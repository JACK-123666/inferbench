"""并发压测：httpx + asyncio（实验五）。

## 为什么必须引入框架

前面四个实验都是**串行**发请求（同步 stdlib HTTP），所以测不出：
    - QPS / 吞吐上限
    - 并发下的 P95 / P99 劣化曲线
    - 首 Token 延迟（TTFT）在排队时的真实表现

而这几项恰好对应"高并发、低延迟在线服务"和"保障 P99 与首 Token 的 SLO"。
串行测不出来不是精度问题，是**能力缺失**——这是引入 httpx + asyncio 的唯一理由，
不是为了少写几行代码。

## 与前面实验的测量差异（重要）

- 前面用**非流式**请求，用 `prompt_eval_duration` **近似** TTFT；
- 这里用**流式**请求（`stream=true`），**实测**从发出到第一个 token 到达的时间。
  这才是真正能写进 SLO 的 TTFT。

## 设计

    asyncio.Semaphore 控制并发度 → 每个请求用 httpx.AsyncClient 流式读 NDJSON
    → 记录 TTFT / 总时延 / 输出 tokens → 汇总 QPS、P50/P95/P99、错误率

用法（由 `python -m inferbench load` 调用）::

    python -m inferbench load --model qwen3:1.7b --levels 1,2,4,8,16 --requests 32
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx

from inferbench import config


class StreamResult(dict):
    """一次请求的结果（dict 子类，便于直接落 CSV）。"""


async def one_request(client: httpx.AsyncClient, model: str, prompt: str, *,
                      max_tokens: int = 96, num_ctx: int | None = None,
                      temperature: float | None = None) -> StreamResult:
    """发一次**流式**请求，实测 TTFT 与总时延。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "options": {
            "num_ctx": config.NUM_CTX if num_ctx is None else num_ctx,
            "num_predict": max_tokens,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "seed": config.SEED,
        },
    }
    t0 = time.perf_counter()
    ttft_ms = None
    out_tokens = 0
    text = ""
    metrics: dict = {}
    try:
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    text += piece
                if chunk.get("done"):
                    metrics = chunk
                    out_tokens = int(chunk.get("eval_count") or 0)
        total_ms = (time.perf_counter() - t0) * 1000.0
        decode_s = float(metrics.get("eval_duration") or 0) / 1e9
        return StreamResult(
            ok=True, error="", ttft_ms=round(ttft_ms or total_ms, 2),
            total_ms=round(total_ms, 2), output_tokens=out_tokens,
            decode_tps=round(out_tokens / decode_s, 1) if decode_s > 0 else 0.0,
            prompt_tokens=int(metrics.get("prompt_eval_count") or 0),
            text=text[:40],
        )
    except Exception as exc:  # noqa: BLE001 - 压测要把异常算成失败率，不能中断整轮
        return StreamResult(ok=False, error=str(exc)[:120],
                            ttft_ms=0.0, total_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                            output_tokens=0, decode_tps=0.0, prompt_tokens=0, text="")


async def run_level(host: str, model: str, prompts: list[str], *, concurrency: int,
                    max_tokens: int = 96, timeout: float = 300.0) -> dict:
    """在一个并发度下跑完全部 prompts，返回该档的汇总。"""
    limits = httpx.Limits(max_connections=max(concurrency * 2, 8),
                          max_keepalive_connections=max(concurrency, 4))
    sem = asyncio.Semaphore(concurrency)
    rows: list[StreamResult] = []

    async with httpx.AsyncClient(base_url=host, timeout=timeout, limits=limits) as client:
        async def guarded(prompt: str) -> None:
            async with sem:
                rows.append(await one_request(client, model, prompt, max_tokens=max_tokens))

        wall0 = time.perf_counter()
        await asyncio.gather(*(guarded(p) for p in prompts))
        wall_s = time.perf_counter() - wall0

    ok = [r for r in rows if r["ok"]]
    ttfts = sorted(r["ttft_ms"] for r in ok)
    totals = sorted(r["total_ms"] for r in ok)
    tokens = sum(r["output_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "error_rate": (len(rows) - len(ok)) / len(rows) if rows else 0.0,
        "wall_s": round(wall_s, 2),
        "qps": round(len(ok) / wall_s, 2) if wall_s > 0 else 0.0,
        "tokens_per_s": round(tokens / wall_s, 1) if wall_s > 0 else 0.0,
        "ttft_p50": round(_pct(ttfts, 50), 1),
        "ttft_p95": round(_pct(ttfts, 95), 1),
        "ttft_p99": round(_pct(ttfts, 99), 1),
        "lat_p50": round(_pct(totals, 50), 1),
        "lat_p95": round(_pct(totals, 95), 1),
        "lat_p99": round(_pct(totals, 99), 1),
        "rows": rows,
    }


def _pct(sorted_vals: list[float], pct: float) -> float:
    """对**已排序**列表取百分位（线性插值）。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


async def run_gradient(host: str, model: str, prompts: list[str], levels: list[int], *,
                       max_tokens: int = 96) -> list[dict]:
    """按并发梯度逐档压测。"""
    results: list[dict] = []
    for lv in levels:
        res = await run_level(host, model, prompts, concurrency=lv, max_tokens=max_tokens)
        results.append(res)
    return results


def saturation_point(results: list[dict]) -> dict:
    """找"饱和点"：QPS 增益开始明显衰减、同时 P95 抬头的那一档。

    判定方式：计算相邻档的 QPS 边际增益，取第一个增益 < 30% 且 P95 已明显高于
    最低档的档位——这个点之后再加并发的性价比就差了。
    """
    if len(results) < 2:
        return {}
    base_p95 = results[0]["lat_p95"]
    for prev, cur in zip(results, results[1:]):
        gain = (cur["qps"] - prev["qps"]) / prev["qps"] if prev["qps"] else 0.0
        if gain < 0.30 and cur["lat_p95"] > base_p95 * 1.5:
            return {"concurrency": cur["concurrency"], "qps_gain": round(gain, 2),
                    "lat_p95": cur["lat_p95"], "base_p95": base_p95}
    return {"concurrency": results[-1]["concurrency"],
            "qps_gain": None, "lat_p95": results[-1]["lat_p95"], "base_p95": base_p95}

