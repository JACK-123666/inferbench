"""测量执行器：把「模型 × 评测集 × 重复次数」跑成可比的记录与汇总。

测量口径（每一条都写进结果文件，保证数字可复现、可被追问）：

1. **先清空显存**：Ollama 会让最近用过的模型常驻（keep_alive），多档量化连着跑
   很容易把 8GB 打满（本机实测 7836/8188 MiB），显存竞争会让解码从 150 tok/s
   掉到 0.6 tok/s（**250 倍**）。所以每次测一个模型前先卸载其它所有模型，
   并记录 GPU 显存基线。
2. **预热**：模型首次加载有 14s 级冷启动，正式测量前先跑一次并**丢弃**。
3. **锁 num_ctx**：默认 131072 会让 KV cache 吃满显存（实测 1.2GB ↔ 5.1GB），
   不锁死就变成"测显存"而不是"测量化"。
4. **think=False**：Qwen3 等思维链模型必须关掉，否则输出混入推理链，token 数不可比。
5. **temperature=0 + 固定 seed**：降低随机性对准确率的影响。
6. **重复 N 次取中位数**并同时给出 P95。
7. **记录显存与 GPU 分流**：`size_vram / size` 算出的 GPU 占比是判断
   「是否真的全量在 GPU 上」的关键证据。
"""
from __future__ import annotations

import time
from typing import Callable

from inferbench import config
from inferbench import gpu, tasks
from inferbench.ollama import Ollama, OllamaError
from inferbench.stats import mean, median, percentile

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    return None


def measure_model(client: Ollama, model: str, items: list[dict], *,
                  repeats: int = config.REPEATS,
                  shots: int = config.SHOTS,
                  num_ctx: int = config.NUM_CTX,
                  num_predict: int = config.NUM_PREDICT,
                  isolate: bool = True,
                  progress: ProgressFn = _noop) -> dict:
    """跑一个模型，返回 {model, footprint, summary, records}。"""
    if isolate:
        stopped = client.unload_all(keep=model)
        if stopped:
            progress(f"[环境] 已卸载其它模型释放显存: {', '.join(stopped)}")

    gpu_before = gpu.state()
    if gpu_before:
        progress(f"[环境] 测量前 GPU: {gpu.describe(gpu_before)}")

    progress(f"[{model}] 预热中（冷启动不计入统计）…")
    try:
        client.chat(model, tasks.build_messages(items[0]["text"], shots),
                    num_ctx=num_ctx, num_predict=num_predict)
    except OllamaError as exc:
        return {"model": model, "error": f"预热失败: {exc}", "records": [], "summary": {}}

    footprint = client.footprint(model)
    gpu_after = gpu.state()
    progress(f"[{model}] 已加载: {footprint['footprint_gb']:.2f} GB "
             f"(VRAM {footprint.get('vram_gb', 0):.2f} GB, {footprint['processor']}), "
             f"context={footprint['context_loaded']}")
    warn = gpu.hygiene_warning(gpu_before, footprint["footprint_gb"])
    if warn:
        progress(f"[{model}] {warn}")
    if footprint.get("gpu_ratio", 100) < 99.5:
        progress(f"[{model}] ⚠ 该模型未全量上 GPU，速度数字会被 CPU 拖累，"
                 f"对比前需要确认显存余量")

    records: list[dict] = []
    total_calls = len(items) * repeats
    done = 0
    for rep in range(repeats):
        for item in items:
            messages = tasks.build_messages(item["text"], shots)
            t0 = time.perf_counter()
            error = ""
            try:
                resp = client.chat(model, messages, num_ctx=num_ctx, num_predict=num_predict)
                met = client.metrics(resp)
                raw = (resp.get("message") or {}).get("content", "") or ""
                pred = tasks.parse_label(raw)
            except OllamaError as exc:
                met = {"total_ms": 0.0, "load_ms": 0.0, "ttft_ms": 0.0, "prefill_tps": 0.0,
                       "decode_tps": 0.0, "prompt_tokens": 0, "output_tokens": 0, "done_reason": ""}
                raw, pred, error = "", None, str(exc)[:200]
            wall_ms = (time.perf_counter() - t0) * 1000.0

            records.append({
                "model": model,
                "rep": rep,
                "item_id": item["id"],
                "gold": item["intent"],
                "pred": pred or "",
                "correct": bool(pred == item["intent"]),
                "hard": bool(item.get("hard")),
                "raw_output": raw.strip()[:60],
                "wall_ms": round(wall_ms, 2),
                "ttft_ms": round(met["ttft_ms"], 2),
                "total_ms": round(met["total_ms"], 2),
                "decode_tps": round(met["decode_tps"], 2),
                "prefill_tps": round(met["prefill_tps"], 2),
                "prompt_tokens": met["prompt_tokens"],
                "output_tokens": met["output_tokens"],
                "footprint_gb": round(footprint["footprint_gb"], 3),
                "vram_gb": round(footprint.get("vram_gb", 0.0), 3),
                "gpu_ratio": round(footprint.get("gpu_ratio", 0.0), 1),
                "num_ctx": num_ctx,
                "error": error,
            })
            done += 1
            if done % 50 == 0 or done == total_calls:
                progress(f"  进度 {done}/{total_calls}")

    summary = _summarize(model, records, footprint, repeats, shots, num_ctx,
                         gpu_before, gpu_after)
    return {"model": model, "footprint": footprint, "summary": summary, "records": records}


def _summarize(model: str, records: list[dict], footprint: dict, repeats: int,
               shots: int, num_ctx: int, gpu_before: dict, gpu_after: dict) -> dict:
    scored = tasks.score(records)
    ok = [r for r in records if not r["error"]]
    ttft = [r["ttft_ms"] for r in ok]
    decode = [r["decode_tps"] for r in ok]
    wall = [r["wall_ms"] for r in ok]
    errors = len(records) - len(ok)
    return {
        "model": model,
        "repeats": repeats,
        "shots": shots,
        "num_ctx": num_ctx,
        "accuracy": scored["accuracy"],
        "accuracy_easy": scored["accuracy_easy"],
        "accuracy_hard": scored["accuracy_hard"],
        "n_hard": scored["n_hard"],
        "invalid_rate": scored["invalid_rate"],
        "confusion": scored["confusion"],
        "ttft_ms_median": round(median(ttft), 2),
        "ttft_ms_p95": round(percentile(ttft, 95), 2),
        "decode_tps_median": round(median(decode), 2),
        "decode_tps_mean": round(mean(decode), 2),
        "wall_ms_median": round(median(wall), 2),
        "wall_ms_p95": round(percentile(wall, 95), 2),
        "footprint_gb": round(footprint["footprint_gb"], 3),
        "vram_gb": round(footprint.get("vram_gb", 0.0), 3),
        "gpu_ratio": round(footprint.get("gpu_ratio", 0.0), 1),
        "processor": footprint["processor"],
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "request_errors": errors,
        "n_records": len(records),
    }


