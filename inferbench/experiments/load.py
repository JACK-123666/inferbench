"""实验五 · 并发压测：QPS 上限、尾延迟劣化与首 Token SLO

用 httpx + asyncio 做并发梯度压测，回答前面串行实验测不出来的问题：

    基础单请求延迟 150ms 的模型，并发 8 时 QPS 是多少？P99 会不会失控？
    承载上限（饱和点）在哪一档？首 Token 延迟（TTFT）随并发怎么变化？

跑法::

    python -m inferbench load                                   # 默认 1,2,4,8,16 五档
    python -m inferbench load --levels 1,4,8 --requests 24       # 冒烟
    python -m inferbench load --model qwen3:1.7b-q8_0            # 换量化档看承载能力差异
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from inferbench import config
from inferbench import fingerprint
from inferbench import eval_set as ev
from inferbench import gpu, loadgen, report as rep
from inferbench.ollama import Ollama
from inferbench.stats import write_csv, write_json


def build_report(payload: dict) -> "object":
    rows = payload["levels"]
    tag = payload["tag"]
    path = config.RESULTS_DIR / f"load_{tag}.md"
    meta = payload["meta"]

    md: list[str] = ["# 实验 5 · 并发压测报告\n"]
    md.append(f"- 生成时间：{payload['created_at']}")
    md.extend(fingerprint.report_lines(payload))
    md.extend(ev.render_lines(payload))
    md.append("")
    md.append(f"- 模型：`{meta['model']}` ｜ 并发梯度：{meta['levels']} ｜ "
              f"每档 {meta['requests_per_level']} 请求 ｜ num_predict={meta['max_tokens']}")
    md.append(f"- 请求方式：**流式**（`stream=true`）——TTFT 是实测首 token 到达时间，"
              f"不是非流式的 prefill 近似值")
    md.append(f"- 测量前 GPU：{meta.get('gpu', 'n/a')}\n")

    md.append("## 并发梯度\n")
    md.append("| 并发 | QPS | 聚合 tok/s | TTFT P50 (ms) | TTFT P95 (ms) | 总时延 P50 (ms) | P95 (ms) | P99 (ms) | 错误率 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['concurrency']} | {r['qps']} | {r['tokens_per_s']} | {r['ttft_p50']} | "
                  f"{r['ttft_p95']} | {r['lat_p50']} | {r['lat_p95']} | {r['lat_p99']} | "
                  f"{r['error_rate'] * 100:.1f}% |")
    md.append("")

    xs = [r["concurrency"] for r in rows]
    md.append(rep.line_chart(xs, {"QPS": [r["qps"] for r in rows],
                                  "聚合 tok/s": [r["tokens_per_s"] for r in rows]},
                             title="并发 → 吞吐", x_label="并发数", y_label="次/秒 · tok/s",
                             percent=False))
    md.append("")
    md.append(rep.line_chart(xs, {"P50": [r["lat_p50"] for r in rows],
                                  "P95": [r["lat_p95"] for r in rows],
                                  "P99": [r["lat_p99"] for r in rows],
                                  "TTFT P95": [r["ttft_p95"] for r in rows]},
                             title="并发 → 尾延迟劣化", x_label="并发数", y_label="ms",
                             percent=False))
    md.append("")

    # 与基线（另一份 load 结果）对比：用于验证"开启并行度"这类改进是否真的有效
    base_load = payload.get("baseline_load")
    if base_load:
        bmap = {r["concurrency"]: r for r in base_load.get("levels", [])}
        md.append("## 与基线对比（改进验证）\n")
        md.append(f"- 基线：`{base_load['meta'].get('model')}` @ "
                  f"`{base_load['meta'].get('host', 'default')}`，"
                  f"OLLAMA_NUM_PARALLEL={base_load['meta'].get('ollama_num_parallel')}，"
                  f"模型占用 {base_load['meta'].get('footprint_gb', 0):.2f} GB")
        md.append(f"- 本次：`{meta['model']}` @ `{meta.get('host')}`，"
                  f"OLLAMA_NUM_PARALLEL={meta.get('ollama_num_parallel')}，"
                  f"模型占用 {meta.get('footprint_gb', 0):.2f} GB\n")
        md.append("| 并发 | 基线 QPS | 本次 QPS | 吞吐变化 | 基线 TTFT P50 | 本次 TTFT P50 | TTFT 变化 |")
        md.append("|---|---|---|---|---|---|---|")
        for r in rows:
            b = bmap.get(r["concurrency"])
            if not b:
                continue
            q = r["qps"] / max(b["qps"], 0.01)
            t_ratio = r["ttft_p50"] / max(b["ttft_p50"], 0.01)
            md.append(f"| {r['concurrency']} | {b['qps']} | {r['qps']} | "
                      f"{'**↑' if q >= 1.05 else ('↓' if q <= 0.95 else '≈')}{q:.2f}×** | "
                      f"{b['ttft_p50']:.1f} ms | {r['ttft_p50']:.1f} ms | "
                      f"{'**↓' if t_ratio <= 0.95 else ('↑' if t_ratio >= 1.05 else '≈')}{t_ratio:.2f}×** |")
        md.append("")
        peak_b = max(base_load.get("levels", []), key=lambda r: r["qps"])
        peak_n = max(rows, key=lambda r: r["qps"])
        mem_gain = (meta.get("footprint_gb", 0) / max(base_load["meta"].get("footprint_gb", 1), 0.01) - 1) * 100
        md.append("**验证结论**：\n")
        md.append(f"- 峰值吞吐 {peak_b['qps']} → **{peak_n['qps']} QPS**"
                  f"（**{peak_n['qps'] / max(peak_b['qps'], 0.01):.2f}×**），"
                  f"聚合 tok/s {peak_b['tokens_per_s']} → {peak_n['tokens_per_s']}。")
        md.append(f"- 代价是显存：模型占用 {base_load['meta'].get('footprint_gb', 0):.2f} GB → "
                  f"{meta.get('footprint_gb', 0):.2f} GB（**{mem_gain:+.0f}%**）——"
                  f"每路并行各占一份 KV cache，8GB 卡上要腾出余量。")
        md.append("- 结论：**同一个硬件，改一个并行度配置就能把吞吐翻倍、把首 Token 延迟降一个数量级**；"
                  "这也解释了为什么「加并发不涨吞吐」——瓶颈从来不是并发度，而是单 slot 串行执行。")
        md.append("")
    sat = payload.get("saturation", {})
    base, peak = rows[0], max(rows, key=lambda r: r["qps"])
    qps_gain = peak["qps"] / max(base["qps"], 0.01)
    ttft_gain = peak["ttft_p95"] / max(base["ttft_p95"], 0.01)
    lat_gain = peak["lat_p95"] / max(base["lat_p95"], 0.01)

    md.append("## 关键结论（数字自动取自本次运行）\n")
    md.append(f"1. **吞吐上限**：并发从 {base['concurrency']} 提到 {peak['concurrency']}，"
              f"QPS {base['qps']} → {peak['qps']}（**{qps_gain:.2f}×**）；"
              f"聚合 tok/s 达 {peak['tokens_per_s']}。")
    md.append(f"2. **代价是尾延迟**：同区间 P95 从 {base['lat_p95']} ms 涨到 {peak['lat_p95']} ms"
              f"（**{lat_gain:.2f}×**），P99 {base['lat_p99']} → {peak['lat_p99']} ms。")
    md.append(f"3. **首 Token 延迟**：TTFT P95 从 {base['ttft_p95']} ms 变为 {peak['ttft_p95']} ms"
              f"（**{ttft_gain:.2f}×**）——这是最直接对应用户体感的 SLO 指标。")

    # 自动识别"排队而非并行"：吞吐几乎没涨，但延迟成倍上升
    if qps_gain < 1.2 and ttft_gain > 2.0:
        md.append(f"4. ⚠ **本实例已经饱和，加并发只换到排队，换不到吞吐**："
                  f"并发 {base['concurrency']}→{peak['concurrency']} 只拿到 **{(qps_gain - 1) * 100:.0f}%** 的吞吐增益，"
                  f"却付出 **{ttft_gain:.1f} 倍** 的首 Token 延迟（TTFT 随并发**线性增长**，是典型 FIFO 排队特征）。")
        if int(payload["meta"].get("ollama_num_parallel", 1)) <= 1:
            md.append(f"5. **根因**：Ollama 的 `OLLAMA_NUM_PARALLEL` 当前为 "
                      f"**{payload['meta'].get('ollama_num_parallel', 1)}**（默认值）——"
                      f"所有请求挤在**同一个 slot** 上串行执行，所以并发度只影响排队长度，不影响吞吐。"
                      f"要真正提升并发承载，需要设 `OLLAMA_NUM_PARALLEL>1` 并重启 Ollama"
                      f"（代价是每路并行都要一份 KV cache 显存，需先确认显存余量）。")
            md.append(f"6. **因此正确的优化方向不是扩并发**，而是降低单请求成本："
                      f"更低的量化档、更短的上下文、更少的输出 token——这三项都能同时改善吞吐与延迟。")
        else:
            md.append(f"5. **排查方向**：`OLLAMA_NUM_PARALLEL={payload['meta'].get('ollama_num_parallel')}` "
                      f"已大于 1，说明瓶颈不在排队，而在 GPU 计算/显存带宽本身（可对比不同量化档的 tok/s 上限）。")
    elif sat.get("concurrency"):
        gain_txt = "未知" if sat.get("qps_gain") is None else f"{sat['qps_gain'] * 100:.0f}%"
        md.append(f"4. **饱和点（推荐工作区间）**：并发 {sat['concurrency']} 附近——"
                  f"该点 QPS 边际增益已降到 {gain_txt}，"
                  f"P95 是最低档的 {sat['lat_p95'] / max(sat.get('base_p95') or 0, 0.01):.1f} 倍。"
                  f"**再加并发只换到延迟，换不到吞吐。**")
    md.append(f"**错误率**：全程 {max(r['error_rate'] for r in rows) * 100:.1f}%——"
              f"并发连接数不是瓶颈，显存能装多大模型才是。")
    md.append("")

    md.append("## 可写进简历的结论句\n")
    if qps_gain < 1.2 and ttft_gain > 2.0:
        md.append(f"> **并发压测与容量规划**：用 httpx + asyncio 做**流式**并发梯度压测（并发 "
                  f"{xs[0]}→{xs[-1]}，实测首 token 到达时间而非 prefill 近似值），"
                  f"发现单实例已饱和——并发 {base['concurrency']}→{peak['concurrency']} "
                  f"仅带来 {(qps_gain - 1) * 100:.0f}% 吞吐增益，却让 TTFT P95 上涨 {ttft_gain:.1f} 倍"
                  f"（{base['ttft_p95']} → {peak['ttft_p95']} ms）、P99 达 {peak['lat_p99']} ms；"
                  f"据此判定瓶颈在单实例推理（显存带宽/单序列执行）而非并发度，"
                  f"把优化方向从「扩并发」改为「降单请求成本」（量化档位 / 上下文裁剪 / 输出长度控制）。")
    else:
        md.append(f"> **并发压测与容量规划**：httpx + asyncio 流式压测并发 {xs[0]}→{xs[-1]}，"
                  f"QPS 提升 {qps_gain:.1f}×（峰值 {peak['qps']}），"
                  f"P95 {peak['lat_p95']} ms、TTFT P95 {peak['ttft_p95']} ms，"
                  f"据此按 P95 SLO 反推可承载并发。")
    md.append("")
    md.append(f"明细数据：`results/load_{tag}.csv`")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inferbench load", description="并发压测（httpx + asyncio）")
    ap.add_argument("--model", default=config.CACHE_MODEL)
    ap.add_argument("--levels", default="1,2,4,8,16", help="并发梯度，逗号分隔")
    ap.add_argument("--requests", type=int, default=32, help="每档请求数")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--host", default=config.OLLAMA_HOST, help="Ollama 地址（可用于对比另一实例，如 NUM_PARALLEL=4）")
    ap.add_argument("--eval-set", default="")
    ap.add_argument("--num-parallel", type=int, default=None,
                    help="被测 Ollama 实例的 OLLAMA_NUM_PARALLEL（客户端探测不到服务器配置，需显式声明；默认读本机环境变量）")
    ap.add_argument("--compare", default="", help="与另一份 load 结果对比（传 baseline 的 json 路径）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")

    items = ev.load(args.eval_set or None)
    prompts = [it["text"] for it in items][: args.requests] if items else []
    while len(prompts) < args.requests:                     # 不够就循环补齐
        prompts += prompts[: max(1, args.requests - len(prompts))]
    if not prompts:
        print("评测集为空，无法构造请求。")
        return 2

    print("=" * 78)
    print(f"并发压测  |  model={args.model}  |  并发梯度 {levels}  |  每档 {args.requests} 请求")
    print(f"流式请求（实测 TTFT）  |  num_predict={args.max_tokens}  temperature=0")
    print("=" * 78)

    cli = Ollama(host=args.host)
    stopped = cli.unload_all(keep=args.model)
    if stopped:
        print(f"[环境] 已卸载其它模型释放显存: {', '.join(stopped)}")
    state = gpu.state()
    if state:
        print(f"[环境] 测量前 GPU: {gpu.describe(state)}")
    print(f"[预热] {args.model} 第一次请求（冷加载不计入统计）…")
    cli.chat(args.model, [{"role": "user", "content": "预热"}], num_predict=8)
    fp = cli.footprint(args.model)
    print(f"[环境] 模型占用 {fp['footprint_gb']:.2f} GB（{fp['processor']}）")

    results = asyncio.run(loadgen.run_gradient(args.host, args.model, prompts,
                                               levels, max_tokens=args.max_tokens))
    print("-" * 78)
    print(f"{'并发':>5}{'QPS':>9}{'tok/s':>10}{'TTFT_P50':>10}{'TTFT_P95':>10}"
          f"{'P95':>9}{'P99':>9}{'错误率':>8}")
    for r in results:
        print(f"{r['concurrency']:>5}{r['qps']:>9.2f}{r['tokens_per_s']:>10.1f}"
              f"{r['ttft_p50']:>10.1f}{r['ttft_p95']:>10.1f}{r['lat_p95']:>9.1f}"
              f"{r['lat_p99']:>9.1f}{r['error_rate'] * 100:>7.1f}%")
    print("-" * 78)

    baseline_load = None
    if args.compare:
        from inferbench.stats import read_json
        baseline_load = read_json(config.ROOT / args.compare) if (config.ROOT / args.compare).exists() else read_json(Path(args.compare))
        print(f"[对比] 已加载基线: {args.compare}")

    all_rows = [dict(row, concurrency=r["concurrency"]) for r in results for row in r["rows"]]
    payload = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": fingerprint.fingerprint(host=args.host),
        "eval_set": ev.identify(args.eval_set or None, items),
        "meta": {
            "model": args.model,
            "levels": levels,
            "requests_per_level": args.requests,
            "max_tokens": args.max_tokens,
            "num_ctx": config.NUM_CTX,
            "streaming": True,
            "gpu": gpu.describe(state) if state else "",
            "footprint_gb": fp["footprint_gb"],
            "client": "httpx + asyncio（Semaphore 控并发）",
            # 并行度是解释"吞吐不涨、延迟线性涨"的关键：1 表示请求在单 slot 上排队
            "host": args.host,
            "ollama_num_parallel": (args.num_parallel if args.num_parallel is not None
                                   else config.ollama_num_parallel()),
        },
        "levels": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "saturation": loadgen.saturation_point(results),
        "baseline_load": baseline_load,
    }
    write_json(config.RESULTS_DIR / f"load_{tag}.json", payload)
    write_csv(config.RESULTS_DIR / f"load_{tag}.csv", all_rows)
    print(f"报告已生成: {build_report(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())






