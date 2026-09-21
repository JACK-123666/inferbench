"""实验 3 · 缓存写入门槛：用自一致性（self-consistency）治 A 类误命中

## 为什么做这个实验

实验 2 的结论是：**误命中 100% 属于 A 类「继承后端模型自身的错误」**——
缓存存的是模型当初的预测，模型答错那一次被永久固化，之后所有近义请求都被这个错答案命中。
阈值再高也救不了，因为错误在**写入缓存的那一刻**就已经产生了。

所以改进方向不是继续调阈值，而是**给"写入缓存"加一道门槛**。

## 门槛怎么设计

**自一致性过滤**：同一条 query 采样 n 次（temperature>0 制造多样性），
- 只有 n 次**完全一致**才允许写入缓存（不一致说明模型自己都没把握，这种答案固化下来最危险）；
- 对外返回**多数票**结果（顺带还能提升一点准确率）。

## 实验设计（保证 A/B 公平）

1. 对请求流里所有**唯一文本**一次性采样 n 次，存成表（`first` = 第一次采样结果，
   `majority` = 多数票，`unanimous` = 是否 n 次一致）。
2. 然后跑两遍模拟，**用同一张采样表**，只有"缓存写入门槛"这一个变量不同：
   - `gate=off`：单次调用，永远写入（= 实验 2 的做法）
   - `gate=on` ：多数票返回，不一致则不写入
3. 对比：误命中率、A 类次数、端到端准确率、命中率、成本、延迟。

这样差异只可能来自门槛，不可能来自采样波动。

## 跑法

    python -m inferbench gate --limit 100 --threshold 0.92 --n 3
    python -m inferbench gate --perturb-only --threshold 0.92   # 快速版
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path


from inferbench import config  # noqa: E402
from inferbench import fingerprint  # noqa: E402
from inferbench.cache import SemanticCache, TimedCache, build_stream, perturb  # noqa: E402
from inferbench import eval_set as ev  # noqa: E402
from inferbench import gpu  # noqa: E402
from inferbench import report as rep  # noqa: E402
from inferbench import tasks  # noqa: E402
from inferbench.ollama import Ollama, OllamaError  # noqa: E402
from inferbench.stats import median, percentile, write_csv, write_json  # noqa: E402


def pre_sample(client: Ollama, texts: list[str], n: int, temperature: float,
               progress) -> dict[str, dict]:
    """对每条唯一文本采样 n 次，得到多数票与是否一致。"""
    table: dict[str, dict] = {}
    for idx, text in enumerate(texts, 1):
        labels: list[str] = []
        total_ms = 0.0
        ptok = otok = 0
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                resp = client.chat(config.CACHE_MODEL, tasks.build_messages(text, config.SHOTS),
                                   num_ctx=config.NUM_CTX, num_predict=config.NUM_PREDICT,
                                   temperature=temperature)
                raw = (resp.get("message") or {}).get("content", "") or ""
                met = client.metrics(resp)
                labels.append(tasks.parse_label(raw) or "")
                ptok += met["prompt_tokens"]
                otok += met["output_tokens"]
            except OllamaError as exc:
                labels.append("")
                print(f"    采样失败: {exc}")
            total_ms += (time.perf_counter() - t0) * 1000.0
        counts = Counter(labels)
        majority, mcount = counts.most_common(1)[0]
        table[text] = {
            "labels": labels,
            "first": labels[0] if labels else "",
            "majority": majority,
            "unanimous": mcount == n,
            "votes": mcount,
            "latency_ms": total_ms,                 # n 次的真实总耗时
            "single_ms": total_ms / max(1, n),      # 单次平均耗时
            "prompt_tokens": ptok,
            "output_tokens": otok,
        }
        if idx % 40 == 0 or idx == len(texts):
            progress(f"  采样进度 {idx}/{len(texts)}")
    return table


def simulate(stream: list[dict], table: dict[str, dict], embedder, *, threshold: float,
             gate: bool) -> dict:
    """重放请求流。gate=False → 单次调用+永远写入；gate=True → 多数票+一致才写入。"""
    cache = SemanticCache(version="v1")
    rows: list[dict] = []
    now_base = time.time()
    write_skipped = 0

    for i, item in enumerate(stream):
        now = now_base + i
        vec, embed_ms = embedder.embed(item["text"])
        t_lookup = time.perf_counter()
        score, entry = cache.lookup(vec, now)
        lookup_ms = (time.perf_counter() - t_lookup) * 1000.0
        rec = table[item["text"]]

        if entry is not None and score >= threshold:
            hit = True
            pred = entry["label"]
            source_gold = entry.get("source_gold", "")
            latency_ms = embed_ms + lookup_ms
            prompt_tokens = output_tokens = 0
        else:
            hit = False
            source_gold = ""
            if gate:
                pred = rec["majority"]
                latency_ms = embed_ms + rec["latency_ms"]
                prompt_tokens, output_tokens = rec["prompt_tokens"], rec["output_tokens"]
                allowed = rec["unanimous"]
            else:
                pred = rec["first"]
                latency_ms = embed_ms + rec["single_ms"]
                prompt_tokens = round(rec["prompt_tokens"] / max(1, len(rec["labels"])))
                output_tokens = round(rec["output_tokens"] / max(1, len(rec["labels"])))
                allowed = True
            if allowed:
                cache.store(vec, pred, item["text"], now, source_gold=item["gold"])
            else:
                write_skipped += 1

        if hit and pred != item["gold"]:
            cause = "A_继承后端错误" if source_gold == item["gold"] else "B_跨意图误匹配"
        else:
            cause = ""

        rows.append({
            "gate": "on" if gate else "off",
            "idx": i,
            "text": item["text"],
            "kind": item["kind"],
            "gold": item["gold"],
            "pred": pred or "",
            "hit": hit,
            "sim": round(score, 4),
            "correct": bool(pred == item["gold"]),
            # 无缓存基线（单次调用）是否正确
            "single_correct": bool(rec["first"] == item["gold"]),
            "false_hit_cause": cause,
            "source_gold": source_gold,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        })

    hits = [r for r in rows if r["hit"]]
    false_hits = [r for r in hits if not r["correct"]]
    false_a = [r for r in false_hits if r["false_hit_cause"].startswith("A")]
    false_b = [r for r in false_hits if r["false_hit_cause"].startswith("B")]
    misses = [r for r in rows if not r["hit"]]

    # 成本：无缓存 = 每个请求单次调用
    ptok_single = sum(round(table[r["text"]]["prompt_tokens"] / max(1, len(table[r["text"]]["labels"])))
                      for r in rows)
    otok_single = sum(round(table[r["text"]]["output_tokens"] / max(1, len(table[r["text"]]["labels"])))
                      for r in rows)
    cost_nocache = (ptok_single / 1000) * config.PRICE_PER_1K_PROMPT_CNY + \
                   (otok_single / 1000) * config.PRICE_PER_1K_OUTPUT_CNY
    ptok_now = sum(r["prompt_tokens"] for r in rows)
    otok_now = sum(r["output_tokens"] for r in rows)
    cost_now = (ptok_now / 1000) * config.PRICE_PER_1K_PROMPT_CNY + \
               (otok_now / 1000) * config.PRICE_PER_1K_OUTPUT_CNY

    return {
        "gate": "on" if gate else "off",
        "threshold": threshold,
        "requests": len(rows),
        "hit_rate": len(hits) / len(rows),
        "false_hit_count": len(false_hits),
        "false_hit_rate": (len(false_hits) / len(hits)) if hits else 0.0,
        "false_hit_a": len(false_a),
        "false_hit_b": len(false_b),
        "cache_entries": len(cache.entries),
        "write_skipped": write_skipped,
        "end_to_end_accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
        "nocache_accuracy": sum(1 for r in rows if r["single_correct"]) / len(rows),
        "latency_ms_median": median([r["latency_ms"] for r in rows]),
        "latency_ms_p95": percentile([r["latency_ms"] for r in rows], 95),
        "miss_latency_p95": percentile([r["latency_ms"] for r in misses], 95) if misses else 0.0,
        "cost_saved_pct": ((cost_nocache - cost_now) / cost_nocache * 100.0) if cost_nocache else 0.0,
        "rows": rows,
    }


def build_report(payload: dict) -> Path:
    tag = payload["tag"]
    path = config.RESULTS_DIR / f"cache_gate_{tag}.md"
    off, on = payload["off"], payload["on"]
    meta = payload["meta"]
    n = meta["n"]

    md: list[str] = ["# 实验 3 · 缓存写入门槛（自一致性）验证报告\n"]
    md.append(f"- 生成时间：{payload['created_at']}")
    md.extend(fingerprint.report_lines(payload))
    md.append("")
    md.append(f"- 请求流：{meta['stream_size']} 条请求 / {meta['unique_texts']} 条唯一文本")
    md.append(f"- 采样：每条唯一文本采样 **{n} 次**（temperature={meta['temperature']}）")
    md.append(f"- **采样一致率：{meta['unanimous_rate'] * 100:.1f}%**"
              f"（{meta['unanimous_count']}/{meta['unique_texts']} 条文本的 {n} 次采样完全一致）"
              f"← 这个数字决定了门槛有没有可能起作用")
    md.append(f"- 阈值固定为 **{meta['threshold']}**，唯一变量是「缓存写入门槛」\n")

    md.append("## A/B 对照\n")
    md.append("| 指标 | 门槛关闭（单次调用+总是写入） | 门槛开启（多数票+一致才写入） | 变化 |")
    md.append("|---|---|---|---|")
    rows = [
        ("命中率", f"{off['hit_rate'] * 100:.1f}%", f"{on['hit_rate'] * 100:.1f}%"),
        ("**误命中率**", f"{off['false_hit_rate'] * 100:.1f}%", f"{on['false_hit_rate'] * 100:.1f}%"),
        ("误命中次数", f"{off['false_hit_count']} 次", f"{on['false_hit_count']} 次"),
        ("├ A 类（继承后端错误）", f"{off['false_hit_a']}", f"{on['false_hit_a']}"),
        ("└ B 类（跨意图误匹配）", f"{off['false_hit_b']}", f"{on['false_hit_b']}"),
        ("缓存条目数", f"{off['cache_entries']}", f"{on['cache_entries']}"),
        ("跳过写入次数", f"{off['write_skipped']}", f"{on['write_skipped']}"),
        ("端到端准确率", f"{off['end_to_end_accuracy'] * 100:.1f}%", f"{on['end_to_end_accuracy'] * 100:.1f}%"),
        ("无缓存基线准确率", f"{off['nocache_accuracy'] * 100:.1f}%", f"{on['nocache_accuracy'] * 100:.1f}%"),
        ("延迟中位数", f"{off['latency_ms_median']:.0f} ms", f"{on['latency_ms_median']:.0f} ms"),
        ("延迟 P95", f"{off['latency_ms_p95']:.0f} ms", f"{on['latency_ms_p95']:.0f} ms"),
        ("成本节省", f"{off['cost_saved_pct']:.1f}%", f"{on['cost_saved_pct']:.1f}%"),
    ]
    for name, a, b in rows:
        md.append(f"| {name} | {a} | {b} | |")
    md.append("")
    md.append(rep.bar_chart(
        ["误命中率 (%)", "端到端准确率 (%)", "命中率 (%)", "成本节省 (%)"],
        {"门槛关闭": [off["false_hit_rate"] * 100, off["end_to_end_accuracy"] * 100,
                      off["hit_rate"] * 100, off["cost_saved_pct"]],
         "门槛开启": [on["false_hit_rate"] * 100, on["end_to_end_accuracy"] * 100,
                      on["hit_rate"] * 100, on["cost_saved_pct"]]},
        title="自一致性门槛 A/B 对照", y_label="%", unit="%", height=300))
    md.append("")

    delta_fh = (on["false_hit_rate"] - off["false_hit_rate"]) * 100
    delta_acc = (on["end_to_end_accuracy"] - off["end_to_end_accuracy"]) * 100
    delta_cost = on["cost_saved_pct"] - off["cost_saved_pct"]
    fh_reduction = off["false_hit_count"] - on["false_hit_count"]
    lat_ratio = (on["latency_ms_median"] / off["latency_ms_median"]) if off["latency_ms_median"] else 1.0

    md.append("## 结论（自动生成）\n")
    md.append(f"**收益侧**：误命中 {off['false_hit_count']} → {on['false_hit_count']} 次"
              f"（少 {fh_reduction} 次），误命中率 {off['false_hit_rate'] * 100:.1f}% → "
              f"{on['false_hit_rate'] * 100:.1f}%（{delta_fh:+.1f} 个百分点）；"
              f"端到端准确率 {off['end_to_end_accuracy'] * 100:.1f}% → "
              f"{on['end_to_end_accuracy'] * 100:.1f}%（{delta_acc:+.1f} 个百分点）。")
    md.append(f"**代价侧**：成本节省 {off['cost_saved_pct']:.1f}% → {on['cost_saved_pct']:.1f}%"
              f"（{delta_cost:+.1f} 个百分点，即**从省成本变成亏成本**）；"
              f"延迟中位数 {off['latency_ms_median']:.0f} → {on['latency_ms_median']:.0f} ms"
              f"（**{lat_ratio:.1f} 倍**）。")
    md.append("")
    if fh_reduction <= 2:
        md.append(f"### 判定：**门槛无效，不建议采用**\n")
        md.append(f"1. 收益只在噪声级别（少 {fh_reduction} 次误命中），而代价是从省 "
                  f"{off['cost_saved_pct']:.1f}% 成本变成亏 {-on['cost_saved_pct']:.1f}%、"
                  f"延迟 {lat_ratio:.1f} 倍。")
        md.append(f"2. **根因在采样一致率**：{meta['unanimous_count']}/{meta['unique_texts']} 条文本"
                  f"（{meta['unanimous_rate'] * 100:.1f}%）在 temperature={meta['temperature']} 下"
                  f"采样 {n} 次完全一致——门槛总共只挡住了 "
                  f"{off['cache_entries'] - on['cache_entries']} 次写入。")
        md.append(f"3. 也就是说：**这个模型对错的题是「稳定地错」，不是「随机地错」**。"
                  f"自一致性过滤只在「模型自己会犹豫」时才起作用，"
                  f"对这种系统性偏差**在原理上就无效**。")
        md.append(f"4. 这个负结果把改进方向钉死了：要治系统性偏差，只能从"
                  f"**换更强的后端模型**、**对易错意图单独处理（命中后二次校验 / 干脆绕过缓存）**"
                  f"入手，而不是继续在缓存层加花样。")
    elif delta_cost < -20:
        md.append(f"### 判定：**有改善，但代价过大，需要更省的实现**\n")
        md.append(f"1. 误命中少 {fh_reduction} 次、误命中率 {delta_fh:+.1f} 个百分点，"
                  f"收益是真的；但成本节省 {delta_cost:+.1f} 个百分点、延迟 {lat_ratio:.1f} 倍，"
                  f"靠「每个未命中采样 {n} 次」来换取并不划算。")
        md.append(f"2. 更省的实现方向：① 只对**低置信度**请求采样（高置信度直接写入）；"
                  f"② 用小模型做校验而不是重复大模型；③ 只对**高价值意图**开启门槛。")
    else:
        md.append(f"### 判定：**门槛有效且代价可接受**\n")
        md.append(f"1. 误命中少 {fh_reduction} 次（{delta_fh:+.1f} 个百分点），"
                  f"端到端准确率 {delta_acc:+.1f} 个百分点，成本 {delta_cost:+.1f} 个百分点、"
                  f"延迟 {lat_ratio:.1f} 倍。")
    md.append("")
    md.append("## 面试怎么讲这个结果\n")
    md.append("> 「实验 2 我发现误命中 100% 来自「继承后端模型自己的错误」，"
              "于是加了自一致性门槛——同一条 query 采样 3 次、结果一致才允许写入缓存。"
              f"结果：{meta['unanimous_count']}/{meta['unique_texts']} 条文本采样 3 次完全一致，"
              f"门槛只挡住了 {off['cache_entries'] - on['cache_entries']} 次写入，"
              f"误命中只少 {fh_reduction} 次，成本却从省 {off['cost_saved_pct']:.1f}% 变成亏 "
              f"{-on['cost_saved_pct']:.1f}%。**结论是这个模型「稳定地错」，"
              "自一致性只能治随机性错误。这个负结果让我把改进方向改到后端模型质量和意图级策略上，"
              "而不是继续在缓存层加花活。**」")
    md.append("")
    md.append(f"明细数据：`results/cache_gate_{tag}.csv`")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def run(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="缓存写入门槛（自一致性）A/B 实验")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--n", type=int, default=3, help="每条 query 采样次数")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--paraphrases", type=int, default=2)
    ap.add_argument("--perturb-only", action="store_true")
    ap.add_argument("--eval-set", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    items = ev.load(args.eval_set or None)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("评测集为空。")
        return 2

    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    client = Ollama()
    embedder = TimedCache(client)

    def progress(msg: str) -> None:
        print(msg, flush=True)

    print("=" * 78)
    print(f"缓存写入门槛实验  |  {len(items)} query × {args.paraphrases} 改写  |  "
          f"阈值 {args.threshold}  |  采样 {args.n} 次 @T={args.temperature}")
    print("=" * 78)

    gpu_before = gpu.state()
    stopped = client.unload_all()
    if stopped:
        progress(f"[环境] 已卸载模型释放显存: {', '.join(stopped)}")
    if gpu_before:
        progress(f"[环境] 测量前 GPU: {gpu.describe(gpu_before)}")

    stream = build_stream(items, client, paraphrases=args.paraphrases,
                             perturb_only=args.perturb_only, progress=progress)
    unique = sorted({s["text"] for s in stream})
    print(f"请求流 {len(stream)} 条，唯一文本 {len(unique)} 条")

    embedder.warmup(rounds=2)
    progress(f"预热完成：embed {embedder.warm_ms['embed']:.1f} ms · "
             f"LLM {embedder.warm_ms['llm']:.1f} ms")

    est_calls = len(unique) * args.n
    progress(f"开始采样：{len(unique)} 条唯一文本 × {args.n} 次 = {est_calls} 次调用"
             f"（约 {est_calls * 0.1 / 60:.1f} 分钟）…")
    t0 = time.time()
    table = pre_sample(client, unique, args.n, args.temperature, progress)
    progress(f"采样完成，用时 {time.time() - t0:.1f}s")

    progress("模拟 A：门槛关闭（单次调用 + 总是写入）…")
    off = simulate(stream, table, embedder, threshold=args.threshold, gate=False)
    progress("模拟 B：门槛开启（多数票 + 一致才写入）…")
    on = simulate(stream, table, embedder, threshold=args.threshold, gate=True)

    payload = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": fingerprint.fingerprint(),
        "meta": {
            "stream_size": len(stream),
            "unique_texts": len(unique),
            "base_queries": len(items),
            "paraphrases": args.paraphrases,
            "threshold": args.threshold,
            "n": args.n,
            "temperature": args.temperature,
            "cache_model": config.CACHE_MODEL,
            "embed_model": config.EMBED_MODEL,
            "gpu_before": gpu_before,
            # 采样一致率：门槛能否起作用的前提条件
            "unanimous_count": sum(1 for v in table.values() if v["unanimous"]),
            "unanimous_rate": (sum(1 for v in table.values() if v["unanimous"]) / len(table)) if table else 0.0,
            "inconsistent_texts": [k for k, v in table.items() if not v["unanimous"]][:5],
        },
        "off": {k: v for k, v in off.items() if k != "rows"},
        "on": {k: v for k, v in on.items() if k != "rows"},
    }
    write_json(config.RESULTS_DIR / f"cache_gate_{tag}.json", payload)
    write_csv(config.RESULTS_DIR / f"cache_gate_{tag}.csv", off["rows"] + on["rows"])

    print("-" * 78)
    print(f"{'指标':<22}{'门槛关闭':>14}{'门槛开启':>14}")
    print("-" * 62)
    # 注意：*_pct 结尾的字段本身已经是百分数，不能再乘 100
    for name, key, fmt in [("命中率", "hit_rate", "pct"), ("误命中率", "false_hit_rate", "pct"),
                           ("误命中次数", "false_hit_count", "int"), ("A 类", "false_hit_a", "int"),
                           ("B 类", "false_hit_b", "int"), ("端到端准确率", "end_to_end_accuracy", "pct"),
                           ("无缓存基线", "nocache_accuracy", "pct"), ("成本节省", "cost_saved_pct", "pct1")]:
        if fmt == "int":
            a, b = f"{off[key]}", f"{on[key]}"
        elif fmt == "pct1":
            a, b = f"{off[key]:.1f}%", f"{on[key]:.1f}%"
        else:
            a, b = f"{off[key] * 100:.1f}%", f"{on[key] * 100:.1f}%"
        print(f"{name:<22}{a:>14}{b:>14}")
    print("-" * 78)
    print(f"报告已生成: {build_report(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())






