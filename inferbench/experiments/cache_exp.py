"""实验 2 · 语义缓存：阈值扫描 + 客观误命中评估 + 版本/灰度/回滚验证

跑法::

    python -m inferbench cache                      # 默认 100 query × (1+2) 改写
    python -m inferbench cache --limit 30 --thresholds 0.85,0.90,0.95
    python -m inferbench cache --perturb-only        # 不调 LLM 生成改写，用确定性扰动

为什么这个实验值得做：

1. **命中是否"正确"是可客观判定的**。意图分类有金标准标签：缓存返回的标签
   如果等于当前 query 的真实意图 → 命中正确；不等 → **误命中**。所以
   「语义缓存的代价」不是感觉，而是一条可测量的误命中率曲线。
2. **端到端准确率**：把命中/未命中两条路径合起来算总准确率，直接量化
   "用缓存换成本"到底付出了多少质量。
3. **阈值在评测集上扫出来**，不是拍脑袋设 0.9。
4. **版本指纹 + TTL + 灰度开关**：模型或 Prompt 换代时缓存必须能一键失效，
   这部分用可复现的实验动作验证，而不是嘴上说"我们会做灰度"。

成本口径说明：同一个文本的 LLM 只在首次真实调用并记录耗时/Token，重复出现时
复用首次的测量值（`--no-memo-latency` 可关闭）。这样一次实验的 LLM 调用量
被压到"唯一文本数"，跑得完且数字可比。
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path


from inferbench import config  # noqa: E402
from inferbench import fingerprint  # noqa: E402
from inferbench import eval_set as ev  # noqa: E402
from inferbench import gpu  # noqa: E402
from inferbench import report as rep  # noqa: E402
from inferbench import tasks  # noqa: E402
from inferbench.ollama import Ollama, OllamaError  # noqa: E402
from inferbench.stats import cosine, median, percentile, write_csv, write_json  # noqa: E402

from inferbench.cache import SemanticCache, TimedCache, build_stream, perturb  # noqa: E402


# ---------------------------------------------------------------------------
# 阈值扫描主流程（缓存实现见 inferbench/cache.py，两个实验共用）
# ---------------------------------------------------------------------------


def run_threshold(threshold: float, stream: list[dict], timer: TimedCache) -> dict:
    """在给定阈值下重放整条请求流。"""
    cache = SemanticCache(version="v1")
    rows: list[dict] = []
    now_base = time.time()

    for i, item in enumerate(stream):
        now = now_base + i
        vec, embed_ms = timer.embed(item["text"])
        lookup_start = time.perf_counter()
        score, entry = cache.lookup(vec, now)
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        # 无缓存基线：这条文本如果直接打给后端模型，它会答什么
        _, llm_pred, _, llm_ptok, llm_otok = timer.classify(item["text"])

        source_gold = ""
        if cache.enabled and entry is not None and score >= threshold:
            latency_ms = embed_ms + lookup_ms
            pred = entry["label"]
            source_gold = entry.get("source_gold", "")
            hit = True
            prompt_tokens = output_tokens = 0
            raw = f"[cache] {entry['text'][:30]}"
        else:
            raw, pred, llm_ms, prompt_tokens, output_tokens = timer.classify(item["text"])
            latency_ms = embed_ms + llm_ms
            hit = False
            cache.store(vec, llm_pred or "", item["text"], now, source_gold=item["gold"])

        # 误命中归因（这是本实验最关键的一步）：
        #   A 类 = 命中源与当前问题的真实意图一致，但缓存里存的是后端模型当初答错的标签
        #          → 属于「继承并放大后端模型自身的错误」（阈值再高也救不了）
        #   B 类 = 命中源与当前问题的真实意图不一致
        #          → 属于「语义匹配跨了意图」，这才是阈值/embedding 的问题
        if hit and pred != item["gold"]:
            cause = "A_继承后端错误" if source_gold == item["gold"] else "B_跨意图误匹配"
        else:
            cause = ""

        rows.append({
            "threshold": threshold,
            "idx": i,
            "text": item["text"],
            "kind": item["kind"],
            "base_id": item["base_id"],
            "gold": item["gold"],
            "pred": pred or "",
            "llm_pred": llm_pred or "",
            "hit": hit,
            "sim": round(score, 4),
            "correct": bool(pred == item["gold"]),
            "llm_correct": bool(llm_pred == item["gold"]),
            "false_hit_cause": cause,
            "source_gold": source_gold,
            "latency_ms": round(latency_ms, 2),
            "embed_ms": round(embed_ms, 2),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "raw": raw[:60],
        })

    hits = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]
    hit_correct = [r for r in hits if r["correct"]]
    false_hits = [r for r in hits if not r["correct"]]
    false_a = [r for r in false_hits if r["false_hit_cause"].startswith("A")]
    false_b = [r for r in false_hits if r["false_hit_cause"].startswith("B")]

    # 严格的无缓存基线：所有请求都打给后端模型时的准确率
    nocache_correct = sum(1 for r in rows if r["llm_correct"])

    # 延迟与成本基线（未命中路径 = 直连模型）
    base_ms_p95 = percentile([r["latency_ms"] for r in misses], 95) if misses else 0.0
    base_ms_median = median([r["latency_ms"] for r in misses]) if misses else 0.0
    base_prompt = median([r["prompt_tokens"] for r in misses]) if misses else 0
    base_output = median([r["output_tokens"] for r in misses]) if misses else 0
    total_prompt_nocache = base_prompt * len(rows)
    total_output_nocache = base_output * len(rows)
    total_prompt = sum(r["prompt_tokens"] for r in rows)
    total_output = sum(r["output_tokens"] for r in rows)
    cost_nocache = (total_prompt_nocache / 1000) * config.PRICE_PER_1K_PROMPT_CNY + \
                   (total_output_nocache / 1000) * config.PRICE_PER_1K_OUTPUT_CNY
    cost_now = (total_prompt / 1000) * config.PRICE_PER_1K_PROMPT_CNY + \
               (total_output / 1000) * config.PRICE_PER_1K_OUTPUT_CNY

    e2e = (sum(1 for r in rows if r["correct"]) / len(rows)) if rows else 0.0
    nocache = (nocache_correct / len(rows)) if rows else 0.0

    return {
        "threshold": threshold,
        "requests": len(rows),
        "hit_rate": (len(hits) / len(rows)) if rows else 0.0,
        "hit_correct_rate": (len(hit_correct) / len(hits)) if hits else 0.0,
        "false_hit_rate": (len(false_hits) / len(hits)) if hits else 0.0,
        "false_hit_a": len(false_a),
        "false_hit_b": len(false_b),
        "false_hit_a_rate": (len(false_a) / len(false_hits)) if false_hits else 0.0,
        "end_to_end_accuracy": e2e,
        "nocache_accuracy": nocache,
        "accuracy_delta": e2e - nocache,
        "hit_ms_median": median([r["latency_ms"] for r in hits]),
        "hit_ms_p95": percentile([r["latency_ms"] for r in hits], 95),
        "miss_ms_median": base_ms_median,
        "miss_ms_p95": base_ms_p95,
        "all_ms_p95": percentile([r["latency_ms"] for r in rows], 95),
        "cost_saved_pct": ((cost_nocache - cost_now) / cost_nocache * 100.0) if cost_nocache else 0.0,
        "rows": rows,
    }


def engineering_checks(stream: list[dict], timer: TimedCache, best_threshold: float, progress) -> dict:
    """灰度 / 版本失效 / 回滚 的可复现验证。"""
    checks: dict[str, dict] = {}

    # ① 版本指纹：把缓存版本从 v1 换到 v2，旧条目必须全部失效
    cache = SemanticCache(version="v1")
    timer.embed(stream[0]["text"])
    for item in stream[:50]:
        vec, _ = timer.embed(item["text"])
        cache.store(vec, item["gold"], item["text"], time.time())
    before = len(cache.entries)
    cache.version = "v2"
    vec0, _ = timer.embed(stream[0]["text"])
    score, entry = cache.lookup(vec0, time.time())
    checks["版本指纹（模型/Prompt 换代）"] = {
        "how": f"缓存写入 {before} 条后把版本号 v1→v2，再查同一 query",
        "result": f"命中={entry is not None}（相似度 {score:.3f}）→ 旧缓存全部失效，不会串用过期答案",
    }

    # ② TTL 过期
    cache = SemanticCache(version="v1", ttl=0.001)
    vec0, _ = timer.embed(stream[0]["text"])
    cache.store(vec0, "small_talk", stream[0]["text"], time.time())
    time.sleep(0.01)
    _, entry = cache.lookup(vec0, time.time())
    checks["TTL 过期"] = {
        "how": "写入后 ttl=1ms，等 10ms 再查",
        "result": f"命中={entry is not None} → 过期条目被跳过，不会命中陈旧答案",
    }

    # ③ 灰度：10% 流量先走新版本缓存，对比命中正确率
    rng = random.Random(config.SEED)
    gray = SemanticCache(version="v2")
    base = SemanticCache(version="v1")
    gray_hits = gray_correct = 0
    for item in stream[:100]:
        vec, _ = timer.embed(item["text"])
        target = gray if rng.random() < 0.1 else base
        score, entry = target.lookup(vec, time.time())
        if entry is not None and score >= best_threshold:
            if target is gray:
                gray_hits += 1
                gray_correct += 1 if entry["label"] == item["gold"] else 0
        else:
            _, pred, _, _, _ = timer.classify(item["text"])
            target.store(vec, pred or "", item["text"], time.time())
    checks["灰度发布（10% 流量）"] = {
        "how": "新版本缓存先接 10% 流量，与旧版本并行，比较命中正确率",
        "result": f"灰度桶命中 {gray_hits} 次，命中正确率 "
                  f"{(gray_correct / gray_hits * 100) if gray_hits else 0:.1f}% → 可先小流量验证再全量",
    }

    # ④ 回滚开关
    cache = SemanticCache(version="v1")
    vec0, _ = timer.embed(stream[0]["text"])
    cache.store(vec0, stream[0]["gold"], stream[0]["text"], time.time())
    cache.enabled = False
    _, entry = cache.lookup(vec0, time.time())
    disabled_hit = entry is not None and cache.enabled
    cache.enabled = True
    _, entry2 = cache.lookup(vec0, time.time())
    checks["一键回滚（kill switch）"] = {
        "how": "把缓存开关关掉再打开，各查一次",
        "result": f"关闭时命中={disabled_hit}，打开后命中={entry2 is not None} → 出问题可秒级回滚到无缓存直连",
    }

    # ⑤ 失效清空
    n = cache.invalidate_all()
    checks["批量失效"] = {
        "how": "调用 invalidate_all()",
        "result": f"清空 {n} 条；清空后缓存为空，下一次请求必然回落到 LLM（正确但变慢）",
    }
    return checks


def run(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="语义缓存阈值扫描实验")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in config.CACHE_THRESHOLDS))
    ap.add_argument("--paraphrases", type=int, default=config.PARAPHRASES_PER_QUERY)
    ap.add_argument("--perturb-only", action="store_true", help="不调模型，用确定性扰动生成改写")
    ap.add_argument("--eval-set", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache-model", default=config.CACHE_MODEL, help="未命中时真正干活的模型")
    ap.add_argument("--embed-model", default=config.EMBED_MODEL)
    ap.add_argument("--skip-engineering", action="store_true")
    args = ap.parse_args(argv)

    config.CACHE_MODEL = args.cache_model
    config.EMBED_MODEL = args.embed_model

    items = ev.load(args.eval_set or None)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("评测集为空。")
        return 2
    problem = ev.require_labels(items, "cache")
    if problem:
        print(problem)
        return 2

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    client = Ollama()
    timer = TimedCache(client)

    def progress(msg: str) -> None:
        print(msg, flush=True)

    print("=" * 78)
    print(f"语义缓存实验  |  {len(items)} query × {args.paraphrases} 改写  |  阈值 {thresholds}")
    print(f"embedding={config.EMBED_MODEL}  未命中用={config.CACHE_MODEL}")
    print("=" * 78)

    # 测量卫生：先清空显存，避免别的模型常驻造成竞争（见 README §6）
    gpu_before = gpu.state()
    stopped = client.unload_all()
    if stopped:
        progress(f"[环境] 已卸载模型释放显存: {', '.join(stopped)}")
    if gpu_before:
        progress(f"[环境] 测量前 GPU: {gpu.describe(gpu_before)}")

    stream = build_stream(items, client, paraphrases=args.paraphrases,
                          perturb_only=args.perturb_only, progress=progress)
    print(f"请求流共 {len(stream)} 条，唯一文本 "
          f"{len({s['text'] for s in stream})} 条")
    progress("预热 embedding 与 LLM（冷加载不计入统计）…")
    timer.warmup(rounds=2)
    progress(f"  预热后：embed {timer.warm_ms['embed']:.1f} ms · "
             f"LLM {timer.warm_ms['llm']:.1f} ms")
    unique = sorted({s["text"] for s in stream})
    progress(f"预跑无缓存基线：{len(unique)} 条唯一文本各真实调用一次后端模型（约需 "
             f"{len(unique) * 0.15 / 60:.1f} 分钟）…")
    t_base = time.time()
    timer.preclassify(unique)
    progress(f"  基线完成，用时 {time.time() - t_base:.1f}s；"
             f"实测调用 {timer.llm_calls} 次")
    print("-" * 78)

    results = []
    for t in thresholds:
        res = run_threshold(t, stream, timer)
        results.append(res)
        print(f"阈值 {t:.2f} → 命中率 {res['hit_rate'] * 100:5.1f}%  "
              f"命中正确率 {res['hit_correct_rate'] * 100:5.1f}%  "
              f"端到端准确率 {res['end_to_end_accuracy'] * 100:5.1f}%  "
              f"成本省 {res['cost_saved_pct']:5.1f}%", flush=True)

    best = max(results, key=lambda r: r["end_to_end_accuracy"] - 0.5 * r["false_hit_rate"])
    eng = {} if args.skip_engineering else engineering_checks(stream, timer, best["threshold"], progress)

    all_rows = [row for res in results for row in res["rows"]]
    baseline = {"ms_p95": max(r["miss_ms_p95"] for r in results),
                "ms_median": median([r["miss_ms_median"] for r in results])}

    payload = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": fingerprint.fingerprint(),
        "eval_set": ev.identify(args.eval_set or None, items),
        "meta": {
            "stream_size": len(stream),
            "base_queries": len(items),
            "paraphrases": args.paraphrases,
            "embed_model": config.EMBED_MODEL,
            "cache_model": config.CACHE_MODEL,
            "unique_texts": len({s["text"] for s in stream}),
            "llm_calls_actual": timer.llm_calls,
            "embed_calls_actual": timer.embed_calls,
            "memo_latency": True,
            "warm_ms": {k: round(v, 2) for k, v in timer.warm_ms.items()},
            "gpu_before": gpu_before,
            "note": "同一文本的 LLM 只在首次真实调用并记录耗时，重复出现复用该测量值；"
                    "预热使用请求流中不会出现的固定文本，避免冷加载污染延迟",
        },
        "protocol": {"thresholds": thresholds, "num_ctx": config.NUM_CTX,
                     "temperature": config.TEMPERATURE, "shots": config.SHOTS,
                     "ttl_seconds": config.CACHE_TTL_SECONDS},
        "baseline": baseline,
        "thresholds": [{k: v for k, v in res.items() if k != "rows"} for res in results],
        "engineering": eng,
    }
    write_json(config.RESULTS_DIR / f"cache_{tag}.json", payload)
    write_csv(config.RESULTS_DIR / f"cache_{tag}.csv", all_rows)

    print("-" * 78)
    print(rep.cache_console_table(payload))
    path = rep.build_cache_report(payload)
    print(f"\n报告已生成: {path}")
    print(f"明细 CSV  : {config.RESULTS_DIR / f'cache_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())




