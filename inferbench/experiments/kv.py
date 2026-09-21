"""实验 6 · KV Cache 扫参：上下文长度 → 显存 / 速度 / 质量

## 回答的问题

    「`num_ctx` 该定多少？」

## 为什么这件事值得单独做一轮实验

**KV cache 是按 `num_ctx` 满额预分配的，跟你实际 prompt 有多长无关。**
所以「上下文设大一点没坏处」是错的——它在你的请求还远没用满之前，
就把显存吃光了。而 `num_ctx` 恰恰是最容易被随手写大的一个数：
很多框架的默认值是模型声明的上限（Qwen3 是 40960），
甚至更离谱的占位值（131072）。

本实验把「上下文长度」从一句话变成三笔账：

1. **显存账**：每 1K 上下文多少 GB（实测斜率），以及它和模型权重谁更贵；
2. **溢出账**：涨到哪一档会从 GPU 溢到 CPU —— 这是**速度断崖**，不是渐变；
3. **质量账**：`num_ctx` 变大会不会改变准确率（如果不变，它就纯粹是成本项）。

## 跑法

    python -m inferbench kv                                  # 默认 8 档，100 条评测集
    python -m inferbench kv --levels 2048,4096,8192 --limit 20   # 冒烟
    python -m inferbench kv --model qwen3:0.6b               # 换模型重扫

## 产出

    results/kv_<tag>.csv    请求级明细
    results/kv_<tag>.json   汇总 + 结构参数 + 理论 KV 值 + 全量记录
    results/kv_<tag>.md     报告（含上下文→显存曲线与结论）
"""
from __future__ import annotations

import argparse
import sys
import time

from inferbench import config
from inferbench import eval_set as ev
from inferbench import fingerprint
from inferbench import gguf
from inferbench import llama
from inferbench import report as rep
from inferbench.bench import measure_model
from inferbench.ollama import Ollama
from inferbench.stats import write_csv, write_json


def _parse_levels(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            print(f"忽略无法解析的上下文档位: {part!r}")
            continue
        if val > 0:
            out.append(val)
    return out


def run(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="KV cache 上下文扫参实验")
    ap.add_argument("--model", default=config.KV_MODEL, help="被扫的模型（保持同一个模型）")
    ap.add_argument("--levels", default=",".join(str(x) for x in config.KV_CONTEXT_LEVELS),
                    help="逗号分隔的 num_ctx 档位")
    ap.add_argument("--repeats", type=int, default=config.KV_REPEATS)
    ap.add_argument("--shots", type=int, default=config.SHOTS)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--eval-set", default="", help="自定义 jsonl 评测集路径")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    levels = _parse_levels(args.levels)
    if not levels:
        print("没有可用的上下文档位。")
        return 2

    items = ev.load(args.eval_set or None)
    if not items:
        print("评测集为空。请确认 D:\\PYTHON\\Synapse\\tools\\intent_testset.py 存在，"
              "或用 --eval-set 指定 jsonl。")
        return 2
    problem = ev.require_labels(items, "kv")
    if problem:
        print(problem)
        return 2
    if args.limit:
        items = items[: args.limit]

    snap = ev.summarize(items)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")

    # ---- 结构参数：KV cache 的理论大小（不需要 GPU，读 GGUF 头即可）----
    geom: dict = {}
    theory: dict = {}
    try:
        geom = gguf.geometry(llama.resolve(args.model))
        theory = {
            "bytes_per_token": gguf.kv_bytes_per_token(geom),
            "gb_per_1k_tokens": round(gguf.kv_gb_per_1k_tokens(geom), 4),
            "declared_context": geom.get("context_length"),
        }
        if geom.get("context_length"):
            theory["kv_gb_at_declared_context"] = round(
                gguf.kv_gb_for_context(geom, int(geom["context_length"])), 3)
    except (gguf.GgufError, FileNotFoundError) as exc:
        print(f"⚠ 读不到 GGUF 结构参数（理论 KV 值将缺失）：{exc}")

    print("=" * 78)
    print(f"KV cache 上下文扫参  |  模型 {args.model}  |  评测集 {snap['total']} 条"
          f"（难例 {snap['hard']}）  |  重复 {args.repeats} 次")
    print(f"档位: {', '.join(str(x) for x in levels)}")
    if geom:
        print(f"结构: {gguf.describe(geom)}")
        print(f"理论 KV: {theory['bytes_per_token']:.0f} B/token = "
              f"{theory['gb_per_1k_tokens']:.4f} GB / 1K token"
              f"（F16 KV，K+V 各一份）")
    print(f"测量口径: temperature={config.TEMPERATURE}  think=False  seed={config.SEED}  "
          f"few-shot {args.shots}  预热后丢弃  每档前先卸载模型清空显存")
    print("=" * 78)

    client = Ollama()
    print(f"Ollama 版本: {client.version()}")

    runs: list[dict] = []
    all_records: list[dict] = []
    for ctx in levels:
        # 每档都先卸载：既要拿干净的显存基线，也要让 Ollama 真的用新 num_ctx 重新加载
        client.unload(args.model)
        time.sleep(2.0)

        print("-" * 78)
        run_data = measure_model(client, args.model, items, repeats=args.repeats,
                                 shots=args.shots, num_ctx=ctx, isolate=False,
                                 progress=lambda msg: print(msg, flush=True))
        summary = run_data.get("summary") or {}
        footprint = run_data.get("footprint") or {}
        gpu_before = (summary.get("gpu_before") or {}).get("mem_used_mb")
        gpu_after = (summary.get("gpu_after") or {}).get("mem_used_mb")

        loaded_ctx = int(footprint.get("context_loaded") or 0)
        entry = {
            "requested_ctx": ctx,
            "loaded_ctx": loaded_ctx,
            "truncated": bool(loaded_ctx and loaded_ctx < ctx),
            "footprint_gb": round(float(footprint.get("footprint_gb") or 0.0), 3),
            "vram_gb": round(float(footprint.get("vram_gb") or 0.0), 3),
            "gpu_ratio": round(float(footprint.get("gpu_ratio") or 0.0), 1),
            "processor": footprint.get("processor", ""),
            "vram_delta_mb": (int(gpu_after) - int(gpu_before))
                             if (gpu_before is not None and gpu_after is not None) else None,
            "summary": summary,
        }
        if entry["truncated"]:
            print(f"⚠ 请求 num_ctx={ctx}，实际生效 {loaded_ctx} —— "
                  f"被静默截断到模型声明的上下文上限")
        if entry["gpu_ratio"] and entry["gpu_ratio"] < config.GPU_RATIO_OK:
            print(f"⚠ 该档未全量上 GPU（{entry['processor']}）—— 速度数字已受 CPU 拖累，"
                  f"这一档就是**溢出拐点**")
        runs.append(entry)
        all_records.extend(run_data.get("records") or [])

    payload = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": fingerprint.fingerprint(),
        "eval_set": ev.identify(args.eval_set or None, items),
        "protocol": {
            "model": args.model, "levels": levels, "temperature": config.TEMPERATURE,
            "think": False, "seed": config.SEED, "shots": args.shots,
            "repeats": args.repeats, "warmup_calls": config.WARMUP_CALLS,
            "kv_cache_type": config.KV_CACHE_TYPE,
        },
        "gguf": geom,
        "kv_theory": theory,
        "runs": runs,
        "records": all_records,
    }
    write_json(config.RESULTS_DIR / f"kv_{tag}.json", payload)
    if all_records:
        write_csv(config.RESULTS_DIR / f"kv_{tag}.csv", all_records)

    print("=" * 78)
    print(rep.kv_console_table(payload))
    md_path = rep.build_kv_report(payload)
    print(f"\n报告已生成: {md_path}")
    print(f"明细 CSV  : {config.RESULTS_DIR / f'kv_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
