"""实验 1 · 量化档位对比：显存 ↓ / 速度 ↑ / 精度 ↓ 三角表

跑法::

    python -m inferbench quant                      # 跑 config.QUANT_MODELS 里的三档
    python -m inferbench quant --models qwen3:1.7b,qwen3:1.7b-q8_0,qwen3:1.7b-fp16
    python -m inferbench quant --limit 20 --repeats 1   # 冒烟测试

产出::

    results/quant_<tag>.csv    单条请求级明细（可自己再切分统计）
    results/quant_<tag>.json   汇总 + 全量记录
    results/quant_<tag>.md     报告（含三角表 + 简历可用结论句）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


from inferbench import config  # noqa: E402
from inferbench import fingerprint  # noqa: E402
from inferbench import eval_set as ev  # noqa: E402
from inferbench import report as rep  # noqa: E402
from inferbench.ollama import Ollama  # noqa: E402
from inferbench.bench import measure_model  # noqa: E402
from inferbench.stats import write_csv, write_json  # noqa: E402


def run(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="量化档位对比实验")
    ap.add_argument("--models", default=",".join(config.QUANT_MODELS),
                    help="逗号分隔的模型 tag")
    ap.add_argument("--repeats", type=int, default=config.REPEATS)
    ap.add_argument("--shots", type=int, default=config.SHOTS)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--eval-set", default="", help="自定义 jsonl 评测集路径")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    items = ev.load(args.eval_set or None)
    if not items:
        print("评测集为空。请确认 D:\\PYTHON\\Synapse\\tools\\intent_testset.py 存在，"
              "或用 --eval-set 指定 jsonl。")
        return 2
    problem = ev.require_labels(items, "quant")
    if problem:
        print(problem)
        return 2
    if args.limit:
        items = items[: args.limit]

    info = ev.summarize(items)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")

    print("=" * 78)
    print(f"量化档位对比实验  |  评测集 {info['total']} 条（难例 {info['hard']}）"
          f"  |  重复 {args.repeats} 次  |  few-shot {args.shots}")
    print(f"测量口径: num_ctx={config.NUM_CTX}  temperature={config.TEMPERATURE}  "
          f"think=False  seed={config.SEED}  预热 {config.WARMUP_CALLS} 次后丢弃")
    print("=" * 78)

    client = Ollama()
    print(f"Ollama 版本: {client.version()}")
    loaded = {m.get("name") for m in client.ps()}
    print(f"当前已加载: {', '.join(sorted(loaded)) if loaded else '（无）'}")
    print("-" * 78)

    runs: list[dict] = []
    all_records: list[dict] = []
    for model in models:
        run = measure_model(client, model, items, repeats=args.repeats, shots=args.shots,
                            progress=lambda msg: print(msg, flush=True))
        runs.append(run)
        all_records.extend(run.get("records", []))
        print("-" * 78)

    payload = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": fingerprint.fingerprint(),
        "eval_set": ev.identify(args.eval_set or None, items),
        "protocol": {
            "num_ctx": config.NUM_CTX, "temperature": config.TEMPERATURE,
            "think": False, "seed": config.SEED, "shots": args.shots,
            "repeats": args.repeats, "warmup_calls": config.WARMUP_CALLS,
        },
        "runs": [{k: v for k, v in r.items() if k != "records"} for r in runs],
        "records": all_records,
    }
    write_json(config.RESULTS_DIR / f"quant_{tag}.json", payload)
    if all_records:
        write_csv(config.RESULTS_DIR / f"quant_{tag}.csv", all_records)

    print(rep.quant_console_table(payload))
    md_path = rep.build_quant_report(payload)
    print(f"\n报告已生成: {md_path}")
    print(f"明细 CSV  : {config.RESULTS_DIR / f'quant_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())




