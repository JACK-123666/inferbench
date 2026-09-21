"""重出报告：从已有的 results/*.json 重新生成 Markdown（不重跑实验）。

改报告文案后用它，避免为了改一句话再跑几百次模型调用。

    python -m inferbench report results/spec_full.json
    python -m inferbench report results/quant_ladder-fp16.json results/cache_full.json
"""
from __future__ import annotations

import sys
from pathlib import Path

from inferbench import report as rep
from inferbench.stats import read_json


def kind_of(payload: dict) -> str:
    """判别结果文件属于哪个实验。

    注意判别顺序：实验三（门槛 A/B）的结果里也有 `meta.stream_size`，
    如果先判 cache 就会把它当成实验二重出，导致 gate 的报告永远不更新（踩过）。
    """
    if payload.get("kind") == "recommend":            # 选型建议：自带 kind，最优先
        return "recommend"
    if payload.get("meta", {}).get("target"):
        return "spec"
    if payload.get("meta", {}).get("levels"):
        return "load"
    if "off" in payload and "on" in payload:          # 实验三：门槛开关 A/B 对照
        return "gate"
    if payload.get("meta", {}).get("stream_size") is not None:
        return "cache"
    if "runs" in payload and "eval_set" in payload:
        return "quant"
    return "unknown"


def run(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    for arg in args:
        path = Path(arg)
        if not path.exists():
            print(f"找不到 {path}")
            continue
        payload = read_json(path)
        kind = kind_of(payload)
        if kind == "quant":
            out = rep.build_quant_report(payload)
        elif kind == "cache":
            out = rep.build_cache_report(payload)
        elif kind == "gate":
            from inferbench.experiments import gate
            out = gate.build_report(payload)
        elif kind == "spec":
            from inferbench.experiments import spec
            out = spec.build_report(payload)
        elif kind == "load":
            from inferbench.experiments import load
            out = load.build_report(payload)
        elif kind == "recommend":
            from inferbench import recommend
            out = recommend.build_report(payload)
        else:
            print(f"{path} 类型无法识别")
            continue
        print(f"已重新生成: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


