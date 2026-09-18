"""统计与输出工具（纯标准库）。"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


def median(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else 0.0


def mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else 0.0


def percentile(values: list[float], pct: float) -> float:
    """线性插值百分位（P95 等）。"""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(vals[int(k)])
    return float(vals[lo] + (vals[hi] - vals[lo]) * (k - lo))


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（纯 Python，向量规模小，不需要 numpy）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM 让 Excel 正确识别中文
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in names})
    return path


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole else 0.0


def fnum(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"
