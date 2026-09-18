"""pytest 公共配置：把工程根目录加进 sys.path，并提供一个"无副作用"的假结果载荷。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def quant_payload() -> dict:
    """一份最小的量化实验结果（三档），用于测报告生成。"""
    def run(model, mem, acc, hard, ttft, tps):
        return {"model": model, "summary": {
            "model": model, "footprint_gb": mem, "accuracy": acc, "accuracy_hard": hard,
            "accuracy_easy": acc, "invalid_rate": 0.0, "n_hard": 23, "n_records": 300,
            "ttft_ms_median": ttft, "ttft_ms_p95": ttft * 1.2,
            "decode_tps_median": tps, "decode_tps_mean": tps,
            "wall_ms_median": 100.0, "wall_ms_p95": 150.0,
            "processor": "100% GPU", "gpu_before": {"mem_used_mb": 1200, "mem_total_mb": 8188},
            "confusion": {}, "request_errors": 0,
        }}
    return {
        "tag": "unit", "created_at": "2026-01-01 00:00:00",
        "eval_set": {"total": 100, "hard": 23, "by_intent": {}},
        "protocol": {"num_ctx": 4096, "temperature": 0.0, "seed": 42, "shots": 3,
                     "repeats": 3, "warmup_calls": 1},
        "runs": [run("qwen3:1.7b-fp16", 3.69, 0.91, 0.783, 18.5, 85.3),
                 run("qwen3:1.7b-q8_0", 2.24, 0.92, 0.783, 13.6, 134.2),
                 run("qwen3:1.7b", 1.59, 0.87, 0.696, 10.7, 191.6)],
        "records": [{"item_id": i, "hard": i < 23} for i in range(100)],
    }


@pytest.fixture
def cache_payload() -> dict:
    """一份最小的语义缓存结果，用于测控制台表格不会把百分数再乘 100。"""
    def row(thr, hit, false_hit, cost):
        return {"threshold": thr, "hit_rate": hit, "hit_correct_rate": 1 - false_hit,
                "false_hit_rate": false_hit, "false_hit_a": 18, "false_hit_b": 0,
                "end_to_end_accuracy": 0.86, "nocache_accuracy": 0.86,
                "hit_ms_p95": 100.0, "miss_ms_p95": 150.0, "cost_saved_pct": cost}
    return {
        "tag": "unit", "created_at": "2026-01-01 00:00:00",
        "meta": {"stream_size": 300, "unique_texts": 230, "base_queries": 100,
                 "paraphrases": 2, "embed_model": "e", "cache_model": "c"},
        "baseline": {"ms_p95": 154.2, "ms_median": 90.0},
        "thresholds": [row(0.80, 0.607, 0.159, 60.4), row(0.92, 0.423, 0.142, 42.4)],
        "engineering": {},
    }
