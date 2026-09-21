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


@pytest.fixture
def kv_payload() -> dict:
    """一份最小的 KV cache 扫参结果。

    前 5 档刻意做成**完全线性**（weight 1.80 GB + 0.1094 GB/1K token），
    最后两档是「被截断且溢出」的坏点 —— 拟合必须把它们排除掉，
    否则斜率会被压平，结论就错了。
    """
    per_token = 0.1094 / 1024.0

    def point(requested: int, *, loaded: int, ratio: float, acc: float, tps: float,
              ttft: float) -> dict:
        footprint = 1.80 + loaded * per_token
        return {
            "requested_ctx": requested, "loaded_ctx": loaded,
            "truncated": loaded < requested,
            "footprint_gb": round(footprint, 3), "vram_gb": round(footprint, 3),
            "gpu_ratio": ratio, "processor": "100% GPU" if ratio >= 99.5 else "86% GPU / 14% CPU",
            "vram_delta_mb": int(footprint * 1024),
            "summary": {
                "model": "qwen3:1.7b-q8_0", "footprint_gb": round(footprint, 3),
                "accuracy": acc, "accuracy_hard": 0.783, "accuracy_easy": acc,
                "invalid_rate": 0.01, "n_hard": 23, "n_records": 300,
                "ttft_ms_median": ttft, "ttft_ms_p95": ttft * 1.2,
                "decode_tps_median": tps, "decode_tps_mean": tps,
                "wall_ms_median": 90.0, "wall_ms_p95": 150.0,
                "processor": "100% GPU", "confusion": {}, "request_errors": 0,
                "gpu_before": {"mem_used_mb": 1500, "mem_total_mb": 8188},
            },
        }

    return {
        "tag": "unit", "created_at": "2026-01-01 00:00:00",
        "eval_set": {"total": 100, "hard": 23, "by_intent": {}},
        "protocol": {"model": "qwen3:1.7b-q8_0", "levels": [2048, 4096, 8192, 16384,
                                                             32768, 40960, 65536],
                     "temperature": 0.0, "think": False, "seed": 42, "shots": 3,
                     "repeats": 3, "warmup_calls": 1, "kv_cache_type": "f16"},
        "gguf": {"architecture": "qwen3", "block_count": 28, "head_count": 16,
                 "head_count_kv": 8, "key_length": 128, "value_length": 128,
                 "embedding_length": 2048, "context_length": 40960, "complete": True},
        "kv_theory": {"bytes_per_token": 114688, "gb_per_1k_tokens": 0.1094,
                      "declared_context": 40960, "kv_gb_at_declared_context": 4.375},
        "runs": [
            point(2048, loaded=2048, ratio=100.0, acc=0.92, tps=115.2, ttft=15.3),
            point(4096, loaded=4096, ratio=100.0, acc=0.92, tps=116.2, ttft=15.3),
            point(8192, loaded=8192, ratio=100.0, acc=0.91, tps=112.0, ttft=18.0),
            point(16384, loaded=16384, ratio=100.0, acc=0.92, tps=105.0, ttft=25.0),
            point(32768, loaded=32768, ratio=100.0, acc=0.92, tps=90.0, ttft=40.0),
            point(40960, loaded=40960, ratio=86.0, acc=0.91, tps=88.0, ttft=41.0),
            point(65536, loaded=40960, ratio=86.0, acc=0.92, tps=90.7, ttft=40.9),
        ],
        # 请求级记录：32768（健康）与 40960（溢出）两档各 8 条，
        # 其中 item 14 在健康档输出非法标签、在溢出档翻成正确标签 ——
        # 报告必须能识别出「准确率差异来自执行路径，不是上下文」。
        "records": (
            [{"item_id": i, "rep": r, "num_ctx": 32768, "hard": i < 23,
              "pred": "chain_of_thought" if i == 14 else "summarize"}
             for i in (1, 2, 14, 15) for r in range(2)]
            + [{"item_id": i, "rep": r, "num_ctx": 40960, "hard": i < 23,
                "pred": "knowledge_retrieval" if i == 14 else "summarize"}
               for i in (1, 2, 14, 15) for r in range(2)]
        ),
    }
