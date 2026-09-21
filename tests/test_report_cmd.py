"""`python -m inferbench report` 的结果分类测试。

包含一条**回归测试**：实验三（门槛 A/B）的结果里也有 `meta.stream_size`，
早先的分类顺序先判 cache，导致 `cache_gate_full.json` 被当成实验二重出成
`cache_full.md`（gate 的报告永远不更新），而且同样的文件名被打印两次。
"""
from __future__ import annotations

import pytest

from inferbench.experiments import report_cmd


@pytest.mark.parametrize("payload,expected", [
    ({"meta": {"target": "qwen3:1.7b"}}, "spec"),
    ({"meta": {"levels": [1, 2]}}, "load"),
    ({"meta": {"stream_size": 300}, "off": {}, "on": {}}, "gate"),
    ({"meta": {"stream_size": 300}, "thresholds": []}, "cache"),
    ({"meta": {}, "runs": [], "eval_set": {}}, "quant"),
    ({"meta": {}, "runs": [], "eval_set": {}, "kv_theory": {}}, "kv"),
    ({"meta": {}}, "unknown"),
])
def test_kind_of_classifies_each_experiment(payload, expected):
    assert report_cmd.kind_of(payload) == expected


def test_kv_is_not_misclassified_as_quant():
    """回归：实验六的结果同样有 `runs` + `eval_set`，不能被 quant 抢走。"""
    kv_like = {"runs": [{"requested_ctx": 2048}], "eval_set": {"total": 100},
               "kv_theory": {"gb_per_1k_tokens": 0.1094}, "gguf": {}}
    assert report_cmd.kind_of(kv_like) == "kv"


def test_gate_is_not_misclassified_as_cache():
    """回归：gate 的结果必须判成 gate，不能被 cache 抢走。"""
    gate_like = {"meta": {"stream_size": 300, "unique_texts": 230},
                 "off": {"hit_rate": 0.42}, "on": {"hit_rate": 0.41},
                 "thresholds": [{"threshold": 0.92}]}
    assert report_cmd.kind_of(gate_like) == "gate"


def test_real_result_files_classify_correctly(tmp_path, monkeypatch):
    """如果 results/ 下已有真实结果，逐个检查分类是否与文件名一致。"""
    from inferbench import config
    from inferbench.stats import read_json

    mapping = {
        "quant_ladder-fp16.json": "quant",
        "cache_full.json": "cache",
        "cache_gate_full.json": "gate",
        "spec_full.json": "spec",
        "load_full.json": "load",
        "kv_full.json": "kv",
    }
    checked = 0
    for name, expected in mapping.items():
        path = config.RESULTS_DIR / name
        if not path.exists():
            continue
        assert report_cmd.kind_of(read_json(path)) == expected, f"{name} 分类错误"
        checked += 1
    _ = (tmp_path, monkeypatch)
    if checked == 0:
        pytest.skip("results/ 下没有结果文件")
