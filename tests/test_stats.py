"""统计与落盘工具测试。"""
from __future__ import annotations

from pathlib import Path

from inferbench import stats


def test_median_robust_to_outlier():
    """中位数必须抗极端值——这是延迟测量用中位数而不是平均数的原因。"""
    values = [10, 11, 12, 13, 1000]
    assert stats.median(values) == 12
    assert stats.mean(values) > 200


def test_percentile_known_values():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert stats.percentile(vals, 50) == 5.5
    assert stats.percentile(vals, 0) == 1
    assert stats.percentile(vals, 100) == 10
    # 线性插值：P95 应落在 9 与 10 之间
    assert 9.0 <= stats.percentile(vals, 95) <= 10.0


def test_percentile_handles_edge_cases():
    assert stats.percentile([], 95) == 0.0
    assert stats.percentile([7], 95) == 7.0


def test_cosine_similarity_properties():
    assert stats.cosine([1, 0], [1, 0]) == 1.0
    assert stats.cosine([1, 0], [0, 1]) == 0.0
    assert stats.cosine([1, 0], [-1, 0]) == -1.0
    # 零向量与长度不一致必须安全返回 0，不能抛异常
    assert stats.cosine([0, 0], [1, 1]) == 0.0
    assert stats.cosine([1, 2], [1, 2, 3]) == 0.0


def test_csv_roundtrip_keeps_chinese(tmp_path: Path):
    path = tmp_path / "out.csv"
    rows = [{"模型": "qwen3:1.7b", "准确率": "92.0%"}, {"模型": "qwen3:0.6b", "准确率": "54.0%"}]
    stats.write_csv(path, rows)
    text = path.read_text(encoding="utf-8-sig")
    assert "qwen3:1.7b" in text and "准确率" in text


def test_write_json_and_read_back(tmp_path: Path):
    path = tmp_path / "out.json"
    payload = {"a": [1, 2], "中文": "值"}
    stats.write_json(path, payload)
    assert stats.read_json(path) == payload

