"""报告生成测试。

包含一条**回归测试**：成本百分比不能被重复放大 100 倍
（曾经在控制台表格里把已经是百分数的 `cost_saved_pct` 又乘了一次 100，
显示成 4996%，报告里却是对的——这类"只错在展示层"的 bug 最容易蒙过眼睛）。
"""
from __future__ import annotations

import re
from pathlib import Path

from inferbench import report as rep, svg


def test_svg_bar_chart_is_wellformed():
    out = svg.bar_chart(["A", "B"], {"x": [1, 2], "y": [3, 4]}, title="t", y_label="u")
    assert out.startswith("<svg") and out.endswith("</svg>")
    assert out.count("<svg") == out.count("</svg>") == 1
    assert "A" in out and "B" in out


def test_svg_line_chart_handles_single_point():
    """只有一个 x 值时不能除零崩掉。"""
    out = svg.line_chart([0.9], {"命中率": [42.0]}, title="t", x_label="thr")
    assert out.startswith("<svg") and "</svg>" in out


def test_svg_escapes_html_chars():
    out = svg.bar_chart(["<script>"], {"a": [1]}, title="t")
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_cost_percent_is_not_multiplied_twice(cache_payload):
    """回归：控制台表格里的成本节省必须和 JSON 里的数值一致。"""
    table = rep.cache_console_table(cache_payload)
    assert "42.4%" in table, "成本节省应显示 42.4%"
    assert "4240" not in table and "4996" not in table, "百分数被重复乘了 100"


def test_quant_rank_orders_precision():
    assert rep.quant_rank("qwen3:1.7b-fp16") > rep.quant_rank("qwen3:1.7b-q8_0")
    assert rep.quant_rank("qwen3:1.7b-q8_0") > rep.quant_rank("qwen3:1.7b")
    # 没写量化后缀时给一个中间默认值，不能崩
    assert rep.quant_rank("some-model:latest") == 2.0


def test_build_quant_report_detects_non_monotonic_accuracy(quant_payload, tmp_path, monkeypatch):
    """Q8 准确率高于 FP16 时，报告必须自动高亮这个反直觉信号。"""
    monkeypatch.setattr(rep.config, "RESULTS_DIR", tmp_path)
    path = rep.build_quant_report(quant_payload)
    text = Path(path).read_text(encoding="utf-8")

    assert "准确率不随精度单调变化" in text
    assert "qwen3:1.7b-q8_0" in text and "92.0%" in text
    # 必须同时给出"FP16 被支配"的选型结论，而不只是陈述现象
    assert "FP16 在纯推理场景被完全支配" in text
    # 三个必需小节
    for section in ["## 三角表", "## 测量卫生", "## 可写进简历的结论句"]:
        assert section in text


def test_build_quant_report_counts_hard_items_once(quant_payload, tmp_path, monkeypatch):
    """难例数必须按唯一题目去重，不能被重复次数放大。"""
    monkeypatch.setattr(rep.config, "RESULTS_DIR", tmp_path)
    text = Path(rep.build_quant_report(quant_payload)).read_text(encoding="utf-8")
    assert "难例 23 题" in text
    assert "难例 69" not in text          # 23 × 3 次重复 = 69，是错误口径


def test_build_cache_report_mentions_attribution(cache_payload, tmp_path, monkeypatch):
    monkeypatch.setattr(rep.config, "RESULTS_DIR", tmp_path)
    text = Path(rep.build_cache_report(cache_payload)).read_text(encoding="utf-8")
    assert "继承后端模型" in text
    assert "复现" in text or "可写进简历" in text


def test_report_files_are_valid_utf8_without_bom(quant_payload, tmp_path, monkeypatch):
    monkeypatch.setattr(rep.config, "RESULTS_DIR", tmp_path)
    raw = Path(rep.build_quant_report(quant_payload)).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert re.search(rb"[\x80-\xff]", raw)          # 确实有中文
    raw.decode("utf-8")

