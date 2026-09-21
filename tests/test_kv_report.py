"""KV cache 扫参报告测试。

核心要守住的一条：**拟合只能用"健康点"**。
被静默截断的档位（请求 65536、实际生效 40960）如果按请求值进拟合，
斜率会被压平；溢出到 CPU 的档位显存已经被 offload 限制住，同样会压平斜率。
这两类点混进去，报告给出的「每 1K 上下文多少 GB」就是错的。
"""
from __future__ import annotations

import pytest

from inferbench import report


def test_fit_uses_only_healthy_points(kv_payload: dict):
    a = report.kv_analysis(kv_payload)
    assert a["n_points"] == 5, "被截断 / 溢出的 2 档必须排除在拟合之外"
    assert a["gb_per_1k"] == pytest.approx(0.1094, abs=0.001)
    assert a["weights_gb"] == pytest.approx(1.80, abs=0.02)


def test_slope_is_cross_checked_against_gguf_theory(kv_payload: dict):
    a = report.kv_analysis(kv_payload)
    # 实测斜率与按 GGUF 结构参数算出的理论值应当对得上（本 fixture 里是构造的，误差为 0）
    assert abs(a["slope_error_pct"]) < 1.0


def test_crossover_is_where_kv_equals_weights(kv_payload: dict):
    a = report.kv_analysis(kv_payload)
    # 1.80 GB / (0.1094/1024 GB per token) ≈ 16848 token
    assert a["crossover_ctx"] == pytest.approx(16848, rel=0.02)
    # 交叉点必须落在扫过的区间内，否则这条结论没有实测支撑
    levels = [r["loaded_ctx"] for r in kv_payload["runs"]]
    assert min(levels) < a["crossover_ctx"] < max(levels)


def test_detects_silent_truncation_and_spill(kv_payload: dict):
    a = report.kv_analysis(kv_payload)
    assert [r["requested_ctx"] for r in a["truncated"]] == [65536]
    assert all(r["loaded_ctx"] == 40960 for r in a["truncated"])
    assert sorted(r["requested_ctx"] for r in a["spilled"]) == [40960, 65536]


def test_console_table_flags_the_two_failure_modes(kv_payload: dict):
    text = report.kv_console_table(kv_payload)
    assert "静默截断" in text
    assert "溢出到CPU" in text
    # 健康档位不应被误标
    assert text.count("静默截断") == 1


def test_console_table_empty_payload():
    assert "没有成功" in report.kv_console_table({})


def test_report_contains_the_four_claims(kv_payload: dict, tmp_path, monkeypatch):
    monkeypatch.setattr(report.config, "RESULTS_DIR", tmp_path)
    path = report.build_kv_report(kv_payload)
    md = path.read_text(encoding="utf-8")

    assert path.name == "kv_unit.md"
    # 1) 斜率 + 与理论值对账
    assert "GB / 1K token" in md
    # 2) KV 追平权重的交叉点
    assert "追平" in md or "超过模型权重" in md
    # 3) 静默截断（本实验最意外的发现）
    assert "静默截断" in md and "40960" in md
    # 4) 溢出到 CPU 是台阶不是斜坡
    assert "offload" in md
    # 5) 质量差异必须归因到执行路径，而不是"上下文变长让模型变准了"
    assert "逐条比对" in md
    assert "不是上下文变长让模型变准了" in md
    # 图表不能是空的
    assert "<svg" in md


def test_quality_diff_is_attributed_to_execution_path(kv_payload: dict, tmp_path, monkeypatch):
    """溢出档准确率**更高**这件事，必须被解释成执行路径分叉，不能当成功劳。"""
    monkeypatch.setattr(report.config, "RESULTS_DIR", tmp_path)
    a = report.kv_analysis(kv_payload)
    assert a["ref_ctx"] == 32768
    healthy = a["per_level"][32768]
    assert healthy["identical"] is True
    spilled = a["per_level"][40960]
    assert spilled["n_diff"] == 2          # item 14 × 2 次重复
    assert spilled["diff_items"] == ["14"]


def test_report_says_pure_cost_when_no_level_differs(kv_payload: dict, tmp_path, monkeypatch):
    """所有档位预测一致时，才可以说「num_ctx 是一笔纯成本」。"""
    monkeypatch.setattr(report.config, "RESULTS_DIR", tmp_path)
    for rec in kv_payload["records"]:
        rec["pred"] = "summarize"
    md = report.build_kv_report(kv_payload).read_text(encoding="utf-8")
    assert "纯成本" in md
    assert "逐条比对" in md


def test_report_survives_missing_gguf_theory(kv_payload: dict, tmp_path, monkeypatch):
    """读不到 GGUF（模型不是本地文件）时报告仍要能出，只是少掉对账那段。"""
    monkeypatch.setattr(report.config, "RESULTS_DIR", tmp_path)
    kv_payload["gguf"] = {}
    kv_payload["kv_theory"] = {}
    md = report.build_kv_report(kv_payload).read_text(encoding="utf-8")
    assert "扫参结果" in md
    assert "关键结论" in md


def test_report_handles_no_successful_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(report.config, "RESULTS_DIR", tmp_path)
    path = report.build_kv_report({"tag": "empty", "runs": [], "protocol": {}})
    assert "没有成功的记录" in path.read_text(encoding="utf-8")


def test_analysis_tolerates_single_point():
    """只有一档时不能除以零，也不能假装拟合出了斜率。"""
    a = report.kv_analysis({"runs": [{"requested_ctx": 2048, "loaded_ctx": 2048,
                                      "truncated": False, "footprint_gb": 2.02,
                                      "gpu_ratio": 100.0, "summary": {}}]})
    assert a["gb_per_token"] == 0.0
    assert a["crossover_ctx"] == 0.0
