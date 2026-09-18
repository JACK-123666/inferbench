"""任务定义测试：标签解析必须严格，评测打分必须能暴露混淆。"""
from __future__ import annotations

import pytest

from inferbench import tasks


@pytest.mark.parametrize("raw,expected", [
    ("knowledge_retrieval", "knowledge_retrieval"),
    ("  summarize  ", "summarize"),
    ("small_talk", "small_talk"),
    ("`summarize`", "summarize"),                    # 带反引号
    ("标签：summarize", "summarize"),                 # 带前缀
    ("知识检索", "knowledge_retrieval"),               # 中文同义
    ("摘要", "summarize"),
    ("闲聊", "small_talk"),
    ("", None),                                       # 空输出 = 非法
    ("我不知道该怎么分类这个问题", None),              # 完全跑题 = 非法
])
def test_parse_label(raw, expected):
    assert tasks.parse_label(raw) == expected


def test_parse_label_never_defaults_to_a_guess():
    """关键性质：解析不出来必须返回 None（记非法输出），而不是随便给个标签。

    非法输出率本身就是一个量化损伤指标，如果这里"兜底成某一类"，
    这个指标就失真了。
    """
    assert tasks.parse_label("The answer is 42") is None
    assert tasks.parse_label("???") is None


def test_build_messages_respects_shot_count():
    zero = tasks.build_messages("你好", shots=0)
    three = tasks.build_messages("你好", shots=3)
    assert len(zero) == 2                    # system + user
    assert len(three) == 2 + 2 * 3           # system + 3 组 (user, assistant) + user
    assert zero[0]["role"] == "system" and zero[-1]["content"] == "你好"


def test_score_separates_easy_and_hard():
    records = [
        {"gold": "summarize", "pred": "summarize", "hard": False, "correct": True},
        {"gold": "summarize", "pred": "small_talk", "hard": True, "correct": False},
        {"gold": "small_talk", "pred": "small_talk", "hard": True, "correct": True},
        {"gold": "small_talk", "pred": "", "hard": True, "correct": False},
    ]
    s = tasks.score(records)
    assert s["n"] == 4 and s["n_hard"] == 3
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["accuracy_easy"] == pytest.approx(1.0)
    assert s["accuracy_hard"] == pytest.approx(1 / 3)
    assert s["invalid_rate"] == pytest.approx(0.25)   # 1 条非法输出
    # 混淆矩阵应把空预测记成 INVALID
    assert s["confusion"]["small_talk"].get("INVALID") == 1


def test_score_on_empty_input_does_not_crash():
    assert tasks.score([]) == {}

