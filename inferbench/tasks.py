"""任务定义：意图分类（与 Synapse 评测集完全对齐）。

为什么选这个任务做量化对比：
- **客观**：三分类，金标准人工标注，准确率是精确匹配，没有 LLM-as-judge 的噪声；
- **省 token**：输出只有 1 个标签（≤8 token），1200 次调用几分钟跑完；
- **有区分度**：含 23 条「模糊难例」（hard），量化掉幅往往先在难例上暴露；
- **自带失败模式**：量化后指令跟随能力下降会直接表现为「输出非法标签」，
  invalid_rate 本身就是一个量化损伤指标。
"""
from __future__ import annotations

import re

LABELS = ["knowledge_retrieval", "summarize", "small_talk"]

# 中文同义词归一（模型可能回中文标签，我们按语义等价接受，避免把格式问题算成错误）
SYNONYMS = {
    "knowledge_retrieval": ["knowledge_retrieval", "knowledge retrieval", "知识检索", "检索", "查询", "概念解释"],
    "summarize": ["summarize", "summary", "摘要", "总结", "概括", "提炼"],
    "small_talk": ["small_talk", "smalltalk", "small talk", "闲聊", "寒暄", "问候"],
}

SYSTEM_PROMPT = (
    "你是一个意图分类器。把用户输入归类为且仅归类为以下三类之一：\n"
    "knowledge_retrieval —— 知识检索、概念解释、文档查询\n"
    "summarize —— 摘要、总结、概括\n"
    "small_talk —— 问候、闲聊、情感表达\n"
    "只输出标签本身，不要标点，不要解释，不要换行。"
)

# few-shot 示例：三个模型完全相同，保证对比公平
FEW_SHOTS: list[tuple[str, str]] = [
    ("Redis和MySQL有什么区别？", "knowledge_retrieval"),
    ("帮我总结一下这段对话", "summarize"),
    ("你好呀，今天心情怎么样？", "small_talk"),
]


def build_messages(text: str, shots: int = 3) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in FEW_SHOTS[:max(0, shots)]:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": text})
    return messages


_CLEAN = re.compile(r"[\s`'\"“”‘’。，,、：:；;！!？?（）()\[\]{}<>《》#*\-]+")


def parse_label(raw: str) -> str | None:
    """把模型输出解析成规范标签；无法判定返回 None（记为非法输出）。"""
    if not raw:
        return None
    text = raw.strip().lower()

    # 1) 直接命中英文标签
    for label in LABELS:
        if text == label:
            return label

    # 2) 去掉噪声字符后命中
    cleaned = _CLEAN.sub("", text)
    for label in LABELS:
        if cleaned == label.replace("_", ""):
            return label

    # 3) 子串/同义词匹配（取最先出现且最长者）
    best: tuple[int, str] | None = None
    for label, words in SYNONYMS.items():
        for word in words:
            idx = cleaned.find(word.replace("_", "").replace(" ", ""))
            if idx >= 0:
                cand = (len(word), label)
                if best is None or cand > best:
                    best = cand
    return best[1] if best else None


def score(records: list[dict]) -> dict:
    """对一次模型运行计算指标（records 为 runner 输出的单条记录）。"""
    total = len(records)
    if total == 0:
        return {}
    correct = sum(1 for r in records if r.get("correct"))
    invalid = sum(1 for r in records if not r.get("pred"))
    hard = [r for r in records if r.get("hard")]
    easy = [r for r in records if not r.get("hard")]
    hard_correct = sum(1 for r in hard if r.get("correct"))
    easy_correct = sum(1 for r in easy if r.get("correct"))

    # 混淆矩阵（预测 None 归入 INVALID）
    confusion: dict[str, dict[str, int]] = {g: {} for g in LABELS}
    for r in records:
        gold = r.get("gold") or "?"
        pred = r.get("pred") or "INVALID"
        confusion.setdefault(gold, {})
        confusion[gold][pred] = confusion[gold].get(pred, 0) + 1

    return {
        "n": total,
        "accuracy": correct / total,
        "accuracy_hard": (hard_correct / len(hard)) if hard else 0.0,
        "accuracy_easy": (easy_correct / len(easy)) if easy else 0.0,
        "n_hard": len(hard),
        "invalid_rate": invalid / total,
        "confusion": confusion,
    }


def format_report(summary: dict) -> str:
    lines = [
        f"  样本数            : {summary['n']}（难例 {summary['n_hard']}）",
        f"  准确率            : {summary['accuracy'] * 100:.1f}%",
        f"  ├ 清晰样本        : {summary['accuracy_easy'] * 100:.1f}%",
        f"  └ 模糊难例        : {summary['accuracy_hard'] * 100:.1f}%",
        f"  非法输出率        : {summary['invalid_rate'] * 100:.1f}%",
    ]
    return "\n".join(lines)
