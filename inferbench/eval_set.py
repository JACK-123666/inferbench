"""评测集加载 —— 直接复用 Synapse 开源工程的 100 条标注集。

数据来源（三选一，按优先级）：
1. ``data/eval_set.jsonl``（本地缓存，首次运行自动从 Synapse 导出）
2. ``D:\\PYTHON\\Synapse\\tools\\intent_testset.py`` 里的 ``TEST_SET``
3. ``--eval-set`` 指定的任意 jsonl

样本结构::

    {"text": "什么是向量数据库？", "intent": "knowledge_retrieval", "hard": false}
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import config


def _load_from_synapse() -> list[dict]:
    path = config.SYNAPSE_TESTSET
    if not path.exists():
        return []
    spec = importlib.util.spec_from_file_location("_synapse_intent_testset", path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "TEST_SET", [])
    return [
        {
            "id": i,
            "text": str(item["text"]),
            "intent": str(item["intent"]),
            "hard": bool(item.get("hard", False)),
            "source": "synapse",
        }
        for i, item in enumerate(raw)
    ]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load(eval_set: str | Path | None = None, *, refresh: bool = False) -> list[dict]:
    """返回评测集（list[dict]），并按需把 Synapse 版本缓存到 data/。"""
    if eval_set:
        return load_jsonl(Path(eval_set))

    cache = config.DATA_DIR / "eval_set.jsonl"
    if cache.exists() and not refresh:
        rows = load_jsonl(cache)
        if rows:
            return rows

    rows = _load_from_synapse()
    if rows:
        save_jsonl(cache, rows)
    return rows


def summarize(eval_set: list[dict]) -> dict:
    counts: dict[str, int] = {}
    hard = 0
    for item in eval_set:
        counts[item["intent"]] = counts.get(item["intent"], 0) + 1
        hard += 1 if item.get("hard") else 0
    return {"total": len(eval_set), "hard": hard, "by_intent": counts}
