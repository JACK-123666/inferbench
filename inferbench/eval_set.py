"""评测集加载 —— 直接复用 Synapse 开源工程的 100 条标注集。

数据来源（三选一，按优先级）：
1. ``data/eval_set.jsonl``（本地缓存，首次运行自动从 Synapse 导出）
2. ``D:\\PYTHON\\Synapse\\tools\\intent_testset.py`` 里的 ``TEST_SET``
3. ``--eval-set`` 指定的任意 jsonl

样本结构::

    {"text": "什么是向量数据库？", "intent": "knowledge_retrieval", "hard": false}
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from inferbench import config


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
    # utf-8-sig：`--eval-set` 指向的文件常是用户在 Windows 上手写/导出的，
    # 带 BOM 时严格 utf-8 会读不出（首个字符变成 \ufeff，json.loads 直接失败）
    with Path(path).open("r", encoding="utf-8-sig") as f:
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


def normalize_rows(rows: list[dict]) -> list[dict]:
    """统一字段：``id`` / ``text`` / ``intent`` / ``hard`` / ``source``。

    为什么必须有这一步（实测踩到）：测量执行器要求每条样本带 ``id``，而从 Synapse 导出的
    缓存会自动补 id、**手写的 jsonl 不会** —— 于是"支持任意 jsonl"这句话以前是假的：
    随便写一份语料喂给 ``--eval-set`` 就会 KeyError: 'id'。缺 id 时按行号补，
    ``intent`` 允许为空（表示未标注），是否可用留给各实验自己判断。
    """
    out: list[dict] = []
    for i, row in enumerate(rows):
        out.append({
            "id": row.get("id", i),
            "text": str(row.get("text", "")),
            "intent": str(row.get("intent") or ""),
            "hard": bool(row.get("hard", False)),
            "source": row.get("source", "user"),
        })
    return out


def labeling(rows: list[dict]) -> dict:
    """标注情况：有多少条有标签、标签是什么、是否落在当前任务定义里。"""
    from inferbench import tasks
    labels = sorted({r["intent"] for r in rows if r.get("intent")})
    labeled = sum(1 for r in rows if r.get("intent"))
    return {
        "total": len(rows), "labeled": labeled, "unlabeled": len(rows) - labeled,
        "labels": labels,
        "unknown_labels": [x for x in labels if x not in tasks.LABELS],
        "task_labels": list(tasks.LABELS),
    }


def require_labels(rows: list[dict], command: str) -> str | None:
    """准确率类实验的前置检查：没标签/标签不匹配就**不让跑**。

    宁可拒绝，也不产出一个恒为 0 的准确率 —— 那种结果会被误读成"模型不行"，
    而真实原因是尺子不对（语料标签与 ``tasks.py`` 的任务定义不一致）。
    返回 None 表示可以跑；否则返回给用户看的说明。
    """
    if not rows:
        return "语料为空：请用 --eval-set 指定，或先跑一次 `python -m inferbench dataset <你的文件>`。"
    info = labeling(rows)
    if info["labeled"] == 0:
        return ("这份语料一条标签都没有 —— 准确率测不出来（只能测吞吐/延迟）。\n"
                "  先补标签（jsonl 里加 \"intent\" 字段），或用 "
                "`python -m inferbench load --eval-set <文件>` 只做并发压测。\n"
                "  体检：python -m inferbench dataset <文件>")
    if info["unknown_labels"]:
        return (f"语料标签 {info['unknown_labels']} 不在当前任务定义的标签集 "
                f"{info['task_labels']} 里 —— 直接跑会得到恒为 0 的准确率。\n"
                f"  **这不是模型不行，是尺子错了**：要么改语料标签，要么改 "
                f"inferbench/tasks.py 的 LABELS / SYSTEM_PROMPT / FEW_SHOTS / SYNONYMS 四处。\n"
                f"  体检：python -m inferbench dataset <文件>")
    return None


def load(eval_set: str | Path | None = None, *, refresh: bool = False) -> list[dict]:
    """返回评测集（list[dict]），并按需把 Synapse 版本缓存到 data/。"""
    if eval_set:
        return normalize_rows(load_jsonl(Path(eval_set)))

    cache = config.DATA_DIR / "eval_set.jsonl"
    if cache.exists() and not refresh:
        rows = normalize_rows(load_jsonl(cache))
        if rows:
            return rows

    rows = normalize_rows(_load_from_synapse())
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


def identify(path: str | Path | None, rows: list[dict]) -> dict:
    """语料身份：来自哪里、多少条、**内容指纹**。

    为什么必须记指纹：``--eval-set`` 允许接任意语料，而**同一个文件名随时可以被改内容**。
    只写路径的话，"这份结果到底是用哪份数据测的"永远说不清 —— 换一版语料重跑，
    两份结果就再也对不上了。指纹只存 sha1 前 8 位，不存数据本身。
    """
    digest = hashlib.sha1(
        "\n".join(str(r.get("text", "")) for r in rows).encode("utf-8")).hexdigest()[:8]
    info = summarize([dict(r, intent=r.get("intent") or "?") for r in rows])
    info["source"] = (str(path) if path
                      else "内置：Synapse 100 条人工标注（缓存于 data/eval_set.jsonl）")
    info["sha1_8"] = digest
    return info


def render_lines(payload: dict) -> list[str]:
    """报告里那行「语料」说明。老结果没有指纹就明说。"""
    info = payload.get("eval_set")
    if not info:
        return ["- 语料：⚠ 本次运行早于语料指纹功能，无法确认用的是哪份数据"]
    src = info.get("source", "未记录")
    sha = info.get("sha1_8") or "未记录"
    return [f"- 语料：{src} · {info.get('total')} 条（难例 {info.get('hard')}）· "
            f"指纹 `{sha}`"]
