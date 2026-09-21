"""语料接入：把你自己的数据接进来之前，先体检。

这个工程的结论全部建立在一个客观评测集上，而它原本是**借来的**（Synapse 的 100 条人工标注）。
真把它当工具用时，第一个问题必然是：**我自己的数据怎么接？接了能测什么？**

本模块回答三个问题：
    1. **格式**对不对 —— jsonl / json / csv / txt 都能读，统一成 ``{"text","intent","hard"}``；
    2. **数据够不够测** —— 有标签才能测准确率；只有 prompt 就只能测 QPS 与延迟；
    3. **和当前任务定义对不对得上** —— 这是最大的坑：语料换了标签、``tasks.py`` 没换，
       准确率会**恒为 0**，看起来像"模型不行"，其实是**尺子错了**。

本模块只做体检与规范化，**不替你打标签**（标签是人的工作）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from inferbench import config
from inferbench import tasks
from inferbench.stats import write_json

# 常见列名别名：尽量让人不用改文件就能接进来
TEXT_KEYS = ("text", "query", "prompt", "question", "content", "输入", "文本", "问题", "内容")
INTENT_KEYS = ("intent", "label", "gold", "target", "category", "标签", "意图", "类别")
HARD_KEYS = ("hard", "is_hard", "difficult", "难例", "模糊")

# 超长判定用的粗估：中英混合按 0.6 字/token 保守估计。只用来"捞出可疑行"，不是精确分词。
CHAR_PER_TOKEN_GUESS = 0.6


def read_any(path: str | Path) -> tuple[list[dict], str]:
    """读入 jsonl / json / csv / tsv / txt，返回 (原始行, 格式名)。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到语料文件: {p}")
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        # utf-8-sig：Windows 上用 PowerShell / 记事本存出来的 jsonl 常带 BOM，
        # 用严格 utf-8 读会直接抛 "Unexpected UTF-8 BOM" —— 这是接真实语料时第一个会撞的墙
        for i, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{p} 第 {i} 行不是合法 JSON: {exc}") from exc
        return rows, "jsonl"

    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("rows") or list(data.values())
        return [r for r in data if isinstance(r, dict)], "json"

    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with p.open(encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f, delimiter=delim)], suffix.lstrip(".")

    # 其它一律当纯文本：一行一条 prompt（最常见的"我只有日志"形态）
    texts = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    return [{"text": t} for t in texts], "txt"


def _pick(row: dict, keys: tuple[str, ...], override: str | None) -> object:
    if override:
        return row.get(override)
    for k in keys:
        if k in row and row[k] not in ("", None):
            return row[k]
    return None


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "是", "真")


def normalize(rows: list[dict], *, text_key: str | None = None, intent_key: str | None = None,
              hard_key: str | None = None) -> list[dict]:
    """统一成 ``{"id","text","intent","hard","source"}``；保留原始行备查。"""
    out: list[dict] = []
    for i, row in enumerate(rows):
        text = _pick(row, TEXT_KEYS, text_key)
        intent = _pick(row, INTENT_KEYS, intent_key)
        hard = _pick(row, HARD_KEYS, hard_key)
        out.append({
            "id": i,
            "text": "" if text is None else str(text).strip(),
            "intent": "" if intent is None else str(intent).strip(),
            "hard": _to_bool(hard) if hard is not None else False,
            "source": "user",
        })
    return out


_WS = re.compile(r"\s+")


def validate(rows: list[dict], *, task_labels: list[str] | None = None,
             num_ctx: int | None = None) -> dict:
    """语料体检：把"能不能拿来测、测出来可不可信"讲清楚。"""
    task_labels = list(task_labels if task_labels is not None else tasks.LABELS)
    num_ctx = num_ctx or config.NUM_CTX
    total = len(rows)

    empty = [r["id"] for r in rows if not r["text"]]
    labeled = [r for r in rows if r["intent"]]
    unlabeled = [r["id"] for r in rows if not r["intent"]]

    by_intent: dict[str, int] = {}
    for r in labeled:
        by_intent[r["intent"]] = by_intent.get(r["intent"], 0) + 1
    unknown = {k: v for k, v in by_intent.items() if k not in task_labels}
    missing = [k for k in task_labels if k not in by_intent]

    seen: dict[str, int] = {}
    dups: list[dict] = []
    for r in rows:
        key = _WS.sub("", r["text"]).lower()
        if not key:
            continue
        if key in seen:
            dups.append({"id": r["id"], "dup_of": seen[key], "text": r["text"][:40]})
        else:
            seen[key] = r["id"]

    max_chars = max((len(r["text"]) for r in rows), default=0)
    limit = int(num_ctx * CHAR_PER_TOKEN_GUESS)
    over = [{"id": r["id"], "chars": len(r["text"])} for r in rows if len(r["text"]) > limit]

    hard_rows = [r for r in rows if r["hard"]]
    max_share = (max(by_intent.values()) / len(labeled)) if labeled else 0.0
    max_label = max(by_intent, key=by_intent.get) if by_intent else ""

    report = {
        "total": total,
        "labeled": len(labeled),
        "unlabeled": len(unlabeled),
        "unlabeled_ids": unlabeled[:10],
        "empty_text_ids": empty[:10],
        "by_intent": by_intent,
        "task_labels": task_labels,
        "unknown_labels": unknown,
        "missing_task_labels": missing,
        "hard_count": len(hard_rows),
        "hard_share": (len(hard_rows) / total) if total else 0.0,
        "duplicate_count": len(dups),
        "duplicate_examples": dups[:5],
        "max_chars": max_chars,
        "char_limit": limit,
        "over_length_count": len(over),
        "over_length_examples": over[:5],
        "balance": {"max_label": max_label, "max_share": max_share, "n_labels": len(by_intent)},
    }
    report["capabilities"] = _capabilities(report)
    report["problems"] = _problems(report)
    return report


def _capabilities(report: dict) -> list[dict]:
    """有这份数据，哪些实验能跑、哪些只能"退化着跑"、哪些跑不了。"""
    total = report["total"]
    labeled = report["labeled"]
    label_ok = labeled > 0 and not report["unknown_labels"]

    def item(name: str, status: str, why: str) -> dict:
        return {"experiment": name, "status": status, "why": why}

    caps = []
    caps.append(item(
        "实验五·并发压测", "ok" if total else "blocked",
        "只需要 prompt 构造请求，不需要标签" if total else "语料为空"))
    if labeled == 0:
        caps.append(item(
            "实验二/三·语义缓存", "degraded",
            "能跑，但「命中是否正确」无法客观判定（那要靠金标准标签）—— "
            "误命中率与 A/B 归因都会失效，只剩成本与延迟可看"))
        caps.append(item(
            "实验一·量化", "blocked",
            "准确率是量化对比的核心指标，没有标签就测不出准确率（只剩显存与速度）"))
    elif report["unknown_labels"]:
        caps.append(item(
            "实验二/三·语义缓存", "blocked",
            f"语料标签 {list(report['unknown_labels'])} 不在当前任务定义的标签集 "
            f"{report['task_labels']} 里 —— 打分会把它们全判错"))
        caps.append(item(
            "实验一·量化", "blocked",
            f"同上：标签与任务定义不一致，准确率会恒为 0。**这不是模型不行，是尺子错了** —— "
            f"请改 inferbench/tasks.py 的 LABELS / SYSTEM_PROMPT / FEW_SHOTS / SYNONYMS 四处"))
    else:
        caps.append(item(
            "实验二/三·语义缓存", "ok" if report["duplicate_count"] == 0 else "degraded",
            "标签齐备，可客观判定命中是否正确"
            + (f"；但有 {report['duplicate_count']} 条重复文本，会让命中率虚高（缓存的天然优势被放大）"
               if report["duplicate_count"] else "")))
        caps.append(item("实验一·量化", "ok", "标签齐备且与任务定义一致"))
    if total and report["hard_count"] == 0:
        caps.append(item(
            "「难例」相关结论", "degraded",
            "语料里一条 hard 都没标 —— 本工程的实测是「量化损伤优先暴露在难例上」，"
            "没有难例列，Q4 与 Q8 的差距会被整体准确率掩盖"))
    return caps


def _problems(report: dict) -> list[dict]:
    """按严重程度列出必须处理的问题（而不是让人自己在报告里找）。"""
    out = []
    if report["empty_text_ids"]:
        out.append({"level": "error", "msg": f"有 {len(report['empty_text_ids'])} 条文本为空"})
    if report["unlabeled"] == report["total"] and report["total"]:
        out.append({"level": "warn",
                    "msg": "一条标签都没有：只能测吞吐/延迟，测不了准确率。"
                           "标注是人的工作，本工具不代劳。"})
    if report["unknown_labels"]:
        out.append({"level": "error",
                    "msg": f"标签 {list(report['unknown_labels'])} 不在任务定义里 —— "
                           f"直接跑会得到恒为 0 的准确率。要么改语料标签，要么改 "
                           f"inferbench/tasks.py（LABELS / SYSTEM_PROMPT / FEW_SHOTS / SYNONYMS）"})
    if report["missing_task_labels"] and not report["unknown_labels"]:
        # unknown_labels 非空时这条只是同因后果（尺子错了），不再重复报一遍
        out.append({"level": "warn",
                    "msg": f"任务定义的标签 {report['missing_task_labels']} 在语料里一条都没有 —— "
                           f"混淆矩阵会出现整行空白，也没法判断模型是否「把这一类全答错」"})
    if report["over_length_count"]:
        out.append({"level": "warn",
                    "msg": f"有 {report['over_length_count']} 条文本超过约 {report['char_limit']} 字"
                           f"（按 {CHAR_PER_TOKEN_GUESS} 字/token 粗估 num_ctx={config.NUM_CTX}）—— "
                           f"超长会被静默截断，测出来的准确率不是这份数据的真实水平"})
    if report["duplicate_count"]:
        out.append({"level": "warn",
                    "msg": f"有 {report['duplicate_count']} 条重复文本 —— 对缓存实验会虚高命中率"})
    if report["labeled"] and report["balance"]["max_share"] > 0.7:
        out.append({"level": "warn",
                    "msg": f"类别不平衡：`{report['balance']['max_label']}` 占 "
                           f"{report['balance']['max_share'] * 100:.0f}%，"
                           f"整体准确率会被多数类主导（全猜它也能拿高分）"})
    return out


# ============================================================
# 报告与 CLI
# ============================================================
def render(report: dict, *, path: str, fmt: str) -> str:
    lines = ["=" * 78,
             f"语料体检  |  {path}  |  格式 {fmt}",
             "=" * 78,
             f"  样本数      : {report['total']}（有标签 {report['labeled']} · "
             f"无标签 {report['unlabeled']}）",
             f"  难例        : {report['hard_count']} 条（占 {report['hard_share'] * 100:.1f}%）",
             f"  重复文本    : {report['duplicate_count']} 条",
             f"  最长文本    : {report['max_chars']} 字"
             f"（超长阈值 {report['char_limit']} 字，疑似超长 {report['over_length_count']} 条）"]
    if report["by_intent"]:
        share = " · ".join(f"{k} {v}" for k, v in sorted(report["by_intent"].items(),
                                                        key=lambda kv: -kv[1]))
        lines.append(f"  标签分布    : {share}")
    lines.append(f"  任务定义标签: {report['task_labels']}")
    lines.append("")
    lines.append("  能测什么：")
    marks = {"ok": "✓", "degraded": "△", "blocked": "✗"}
    for cap in report["capabilities"]:
        lines.append(f"    {marks.get(cap['status'], '?')} {cap['experiment']}：{cap['why']}")
    if report["problems"]:
        lines.append("")
        lines.append("  必须处理的问题：")
        for prob in report["problems"]:
            tag = "✗" if prob["level"] == "error" else "⚠"
            lines.append(f"    {tag} {prob['msg']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m inferbench dataset",
        description="语料体检：读你自己的数据（jsonl/json/csv/txt），检查格式、可测性与标签对齐。",
        epilog="示例：\n"
               "  python -m inferbench dataset data/example_corpus.jsonl\n"
               "  python -m inferbench dataset raw_queries.txt --write data/my_corpus.jsonl\n"
               "  python -m inferbench dataset q.csv --text-col question --intent-col category\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="语料文件：.jsonl / .json / .csv / .tsv / .txt")
    p.add_argument("--text-col", default=None, help="文本列名（默认自动识别 text/query/prompt/…）")
    p.add_argument("--intent-col", default=None, help="标签列名（默认自动识别 intent/label/gold/…）")
    p.add_argument("--hard-col", default=None, help="难例列名（默认自动识别 hard/is_hard/…）")
    p.add_argument("--write", default=None, help="把规范化结果写成 jsonl，直接给 --eval-set 用")
    p.add_argument("--json", action="store_true", help="只输出 JSON 体检报告")
    p.add_argument("--report-tag", default="", help="体检报告的产物标签")
    return p.parse_args(argv)


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    raw, fmt = read_any(args.path)
    rows = normalize(raw, text_key=args.text_col, intent_key=args.intent_col,
                     hard_key=args.hard_col)
    report = validate(rows)
    report["source_path"] = str(args.path)
    report["source_format"] = fmt
    report["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["kind"] = "dataset"

    out_json = config.RESULTS_DIR / f"dataset_{args.report_tag or 'check'}.json"
    write_json(out_json, report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report, path=str(args.path), fmt=fmt))
        print(f"体检报告: {out_json}")

    if args.write:
        target = Path(args.write)
        if not target.is_absolute():
            target = config.ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已写出规范化语料: {target}")
        usable = [c["experiment"] for c in report["capabilities"] if c["status"] == "ok"]
        if usable:
            print(f"可跑的实验: {'、'.join(usable)}")
            print(f"例如: python -m inferbench load --eval-set {target} --levels 1,4 --requests 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
