"""语料接入（dataset）测试。

守的是"接真实语料时最先撞的几堵墙"：
    1. **标签与任务定义不一致** → 准确率恒为 0，看起来像模型不行，其实是尺子错了（必须在体检里报出来）；
    2. **Windows 上导出/手写的 jsonl 带 BOM** → 严格 utf-8 直接读不出（dataset 与 --eval-set 两条路都要能读）；
    3. **只有 prompt、没有标签** → 只能测吞吐/延迟，不能让用户以为能测准确率；
    4. **没有 hard 标注** → 量化损伤优先暴露在难例上，缺了这条列，Q4 与 Q8 的差距会被掩盖。
"""
from __future__ import annotations

import json
from pathlib import Path

from inferbench import dataset as ds
from inferbench import eval_set as ev
from inferbench import tasks


def _rows(*specs) -> list[dict]:
    return [{"id": i, "text": t, "intent": intent, "hard": hard, "source": "user"}
            for i, (t, intent, hard) in enumerate(specs)]


# ---------- 读入：格式与 BOM ----------

def test_reads_jsonl_with_bom(tmp_path: Path):
    """回归：Windows 上 Out-File / 记事本存出的 jsonl 带 BOM，必须能读。"""
    p = tmp_path / "bom.jsonl"
    p.write_text('{"text": "你好", "intent": "small_talk", "hard": false}\n', encoding="utf-8-sig")
    rows, fmt = ds.read_any(p)
    assert fmt == "jsonl" and rows[0]["text"] == "你好"
    # --eval-set 走的是另一条读取路径，也必须能读
    assert ev.load_jsonl(p)[0]["intent"] == "small_talk"


def test_reads_txt_as_unlabeled_prompts(tmp_path: Path):
    p = tmp_path / "logs.txt"
    p.write_text("第一条\n\n第二条\n", encoding="utf-8")
    rows, fmt = ds.read_any(p)
    assert fmt == "txt" and [r["text"] for r in rows] == ["第一条", "第二条"]
    assert all("intent" not in r for r in rows)


def test_reads_csv_with_column_aliases(tmp_path: Path):
    p = tmp_path / "q.csv"
    p.write_text("question,category\n怎么查批次记录,知识检索\n", encoding="utf-8-sig")
    rows, fmt = ds.read_any(p)
    assert fmt == "csv"
    norm = ds.normalize(rows)
    assert norm[0]["text"] == "怎么查批次记录" and norm[0]["intent"] == "知识检索"


def test_unknown_extension_treated_as_text(tmp_path: Path):
    p = tmp_path / "queries.log"
    p.write_text("a\nb\n", encoding="utf-8")
    rows, fmt = ds.read_any(p)
    assert fmt == "txt" and len(rows) == 2


# ---------- 体检 ----------

def test_label_mismatch_blocks_accuracy_experiments():
    """最大的坑：语料标签换了、tasks.py 没换 → 准确率恒为 0。必须报 error 并指出改哪四处。"""
    rows = _rows(("这批货延误了几天", "logistics_anomaly", False),
                 ("成本超支多少", "cost_query", False))
    rep = ds.validate(rows)
    assert rep["unknown_labels"] == {"logistics_anomaly": 1, "cost_query": 1}
    blocked = {c["experiment"] for c in rep["capabilities"] if c["status"] == "blocked"}
    assert "实验一·量化" in blocked and "实验二/三·语义缓存" in blocked
    errs = [p["msg"] for p in rep["problems"] if p["level"] == "error"]
    assert any("tasks.py" in m and "LABELS" in m for m in errs)
    # 尺子错了这件事必须说破，而不是让用户以为模型不行
    why = next(c["why"] for c in rep["capabilities"] if c["experiment"] == "实验一·量化")
    assert "尺子错了" in why


def test_unlabeled_corpus_only_supports_throughput():
    rep = ds.validate(_rows(("上个月运费涨了多少", "", False)))
    caps = {c["experiment"]: c["status"] for c in rep["capabilities"]}
    assert caps["实验五·并发压测"] == "ok"
    assert caps["实验一·量化"] == "blocked"
    assert caps["实验二/三·语义缓存"] == "degraded"
    assert any("不代劳" in p["msg"] for p in rep["problems"])


def test_matching_labels_are_ok():
    rep = ds.validate(_rows(("你好", "small_talk", False), ("总结一下", "summarize", True)))
    caps = {c["experiment"]: c["status"] for c in rep["capabilities"]}
    assert caps["实验一·量化"] == "ok"
    assert caps["实验二/三·语义缓存"] == "ok"
    assert rep["hard_count"] == 1 and rep["hard_share"] == 0.5
    assert not [p for p in rep["problems"] if p["level"] == "error"]


def test_missing_hard_flag_is_reported():
    """没有 hard 标注 → 「难例」相关结论降级（量化损伤优先暴露在难例上，缺这列会掩盖差距）。"""
    rep = ds.validate(_rows(("你好", "small_talk", False), ("谢谢", "small_talk", False)))
    degraded = [c for c in rep["capabilities"] if c["status"] == "degraded"]
    assert any("难例" in c["experiment"] and "掩盖" in c["why"] for c in degraded)


def test_duplicates_flagged_for_cache_experiment():
    rep = ds.validate(_rows(("你好", "small_talk", False), ("你好", "small_talk", False)))
    assert rep["duplicate_count"] == 1
    cap = next(c for c in rep["capabilities"] if "缓存" in c["experiment"])
    assert cap["status"] == "degraded" and "虚高" in cap["why"]


def test_empty_text_and_over_length_flagged():
    long_text = "字" * 5000                       # 默认阈值 = num_ctx(4096) × 0.6 ≈ 2457
    rep = ds.validate(_rows(("", "small_talk", False), (long_text, "summarize", False)))
    assert rep["empty_text_ids"] == [0]
    assert rep["over_length_count"] == 1
    msgs = " ".join(p["msg"] for p in rep["problems"])
    assert "为空" in msgs and "截断" in msgs


def test_class_imbalance_flagged():
    rows = _rows(*[("问" + str(i), "knowledge_retrieval", False) for i in range(9)])
    rows += _rows(("你好", "small_talk", False))
    rows = [dict(r, id=i) for i, r in enumerate(rows)]
    rep = ds.validate(rows)
    assert rep["balance"]["max_share"] == 0.9
    assert any("不平衡" in p["msg"] for p in rep["problems"])


# ---------- CLI ----------

def test_cli_writes_normalized_corpus(tmp_path: Path, monkeypatch):
    src = tmp_path / "logs.txt"
    src.write_text("第一条 query\n第二条 query\n", encoding="utf-8")
    out = tmp_path / "normalized.jsonl"
    monkeypatch.setattr(ds.config, "RESULTS_DIR", tmp_path)     # 别把体检报告写进仓库 results/
    code = ds.run([str(src), "--write", str(out), "--json"])
    assert code == 0
    written = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    assert set(written[0]) == {"id", "text", "intent", "hard", "source"}
    # 写出的文件必须能被 --eval-set 那条路读回来
    assert ev.load_jsonl(out)[0]["text"] == "第一条 query"


def test_example_corpus_in_repo_is_valid_and_runnable():
    """仓库里那份样例语料必须自洽：标签与任务定义一致、无重复、无空文本。"""
    path = Path(__file__).resolve().parents[1] / "data" / "example_corpus.jsonl"
    rows, fmt = ds.read_any(path)
    rep = ds.validate(ds.normalize(rows))
    assert fmt == "jsonl"
    assert rep["total"] == 15
    assert rep["unknown_labels"] == {} and rep["empty_text_ids"] == []
    assert rep["duplicate_count"] == 0 and rep["hard_count"] == 3
    assert not [p for p in rep["problems"] if p["level"] == "error"]
    assert {c["status"] for c in rep["capabilities"] if "量化" in c["experiment"]} == {"ok"}


def test_task_labels_are_read_from_tasks_module():
    """体检的"尺子"必须取自 tasks.py，不能在 dataset.py 里另写一份。"""
    rep = ds.validate(_rows(("你好", tasks.LABELS[0], False)))
    assert rep["task_labels"] == list(tasks.LABELS)


# ---------- 载入边界：手写 jsonl 必须能直接喂给实验 ----------

def test_normalize_rows_backfills_missing_id():
    """回归：手写的 jsonl 没有 id，而测量执行器要求每条带 id —— 以前会 KeyError: 'id'。"""
    rows = ev.normalize_rows([{"text": "你好", "intent": "small_talk"},
                              {"text": "总结一下", "intent": "summarize"}])
    assert [r["id"] for r in rows] == [0, 1]
    assert rows[0]["hard"] is False and rows[0]["source"] == "user"
    # 原有 id 必须保留（内置评测集的记录要稳定）
    assert ev.normalize_rows([{"id": 42, "text": "x"}])[0]["id"] == 42


def test_require_labels_blocks_unlabeled_and_mismatched():
    unlabeled = ev.normalize_rows([{"text": "你好"}])
    msg = ev.require_labels(unlabeled, "quant")
    assert msg and "一条标签都没有" in msg
    mismatched = ev.normalize_rows([{"text": "你好", "intent": "logistics_anomaly"}])
    msg2 = ev.require_labels(mismatched, "quant")
    assert msg2 and "尺子错了" in msg2 and "tasks.py" in msg2
    ok = ev.normalize_rows([{"text": "你好", "intent": "small_talk"}])
    assert ev.require_labels(ok, "quant") is None
    assert "语料为空" in ev.require_labels([], "quant")


def test_identify_fingerprint_is_stable_and_content_sensitive():
    rows = ev.normalize_rows([{"text": "a", "intent": "small_talk", "hard": True},
                              {"text": "b", "intent": "summarize"}])
    info = ev.identify("my.jsonl", rows)
    assert info["source"] == "my.jsonl" and info["total"] == 2 and info["hard"] == 1
    assert info["sha1_8"] == ev.identify("other-name.jsonl", rows)["sha1_8"]   # 只看内容
    rows2 = ev.normalize_rows([{"text": "a", "intent": "small_talk", "hard": True},
                               {"text": "b 改了一点", "intent": "summarize"}])
    assert ev.identify("my.jsonl", rows2)["sha1_8"] != info["sha1_8"]          # 内容变了就变
    assert len(info["sha1_8"]) == 8


def test_render_lines_handles_old_results():
    assert "无法确认" in ev.render_lines({"tag": "old"})[0]
    line = ev.render_lines({"eval_set": {"source": "内置", "total": 100, "hard": 23,
                                        "sha1_8": "abc12345"}})[0]
    assert "内置" in line and "100 条" in line and "abc12345" in line
