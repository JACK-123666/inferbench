"""收敛环节（recommend）测试。

重点守三类**真实踩过的陷阱**：
    1. 文件名通配撞车：`cache_*.json` 会连 `cache_gate_*.json` 一起吃进来，
       而后者没有 `thresholds` —— 缓存决策会被误判成"无数据"；
    2. 把基线当候选：投机解码结果里的 `base2` 是**重复性对照**，凭 ±1% 抖动就能"赢"过 `base`，
       于是工具推荐"开 base2"；它真正的用处是给出**噪声下限**；
    3. 新跑的冒烟顶掉正式结果：同一个实验有多份结果时，必须按"信息量"挑，而不是按时间挑。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferbench import recommend as rec

FAKE_ENV = {
    "captured_at": "2026-09-21 00:00:00", "python": "3.13.5",
    "gpu": {"name": "NVIDIA GeForce RTX 4060 Laptop GPU", "memory_total_mb": 8188.0,
            "driver": "616.92", "cuda": "13.4", "compute_cap": "8.9"},
    "ollama": {"version": "0.32.15", "host": "http://127.0.0.1:11434", "num_parallel_env": 1},
    "code": {"inferbench": "0.3.0", "git": "test"},
}
FAKE_ENV["scope"] = "NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB） · Ollama 0.32.15"


def _quant(runs, records=900, env=None):
    def r(model, mem, acc, hard, ttft, tps):
        return {"model": model, "summary": {
            "model": model, "footprint_gb": mem, "accuracy": acc, "accuracy_hard": hard,
            "invalid_rate": 0.0, "ttft_ms_median": ttft, "decode_tps_median": tps,
            "wall_ms_p95": 150.0, "request_errors": 0}}
    return {"tag": "t", "created_at": "2026-09-18 10:00:00",
            "eval_set": {"total": 100, "hard": 23, "by_intent": {}},
            "protocol": {"num_ctx": 4096}, "runs": [r(*a) for a in runs],
            "records": [{"item_id": i} for i in range(records)], "env": env}


QUANT_RUNS = [("qwen3:1.7b-fp16", 3.69, 0.91, 0.783, 18.5, 85.3),
              ("qwen3:1.7b-q8_0", 2.24, 0.92, 0.783, 13.6, 134.2),
              ("qwen3:1.7b", 1.59, 0.87, 0.696, 10.7, 191.6)]


def _cache(rows=None):
    rows = rows if rows is not None else [
        {"threshold": 0.80, "hit_rate": 0.607, "hit_correct_rate": 0.84, "false_hit_rate": 0.159,
         "false_hit_a": 29, "false_hit_b": 0, "end_to_end_accuracy": 0.86,
         "nocache_accuracy": 0.86, "hit_ms_p95": 100.0, "miss_ms_p95": 150.0, "cost_saved_pct": 60.4},
        {"threshold": 0.92, "hit_rate": 0.423, "hit_correct_rate": 0.858, "false_hit_rate": 0.142,
         "false_hit_a": 18, "false_hit_b": 0, "end_to_end_accuracy": 0.86,
         "nocache_accuracy": 0.86, "hit_ms_p95": 89.0, "miss_ms_p95": 144.0, "cost_saved_pct": 42.4},
        {"threshold": 0.98, "hit_rate": 0.277, "hit_correct_rate": 0.819, "false_hit_rate": 0.181,
         "false_hit_a": 15, "false_hit_b": 0, "end_to_end_accuracy": 0.857,
         "nocache_accuracy": 0.86, "hit_ms_p95": 80.0, "miss_ms_p95": 144.0, "cost_saved_pct": 27.2},
    ]
    return {"tag": "full", "created_at": "2026-09-18 08:59:02",
            "meta": {"stream_size": 300, "unique_texts": 230, "embed_model": "e", "cache_model": "c"},
            "baseline": {"ms_p95": 154.2, "ms_median": 90.0}, "thresholds": rows, "engineering": {}}


def _gate():
    return {"tag": "full", "created_at": "2026-09-18 09:34:43",
            "meta": {"stream_size": 300, "unique_texts": 230, "n": 3, "temperature": 0.7,
                     "threshold": 0.92, "unanimous_count": 227, "unanimous_rate": 0.987,
                     "cache_model": "c", "embed_model": "e"},
            "off": {"gate": "off", "threshold": 0.92, "hit_rate": 0.423, "false_hit_count": 18,
                    "false_hit_rate": 0.142, "false_hit_a": 18, "false_hit_b": 0,
                    "cache_entries": 173, "write_skipped": 0, "end_to_end_accuracy": 0.863,
                    "nocache_accuracy": 0.863, "latency_ms_median": 89.0,
                    "latency_ms_p95": 127.0, "cost_saved_pct": 42.4},
            "on": {"gate": "on", "threshold": 0.92, "hit_rate": 0.417, "false_hit_count": 16,
                   "false_hit_rate": 0.128, "false_hit_a": 16, "false_hit_b": 0,
                   "cache_entries": 172, "write_skipped": 1, "end_to_end_accuracy": 0.867,
                   "nocache_accuracy": 0.863, "latency_ms_median": 165.0,
                   "latency_ms_p95": 230.0, "cost_saved_pct": -74.7}}


def _spec():
    def run(name, a_tps, b_tps, a_acc=100.0, b_acc=94.0):
        return {"config": name, "workloads": {
            "A": {"decode_tps_median": a_tps, "accept_rate_median": a_acc, "ttft_ms_median": 20.0},
            "B": {"decode_tps_median": b_tps, "accept_rate_median": b_acc, "ttft_ms_median": 20.0}}}
    return {"tag": "full", "created_at": "2026-09-18 12:00:01",
            "meta": {"target": "qwen3:1.7b", "target_gb": 1.27, "draft": "qwen3:0.6b",
                     "draft_gb": 0.49, "n_ctx": 4096, "np": 1},
            "runs": [run("base", 154.2, 152.5), run("base2", 155.3, 152.2),
                     run("draft-k3", 107.4, 100.2, 69.9, 65.3),
                     run("draft-k8", 109.2, 94.8, 56.6, 45.7),
                     run("draft-k16", 76.5, 57.3, 30.8, 22.0),
                     run("ngram", 282.0, 181.2, 100.0, 94.2)]}


def _load(num_parallel, footprint, ttfts):
    """ttfts: {并发: TTFT P95}"""
    return {"tag": f"p{num_parallel}", "created_at": "2026-09-18 12:25:13",
            "meta": {"model": "qwen3:1.7b", "ollama_num_parallel": num_parallel,
                     "footprint_gb": footprint, "host": "http://127.0.0.1:11434",
                     "num_ctx": 4096, "levels": sorted(ttfts)},
            "levels": [{"concurrency": c, "requests": 24, "error_rate": 0.0, "qps": 2.4 * c ** 0.6,
                        "tokens_per_s": 150.0, "ttft_p50": p * 0.9, "ttft_p95": p,
                        "lat_p95": 500.0, "lat_p99": 520.0} for c, p in sorted(ttfts.items())],
            "saturation": {}}


# ============================================================
# 量化选型
# ============================================================
def test_quant_without_floor_picks_highest_accuracy():
    """没给质量下限时，不该主动用精度换显存 —— 应取准确率最高的一档。"""
    d = rec.decide_quant(_quant(QUANT_RUNS), vram_gb=8.0, quality_floor=None)
    assert d["status"] == "ok"
    assert d["recommended"]["model"] == "qwen3:1.7b-q8_0"          # 92.0%，不是最省的 1.59GB
    assert d["cheaper_alternative"]["model"] == "qwen3:1.7b"
    assert d["cheaper_alternative"]["accuracy_cost_pp"] == pytest.approx(5.0)
    assert d["cheaper_alternative"]["hard_cost_pp"] > 8.0


def test_quant_with_floor_picks_cheapest_that_meets_it():
    d = rec.decide_quant(_quant(QUANT_RUNS), vram_gb=8.0, quality_floor=0.85)
    assert d["recommended"]["model"] == "qwen3:1.7b"
    assert "质量下限" in d["basis"] or "下限" in d["basis"]
    # 推荐档就是难例最优时，不该输出"低 0.0pp"这种空警告
    assert "hard_case_cost" not in d or d["hard_case_cost"]["hard_drop_pp"] > 0.05


def test_quant_infeasible_reports_why():
    d = rec.decide_quant(_quant(QUANT_RUNS), vram_gb=1.0, quality_floor=None)
    assert d["status"] == "infeasible"
    assert any("显存不够" in r for r in d["reasons"])
    d2 = rec.decide_quant(_quant(QUANT_RUNS), vram_gb=8.0, quality_floor=0.99)
    assert d2["status"] == "infeasible"
    assert any("准确率下限" in r for r in d2["reasons"])


def test_quant_flags_hard_case_amplification():
    """推荐档在难例上更差时必须给出放大倍数（Q4 难例掉幅是整体的近 2 倍）。"""
    d = rec.decide_quant(_quant(QUANT_RUNS), vram_gb=8.0, quality_floor=0.85)
    hc = d.get("hard_case_cost")
    assert hc and hc["hard_drop_pp"] > hc["overall_drop_pp"]
    assert hc["amplification"] and hc["amplification"] > 1.5


# ============================================================
# 投机解码：不能把基线当候选
# ============================================================
def test_spec_never_recommends_baseline_or_control():
    d = rec.decide_spec(_spec())
    bests = {w: v["best"]["config"] for w, v in d["per_workload"].items()}
    assert set(bests.values()) == {"ngram"}, f"推荐了非投机配置: {bests}"
    assert "base2" not in {c["config"] for v in d["per_workload"].values() for c in v["candidates"]}


def test_spec_uses_control_as_noise_floor():
    """base2 与 base 的差异 = 测量噪声下限；低于它的加速不足采信。"""
    d = rec.decide_spec(_spec())
    assert d["controls"] == ["base2"]
    assert 0.0 < d["noise_floor_pct"] < 2.0


def test_spec_marks_all_draft_configs_slower():
    d = rec.decide_spec(_spec())
    assert d["all_draft_configs_slower"] is True
    assert d["draft_speedups"]["draft-k16"] == [0.496, 0.376]


def test_spec_absent_gives_no_conclusion():
    assert rec.decide_spec(None)["status"] == "no_data"


# ============================================================
# 缓存与门槛
# ============================================================
def test_cache_picks_max_saving_within_accuracy_and_no_cross_intent():
    d = rec.decide_cache(_cache(), _gate())
    assert d["cache"]["threshold"] == 0.80        # 0.80 省 60.4% 最多，且准确率不低于基线、B类=0
    assert d["cache"]["selection_met"] is True


def test_cache_rejects_threshold_that_costs_accuracy():
    """如果某档端到端准确率低于无缓存基线，就不能选它 —— 哪怕它省得最多。"""
    rows = [dict(r) for r in _cache()["thresholds"]]
    rows[0]["end_to_end_accuracy"] = 0.70         # 0.80 档掉准确率
    d = rec.decide_cache(_cache(rows), None)
    assert d["cache"]["threshold"] != 0.80
    assert d["cache"]["selection_met"] is False or d["cache"]["threshold"] in (0.92, 0.98)


def test_cache_detects_threshold_is_not_a_lever():
    d = rec.decide_cache(_cache(), None)
    assert "不随阈值单调下降" in d["cache"]["threshold_is_lever"]["note"]


def test_gate_negative_result_marked_not_worth_it():
    d = rec.decide_cache(_cache(), _gate())
    assert d["gate"]["worth_it"] is False
    assert d["gate"]["cost_saved_pct_on"] < d["gate"]["cost_saved_pct_off"]


# ============================================================
# 并发与容量
# ============================================================
def test_load_capacity_respects_slo():
    loads = [(Path("load_full.json"), _load(1, 1.586, {1: 40.7, 2: 448.0, 4: 1298.3,
                                                        8: 2813.9, 16: 6112.1})),
             (Path("load_parallel4.json"), _load(4, 2.949, {1: 36.0, 2: 34.9, 4: 51.9,
                                                            8: 676.6, 16: 1641.9}))]
    d = rec.decide_load(loads, slo_ttft=200.0, want_concurrency=4)
    assert d["recommended"]["num_parallel"] == 4          # 只有 P=4 能在这个 SLO 下扛住
    assert d["recommended"]["capacity_at_slo"]["concurrency"] == 4
    assert d["meets_target"]["ok"] is True


def test_load_does_not_extrapolate_beyond_measured():
    """要 32 路但没测过 → 必须说"给不出结论"，不能外推。"""
    loads = [(Path("load_full.json"), _load(1, 1.586, {1: 40.7, 2: 448.0}))]
    d = rec.decide_load(loads, slo_ttft=200.0, want_concurrency=32)
    assert d["meets_target"]["ok"] is False
    assert "不外推" in d["meets_target"]["reason"]


def test_load_reports_parallel_upgrade_vram_cost():
    loads = [(Path("load_full.json"), _load(1, 1.586, {1: 40.7, 2: 448.0})),
             (Path("load_parallel4.json"), _load(4, 2.949, {2: 34.9, 4: 51.9}))]
    d = rec.decide_load(loads, slo_ttft=200.0, want_concurrency=None)
    pu = d["parallel_upgrade"]
    assert pu["capacity_from"] == 1 and pu["capacity_to"] == 4
    assert 80 < pu["vram_cost_pct"] < 90                    # 实测 +86%


def test_load_absent_gives_no_conclusion():
    assert rec.decide_load([], slo_ttft=200.0, want_concurrency=None)["status"] == "no_data"


def test_load_without_slo_does_not_claim_capacity():
    """不设延迟目标就没有"承载能力"可言：P=1 也能收下 16 路，只是首 Token 要等几秒。"""
    loads = [(Path("load_full.json"), _load(1, 1.586, {1: 40.7, 16: 6112.1}))]
    d = rec.decide_load(loads, slo_ttft=None, want_concurrency=16)
    assert d["recommended"] is None
    assert "给不出承载能力结论" in d["capacity_note"]
    assert d["meets_target"]["ok"] is False and "没有意义" in d["meets_target"]["reason"]


def test_corpus_check_warns_when_results_use_different_corpora():
    a = {"eval_set": {"source": "内置", "total": 100, "sha1_8": "aaaaaaaa"}}
    b = {"eval_set": {"source": "我的语料.jsonl", "total": 15, "sha1_8": "bbbbbbbb"}}
    chk = rec.check_corpus([("实验一·量化", a), ("实验二·缓存", b)])
    assert len(chk["corpora"]) == 2
    assert any("不是同一份语料" in w for w in chk["warnings"])
    # 同一份语料则不告警
    assert rec.check_corpus([("实验一·量化", a), ("实验二·缓存", dict(a))])["warnings"] == []
    # 没有语料信息的老结果进 unknown，不误报"混用"
    chk2 = rec.check_corpus([("实验一·量化", a), ("实验二·缓存", {"tag": "old"})])
    assert chk2["warnings"] == [] and chk2["unknown"] == ["实验二·缓存"]


# ============================================================
# 数据源挑选与适用范围
# ============================================================
def test_candidates_exclude_gate_files(tmp_path: Path):
    """回归：cache_*.json 不能把 cache_gate_*.json 吃进来（后者没有 thresholds）。"""
    (tmp_path / "cache_full.json").write_text(json.dumps(_cache()), encoding="utf-8")
    (tmp_path / "cache_gate_full.json").write_text(json.dumps(_gate()), encoding="utf-8")
    picked = rec._candidates(tmp_path, "cache_*.json", exclude=("cache_gate_",))
    assert [p.name for p, _ in picked] == ["cache_full.json"]
    gate = rec._candidates(tmp_path, "cache_gate_*.json")
    assert [p.name for p, _ in gate] == ["cache_gate_full.json"]


def test_pick_prefers_more_records_over_newer(tmp_path: Path):
    """回归：新跑的一次冒烟不能顶掉正式的完整结果。"""
    full = tmp_path / "quant_full.json"
    smoke = tmp_path / "quant_smoke.json"
    full.write_text(json.dumps(_quant(QUANT_RUNS, records=900)), encoding="utf-8")
    d = _quant(QUANT_RUNS, records=24)
    d["created_at"] = "2026-09-21 23:59:59"          # 冒烟更新
    smoke.write_text(json.dumps(d), encoding="utf-8")
    picked = rec._pick("quant", rec._candidates(tmp_path, "quant_*.json"))
    assert picked is not None and picked[0].name == "quant_full.json"


def test_scope_check_warns_on_gpu_mismatch():
    other = {"gpu": {"name": "NVIDIA A100", "memory_total_mb": 40960.0},
             "ollama": {"version": "0.30.0", "num_parallel_env": 4}}
    chk = rec.check_scope([("实验五·并发", {"env": other})], FAKE_ENV)
    text = "\n".join(chk["warnings"])
    assert "A100" in text and "可能不成立" in text
    assert "Ollama 0.30.0" in text


def test_scope_check_lists_results_without_fingerprint():
    chk = rec.check_scope([("实验二·缓存", {"tag": "old"}),
                           ("实验五·并发（load_x.json）", {"tag": "old"})], FAKE_ENV)
    assert chk["unknown_scope"] == ["实验二·缓存", "实验五·并发（load_x.json）"]
    assert chk["warnings"] == []


# ============================================================
# 端到端：build() 全流程（用假结果目录，不碰 GPU）
# ============================================================
def _write_results_dir(tmp_path: Path) -> Path:
    (tmp_path / "quant_full.json").write_text(json.dumps(_quant(QUANT_RUNS)), encoding="utf-8")
    (tmp_path / "quant_smoke.json").write_text(
        json.dumps(_quant(QUANT_RUNS, records=12)), encoding="utf-8")
    (tmp_path / "cache_full.json").write_text(json.dumps(_cache()), encoding="utf-8")
    (tmp_path / "cache_gate_full.json").write_text(json.dumps(_gate()), encoding="utf-8")
    (tmp_path / "spec_full.json").write_text(json.dumps(_spec()), encoding="utf-8")
    (tmp_path / "load_full.json").write_text(
        json.dumps(_load(1, 1.586, {1: 40.7, 2: 448.0})), encoding="utf-8")
    (tmp_path / "load_parallel4.json").write_text(
        json.dumps(_load(4, 2.949, {2: 34.9, 4: 51.9})), encoding="utf-8")
    return tmp_path


def test_build_end_to_end_uses_all_five_experiments(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rec.fp, "fingerprint", lambda host=None: dict(FAKE_ENV))
    monkeypatch.setattr(rec.config, "RESULTS_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    rdir = _write_results_dir(tmp_path)

    payload, paths = rec.build(["--results", str(rdir), "--vram", "8",
                               "--slo-ttft", "200", "--concurrency", "4", "--tag", "unit"])

    assert payload["quant"]["recommended"]["model"] == "qwen3:1.7b-q8_0"
    assert payload["cache_decision"]["cache"]["threshold"] == 0.80
    assert payload["cache_decision"]["gate"]["worth_it"] is False
    assert payload["spec_decision"]["per_workload"]["A"]["best"]["config"] == "ngram"
    assert payload["load_decision"]["recommended"]["num_parallel"] == 4
    # 缓存文件必须挑对：挑到 gate 文件的话这一项会变成 no_data
    assert payload["sources"]["cache"]["file"] == "cache_full.json"
    assert payload["sources"]["quant"]["file"] == "quant_full.json"
    assert "quant_smoke.json" in payload["unused_candidates"]["quant"]
    assert len(paths) == 2 and paths[1].exists()
    md = paths[1].read_text(encoding="utf-8")
    assert "| 量化档位 |" in md and "## 7. 这份建议的边界" in md


def test_build_degrades_when_experiments_missing(tmp_path: Path, monkeypatch):
    """缺实验五时，其余决策照常给出，并发一项明确"给不出结论"。"""
    monkeypatch.setattr(rec.fp, "fingerprint", lambda host=None: dict(FAKE_ENV))
    monkeypatch.setattr(rec.config, "RESULTS_DIR", tmp_path / "out2")
    (tmp_path / "out2").mkdir()
    rdir = tmp_path / "only_quant"
    rdir.mkdir()
    (rdir / "quant_full.json").write_text(json.dumps(_quant(QUANT_RUNS)), encoding="utf-8")

    payload, paths = rec.build(["--results", str(rdir), "--vram", "8", "--tag", "unit2"])

    assert payload["quant"]["status"] == "ok"
    assert payload["load_decision"]["status"] == "no_data"
    md = paths[1].read_text(encoding="utf-8")
    assert "实验五·并发 | ⚠ 缺" in md
    assert "缺失的实验" in md
