"""收敛环节：把 5 个实验的数字收敛成「你这台机器上该怎么配」。

为什么需要它（由这个工程的定位决定）：
    前面 5 个实验产出的是**数据**，不是**决策**。报告里全是表格，人要自己把 5 份表读成一句话 ——
    「我该用哪个量化档、缓存阈值定多少、并行度设几、每一项的代价是多少」。这一步就是那句话。

三条硬规矩（与工程其它部分一致）：
    1. **所有数字只能来自 ``results/*.json``**，不手抄、不在代码里内置结论常量；
    2. **缺哪个实验的数据就明说"这项给不出结论"**，绝不默认、绝不外推；
    3. **推荐必须带代价**，并带上这些数字的**适用范围**（取自结果自带的 ``env`` 指纹）。

输出：``results/recommend_<tag>.{json,md}``，Markdown 可再用
``python -m inferbench report results/recommend_<tag>.json`` 重出。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

from inferbench import config
from inferbench import fingerprint as fp
from inferbench.stats import read_json, write_json

# 显存安全余量：footprint 已含该模型在 num_ctx 下的 KV cache，但机器上还有别的进程
VRAM_MARGIN = 0.9


# ============================================================
# 载入与挑选数据源
# ============================================================
def _candidates(results_dir: Path, pattern: str, exclude: tuple[str, ...] = ()) -> list[tuple[Path, dict]]:
    """按文件名模式收集候选结果。

    ``exclude`` 不是洁癖：``cache_*.json`` 会连 ``cache_gate_*.json`` 一起匹配到，
    而后者根本没有 ``thresholds`` 字段 —— 混进来会让"缓存决策"误判成无数据。
    """
    out: list[tuple[Path, dict]] = []
    for name in sorted(glob.glob(str(Path(results_dir) / pattern))):
        p = Path(name)
        if any(p.name.startswith(prefix) for prefix in exclude):
            continue
        try:
            out.append((p, read_json(p)))
        except Exception:  # noqa: BLE001
            continue
    return out


def _info_score(kind: str, payload: dict) -> float:
    """信息量打分：用于在同一实验的多份结果里挑"证据最全"的那份。

    quant 有多份（完整阶梯 / 含 0.6b 对照 / 冒烟），必须挑记录最多的那份 ——
    否则新跑的一次 12 条冒烟会把正式结果顶掉（这是真实会踩的坑）。
    """
    if kind == "quant":
        return float(len(payload.get("records", [])))
    if kind == "cache":
        return float((payload.get("meta") or {}).get("stream_size") or 0)
    if kind == "gate":
        return float((payload.get("meta") or {}).get("stream_size") or 0)
    if kind == "spec":
        return float(len(payload.get("runs", [])))
    if kind == "load":
        return float(len(payload.get("levels", [])))
    return 0.0


def _pick(kind: str, items: list[tuple[Path, dict]], override: str | None = None) -> tuple[Path, dict] | None:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = config.ROOT / p
        if not p.exists():
            raise SystemExit(f"指定的 {kind} 结果文件不存在: {p}")
        return p, read_json(p)
    if not items:
        return None
    return max(items, key=lambda it: (_info_score(kind, it[1]), it[1].get("created_at", "")))


def _scope_of(payload: dict) -> str:
    env = payload.get("env") or {}
    return env.get("scope") or "未记录（该结果早于环境指纹功能）"


def _corpus_of(payload: dict) -> dict | None:
    """结果用的语料身份（来源 / 条数 / 内容指纹）。

    准确率类结论**只在那份语料上成立**，所以推荐必须把它一起带上 ——
    否则"Q8 比 Q4 准 5pp"这句话在换了一份语料之后就没人说得清还成不成立。
    """
    info = payload.get("eval_set")
    if not info:
        return None
    return {"source": info.get("source", "未记录"), "total": info.get("total"),
            "hard": info.get("hard"), "sha1_8": info.get("sha1_8")}


# ============================================================
# 四项决策
# ============================================================
def decide_quant(payload: dict, vram_gb: float | None, quality_floor: float | None) -> dict:
    """量化档位：在约束内取**显存最省**的一档（够用就好），并把代价摆全。"""
    summaries = [r["summary"] for r in payload.get("runs", [])
                 if r.get("summary") and not r.get("error")]
    if not summaries:
        return {"status": "no_data", "reason": "量化结果里没有成功的运行记录"}

    usable = (vram_gb * VRAM_MARGIN) if vram_gb else None
    pool = [s for s in summaries if s.get("footprint_gb") is not None]
    feasible = [s for s in pool
                if (usable is None or s["footprint_gb"] <= usable)
                and (quality_floor is None or s["accuracy"] >= quality_floor)]

    table = [{
        "model": s["model"], "footprint_gb": s["footprint_gb"],
        "accuracy": s["accuracy"], "accuracy_hard": s["accuracy_hard"],
        "invalid_rate": s.get("invalid_rate"), "ttft_ms": s.get("ttft_ms_median"),
        "decode_tps": s.get("decode_tps_median"), "wall_ms_p95": s.get("wall_ms_p95"),
        "fits_vram": bool(usable is None or s["footprint_gb"] <= usable),
        "meets_floor": bool(quality_floor is None or s["accuracy"] >= quality_floor),
    } for s in sorted(pool, key=lambda x: -x["footprint_gb"])]

    if not feasible:
        cheapest = min(pool, key=lambda s: s["footprint_gb"])
        reasons = []
        if usable is not None and cheapest["footprint_gb"] > usable:
            reasons.append(f"显存不够：最省的一档也要 {cheapest['footprint_gb']:.2f} GB，"
                           f"可用 {usable:.2f} GB（已留 {(1 - VRAM_MARGIN) * 100:.0f}% 余量）")
        if quality_floor is not None and max(s["accuracy"] for s in pool) < quality_floor:
            reasons.append(f"没有一档能达到准确率下限 {quality_floor * 100:.1f}%"
                           f"（最高 {max(s['accuracy'] for s in pool) * 100:.1f}%）")
        return {"status": "infeasible", "reasons": reasons, "table": table}

    # 选型目标取决于你有没有给质量下限：
    #   给了下限 → 「满足下限的最省一档」（够用就好，省显存是明确目标）；
    #   没给下限 → 「显存允许内准确率最高的一档」。此时**没有理由用精度换显存** ——
    #   本工程的实测就是：更小的一档会在难例上多掉一截，而你并没有说可以接受。
    if quality_floor is None:
        best = max(feasible, key=lambda s: (s["accuracy"], -s["footprint_gb"]))
        basis = "未给准确率下限 → 在显存允许的候选里取**准确率最高**的一档（没有理由主动用精度换显存）"
    else:
        best = min(feasible, key=lambda s: s["footprint_gb"])
        basis = ("给了准确率下限 → 在满足「显存 + 下限」的候选里取**显存最省**的一档（够用就好）")

    best_hard = max(feasible, key=lambda s: s["accuracy_hard"])
    cheapest = min(feasible, key=lambda s: s["footprint_gb"])
    dec = {
        "status": "ok",
        "recommended": {
            "model": best["model"], "footprint_gb": best["footprint_gb"],
            "accuracy": best["accuracy"], "accuracy_hard": best["accuracy_hard"],
            "invalid_rate": best.get("invalid_rate"), "ttft_ms": best.get("ttft_ms_median"),
            "decode_tps": best.get("decode_tps_median"), "wall_ms_p95": best.get("wall_ms_p95"),
        },
        "hard_case_best": {"model": best_hard["model"], "accuracy_hard": best_hard["accuracy_hard"],
                           "footprint_gb": best_hard["footprint_gb"]},
        "basis": basis,
        "table": table,
    }
    if cheapest["model"] != best["model"]:
        dec["cheaper_alternative"] = {
            "model": cheapest["model"], "footprint_gb": cheapest["footprint_gb"],
            "accuracy": cheapest["accuracy"], "accuracy_hard": cheapest["accuracy_hard"],
            "saves_gb": best["footprint_gb"] - cheapest["footprint_gb"],
            "accuracy_cost_pp": (best["accuracy"] - cheapest["accuracy"]) * 100,
            "hard_cost_pp": (best["accuracy_hard"] - cheapest["accuracy_hard"]) * 100,
        }

    # 难例代价：用数据本身算，不写死倍数。只有"推荐档确实在难例上更差"才值得说 ——
    # 否则会输出"低 0.0pp"这种废话（推荐档本身就是难例最优时就会这样）。
    hard_gap = (best_hard["accuracy_hard"] - best["accuracy_hard"]) * 100
    if best["model"] != best_hard["model"] and hard_gap > 0.05:
        overall_drop = (max(s["accuracy"] for s in feasible) - best["accuracy"]) * 100
        dec["hard_case_cost"] = {
            "vs_model": best_hard["model"],
            "overall_drop_pp": overall_drop, "hard_drop_pp": hard_gap,
            "amplification": (hard_gap / overall_drop) if overall_drop > 1e-9 else None,
        }
    es = payload.get("eval_set") or {}
    if es.get("total"):
        dec["eval_set"] = {"total": es["total"], "hard": es.get("hard"),
                           "hard_share": (es.get("hard") or 0) / es["total"]}
    return dec


def decide_cache(cache: dict | None, gate: dict | None) -> dict:
    """语义缓存：阈值取「省得最多 且 不牺牲端到端准确率 且 无跨意图误命中」的一档。"""
    dec: dict = {}
    rows = (cache or {}).get("thresholds") or []
    if rows:
        tolerable = [r for r in rows
                     if r["end_to_end_accuracy"] >= r["nocache_accuracy"] - 1e-9
                     and (r.get("false_hit_b") or 0) == 0]
        pick = max(tolerable or rows, key=lambda r: r["cost_saved_pct"])
        dec["cache"] = {
            "status": "ok",
            "threshold": pick["threshold"], "cost_saved_pct": pick["cost_saved_pct"],
            "hit_rate": pick["hit_rate"], "false_hit_rate": pick["false_hit_rate"],
            "false_hit_a": pick.get("false_hit_a"), "false_hit_b": pick.get("false_hit_b"),
            "end_to_end_accuracy": pick["end_to_end_accuracy"],
            "nocache_accuracy": pick["nocache_accuracy"],
            "hit_ms_p95": pick.get("hit_ms_p95"), "miss_ms_p95": pick.get("miss_ms_p95"),
            "basis": "在「端到端准确率不低于无缓存基线 且 跨意图误命中(B类)=0」的档位里，取成本节省最大的一档",
            "selection_met": bool(tolerable),
            "table": [{"threshold": r["threshold"], "cost_saved_pct": r["cost_saved_pct"],
                       "hit_rate": r["hit_rate"], "false_hit_rate": r["false_hit_rate"],
                       "false_hit_a": r.get("false_hit_a"), "false_hit_b": r.get("false_hit_b"),
                       "end_to_end_accuracy": r["end_to_end_accuracy"],
                       "nocache_accuracy": r["nocache_accuracy"]} for r in rows],
        }
        # 阈值是不是杠杆：用数据里的极差判断，不引用结论
        hi = max(rows, key=lambda r: r["threshold"])
        lo = min(rows, key=lambda r: r["threshold"])
        dec["cache"]["threshold_is_lever"] = {
            "false_hit_rate_range": [lo["false_hit_rate"], hi["false_hit_rate"]],
            "note": "误命中率随阈值上升而下降 → 阈值是可用的杠杆"
                    if hi["false_hit_rate"] < lo["false_hit_rate"] - 1e-9
                    else "误命中率不随阈值单调下降 → 调阈值治不了误命中",
        }
    else:
        dec["cache"] = {"status": "no_data", "reason": "没有实验二（语义缓存）的结果文件"}

    if gate:
        off, on = gate.get("off") or {}, gate.get("on") or {}
        meta = gate.get("meta") or {}
        dec["gate"] = {
            "status": "ok",
            "cost_saved_pct_off": off.get("cost_saved_pct"),
            "cost_saved_pct_on": on.get("cost_saved_pct"),
            "false_hit_off": off.get("false_hit_count"), "false_hit_on": on.get("false_hit_count"),
            "latency_median_off": off.get("latency_ms_median"), "latency_median_on": on.get("latency_ms_median"),
            "unanimous_rate": meta.get("unanimous_rate"),
            "unanimous_count": meta.get("unanimous_count"), "unique_texts": meta.get("unique_texts"),
            "n": meta.get("n"),
            "worth_it": bool((on.get("cost_saved_pct") or 0) > (off.get("cost_saved_pct") or 0)
                             and (on.get("false_hit_count") or 0) < (off.get("false_hit_count") or 0)),
        }
    else:
        dec["gate"] = {"status": "no_data", "reason": "没有实验三（写入门槛）的结果文件"}
    return dec


def decide_spec(spec: dict | None) -> dict:
    """投机解码：逐负载取最优**投机配置**，并给出噪声下限与草稿成本。

    两个必须排除的干扰：
      * ``base`` / ``base2`` 是**基线与其重复性对照**，不是投机配置 —— 如果把它们混进候选，
        ``base2`` 凭 ±1% 的测量抖动就可能"赢"（实测 155.3 vs 154.2 tok/s），
        于是工具会推荐"开 base2"这种荒唐结论；
      * 反过来，``base2`` 与 ``base`` 的差值恰好就是**测量噪声下限** ——
        小于这个幅度的加速都不足采信。这个数要算出来告诉用户。
    """
    if not spec:
        return {"status": "no_data", "reason": "没有实验四（投机解码）的结果文件"}
    runs = {r["config"]: r for r in spec.get("runs", []) if not r.get("error")}
    if not runs:
        return {"status": "no_data", "reason": "投机解码结果里没有成功的配置"}

    names = list(runs)
    baseline_names = [n for n in names if re.fullmatch(r"base\d*", n)]
    base_name = "base" if "base" in runs else (baseline_names[0] if baseline_names else names[0])
    controls = [n for n in baseline_names if n != base_name]
    cand_names = [n for n in names if n not in baseline_names]
    if not cand_names:
        return {"status": "no_data", "reason": "结果里只有基线，没有投机解码配置"}

    draft_names = [n for n in cand_names if n.startswith("draft")]
    ngram_names = [n for n in cand_names if n.startswith("ngram")]

    workloads = list((runs[base_name].get("workloads") or {}).keys())
    per_workload = {}
    noise = 0.0
    for w in workloads:
        base = (runs[base_name].get("workloads") or {}).get(w) or {}
        base_tps = base.get("decode_tps_median") or 0.0
        for ctrl in controls:
            cw = (runs[ctrl].get("workloads") or {}).get(w) or {}
            if base_tps and cw.get("decode_tps_median"):
                noise = max(noise, abs(cw["decode_tps_median"] / base_tps - 1.0))
        cands = []
        for n in cand_names:
            wd = (runs[n].get("workloads") or {}).get(w)
            if not wd or wd.get("decode_tps_median") is None:
                continue
            cands.append({
                "config": n, "decode_tps": wd["decode_tps_median"],
                "speedup": (wd["decode_tps_median"] / base_tps) if base_tps else None,
                "accept_rate": wd.get("accept_rate_median"), "ttft_ms": wd.get("ttft_ms_median"),
            })
        if not cands:
            continue
        best = max(cands, key=lambda c: c["decode_tps"])
        per_workload[w] = {"baseline_config": base_name, "baseline_tps": base_tps,
                           "best": best, "candidates": sorted(cands, key=lambda c: -c["decode_tps"])}

    meta = spec.get("meta") or {}
    draft_cost = {
        "target": meta.get("target"), "target_gb": meta.get("target_gb"),
        "draft": meta.get("draft"), "draft_gb": meta.get("draft_gb"),
        "draft_share_of_target": ((meta.get("draft_gb") or 0) / meta["target_gb"])
        if meta.get("target_gb") else None,
    }
    losers = {n: [round((c["speedup"] or 0), 3) for w in per_workload.values()
                  for c in w["candidates"] if c["config"] == n] for n in draft_names}
    return {"status": "ok", "baseline": base_name, "controls": controls,
            "noise_floor_pct": noise * 100,
            "per_workload": per_workload,
            "draft_cost": draft_cost, "draft_configs": draft_names, "ngram_configs": ngram_names,
            "draft_speedups": losers,
            "all_draft_configs_slower": bool(losers) and all(
                s < 1.0 for v in losers.values() for s in v)}


def decide_load(loads: list[tuple[Path, dict]], slo_ttft: float | None,
                want_concurrency: int | None) -> dict:
    """并发与容量：给定 TTFT P95 目标，找出各并行度下能承载的最大并发。"""
    if not loads:
        return {"status": "no_data", "reason": "没有实验五（并发压测）的结果文件"}

    configs = []
    for path, payload in loads:
        meta = payload.get("meta") or {}
        levels = [L for L in payload.get("levels", []) if L.get("ttft_p95") is not None]
        if not levels:
            continue
        levels.sort(key=lambda L: L["concurrency"])
        # 没有延迟目标就没有"承载能力"这一说：P=1 也能收下 16 路请求，
        # 只是 TTFT P95 会到几秒 —— 那不是容量，那是排队。
        within = [L for L in levels if L["ttft_p95"] <= slo_ttft] if slo_ttft is not None else []
        cap = within[-1] if within else None
        configs.append({
            "file": path.name,
            "num_parallel": meta.get("ollama_num_parallel"),
            "host": meta.get("host"),
            "model": meta.get("model"),
            "footprint_gb": meta.get("footprint_gb"),
            "levels": [{"concurrency": L["concurrency"], "qps": L["qps"],
                        "tokens_per_s": L.get("tokens_per_s"), "ttft_p50": L.get("ttft_p50"),
                        "ttft_p95": L["ttft_p95"], "lat_p95": L.get("lat_p95"),
                        "lat_p99": L.get("lat_p99"), "error_rate": L.get("error_rate")}
                       for L in levels],
            "measured_max_concurrency": levels[-1]["concurrency"],
            "capacity_at_slo": ({"concurrency": cap["concurrency"], "qps": cap["qps"],
                                 "ttft_p95": cap["ttft_p95"]} if cap else None),
            "saturation": payload.get("saturation"),
        })
    if not configs:
        return {"status": "no_data", "reason": "并发结果里没有可用的并发梯度"}

    configs.sort(key=lambda c: (c["num_parallel"] if c["num_parallel"] is not None else 0))
    result = {"status": "ok", "slo_ttft_ms": slo_ttft, "configs": configs}
    if slo_ttft is None:
        result["recommended"] = None
        result["capacity_note"] = ("未设 `--slo-ttft`：**给不出承载能力结论**。"
                                   "并发数本身不是容量 —— 不设延迟目标的话，单 slot 也能收下 16 路请求，"
                                   "只是首 Token 要等几秒。加上 `--slo-ttft <ms>` 再来看能扛几路。")
        if want_concurrency is not None:
            result["meets_target"] = {
                "ok": False, "want_concurrency": want_concurrency,
                "reason": f"没设 `--slo-ttft`，无法判断能否承载 {want_concurrency} 路 —— "
                          f"并发不配延迟目标是没有意义的（补上目标延迟再问）",
            }
        return result

    best = max(configs, key=lambda c: (c["capacity_at_slo"] or {}).get("concurrency") or 0)
    result["recommended"] = {"file": best["file"], "num_parallel": best["num_parallel"],
                             "capacity_at_slo": best["capacity_at_slo"],
                             "footprint_gb": best["footprint_gb"]}

    if len(configs) >= 2:
        lo, hi = configs[0], configs[-1]
        cap_lo = (lo["capacity_at_slo"] or {}).get("concurrency")
        cap_hi = (hi["capacity_at_slo"] or {}).get("concurrency")
        fp_lo, fp_hi = lo.get("footprint_gb"), hi.get("footprint_gb")
        result["parallel_upgrade"] = {
            "from_num_parallel": lo["num_parallel"], "to_num_parallel": hi["num_parallel"],
            "capacity_from": cap_lo, "capacity_to": cap_hi,
            "vram_from_gb": fp_lo, "vram_to_gb": fp_hi,
            "vram_cost_pct": (((fp_hi - fp_lo) / fp_lo * 100) if fp_lo else None),
        }

    if want_concurrency is not None:
        caps = [(c["num_parallel"], (c["capacity_at_slo"] or {}).get("concurrency")) for c in configs]
        fits = [c for c in configs
                if ((c["capacity_at_slo"] or {}).get("concurrency") or 0) >= want_concurrency]
        if fits:
            chosen = min(fits, key=lambda c: (c["footprint_gb"] or 0))
            result["meets_target"] = {"ok": True, "want_concurrency": want_concurrency,
                                      "num_parallel": chosen["num_parallel"],
                                      "file": chosen["file"],
                                      "capacity_at_slo": chosen["capacity_at_slo"],
                                      "footprint_gb": chosen["footprint_gb"]}
        else:
            result["meets_target"] = {
                "ok": False, "want_concurrency": want_concurrency, "measured_caps": caps,
                "reason": f"已测的并行度下，满足 SLO 的最大并发都不足 {want_concurrency} 路"
                          f"（{caps}）；更高并行度没有实测数据，本工具不外推 —— "
                          f"需要的话请跑一次 `python -m inferbench load --num-parallel N`。",
            }
    return result


# ============================================================
# 适用范围比对
# ============================================================
def check_scope(used: list[tuple[str, dict]], current: dict) -> dict:
    """拿用到的每份结果的 env 指纹，与本机指纹比对，指出结论可能不适用的地方。"""
    cur_gpu = (current.get("gpu") or {})
    cur_oll = (current.get("ollama") or {})
    warnings: list[str] = []
    unknown: list[str] = []
    for label, payload in used:
        env = payload.get("env")
        if not env:
            unknown.append(label)          # 具体"未记录"到什么程度，报告第 2 节的表里已经写了
            continue
        gpu = env.get("gpu") or {}
        oll = env.get("ollama") or {}
        if gpu.get("name") and cur_gpu.get("name") and gpu["name"] != cur_gpu["name"]:
            warnings.append(f"{label}：数据来自 `{gpu['name']}`，本机是 `{cur_gpu['name']}` "
                            f"—— 显存相关结论（并发容量、量化档位）可能不成立")
        elif gpu.get("memory_total_mb") and cur_gpu.get("memory_total_mb") and \
                abs(gpu["memory_total_mb"] - cur_gpu["memory_total_mb"]) > 256:
            warnings.append(f"{label}：数据来自 {gpu['memory_total_mb']:.0f} MiB 显存，"
                            f"本机 {cur_gpu['memory_total_mb']:.0f} MiB —— 容量结论需重测")
        if oll.get("version") and cur_oll.get("version") and \
                oll["version"] != "unknown" and cur_oll["version"] != "unknown" and \
                oll["version"] != cur_oll["version"]:
            warnings.append(f"{label}：数据来自 Ollama {oll['version']}，本机 {cur_oll['version']} "
                            f"—— 调度/并发行为可能与版本相关")
        if oll.get("num_parallel_env") != cur_oll.get("num_parallel_env"):
            warnings.append(f"{label}：数据采集时 OLLAMA_NUM_PARALLEL="
                            f"{oll.get('num_parallel_env')}，本机 {cur_oll.get('num_parallel_env')} "
                            f"—— 并发结论直接受这个值影响")
    return {"warnings": warnings, "unknown_scope": unknown}


def check_corpus(used: list[tuple[str, dict]]) -> dict:
    """多份结果混用不同语料时，准确率的横向对比是不成立的。"""
    seen: dict[str, list[str]] = {}
    unknown: list[str] = []
    for label, payload in used:
        info = _corpus_of(payload)
        if not info or not info.get("sha1_8"):
            unknown.append(label)
            continue
        seen.setdefault(info["sha1_8"], []).append(label)
    warnings = []
    if len(seen) > 1:
        groups = "；".join(f"指纹 `{sha}`：{'、'.join(labels)}" for sha, labels in seen.items())
        warnings.append(f"这些结果用的**不是同一份语料**（{groups}）—— "
                        f"跨语料的准确率横向对比不成立，请用同一份语料重跑后再比")
    return {"corpora": seen, "unknown": unknown, "warnings": warnings}


# ============================================================
# 组装
# ============================================================
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m inferbench recommend",
        description="把实验结果收敛成「本机该怎么配」：读 results/*.json，输出推荐配置与代价。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python -m inferbench recommend --vram 8 --slo-ttft 200 --concurrency 4\n"
               "  python -m inferbench recommend --quality-floor 0.9 --json\n",
    )
    p.add_argument("--results", default=str(config.RESULTS_DIR), help="结果目录（默认 results/）")
    p.add_argument("--vram", type=float, default=None, help="可用显存 GB（默认探测本机）")
    p.add_argument("--slo-ttft", type=float, default=None, help="首 Token 延迟目标：TTFT P95 (ms)")
    p.add_argument("--concurrency", type=int, default=None, help="需要承载的并发路数")
    p.add_argument("--quality-floor", type=float, default=None,
                   help="可接受的最低整体准确率（0~1，例如 0.9）")
    p.add_argument("--quant-file", default=None, help="手动指定实验一结果文件")
    p.add_argument("--cache-file", default=None, help="手动指定实验二结果文件")
    p.add_argument("--gate-file", default=None, help="手动指定实验三结果文件")
    p.add_argument("--spec-file", default=None, help="手动指定实验四结果文件")
    p.add_argument("--load-file", default=None, help="手动指定实验五结果文件（可重复用逗号分隔）")
    p.add_argument("--tag", default="", help="输出标签，产物为 results/recommend_<tag>.{json,md}")
    p.add_argument("--json", action="store_true", help="只打印 JSON")
    return p.parse_args(argv)


def build(argv: list[str]) -> tuple[dict, list[Path]]:
    args = _parse_args(argv)
    rdir = Path(args.results)
    if not rdir.is_absolute():
        rdir = config.ROOT / rdir

    quant_cands = _candidates(rdir, "quant_*.json")
    cache_cands = _candidates(rdir, "cache_*.json", exclude=("cache_gate_",))
    gate_cands = _candidates(rdir, "cache_gate_*.json")
    spec_cands = _candidates(rdir, "spec_*.json")
    load_items = _candidates(rdir, "load_*.json")

    quant = _pick("quant", quant_cands, args.quant_file)
    cache = _pick("cache", cache_cands, args.cache_file)
    gate = _pick("gate", gate_cands, args.gate_file)
    spec = _pick("spec", spec_cands, args.spec_file)
    if args.load_file:
        load_items = []
        for f in args.load_file.split(","):
            p = Path(f.strip())
            if not p.is_absolute():
                p = config.ROOT / p
            if not p.exists():
                raise SystemExit(f"指定的并发结果文件不存在: {p}")
            load_items.append((p, read_json(p)))
    # 同一实验可能有多份结果：全部纳进来，并在报告里列出"未采用"的是哪些
    loads = load_items
    unused = {}
    for key, cands, picked in (("quant", quant_cands, quant), ("cache", cache_cands, cache),
                               ("gate", gate_cands, gate), ("spec", spec_cands, spec)):
        others = [c[0].name for c in cands if not picked or c[0] != picked[0]]
        if others:
            unused[key] = others

    current_env = fp.fingerprint()
    if args.vram is not None:
        vram, vram_source = args.vram, "命令行指定"
    elif (current_env.get("gpu") or {}).get("memory_total_mb"):
        vram, vram_source = current_env["gpu"]["memory_total_mb"] / 1024.0, "本机探测"
    else:
        vram, vram_source = None, "未指定（因此不按显存过滤候选）"

    used = []
    for label, item in (("实验一·量化", quant), ("实验二·缓存", cache), ("实验三·门槛", gate),
                        ("实验四·投机解码", spec)):
        if item:
            used.append((label, item[1]))
    for path, payload in loads:
        used.append((f"实验五·并发（{path.name}）", payload))

    payload = {
        "kind": "recommend",
        "tag": args.tag or "default",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": {
            "vram_gb": vram, "vram_source": vram_source, "vram_margin": VRAM_MARGIN,
            "slo_ttft_ms": args.slo_ttft, "want_concurrency": args.concurrency,
            "quality_floor": args.quality_floor,
        },
        "sources": {
            "quant": {"file": quant[0].name, "created_at": quant[1].get("created_at"),
                      "scope": _scope_of(quant[1]), "corpus": _corpus_of(quant[1])} if quant else None,
            "cache": {"file": cache[0].name, "created_at": cache[1].get("created_at"),
                      "scope": _scope_of(cache[1]), "corpus": _corpus_of(cache[1])} if cache else None,
            "gate": {"file": gate[0].name, "created_at": gate[1].get("created_at"),
                     "scope": _scope_of(gate[1]), "corpus": _corpus_of(gate[1])} if gate else None,
            "spec": {"file": spec[0].name, "created_at": spec[1].get("created_at"),
                     "scope": _scope_of(spec[1]), "corpus": _corpus_of(spec[1])} if spec else None,
            "load": [{"file": p.name, "created_at": pl.get("created_at"),
                      "scope": _scope_of(pl), "corpus": _corpus_of(pl)} for p, pl in loads],
        },
        "unused_candidates": unused,
        "quant": decide_quant(quant[1], vram, args.quality_floor) if quant else
        {"status": "no_data", "reason": "没有实验一（量化）的结果文件"},
        "cache_decision": decide_cache(cache[1] if cache else None, gate[1] if gate else None),
        "spec_decision": decide_spec(spec[1] if spec else None),
        "load_decision": decide_load(loads, args.slo_ttft, args.concurrency),
        "scope_check": check_scope(used, current_env),
        "corpus_check": check_corpus(used),
        "env": current_env,
    }
    tag = payload["tag"]
    json_path = config.RESULTS_DIR / f"recommend_{tag}.json"
    write_json(json_path, payload)
    md_path = build_report(payload)
    return payload, [json_path, md_path]


# ============================================================
# 报告
# ============================================================
def _fmt(value, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def build_report(payload: dict) -> Path:
    sc = payload["scenario"]
    tag = payload.get("tag", "default")
    path = config.RESULTS_DIR / f"recommend_{tag}.md"

    md: list[str] = ["# 选型建议（从实验结果收敛而来）\n"]
    md.append(f"- 生成时间：{payload['created_at']}")
    md.extend(fp.report_lines(payload))
    md.append("")

    md.append("## 1. 场景（你的约束）\n")
    md.append(f"- 可用显存：**{_fmt(sc['vram_gb'], ' GB')}**（{sc['vram_source']}，"
              f"实际按 {sc['vram_margin'] * 100:.0f}% 计 = {_fmt((sc['vram_gb'] or 0) * sc['vram_margin'], ' GB')}）")
    md.append(f"- 首 Token 延迟目标：**{_fmt(sc['slo_ttft_ms'], ' ms')}**（TTFT P95）"
              if sc["slo_ttft_ms"] else "- 首 Token 延迟目标：**未设**（不按 SLO 过滤容量）")
    md.append(f"- 需要承载并发：**{_fmt(sc['want_concurrency'], ' 路')}**"
              if sc["want_concurrency"] else "- 需要承载并发：**未设**")
    md.append(f"- 准确率下限：**{_fmt((sc['quality_floor'] or 0) * 100, '%', 1)}**"
              if sc["quality_floor"] else "- 准确率下限：**未设**")
    md.append("")

    md.append("## 2. 用到的数据与适用范围\n")
    md.append("| 实验 | 文件 | 采集时间 | 适用范围 |")
    md.append("|---|---|---|---|")
    src = payload["sources"]
    for label, key in (("实验一·量化", "quant"), ("实验二·缓存", "cache"),
                       ("实验三·门槛", "gate"), ("实验四·投机解码", "spec")):
        item = src.get(key)
        md.append(f"| {label} | `{item['file']}` | {item.get('created_at') or '未记录'} | "
                  f"{item.get('scope') or '—'} |" if item else f"| {label} | ⚠ 缺 | — | — |")
    for item in src.get("load", []):
        md.append(f"| 实验五·并发 | `{item['file']}` | {item.get('created_at') or '未记录'} | "
                  f"{item.get('scope') or '—'} |")
    if not src.get("load"):
        md.append("| 实验五·并发 | ⚠ 缺 | — | — |")
    md.append("")
    unused = payload.get("unused_candidates") or {}
    if unused:
        md.append("同一实验有多份结果时，本工具按「记录最多 → 采集最新」自动挑选；"
                  "**本次未采用**的候选如下（可用 `--quant-file` 等参数手动指定）：\n")
        for key, names in unused.items():
            md.append(f"- {key}：{'、'.join('`' + n + '`' for n in names)}")
        md.append("")

    corpora = payload.get("corpus_check") or {}
    listed = [("实验一·量化", src.get("quant")), ("实验二·缓存", src.get("cache")),
              ("实验三·门槛", src.get("gate"))]
    listed += [("实验五·并发", item) for item in src.get("load", [])]
    with_corpus = [(label, item) for label, item in listed if item and item.get("corpus")]
    if with_corpus:
        md.append("**这些准确率只在下列语料上成立**（语料换了结论要重测）：\n")
        for label, item in with_corpus:
            c = item["corpus"]
            md.append(f"- {label}（`{item['file']}`）：{c['source']} · {c['total']} 条"
                      f"（难例 {c['hard']}）· 指纹 `{c['sha1_8'] or '未记录'}`")
        md.append("")
    for w in corpora.get("warnings") or []:
        md.append(f"⚠ {w}\n")

    chk = payload.get("scope_check") or {}
    if chk.get("warnings"):
        md.append("### ⚠ 适用范围警示\n")
        for w in chk["warnings"]:
            md.append(f"- {w}")
        md.append("")
    if chk.get("unknown_scope"):
        md.append("### 适用边界未知的数据\n")
        md.append("下面这些结果没有环境指纹，**不要直接跨机器套用**：\n")
        for u in chk["unknown_scope"]:
            md.append(f"- {u}")
        md.append("")

    md.append("## 3. 推荐配置\n")
    md.append("| 决策 | 建议 | 代价 | 依据（数字来自哪份结果） |")
    md.append("|---|---|---|---|")

    # 3.1 量化
    q = payload["quant"]
    if q.get("status") == "ok":
        r = q["recommended"]
        cost = (f"显存 {r['footprint_gb']:.2f} GB · 整体准确率 {r['accuracy'] * 100:.1f}% · "
                f"难例 {r['accuracy_hard'] * 100:.1f}% · 解码 {_fmt(r['decode_tps'], ' tok/s', 1)} · "
                f"TTFT {_fmt(r['ttft_ms'], ' ms', 1)}")
        md.append(f"| 量化档位 | `{r['model']}` | {cost} | 实验一（{q['basis']}） |")
    elif q.get("status") == "infeasible":
        md.append(f"| 量化档位 | ⚠ 约束不可满足 | {'；'.join(q.get('reasons') or [])} | 实验一 |")
    else:
        md.append(f"| 量化档位 | ⚠ 给不出结论 | {q.get('reason')} | — |")

    # 3.2 缓存
    c = (payload.get("cache_decision") or {}).get("cache") or {}
    if c.get("status") == "ok":
        cost = (f"成本省 {c['cost_saved_pct']:.1f}% · 命中率 {c['hit_rate'] * 100:.1f}% · "
                f"端到端准确率 {c['end_to_end_accuracy'] * 100:.1f}%"
                f"（无缓存基线 {c['nocache_accuracy'] * 100:.1f}%）· "
                f"误命中 {c['false_hit_rate'] * 100:.1f}%（A类 {c['false_hit_a']} / B类 {c['false_hit_b']}）")
        md.append(f"| 语义缓存阈值 | **{c['threshold']}** | {cost} | 实验二（{c['basis']}） |")
        lever = c.get("threshold_is_lever") or {}
        if lever:
            md.append(f"| 阈值还能再调吗 | "
                      f"{'可以再扫' if '可用的杠杆' in (lever.get('note') or '') else '**不要再调**'} | "
                      f"误命中率在阈值两端为 {lever['false_hit_rate_range'][0] * 100:.1f}% → "
                      f"{lever['false_hit_rate_range'][1] * 100:.1f}% | 实验二 |")
    elif c:
        md.append(f"| 语义缓存阈值 | ⚠ 给不出结论 | {c.get('reason')} | — |")

    # 3.3 门槛（负结果 → 明确"不要做"）
    g = (payload.get("cache_decision") or {}).get("gate") or {}
    if g.get("status") == "ok":
        if g.get("worth_it"):
            md.append(f"| 加自一致性写入门槛 | 可以试 | 成本节省 "
                      f"{_fmt(g['cost_saved_pct_off'], '%', 1)} → {_fmt(g['cost_saved_pct_on'], '%', 1)} | 实验三 |")
        else:
            md.append(f"| 加自一致性写入门槛 | **不要做**（负结果） | "
                      f"成本节省 {_fmt(g['cost_saved_pct_off'], '%', 1)} → **{_fmt(g['cost_saved_pct_on'], '%', 1)}**"
                      f"（由省转亏），误命中只从 {g['false_hit_off']} 降到 {g['false_hit_on']} 次 | "
                      f"实验三（采样一致率 {_fmt((g['unanimous_rate'] or 0) * 100, '%', 1)}："
                      f"{g['unanimous_count']}/{g['unique_texts']} 条唯一文本的 {g['n']} 次采样完全一致） |")

    # 3.4 投机解码
    s = payload.get("spec_decision") or {}
    if s.get("status") == "ok":
        for w, item in s["per_workload"].items():
            best = item["best"]
            md.append(f"| 投机解码（{w}） | `{best['config']}` | 加速 "
                      f"**{_fmt(best['speedup'], '×')}**（{best['decode_tps']:.1f} vs 基线 "
                      f"{item['baseline_tps']:.1f} tok/s）· 接受率 "
                      f"{_fmt(best['accept_rate'], '%', 1)} | 实验四 |")
        dc = s.get("draft_cost") or {}
        if s.get("all_draft_configs_slower"):
            detail = "、".join(f"`{name}` {'/'.join(f'{v:.2f}×' for v in vals)}"
                              for name, vals in s["draft_speedups"].items())
            md.append(f"| 用草稿模型（draft） | **不要做**（负结果） | 所有草稿配置都慢于基线"
                      f"（按各负载的加速比）：{detail}；草稿模型体积是 target 的 "
                      f"{_fmt((dc.get('draft_share_of_target') or 0) * 100, '%', 1)}"
                      f"（{dc.get('draft_gb')} / {dc.get('target_gb')} GB） | 实验四 |")
        nf = s.get("noise_floor_pct")
        if nf:
            md.append(f"| 多快才算真快（噪声下限） | 重复性对照 `{'、'.join(s.get('controls') or [])}` "
                      f"vs `{s.get('baseline')}` | 同一配置重复测量的差异 ≤ **{nf:.1f}%** —— "
                      f"低于此幅度的「加速」只是抖动，不足采信 | 实验四 |")
    else:
        md.append(f"| 投机解码 | ⚠ 给不出结论 | {s.get('reason')} | — |")

    # 3.5 并发
    L = payload.get("load_decision") or {}
    if L.get("status") == "ok":
        rec = L.get("recommended") or {}
        cap = rec.get("capacity_at_slo") or {}
        if cap:
            md.append(f"| 并行度 / 容量 | `OLLAMA_NUM_PARALLEL={rec.get('num_parallel')}` | "
                      f"在 TTFT P95 ≤ {_fmt(L.get('slo_ttft_ms'), ' ms')} 下可承载 "
                      f"**{cap.get('concurrency')} 路**（QPS {_fmt(cap.get('qps'))}），"
                      f"模型显存 {_fmt(rec.get('footprint_gb'), ' GB')} | 实验五（`{rec.get('file')}`） |")
        elif L.get("capacity_note"):
            md.append(f"| 并行度 / 容量 | ⚠ **给不出结论** | {L['capacity_note']} | 实验五 |")
        else:
            md.append(f"| 并行度 / 容量 | ⚠ 该 SLO 下没有任何并发档达标 | "
                      f"最低实测 TTFT P95 都超过目标 | 实验五 |")
        pu = L.get("parallel_upgrade")
        if pu and pu.get("capacity_to") is not None and pu.get("capacity_from") is not None:
            md.append(f"| 提升并行度值不值 | `NUM_PARALLEL` {pu['from_num_parallel']} → "
                      f"{pu['to_num_parallel']} | 同 SLO 承载力 {pu['capacity_from']} → "
                      f"**{pu['capacity_to']} 路**，代价显存 {_fmt(pu['vram_from_gb'], ' GB')} → "
                      f"{_fmt(pu['vram_to_gb'], ' GB')}"
                      f"（+{_fmt(pu['vram_cost_pct'], '%', 1)}） | 实验五对照 |")
        mt = L.get("meets_target")
        if mt and not mt.get("ok"):
            md.append(f"| 能否承载 {mt['want_concurrency']} 路 | ⚠ **给不出结论** | {mt['reason']} | — |")
        elif mt and mt.get("ok"):
            md.append(f"| 承载 {mt['want_concurrency']} 路 | "
                      f"`NUM_PARALLEL={mt['num_parallel']}` | 达标（该配置下可承载 "
                      f"{(mt['capacity_at_slo'] or {}).get('concurrency')} 路） | 实验五 |")
    else:
        md.append(f"| 并行度 / 容量 | ⚠ 给不出结论 | {L.get('reason')} | — |")
    md.append("")

    # 4. 明细
    if q.get("status") == "ok" and q.get("table"):
        md.append("## 4. 量化候选明细（含被约束排除的档位）\n")
        md.append("| 模型 | 显存 (GB) | 整体准确率 | 难例准确率 | 解码 (tok/s) | TTFT (ms) | 满足显存 | 满足下限 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for t in q["table"]:
            md.append(f"| `{t['model']}` | {t['footprint_gb']:.2f} | {t['accuracy'] * 100:.1f}% | "
                      f"{t['accuracy_hard'] * 100:.1f}% | {_fmt(t['decode_tps'], '', 1)} | "
                      f"{_fmt(t['ttft_ms'], '', 1)} | {'✓' if t['fits_vram'] else '✗'} | "
                      f"{'✓' if t['meets_floor'] else '✗'} |")
        md.append("")
        hc = q.get("hard_case_cost")
        es = q.get("eval_set") or {}
        if hc:
            amp = hc.get("amplification")
            md.append(f"> **难例代价**：推荐档比 `{hc['vs_model']}` 整体低 "
                      f"{hc['overall_drop_pp']:.1f}pp，但**难例低 {hc['hard_drop_pp']:.1f}pp**"
                      + (f"（放大 {amp:.1f} 倍）" if amp else "")
                      + f"。本评测集难例占 {(es.get('hard_share') or 0) * 100:.0f}%"
                        f"（{es.get('hard')}/{es.get('total')} 条）—— "
                        f"你真实流量里模糊输入占比更高的话，请按难例列评估。\n")
        ca = q.get("cheaper_alternative")
        if ca:
            md.append(f"> **还有更省的一档**：`{ca['model']}`（{ca['footprint_gb']:.2f} GB）能再省 "
                      f"{ca['saves_gb']:.2f} GB，代价是整体准确率 {ca['accuracy_cost_pp']:.1f}pp、"
                      f"难例 {ca['hard_cost_pp']:.1f}pp。若这个代价可接受，用 "
                      f"`--quality-floor` 把它表达出来，本工具会改推它。\n")

    if c.get("status") == "ok" and c.get("table"):
        md.append("## 5. 缓存阈值扫描明细\n")
        md.append("| 阈值 | 成本节省 | 命中率 | 误命中率 | A类 | B类 | 端到端准确率 | 无缓存基线 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for t in c["table"]:
            md.append(f"| {t['threshold']:.2f} | {t['cost_saved_pct']:.1f}% | {t['hit_rate'] * 100:.1f}% | "
                      f"{t['false_hit_rate'] * 100:.1f}% | {t['false_hit_a']} | {t['false_hit_b']} | "
                      f"{t['end_to_end_accuracy'] * 100:.1f}% | {t['nocache_accuracy'] * 100:.1f}% |")
        md.append("")

    if L.get("status") == "ok":
        md.append("## 6. 并发容量明细\n")
        for cfg in L["configs"]:
            md.append(f"**`NUM_PARALLEL={cfg['num_parallel']}`**（`{cfg['file']}`"
                      f"，模型 {cfg['model']}，显存 {_fmt(cfg['footprint_gb'], ' GB')}）\n")
            md.append("| 并发 | QPS | tok/s | TTFT P50 | TTFT P95 | P95 | P99 |")
            md.append("|---|---|---|---|---|---|---|")
            for lv in cfg["levels"]:
                md.append(f"| {lv['concurrency']} | {_fmt(lv['qps'])} | {_fmt(lv['tokens_per_s'], '', 1)} | "
                          f"{_fmt(lv['ttft_p50'], '', 1)} | {_fmt(lv['ttft_p95'], '', 1)} | "
                          f"{_fmt(lv['lat_p95'], '', 1)} | {_fmt(lv['lat_p99'], '', 1)} |")
            md.append("")

    md.append("## 7. 这份建议的边界\n")
    md.append("- 所有数字来自上表列出的结果文件，逐项可回溯到 `results/*.json`；本工具不内置任何结论常量。")
    md.append("- 未实测的配置**一律不外推**（例如没跑过 `NUM_PARALLEL=8`，就不会给你 8 路的容量数字）。")
    missing = [k for k, v in (("实验一·量化", src.get("quant")), ("实验二·缓存", src.get("cache")),
                              ("实验三·门槛", src.get("gate")), ("实验四·投机解码", src.get("spec")))
               if not v]
    if not src.get("load"):
        missing.append("实验五·并发")
    if missing:
        md.append(f"- ⚠ **缺失的实验**：{'、'.join(missing)} —— 对应决策给不出结论。"
                  f"补齐后重跑本命令即可。")
    md.append("- 换机器 / 升级 Ollama 之后，先看第 2 节的「适用范围」，必要时重跑对应实验。")
    md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path


def console_summary(payload: dict) -> str:
    sc = payload["scenario"]
    lines = ["=" * 78,
             f"选型建议  |  显存 {_fmt(sc['vram_gb'], ' GB')}  |  "
             f"TTFT P95 ≤ {_fmt(sc['slo_ttft_ms'], ' ms')}  |  目标并发 {_fmt(sc['want_concurrency'])}",
             "=" * 78]
    q = payload["quant"]
    if q.get("status") == "ok":
        r = q["recommended"]
        lines.append(f"  量化档位    : {r['model']}  "
                     f"({r['footprint_gb']:.2f} GB · 准确率 {r['accuracy'] * 100:.1f}% · "
                     f"难例 {r['accuracy_hard'] * 100:.1f}% · {_fmt(r['decode_tps'], ' tok/s', 1)})")
        ca = q.get("cheaper_alternative")
        if ca:
            lines.append(f"  更省的一档  : {ca['model']} 省 {ca['saves_gb']:.2f} GB，"
                         f"准确率代价 -{ca['accuracy_cost_pp']:.1f}pp / 难例 -{ca['hard_cost_pp']:.1f}pp"
                         f"（可接受就用 --quality-floor 表达）")
    elif q.get("status") == "infeasible":
        lines.append(f"  量化档位    : 约束不可满足 —— {'；'.join(q.get('reasons') or [])}")
    else:
        lines.append(f"  量化档位    : 给不出结论（{q.get('reason')}）")

    c = (payload.get("cache_decision") or {}).get("cache") or {}
    if c.get("status") == "ok":
        lines.append(f"  缓存阈值    : {c['threshold']}  "
                     f"(成本省 {c['cost_saved_pct']:.1f}% · 端到端 {c['end_to_end_accuracy'] * 100:.1f}% "
                     f"= 无缓存基线 {c['nocache_accuracy'] * 100:.1f}%)")
    else:
        lines.append(f"  缓存阈值    : 给不出结论（{c.get('reason')}）")

    g = (payload.get("cache_decision") or {}).get("gate") or {}
    if g.get("status") == "ok":
        lines.append(f"  写入门槛    : {'可以试' if g.get('worth_it') else '不要做（负结果）'}  "
                     f"(成本节省 {_fmt(g['cost_saved_pct_off'], '%', 1)} → "
                     f"{_fmt(g['cost_saved_pct_on'], '%', 1)})")

    s = payload.get("spec_decision") or {}
    if s.get("status") == "ok":
        for w, item in s["per_workload"].items():
            lines.append(f"  投机解码    : [{w}] {item['best']['config']}  "
                         f"{_fmt(item['best']['speedup'], '×')}")
    else:
        lines.append(f"  投机解码    : 给不出结论（{s.get('reason')}）")

    L = payload.get("load_decision") or {}
    if L.get("status") == "ok":
        rec = L.get("recommended") or {}
        cap = rec.get("capacity_at_slo") or {}
        if cap:
            lines.append(f"  并发容量    : NUM_PARALLEL={rec.get('num_parallel')} "
                         f"→ {cap.get('concurrency')} 路 @ SLO（QPS {_fmt(cap.get('qps'))}）")
        elif L.get("capacity_note"):
            lines.append("  并发容量    : 给不出结论 —— 未设 --slo-ttft（并发不配延迟目标没有意义）")
        else:
            lines.append("  并发容量    : 该 SLO 下没有任何并发档达标")
        mt = L.get("meets_target")
        if mt and not mt.get("ok"):
            lines.append(f"  目标并发    : 给不出结论 —— {mt['reason']}")
    else:
        lines.append(f"  并发容量    : 给不出结论（{L.get('reason')}）")

    chk = payload.get("scope_check") or {}
    for w in chk.get("warnings") or []:
        lines.append(f"  ⚠ {w}")
    for w in (payload.get("corpus_check") or {}).get("warnings") or []:
        lines.append(f"  ⚠ {w}")
    lines.append("=" * 78)
    return "\n".join(lines)


def run(argv: list[str]) -> int:
    payload, paths = build(argv)
    if "--json" in argv:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(console_summary(payload))
    print(f"JSON: {paths[0]}")
    print(f"报告: {paths[1]}")
    return 0
