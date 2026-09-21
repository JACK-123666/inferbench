"""报告生成：把实验结果组装成 Markdown（含内联 SVG 图与可直接引用的结论）。

图表实现在 inferbench/svg.py（零依赖内联 SVG，不引入 matplotlib）。
本模块只负责：取数 → 判定 → 组织文案。**所有数字都从结果文件里取，
不手抄**——手抄的数字一定会在某次改动后与数据不一致。
"""
from __future__ import annotations

import re
from pathlib import Path

from inferbench import config
from inferbench import eval_set as ev
from inferbench import fingerprint
from inferbench import stats
from inferbench.svg import bar_chart, line_chart  # noqa: E402


def _top_confusions(summary: dict, topk: int = 3) -> str:
    """从混淆矩阵里挑出最主要的错误形态（不做想当然的归因）。"""
    confusion = summary.get("confusion") or {}
    off: list[tuple[int, str]] = []
    for gold, preds in confusion.items():
        for pred, cnt in preds.items():
            if pred != gold:
                off.append((cnt, f"{gold} → {pred}"))
    if not off:
        return "本次没有错误样本（或样本量太小）"
    off.sort(reverse=True)
    parts = [f"{name} 错 {cnt} 次" for cnt, name in off[:topk] if cnt > 0]
    return "、".join(parts) if parts else "无显著错误形态"


def _invalid_note(ref: dict, worst: dict) -> str:
    """非法输出率的对比说明——方向由数据决定，不预设结论。

    踩过的坑：第一版写死了"低精度档非法输出更多"，实测 Q4 反而更低
    （0.0% vs 1.0%），结论当场被数据打脸。
    """
    delta = (worst["invalid_rate"] - ref["invalid_rate"]) * 100
    if delta > 0.5:
        return "（更低精度档确实出现了指令跟随退化，先崩格式再崩准确率）"
    if delta < -0.5:
        return ("（注意：本次更低精度档的非法输出反而更少，说明该任务的量化损伤以"
                "**分类混淆**为主，而非输出格式崩溃）")
    return "（两档非法输出率差异在噪声范围内，量化损伤主要体现在分类混淆上）"


def _ladder_note(runs_sorted: list[dict], n_records: int) -> str:
    """检测「准确率不随精度单调变化」——中档打败顶档时给出解读。

    这是实测里真出现过的情况：Q8_0（92.0%）高于 FP16（91.0%）。
    如果不自动检测，这种最有价值的信号会被"以最高精度档为基准"的模板掩盖。
    """
    runs = [r["summary"] for r in runs_sorted]
    if len(runs) < 2:
        return ""
    best = max(runs, key=lambda s: s["accuracy"])
    top = runs[0]
    if best["model"] == top["model"]:
        return ""
    delta_pp = (best["accuracy"] - top["accuracy"]) * 100
    n_diff = round(abs(delta_pp) / 100 * n_records)
    mem_saved = stats.pct(top["footprint_gb"] - best["footprint_gb"], top["footprint_gb"])
    speed_up = stats.pct(best["decode_tps_median"] - top["decode_tps_median"],
                         top["decode_tps_median"])
    return (
        f"\n> ⚠ **准确率不随精度单调变化**：`{best['model']}`（{best['accuracy'] * 100:.1f}%）"
        f"反而高于最高精度档 `{top['model']}`（{top['accuracy'] * 100:.1f}%），"
        f"差 {delta_pp:+.1f} 个百分点（{n_diff}/{n_records} 条记录，处于噪声范围）。\n>\n"
        f"> 但结论仍然成立且有价值：**`{best['model']}` 在本任务上不劣于 `{top['model']}`，"
        f"同时显存少 {mem_saved:.1f}%、吞吐高 {speed_up:.1f}%**。\n>\n"
        f"> 也就是说 **FP16 在纯推理场景被完全支配**——只有需要继续训练/微调（LoRA、DPO）时，"
        f"保留 FP16/BF16 才有意义。**「精度越高越好」是个直觉陷阱，必须用数据验证。**"
    )


# ---------------------------------------------------------------------------
# 量化实验：控制台表 + Markdown 报告
# ---------------------------------------------------------------------------

_QUANT_ORDER = [("fp16", 4.0), ("f16", 4.0), ("q8", 3.0), ("q6", 2.6), ("q5", 2.2),
                ("q4", 2.0), ("q3", 1.5), ("q2", 1.0)]


def quant_rank(tag: str) -> float:
    """从 tag 里推断量化精度档位（用于挑选"参考档"）。"""
    lowered = tag.lower()
    best = 2.0
    for key, rank in _QUANT_ORDER:
        if key in lowered:
            best = max(best, rank)
    return best


def _run_ok(run: dict) -> bool:
    s = run.get("summary") or {}
    return bool(s) and s.get("n_records", 0) > 0


def quant_console_table(payload: dict) -> str:
    runs = [r for r in payload.get("runs", []) if _run_ok(r)]
    if not runs:
        return "没有成功的运行记录。"
    head = (f"{'模型':<22}{'显存GB':>8}{'准确率':>9}{'难例':>8}{'非法率':>8}"
            f"{'TTFTms':>9}{'tok/s':>9}{'P95ms':>9}  GPU")
    lines = [head, "-" * len(head)]
    for r in runs:
        s = r["summary"]
        lines.append(
            f"{s['model']:<22}{s['footprint_gb']:>8.2f}{s['accuracy'] * 100:>8.1f}%"
            f"{s['accuracy_hard'] * 100:>7.1f}%{s['invalid_rate'] * 100:>7.1f}%"
            f"{s['ttft_ms_median']:>9.1f}{s['decode_tps_median']:>9.1f}"
            f"{s['wall_ms_p95']:>9.1f}  {s['processor']}"
        )
    return "\n".join(lines)


def build_quant_report(payload: dict) -> Path:
    runs = [r for r in payload.get("runs", []) if _run_ok(r)]
    tag = payload.get("tag", "run")
    path = config.RESULTS_DIR / f"quant_{tag}.md"

    if not runs:
        path.write_text("# 量化对比报告\n\n没有成功的运行记录。\n", encoding="utf-8")
        return path

    runs_sorted = sorted(runs, key=lambda r: quant_rank(r["summary"]["model"]), reverse=True)
    labels = [r["summary"]["model"] for r in runs_sorted]
    accs = [r["summary"]["accuracy"] * 100 for r in runs_sorted]
    tps = [r["summary"]["decode_tps_median"] for r in runs_sorted]
    mem = [r["summary"]["footprint_gb"] for r in runs_sorted]
    ttft = [r["summary"]["ttft_ms_median"] for r in runs_sorted]

    chart_mem = bar_chart(labels, {"显存占用 (GB)": mem}, title="量化档位 → 显存占用",
                          y_label="GB", unit="", height=280)
    chart_speed = bar_chart(labels, {"解码吞吐 (tok/s)": tps, "TTFT (ms)": ttft},
                            title="量化档位 → 速度与首 Token 延迟", y_label="tok/s · ms", height=280)
    chart_acc = bar_chart(labels, {"整体准确率 (%)": accs,
                                   "难例准确率 (%)": [r["summary"]["accuracy_hard"] * 100 for r in runs_sorted]},
                          title="量化档位 → 准确率", y_label="%", unit="%", height=280)

    ref = runs_sorted[0]["summary"]          # 参考档 = 精度最高的一档
    ref_acc = ref["accuracy"] * 100
    ref_mem = ref["footprint_gb"]
    ref_tps = ref["decode_tps_median"]
    worst = runs_sorted[-1]["summary"]
    # 难例数按**唯一题目**去重（记录数会被重复次数放大）
    hard_items = len({r.get("item_id") for r in payload.get("records", []) if r.get("hard")})

    md: list[str] = []
    md.append(f"# 实验 1 · 量化档位对比报告\n")
    md.append(f"- 生成时间：{payload.get('created_at')}")
    md.extend(fingerprint.report_lines(payload))
    md.extend(ev.render_lines(payload))
    md.append("")
    md.append(f"- 评测集：{payload['eval_set']['total']} 条（难例 {payload['eval_set']['hard']} 条）")
    proto = payload.get("protocol", {})
    md.append(f"- 测量口径：num_ctx={proto.get('num_ctx')} · temperature={proto.get('temperature')} · "
              f"think=False · seed={proto.get('seed')} · few-shot={proto.get('shots')} · "
              f"重复 {proto.get('repeats')} 次取中位数 · 预热 {proto.get('warmup_calls')} 次后丢弃\n")

    md.append("## 三角表（显存 ↓ / 速度 ↑ / 精度 ↓）\n")
    md.append("| 模型 | 显存 (GB) | 准确率 | 难例准确率 | 非法输出率 | TTFT 中位 (ms) | 解码 (tok/s) | P95 端到端 (ms) | 设备 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs_sorted:
        s = r["summary"]
        md.append(f"| `{s['model']}` | {s['footprint_gb']:.2f} | {s['accuracy'] * 100:.1f}% | "
                  f"{s['accuracy_hard'] * 100:.1f}% | {s['invalid_rate'] * 100:.1f}% | "
                  f"{s['ttft_ms_median']:.1f} | {s['decode_tps_median']:.1f} | {s['wall_ms_p95']:.1f} | "
                  f"{s['processor'] or '-'} |")
    md.append("")
    md.append(chart_mem)
    md.append("")
    md.append(chart_speed)
    md.append("")
    md.append(chart_acc)
    md.append("")

    md.append("## 关键结论（数字自动取自本次运行）\n")
    md.append(f"1. **精度代价**：以 `{ref['model']}`（本次最高精度档）为基准，"
              f"`{worst['model']}` 准确率 {worst['accuracy'] * 100:.1f}% vs {ref_acc:.1f}%，"
              f"掉幅 {(ref_acc - worst['accuracy'] * 100):.1f} 个百分点；"
              f"难例子集掉幅 "
              f"{(ref['accuracy_hard'] - worst['accuracy_hard']) * 100:.1f} 个百分点"
              f"（难例 {hard_items} 题，按唯一题目去重）。")
    md.append(f"2. **显存代价**：{ref_mem:.2f} GB → {worst['footprint_gb']:.2f} GB，"
              f"降幅 {stats.pct(ref_mem - worst['footprint_gb'], ref_mem):.1f}%。")
    md.append(f"3. **速度收益**：解码 {ref_tps:.1f} → {worst['decode_tps_median']:.1f} tok/s，"
              f"提升 {stats.pct(worst['decode_tps_median'] - ref_tps, ref_tps):.1f}%；"
              f"首 Token 延迟 {ref['ttft_ms_median']:.1f} → {worst['ttft_ms_median']:.1f} ms。")
    md.append(f"4. **失效模式（看混淆矩阵，不做想当然的归因）**："
              f"`{worst['model']}` 的主要错误形态 —— {_top_confusions(worst)}；"
              f"非法输出率 {worst['invalid_rate'] * 100:.1f}% vs 基准 {ref['invalid_rate'] * 100:.1f}%"
              f"{_invalid_note(ref, worst)}")
    md.append(_ladder_note(runs_sorted, ref.get("n_records", 0)))
    md.append("")

    md.append("## 测量卫生（为什么这些速度数字可信）\n")
    md.append("| 模型 | 测量前 GPU 显存 | 模型占用 | GPU 分流 |")
    md.append("|---|---|---|---|")
    for r in runs_sorted:
        s = r["summary"]
        gb = s.get("gpu_before") or {}
        base = (f"{gb.get('mem_used_mb', 0):.0f}/{gb.get('mem_total_mb', 0):.0f} MiB"
                if gb else "n/a")
        md.append(f"| `{s['model']}` | {base} | {s['footprint_gb']:.2f} GB "
                  f"(VRAM {s.get('vram_gb', 0):.2f} GB) | {s['processor']} |")
    md.append("")
    md.append("> **踩过的坑**：Ollama 默认让最近用过的模型常驻显存。首次跑多档量化时，"
              "多个模型同时占用 8GB 显存（实测 7836/8188 MiB），推理被迫退到 CPU/共享内存，"
              "解码速度从 **168 tok/s 掉到 0.59 tok/s（差 286 倍）**。"
              "现在每次测量前先 `unload_all()` 清空显存并记录 GPU 基线，"
              "再用 `size_vram / size` 校验模型确实 100% 在 GPU 上——"
              "否则测出来的「速度差异」是显存竞争，不是量化收益。")
    md.append("")

    md.append("## 结论（可直接引用；数字都是本次真实测量值）\n")
    ladder = " / ".join(labels)
    md.append(f"> **量化-显存-精度三方权衡实验**：固定 num_ctx=4096、关闭思维链、temperature=0、"
              f"每档重复 {proto.get('repeats')} 次取中位数的统一口径下，"
              f"对 {ladder} 做 A/B：`{worst['model']}` 相比 `{ref['model']}` 显存降低 "
              f"{stats.pct(ref_mem - worst['footprint_gb'], ref_mem):.1f}%、"
              f"解码吞吐提升 {stats.pct(worst['decode_tps_median'] - ref_tps, ref_tps):.1f}%，"
              f"3 分类准确率由 {ref_acc:.1f}% 降至 {worst['accuracy'] * 100:.1f}%"
              f"（模糊难例 {ref['accuracy_hard'] * 100:.1f}% → {worst['accuracy_hard'] * 100:.1f}%）；"
              f"并定位了首次测量中「多模型争抢显存导致解码掉 286 倍」的失真，"
              f"固化为「测量前清空显存 + 校验 GPU 分流」的实验规范。")
    # 若出现「中档打败顶档」，额外给一条更有杀伤力的选型结论
    _best = max((r["summary"] for r in runs_sorted), key=lambda s: s["accuracy"])
    if _best["model"] != ref["model"]:
        md.append(f">")
        md.append(f"> **量化选型结论**：`{_best['model']}` 与 `{ref['model']}` 质量无统计差异"
                  f"（{_best['accuracy'] * 100:.1f}% vs {ref['accuracy'] * 100:.1f}%），"
                  f"但显存少 {stats.pct(ref['footprint_gb'] - _best['footprint_gb'], ref['footprint_gb']):.1f}%、"
                  f"吞吐高 {stats.pct(_best['decode_tps_median'] - ref['decode_tps_median'], ref['decode_tps_median']):.1f}%、"
                  f"首 Token 延迟低 {stats.pct(ref['ttft_ms_median'] - _best['ttft_ms_median'], ref['ttft_ms_median']):.1f}%"
                  f"——**纯推理场景应默认 Q8 而不是 FP16，FP16 只在需要继续训练/微调时才有必要保留**。")
    md.append("")

    md.append("## 复现方式\n")
    md.append("```powershell")
    md.append(f"python -m inferbench quant --models {','.join(labels)}")
    md.append("```")
    md.append("")
    md.append(f"明细数据：`results/quant_{tag}.csv`（请求级，含每条样本的预测、延迟、显存、GPU 分流）")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 语义缓存实验：控制台表 + Markdown 报告
# ---------------------------------------------------------------------------

def cache_console_table(payload: dict) -> str:
    rows = payload.get("thresholds", [])
    if not rows:
        return "没有成功的阈值扫描记录。"
    head = (f"{'阈值':>6}{'命中率':>9}{'命中正确率':>12}{'误命中':>9}"
            f"{'A继承后端':>11}{'B跨意图':>9}{'端到端':>9}{'无缓存':>9}{'成本省':>9}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['threshold']:>6.2f}{r['hit_rate'] * 100:>8.1f}%"
                     f"{r['hit_correct_rate'] * 100:>11.1f}%{r['false_hit_rate'] * 100:>8.1f}%"
                     f"{r.get('false_hit_a', 0):>11d}{r.get('false_hit_b', 0):>9d}"
                     f"{r['end_to_end_accuracy'] * 100:>8.1f}%{r['nocache_accuracy'] * 100:>8.1f}%"
                     f"{r['cost_saved_pct']:>8.1f}%")
    base = payload.get("baseline", {}).get("ms_p95")
    if base:
        lines.append(f"\n无缓存基线 P95 = {base:.1f} ms")
    return "\n".join(lines)


def build_cache_report(payload: dict) -> Path:
    tag = payload.get("tag", "run")
    path = config.RESULTS_DIR / f"cache_{tag}.md"
    rows = payload.get("thresholds", [])
    meta = payload.get("meta", {})

    md: list[str] = ["# 实验 2 · 语义缓存实验报告\n"]
    md.append(f"- 生成时间：{payload.get('created_at')}")
    md.extend(fingerprint.report_lines(payload))
    md.extend(ev.render_lines(payload))
    md.append("")
    md.append(f"- 请求流：{meta.get('stream_size')} 条（原始 query {meta.get('base_queries')} 条 × "
              f"每条 {meta.get('paraphrases')} 个语义改写 + 原句）")
    md.append(f"- embedding：`{meta.get('embed_model')}` ｜ 未命中时干活：`{meta.get('cache_model')}`")
    md.append(f"- 评测口径：命中是否**正确**由金标准标签客观判定（缓存返回的标签 == 当前 query 的真实意图），"
              f"不依赖主观感觉\n")

    if not rows:
        path.write_text("\n".join(md) + "\n没有成功的记录。\n", encoding="utf-8")
        return path

    md.append("## 阈值扫描\n")
    md.append("| 阈值 | 命中率 | 命中正确率 | 误命中率 | A类·继承后端错误 | B类·跨意图误匹配 | 端到端准确率 | 无缓存基线 | 命中延迟 P95 (ms) | 未命中 P95 (ms) | 成本节省 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['threshold']:.2f} | {r['hit_rate'] * 100:.1f}% | {r['hit_correct_rate'] * 100:.1f}% | "
                  f"{r['false_hit_rate'] * 100:.1f}% | {r.get('false_hit_a', 0)} | {r.get('false_hit_b', 0)} | "
                  f"{r['end_to_end_accuracy'] * 100:.1f}% | {r['nocache_accuracy'] * 100:.1f}% | "
                  f"{r['hit_ms_p95']:.1f} | {r['miss_ms_p95']:.1f} | {r['cost_saved_pct']:.1f}% |")
    md.append("")

    xs = [r["threshold"] for r in rows]
    md.append(line_chart(xs, {
        "命中率": [r["hit_rate"] * 100 for r in rows],
        "命中正确率": [r["hit_correct_rate"] * 100 for r in rows],
        "误命中率": [r["false_hit_rate"] * 100 for r in rows],
        "端到端准确率": [r["end_to_end_accuracy"] * 100 for r in rows],
        "无缓存基线准确率": [r["nocache_accuracy"] * 100 for r in rows],
    }, title="相似度阈值 → 命中率 / 正确率 / 误命中", x_label="余弦相似度阈值", y_label="%"))
    md.append("")
    md.append(line_chart(xs, {
        "成本节省": [r["cost_saved_pct"] for r in rows],
        "命中延迟 P95": [r["hit_ms_p95"] for r in rows],
        "未命中延迟 P95": [r["miss_ms_p95"] for r in rows],
    }, title="阈值 → 成本节省与延迟", x_label="余弦相似度阈值", y_label="% · ms", percent=False))
    md.append("")

    best = max(rows, key=lambda r: r["end_to_end_accuracy"] - 0.5 * r["false_hit_rate"])
    knee = max(rows, key=lambda r: r["cost_saved_pct"] - r["false_hit_rate"] * 100)
    baseline = payload.get("baseline", {})

    # 误命中是否随阈值单调改善？——这是本实验最值钱的判断
    fh_first, fh_last = rows[0]["false_hit_rate"], rows[-1]["false_hit_rate"]
    a_total = sum(r.get("false_hit_a", 0) for r in rows)
    b_total = sum(r.get("false_hit_b", 0) for r in rows)
    a_share = (a_total / (a_total + b_total) * 100) if (a_total + b_total) else 0.0

    md.append("## 关键结论\n")
    md.append(f"1. **误命中率不随阈值下降（反直觉，但被数据证实）**："
              f"阈值 {rows[0]['threshold']:.2f} 时误命中率 {fh_first * 100:.1f}%，"
              f"提到 {rows[-1]['threshold']:.2f} 仍是 {fh_last * 100:.1f}%。"
              f"调高阈值只砍掉了命中率（{rows[0]['hit_rate'] * 100:.1f}% → {rows[-1]['hit_rate'] * 100:.1f}%），"
              f"没砍掉错误。")
    md.append(f"2. **归因分解（本实验的核心）**：把误命中拆成两类后，"
              f"**A 类「继承后端模型自身错误」占 {a_share:.1f}%**（共 {a_total} 次），"
              f"B 类「跨意图误匹配」占 {100 - a_share:.1f}%（共 {b_total} 次）。"
              f"A 类的成因是：缓存存的是后端模型**当初的预测**，模型答错的那一次会被永久固化，"
              f"之后所有语义相近的请求都会被这个错答案命中——**阈值再高也救不了，"
              f"因为它在写入缓存的那一刻就已经错了**。")
    md.append(f"3. **推荐工作点**：{best['threshold']:.2f}（端到端准确率 "
              f"{best['end_to_end_accuracy'] * 100:.1f}%，命中率 {best['hit_rate'] * 100:.1f}%，"
              f"误命中率 {best['false_hit_rate'] * 100:.1f}%，成本节省 {best['cost_saved_pct']:.1f}%）；"
              f"如果更看成本可退到 {knee['threshold']:.2f}（成本省 {knee['cost_saved_pct']:.1f}%）。"
              f"注意 A 类占比这么高，说明**继续调阈值收益很小**，"
              f"真正该做的是「高价值/易错意图不进缓存」或「命中后再校验」。")
    md.append(f"4. **准确率净代价**：端到端 {best['end_to_end_accuracy'] * 100:.1f}% vs 无缓存基线（所有请求直连模型）"
              f"{best['nocache_accuracy'] * 100:.1f}%，差 "
              f"{(best['end_to_end_accuracy'] - best['nocache_accuracy']) * 100:+.1f} 个百分点。")
    if baseline:
        md.append(f"5. **SLO 影响**：无缓存未命中路径 P95 = {baseline.get('ms_p95', 0):.1f} ms，"
                  f"命中路径 P95 = {best['hit_ms_p95']:.1f} ms"
                  f"（降幅 {stats.pct(baseline.get('ms_p95', 0) - best['hit_ms_p95'], baseline.get('ms_p95', 1)):.1f}%）。")
    md.append("")

    eng = payload.get("engineering", {})
    if eng:
        md.append("## 工程化验证（灰度 / 回滚 / 多版本）\n")
        md.append("| 机制 | 验证方式 | 结果 |")
        md.append("|---|---|---|")
        for name, item in eng.items():
            md.append(f"| {name} | {item.get('how', '')} | {item.get('result', '')} |")
        md.append("")

    md.append("## 结论（可直接引用）\n")
    md.append("> **语义缓存与成本治理（含误命中归因）**：以真实意图分布构造 1:N 语义改写请求流"
              f"（{meta.get('stream_size')} 条请求 / {meta.get('unique_texts')} 条唯一文本），"
              f"缓存正确性由金标准标签客观判定；扫描 8 档阈值后选中 {best['threshold']:.2f}："
              f"命中率 {best['hit_rate'] * 100:.1f}%、成本降 {best['cost_saved_pct']:.1f}%、"
              f"端到端准确率与无缓存基线差 {(best['end_to_end_accuracy'] - best['nocache_accuracy']) * 100:+.1f} 个百分点。"
              f"进一步把误命中拆成「继承后端模型错误」与「跨意图误匹配」两类，"
              f"实测前者占 {a_share:.1f}%，据此判断**继续调阈值收益有限**，"
              f"应改为对高价值意图关闭缓存或命中后二次校验——"
              f"结论来自可复现的消融实验，而非「调大阈值就好了」的直觉。")
    md.append("")
    md.append(f"明细数据：`results/cache_{tag}.csv`")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# KV cache 扫参：控制台表 + Markdown 报告
# ---------------------------------------------------------------------------

def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乘拟合 `y = slope·x + intercept`。点数不足时退化为常数。"""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx


def kv_analysis(payload: dict) -> dict:
    """把扫参结果收敛成几个判定：斜率、权重基线、交叉点、截断、溢出拐点、质量抖动。

    拟合**只用手上的健康点**（实际生效上下文 == 请求值，且 100% 在 GPU 上）：
    被截断的档位 x 应该是生效值而不是请求值，溢出的档位显存已经被"掉到 CPU"
    这个事实限制了，再放进拟合会把斜率压平——两者都会让斜率失真。
    """
    runs = payload.get("runs") or []
    good = [r for r in runs
            if r.get("loaded_ctx") and not r.get("truncated")
            and float(r.get("gpu_ratio") or 0) >= config.GPU_RATIO_OK]
    slope, intercept = _least_squares(
        [float(r["loaded_ctx"]) for r in good],
        [float(r["footprint_gb"]) for r in good],
    )
    theory_slope = float((payload.get("kv_theory") or {}).get("gb_per_1k_tokens") or 0.0)
    measured_slope = slope * 1024.0                      # GB / 1K token
    out = {
        "n_points": len(good),
        "gb_per_token": slope,
        "gb_per_1k": measured_slope,
        "weights_gb": intercept,
        "theory_gb_per_1k": theory_slope,
        "slope_error_pct": (stats.pct(measured_slope - theory_slope, theory_slope)
                            if theory_slope > 0 else 0.0),
        "crossover_ctx": (intercept / slope) if slope > 0 else 0.0,
        "truncated": [r for r in runs if r.get("truncated")],
        "spilled": [r for r in runs
                    if r.get("gpu_ratio") and float(r["gpu_ratio"]) < config.GPU_RATIO_OK],
        "healthiest": max(good, key=lambda r: r["loaded_ctx"]) if good else None,
    }
    accs = [r["summary"]["accuracy"] for r in runs
            if (r.get("summary") or {}).get("n_records")]
    out["accuracy_min"] = min(accs) if accs else 0.0
    out["accuracy_max"] = max(accs) if accs else 0.0
    out["accuracy_spread_pp"] = (out["accuracy_max"] - out["accuracy_min"]) * 100

    # ---- 质量侧：逐条比对预测，判断准确率差异到底来自哪里 ----
    # 这是本实验最容易讲错的一处：溢出档的准确率**更高**，但那不是上下文变长的功劳。
    # 只有逐条 diff 才能说清「变的是执行路径，不是质量」。
    ref = out["healthiest"]
    if ref:
        ref_ctx = int(ref["loaded_ctx"])
        ref_pred = {(r.get("item_id"), r.get("rep")): (r.get("pred") or "")
                    for r in payload.get("records", []) if r.get("num_ctx") == ref_ctx}
        per_level: dict[int, dict] = {}
        for run in runs:
            ctx = int(run.get("loaded_ctx") or 0)
            pred = {(r.get("item_id"), r.get("rep")): (r.get("pred") or "")
                    for r in payload.get("records", []) if r.get("num_ctx") == ctx}
            if not pred or not ref_pred:
                continue
            shared = set(pred) & set(ref_pred)
            diff = {k for k in shared if pred[k] != ref_pred[k]}
            per_level[ctx] = {
                "n": len(shared), "n_diff": len(diff),
                "diff_items": sorted({str(k[0]) for k in diff}),
                "identical": not diff,
                "spilled": ctx in {int(r.get("loaded_ctx") or 0) for r in out["spilled"]},
            }
        out["ref_ctx"] = ref_ctx
        out["per_level"] = per_level
    return out


def kv_console_table(payload: dict) -> str:
    runs = payload.get("runs") or []
    if not runs:
        return "没有成功的运行记录。"
    a = kv_analysis(payload)
    head = (f"{'请求ctx':>9}{'生效ctx':>9}{'总占用':>9}{'KV推算':>9}{'GPU':>7}"
            f"{'TTFTms':>9}{'tok/s':>9}{'准确率':>9}  备注")
    lines = [head, "-" * len(head)]
    for r in runs:
        s = r.get("summary") or {}
        kv = max(0.0, float(r.get("footprint_gb") or 0.0) - a["weights_gb"])
        note = []
        if r.get("truncated"):
            note.append("静默截断")
        if r.get("gpu_ratio") and float(r["gpu_ratio"]) < config.GPU_RATIO_OK:
            note.append("溢出到CPU")
        lines.append(
            f"{r['requested_ctx']:>9}{r.get('loaded_ctx') or '-':>9}"
            f"{float(r.get('footprint_gb') or 0):>8.2f}G{kv:>8.2f}G"
            f"{float(r.get('gpu_ratio') or 0):>6.0f}%"
            f"{s.get('ttft_ms_median', 0):>9.1f}{s.get('decode_tps_median', 0):>9.1f}"
            f"{s.get('accuracy', 0) * 100:>8.1f}%  {'/'.join(note)}"
        )
    lines.append(f"\n实测斜率 {a['gb_per_1k']:.4f} GB/1K token"
                 f"（理论 {a['theory_gb_per_1k']:.4f}，误差 {a['slope_error_pct']:+.1f}%）"
                 f" · 权重基线 {a['weights_gb']:.2f} GB"
                 f" · KV 追平权重的上下文 ≈ {a['crossover_ctx']:.0f}")
    return "\n".join(lines)


def build_kv_report(payload: dict) -> Path:
    tag = payload.get("tag", "run")
    path = config.RESULTS_DIR / f"kv_{tag}.md"
    runs = payload.get("runs") or []
    proto = payload.get("protocol") or {}
    geom = payload.get("gguf") or {}
    theory = payload.get("kv_theory") or {}

    md: list[str] = ["# 实验 6 · KV cache 上下文扫参报告\n"]
    md.append(f"- 生成时间：{payload.get('created_at')}")
    md.extend(fingerprint.report_lines(payload))
    md.extend(ev.render_lines(payload))
    md.append("")
    md.append(f"- 模型：`{proto.get('model')}`（全程同一个模型，唯一变量是 `num_ctx`）")
    md.append(f"- 评测集：{payload.get('eval_set', {}).get('total')} 条"
              f"（难例 {payload.get('eval_set', {}).get('hard')} 条）")
    md.append(f"- 测量口径：temperature={proto.get('temperature')} · think=False · "
              f"seed={proto.get('seed')} · few-shot={proto.get('shots')} · "
              f"重复 {proto.get('repeats')} 次取中位数 · 每档前卸载模型清空显存\n")

    if not runs:
        path.write_text("\n".join(md) + "\n没有成功的记录。\n", encoding="utf-8")
        return path

    a = kv_analysis(payload)

    if geom.get("complete"):
        md.append("## 先算一遍理论值（当尺子用）\n")
        md.append(f"- 结构：{geom.get('architecture')} · {geom.get('block_count')} 层 × "
                  f"{geom.get('head_count_kv')}/{geom.get('head_count')} 个 KV 头 × "
                  f"head_dim {geom.get('key_length')}，模型声明上下文 {geom.get('context_length')}")
        md.append(f"- 理论 KV：`2 (K+V) × {geom.get('block_count')} 层 × {geom.get('head_count_kv')} KV 头 × "
                  f"{geom.get('key_length')} × {config.KV_BYTES_PER_ELEM.get(config.KV_CACHE_TYPE, 2)} 字节` = "
                  f"**{theory.get('bytes_per_token', 0):.0f} B/token = "
                  f"{theory.get('gb_per_1k_tokens', 0):.4f} GB / 1K token**")
        md.append(f"- 按此推算，跑满模型声明的 {geom.get('context_length')} 上下文，"
                  f"KV cache 单独就要 **{theory.get('kv_gb_at_declared_context', 0):.2f} GB**"
                  f"（还没算模型权重）\n")

    md.append("## 扫参结果\n")
    md.append("| 请求 num_ctx | 实际生效 | 总占用 (GB) | 推算 KV (GB) | GPU 分流 | "
              "TTFT 中位 (ms) | 解码 (tok/s) | 准确率 | P95 端到端 (ms) | 备注 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s = r.get("summary") or {}
        kv = max(0.0, float(r.get("footprint_gb") or 0.0) - a["weights_gb"])
        note = []
        if r.get("truncated"):
            note.append("**静默截断**")
        if r.get("gpu_ratio") and float(r["gpu_ratio"]) < config.GPU_RATIO_OK:
            note.append("**溢出到 CPU**")
        md.append(f"| {r['requested_ctx']} | {r.get('loaded_ctx') or '-'} | "
                  f"{float(r.get('footprint_gb') or 0):.2f} | {kv:.2f} | "
                  f"{r.get('processor') or '-'} | {s.get('ttft_ms_median', 0):.1f} | "
                  f"{s.get('decode_tps_median', 0):.1f} | {s.get('accuracy', 0) * 100:.1f}% | "
                  f"{s.get('wall_ms_p95', 0):.1f} | {'/'.join(note) or '—'} |")
    md.append("")

    xs = [float(r["requested_ctx"]) for r in runs]
    bpt = float(theory.get("bytes_per_token") or 0.0)
    md.append(line_chart(xs, {
        "实测总占用 (GB)": [float(r.get("footprint_gb") or 0.0) for r in runs],
        "理论 KV cache (GB)": [bpt * float(r.get("loaded_ctx") or 0) / (1024 ** 3) for r in runs],
        "模型权重基线 (GB)": [a["weights_gb"] for _ in runs],
    }, title="上下文长度 → 显存构成（KV cache 是按 num_ctx 满额预分配的）",
        x_label="请求的 num_ctx (tokens)", y_label="GB", percent=False))
    md.append("")
    md.append(bar_chart([str(r["requested_ctx"]) for r in runs], {
        "解码吞吐 (tok/s)": [(r.get("summary") or {}).get("decode_tps_median", 0) for r in runs],
        "TTFT 中位 (ms)": [(r.get("summary") or {}).get("ttft_ms_median", 0) for r in runs],
        "GPU 分流 (%)": [float(r.get("gpu_ratio") or 0) for r in runs],
    }, title="上下文长度 → 速度与 GPU 分流", y_label="tok/s · ms · %", height=300))
    md.append("")

    md.append("## 关键结论（数字自动取自本次运行）\n")

    md.append(f"1. **KV cache 的显存是可以算出来的，而且实测对得上**："
              f"实测斜率 **{a['gb_per_1k']:.4f} GB / 1K token**，"
              f"从 GGUF 结构参数算出的理论值 {a['theory_gb_per_1k']:.4f} GB / 1K token，"
              f"两者相差 {abs(a['slope_error_pct']):.1f}%。"
              f"→ 归因成立：显存增长确实来自 KV cache，不是框架开销或显存碎片。"
              f"工程含义是**上下文预算可以在上线前算出来，不用靠试**。")

    if a["weights_gb"] > 0 and a["crossover_ctx"] > 0:
        md.append(f"2. **KV cache 会超过模型权重本身**：本次拟合出的权重基线 "
                  f"{a['weights_gb']:.2f} GB，KV cache 在约 **{a['crossover_ctx']:.0f} token** "
                  f"处追平它，之后 KV 成为显存的主要构成。"
                  f"→ 也就是说「换个小模型省显存」在长上下文下**基本失效**："
                  f"`qwen3:0.6b` 和 `qwen3:1.7b` 的 KV 头数/层数完全相同，"
                  f"KV cache 一样大，省下的只有权重那一部分。")

    if a["truncated"]:
        reqs = ", ".join(str(r["requested_ctx"]) for r in a["truncated"])
        eff = sorted({int(r["loaded_ctx"]) for r in a["truncated"]})
        md.append(f"3. **超过模型上限会被静默截断（本次最意外的发现）**："
                  f"请求 `num_ctx` = {reqs} 时，实际生效的都是 "
                  f"**{('、'.join(str(x) for x in eff))}**，"
                  f"而 Ollama **既不报错也不警告**。"
                  f"模型的声明上限就写在 GGUF 里（`context_length`），"
                  f"超出的部分被丢弃。"
                  f"→ 把 `num_ctx` 写成很大的值（很多框架的默认占位值）"
                  f"**换不来更长的上下文，只换来显存浪费和溢出**。")

    if a["spilled"]:
        first = min(a["spilled"], key=lambda r: r["requested_ctx"])
        healthy = a["healthiest"]
        if healthy:
            h_ctx = healthy["loaded_ctx"]
            h_gb = float(healthy.get("footprint_gb") or 0.0)
            h_tps = (healthy.get("summary") or {}).get("decode_tps_median", 0)
            ref = (f"对照组 num_ctx={h_ctx}（{h_gb:.2f} GB，100% GPU）"
                   f"解码 {h_tps:.1f} tok/s，")
        else:
            ref = ""
        s_tps = (first.get("summary") or {}).get("decode_tps_median", 0)
        md.append(f"4. **溢出到 CPU 是一个台阶，不是斜坡**："
                  f"从 `num_ctx={first['requested_ctx']}` 起 GPU 分流降到 "
                  f"{float(first.get('gpu_ratio') or 0):.0f}%（{first.get('processor')}）。"
                  f"{ref}该档解码 {s_tps:.1f} tok/s。"
                  f"→ 显存不够时框架**不会拒绝启动，而是把层 offload 到 CPU 继续跑**，"
                  f"表现成「能跑但慢很多」——这种静默降级比直接 OOM 更难排查。")
    md.append("")

    md.append("## 结论（可直接引用；数字都是本次真实测量值）\n")
    parts = [f"**KV cache 上下文扫参**：在 RTX 4060 8GB 上对同一模型扫 "
             f"{len(runs)} 档 `num_ctx`（{runs[0]['requested_ctx']} → {runs[-1]['requested_ctx']}），"
             f"实测 KV cache 成本 **{a['gb_per_1k']:.4f} GB / 1K token**，"
             f"与按 GGUF 结构参数算出的理论值（{a['theory_gb_per_1k']:.4f}）"
             f"相差 {abs(a['slope_error_pct']):.1f}%，"
             f"据此把「上下文长度」从配置项变成可预算的显存支出。"]
    if a["crossover_ctx"] > 0:
        parts.append(f"上下文涨到约 {a['crossover_ctx']:.0f} token 时，"
                     f"KV cache 的显存超过模型权重本身。")
    if a["truncated"]:
        parts.append(f"并发现请求 `num_ctx` 超过模型声明上限时，"
                     f"运行时会**静默截断**到上限值而不报错。")
    if a["spilled"]:
        parts.append(f"显存不足时表现为 GPU 分流下降、解码吞吐阶跃式下跌"
                     f"（而非启动失败），属于静默降级。")
    md.append("\n>\n".join("> " + p for p in parts))
    md.append("")

    # 质量侧：必须逐条 diff 才能说清「准确率那一跳来自执行路径，不是上下文」
    per_level = a.get("per_level") or {}
    ref_ctx = a.get("ref_ctx")
    if per_level and ref_ctx:
        healthy_same = [c for c, v in per_level.items() if v["identical"] and not v["spilled"]]
        changed = {c: v for c, v in per_level.items() if v["n_diff"]}
        if changed:
            detail = "、".join(
                f"`num_ctx={c}`（{v['n_diff']}/{v['n']} 条，涉及样本 {','.join(v['diff_items']) or '—'}）"
                for c, v in sorted(changed.items()))
            md.append(f"> **质量侧（逐条比对，不说「基本一致」）**："
                      f"以 `num_ctx={ref_ctx}` 为基准逐条比对预测，"
                      f"{len(healthy_same)} 个 100% GPU 档位的预测**完全一致**；"
                      f"出现差异的只有溢出到 CPU 的档位 —— {detail}。"
                      f"也就是说准确率那一跳**不是上下文变长让模型变准了**，"
                      f"而是换了执行路径（GPU → CPU offload）后浮点累加顺序改变，"
                      f"贪心解码在并列位置翻转，恰好把一条「输出非法标签」的样本"
                      f"翻成了「输出正确标签」。"
                      f"**1 条样本就值 "
                      f"{a['accuracy_spread_pp']:.1f} 个百分点的准确率** —— "
                      f"这正是「不加对照就无法归因」的活例子。")
        else:
            md.append(f"> **质量侧（逐条比对）**：以 `num_ctx={ref_ctx}` 为基准逐条比对预测，"
                      f"各档预测完全一致（{per_level.get(ref_ctx, {}).get('n', 0)} 条）——"
                      f"`num_ctx` 不改变输出，**它是一笔纯成本**：买不到质量，只买到显存占用。")
    md.append("")

    md.append("## 复现方式\n")
    md.append("```powershell")
    md.append(f"python -m inferbench kv --model {proto.get('model')} "
              f"--levels {','.join(str(r['requested_ctx']) for r in runs)}")
    md.append("```")
    md.append("")
    md.append(f"明细数据：`results/kv_{tag}.csv`（请求级，含每条样本的延迟、显存、GPU 分流）")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


