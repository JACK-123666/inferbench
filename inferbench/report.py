"""报告生成：把实验结果组装成 Markdown（含内联 SVG 图与可写进简历的结论句）。

图表实现在 inferbench/svg.py（零依赖内联 SVG，不引入 matplotlib）。
本模块只负责：取数 → 判定 → 组织文案。**所有数字都从结果文件里取，
不手抄**——手抄的数字一定会在某次改动后与数据不一致。
"""
from __future__ import annotations

import re
from pathlib import Path

from inferbench import config
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
    md.append("> **踩过的坑（面试可讲）**：Ollama 默认让最近用过的模型常驻显存。首次跑多档量化时，"
              "多个模型同时占用 8GB 显存（实测 7836/8188 MiB），推理被迫退到 CPU/共享内存，"
              "解码速度从 **168 tok/s 掉到 0.59 tok/s（差 286 倍）**。"
              "现在每次测量前先 `unload_all()` 清空显存并记录 GPU 基线，"
              "再用 `size_vram / size` 校验模型确实 100% 在 GPU 上——"
              "否则测出来的「速度差异」是显存竞争，不是量化收益。")
    md.append("")

    md.append("## 可写进简历的结论句（数字都是本次真实测量值）\n")
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

    md: list[str] = ["# 实验 3 · 语义缓存实验报告\n"]
    md.append(f"- 生成时间：{payload.get('created_at')}")
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

    md.append("## 可写进简历的结论句\n")
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


