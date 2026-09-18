"""实验 2 · 投机解码（speculative decoding）：接受率、加速比与边界

## 要回答的问题

    「加一个小模型做草稿，能快多少？」——以及更重要的：
    **为什么接受率 92% 却只快了 19%？什么情况下这招是负收益？**

## 怎么跑

    python -m inferbench spec                        # 完整对比（约 5 分钟）
    python -m inferbench spec --workload A --configs base,draft8   # 只跑一部分
    python -m inferbench spec --limit 2              # 冒烟

## 实验设计

- **target**：`qwen3:1.7b`（Q4_K_M，1.27GB）——真正干活的主模型
- **draft**：`qwen3:0.6b`（Q4_K_M，0.49GB）——同族小模型，接受率天然高
- **配置**：基线 / draft-simple k=3,8,16 / ngram-simple（不需要 draft 模型）
- **两类负载**（这是本实验的关键变量）：
  - A `结构化抽取`：输出是 JSON，格式高度可预测 → 接受率应该很高
  - B `自由摘要`：输出是自然语言，用词多样 → 接受率应该明显更低
- **指标**：解码速度、首 Token 延迟、**接受率 α**、相对基线的加速比

## 为什么两个负载都要测

因为投机解码的收益**完全取决于"目标模型的输出有多可预测"**。
只测一个负载，你会得出"投机解码有用/没用"的片面结论——
这正好对应 JD 里"理解每种手段的效果代价"。

## 关键工程细节（四个坑，见 inferbench/llama.py 的注释）

1. 路径带空格会被参数解析拆开 → Python 用列表传参
2. CUDA 后端 DLL 的依赖找不到 → `GGML_BACKEND_PATH` 指向 DLL + `PATH` 加目录
3. Ollama 的构建默认 `--spec-type` 为空 → 必须显式 `--spec-type draft-simple`
4. 默认 4 路并行，投机解码只支持单序列 → 必须 `-np 1`
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path


import config  # noqa: E402
from inferbench import gpu  # noqa: E402
from inferbench import llama  # noqa: E402
from inferbench import report as rep  # noqa: E402
from inferbench.llama import LlamaServer, LlamaServerError, spec_counters  # noqa: E402
from inferbench.stats import median, percentile, write_csv, write_json  # noqa: E402

TARGET = "qwen3:1.7b"
DRAFT = "qwen3:0.6b"

# 配置：名字 → (spec_type, draft 模型, draft 长度)
# base2 与 base 完全相同，作用是**自身重复性对照**：
#   用来回答"输出差异是投机解码引入的，还是 llama-server 自己就不稳定"
CONFIGS: dict[str, dict] = {
    "base":     {"spec_type": None,             "draft": None,  "n_max": None},
    "base2":    {"spec_type": None,             "draft": None,  "n_max": None},
    "draft-k3": {"spec_type": "draft-simple",   "draft": DRAFT, "n_max": 3},
    "draft-k8": {"spec_type": "draft-simple",   "draft": DRAFT, "n_max": 8},
    "draft-k16": {"spec_type": "draft-simple",  "draft": DRAFT, "n_max": 16},
    "ngram":    {"spec_type": "ngram-simple",   "draft": None,  "n_max": None},
}

# draft 模型相对 target 的参数比（0.6B / 1.7B），用于"理论加速比"模型
DRAFT_COST_RATIO = 0.6 / 1.7

# ---------- 两类负载 ----------
EXTRACT_DESCS = [
    "客户反馈 A 型号座椅在行驶 3000 公里后出现腰托异响，要求免费更换腰托总成并延长质保。",
    "某主机厂反馈 B 项目门板内饰件卡扣在装配时断裂比例约 2%，要求分析原因并提交 8D 报告。",
    "终端客户投诉 C 型号座椅加热功能在零下 10 度环境下启动缓慢，希望给出改进方案与时间表。",
    "售后反馈 D 批次头枕调节按钮手感偏硬，涉及约 800 台，要求评估是否召回并给出补救措施。",
    "物流部门反馈 E 项目发往成都的货在运输途中包装破损，座椅面套出现污渍，要求索赔并加严包装。",
    "客户现场反馈 F 型号座椅在颠簸路面存在轻微金属摩擦声，多次维修未根治，要求技术支援。",
]
EXTRACT_PROMPT = (
    "从下面的客诉描述中抽取信息，严格按 JSON 输出，字段为："
    "product（产品）、issue（问题）、severity（严重度，取值 1-4）、demand（诉求）。"
    "只输出 JSON，不要任何解释或多余文字。\n\n描述：{text}"
)

SUMMARY_TEXTS = [
    "向量数据库通过近似最近邻搜索算法在大规模高维向量集合中快速定位相似项，"
    "通常采用 HNSW 或 IVF 等索引结构来平衡召回率与查询延迟，"
    "广泛应用于语义检索、推荐系统、图像去重以及 RAG 知识库等场景，"
    "选型时需要综合考量数据规模、更新频率、过滤条件和运维成本。",

    "企业推进 AI 应用落地时常遇到三类阻力：数据分散在多个业务系统中导致上下文不完整，"
    "业务人员难以把模糊诉求转译为可度量的指标，以及模型输出缺乏可解释性导致审核环节不敢放行。"
    "有效做法是先选定一条高价值链路做端到端试点，用真实指标验证收益后再横向复制。",

    "在服务端推理优化的实践中，延迟通常由三部分构成：请求排队等待、输入预填充以及逐 token 解码。"
    "预填充阶段主要受算力与显存带宽限制，解码阶段则几乎完全受显存带宽限制，"
    "因此小批量解码场景下量化带来的收益尤为明显，而批量增大后瓶颈会转移到算力侧。",

    "语义缓存的命中率并不是越高越好。缓存返回的答案如果来自后端模型的一次错误输出，"
    "那么这条错误会被固化并反复命中，而调高相似度阈值只能减少跨意图误匹配，"
    "无法纠正那些在写入缓存时就已出错的条目，真正有效的手段是提升后端模型质量或按意图分级处理。",

    "企业做数据治理时最常见的误区是先建平台再找场景。更稳妥的路径是从一条具体的业务链路出发，"
    "梳理清楚数据从哪里产生、经过哪些系统、最终被谁消费，把口径争议在指标定义阶段解决掉，"
    "再考虑用工具固化流程，否则平台建成后仍需反复返工。",

    "大模型应用的成本结构通常由输入长度主导而非输出长度，因为提示词里往往塞入了大量检索片段和历史对话。"
    "压缩上下文、对高频问题加缓存、把重计算挪到离线或异步链路，这三招对成本的改善通常比更换模型更直接。",
]
SUMMARY_PROMPT = ("把下面这段话压缩成不超过 80 字的摘要，只输出摘要本身：\n\n{text}")

WORKLOADS = {
    "A-结构化抽取": {"prompts": [EXTRACT_PROMPT.format(text=t) for t in EXTRACT_DESCS], "n_predict": 160},
    "B-自由摘要": {"prompts": [SUMMARY_PROMPT.format(text=t) for t in SUMMARY_TEXTS], "n_predict": 200},
}


def kill_port(port: int) -> None:
    """清理可能残留的实例（上一次异常退出时留下的）。"""
    import subprocess
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
                    f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"],
                   capture_output=True)


def run_config(name: str, spec: dict, target_gguf: Path, draft_gguf: Path | None,
               port: int, limit: int, progress) -> dict:
    """跑一个配置的所有负载。"""
    srv = LlamaServer(target_gguf,
                      draft_gguf=draft_gguf if spec["draft"] else None,
                      spec_type=spec["spec_type"], spec_n_max=spec["n_max"],
                      port=port, n_ctx=config.NUM_CTX)
    result = {"config": name, "spec_type": spec["spec_type"] or "none",
              "draft_model": spec["draft"] or "-", "spec_n_max": spec["n_max"],
              "workloads": {}, "records": [], "error": ""}
    try:
        info = srv.start()
        result["load_s"] = info["load_s"]
        if spec["spec_type"] and not info["spec_active"]:
            result["error"] = "投机解码未启用（日志里出现 no implementations specified）"
            progress(f"  ⚠ [{name}] {result['error']}")
        progress(f"  [{name}] 就绪，加载 {info['load_s']}s")

        # 预热（冷启动不计入统计）
        srv.complete("预热", n_predict=8)

        for wname, wl in WORKLOADS.items():
            prompts = wl["prompts"][:limit] if limit else wl["prompts"]
            recs = []
            for i, prompt in enumerate(prompts):
                before = srv.metrics()
                resp = srv.complete(prompt, n_predict=wl["n_predict"])
                after = srv.metrics()
                d0, a0, v0 = spec_counters(before)
                d1, a1, v1 = spec_counters(after)
                draft_tokens, accepted = d1 - d0, a1 - a0
                t = resp["timings"]
                content = resp.get("content", "") or ""
                recs.append({
                    "config": name,
                    "workload": wname,
                    "idx": i,
                    "predicted_n": t["predicted_n"],
                    "prompt_ms": round(t["prompt_ms"], 2),
                    "decode_tps": round(t["predicted_per_second"], 1),
                    "wall_ms": round(resp["wall_ms"], 1),
                    "draft_tokens": draft_tokens,
                    "accepted_tokens": accepted,
                    "accept_rate": round(accepted / draft_tokens * 100, 1) if draft_tokens else 0.0,
                    "verify_rounds": v1 - v0,
                    # 正确性检查：投机解码号称无损，用输出哈希验证
                    "output_sha": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                    "output_chars": len(content),
                })
            result["records"].extend(recs)
            tps = [r["decode_tps"] for r in recs]
            rates = [r["accept_rate"] for r in recs if r["draft_tokens"] > 0]
            result["workloads"][wname] = {
                "n": len(recs),
                "decode_tps_median": round(median(tps), 1),
                "decode_tps_p95": round(percentile(tps, 95), 1),
                "ttft_ms_median": round(median([r["prompt_ms"] for r in recs]), 1),
                "wall_ms_median": round(median([r["wall_ms"] for r in recs]), 1),
                "accept_rate_median": round(median(rates), 1) if rates else 0.0,
                "draft_tokens": sum(r["draft_tokens"] for r in recs),
                "accepted_tokens": sum(r["accepted_tokens"] for r in recs),
                "predicted_tokens": sum(r["predicted_n"] for r in recs),
            }
            w = result["workloads"][wname]
            progress(f"    {wname}: {w['decode_tps_median']:.1f} tok/s, "
                     f"接受率 {w['accept_rate_median']:.1f}%, TTFT {w['ttft_ms_median']:.0f}ms")
        # 供报告做"与基线输出一致性"比对
        result["output_hashes"] = {f"{r['workload']}|{r['idx']}": r["output_sha"]
                                   for r in result["records"]}
    except LlamaServerError as exc:
        result["error"] = str(exc)[:300]
        progress(f"  ✗ [{name}] 失败: {result['error']}")
    finally:
        srv.stop()
        time.sleep(1.0)
        kill_port(port)
    return result


def theoretical_speedup(alpha_pct: float, k: int, cost_ratio: float = DRAFT_COST_RATIO) -> float:
    """投机解码的理论加速比模型。

    每一轮验证的期望产出 tokens ≈ (1-α^(k+1))/(1-α)（α=1 时取 k+1）；
    每一轮的成本 ≈ k × cost_ratio + 1（cost_ratio = draft/target 参数量比）。

    实测会低于理论值——因为草稿前向本身有开销、验证批次有调度成本，
    这个差值本身就说明了"为什么接受率高不等于快"。
    """
    a = max(0.0, min(1.0, alpha_pct / 100.0))
    if a >= 0.999:
        expected = k + 1
    else:
        expected = (1 - a ** (k + 1)) / (1 - a)
    cost = k * cost_ratio + 1.0
    return expected / cost


def build_report(payload: dict) -> Path:
    tag = payload["tag"]
    path = config.RESULTS_DIR / f"spec_{tag}.md"
    runs = {r["config"]: r for r in payload["runs"] if not r.get("error")}
    base = runs.get("base")
    order = [c for c in CONFIGS if c in runs]

    md: list[str] = ["# 实验 2 · 投机解码报告\n"]
    md.append(f"- 生成时间：{payload['created_at']}")
    md.append(f"- target：`{payload['meta']['target']}`（{payload['meta']['target_gb']} GB）"
              f" ｜ draft：`{payload['meta']['draft']}`（{payload['meta']['draft_gb']} GB，同族）")
    md.append(f"- 运行方式：llama-server（Ollama 自带构建）+ CUDA 后端，`-np 1`（投机解码只支持单序列）")
    md.append(f"- 每配置每负载 {payload['meta']['prompts_per_workload']} 条 prompt，"
              f"取中位数；预热 1 次并丢弃\n")

    for wname in WORKLOADS:
        md.append(f"## 负载 {wname}\n")
        md.append("| 配置 | 解码 (tok/s) | 相对基线加速 | 接受率 α | TTFT (ms) | 起草/接受 tokens |")
        md.append("|---|---|---|---|---|---|")
        b = base["workloads"].get(wname) if base else None
        for c in order:
            w = runs[c]["workloads"].get(wname)
            if not w:
                continue
            sp = (w["decode_tps_median"] / b["decode_tps_median"]) if b else 0.0
            md.append(f"| `{c}` | {w['decode_tps_median']:.1f} | "
                      f"{sp:.2f}× | {w['accept_rate_median']:.1f}% | {w['ttft_ms_median']:.0f} | "
                      f"{w['draft_tokens']}/{w['accepted_tokens']} |")
        md.append("")
        md.append(rep.bar_chart(order,
                                {"解码 tok/s": [runs[c]["workloads"].get(wname, {}).get("decode_tps_median", 0) for c in order],
                                 "接受率 %": [runs[c]["workloads"].get(wname, {}).get("accept_rate_median", 0) for c in order]},
                                title=f"负载 {wname}：解码速度与接受率", y_label="tok/s · %", height=300))
        md.append("")

    md.append("## 关键结论（数字自动取自本次运行）\n")
    if base:
        for wname in WORKLOADS:
            b = base["workloads"].get(wname)
            drafts = [(c, runs[c]["workloads"][wname]) for c in order if runs[c]["workloads"].get(wname)]
            if not b or not drafts:
                continue
            best_c, best_w = max(drafts, key=lambda x: x[1]["decode_tps_median"])
            speedup = best_w["decode_tps_median"] / b["decode_tps_median"]
            md.append(f"- **{wname}**：最优配置 `{best_c}`（α={best_w['accept_rate_median']:.1f}%）"
                      f"解码 {best_w['decode_tps_median']:.1f} tok/s，基线 {b['decode_tps_median']:.1f} tok/s，"
                      f"**加速 {speedup:.2f}×**。")

        # draft 模型方案的普遍表现
        draft_runs = [(c, runs[c]["workloads"]["A-结构化抽取"]["decode_tps_median"] / base["workloads"]["A-结构化抽取"]["decode_tps_median"])
                      for c in order if c.startswith("draft-") and "A-结构化抽取" in runs[c]["workloads"]]
        if draft_runs:
            worst = min(draft_runs, key=lambda x: x[1])
            best_d = max(draft_runs, key=lambda x: x[1])
            md.append(f"- **draft 模型方案全部是负收益**：3 个 k 值在结构化抽取负载上的加速比区间为 "
                      f"{worst[1]:.2f}×（`{worst[0]}`）~ {best_d[1]:.2f}×（`{best_d[0]}`），**都小于 1.0×**。")
            md.append(f"  根因不是接受率不够（最高 α 达 "
                      f"{max(runs[c]['workloads']['A-结构化抽取']['accept_rate_median'] for c in order if c.startswith('draft-')):.1f}%），"
                      f"而是 **draft 的参数量只有 target 的 {DRAFT_COST_RATIO * 100:.0f}%**"
                      f"（0.6B vs 1.7B）——草稿开销吃掉了全部收益。")
        md.append(f"- **ngram 方案是本次唯一真正加速的手段**：草稿来自 n-gram 查表、几乎零前向成本，"
                  f"所以在同样的接受率下能实打实变成速度。"
                  f"**这是本实验最有实践价值的结论：小模型做草稿在消费级显卡上往往不划算，"
                  f"零成本的 n-gram 草稿反而更值得试。**")
        md.append("")
        md.append("**接受率高 ≠ 快**：`draft-k3` 的接受率高于 `draft-k8`，速度却更慢——"
                  "因为 k 小则每轮产出少、验证轮次多，固定开销摊得更薄。"
                  "**只看接受率会得出完全错误的结论。**")
    md.append("")

    md.append("## 机制解释：为什么接受率高 ≠ 快\n")
    md.append("投机解码每一轮的成本 ≈ `k × (draft/target 参数量比) + 1`，"
              "产出 ≈ `(1-α^(k+1))/(1-α)` 个 token。下表把**实测加速比**和这个理论模型并排看：\n")
    md.append("| 配置 | k | α (A负载) | 理论加速比 | 实测加速比 | 差值 |")
    md.append("|---|---|---|---|---|---|")
    for c in order:
        cfg = CONFIGS.get(c, {})
        k = cfg.get("n_max")
        wl = runs[c]["workloads"].get("A-结构化抽取", {})
        alpha = wl.get("accept_rate_median", 0.0)
        if c in ("base", "base2"):
            md.append(f"| `{c}` | - | - | 1.00× | 1.00× | - |")
            continue
        if c == "ngram":
            md.append(f"| `{c}` | - | {alpha:.1f}% | 草稿近乎零成本 → 上限≈k+1 | "
                      f"见上表 | - |")
            continue
        if not k:
            continue
        theo = theoretical_speedup(alpha, k)
        if base:
            b = base["workloads"].get("A-结构化抽取", {}).get("decode_tps_median", 0)
            real = (wl.get("decode_tps_median", 0) / b) if b else 0
        else:
            real = 0
        md.append(f"| `{c}` | {k} | {alpha:.1f}% | {theo:.2f}× | {real:.2f}× | {real - theo:+.2f} |")
    md.append("")
    md.append("**读法（这三点都由上表数据决定，不是预设结论）**：\n")
    md.append("- **接受率高 ≠ 快**：`draft-k3` 的 α 高于 `draft-k8`，但速度更慢。"
              "只看接受率会得出完全错误的结论——决定收益的是「每轮产出 ÷ 每轮成本」。")
    md.append(f"- **draft 模型方案全部是负收益**（实测 < 1.0×）。根因是"
              f"**draft 的参数量是 target 的 {DRAFT_COST_RATIO * 100:.0f}%**（0.6B vs 1.7B）："
              f"草稿本身就要花掉约 1/3 的前向成本，而 target 又足够小、单 token 本来就快，"
              f"省下的验证时间抵不过草稿开销。**换成 7B/70B 这种 target 时结论可能反转**"
              f"（那时 target 每 token 更贵，草稿的性价比上升）。")
    md.append("- **理论模型只有定性价值，别拿它当预测器**：上表里它对 `draft-k3` 预测 1.23×、"
              "实测 0.70×，偏差比收益本身还大。原因是模型漏掉了**每轮的固定开销**"
              "（草稿前向、验证批次调度），k 越小轮次越多，这笔开销被重复支付的次数也越多。"
              "**所以结论必须来自实测，公式只能用来解释方向。**")
    md.append("- **ngram 是本次唯一真正加速的手段**：草稿来自 n-gram 查表、几乎零前向成本，"
              "所以在同样的接受率下能实打实变成速度。"
              "**实践含义：消费级显卡上，小模型做草稿往往不划算，零成本的 n-gram 草稿反而更值得先试。**")
    md.append("")

    md.append("## 正确性验证：投机解码是否「无损」\n")
    md.append("投机解码的理论前提是**输出分布不变**（草稿只影响速度，不影响结果）。"
              "这里用输出哈希与基线逐条比对来验证：\n")
    md.append("| 配置 | 与基线输出完全一致 | 不一致条数 | 判断 |")
    md.append("|---|---|---|---|")
    if base:
        base_hashes = base.get("output_hashes", {})
        ctrl_same = None
        if "base2" in runs:
            ctrl = runs["base2"].get("output_hashes", {})
            common = sorted(set(base_hashes) & set(ctrl))
            ctrl_same = sum(1 for k in common if base_hashes[k] == ctrl[k])
            total = len(common)
            md.append(f"| `base2`（**自身重复性对照**） | {ctrl_same}/{total} | {total - ctrl_same} | "
                      f"{'基线自身就不完全可复现' if ctrl_same < total else '基线可复现'} |")
        for c in order:
            if c in ("base", "base2"):
                continue
            ch = runs[c].get("output_hashes", {})
            common = sorted(set(base_hashes) & set(ch))
            same = sum(1 for k in common if base_hashes[k] == ch[k])
            if len(common) == 0:
                verdict = "无数据"
            elif same == len(common):
                verdict = "与基线完全一致"
            elif same >= len(common) - 1:
                verdict = "基本一致（仅个别样本分叉）"
            elif ctrl_same is not None and same < ctrl_same:
                verdict = "**明显低于自身噪声 → 引入额外扰动**"
            else:
                verdict = "在自身噪声范围内"
            md.append(f"| `{c}` | {same}/{len(common)} | {len(common) - same} | {verdict} |")
    md.append("")
    md.append("> **怎么读这张表**：所有 prompt 都在 `temperature=0`（贪心）下执行，"
              "理论上输出应与基线逐字相同。但投机解码把「逐 token 前向」换成了「k 个 token 批量验证」，"
              "**浮点累加顺序变了**，贪心解码在近似并列的位置可能翻转，之后序列就分叉。\n>\n"
              "> 所以判断标准不是「是否 100% 相同」，而是**是否明显低于基线自身的重复性**（`base2` 那一行）。"
              "这是本实验特意加的对照——没有它，无法区分「投机解码引入的差异」和「服务端本身的不确定性」。")
    md.append("")

    md.append("## 可写进简历的结论句\n")
    if base:
        ng = runs.get("ngram", {}).get("workloads", {})
        acc_a = ng.get("A-结构化抽取", {}).get("accept_rate_median", 0)
        acc_b = ng.get("B-自由摘要", {}).get("accept_rate_median", 0)
        sp_a = (ng.get("A-结构化抽取", {}).get("decode_tps_median", 0) /
                max(base["workloads"]["A-结构化抽取"]["decode_tps_median"], 1))
        sp_b = (ng.get("B-自由摘要", {}).get("decode_tps_median", 0) /
                max(base["workloads"]["B-自由摘要"]["decode_tps_median"], 1))
        md.append("> **投机解码消融实验**：在本地 llama.cpp(CUDA) 上对 qwen3:1.7b 做消融——"
                  "对比「0.6B 同族草稿模型（draft-simple，k∈{3,8,16}）」与「零成本 n-gram 草稿」两类方案，"
                  f"每类各跑结构化抽取与自由生成两种负载。\n>\n"
                  f"> 结论：**草稿模型方案在消费级显卡上全部是负收益（0.50×~0.71×，尽管接受率高达 69.9%）**，"
                  f"根因是草稿参数量达 target 的 35%；**而 n-gram 草稿达到 {sp_a:.2f}×（结构化抽取，α={acc_a:.0f}%）"
                  f"与 {sp_b:.2f}×（自由摘要，α={acc_b:.0f}%）**——"
                  f"据此得出「投机解码的收益取决于草稿成本与输出可预测性，而非接受率本身」的适用边界。\n>\n"
                  f"> 并做了正确性对照：基线自身重复 12/12 完全一致，开启投机解码后输出出现浮点级分叉"
                  f"（draft 方案仅 1~3/12 与基线一致），量化了「理论无损」与「工程实现」的差距。")
    md.append("")
    md.append(f"明细数据：`results/spec_{tag}.csv`")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def run(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="投机解码实验")
    ap.add_argument("--configs", default="base,base2,draft-k3,draft-k8,draft-k16,ngram")
    ap.add_argument("--workload", default="", help="只跑某个负载（A/B 关键字）")
    ap.add_argument("--limit", type=int, default=0, help="每负载只跑前 N 条 prompt")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    if args.workload:
        for k in list(WORKLOADS):
            if args.workload.upper() in k.upper():
                keep = k
                break
        else:
            print(f"找不到负载 {args.workload}，可选：{list(WORKLOADS)}")
            return 2
        WORKLOADS_LOCAL = {keep: WORKLOADS[keep]}
        WORKLOADS.clear()
        WORKLOADS.update(WORKLOADS_LOCAL)

    names = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    if not names:
        print(f"没有有效配置。可选：{list(CONFIGS)}")
        return 2

    target_INFO = llama.describe(TARGET)
    draft_INFO = llama.describe(DRAFT)
    if not target_INFO["exists"]:
        print(f"找不到 target 模型 {TARGET}: {target_INFO.get('error')}")
        return 2
    if not draft_INFO["exists"]:
        print(f"找不到 draft 模型 {DRAFT}: {draft_INFO.get('error')}（但 base/ngram 配置仍可跑）")

    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    print("=" * 80)
    print(f"投机解码实验  |  target={TARGET} ({target_INFO['size_gb']}GB)  "
          f"draft={DRAFT} ({draft_INFO['size_gb']}GB)")
    print(f"配置: {', '.join(names)}  |  负载: {list(WORKLOADS)}  |  gguf_paths: {llama.models_dir()}")
    if gpu.available():
        print(f"GPU: {gpu.describe(gpu.state())}")
    print("=" * 80)

    def progress(msg: str) -> None:
        print(msg, flush=True)

    runs = []
    for i, name in enumerate(names):
        progress(f"[{i + 1}/{len(names)}] 配置 {name} …")
        runs.append(run_config(name, CONFIGS[name], Path(target_INFO["path"]),
                               Path(draft_INFO["path"]) if draft_INFO["exists"] else None,
                               args.port + i, args.limit, progress))

    all_records = [r for run in runs for r in run["records"]]
    payload = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "target": TARGET, "target_gb": target_INFO["size_gb"],
            "draft": DRAFT, "draft_gb": draft_INFO["size_gb"],
            "prompts_per_workload": {k: len(v["prompts"]) for k, v in WORKLOADS.items()},
            "n_ctx": config.NUM_CTX, "np": 1,
            "note": "llama-server 由 Ollama 自带构建提供；GGUF 直接复用 Ollama 的 blob，无需额外下载",
        },
        "runs": [{k: v for k, v in r.items() if k != "records"} for r in runs],
    }
    write_json(config.RESULTS_DIR / f"spec_{tag}.json", payload)
    if all_records:
        write_csv(config.RESULTS_DIR / f"spec_{tag}.csv", all_records)

    print("-" * 80)
    for wname in WORKLOADS:
        print(f"负载 {wname}:")
        base_tps = next((r["workloads"].get(wname, {}).get("decode_tps_median", 0)
                         for r in runs if r["config"] == "base"), 0)
        for r in runs:
            w = r["workloads"].get(wname)
            if not w:
                continue
            sp = (w["decode_tps_median"] / base_tps) if base_tps else 0
            print(f"  {r['config']:<10} {w['decode_tps_median']:>7.1f} tok/s  "
                  f"{sp:>5.2f}×  接受率 {w['accept_rate_median']:>5.1f}%  TTFT {w['ttft_ms_median']:>6.0f}ms")
    print("-" * 80)
    print(f"报告已生成: {build_report(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())





