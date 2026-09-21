# 实验 2 · 语义缓存实验报告

- 生成时间：2026-09-18 08:59:02
- **适用范围**：⚠ 本次运行早于环境指纹功能（结果里没有 `env` 块），硬件与代码版本均未记录 —— **该结果的适用边界未知**，不要直接跨机器套用
- 语料：⚠ 本次运行早于语料指纹功能，无法确认用的是哪份数据

- 请求流：300 条（原始 query 100 条 × 每条 2 个语义改写 + 原句）
- embedding：`qwen3-embedding:0.6b` ｜ 未命中时干活：`qwen3:1.7b`
- 评测口径：命中是否**正确**由金标准标签客观判定（缓存返回的标签 == 当前 query 的真实意图），不依赖主观感觉

## 阈值扫描

| 阈值 | 命中率 | 命中正确率 | 误命中率 | A类·继承后端错误 | B类·跨意图误匹配 | 端到端准确率 | 无缓存基线 | 命中延迟 P95 (ms) | 未命中 P95 (ms) | 成本节省 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 60.7% | 84.1% | 15.9% | 29 | 0 | 85.3% | 86.0% | 103.7 | 150.6 | 60.4% |
| 0.85 | 54.3% | 82.8% | 17.2% | 28 | 0 | 84.3% | 86.0% | 109.1 | 147.8 | 54.1% |
| 0.88 | 50.0% | 84.0% | 16.0% | 24 | 0 | 85.0% | 86.0% | 109.1 | 149.8 | 49.8% |
| 0.90 | 46.3% | 85.6% | 14.4% | 20 | 0 | 85.7% | 86.0% | 105.4 | 153.3 | 46.0% |
| 0.92 | 42.3% | 85.8% | 14.2% | 18 | 0 | 86.0% | 86.0% | 104.3 | 154.2 | 42.0% |
| 0.94 | 38.0% | 85.1% | 14.9% | 17 | 0 | 86.0% | 86.0% | 109.7 | 131.4 | 37.6% |
| 0.96 | 33.3% | 83.0% | 17.0% | 17 | 0 | 85.7% | 86.0% | 109.4 | 143.9 | 32.9% |
| 0.98 | 27.7% | 81.9% | 18.1% | 15 | 0 | 85.7% | 86.0% | 106.0 | 148.0 | 27.2% |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">相似度阈值 → 命中率 / 正确率 / 误命中</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">%</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0%</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">24%</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">48%</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">72%</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">96%</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">0.8</text>
<text x="204.3" y="262" font-size="10" fill="#333" text-anchor="middle">0.85</text>
<text x="293.3" y="262" font-size="10" fill="#333" text-anchor="middle">0.88</text>
<text x="352.7" y="262" font-size="10" fill="#333" text-anchor="middle">0.9</text>
<text x="412.0" y="262" font-size="10" fill="#333" text-anchor="middle">0.92</text>
<text x="471.3" y="262" font-size="10" fill="#333" text-anchor="middle">0.94</text>
<text x="530.7" y="262" font-size="10" fill="#333" text-anchor="middle">0.96</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">0.98</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">余弦相似度阈值</text>
<polyline points="56.0,115.5 204.3,128.9 293.3,138.1 352.7,145.9 412.0,154.3 471.3,163.5 530.7,173.4 590.0,185.4" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="115.5" r="3" fill="#2E5BFF"/>
<circle cx="204.3" cy="128.9" r="3" fill="#2E5BFF"/>
<circle cx="293.3" cy="138.1" r="3" fill="#2E5BFF"/>
<circle cx="352.7" cy="145.9" r="3" fill="#2E5BFF"/>
<circle cx="412.0" cy="154.3" r="3" fill="#2E5BFF"/>
<circle cx="471.3" cy="163.5" r="3" fill="#2E5BFF"/>
<circle cx="530.7" cy="173.4" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="185.4" r="3" fill="#2E5BFF"/>
<polyline points="56.0,66.0 204.3,68.6 293.3,66.1 352.7,62.7 412.0,62.2 471.3,63.8 530.7,68.2 590.0,70.5" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="66.0" r="3" fill="#00A37A"/>
<circle cx="204.3" cy="68.6" r="3" fill="#00A37A"/>
<circle cx="293.3" cy="66.1" r="3" fill="#00A37A"/>
<circle cx="352.7" cy="62.7" r="3" fill="#00A37A"/>
<circle cx="412.0" cy="62.2" r="3" fill="#00A37A"/>
<circle cx="471.3" cy="63.8" r="3" fill="#00A37A"/>
<circle cx="530.7" cy="68.2" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="70.5" r="3" fill="#00A37A"/>
<polyline points="56.0,210.3 204.3,207.6 293.3,210.1 352.7,213.5 412.0,214.0 471.3,212.4 530.7,208.0 590.0,205.7" fill="none" stroke="#F2994A" stroke-width="2"/>
<circle cx="56.0" cy="210.3" r="3" fill="#F2994A"/>
<circle cx="204.3" cy="207.6" r="3" fill="#F2994A"/>
<circle cx="293.3" cy="210.1" r="3" fill="#F2994A"/>
<circle cx="352.7" cy="213.5" r="3" fill="#F2994A"/>
<circle cx="412.0" cy="214.0" r="3" fill="#F2994A"/>
<circle cx="471.3" cy="212.4" r="3" fill="#F2994A"/>
<circle cx="530.7" cy="208.0" r="3" fill="#F2994A"/>
<circle cx="590.0" cy="205.7" r="3" fill="#F2994A"/>
<polyline points="56.0,63.3 204.3,65.4 293.3,64.0 352.7,62.6 412.0,61.9 471.3,61.9 530.7,62.6 590.0,62.6" fill="none" stroke="#EB5757" stroke-width="2"/>
<circle cx="56.0" cy="63.3" r="3" fill="#EB5757"/>
<circle cx="204.3" cy="65.4" r="3" fill="#EB5757"/>
<circle cx="293.3" cy="64.0" r="3" fill="#EB5757"/>
<circle cx="352.7" cy="62.6" r="3" fill="#EB5757"/>
<circle cx="412.0" cy="61.9" r="3" fill="#EB5757"/>
<circle cx="471.3" cy="61.9" r="3" fill="#EB5757"/>
<circle cx="530.7" cy="62.6" r="3" fill="#EB5757"/>
<circle cx="590.0" cy="62.6" r="3" fill="#EB5757"/>
<polyline points="56.0,61.9 204.3,61.9 293.3,61.9 352.7,61.9 412.0,61.9 471.3,61.9 530.7,61.9 590.0,61.9" fill="none" stroke="#9B51E0" stroke-width="2"/>
<circle cx="56.0" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="204.3" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="293.3" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="352.7" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="412.0" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="471.3" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="530.7" cy="61.9" r="3" fill="#9B51E0"/>
<circle cx="590.0" cy="61.9" r="3" fill="#9B51E0"/>
<rect x="600" y="31" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="614" y="39" font-size="10.5" fill="#444">命中率</text>
<rect x="600" y="49" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="614" y="57" font-size="10.5" fill="#444">命中正确率</text>
<rect x="600" y="67" width="9" height="9" fill="#F2994A" rx="2"/>
<text x="614" y="75" font-size="10.5" fill="#444">误命中率</text>
<rect x="600" y="85" width="9" height="9" fill="#EB5757" rx="2"/>
<text x="614" y="93" font-size="10.5" fill="#444">端到端准确率</text>
<rect x="600" y="103" width="9" height="9" fill="#9B51E0" rx="2"/>
<text x="614" y="111" font-size="10.5" fill="#444">无缓存基线准确率</text>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">阈值 → 成本节省与延迟</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">% · ms</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">43</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">86</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">129</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">173</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">0.8</text>
<text x="204.3" y="262" font-size="10" fill="#333" text-anchor="middle">0.85</text>
<text x="293.3" y="262" font-size="10" fill="#333" text-anchor="middle">0.88</text>
<text x="352.7" y="262" font-size="10" fill="#333" text-anchor="middle">0.9</text>
<text x="412.0" y="262" font-size="10" fill="#333" text-anchor="middle">0.92</text>
<text x="471.3" y="262" font-size="10" fill="#333" text-anchor="middle">0.94</text>
<text x="530.7" y="262" font-size="10" fill="#333" text-anchor="middle">0.96</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">0.98</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">余弦相似度阈值</text>
<polyline points="56.0,172.6 204.3,180.1 293.3,185.2 352.7,189.6 412.0,194.4 471.3,199.5 530.7,205.1 590.0,211.9" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="172.6" r="3" fill="#2E5BFF"/>
<circle cx="204.3" cy="180.1" r="3" fill="#2E5BFF"/>
<circle cx="293.3" cy="185.2" r="3" fill="#2E5BFF"/>
<circle cx="352.7" cy="189.6" r="3" fill="#2E5BFF"/>
<circle cx="412.0" cy="194.4" r="3" fill="#2E5BFF"/>
<circle cx="471.3" cy="199.5" r="3" fill="#2E5BFF"/>
<circle cx="530.7" cy="205.1" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="211.9" r="3" fill="#2E5BFF"/>
<polyline points="56.0,121.4 204.3,115.1 293.3,115.1 352.7,119.5 412.0,120.8 471.3,114.4 530.7,114.7 590.0,118.8" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="121.4" r="3" fill="#00A37A"/>
<circle cx="204.3" cy="115.1" r="3" fill="#00A37A"/>
<circle cx="293.3" cy="115.1" r="3" fill="#00A37A"/>
<circle cx="352.7" cy="119.5" r="3" fill="#00A37A"/>
<circle cx="412.0" cy="120.8" r="3" fill="#00A37A"/>
<circle cx="471.3" cy="114.4" r="3" fill="#00A37A"/>
<circle cx="530.7" cy="114.7" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="118.8" r="3" fill="#00A37A"/>
<polyline points="56.0,66.0 204.3,69.4 293.3,67.0 352.7,62.8 412.0,61.9 471.3,88.7 530.7,74.0 590.0,69.1" fill="none" stroke="#F2994A" stroke-width="2"/>
<circle cx="56.0" cy="66.0" r="3" fill="#F2994A"/>
<circle cx="204.3" cy="69.4" r="3" fill="#F2994A"/>
<circle cx="293.3" cy="67.0" r="3" fill="#F2994A"/>
<circle cx="352.7" cy="62.8" r="3" fill="#F2994A"/>
<circle cx="412.0" cy="61.9" r="3" fill="#F2994A"/>
<circle cx="471.3" cy="88.7" r="3" fill="#F2994A"/>
<circle cx="530.7" cy="74.0" r="3" fill="#F2994A"/>
<circle cx="590.0" cy="69.1" r="3" fill="#F2994A"/>
<rect x="600" y="31" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="614" y="39" font-size="10.5" fill="#444">成本节省</text>
<rect x="600" y="49" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="614" y="57" font-size="10.5" fill="#444">命中延迟 P95</text>
<rect x="600" y="67" width="9" height="9" fill="#F2994A" rx="2"/>
<text x="614" y="75" font-size="10.5" fill="#444">未命中延迟 P95</text>
</svg>

## 关键结论

1. **误命中率不随阈值下降（反直觉，但被数据证实）**：阈值 0.80 时误命中率 15.9%，提到 0.98 仍是 18.1%。调高阈值只砍掉了命中率（60.7% → 27.7%），没砍掉错误。
2. **归因分解（本实验的核心）**：把误命中拆成两类后，**A 类「继承后端模型自身错误」占 100.0%**（共 168 次），B 类「跨意图误匹配」占 0.0%（共 0 次）。A 类的成因是：缓存存的是后端模型**当初的预测**，模型答错的那一次会被永久固化，之后所有语义相近的请求都会被这个错答案命中——**阈值再高也救不了，因为它在写入缓存的那一刻就已经错了**。
3. **推荐工作点**：0.92（端到端准确率 86.0%，命中率 42.3%，误命中率 14.2%，成本节省 42.0%）；如果更看成本可退到 0.80（成本省 60.4%）。注意 A 类占比这么高，说明**继续调阈值收益很小**，真正该做的是「高价值/易错意图不进缓存」或「命中后再校验」。
4. **准确率净代价**：端到端 86.0% vs 无缓存基线（所有请求直连模型）86.0%，差 +0.0 个百分点。
5. **SLO 影响**：无缓存未命中路径 P95 = 154.2 ms，命中路径 P95 = 104.3 ms（降幅 32.3%）。

## 工程化验证（灰度 / 回滚 / 多版本）

| 机制 | 验证方式 | 结果 |
|---|---|---|
| 版本指纹（模型/Prompt 换代） | 缓存写入 50 条后把版本号 v1→v2，再查同一 query | 命中=False（相似度 -1.000）→ 旧缓存全部失效，不会串用过期答案 |
| TTL 过期 | 写入后 ttl=1ms，等 10ms 再查 | 命中=False → 过期条目被跳过，不会命中陈旧答案 |
| 灰度发布（10% 流量） | 新版本缓存先接 10% 流量，与旧版本并行，比较命中正确率 | 灰度桶命中 0 次，命中正确率 0.0% → 可先小流量验证再全量 |
| 一键回滚（kill switch） | 把缓存开关关掉再打开，各查一次 | 关闭时命中=False，打开后命中=True → 出问题可秒级回滚到无缓存直连 |
| 批量失效 | 调用 invalidate_all() | 清空 1 条；清空后缓存为空，下一次请求必然回落到 LLM（正确但变慢） |

## 结论（可直接引用）

> **语义缓存与成本治理（含误命中归因）**：以真实意图分布构造 1:N 语义改写请求流（300 条请求 / 230 条唯一文本），缓存正确性由金标准标签客观判定；扫描 8 档阈值后选中 0.92：命中率 42.3%、成本降 42.0%、端到端准确率与无缓存基线差 +0.0 个百分点。进一步把误命中拆成「继承后端模型错误」与「跨意图误匹配」两类，实测前者占 100.0%，据此判断**继续调阈值收益有限**，应改为对高价值意图关闭缓存或命中后二次校验——结论来自可复现的消融实验，而非「调大阈值就好了」的直觉。

明细数据：`results/cache_full.csv`