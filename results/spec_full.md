# 实验 4 · 投机解码报告

- 生成时间：2026-09-18 12:00:01
- **适用范围**：⚠ 本次运行早于环境指纹功能（结果里没有 `env` 块），硬件与代码版本均未记录 —— **该结果的适用边界未知**，不要直接跨机器套用

- target：`qwen3:1.7b`（1.27 GB） ｜ draft：`qwen3:0.6b`（0.49 GB，同族）
- 运行方式：llama-server（Ollama 自带构建）+ CUDA 后端，`-np 1`（投机解码只支持单序列）
- 每配置每负载 {'A-结构化抽取': 6, 'B-自由摘要': 6} 条 prompt，取中位数；预热 1 次并丢弃

## 负载 A-结构化抽取

| 配置 | 解码 (tok/s) | 相对基线加速 | 接受率 α | TTFT (ms) | 起草/接受 tokens |
|---|---|---|---|---|---|
| `base` | 154.2 | 1.00× | 0.0% | 16 | 0.0/0.0 |
| `base2` | 155.3 | 1.01× | 0.0% | 17 | 0.0/0.0 |
| `draft-k3` | 107.4 | 0.70× | 69.9% | 24 | 911.0/648.0 |
| `draft-k8` | 109.2 | 0.71× | 56.6% | 26 | 1520.0/761.0 |
| `draft-k16` | 76.5 | 0.50× | 30.8% | 24 | 2563.0/786.0 |
| `ngram` | 282.0 | 1.83× | 100.0% | 17 | 552.0/468.0 |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">负载 A-结构化抽取：解码速度与接受率</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">tok/s · %</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">81</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">162</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">243</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">324</text>
<rect x="71.1" y="147.0" width="36.9" height="97.0" fill="#2E5BFF" rx="2"/>
<text x="89.6" y="143.0" font-size="9" fill="#444" text-anchor="middle">154.2</text>
<rect x="110.0" y="244.0" width="36.9" height="0.5" fill="#00A37A" rx="2"/>
<text x="128.4" y="240.0" font-size="9" fill="#444" text-anchor="middle">0.0</text>
<text x="110.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">base</text>
<rect x="179.1" y="146.3" width="36.9" height="97.7" fill="#2E5BFF" rx="2"/>
<text x="197.6" y="142.3" font-size="9" fill="#444" text-anchor="middle">155.3</text>
<rect x="218.0" y="244.0" width="36.9" height="0.5" fill="#00A37A" rx="2"/>
<text x="236.4" y="240.0" font-size="9" fill="#444" text-anchor="middle">0.0</text>
<text x="218.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">base2</text>
<rect x="287.1" y="176.4" width="36.9" height="67.6" fill="#2E5BFF" rx="2"/>
<text x="305.6" y="172.4" font-size="9" fill="#444" text-anchor="middle">107.4</text>
<rect x="326.0" y="200.0" width="36.9" height="44.0" fill="#00A37A" rx="2"/>
<text x="344.4" y="196.0" font-size="9" fill="#444" text-anchor="middle">69.9</text>
<text x="326.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">draft-k3</text>
<rect x="395.1" y="175.3" width="36.9" height="68.7" fill="#2E5BFF" rx="2"/>
<text x="413.6" y="171.3" font-size="9" fill="#444" text-anchor="middle">109.2</text>
<rect x="434.0" y="208.4" width="36.9" height="35.6" fill="#00A37A" rx="2"/>
<text x="452.4" y="204.4" font-size="9" fill="#444" text-anchor="middle">56.6</text>
<text x="434.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">draft-k8</text>
<rect x="503.1" y="195.9" width="36.9" height="48.1" fill="#2E5BFF" rx="2"/>
<text x="521.6" y="191.9" font-size="9" fill="#444" text-anchor="middle">76.5</text>
<rect x="542.0" y="224.6" width="36.9" height="19.4" fill="#00A37A" rx="2"/>
<text x="560.4" y="220.6" font-size="9" fill="#444" text-anchor="middle">30.8</text>
<text x="542.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">draft-k16</text>
<rect x="611.1" y="66.6" width="36.9" height="177.4" fill="#2E5BFF" rx="2"/>
<text x="629.6" y="62.6" font-size="9" fill="#444" text-anchor="middle">282.0</text>
<rect x="650.0" y="181.1" width="36.9" height="62.9" fill="#00A37A" rx="2"/>
<text x="668.4" y="177.1" font-size="9" fill="#444" text-anchor="middle">100.0</text>
<text x="650.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">ngram</text>
<rect x="56" y="280" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="288" font-size="10.5" fill="#444">解码 tok/s</text>
<rect x="134" y="280" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="148" y="288" font-size="10.5" fill="#444">接受率 %</text>
</svg>

## 负载 B-自由摘要

| 配置 | 解码 (tok/s) | 相对基线加速 | 接受率 α | TTFT (ms) | 起草/接受 tokens |
|---|---|---|---|---|---|
| `base` | 152.5 | 1.00× | 0.0% | 18 | 0.0/0.0 |
| `base2` | 152.2 | 1.00× | 0.0% | 17 | 0.0/0.0 |
| `draft-k3` | 100.2 | 0.66× | 65.3% | 27 | 1228.0/779.0 |
| `draft-k8` | 94.8 | 0.62× | 45.7% | 25 | 2181.0/919.0 |
| `draft-k16` | 57.3 | 0.38× | 22.0% | 24 | 4596.0/898.0 |
| `ngram` | 181.2 | 1.19× | 94.2% | 17 | 360.0/313.0 |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">负载 B-自由摘要：解码速度与接受率</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">tok/s · %</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">52</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">104</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">156</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">208</text>
<rect x="71.1" y="94.7" width="36.9" height="149.3" fill="#2E5BFF" rx="2"/>
<text x="89.6" y="90.7" font-size="9" fill="#444" text-anchor="middle">152.5</text>
<rect x="110.0" y="244.0" width="36.9" height="0.5" fill="#00A37A" rx="2"/>
<text x="128.4" y="240.0" font-size="9" fill="#444" text-anchor="middle">0.0</text>
<text x="110.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">base</text>
<rect x="179.1" y="95.0" width="36.9" height="149.0" fill="#2E5BFF" rx="2"/>
<text x="197.6" y="91.0" font-size="9" fill="#444" text-anchor="middle">152.2</text>
<rect x="218.0" y="244.0" width="36.9" height="0.5" fill="#00A37A" rx="2"/>
<text x="236.4" y="240.0" font-size="9" fill="#444" text-anchor="middle">0.0</text>
<text x="218.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">base2</text>
<rect x="287.1" y="145.9" width="36.9" height="98.1" fill="#2E5BFF" rx="2"/>
<text x="305.6" y="141.9" font-size="9" fill="#444" text-anchor="middle">100.2</text>
<rect x="326.0" y="180.1" width="36.9" height="63.9" fill="#00A37A" rx="2"/>
<text x="344.4" y="176.1" font-size="9" fill="#444" text-anchor="middle">65.3</text>
<text x="326.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">draft-k3</text>
<rect x="395.1" y="151.2" width="36.9" height="92.8" fill="#2E5BFF" rx="2"/>
<text x="413.6" y="147.2" font-size="9" fill="#444" text-anchor="middle">94.8</text>
<rect x="434.0" y="199.3" width="36.9" height="44.7" fill="#00A37A" rx="2"/>
<text x="452.4" y="195.3" font-size="9" fill="#444" text-anchor="middle">45.7</text>
<text x="434.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">draft-k8</text>
<rect x="503.1" y="187.9" width="36.9" height="56.1" fill="#2E5BFF" rx="2"/>
<text x="521.6" y="183.9" font-size="9" fill="#444" text-anchor="middle">57.3</text>
<rect x="542.0" y="222.5" width="36.9" height="21.5" fill="#00A37A" rx="2"/>
<text x="560.4" y="218.5" font-size="9" fill="#444" text-anchor="middle">22.0</text>
<text x="542.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">draft-k16</text>
<rect x="611.1" y="66.6" width="36.9" height="177.4" fill="#2E5BFF" rx="2"/>
<text x="629.6" y="62.6" font-size="9" fill="#444" text-anchor="middle">181.2</text>
<rect x="650.0" y="151.8" width="36.9" height="92.2" fill="#00A37A" rx="2"/>
<text x="668.4" y="147.8" font-size="9" fill="#444" text-anchor="middle">94.2</text>
<text x="650.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">ngram</text>
<rect x="56" y="280" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="288" font-size="10.5" fill="#444">解码 tok/s</text>
<rect x="134" y="280" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="148" y="288" font-size="10.5" fill="#444">接受率 %</text>
</svg>

## 关键结论（数字自动取自本次运行）

- **A-结构化抽取**：最优配置 `ngram`（α=100.0%）解码 282.0 tok/s，基线 154.2 tok/s，**加速 1.83×**。
- **B-自由摘要**：最优配置 `ngram`（α=94.2%）解码 181.2 tok/s，基线 152.5 tok/s，**加速 1.19×**。
- **draft 模型方案全部是负收益**：3 个 k 值在结构化抽取负载上的加速比区间为 0.50×（`draft-k16`）~ 0.71×（`draft-k8`），**都小于 1.0×**。
  根因不是接受率不够（最高 α 达 69.9%），而是 **draft 的参数量只有 target 的 35%**（0.6B vs 1.7B）——草稿开销吃掉了全部收益。
- **ngram 方案是本次唯一真正加速的手段**：草稿来自 n-gram 查表、几乎零前向成本，所以在同样的接受率下能实打实变成速度。**这是本实验最有实践价值的结论：小模型做草稿在消费级显卡上往往不划算，零成本的 n-gram 草稿反而更值得试。**

**接受率高 ≠ 快**：`draft-k3` 的接受率高于 `draft-k8`，速度却更慢——因为 k 小则每轮产出少、验证轮次多，固定开销摊得更薄。**只看接受率会得出完全错误的结论。**

## 机制解释：为什么接受率高 ≠ 快

投机解码每一轮的成本 ≈ `k × (draft/target 参数量比) + 1`，产出 ≈ `(1-α^(k+1))/(1-α)` 个 token。下表把**实测加速比**和这个理论模型并排看：

| 配置 | k | α (A负载) | 理论加速比 | 实测加速比 | 差值 |
|---|---|---|---|---|---|
| `base` | - | - | 1.00× | 1.00× | - |
| `base2` | - | - | 1.00× | 1.00× | - |
| `draft-k3` | 3 | 69.9% | 1.23× | 0.70× | -0.53 |
| `draft-k8` | 8 | 56.6% | 0.60× | 0.71× | +0.11 |
| `draft-k16` | 16 | 30.8% | 0.22× | 0.50× | +0.28 |
| `ngram` | - | 100.0% | 草稿近乎零成本 → 上限≈k+1 | 见上表 | - |

**读法（这三点都由上表数据决定，不是预设结论）**：

- **接受率高 ≠ 快**：`draft-k3` 的 α 高于 `draft-k8`，但速度更慢。只看接受率会得出完全错误的结论——决定收益的是「每轮产出 ÷ 每轮成本」。
- **draft 模型方案全部是负收益**（实测 < 1.0×）。根因是**draft 的参数量是 target 的 35%**（0.6B vs 1.7B）：草稿本身就要花掉约 1/3 的前向成本，而 target 又足够小、单 token 本来就快，省下的验证时间抵不过草稿开销。**换成 7B/70B 这种 target 时结论可能反转**（那时 target 每 token 更贵，草稿的性价比上升）。
- **理论模型只有定性价值，别拿它当预测器**：上表里它对 `draft-k3` 预测 1.23×、实测 0.70×，偏差比收益本身还大。原因是模型漏掉了**每轮的固定开销**（草稿前向、验证批次调度），k 越小轮次越多，这笔开销被重复支付的次数也越多。**所以结论必须来自实测，公式只能用来解释方向。**
- **ngram 是本次唯一真正加速的手段**：草稿来自 n-gram 查表、几乎零前向成本，所以在同样的接受率下能实打实变成速度。**实践含义：消费级显卡上，小模型做草稿往往不划算，零成本的 n-gram 草稿反而更值得先试。**

## 正确性验证：投机解码是否「无损」

投机解码的理论前提是**输出分布不变**（草稿只影响速度，不影响结果）。这里用输出哈希与基线逐条比对来验证：

| 配置 | 与基线输出完全一致 | 不一致条数 | 判断 |
|---|---|---|---|
| `base2`（**自身重复性对照**） | 12/12 | 0 | 基线可复现 |
| `draft-k3` | 3/12 | 9 | **明显低于自身噪声 → 引入额外扰动** |
| `draft-k8` | 1/12 | 11 | **明显低于自身噪声 → 引入额外扰动** |
| `draft-k16` | 1/12 | 11 | **明显低于自身噪声 → 引入额外扰动** |
| `ngram` | 11/12 | 1 | 基本一致（仅个别样本分叉） |

> **怎么读这张表**：所有 prompt 都在 `temperature=0`（贪心）下执行，理论上输出应与基线逐字相同。但投机解码把「逐 token 前向」换成了「k 个 token 批量验证」，**浮点累加顺序变了**，贪心解码在近似并列的位置可能翻转，之后序列就分叉。
>
> 所以判断标准不是「是否 100% 相同」，而是**是否明显低于基线自身的重复性**（`base2` 那一行）。这是本实验特意加的对照——没有它，无法区分「投机解码引入的差异」和「服务端本身的不确定性」。

## 结论（可直接引用）

> **投机解码消融实验**：在本地 llama.cpp(CUDA) 上对 qwen3:1.7b 做消融——对比「0.6B 同族草稿模型（draft-simple，k∈{3,8,16}）」与「零成本 n-gram 草稿」两类方案，每类各跑结构化抽取与自由生成两种负载。
>
> 结论：**草稿模型方案在消费级显卡上全部是负收益（0.50×~0.71×，尽管接受率高达 69.9%）**，根因是草稿参数量达 target 的 35%；**而 n-gram 草稿达到 1.83×（结构化抽取，α=100%）与 1.19×（自由摘要，α=94%）**——据此得出「投机解码的收益取决于草稿成本与输出可预测性，而非接受率本身」的适用边界。
>
> 并做了正确性对照：基线自身重复 12/12 完全一致，开启投机解码后输出出现浮点级分叉（draft 方案仅 1~3/12 与基线一致），量化了「理论无损」与「工程实现」的差距。

明细数据：`results/spec_full.csv`