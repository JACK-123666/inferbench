# 实验 6 · KV cache 上下文扫参报告

- 生成时间：2026-09-21 13:24:52
- **适用范围**：NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB） · 驱动 616.92 · CUDA 13.4 · Ollama 0.32.15 · OLLAMA_NUM_PARALLEL=1（客户端环境变量） · Python 3.13.12 · inferbench 0.3.0+d752376
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU` · 显存 8188 MiB · 驱动 616.92 · CUDA 13.4 · 算力 8.9
  - 运行时：Ollama `0.32.15` @ `http://127.0.0.1:11434` · `OLLAMA_NUM_PARALLEL=1`（客户端环境变量；服务端探测不到，若服务端在别处运行请以服务端为准）
  - 代码版本：inferbench `0.3.0` · git `d752376` · Python 3.13.12 · 指纹采集于 2026-09-21 13:24:52
- 语料：内置：Synapse 100 条人工标注（缓存于 data/eval_set.jsonl） · 100 条（难例 23）· 指纹 `665ef7c3`

- 模型：`qwen3:1.7b-q8_0`（全程同一个模型，唯一变量是 `num_ctx`）
- 评测集：100 条（难例 23 条）
- 测量口径：temperature=0.0 · think=False · seed=42 · few-shot=3 · 重复 3 次取中位数 · 每档前卸载模型清空显存

## 先算一遍理论值（当尺子用）

- 结构：qwen3 · 28 层 × 8/16 个 KV 头 × head_dim 128，模型声明上下文 40960
- 理论 KV：`2 (K+V) × 28 层 × 8 KV 头 × 128 × 2 字节` = **114688 B/token = 0.1094 GB / 1K token**
- 按此推算，跑满模型声明的 40960 上下文，KV cache 单独就要 **4.38 GB**（还没算模型权重）

## 扫参结果

| 请求 num_ctx | 实际生效 | 总占用 (GB) | 推算 KV (GB) | GPU 分流 | TTFT 中位 (ms) | 解码 (tok/s) | 准确率 | P95 端到端 (ms) | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 2.02 | 0.20 | 100% GPU | 16.3 | 111.1 | 92.0% | 101.4 | — |
| 4096 | 4096 | 2.24 | 0.42 | 100% GPU | 16.6 | 110.2 | 92.0% | 105.3 | — |
| 8192 | 8192 | 2.74 | 0.91 | 100% GPU | 15.3 | 117.4 | 92.0% | 87.9 | — |
| 16384 | 16384 | 3.63 | 1.80 | 100% GPU | 15.8 | 114.6 | 92.0% | 92.6 | — |
| 32768 | 32768 | 5.33 | 3.51 | 100% GPU | 15.9 | 114.5 | 92.0% | 92.8 | — |
| 40960 | 40960 | 6.77 | 4.95 | 86% GPU / 14% CPU | 49.2 | 71.0 | 93.0% | 265.3 | **溢出到 CPU** |
| 65536 | 40960 | 6.77 | 4.95 | 86% GPU / 14% CPU | 48.0 | 84.5 | 93.0% | 148.0 | **静默截断**/**溢出到 CPU** |
| 131072 | 40960 | 6.77 | 4.95 | 86% GPU / 14% CPU | 42.7 | 84.3 | 93.0% | 190.8 | **静默截断**/**溢出到 CPU** |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">上下文长度 → 显存构成（KV cache 是按 num_ctx 满额预分配的）</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">GB</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">2</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">4</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">6</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">8</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">2048</text>
<text x="64.5" y="262" font-size="10" fill="#333" text-anchor="middle">4096</text>
<text x="81.4" y="262" font-size="10" fill="#333" text-anchor="middle">8192</text>
<text x="115.3" y="262" font-size="10" fill="#333" text-anchor="middle">16384</text>
<text x="183.1" y="262" font-size="10" fill="#333" text-anchor="middle">32768</text>
<text x="217.0" y="262" font-size="10" fill="#333" text-anchor="middle">40960</text>
<text x="318.8" y="262" font-size="10" fill="#333" text-anchor="middle">65536</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">131072</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">请求的 num_ctx (tokens)</text>
<polyline points="56.0,189.6 64.5,183.7 81.4,170.3 115.3,146.4 183.1,100.5 217.0,61.9 318.8,61.9 590.0,61.9" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="189.6" r="3" fill="#2E5BFF"/>
<circle cx="64.5" cy="183.7" r="3" fill="#2E5BFF"/>
<circle cx="81.4" cy="170.3" r="3" fill="#2E5BFF"/>
<circle cx="115.3" cy="146.4" r="3" fill="#2E5BFF"/>
<circle cx="183.1" cy="100.5" r="3" fill="#2E5BFF"/>
<circle cx="217.0" cy="61.9" r="3" fill="#2E5BFF"/>
<circle cx="318.8" cy="61.9" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="61.9" r="3" fill="#2E5BFF"/>
<polyline points="56.0,238.1 64.5,232.2 81.4,220.5 115.3,196.9 183.1,149.8 217.0,126.3 318.8,126.3 590.0,126.3" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="238.1" r="3" fill="#00A37A"/>
<circle cx="64.5" cy="232.2" r="3" fill="#00A37A"/>
<circle cx="81.4" cy="220.5" r="3" fill="#00A37A"/>
<circle cx="115.3" cy="196.9" r="3" fill="#00A37A"/>
<circle cx="183.1" cy="149.8" r="3" fill="#00A37A"/>
<circle cx="217.0" cy="126.3" r="3" fill="#00A37A"/>
<circle cx="318.8" cy="126.3" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="126.3" r="3" fill="#00A37A"/>
<polyline points="56.0,194.9 64.5,194.9 81.4,194.9 115.3,194.9 183.1,194.9 217.0,194.9 318.8,194.9 590.0,194.9" fill="none" stroke="#F2994A" stroke-width="2"/>
<circle cx="56.0" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="64.5" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="81.4" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="115.3" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="183.1" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="217.0" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="318.8" cy="194.9" r="3" fill="#F2994A"/>
<circle cx="590.0" cy="194.9" r="3" fill="#F2994A"/>
<rect x="600" y="31" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="614" y="39" font-size="10.5" fill="#444">实测总占用 (GB)</text>
<rect x="600" y="49" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="614" y="57" font-size="10.5" fill="#444">理论 KV cache (GB)</text>
<rect x="600" y="67" width="9" height="9" fill="#F2994A" rx="2"/>
<text x="614" y="75" font-size="10.5" fill="#444">模型权重基线 (GB)</text>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">上下文长度 → 速度与 GPU 分流</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">tok/s · ms · %</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">34</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">68</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">101</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">135</text>
<rect x="67.3" y="76.1" width="17.4" height="167.9" fill="#2E5BFF" rx="2"/>
<text x="76.1" y="72.1" font-size="9" fill="#444" text-anchor="middle">111.1</text>
<rect x="86.8" y="219.4" width="17.4" height="24.6" fill="#00A37A" rx="2"/>
<text x="95.5" y="215.4" font-size="9" fill="#444" text-anchor="middle">16.3</text>
<rect x="106.2" y="92.9" width="17.4" height="151.1" fill="#F2994A" rx="2"/>
<text x="114.9" y="88.9" font-size="9" fill="#444" text-anchor="middle">100.0</text>
<text x="96.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">2048</text>
<rect x="148.3" y="77.6" width="17.4" height="166.4" fill="#2E5BFF" rx="2"/>
<text x="157.1" y="73.6" font-size="9" fill="#444" text-anchor="middle">110.2</text>
<rect x="167.8" y="219.0" width="17.4" height="25.0" fill="#00A37A" rx="2"/>
<text x="176.5" y="215.0" font-size="9" fill="#444" text-anchor="middle">16.6</text>
<rect x="187.2" y="92.9" width="17.4" height="151.1" fill="#F2994A" rx="2"/>
<text x="195.9" y="88.9" font-size="9" fill="#444" text-anchor="middle">100.0</text>
<text x="177.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">4096</text>
<rect x="229.3" y="66.6" width="17.4" height="177.4" fill="#2E5BFF" rx="2"/>
<text x="238.1" y="62.6" font-size="9" fill="#444" text-anchor="middle">117.4</text>
<rect x="248.8" y="220.8" width="17.4" height="23.2" fill="#00A37A" rx="2"/>
<text x="257.5" y="216.8" font-size="9" fill="#444" text-anchor="middle">15.3</text>
<rect x="268.2" y="92.9" width="17.4" height="151.1" fill="#F2994A" rx="2"/>
<text x="276.9" y="88.9" font-size="9" fill="#444" text-anchor="middle">100.0</text>
<text x="258.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">8192</text>
<rect x="310.3" y="70.9" width="17.4" height="173.1" fill="#2E5BFF" rx="2"/>
<text x="319.1" y="66.9" font-size="9" fill="#444" text-anchor="middle">114.6</text>
<rect x="329.8" y="220.1" width="17.4" height="23.9" fill="#00A37A" rx="2"/>
<text x="338.5" y="216.1" font-size="9" fill="#444" text-anchor="middle">15.8</text>
<rect x="349.2" y="92.9" width="17.4" height="151.1" fill="#F2994A" rx="2"/>
<text x="357.9" y="88.9" font-size="9" fill="#444" text-anchor="middle">100.0</text>
<text x="339.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">16384</text>
<rect x="391.3" y="71.0" width="17.4" height="173.0" fill="#2E5BFF" rx="2"/>
<text x="400.1" y="67.0" font-size="9" fill="#444" text-anchor="middle">114.5</text>
<rect x="410.8" y="220.0" width="17.4" height="24.0" fill="#00A37A" rx="2"/>
<text x="419.5" y="216.0" font-size="9" fill="#444" text-anchor="middle">15.9</text>
<rect x="430.2" y="92.9" width="17.4" height="151.1" fill="#F2994A" rx="2"/>
<text x="438.9" y="88.9" font-size="9" fill="#444" text-anchor="middle">100.0</text>
<text x="420.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">32768</text>
<rect x="472.3" y="136.7" width="17.4" height="107.3" fill="#2E5BFF" rx="2"/>
<text x="481.1" y="132.7" font-size="9" fill="#444" text-anchor="middle">71.0</text>
<rect x="491.8" y="169.7" width="17.4" height="74.3" fill="#00A37A" rx="2"/>
<text x="500.5" y="165.7" font-size="9" fill="#444" text-anchor="middle">49.2</text>
<rect x="511.2" y="114.8" width="17.4" height="129.2" fill="#F2994A" rx="2"/>
<text x="519.9" y="110.8" font-size="9" fill="#444" text-anchor="middle">85.5</text>
<text x="501.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">40960</text>
<rect x="553.3" y="116.4" width="17.4" height="127.6" fill="#2E5BFF" rx="2"/>
<text x="562.1" y="112.4" font-size="9" fill="#444" text-anchor="middle">84.5</text>
<rect x="572.8" y="171.4" width="17.4" height="72.6" fill="#00A37A" rx="2"/>
<text x="581.5" y="167.4" font-size="9" fill="#444" text-anchor="middle">48.0</text>
<rect x="592.2" y="114.8" width="17.4" height="129.2" fill="#F2994A" rx="2"/>
<text x="600.9" y="110.8" font-size="9" fill="#444" text-anchor="middle">85.5</text>
<text x="582.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">65536</text>
<rect x="634.3" y="116.7" width="17.4" height="127.3" fill="#2E5BFF" rx="2"/>
<text x="643.1" y="112.7" font-size="9" fill="#444" text-anchor="middle">84.3</text>
<rect x="653.8" y="179.5" width="17.4" height="64.5" fill="#00A37A" rx="2"/>
<text x="662.5" y="175.5" font-size="9" fill="#444" text-anchor="middle">42.7</text>
<rect x="673.2" y="114.8" width="17.4" height="129.2" fill="#F2994A" rx="2"/>
<text x="681.9" y="110.8" font-size="9" fill="#444" text-anchor="middle">85.5</text>
<text x="663.5" y="262" font-size="10.5" fill="#333" text-anchor="middle">131072</text>
<rect x="56" y="280" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="288" font-size="10.5" fill="#444">解码吞吐 (tok/s)</text>
<rect x="162" y="280" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="176" y="288" font-size="10.5" fill="#444">TTFT 中位 (ms)</text>
<rect x="268" y="280" width="9" height="9" fill="#F2994A" rx="2"/>
<text x="282" y="288" font-size="10.5" fill="#444">GPU 分流 (%)</text>
</svg>

## 关键结论（数字自动取自本次运行）

1. **KV cache 的显存是可以算出来的，而且实测对得上**：实测斜率 **0.1103 GB / 1K token**，从 GGUF 结构参数算出的理论值 0.1094 GB / 1K token，两者相差 0.8%。→ 归因成立：显存增长确实来自 KV cache，不是框架开销或显存碎片。工程含义是**上下文预算可以在上线前算出来，不用靠试**。
2. **KV cache 会超过模型权重本身**：本次拟合出的权重基线 1.82 GB，KV cache 在约 **16940 token** 处追平它，之后 KV 成为显存的主要构成。→ 也就是说「换个小模型省显存」在长上下文下**基本失效**：`qwen3:0.6b` 和 `qwen3:1.7b` 的 KV 头数/层数完全相同，KV cache 一样大，省下的只有权重那一部分。
3. **超过模型上限会被静默截断（本次最意外的发现）**：请求 `num_ctx` = 65536, 131072 时，实际生效的都是 **40960**，而 Ollama **既不报错也不警告**。模型的声明上限就写在 GGUF 里（`context_length`），超出的部分被丢弃。→ 把 `num_ctx` 写成很大的值（很多框架的默认占位值）**换不来更长的上下文，只换来显存浪费和溢出**。
4. **溢出到 CPU 是一个台阶，不是斜坡**：从 `num_ctx=40960` 起 GPU 分流降到 86%（86% GPU / 14% CPU）。对照组 num_ctx=32768（5.33 GB，100% GPU）解码 114.5 tok/s，该档解码 71.0 tok/s。→ 显存不够时框架**不会拒绝启动，而是把层 offload 到 CPU 继续跑**，表现成「能跑但慢很多」——这种静默降级比直接 OOM 更难排查。

## 结论（可直接引用；数字都是本次真实测量值）

> **KV cache 上下文扫参**：在 RTX 4060 8GB 上对同一模型扫 8 档 `num_ctx`（2048 → 131072），实测 KV cache 成本 **0.1103 GB / 1K token**，与按 GGUF 结构参数算出的理论值（0.1094）相差 0.8%，据此把「上下文长度」从配置项变成可预算的显存支出。
>
> 上下文涨到约 16940 token 时，KV cache 的显存超过模型权重本身。
>
> 并发现请求 `num_ctx` 超过模型声明上限时，运行时会**静默截断**到上限值而不报错。
>
> 显存不足时表现为 GPU 分流下降、解码吞吐阶跃式下跌（而非启动失败），属于静默降级。

> **质量侧（逐条比对，不说「基本一致」）**：以 `num_ctx=32768` 为基准逐条比对预测，5 个 100% GPU 档位的预测**完全一致**；出现差异的只有溢出到 CPU 的档位 —— `num_ctx=40960`（3/300 条，涉及样本 14）。也就是说准确率那一跳**不是上下文变长让模型变准了**，而是换了执行路径（GPU → CPU offload）后浮点累加顺序改变，贪心解码在并列位置翻转，恰好把一条「输出非法标签」的样本翻成了「输出正确标签」。**1 条样本就值 1.0 个百分点的准确率** —— 这正是「不加对照就无法归因」的活例子。

## 复现方式

```powershell
python -m inferbench kv --model qwen3:1.7b-q8_0 --levels 2048,4096,8192,16384,32768,40960,65536,131072
```

明细数据：`results/kv_full.csv`（请求级，含每条样本的延迟、显存、GPU 分流）