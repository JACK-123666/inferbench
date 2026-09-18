# 实验 5 · 并发压测报告

- 生成时间：2026-09-18 12:25:13
- 模型：`qwen3:1.7b` ｜ 并发梯度：[1, 2, 4, 8, 16] ｜ 每档 24 请求 ｜ num_predict=64
- 请求方式：**流式**（`stream=true`）——TTFT 是实测首 token 到达时间，不是非流式的 prefill 近似值
- 测量前 GPU：3684/8188 MiB (45%) · util 10% · SM 225MHz · 7.8W · 54℃

## 并发梯度

| 并发 | QPS | 聚合 tok/s | TTFT P50 (ms) | TTFT P95 (ms) | 总时延 P50 (ms) | P95 (ms) | P99 (ms) | 错误率 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.36 | 150.9 | 27.4 | 40.7 | 422.9 | 443.8 | 445.8 | 0.0% |
| 2 | 2.42 | 154.3 | 434.6 | 448.0 | 828.3 | 860.4 | 874.4 | 0.0% |
| 4 | 2.48 | 157.7 | 1216.8 | 1298.3 | 1602.0 | 1706.7 | 1715.2 | 0.0% |
| 8 | 2.51 | 160.2 | 2790.3 | 2813.9 | 3171.7 | 3192.0 | 3194.1 | 0.0% |
| 16 | 2.47 | 157.3 | 4711.4 | 6112.1 | 5084.6 | 6517.6 | 6529.7 | 0.0% |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">并发 → 吞吐</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">次/秒 · tok/s</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">45</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">90</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">135</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">179</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">1</text>
<text x="91.6" y="262" font-size="10" fill="#333" text-anchor="middle">2</text>
<text x="162.8" y="262" font-size="10" fill="#333" text-anchor="middle">4</text>
<text x="305.2" y="262" font-size="10" fill="#333" text-anchor="middle">8</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">16</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">并发数</text>
<polyline points="56.0,241.3 91.6,241.2 162.8,241.2 305.2,241.1 590.0,241.2" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="241.3" r="3" fill="#2E5BFF"/>
<circle cx="91.6" cy="241.2" r="3" fill="#2E5BFF"/>
<circle cx="162.8" cy="241.2" r="3" fill="#2E5BFF"/>
<circle cx="305.2" cy="241.1" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="241.2" r="3" fill="#2E5BFF"/>
<polyline points="56.0,72.4 91.6,68.6 162.8,64.7 305.2,61.9 590.0,65.2" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="72.4" r="3" fill="#00A37A"/>
<circle cx="91.6" cy="68.6" r="3" fill="#00A37A"/>
<circle cx="162.8" cy="64.7" r="3" fill="#00A37A"/>
<circle cx="305.2" cy="61.9" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="65.2" r="3" fill="#00A37A"/>
<rect x="600" y="31" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="614" y="39" font-size="10.5" fill="#444">QPS</text>
<rect x="600" y="49" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="614" y="57" font-size="10.5" fill="#444">聚合 tok/s</text>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">并发 → 尾延迟劣化</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">ms</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">1828</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">3657</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">5485</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">7313</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">1</text>
<text x="91.6" y="262" font-size="10" fill="#333" text-anchor="middle">2</text>
<text x="162.8" y="262" font-size="10" fill="#333" text-anchor="middle">4</text>
<text x="305.2" y="262" font-size="10" fill="#333" text-anchor="middle">8</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">16</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">并发数</text>
<polyline points="56.0,232.2 91.6,220.9 162.8,199.3 305.2,155.5 590.0,102.2" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="232.2" r="3" fill="#2E5BFF"/>
<circle cx="91.6" cy="220.9" r="3" fill="#2E5BFF"/>
<circle cx="162.8" cy="199.3" r="3" fill="#2E5BFF"/>
<circle cx="305.2" cy="155.5" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="102.2" r="3" fill="#2E5BFF"/>
<polyline points="56.0,231.6 91.6,220.0 162.8,196.4 305.2,155.0 590.0,62.2" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="231.6" r="3" fill="#00A37A"/>
<circle cx="91.6" cy="220.0" r="3" fill="#00A37A"/>
<circle cx="162.8" cy="196.4" r="3" fill="#00A37A"/>
<circle cx="305.2" cy="155.0" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="62.2" r="3" fill="#00A37A"/>
<polyline points="56.0,231.6 91.6,219.6 162.8,196.2 305.2,154.9 590.0,61.9" fill="none" stroke="#F2994A" stroke-width="2"/>
<circle cx="56.0" cy="231.6" r="3" fill="#F2994A"/>
<circle cx="91.6" cy="219.6" r="3" fill="#F2994A"/>
<circle cx="162.8" cy="196.2" r="3" fill="#F2994A"/>
<circle cx="305.2" cy="154.9" r="3" fill="#F2994A"/>
<circle cx="590.0" cy="61.9" r="3" fill="#F2994A"/>
<polyline points="56.0,242.9 91.6,231.5 162.8,207.8 305.2,165.5 590.0,73.5" fill="none" stroke="#EB5757" stroke-width="2"/>
<circle cx="56.0" cy="242.9" r="3" fill="#EB5757"/>
<circle cx="91.6" cy="231.5" r="3" fill="#EB5757"/>
<circle cx="162.8" cy="207.8" r="3" fill="#EB5757"/>
<circle cx="305.2" cy="165.5" r="3" fill="#EB5757"/>
<circle cx="590.0" cy="73.5" r="3" fill="#EB5757"/>
<rect x="600" y="31" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="614" y="39" font-size="10.5" fill="#444">P50</text>
<rect x="600" y="49" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="614" y="57" font-size="10.5" fill="#444">P95</text>
<rect x="600" y="67" width="9" height="9" fill="#F2994A" rx="2"/>
<text x="614" y="75" font-size="10.5" fill="#444">P99</text>
<rect x="600" y="85" width="9" height="9" fill="#EB5757" rx="2"/>
<text x="614" y="93" font-size="10.5" fill="#444">TTFT P95</text>
</svg>

## 关键结论（数字自动取自本次运行）

1. **吞吐上限**：并发从 1 提到 8，QPS 2.36 → 2.51（**1.06×**）；聚合 tok/s 达 160.2。
2. **代价是尾延迟**：同区间 P95 从 443.8 ms 涨到 3192.0 ms（**7.19×**），P99 445.8 → 3194.1 ms。
3. **首 Token 延迟**：TTFT P95 从 40.7 ms 变为 2813.9 ms（**69.14×**）——这是最直接对应用户体感的 SLO 指标。
4. ⚠ **本实例已经饱和，加并发只换到排队，换不到吞吐**：并发 1→8 只拿到 **6%** 的吞吐增益，却付出 **69.1 倍** 的首 Token 延迟（TTFT 随并发**线性增长**，是典型 FIFO 排队特征）。
5. **根因**：Ollama 的 `OLLAMA_NUM_PARALLEL` 当前为 **1**（默认值）——所有请求挤在**同一个 slot** 上串行执行，所以并发度只影响排队长度，不影响吞吐。要真正提升并发承载，需要设 `OLLAMA_NUM_PARALLEL>1` 并重启 Ollama（代价是每路并行都要一份 KV cache 显存，需先确认显存余量）。
6. **因此正确的优化方向不是扩并发**，而是降低单请求成本：更低的量化档、更短的上下文、更少的输出 token——这三项都能同时改善吞吐与延迟。
**错误率**：全程 0.0%——并发连接数不是瓶颈，显存能装多大模型才是。

## 可写进简历的结论句

> **并发压测与容量规划**：用 httpx + asyncio 做**流式**并发梯度压测（并发 1→16，实测首 token 到达时间而非 prefill 近似值），发现单实例已饱和——并发 1→8 仅带来 6% 吞吐增益，却让 TTFT P95 上涨 69.1 倍（40.7 → 2813.9 ms）、P99 达 3194.1 ms；据此判定瓶颈在单实例推理（显存带宽/单序列执行）而非并发度，把优化方向从「扩并发」改为「降单请求成本」（量化档位 / 上下文裁剪 / 输出长度控制）。

明细数据：`results/load_full.csv`