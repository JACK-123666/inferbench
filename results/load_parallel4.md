# 实验 5 · 并发压测报告

- 生成时间：2026-09-18 13:03:21
- **适用范围**：⚠ 本次运行早于环境指纹功能（结果里没有 `env` 块），硬件与代码版本均未记录 —— **该结果的适用边界未知**，不要直接跨机器套用

- 模型：`qwen3:1.7b` ｜ 并发梯度：[1, 2, 4, 8, 16] ｜ 每档 24 请求 ｜ num_predict=64
- 请求方式：**流式**（`stream=true`）——TTFT 是实测首 token 到达时间，不是非流式的 prefill 近似值
- 测量前 GPU：5231/8188 MiB (64%) · util 25% · SM 210MHz · 9.1W · 59℃

## 并发梯度

| 并发 | QPS | 聚合 tok/s | TTFT P50 (ms) | TTFT P95 (ms) | 总时延 P50 (ms) | P95 (ms) | P99 (ms) | 错误率 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.24 | 143.0 | 25.6 | 36.0 | 437.5 | 496.3 | 569.3 | 0.0% |
| 2 | 4.01 | 255.7 | 34.9 | 47.4 | 497.3 | 514.1 | 515.8 | 0.0% |
| 4 | 6.09 | 387.9 | 51.9 | 62.6 | 643.5 | 702.8 | 704.0 | 0.0% |
| 8 | 6.27 | 399.5 | 676.6 | 696.9 | 1269.4 | 1287.6 | 1287.9 | 0.0% |
| 16 | 6.18 | 393.6 | 1641.9 | 1982.2 | 2195.8 | 2580.7 | 2582.8 | 0.0% |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">并发 → 吞吐</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">次/秒 · tok/s</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">112</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">224</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">336</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">447</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">1</text>
<text x="91.6" y="262" font-size="10" fill="#333" text-anchor="middle">2</text>
<text x="162.8" y="262" font-size="10" fill="#333" text-anchor="middle">4</text>
<text x="305.2" y="262" font-size="10" fill="#333" text-anchor="middle">8</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">16</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">并发数</text>
<polyline points="56.0,243.0 91.6,242.2 162.8,241.2 305.2,241.1 590.0,241.2" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="243.0" r="3" fill="#2E5BFF"/>
<circle cx="91.6" cy="242.2" r="3" fill="#2E5BFF"/>
<circle cx="162.8" cy="241.2" r="3" fill="#2E5BFF"/>
<circle cx="305.2" cy="241.1" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="241.2" r="3" fill="#2E5BFF"/>
<polyline points="56.0,178.8 91.6,127.4 162.8,67.1 305.2,61.9 590.0,64.5" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="178.8" r="3" fill="#00A37A"/>
<circle cx="91.6" cy="127.4" r="3" fill="#00A37A"/>
<circle cx="162.8" cy="67.1" r="3" fill="#00A37A"/>
<circle cx="305.2" cy="61.9" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="64.5" r="3" fill="#00A37A"/>
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
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">723</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">1446</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">2170</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">2893</text>
<text x="56.0" y="262" font-size="10" fill="#333" text-anchor="middle">1</text>
<text x="91.6" y="262" font-size="10" fill="#333" text-anchor="middle">2</text>
<text x="162.8" y="262" font-size="10" fill="#333" text-anchor="middle">4</text>
<text x="305.2" y="262" font-size="10" fill="#333" text-anchor="middle">8</text>
<text x="590.0" y="262" font-size="10" fill="#333" text-anchor="middle">16</text>
<text x="323.0" y="292" font-size="10.5" fill="#333" text-anchor="middle">并发数</text>
<polyline points="56.0,213.1 91.6,208.9 162.8,198.6 305.2,154.5 590.0,89.1" fill="none" stroke="#2E5BFF" stroke-width="2"/>
<circle cx="56.0" cy="213.1" r="3" fill="#2E5BFF"/>
<circle cx="91.6" cy="208.9" r="3" fill="#2E5BFF"/>
<circle cx="162.8" cy="198.6" r="3" fill="#2E5BFF"/>
<circle cx="305.2" cy="154.5" r="3" fill="#2E5BFF"/>
<circle cx="590.0" cy="89.1" r="3" fill="#2E5BFF"/>
<polyline points="56.0,209.0 91.6,207.7 162.8,194.4 305.2,153.2 590.0,62.0" fill="none" stroke="#00A37A" stroke-width="2"/>
<circle cx="56.0" cy="209.0" r="3" fill="#00A37A"/>
<circle cx="91.6" cy="207.7" r="3" fill="#00A37A"/>
<circle cx="162.8" cy="194.4" r="3" fill="#00A37A"/>
<circle cx="305.2" cy="153.2" r="3" fill="#00A37A"/>
<circle cx="590.0" cy="62.0" r="3" fill="#00A37A"/>
<polyline points="56.0,203.9 91.6,207.6 162.8,194.4 305.2,153.2 590.0,61.9" fill="none" stroke="#F2994A" stroke-width="2"/>
<circle cx="56.0" cy="203.9" r="3" fill="#F2994A"/>
<circle cx="91.6" cy="207.6" r="3" fill="#F2994A"/>
<circle cx="162.8" cy="194.4" r="3" fill="#F2994A"/>
<circle cx="305.2" cy="153.2" r="3" fill="#F2994A"/>
<circle cx="590.0" cy="61.9" r="3" fill="#F2994A"/>
<polyline points="56.0,241.5 91.6,240.7 162.8,239.6 305.2,194.9 590.0,104.2" fill="none" stroke="#EB5757" stroke-width="2"/>
<circle cx="56.0" cy="241.5" r="3" fill="#EB5757"/>
<circle cx="91.6" cy="240.7" r="3" fill="#EB5757"/>
<circle cx="162.8" cy="239.6" r="3" fill="#EB5757"/>
<circle cx="305.2" cy="194.9" r="3" fill="#EB5757"/>
<circle cx="590.0" cy="104.2" r="3" fill="#EB5757"/>
<rect x="600" y="31" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="614" y="39" font-size="10.5" fill="#444">P50</text>
<rect x="600" y="49" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="614" y="57" font-size="10.5" fill="#444">P95</text>
<rect x="600" y="67" width="9" height="9" fill="#F2994A" rx="2"/>
<text x="614" y="75" font-size="10.5" fill="#444">P99</text>
<rect x="600" y="85" width="9" height="9" fill="#EB5757" rx="2"/>
<text x="614" y="93" font-size="10.5" fill="#444">TTFT P95</text>
</svg>

## 与基线对比（改进验证）

- 基线：`qwen3:1.7b` @ `default`，OLLAMA_NUM_PARALLEL=1，模型占用 1.59 GB
- 本次：`qwen3:1.7b` @ `http://127.0.0.1:11440`，OLLAMA_NUM_PARALLEL=4，模型占用 2.95 GB

| 并发 | 基线 QPS | 本次 QPS | 吞吐变化 | 基线 TTFT P50 | 本次 TTFT P50 | TTFT 变化 |
|---|---|---|---|---|---|---|
| 1 | 2.36 | 2.24 | ↓0.95×** | 27.4 ms | 25.6 ms | **↓0.93×** |
| 2 | 2.42 | 4.01 | **↑1.66×** | 434.6 ms | 34.9 ms | **↓0.08×** |
| 4 | 2.48 | 6.09 | **↑2.46×** | 1216.8 ms | 51.9 ms | **↓0.04×** |
| 8 | 2.51 | 6.27 | **↑2.50×** | 2790.3 ms | 676.6 ms | **↓0.24×** |
| 16 | 2.47 | 6.18 | **↑2.50×** | 4711.4 ms | 1641.9 ms | **↓0.35×** |

**验证结论**：

- 峰值吞吐 2.51 → **6.27 QPS**（**2.50×**），聚合 tok/s 160.2 → 399.5。
- 代价是显存：模型占用 1.59 GB → 2.95 GB（**+86%**）——每路并行各占一份 KV cache，8GB 卡上要腾出余量。
- 结论：**同一个硬件，改一个并行度配置就能把吞吐翻倍、把首 Token 延迟降一个数量级**；这也解释了为什么「加并发不涨吞吐」——瓶颈从来不是并发度，而是单 slot 串行执行。

## 关键结论（数字自动取自本次运行）

1. **吞吐上限**：并发从 1 提到 8，QPS 2.24 → 6.27（**2.80×**）；聚合 tok/s 达 399.5。
2. **代价是尾延迟**：同区间 P95 从 496.3 ms 涨到 1287.6 ms（**2.59×**），P99 569.3 → 1287.9 ms。
3. **首 Token 延迟**：TTFT P95 从 36.0 ms 变为 696.9 ms（**19.36×**）——这是最直接对应用户体感的 SLO 指标。
4. **饱和点（推荐工作区间）**：并发 8 附近——该点 QPS 边际增益已降到 3%，P95 是最低档的 2.6 倍。**再加并发只换到延迟，换不到吞吐。**
**错误率**：全程 0.0%——并发连接数不是瓶颈，显存能装多大模型才是。

## 可写进简历的结论句

> **并发压测与容量规划**：httpx + asyncio 流式压测并发 1→16，QPS 提升 2.8×（峰值 6.27），P95 1287.6 ms、TTFT P95 696.9 ms，据此按 P95 SLO 反推可承载并发。

明细数据：`results/load_parallel4.csv`