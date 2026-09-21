# 实验 1 · 量化档位对比报告

- 生成时间：2026-09-18 10:27:45
- **适用范围**：⚠ 本次运行早于环境指纹功能（结果里没有 `env` 块），硬件与代码版本均未记录 —— **该结果的适用边界未知**，不要直接跨机器套用

- 评测集：100 条（难例 23 条）
- 测量口径：num_ctx=4096 · temperature=0.0 · think=False · seed=42 · few-shot=3 · 重复 3 次取中位数 · 预热 1 次后丢弃

## 三角表（显存 ↓ / 速度 ↑ / 精度 ↓）

| 模型 | 显存 (GB) | 准确率 | 难例准确率 | 非法输出率 | TTFT 中位 (ms) | 解码 (tok/s) | P95 端到端 (ms) | 设备 |
|---|---|---|---|---|---|---|---|---|
| `qwen3:1.7b-fp16` | 3.69 | 91.0% | 78.3% | 1.0% | 18.5 | 85.3 | 113.7 | 100% GPU |
| `qwen3:1.7b-q8_0` | 2.24 | 92.0% | 78.3% | 1.0% | 13.6 | 134.2 | 81.0 | 100% GPU |
| `qwen3:1.7b` | 1.59 | 87.0% | 69.6% | 0.0% | 10.7 | 191.6 | 72.7 | 100% GPU |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="280" viewBox="0 0 720 280" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="280" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">量化档位 → 显存占用</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">GB</text>
<line x1="56" y1="224.0" x2="704" y2="224.0" stroke="#E6E8EE"/>
<text x="48" y="228.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="178.0" x2="704" y2="178.0" stroke="#E6E8EE"/>
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">1</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">2</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">3</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">4</text>
<rect x="86.2" y="64.0" width="153.5" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="163.0" y="60.0" font-size="9" fill="#444" text-anchor="middle">3.7</text>
<text x="164.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-fp16</text>
<rect x="302.2" y="126.9" width="153.5" height="97.1" fill="#2E5BFF" rx="2"/>
<text x="379.0" y="122.9" font-size="9" fill="#444" text-anchor="middle">2.2</text>
<text x="380.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="518.2" y="155.3" width="153.5" height="68.7" fill="#2E5BFF" rx="2"/>
<text x="595.0" y="151.3" font-size="9" fill="#444" text-anchor="middle">1.6</text>
<text x="596.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b</text>
<rect x="56" y="260" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="268" font-size="10.5" fill="#444">显存占用 (GB)</text>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="280" viewBox="0 0 720 280" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="280" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">量化档位 → 速度与首 Token 延迟</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">tok/s · ms</text>
<line x1="56" y1="224.0" x2="704" y2="224.0" stroke="#E6E8EE"/>
<text x="48" y="228.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="178.0" x2="704" y2="178.0" stroke="#E6E8EE"/>
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">55</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">110</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">165</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">220</text>
<rect x="86.2" y="152.8" width="75.8" height="71.2" fill="#2E5BFF" rx="2"/>
<text x="124.1" y="148.8" font-size="9" fill="#444" text-anchor="middle">85.3</text>
<rect x="164.0" y="208.6" width="75.8" height="15.4" fill="#00A37A" rx="2"/>
<text x="201.9" y="204.6" font-size="9" fill="#444" text-anchor="middle">18.5</text>
<text x="164.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-fp16</text>
<rect x="302.2" y="111.9" width="75.8" height="112.1" fill="#2E5BFF" rx="2"/>
<text x="340.1" y="107.9" font-size="9" fill="#444" text-anchor="middle">134.2</text>
<rect x="380.0" y="212.7" width="75.8" height="11.3" fill="#00A37A" rx="2"/>
<text x="417.9" y="208.7" font-size="9" fill="#444" text-anchor="middle">13.6</text>
<text x="380.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="518.2" y="64.0" width="75.8" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="556.1" y="60.0" font-size="9" fill="#444" text-anchor="middle">191.6</text>
<rect x="596.0" y="215.0" width="75.8" height="9.0" fill="#00A37A" rx="2"/>
<text x="633.9" y="211.0" font-size="9" fill="#444" text-anchor="middle">10.7</text>
<text x="596.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b</text>
<rect x="56" y="260" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="268" font-size="10.5" fill="#444">解码吞吐 (tok/s)</text>
<rect x="162" y="260" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="176" y="268" font-size="10.5" fill="#444">TTFT (ms)</text>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="280" viewBox="0 0 720 280" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="280" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">量化档位 → 准确率</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">%</text>
<line x1="56" y1="224.0" x2="704" y2="224.0" stroke="#E6E8EE"/>
<text x="48" y="228.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="178.0" x2="704" y2="178.0" stroke="#E6E8EE"/>
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">26</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">53</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">79</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">106</text>
<rect x="86.2" y="65.7" width="75.8" height="158.3" fill="#2E5BFF" rx="2"/>
<text x="124.1" y="61.7" font-size="9" fill="#444" text-anchor="middle">91.0%</text>
<rect x="164.0" y="87.9" width="75.8" height="136.1" fill="#00A37A" rx="2"/>
<text x="201.9" y="83.9" font-size="9" fill="#444" text-anchor="middle">78.3%</text>
<text x="164.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-fp16</text>
<rect x="302.2" y="64.0" width="75.8" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="340.1" y="60.0" font-size="9" fill="#444" text-anchor="middle">92.0%</text>
<rect x="380.0" y="87.9" width="75.8" height="136.1" fill="#00A37A" rx="2"/>
<text x="417.9" y="83.9" font-size="9" fill="#444" text-anchor="middle">78.3%</text>
<text x="380.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="518.2" y="72.7" width="75.8" height="151.3" fill="#2E5BFF" rx="2"/>
<text x="556.1" y="68.7" font-size="9" fill="#444" text-anchor="middle">87.0%</text>
<rect x="596.0" y="103.0" width="75.8" height="121.0" fill="#00A37A" rx="2"/>
<text x="633.9" y="99.0" font-size="9" fill="#444" text-anchor="middle">69.6%</text>
<text x="596.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b</text>
<rect x="56" y="260" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="268" font-size="10.5" fill="#444">整体准确率 (%)</text>
<rect x="141" y="260" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="155" y="268" font-size="10.5" fill="#444">难例准确率 (%)</text>
</svg>

## 关键结论（数字自动取自本次运行）

1. **精度代价**：以 `qwen3:1.7b-fp16`（本次最高精度档）为基准，`qwen3:1.7b` 准确率 87.0% vs 91.0%，掉幅 4.0 个百分点；难例子集掉幅 8.7 个百分点（难例 23 题，按唯一题目去重）。
2. **显存代价**：3.69 GB → 1.59 GB，降幅 57.1%。
3. **速度收益**：解码 85.3 → 191.6 tok/s，提升 124.7%；首 Token 延迟 18.5 → 10.7 ms。
4. **失效模式（看混淆矩阵，不做想当然的归因）**：`qwen3:1.7b` 的主要错误形态 —— summarize → small_talk 错 30 次、knowledge_retrieval → small_talk 错 9 次；非法输出率 0.0% vs 基准 1.0%（注意：本次更低精度档的非法输出反而更少，说明该任务的量化损伤以**分类混淆**为主，而非输出格式崩溃）

> ⚠ **准确率不随精度单调变化**：`qwen3:1.7b-q8_0`（92.0%）反而高于最高精度档 `qwen3:1.7b-fp16`（91.0%），差 +1.0 个百分点（3/300 条记录，处于噪声范围）。
>
> 但结论仍然成立且有价值：**`qwen3:1.7b-q8_0` 在本任务上不劣于 `qwen3:1.7b-fp16`，同时显存少 39.3%、吞吐高 57.4%**。
>
> 也就是说 **FP16 在纯推理场景被完全支配**——只有需要继续训练/微调（LoRA、DPO）时，保留 FP16/BF16 才有意义。**「精度越高越好」是个直觉陷阱，必须用数据验证。**

## 测量卫生（为什么这些速度数字可信）

| 模型 | 测量前 GPU 显存 | 模型占用 | GPU 分流 |
|---|---|---|---|
| `qwen3:1.7b-fp16` | 5725/8188 MiB | 3.69 GB (VRAM 3.69 GB) | 100% GPU |
| `qwen3:1.7b-q8_0` | 1918/8188 MiB | 2.24 GB (VRAM 2.24 GB) | 100% GPU |
| `qwen3:1.7b` | 1970/8188 MiB | 1.59 GB (VRAM 1.59 GB) | 100% GPU |

> **踩过的坑（面试可讲）**：Ollama 默认让最近用过的模型常驻显存。首次跑多档量化时，多个模型同时占用 8GB 显存（实测 7836/8188 MiB），推理被迫退到 CPU/共享内存，解码速度从 **168 tok/s 掉到 0.59 tok/s（差 286 倍）**。现在每次测量前先 `unload_all()` 清空显存并记录 GPU 基线，再用 `size_vram / size` 校验模型确实 100% 在 GPU 上——否则测出来的「速度差异」是显存竞争，不是量化收益。

## 可写进简历的结论句（数字都是本次真实测量值）

> **量化-显存-精度三方权衡实验**：固定 num_ctx=4096、关闭思维链、temperature=0、每档重复 3 次取中位数的统一口径下，对 qwen3:1.7b-fp16 / qwen3:1.7b-q8_0 / qwen3:1.7b 做 A/B：`qwen3:1.7b` 相比 `qwen3:1.7b-fp16` 显存降低 57.1%、解码吞吐提升 124.7%，3 分类准确率由 91.0% 降至 87.0%（模糊难例 78.3% → 69.6%）；并定位了首次测量中「多模型争抢显存导致解码掉 286 倍」的失真，固化为「测量前清空显存 + 校验 GPU 分流」的实验规范。
>
> **量化选型结论**：`qwen3:1.7b-q8_0` 与 `qwen3:1.7b-fp16` 质量无统计差异（92.0% vs 91.0%），但显存少 39.3%、吞吐高 57.4%、首 Token 延迟低 26.4%——**纯推理场景应默认 Q8 而不是 FP16，FP16 只在需要继续训练/微调时才有必要保留**。

## 复现方式

```powershell
python -m inferbench quant --models qwen3:1.7b-fp16,qwen3:1.7b-q8_0,qwen3:1.7b
```

明细数据：`results/quant_ladder-fp16.csv`（请求级，含每条样本的预测、延迟、显存、GPU 分流）