# 实验 1 · 量化档位对比报告

- 生成时间：2026-09-21 12:26:43
- **适用范围**：NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB） · 驱动 616.92 · CUDA 13.4 · Ollama 0.32.15 · OLLAMA_NUM_PARALLEL=1（客户端环境变量） · Python 3.13.12 · inferbench 0.3.0+fdf6d47
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU` · 显存 8188 MiB · 驱动 616.92 · CUDA 13.4 · 算力 8.9
  - 运行时：Ollama `0.32.15` @ `http://127.0.0.1:11434` · `OLLAMA_NUM_PARALLEL=1`（客户端环境变量；服务端探测不到，若服务端在别处运行请以服务端为准）
  - 代码版本：inferbench `0.3.0` · git `fdf6d47` · Python 3.13.12 · 指纹采集于 2026-09-21 12:26:43
- 语料：data/example_corpus.jsonl · 15 条（难例 3）· 指纹 `660f6dca`

- 评测集：15 条（难例 3 条）
- 测量口径：num_ctx=4096 · temperature=0.0 · think=False · seed=42 · few-shot=3 · 重复 1 次取中位数 · 预热 1 次后丢弃

## 三角表（显存 ↓ / 速度 ↑ / 精度 ↓）

| 模型 | 显存 (GB) | 准确率 | 难例准确率 | 非法输出率 | TTFT 中位 (ms) | 解码 (tok/s) | P95 端到端 (ms) | 设备 |
|---|---|---|---|---|---|---|---|---|
| `qwen3:1.7b-q8_0` | 2.24 | 86.7% | 66.7% | 0.0% | 15.5 | 124.6 | 87.9 | 100% GPU |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="280" viewBox="0 0 720 280" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="280" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">量化档位 → 显存占用</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">GB</text>
<line x1="56" y1="224.0" x2="704" y2="224.0" stroke="#E6E8EE"/>
<text x="48" y="228.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="178.0" x2="704" y2="178.0" stroke="#E6E8EE"/>
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">1</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">1</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">2</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">3</text>
<rect x="146.7" y="64.0" width="464.6" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="379.0" y="60.0" font-size="9" fill="#444" text-anchor="middle">2.2</text>
<text x="380.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
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
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">36</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">72</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">108</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">143</text>
<rect x="146.7" y="64.0" width="231.3" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="262.4" y="60.0" font-size="9" fill="#444" text-anchor="middle">124.6</text>
<rect x="380.0" y="204.1" width="231.3" height="19.9" fill="#00A37A" rx="2"/>
<text x="495.6" y="200.1" font-size="9" fill="#444" text-anchor="middle">15.5</text>
<text x="380.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
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
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">25</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">50</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">75</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">100</text>
<rect x="146.7" y="64.0" width="231.3" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="262.4" y="60.0" font-size="9" fill="#444" text-anchor="middle">86.7%</text>
<rect x="380.0" y="100.9" width="231.3" height="123.1" fill="#00A37A" rx="2"/>
<text x="495.6" y="96.9" font-size="9" fill="#444" text-anchor="middle">66.7%</text>
<text x="380.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="56" y="260" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="268" font-size="10.5" fill="#444">整体准确率 (%)</text>
<rect x="141" y="260" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="155" y="268" font-size="10.5" fill="#444">难例准确率 (%)</text>
</svg>

## 关键结论（数字自动取自本次运行）

1. **精度代价**：以 `qwen3:1.7b-q8_0`（本次最高精度档）为基准，`qwen3:1.7b-q8_0` 准确率 86.7% vs 86.7%，掉幅 0.0 个百分点；难例子集掉幅 0.0 个百分点（难例 3 题，按唯一题目去重）。
2. **显存代价**：2.24 GB → 2.24 GB，降幅 0.0%。
3. **速度收益**：解码 124.6 → 124.6 tok/s，提升 0.0%；首 Token 延迟 15.5 → 15.5 ms。
4. **失效模式（看混淆矩阵，不做想当然的归因）**：`qwen3:1.7b-q8_0` 的主要错误形态 —— summarize → knowledge_retrieval 错 1 次、knowledge_retrieval → small_talk 错 1 次；非法输出率 0.0% vs 基准 0.0%（两档非法输出率差异在噪声范围内，量化损伤主要体现在分类混淆上）


## 测量卫生（为什么这些速度数字可信）

| 模型 | 测量前 GPU 显存 | 模型占用 | GPU 分流 |
|---|---|---|---|
| `qwen3:1.7b-q8_0` | 4482/8188 MiB | 2.24 GB (VRAM 2.24 GB) | 100% GPU |

> **踩过的坑（面试可讲）**：Ollama 默认让最近用过的模型常驻显存。首次跑多档量化时，多个模型同时占用 8GB 显存（实测 7836/8188 MiB），推理被迫退到 CPU/共享内存，解码速度从 **168 tok/s 掉到 0.59 tok/s（差 286 倍）**。现在每次测量前先 `unload_all()` 清空显存并记录 GPU 基线，再用 `size_vram / size` 校验模型确实 100% 在 GPU 上——否则测出来的「速度差异」是显存竞争，不是量化收益。

## 可写进简历的结论句（数字都是本次真实测量值）

> **量化-显存-精度三方权衡实验**：固定 num_ctx=4096、关闭思维链、temperature=0、每档重复 1 次取中位数的统一口径下，对 qwen3:1.7b-q8_0 做 A/B：`qwen3:1.7b-q8_0` 相比 `qwen3:1.7b-q8_0` 显存降低 0.0%、解码吞吐提升 0.0%，3 分类准确率由 86.7% 降至 86.7%（模糊难例 66.7% → 66.7%）；并定位了首次测量中「多模型争抢显存导致解码掉 286 倍」的失真，固化为「测量前清空显存 + 校验 GPU 分流」的实验规范。

## 复现方式

```powershell
python -m inferbench quant --models qwen3:1.7b-q8_0
```

明细数据：`results/quant_owncorpus.csv`（请求级，含每条样本的预测、延迟、显存、GPU 分流）