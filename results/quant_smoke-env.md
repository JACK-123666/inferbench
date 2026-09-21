# 实验 1 · 量化档位对比报告

- 生成时间：2026-09-21 12:15:45
- **适用范围**：NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB） · 驱动 616.92 · CUDA 13.4 · Ollama 0.32.15 · OLLAMA_NUM_PARALLEL=1（客户端环境变量） · Python 3.13.12 · inferbench 0.3.0+e6469a8
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU` · 显存 8188 MiB · 驱动 616.92 · CUDA 13.4 · 算力 8.9
  - 运行时：Ollama `0.32.15` @ `http://127.0.0.1:11434` · `OLLAMA_NUM_PARALLEL=1`（客户端环境变量；服务端探测不到，若服务端在别处运行请以服务端为准）
  - 代码版本：inferbench `0.3.0` · git `e6469a8` · Python 3.13.12 · 指纹采集于 2026-09-21 12:15:45

- 评测集：12 条（难例 0 条）
- 测量口径：num_ctx=4096 · temperature=0.0 · think=False · seed=42 · few-shot=3 · 重复 1 次取中位数 · 预热 1 次后丢弃

## 三角表（显存 ↓ / 速度 ↑ / 精度 ↓）

| 模型 | 显存 (GB) | 准确率 | 难例准确率 | 非法输出率 | TTFT 中位 (ms) | 解码 (tok/s) | P95 端到端 (ms) | 设备 |
|---|---|---|---|---|---|---|---|---|
| `qwen3:1.7b-q8_0` | 2.24 | 100.0% | 0.0% | 0.0% | 16.1 | 110.1 | 95.0 | 100% GPU |
| `qwen3:1.7b` | 1.59 | 100.0% | 0.0% | 0.0% | 13.5 | 157.5 | 73.0 | 100% GPU |

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
<rect x="101.4" y="64.0" width="231.3" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="217.0" y="60.0" font-size="9" fill="#444" text-anchor="middle">2.2</text>
<text x="218.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="425.4" y="110.8" width="231.3" height="113.2" fill="#2E5BFF" rx="2"/>
<text x="541.0" y="106.8" font-size="9" fill="#444" text-anchor="middle">1.6</text>
<text x="542.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b</text>
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
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">45</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">91</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">136</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">181</text>
<rect x="101.4" y="112.2" width="114.6" height="111.8" fill="#2E5BFF" rx="2"/>
<text x="158.7" y="108.2" font-size="9" fill="#444" text-anchor="middle">110.1</text>
<rect x="218.0" y="207.6" width="114.6" height="16.4" fill="#00A37A" rx="2"/>
<text x="275.3" y="203.6" font-size="9" fill="#444" text-anchor="middle">16.1</text>
<text x="218.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="425.4" y="64.0" width="114.6" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="482.7" y="60.0" font-size="9" fill="#444" text-anchor="middle">157.5</text>
<rect x="542.0" y="210.3" width="114.6" height="13.7" fill="#00A37A" rx="2"/>
<text x="599.3" y="206.3" font-size="9" fill="#444" text-anchor="middle">13.5</text>
<text x="542.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b</text>
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
<text x="48" y="182.0" font-size="10" fill="#888" text-anchor="end">29</text>
<line x1="56" y1="132.0" x2="704" y2="132.0" stroke="#E6E8EE"/>
<text x="48" y="136.0" font-size="10" fill="#888" text-anchor="end">57</text>
<line x1="56" y1="86.0" x2="704" y2="86.0" stroke="#E6E8EE"/>
<text x="48" y="90.0" font-size="10" fill="#888" text-anchor="end">86</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">115</text>
<rect x="101.4" y="64.0" width="114.6" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="158.7" y="60.0" font-size="9" fill="#444" text-anchor="middle">100.0%</text>
<rect x="218.0" y="224.0" width="114.6" height="0.5" fill="#00A37A" rx="2"/>
<text x="275.3" y="220.0" font-size="9" fill="#444" text-anchor="middle">0.0%</text>
<text x="218.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b-q8_0</text>
<rect x="425.4" y="64.0" width="114.6" height="160.0" fill="#2E5BFF" rx="2"/>
<text x="482.7" y="60.0" font-size="9" fill="#444" text-anchor="middle">100.0%</text>
<rect x="542.0" y="224.0" width="114.6" height="0.5" fill="#00A37A" rx="2"/>
<text x="599.3" y="220.0" font-size="9" fill="#444" text-anchor="middle">0.0%</text>
<text x="542.0" y="242" font-size="10.5" fill="#333" text-anchor="middle">qwen3:1.7b</text>
<rect x="56" y="260" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="268" font-size="10.5" fill="#444">整体准确率 (%)</text>
<rect x="141" y="260" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="155" y="268" font-size="10.5" fill="#444">难例准确率 (%)</text>
</svg>

## 关键结论（数字自动取自本次运行）

1. **精度代价**：以 `qwen3:1.7b-q8_0`（本次最高精度档）为基准，`qwen3:1.7b` 准确率 100.0% vs 100.0%，掉幅 0.0 个百分点；难例子集掉幅 0.0 个百分点（难例 0 题，按唯一题目去重）。
2. **显存代价**：2.24 GB → 1.59 GB，降幅 29.3%。
3. **速度收益**：解码 110.1 → 157.5 tok/s，提升 43.1%；首 Token 延迟 16.1 → 13.5 ms。
4. **失效模式（看混淆矩阵，不做想当然的归因）**：`qwen3:1.7b` 的主要错误形态 —— 本次没有错误样本（或样本量太小）；非法输出率 0.0% vs 基准 0.0%（两档非法输出率差异在噪声范围内，量化损伤主要体现在分类混淆上）


## 测量卫生（为什么这些速度数字可信）

| 模型 | 测量前 GPU 显存 | 模型占用 | GPU 分流 |
|---|---|---|---|
| `qwen3:1.7b-q8_0` | 2143/8188 MiB | 2.24 GB (VRAM 2.24 GB) | 100% GPU |
| `qwen3:1.7b` | 2100/8188 MiB | 1.59 GB (VRAM 1.59 GB) | 100% GPU |

> **踩过的坑（面试可讲）**：Ollama 默认让最近用过的模型常驻显存。首次跑多档量化时，多个模型同时占用 8GB 显存（实测 7836/8188 MiB），推理被迫退到 CPU/共享内存，解码速度从 **168 tok/s 掉到 0.59 tok/s（差 286 倍）**。现在每次测量前先 `unload_all()` 清空显存并记录 GPU 基线，再用 `size_vram / size` 校验模型确实 100% 在 GPU 上——否则测出来的「速度差异」是显存竞争，不是量化收益。

## 可写进简历的结论句（数字都是本次真实测量值）

> **量化-显存-精度三方权衡实验**：固定 num_ctx=4096、关闭思维链、temperature=0、每档重复 1 次取中位数的统一口径下，对 qwen3:1.7b-q8_0 / qwen3:1.7b 做 A/B：`qwen3:1.7b` 相比 `qwen3:1.7b-q8_0` 显存降低 29.3%、解码吞吐提升 43.1%，3 分类准确率由 100.0% 降至 100.0%（模糊难例 0.0% → 0.0%）；并定位了首次测量中「多模型争抢显存导致解码掉 286 倍」的失真，固化为「测量前清空显存 + 校验 GPU 分流」的实验规范。

## 复现方式

```powershell
python -m inferbench quant --models qwen3:1.7b-q8_0,qwen3:1.7b
```

明细数据：`results/quant_smoke-env.csv`（请求级，含每条样本的预测、延迟、显存、GPU 分流）