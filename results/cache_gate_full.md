# 实验 3b · 缓存写入门槛（自一致性）验证报告

- 生成时间：2026-09-18 09:34:43
- 请求流：300 条请求 / 230 条唯一文本
- 采样：每条唯一文本采样 **3 次**（temperature=0.7）
- **采样一致率：98.7%**（227/230 条文本的 3 次采样完全一致）← 这个数字决定了门槛有没有可能起作用
- 阈值固定为 **0.92**，唯一变量是「缓存写入门槛」

## A/B 对照

| 指标 | 门槛关闭（单次调用+总是写入） | 门槛开启（多数票+一致才写入） | 变化 |
|---|---|---|---|
| 命中率 | 42.3% | 41.7% | |
| **误命中率** | 14.2% | 12.8% | |
| 误命中次数 | 18 次 | 16 次 | |
| ├ A 类（继承后端错误） | 18 | 16 | |
| └ B 类（跨意图误匹配） | 0 | 0 | |
| 缓存条目数 | 173 | 172 | |
| 跳过写入次数 | 0 | 3 | |
| 端到端准确率 | 86.3% | 86.7% | |
| 无缓存基线准确率 | 86.3% | 86.3% | |
| 延迟中位数 | 89 ms | 165 ms | |
| 延迟 P95 | 127 ms | 223 ms | |
| 成本节省 | 42.4% | -74.7% | |

<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300" font-family="Segoe UI, Microsoft YaHei, sans-serif">
<rect width="720" height="300" fill="#ffffff"/>
<text x="56" y="22" font-size="14" font-weight="600" fill="#111">自一致性门槛 A/B 对照</text>
<text x="704" y="22" font-size="11" fill="#666" text-anchor="end">%</text>
<line x1="56" y1="244.0" x2="704" y2="244.0" stroke="#E6E8EE"/>
<text x="48" y="248.0" font-size="10" fill="#888" text-anchor="end">0</text>
<line x1="56" y1="193.0" x2="704" y2="193.0" stroke="#E6E8EE"/>
<text x="48" y="197.0" font-size="10" fill="#888" text-anchor="end">25</text>
<line x1="56" y1="142.0" x2="704" y2="142.0" stroke="#E6E8EE"/>
<text x="48" y="146.0" font-size="10" fill="#888" text-anchor="end">50</text>
<line x1="56" y1="91.0" x2="704" y2="91.0" stroke="#E6E8EE"/>
<text x="48" y="95.0" font-size="10" fill="#888" text-anchor="end">75</text>
<line x1="56" y1="40.0" x2="704" y2="40.0" stroke="#E6E8EE"/>
<text x="48" y="44.0" font-size="10" fill="#888" text-anchor="end">100</text>
<rect x="78.7" y="215.0" width="56.3" height="29.0" fill="#2E5BFF" rx="2"/>
<text x="106.8" y="211.0" font-size="9" fill="#444" text-anchor="middle">14.2%</text>
<rect x="137.0" y="217.8" width="56.3" height="26.2" fill="#00A37A" rx="2"/>
<text x="165.2" y="213.8" font-size="9" fill="#444" text-anchor="middle">12.8%</text>
<text x="137.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">误命中率 (%)</text>
<rect x="240.7" y="67.3" width="56.3" height="176.7" fill="#2E5BFF" rx="2"/>
<text x="268.8" y="63.3" font-size="9" fill="#444" text-anchor="middle">86.3%</text>
<rect x="299.0" y="66.6" width="56.3" height="177.4" fill="#00A37A" rx="2"/>
<text x="327.2" y="62.6" font-size="9" fill="#444" text-anchor="middle">86.7%</text>
<text x="299.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">端到端准确率 (%)</text>
<rect x="402.7" y="157.4" width="56.3" height="86.6" fill="#2E5BFF" rx="2"/>
<text x="430.8" y="153.4" font-size="9" fill="#444" text-anchor="middle">42.3%</text>
<rect x="461.0" y="158.7" width="56.3" height="85.3" fill="#00A37A" rx="2"/>
<text x="489.2" y="154.7" font-size="9" fill="#444" text-anchor="middle">41.7%</text>
<text x="461.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">命中率 (%)</text>
<rect x="564.7" y="157.1" width="56.3" height="86.9" fill="#2E5BFF" rx="2"/>
<text x="592.8" y="153.1" font-size="9" fill="#444" text-anchor="middle">42.4%</text>
<rect x="623.0" y="396.9" width="56.3" height="0.5" fill="#00A37A" rx="2"/>
<text x="651.2" y="392.9" font-size="9" fill="#444" text-anchor="middle">-74.7%</text>
<text x="623.0" y="262" font-size="10.5" fill="#333" text-anchor="middle">成本节省 (%)</text>
<rect x="56" y="280" width="9" height="9" fill="#2E5BFF" rx="2"/>
<text x="70" y="288" font-size="10.5" fill="#444">门槛关闭</text>
<rect x="106" y="280" width="9" height="9" fill="#00A37A" rx="2"/>
<text x="120" y="288" font-size="10.5" fill="#444">门槛开启</text>
</svg>

## 结论（自动生成）

**收益侧**：误命中 18 → 16 次（少 2 次），误命中率 14.2% → 12.8%（-1.4 个百分点）；端到端准确率 86.3% → 86.7%（+0.3 个百分点）。
**代价侧**：成本节省 42.4% → -74.7%（-117.1 个百分点，即**从省成本变成亏成本**）；延迟中位数 89 → 165 ms（**1.8 倍**）。

### 判定：**门槛无效，不建议采用**

1. 收益只在噪声级别（少 2 次误命中），而代价是从省 42.4% 成本变成亏 74.7%、延迟 1.8 倍。
2. **根因在采样一致率**：227/230 条文本（98.7%）在 temperature=0.7 下采样 3 次完全一致——门槛总共只挡住了 1 次写入。
3. 也就是说：**这个模型对错的题是「稳定地错」，不是「随机地错」**。自一致性过滤只在「模型自己会犹豫」时才起作用，对这种系统性偏差**在原理上就无效**。
4. 这个负结果把改进方向钉死了：要治系统性偏差，只能从**换更强的后端模型**、**对易错意图单独处理（命中后二次校验 / 干脆绕过缓存）**入手，而不是继续在缓存层加花样。

## 面试怎么讲这个结果

> 「实验 3 我发现误命中 100% 来自「继承后端模型自己的错误」，于是加了自一致性门槛——同一条 query 采样 3 次、结果一致才允许写入缓存。结果：227/230 条文本采样 3 次完全一致，门槛只挡住了 1 次写入，误命中只少 2 次，成本却从省 42.4% 变成亏 74.7%。**结论是这个模型「稳定地错」，自一致性只能治随机性错误。这个负结果让我把改进方向改到后端模型质量和意图级策略上，而不是继续在缓存层加花活。**」

明细数据：`results/cache_gate_full.csv`