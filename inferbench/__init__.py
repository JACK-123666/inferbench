"""inferbench：本地大模型推理实验台。

统一入口是 `python -m inferbench`（见 inferbench/cli.py）。包内分工：

    ollama.py   Ollama HTTP 客户端（stdlib）+ 指标提取 + 显存治理
    llama.py    llama.cpp 集成：GGUF 路径解析 + llama-server 管理（投机解码）
    gpu.py      nvidia-smi 采集 + 测量卫生检查
    cache.py    语义缓存核心（实验二/三共用）
    loadgen.py  并发压测（httpx + asyncio，实验五）
    bench.py    通用测量执行器（清显存 → 预热 → 重复 → 记录）
    report.py   报告框架；svg.py 内联图表
    tasks.py    任务定义（意图分类）；eval_set.py 评测集；stats.py 统计与落盘
"""

__version__ = "0.3.0"

