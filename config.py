"""inferbench 全局配置。

设计原则：整个工程**零第三方依赖**（只用 Python 标准库），
这样任何一台装了 Python 3.10+ 的机器都能直接跑，不折腾环境。
"""
import os
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Synapse 开源工程（复用它的 100 条评测集）
# 换机器时用环境变量覆盖：INFERBENCH_SYNAPSE_DIR
SYNAPSE_DIR = Path(os.environ.get("INFERBENCH_SYNAPSE_DIR", r"D:\PYTHON\Synapse"))
SYNAPSE_TESTSET = SYNAPSE_DIR / "tools" / "intent_testset.py"

# ---------- Ollama ----------
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 300.0      # 单次请求超时（秒）
OLLAMA_RETRIES = 2          # 失败重试次数（指数退避）
KEEP_ALIVE = "5m"           # 模型常驻时间：太长会让多档量化互相争显存（实测会掉 250 倍速度）

# ---------- 测量口径（**必须锁死**，否则数字不可比） ----------
NUM_CTX = 4096              # 上下文长度：默认 131072 会让 KV cache 吃满显存
EMBED_NUM_CTX = 512         # embedding 模型的上下文：默认 32768 会让 0.6GB 的模型占 5.8GB 显存
TEMPERATURE = 0.0
SEED = 42
NUM_PREDICT = 16            # 只要标签，限制输出长度
REPEATS = 3                 # 重复次数，取中位数
SHOTS = 3                   # few-shot 示例数（所有模型完全相同）
WARMUP_CALLS = 1            # 预热调用次数（丢弃，不计入统计）

# ---------- 实验 1：量化对比 ----------
# 默认跑已有的三档：Q4_K_M / Q8_0 / 小模型对照
QUANT_MODELS = [
    "qwen3:0.6b",        # Q4_K_M  0.49GB
    "qwen3:1.7b",        # Q4_K_M  1.27GB
    "qwen3:1.7b-q8_0",   # Q8_0    2.07GB
    # "qwen3:1.7b-fp16", # FP16    约 3.8GB（需要时再拉，见 README）
]

# ---------- 实验 3：语义缓存 ----------
EMBED_MODEL = "qwen3-embedding:0.6b"
CACHE_MODEL = "qwen3:1.7b"          # 缓存后端（未命中时真正干活的模型）
CACHE_THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
PARAPHRASES_PER_QUERY = 2           # 每条 query 生成几个语义近义改写
CACHE_TTL_SECONDS = 3600            # 缓存 TTL
CACHE_TOP_K = 1                     # 取 top-1 近似即命中

# ---------- 成本折算（用于"省了多少钱"的估算，单位可改） ----------
# 以 qwen-turbo 量级刊例价估算（元 / 1K tokens），**以控制台账单为准**
PRICE_PER_1K_PROMPT_CNY = 0.0003
PRICE_PER_1K_OUTPUT_CNY = 0.0006


# ---------- 运行时环境探测 ----------
def ollama_num_parallel() -> int:
    """读取 Ollama 的并行度（`OLLAMA_NUM_PARALLEL`），未设置则为默认值 1。

    这个值直接决定并发压测的结果形态：
    - =1（默认）：所有请求挤在**同一个 slot** 上串行执行 → 并发只增加排队，
      表现为「吞吐不涨、TTFT 随并发线性增长」
    - >1：才可能出现真正的并行计算，但每路并行都要一份 KV cache 显存
    """
    import os
    for scope in ("OLLAMA_NUM_PARALLEL",):
        val = os.environ.get(scope)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


