"""统一命令入口：`python -m inferbench <命令>`。

设计意图：这个工程有 5 个实验 + 若干工具，散成 5 个脚本会让"从哪开始"变成一个问题。
所以这里做一个薄薄的路由层——**每个子命令背后都是 `inferbench/experiments/*.py` 里的 `run(argv)`**，
CLI 只负责分发、帮助和参数透传，不塞业务逻辑。
"""
from __future__ import annotations

import subprocess
import sys

from inferbench import gpu, llama
from inferbench.ollama import Ollama

# 命令表：名称 → (模块, 一句话说明, 示例)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "env":    ("", "环境体检：显卡 / Ollama / 模型 / 依赖 / 测量口径",
               "python -m inferbench env"),
    "dataset": ("inferbench.dataset", "语料体检：接自己的数据（格式 / 可测性 / 标签对齐）",
                "python -m inferbench dataset data/example_corpus.jsonl"),
    "quant":  ("inferbench.experiments.quant", "实验一 · 量化档位对比（显存↓ / 速度↑ / 精度↓）",
               "python -m inferbench quant --models qwen3:1.7b-fp16,qwen3:1.7b-q8_0,qwen3:1.7b"),
    "cache":  ("inferbench.experiments.cache_exp", "实验二 · 语义缓存（阈值扫描 + 误命中归因）",
               "python -m inferbench cache --paraphrases 2"),
    "gate":   ("inferbench.experiments.gate", "实验三 · 缓存写入门槛（自一致性 A/B，负结果）",
               "python -m inferbench gate --n 3 --threshold 0.92"),
    "spec":   ("inferbench.experiments.spec", "实验四 · 投机解码（draft 模型 vs n-gram）",
               "python -m inferbench spec --configs base,draft-k8,ngram"),
    "load":   ("inferbench.experiments.load", "实验五 · 并发压测（QPS / 尾延迟 / TTFT）",
               "python -m inferbench load --levels 1,2,4,8,16 --requests 32"),
    "recommend": ("inferbench.recommend", "收敛 · 选型建议：读结果文件算出「本机该怎么配 + 代价」",
                  "python -m inferbench recommend --vram 8 --slo-ttft 200 --concurrency 4"),
    "report": ("inferbench.experiments.report_cmd", "重出报告：用旧 JSON 重新生成 Markdown（不重跑实验）",
               "python -m inferbench report results/spec_full.json"),
    "test":   ("", "跑单元测试（pytest）", "python -m inferbench test"),
    "models": ("", "列出 Ollama 模型及其对应的 GGUF 文件路径",
               "python -m inferbench models"),
}


def _print_help() -> None:
    print("inferbench · 本地大模型推理实验台\n")
    print("用法: python -m inferbench <命令> [参数]\n")
    width = max(len(k) for k in COMMANDS)
    for name, (_, desc, example) in COMMANDS.items():
        print(f"  {name:<{width}}  {desc}")
    print("\n示例:\n")
    for name, (_, _, example) in COMMANDS.items():
        print(f"  {example}")
    print("\n加 --help 看某个命令的全部参数，例如：python -m inferbench load --help")


def cmd_env(argv: list[str]) -> int:
    """环境体检：一眼看清这台机器能不能跑、跑之前要确认什么。"""
    from inferbench import tasks
    from inferbench import config

    print("=" * 78)
    print("环境体检")
    print("=" * 78)
    state = gpu.state()
    print(f"GPU        : {gpu.describe(state) if state else 'nvidia-smi 不可用（无法做量化对比）'}")
    print(f"模型目录    : {llama.models_dir()}")

    cli = Ollama()
    print(f"Ollama     : {cli.version()} @ {config.OLLAMA_HOST}")
    models = cli.list_models()
    print(f"已装模型    : {len(models)} 个")
    for m in models:
        name = m.get("name", "?")
        size = (m.get("size") or 0) / (1024 ** 3)
        quant = (m.get("details") or {}).get("quantization_level", "?")
        print(f"   - {name:<24} {size:>5.2f} GB  {quant}")
    loaded = cli.loaded_models()
    print(f"当前加载    : {', '.join(loaded) if loaded else '（无）'}"
          f"{'  ⚠ 多个模型会争抢显存，测速前请只留一个' if len(loaded) > 1 else ''}")

    info = llama.describe("qwen3:1.7b")
    print(f"GGUF 复用   : {info['path'] if info['exists'] else '未找到 qwen3:1.7b（投机解码实验需要）'}")

    print("\n依赖:")
    for mod, why in [("httpx", "并发压测（实验五）"), ("pytest", "单元测试")]:
        try:
            __import__(mod)
            print(f"   ✓ {mod:<8} {why}")
        except ImportError:
            print(f"   ✗ {mod:<8} 未安装 —— {why}：pip install {mod}")
    print("\n测量口径（写在 inferbench/config.py，所有实验共用）:")
    print(f"   num_ctx={config.NUM_CTX}  embed_ctx={config.EMBED_NUM_CTX}  "
          f"temperature={config.TEMPERATURE}  seed={config.SEED}  repeats={config.REPEATS}")
    print(f"   think=False（思维链模型必须关）  few-shot={config.SHOTS}  "
          f"预热后丢弃（冷加载 14s 不计入统计）")
    print("=" * 78)
    _ = tasks
    return 0


def cmd_test(argv: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *argv])


def cmd_models(argv: list[str]) -> int:
    cli = Ollama()
    print(f"{'模型':<26}{'体积':>9}  {'GGUF 路径'}")
    for m in cli.list_models():
        name = m.get("name", "?")
        size = (m.get("size") or 0) / (1024 ** 3)
        info = llama.describe(name)
        print(f"{name:<26}{size:>7.2f}GB  {info['path'] if info['exists'] else '(未找到)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"未知命令: {name}\n")
        _print_help()
        return 2

    if name == "env":
        return cmd_env(rest)
    if name == "test":
        return cmd_test(rest)
    if name == "models":
        return cmd_models(rest)

    module_name = COMMANDS[name][0]
    # 延迟导入：只加载要跑的那个实验，避免 import 全部依赖
    module = __import__(module_name, fromlist=["run"])
    return module.run(rest)

