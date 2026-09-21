"""环境指纹：让每份结果自带「适用范围」。

为什么必须有（由这个工程的定位决定）：
    结果里的每个数字都绑定硬件与运行时 ——「8GB 卡上草稿模型全负收益」「开 4 路并行显存 +86%」
    都只在特定的显存 / 驱动 / Ollama 版本下成立。如果结果文件里不记硬件身份，别人（包括三个月后的
    自己）拿到一份历史结果时，无法判断它适不适用自己的机器 —— 只能去翻 README 里手写的环境说明，
    而手写说明与结果文件之间没有任何绑定。

本模块的纪律：
    - **测不到就留空，绝不猜、绝不抛异常**（采集失败不能把实验搞崩）；
    - 老结果没有 ``env`` 块时，报告显式写「未记录、适用边界未知」，不做任何推测性补全。
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from inferbench import config
from inferbench import __version__

_SMI = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"

# 表头写法随驱动版本变化：老驱动是 `CUDA Version: 13.4`，616.92 驱动改成 `CUDA UMD Version: 13.4`。
# 只认前者会**永远采不到 CUDA**（本机实测踩到），所以这里两种都认。
_CUDA_RE = re.compile(r"CUDA(?:\s+[A-Za-z]+)?\s+Version:\s*([\d.]+)")


def cuda_version_from_header(text: str) -> str:
    """从 nvidia-smi 默认输出的表头里取 CUDA 版本；取不到返回空串。"""
    m = _CUDA_RE.search(text or "")
    return m.group(1) if m else ""



def _run(cmd: list[str], timeout: float = 15.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _num(text: str) -> float:
    m = re.search(r"[\d.]+", text or "")
    return float(m.group(0)) if m else 0.0


def gpu_identity() -> dict:
    """GPU 硬件身份：型号 / 显存总量 / 驱动 / CUDA / 算力。无 N 卡时返回 {}。"""
    if not (shutil.which(_SMI) or Path(_SMI).exists()):
        return {}
    line = _run([_SMI, "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"])
    if not line:
        return {}
    parts = [p.strip() for p in line.splitlines()[0].split(",")]
    if len(parts) < 3:
        return {}
    ident = {
        "name": parts[0],
        "memory_total_mb": _num(parts[1]),
        "driver": parts[2],
    }
    if len(parts) >= 4:
        ident["compute_cap"] = parts[3]
    # CUDA 版本只出现在 nvidia-smi 的默认输出表头里，查询模式不提供
    cuda = cuda_version_from_header(_run([_SMI]))
    if cuda:
        ident["cuda"] = cuda
    return ident


def ollama_identity(host: str | None = None) -> dict:
    """Ollama 版本 + 实际访问的 host + 并行度。

    ``num_parallel_env`` 是**客户端进程的环境变量**，不是对服务端的探测：
    Ollama 不暴露该配置，所以只有当服务端跑在同一环境时才等价（README §6 记过这个坑）。
    """
    host = (host or config.OLLAMA_HOST).rstrip("/")
    version = ""
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=3.0) as resp:
            version = str(json.loads(resp.read().decode("utf-8")).get("version", ""))
    except Exception:  # noqa: BLE001
        version = ""
    return {
        "version": version or "unknown",
        "host": host,
        "num_parallel_env": config.ollama_num_parallel(),
    }


def code_identity() -> dict:
    """产出这份结果的代码版本：包版本 + git 短 SHA（仓库里没有 git 就留空）。"""
    sha = _run(["git", "-C", str(config.ROOT), "rev-parse", "--short", "HEAD"], timeout=5.0)
    return {"inferbench": __version__, "git": sha}


def scope_line(env: dict) -> str:
    """一句话「适用范围」—— 报告开头就印它。"""
    gpu = env.get("gpu") or {}
    oll = env.get("ollama") or {}
    code = env.get("code") or {}
    bits: list[str] = []
    if gpu.get("name"):
        bits.append(f"{gpu['name']}（{_num(str(gpu.get('memory_total_mb', 0))):.0f} MiB）")
    else:
        bits.append("无 N 卡或 nvidia-smi 不可用")
    if gpu.get("driver"):
        bits.append(f"驱动 {gpu['driver']}")
    if gpu.get("cuda"):
        bits.append(f"CUDA {gpu['cuda']}")
    bits.append(f"Ollama {oll.get('version', 'unknown')}")
    bits.append(f"OLLAMA_NUM_PARALLEL={oll.get('num_parallel_env', '?')}（客户端环境变量）")
    bits.append(f"Python {env.get('python', '?')}")
    if code:
        tag = str(code.get("inferbench", "?"))
        if code.get("git"):
            tag += f"+{code['git']}"
        bits.append(f"inferbench {tag}")
    return " · ".join(bits)


def fingerprint(host: str | None = None) -> dict:
    """采集一次完整指纹。永不抛异常：测不到的字段就是空的。"""
    env = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_identity(),
        "ollama": ollama_identity(host),
        "code": code_identity(),
    }
    env["scope"] = scope_line(env)
    return env


def report_lines(payload: dict) -> list[str]:
    """报告头部的「适用范围」段（Markdown 行）。

    老结果没有 ``env`` 块 → 返回一条**明确的未记录声明**，而不是悄悄省略：
    省略会让人误以为该结果没有适用边界问题。
    """
    env = payload.get("env")
    if not env:
        return ["- **适用范围**：⚠ 本次运行早于环境指纹功能（结果里没有 `env` 块），"
                "硬件与代码版本均未记录 —— **该结果的适用边界未知**，不要直接跨机器套用"]

    lines = [f"- **适用范围**：{env.get('scope', '未记录')}"]
    gpu = env.get("gpu") or {}
    oll = env.get("ollama") or {}
    code = env.get("code") or {}
    if gpu.get("name"):
        lines.append(f"  - GPU：`{gpu['name']}` · 显存 {_num(str(gpu.get('memory_total_mb', 0))):.0f} MiB · "
                     f"驱动 {gpu.get('driver', '未记录')} · CUDA {gpu.get('cuda', '未记录')} · "
                     f"算力 {gpu.get('compute_cap', '未记录')}")
    else:
        lines.append("  - GPU：未记录（nvidia-smi 不可用）—— 显存相关结论的适用边界未知")
    # 下面两行**无条件输出**（缺的字段写"未记录"）：省略会让人以为没这回事，
    # 而"哪一项没采到"本身就是适用范围的一部分。
    lines.append(f"  - 运行时：Ollama `{oll.get('version', '未记录')}` @ `{oll.get('host', '未记录')}` · "
                 f"`OLLAMA_NUM_PARALLEL={oll.get('num_parallel_env', '未记录')}`"
                 f"（客户端环境变量；服务端探测不到，若服务端在别处运行请以服务端为准）")
    version = (f"inferbench `{code['inferbench']}`" if code.get("inferbench")
               else "inferbench `未记录`")
    if code.get("git"):
        version += f" · git `{code['git']}`"
    lines.append(f"  - 代码版本：{version} · Python {env.get('python', '未记录')} · "
                 f"指纹采集于 {env.get('captured_at', '未记录')}")
    return lines
