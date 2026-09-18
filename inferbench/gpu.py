"""GPU 状态采集（测量卫生）。

为什么必须做这件事（本机实测教训）：
    Ollama 默认会把最近用过的模型常驻显存（keep_alive），跑多档量化时很容易出现
    **多个模型同时占着显存**，把 8GB 打满（实测 7836/8188 MiB）。显存打满后
    推理会退化到 CPU/共享内存，解码速度从 150 tok/s 掉到 0.6 tok/s——**差了 250 倍**。
    如果不记录 GPU 基线，量化对比出来的"速度差异"其实测的是显存竞争。

因此每次测量都会记录：
    - 测量前的显存基线（判断是否有别的进程抢占）
    - 测量中该模型的显存占用与 GPU 分流比例
    - 是否有其它模型同时常驻（unload_all 之后应当为 0）
"""
from __future__ import annotations

import shutil
import subprocess

_SMI = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"


def available() -> bool:
    return bool(_SMI) and shutil.which(_SMI) is not None


def state() -> dict:
    """返回 {mem_used_mb, mem_total_mb, util, clock_sm, power_w, temp_c}；失败返回空 dict。"""
    if not available():
        return {}
    try:
        out = subprocess.run(
            [_SMI, "--query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        line = (out.stdout or "").strip().splitlines()
        if not line:
            return {}
        parts = [p.strip() for p in line[0].split(",")]
        return {
            "mem_used_mb": float(parts[0]),
            "mem_total_mb": float(parts[1]),
            "util": float(parts[2]),
            "clock_sm": float(parts[3]),
            "power_w": float(parts[4]),
            "temp_c": float(parts[5]),
        }
    except Exception:  # noqa: BLE001
        return {}


def describe(st: dict) -> str:
    if not st:
        return "nvidia-smi 不可用"
    return (f"{st['mem_used_mb']:.0f}/{st['mem_total_mb']:.0f} MiB "
            f"({st['mem_used_mb'] / max(st['mem_total_mb'], 1) * 100:.0f}%) · "
            f"util {st['util']:.0f}% · SM {st['clock_sm']:.0f}MHz · "
            f"{st['power_w']:.1f}W · {st['temp_c']:.0f}℃")


def hygiene_warning(baseline: dict, footprint_gb: float) -> str:
    """给出环境是否适合测量的判定。"""
    if not baseline:
        return ""
    free_gb = (baseline["mem_total_mb"] - baseline["mem_used_mb"]) / 1024
    if free_gb < footprint_gb * 1.5:
        return (f"⚠ 显存可能不足：可用 {free_gb:.2f} GB，模型需约 {footprint_gb:.2f} GB"
                f"（建议关掉 Wallpaper Engine / 抖音 / 浏览器等吃显存的程序）")
    return ""
