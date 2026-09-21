"""环境指纹测试。

这个模块守的是一条**诚实性纪律**，不是功能：
    结果绑定硬件，所以每份结果必须能回答"读数是在什么机器/什么版本下测的"；
    测不到就写"未记录"，绝不用推测填坑——宁可显示适用边界未知，也不能让人误以为它通用。
"""
from __future__ import annotations

from pathlib import Path

from inferbench import fingerprint as fp, report as rep

FULL_ENV = {
    "captured_at": "2026-09-20 12:00:00",
    "python": "3.13.5",
    "platform": "Windows-11-10.0.26200-SP0",
    "gpu": {"name": "NVIDIA GeForce RTX 4060 Laptop GPU", "memory_total_mb": 8188.0,
            "driver": "616.92", "cuda": "13.4", "compute_cap": "8.9"},
    "ollama": {"version": "0.32.15", "host": "http://127.0.0.1:11434", "num_parallel_env": 1},
    "code": {"inferbench": "0.3.0", "git": "e6469a8"},
}


# ---------- 报告渲染 ----------

def test_report_lines_warns_when_env_missing():
    """老结果没有 env 块时必须**显式声明适用边界未知**，不能悄悄省略这一段。"""
    lines = fp.report_lines({"tag": "old", "created_at": "2026-01-01 00:00:00"})
    assert len(lines) == 1
    assert "适用边界未知" in lines[0]
    assert "未记录" in lines[0]


def test_report_lines_renders_gpu_runtime_and_code():
    env = dict(FULL_ENV, scope=fp.scope_line(FULL_ENV))
    lines = fp.report_lines({"env": env})
    text = "\n".join(lines)
    assert "NVIDIA GeForce RTX 4060 Laptop GPU" in text
    assert "8188 MiB" in text and "616.92" in text and "CUDA 13.4" in text
    assert "Ollama `0.32.15`" in text
    assert "git `e6469a8`" in text and "Python 3.13.5" in text


def test_report_lines_tolerates_partial_env():
    """只采到一半（比如没 N 卡）也要能出报告，且明确标出哪一项没记录。"""
    lines = fp.report_lines({"env": {"python": "3.13.5", "gpu": {}, "ollama": {}, "code": {}}})
    text = "\n".join(lines)
    assert "未记录（nvidia-smi 不可用）" in text
    assert "Python 3.13.5" in text


def test_scope_line_does_not_invent_gpu_when_missing():
    scope = fp.scope_line({"gpu": {}, "ollama": {}, "code": {}, "python": "3.13.5"})
    assert "无 N 卡或 nvidia-smi 不可用" in scope
    assert "RTX" not in scope and "NVIDIA" not in scope


def test_scope_line_marks_num_parallel_as_client_env():
    """并行度只能从客户端环境变量读，必须在文案里说清它不是服务端探测（README 记过这个坑）。"""
    scope = fp.scope_line(dict(FULL_ENV, scope=None))
    assert "OLLAMA_NUM_PARALLEL=1" in scope
    assert "客户端环境变量" in scope


# ---------- 采集 ----------

def test_fingerprint_returns_stable_keys():
    """真采一次：字段齐全、scope 非空，且**不抛异常**（没有 N 卡的机器也要能跑）。"""
    env = fp.fingerprint()
    for key in ("captured_at", "python", "platform", "gpu", "ollama", "code", "scope"):
        assert key in env, f"缺字段 {key}"
    assert isinstance(env["scope"], str) and env["scope"]
    assert env["code"]["inferbench"]        # 包版本必须拿得到


def test_gpu_identity_returns_empty_when_smi_missing(monkeypatch):
    monkeypatch.setattr(fp, "_SMI", r"C:\definitely\not\here\nvidia-smi.exe")
    assert fp.gpu_identity() == {}


def test_cuda_version_parses_both_header_variants():
    """回归：表头写法随驱动变化，两种都得认，否则 CUDA 永远采不到。"""
    old = "| NVIDIA-SMI 552.22    Driver Version: 552.22    CUDA Version: 12.4  |"
    new = "| NVIDIA-SMI 616.92    KMD Version: 616.92    CUDA UMD Version: 13.4  |"
    assert fp.cuda_version_from_header(old) == "12.4"
    assert fp.cuda_version_from_header(new) == "13.4"
    assert fp.cuda_version_from_header("no header here") == ""
    assert fp.cuda_version_from_header("") == ""


# ---------- 与报告打通 ----------

def test_quant_report_shows_scope_section(quant_payload, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rep.config, "RESULTS_DIR", tmp_path)
    quant_payload["env"] = dict(FULL_ENV, scope=fp.scope_line(FULL_ENV))
    text = Path(rep.build_quant_report(quant_payload)).read_text(encoding="utf-8")
    assert "**适用范围**" in text
    assert "NVIDIA GeForce RTX 4060 Laptop GPU" in text
    assert "git `e6469a8`" in text


def test_quant_report_degrades_without_env(quant_payload, tmp_path: Path, monkeypatch):
    """用现存 fixture（没有 env）跑一次：报告照常生成，并带上未记录声明。"""
    monkeypatch.setattr(rep.config, "RESULTS_DIR", tmp_path)
    assert "env" not in quant_payload
    text = Path(rep.build_quant_report(quant_payload)).read_text(encoding="utf-8")
    assert "适用边界未知" in text
    assert "## 三角表" in text          # 其余小节不受影响
