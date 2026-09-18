"""llama.cpp 集成：GGUF 路径解析 + llama-server 进程管理（用于投机解码实验）。

为什么用 Ollama 自带的 llama-server：
    Ollama 不暴露 draft 模型参数，没法开投机解码；但它自带的 `llama-server.exe`
    就是完整的 llama.cpp 构建，参数齐全——而且**不用额外下载任何东西**。
    另外 Ollama 的 blob 就是标准 GGUF（文件名是内容的 sha256），
    可以直接交给 llama.cpp 加载，省掉几 GB 流量。

四个工程前提（都是实测踩出来的，详见 README）：
    1. 路径带空格会被参数解析拆开 → Python 用列表传参
    2. CUDA 后端 DLL 的依赖找不到 → GGML_BACKEND_PATH 指向 DLL + 目录进 PATH
    3. Ollama 构建默认 --spec-type 为空 → 必须显式 --spec-type draft-simple
    4. 默认 4 路并行，而投机解码只支持单序列 → 必须 -np 1
"""
# ---------------------------------------------------------------------------
# GGUF 路径解析：把 `qwen3:0.6b` 这样的 tag 映射到磁盘上的 GGUF 文件
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path


def models_dir() -> Path:
    """定位 Ollama 的模型根目录（blobs/ 和 manifests/ 的父目录）。"""
    for scope in ("OLLAMA_MODELS",):
        val = os.environ.get(scope)
        if val:
            p = Path(val)
            # 有的安装把 OLLAMA_MODELS 指到程序目录（如 D:\Program Files\ollama）
            if (p / "manifests").exists() or (p / "blobs").exists():
                return p
            if (p / "models" / "manifests").exists():
                return p / "models"
    default = Path.home() / ".ollama" / "models"
    return default


def resolve(model_tag: str, root: Path | None = None) -> Path:
    """把 `qwen3:0.6b` 这样的 tag 解析成 GGUF 文件路径。找不到会抛 FileNotFoundError。"""
    root = root or models_dir()
    name, _, tag = model_tag.partition(":")
    tag = tag or "latest"
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest}")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    for layer in data.get("layers", []):
        if "image.model" in str(layer.get("mediaType", "")):
            blob = root / "blobs" / str(layer["digest"]).replace(":", "-")
            if not blob.exists():
                raise FileNotFoundError(f"blob 不存在: {blob}")
            return blob
    raise FileNotFoundError(f"{model_tag} 的 manifest 里没有模型层: {manifest}")


def describe(model_tag: str, root: Path | None = None) -> dict:
    """返回 {tag, path, size_gb, exists}，用于报告与日志。"""
    try:
        p = resolve(model_tag, root)
        return {"tag": model_tag, "path": str(p),
                "size_gb": round(p.stat().st_size / (1024 ** 3), 2), "exists": True}
    except FileNotFoundError as exc:
        return {"tag": model_tag, "path": "", "size_gb": 0.0, "exists": False, "error": str(exc)}


"""llama-server 进程管理器 —— 用于投机解码实验。

## 为什么不用 Ollama 做这个实验

Ollama **不暴露 draft 模型参数**，没法开投机解码。但它自带的
`llama-server.exe` 就是完整的 llama.cpp 构建，参数齐全——
而且**不用额外下载任何东西**（GitHub release 里本环境只有个 txt）。

## 踩过的四个坑（都在这里解决了）

1. **路径带空格**：`D:\\Program Files\\...` 会被参数解析拆开。
   → 用列表传参（subprocess），不走 shell 拼接。
2. **CUDA 后端加载失败**：backend DLL 在子目录 `cuda_v12\\`，
   且它的依赖（cublas64_12.dll 等）找不到。
   → `GGML_BACKEND_PATH` 指向 **DLL 文件本身** + 把 `cuda_v12` 目录加进 `PATH`。
   实测：CPU 34.6 tok/s → GPU 148.6 tok/s。
3. **投机解码不生效**：Ollama 的构建默认 `--spec-type` 为空，
   日志报 `no implementations specified for speculative decoding`。
   → 必须显式指定 `--spec-type draft-simple`。
4. **默认 4 路并行**：Ollama 的 llama-server 默认 `n_slots = 4`，
   而投机解码只支持单序列。
   → 必须加 `-np 1`。
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# llama.cpp 二进制目录（Ollama 自带的那份，见本文件顶部说明）
# 换机器时用环境变量覆盖：INFERBENCH_LLAMA_DIR / INFERBENCH_CUDA_SUBDIR
LLAMA_DIR = Path(os.environ.get("INFERBENCH_LLAMA_DIR", r"D:\Program\lib\ollama"))
CUDA_SUBDIR = os.environ.get("INFERBENCH_CUDA_SUBDIR", "cuda_v12")


class LlamaServerError(RuntimeError):
    pass


class LlamaServer:
    """启动 / 查询 / 停止一个 llama-server 实例。"""

    def __init__(self, target_gguf: str | Path, *, draft_gguf: str | Path | None = None,
                 spec_type: str | None = None, spec_n_max: int | None = None,
                 port: int = 8087, n_ctx: int = 4096, ngl: int = 99,
                 llama_dir: Path = LLAMA_DIR, cuda_subdir: str = CUDA_SUBDIR,
                 log_path: Path | None = None) -> None:
        self.target = str(target_gguf)
        self.draft = str(draft_gguf) if draft_gguf else None
        self.spec_type = spec_type
        self.spec_n_max = spec_n_max
        self.port = port
        self.n_ctx = n_ctx
        self.ngl = ngl
        self.exe = Path(llama_dir) / "llama-server.exe"
        self.cuda_dir = Path(llama_dir) / cuda_subdir
        self.log_path = Path(log_path) if log_path else Path(os.environ.get("TEMP", ".")) / f"llama_server_{port}.log"
        self.proc: subprocess.Popen | None = None

    # ---------- 生命周期 ----------
    def _args(self) -> list[str]:
        args = [str(self.exe), "-m", self.target, "-ngl", str(self.ngl),
                "-c", str(self.n_ctx), "-np", "1",           # 坑 4：投机解码只支持单序列
                "--port", str(self.port), "--host", "127.0.0.1", "--metrics"]
        if self.draft:
            args += ["-md", self.draft, "-ngld", str(self.ngl)]
        if self.spec_type:
            args += ["--spec-type", self.spec_type]           # 坑 3：必须显式指定
        if self.spec_n_max is not None:
            args += ["--spec-draft-n-max", str(self.spec_n_max)]
        return args

    def _env(self) -> dict:
        env = dict(os.environ)
        # 坑 2：backend 要指向 DLL 文件，且依赖目录必须进 PATH
        if self.cuda_dir.exists():
            env["GGML_BACKEND_PATH"] = str(self.cuda_dir / "ggml-cuda.dll")
            env["PATH"] = str(self.cuda_dir) + os.pathsep + env.get("PATH", "")
        return env

    def start(self, timeout: float = 180.0) -> dict:
        if not self.exe.exists():
            raise LlamaServerError(f"找不到 {self.exe}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w", encoding="utf-8", errors="replace")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self._args(), env=self._env(), stdout=self.log,
                                     stderr=subprocess.STDOUT, cwd=str(self.exe.parent))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise LlamaServerError(f"进程提前退出（exit={self.proc.returncode}），日志：{self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if json.loads(r.read().decode("utf-8")).get("status") == "ok":
                        return {"load_s": round(time.perf_counter() - t0, 1),
                                "spec_active": self._spec_active()}
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.6)
        raise LlamaServerError(f"启动超时，日志：{self.log_path}")

    def _spec_active(self) -> bool:
        """从日志确认投机解码真的启用了（而不是只加载了 draft 模型）。"""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if "no implementations specified for speculative decoding" in text:
            return False
        return True

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            if hasattr(self, "log"):
                self.log.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------- 请求 ----------
    def complete(self, prompt: str, *, n_predict: int = 200, temperature: float = 0.0,
                 timeout: float = 300.0) -> dict:
        body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                           "temperature": temperature, "cache_prompt": False},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        data["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return data

    def metrics(self) -> dict:
        """读 Prometheus 指标，取出投机解码计数器。"""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=10) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return {}
        out: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("llamacpp:") and " " in line:
                key, _, val = line.rpartition(" ")
                try:
                    out[key.strip()] = float(val)
                except ValueError:
                    continue
        return out


def spec_counters(m: dict) -> tuple[float, float, float]:
    """返回 (起草 tokens, 接受 tokens, 验证轮次)。"""
    return (m.get("llamacpp:spec_decode_num_draft_tokens_total", 0.0),
            m.get("llamacpp:spec_decode_num_accepted_tokens_total", 0.0),
            m.get("llamacpp:spec_decode_num_drafts_total", 0.0))



