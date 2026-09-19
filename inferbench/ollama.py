"""Ollama HTTP 客户端 —— 纯标准库实现（urllib），零依赖。

为什么要自己写：
1. Ollama 的 /api/chat 返回里带 nanosecond 级计时字段（prompt_eval_duration /
   eval_duration），这是做量化对比最干净的延迟来源，不需要外部压测工具。
2. /api/ps 返回**真实显存占用 + CPU/GPU 分流**，是「量化档位 → 显存代价」的
   唯一可信来源（比 GGUF 文件大小真实，因为它含 KV cache）。

用法::

    cli = Ollama()
    r = cli.chat("qwen3:1.7b", [{"role": "user", "content": "hi"}])
    print(r["message"]["content"], cli.metrics(r))
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from inferbench import config

NS = 1_000_000_000


class OllamaError(RuntimeError):
    """Ollama 请求失败（含重试耗尽）。"""


class Ollama:
    def __init__(self, host: str | None = None, timeout: float | None = None,
                 retries: int | None = None) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout if timeout is not None else config.OLLAMA_TIMEOUT
        self.retries = retries if retries is not None else config.OLLAMA_RETRIES

    # ---------- 底层 ----------
    def _request(self, path: str, payload: dict | None = None, method: str = "POST") -> dict:
        url = self.host + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body) if body.strip() else {}
            except Exception as exc:  # noqa: BLE001 - 统一转成 OllamaError
                last_err = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2 ** attempt))
        raise OllamaError(f"{method} {path} 失败: {last_err}") from last_err

    def _get(self, path: str) -> dict:
        return self._request(path, payload=None, method="GET")

    # ---------- 能力 ----------
    def version(self) -> str:
        try:
            return self._get("/api/version").get("version", "unknown")
        except OllamaError:
            return "unknown"

    def list_models(self) -> list[dict]:
        return self._get("/api/tags").get("models", [])

    def ps(self) -> list[dict]:
        """当前加载在内存里的模型（含 SIZE 与 PROCESSOR 分流）。"""
        return self._get("/api/ps").get("models", [])

    def chat(self, model: str, messages: list[dict], *, think: bool = False,
             num_ctx: int | None = None, num_predict: int | None = None,
             temperature: float | None = None, seed: int | None = None,
             keep_alive: str | None = None, fmt: str | None = None) -> dict:
        """单轮对话（非流式）。

        think=False 对 Qwen3 这类思维链模型是**必须**的：否则输出里混入推理
        链，token 数被污染，量化档位之间就不可比了。
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": keep_alive or config.KEEP_ALIVE,
            "options": {
                "temperature": config.TEMPERATURE if temperature is None else temperature,
                "num_ctx": config.NUM_CTX if num_ctx is None else num_ctx,
                "num_predict": config.NUM_PREDICT if num_predict is None else num_predict,
                "seed": config.SEED if seed is None else seed,
            },
        }
        if fmt:
            payload["format"] = fmt
        return self._request("/api/chat", payload)

    def embed(self, model: str, texts: list[str], keep_alive: str | None = None,
              num_ctx: int | None = None) -> list[list[float]]:
        """批量取 embedding（/api/embed）。

        **必须锁 num_ctx**：本机实测 `qwen3-embedding:0.6b` 的默认上下文是 32768，
        KV cache 把它撑到 **5.8 GB**（模型本体只有 0.6 GB），直接把 8GB 显存顶满，
        随后所有推理退到 CPU（解码从 168 tok/s 掉到 0.6 tok/s）。
        单句 query 给 512 上下文绰绰有余。
        """
        payload = {
            "model": model,
            "input": texts,
            "keep_alive": keep_alive or config.KEEP_ALIVE,
            "options": {"num_ctx": config.EMBED_NUM_CTX if num_ctx is None else num_ctx},
        }
        resp = self._request("/api/embed", payload)
        return resp.get("embeddings", [])

    # ---------- 指标提取 ----------
    @staticmethod
    def metrics(resp: dict) -> dict:
        """把 Ollama 的纳秒计时字段折算成可比指标。"""
        def sec(key: str) -> float:
            return float(resp.get(key) or 0) / NS

        prompt_eval_s = sec("prompt_eval_duration")
        eval_s = sec("eval_duration")
        prompt_tokens = int(resp.get("prompt_eval_count") or 0)
        out_tokens = int(resp.get("eval_count") or 0)
        return {
            "total_ms": sec("total_duration") * 1000,
            "load_ms": sec("load_duration") * 1000,
            # 非流式请求下，prefill 耗时 ≈ 首 Token 延迟（TTFT）
            "ttft_ms": prompt_eval_s * 1000,
            "prefill_tps": (prompt_tokens / prompt_eval_s) if prompt_eval_s > 0 else 0.0,
            "decode_tps": (out_tokens / eval_s) if eval_s > 0 else 0.0,
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "done_reason": resp.get("done_reason", ""),
        }

    def footprint(self, model: str) -> dict:
        """模型当前在内存/显存里的占用与 CPU-GPU 分流。

        注意：`/api/ps` 不返回 `processor` 字段（那是 CLI 表格里算出来的），
        所以这里用 size_vram / size 自己算分流比例——这是判断「模型是否真的
        全量在 GPU 上」的关键证据。
        """
        for m in self.ps():
            if m.get("name") == model or m.get("model") == model:
                size = float(m.get("size") or 0)
                size_vram = float(m.get("size_vram") or 0)
                ratio = (size_vram / size * 100.0) if size > 0 else 0.0
                if ratio >= 99.5:
                    processor = "100% GPU"
                elif ratio > 0:
                    processor = f"{ratio:.0f}% GPU / {100 - ratio:.0f}% CPU"
                else:
                    processor = "100% CPU"
                return {
                    "footprint_gb": size / (1024 ** 3),
                    "vram_gb": size_vram / (1024 ** 3),
                    "gpu_ratio": ratio,
                    "processor": processor,
                    "context_loaded": m.get("context_length") or m.get("context") or 0,
                }
        return {"footprint_gb": 0.0, "vram_gb": 0.0, "gpu_ratio": 0.0,
                "processor": "not loaded", "context_loaded": 0}

    # ---------- 显存治理 ----------
    def unload(self, model: str) -> None:
        """立即卸载模型，释放显存（等价于 `ollama stop`）。"""
        try:
            self._request("/api/generate", {"model": model, "keep_alive": 0})
        except OllamaError:
            pass

    def loaded_models(self) -> list[str]:
        return [m.get("name") or m.get("model") or "" for m in self.ps()]

    def unload_all(self, keep: str | None = None) -> list[str]:
        """卸载除 `keep` 之外的所有模型，避免多模型争抢显存污染测量。

        返回被卸载的模型列表。
        """
        stopped: list[str] = []
        for name in self.loaded_models():
            if keep and name == keep:
                continue
            self.unload(name)
            stopped.append(name)
        if stopped:
            time.sleep(1.5)  # 给驱动一点回收时间
        return stopped
