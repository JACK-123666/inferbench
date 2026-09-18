"""语义缓存核心：缓存实现、请求流构造、计时器。

从实验脚本里抽出来的原因：实验二（阈值扫描）和实验三（写入门槛）都要用它，
放在 `inferbench/experiments/cache_exp.py` 里会让 gate 反向依赖 cache_exp——那是臃肿的开始。

设计要点：
- `SemanticCache`：版本指纹 + TTL + 分区 + 开关，四个都对应生产上的真实需求
- `TimedCache`：同一文本只真实调用一次（把 LLM 调用量压到"唯一文本数"），
  并带**预热**——预热文本必须用请求流里不会出现的文本，否则那条请求的"延迟"
  会变成冷加载时间（实测把未命中 P95 污染成 8182 ms）
- `build_stream`：把 100 条 query 扩成 300 条 1:N 语义改写请求流，模拟真实流量
"""
from __future__ import annotations

import random
import time

import config
from inferbench import tasks
from inferbench.ollama import Ollama, OllamaError
from inferbench.stats import cosine

PARAPHRASE_PROMPT = (
    "把下面这句话换一种说法（保持语义完全相同、长度相近），只输出改写后的这一句话，"
    "不要解释、不要引号、不要序号：\n{text}"
)
POLITE_PREFIXES = ["请问", "麻烦问一下", "你好，", "我想问下", ""]
POLITE_SUFFIXES = ["", "呢？", "呀", "谢谢"]


def perturb(text: str, rng: random.Random) -> str:
    """确定性扰动：生成"同一意图的另一种说法"，不依赖模型（用于快速跑通）。"""
    prefix = rng.choice(POLITE_PREFIXES)
    suffix = rng.choice(POLITE_SUFFIXES)
    core = text.rstrip("？?。！!")
    return f"{prefix}{core}{suffix}" if (prefix or suffix) else f"{core}，谢谢"


def build_stream(items: list[dict], client: Ollama, *, paraphrases: int,
                 perturb_only: bool, progress=lambda _: None) -> list[dict]:
    """构造 1:N 语义改写请求流（模拟真实流量里同一个问题被反复问）。

    改写用 temperature=0.7 + 固定 seed 生成，保证**可复现**——
    不加 seed 会导致两次实验的请求流不同，指标跟着变（踩过）。
    """
    rng = random.Random(config.SEED)
    stream: list[dict] = []
    if perturb_only:
        for item in items:
            stream.append({"text": item["text"], "gold": item["intent"],
                           "base_id": item["id"], "kind": "base"})
            for i in range(paraphrases):
                stream.append({"text": perturb(item["text"], rng), "gold": item["intent"],
                               "base_id": item["id"], "kind": f"perturb{i + 1}"})
    else:
        progress("用本地模型生成语义改写…")
        for idx, item in enumerate(items, 1):
            stream.append({"text": item["text"], "gold": item["intent"],
                           "base_id": item["id"], "kind": "base"})
            for i in range(paraphrases):
                text = ""
                try:
                    resp = client.chat(config.CACHE_MODEL,
                                       [{"role": "user",
                                         "content": PARAPHRASE_PROMPT.format(text=item["text"])}],
                                       num_predict=64, temperature=0.7, seed=config.SEED)
                    text = (resp.get("message") or {}).get("content", "").strip().strip('"“”')
                    text = text.split("\n")[0].strip()
                except OllamaError as exc:
                    progress(f"  改写失败（回退扰动）: {exc}")
                if not text or text == item["text"]:
                    text, kind = perturb(item["text"], rng), f"perturb{i + 1}"
                else:
                    kind = f"para{i + 1}"
                stream.append({"text": text, "gold": item["intent"],
                               "base_id": item["id"], "kind": kind})
            if idx % 20 == 0:
                progress(f"  改写进度 {idx}/{len(items)}")
    rng.shuffle(stream)          # 打乱，避免同一条 query 的变体挨在一起
    return stream


class SemanticCache:
    """最小可用语义缓存：版本指纹、TTL、按意图分区、灰度开关。"""

    def __init__(self, version: str = "v1", ttl: float = config.CACHE_TTL_SECONDS) -> None:
        self.version = version
        self.ttl = ttl
        self.enabled = True
        self.entries: list[dict] = []      # {vec, label, text, ts, version, partition, source_gold}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def lookup(self, vec: list[float], now: float) -> tuple[float, dict | None]:
        best_score, best_entry = -1.0, None
        for entry in self.entries:
            if entry["version"] != self.version:
                continue                                   # 版本指纹：换代后旧缓存直接失效
            if self.ttl > 0 and now - entry["ts"] > self.ttl:
                self.evictions += 1
                continue                                   # TTL 过期
            score = cosine(vec, entry["vec"])
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return best_score, best_entry

    def store(self, vec: list[float], label: str, text: str, now: float,
              partition: str = "", source_gold: str = "") -> None:
        self.entries.append({"vec": vec, "label": label, "text": text, "ts": now,
                             "version": self.version, "partition": partition,
                             "source_gold": source_gold})

    def invalidate_all(self) -> int:
        """失效开关：模型/Prompt 换代时调用。"""
        n = len(self.entries)
        self.entries.clear()
        return n


class TimedCache:
    """带记忆化的计时器：同一文本只真实调用一次，并区分 embed / LLM 耗时。"""

    WARMUP_TEXT = "《预热文本，不会出现在请求流中》"

    def __init__(self, client: Ollama, memo_latency: bool = True) -> None:
        self.client = client
        self.memo_latency = memo_latency
        self._llm: dict[str, tuple[str, str | None, float, int, int]] = {}
        self.llm_calls = 0
        self.embed_calls = 0
        self.warm_ms = {"embed": 0.0, "llm": 0.0}

    def warmup(self, rounds: int = 2) -> None:
        """预热两个模型（冷加载不计入统计）。用当前时间重新计一次。"""
        for _ in range(max(1, rounds)):
            _, ms = self.embed(self.WARMUP_TEXT)
            self.warm_ms["embed"] = ms
            _, _, ms_llm, _, _ = self.classify(self.WARMUP_TEXT)
            self.warm_ms["llm"] = ms_llm

    def preclassify(self, texts: list[str]) -> None:
        """先把所有**唯一文本**真实跑一遍 LLM —— 这是"无缓存基线"的严格定义。"""
        for text in texts:
            self.classify(text)

    def embed(self, text: str) -> tuple[list[float], float]:
        t0 = time.perf_counter()
        vecs = self.client.embed(config.EMBED_MODEL, [text])
        ms = (time.perf_counter() - t0) * 1000.0
        self.embed_calls += 1
        return (vecs[0] if vecs else []), ms

    def classify(self, text: str) -> tuple[str, str | None, float, int, int]:
        """返回 (原始输出, 解析标签, 耗时ms, prompt_tokens, output_tokens)。"""
        if self.memo_latency and text in self._llm:
            return self._llm[text]
        t0 = time.perf_counter()
        try:
            resp = self.client.chat(config.CACHE_MODEL,
                                    tasks.build_messages(text, config.SHOTS),
                                    num_ctx=config.NUM_CTX, num_predict=config.NUM_PREDICT)
            raw = (resp.get("message") or {}).get("content", "") or ""
            met = self.client.metrics(resp)
            result = (raw.strip(), tasks.parse_label(raw),
                      (time.perf_counter() - t0) * 1000.0,
                      met["prompt_tokens"], met["output_tokens"])
        except OllamaError as exc:
            result = ("", None, (time.perf_counter() - t0) * 1000.0, 0, 0)
            print(f"    LLM 调用失败: {exc}")
        self.llm_calls += 1
        self._llm[text] = result
        return result

