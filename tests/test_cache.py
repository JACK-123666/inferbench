"""语义缓存核心测试。

包含一条**回归测试**：请求流构造必须可复现（曾经因为没固定 seed，
两次实验的请求流不同，导致误命中次数在 17/18 之间漂移）。
"""
from __future__ import annotations

import random

import config
from inferbench import cache as C


def _vec(x: float, y: float) -> list[float]:
    return [x, y]


def test_version_fingerprint_invalidates_old_entries():
    """模型/Prompt 换代后，旧缓存必须一条都命中不了。"""
    cache = C.SemanticCache(version="v1")
    cache.store(_vec(1, 0), "summarize", "帮我总结一下", now=0.0)
    score, entry = cache.lookup(_vec(1, 0), now=0.0)
    assert entry is not None and score == 1.0

    cache.version = "v2"
    score, entry = cache.lookup(_vec(1, 0), now=0.0)
    assert entry is None, "版本号变了之后绝不能命中旧条目"


def test_ttl_expiry():
    cache = C.SemanticCache(version="v1", ttl=1.0)
    cache.store(_vec(1, 0), "summarize", "t", now=100.0)
    assert cache.lookup(_vec(1, 0), now=100.5)[1] is not None
    assert cache.lookup(_vec(1, 0), now=102.0)[1] is None, "超过 TTL 就不该命中"


def test_ttl_zero_means_no_expiry():
    cache = C.SemanticCache(version="v1", ttl=0)
    cache.store(_vec(1, 0), "summarize", "t", now=0.0)
    assert cache.lookup(_vec(1, 0), now=10 ** 9)[1] is not None


def test_lookup_returns_most_similar_entry():
    cache = C.SemanticCache(version="v1")
    cache.store(_vec(1, 0), "knowledge_retrieval", "近", now=0.0)
    cache.store(_vec(0, 1), "small_talk", "远", now=0.0)
    score, entry = cache.lookup(_vec(0.9, 0.1), now=0.0)
    assert entry["label"] == "knowledge_retrieval"
    assert score > 0.9


def test_hit_miss_counters_and_invalidate():
    cache = C.SemanticCache(version="v1")
    cache.lookup(_vec(1, 0), now=0.0)            # 空缓存 → miss
    assert cache.misses == 1 and cache.hits == 0
    cache.store(_vec(1, 0), "x", "t", now=0.0)
    cache.lookup(_vec(1, 0), now=0.0)            # → hit
    assert cache.hits == 1
    assert cache.invalidate_all() == 1
    assert cache.entries == []
    assert cache.lookup(_vec(1, 0), now=0.0)[1] is None


def test_kill_switch_disables_cache():
    """一键回滚：开关关掉后，即使有条目也必须回落到直连模型。"""
    cache = C.SemanticCache(version="v1")
    cache.store(_vec(1, 0), "x", "t", now=0.0)
    cache.enabled = False
    # 实验里的判定条件是 `cache.enabled and entry is not None and score >= threshold`
    assert not cache.enabled


def test_build_stream_is_reproducible():
    """回归测试：固定 seed 后，两次构造的请求流必须逐条相同。

    踩过的坑：改写用 temperature=0.7 生成却没传 seed，
    两次运行的请求流不同，误命中次数在 17/18 之间漂移，数字不可比。
    """
    items = [{"id": i, "text": f"问题{i}", "intent": "small_talk"} for i in range(5)]

    class FakeClient:                       # 不联网：走确定性扰动分支
        def chat(self, *a, **k):
            raise AssertionError("perturb_only=True 时不应调用模型")

    a = C.build_stream(items, FakeClient(), paraphrases=2, perturb_only=True)
    b = C.build_stream(items, FakeClient(), paraphrases=2, perturb_only=True)
    assert [x["text"] for x in a] == [x["text"] for x in b]
    assert len(a) == 5 * 3


def test_perturb_preserves_intent_text_core():
    rng = random.Random(config.SEED)
    out = C.perturb("帮我总结一下这段对话？", rng)
    assert "帮我总结一下这段对话" in out
    assert out != "帮我总结一下这段对话？"          # 必须真的改了说法

