"""并发压测模块测试（纯函数部分，不联网）。"""
from __future__ import annotations

from inferbench import loadgen


def test_pct_matches_known_values():
    vals = sorted([10, 20, 30, 40, 50])
    assert loadgen._pct(vals, 50) == 30
    assert loadgen._pct(vals, 0) == 10
    assert loadgen._pct(vals, 100) == 50
    assert loadgen._pct([], 95) == 0.0
    assert loadgen._pct([7], 99) == 7.0


def _level(conc, qps, p95):
    return {"concurrency": conc, "qps": qps, "lat_p95": p95}


def test_saturation_point_finds_knee():
    """QPS 增益跌破 30% 且 P95 明显抬头的那一档，才是真正的饱和点。"""
    results = [_level(1, 10, 150), _level(2, 19, 160), _level(4, 32, 190), _level(8, 36, 400)]
    sat = loadgen.saturation_point(results)
    assert sat["concurrency"] == 8
    assert sat["qps_gain"] is not None and sat["qps_gain"] < 0.30


def test_saturation_point_when_still_scaling():
    """还在线性扩展时（QPS 增益都很高），退化为报告最后一档。"""
    results = [_level(1, 10, 150), _level(2, 20, 155), _level(4, 40, 160)]
    sat = loadgen.saturation_point(results)
    assert sat["concurrency"] == 4


def test_saturation_point_needs_at_least_two_levels():
    assert loadgen.saturation_point([_level(1, 10, 150)]) == {}


def test_run_level_aggregates_synthetic_traffic(monkeypatch):
    """用一个假的 one_request 驱动 run_level，验证聚合口径（QPS / P95 / 错误率）。"""
    import asyncio

    calls = {"n": 0}

    async def fake_request(client, model, prompt, *, max_tokens=96, num_ctx=None, temperature=None):
        calls["n"] += 1
        await asyncio.sleep(0.01)
        ok = calls["n"] % 5 != 0                     # 每 5 个失败 1 个
        base = 100 + calls["n"]
        return loadgen.StreamResult(
            ok=ok, error="" if ok else "boom",
            ttft_ms=base, total_ms=base * 2, output_tokens=10 if ok else 0,
            decode_tps=50.0 if ok else 0.0, prompt_tokens=5, text="t")

    monkeypatch.setattr(loadgen, "one_request", fake_request)
    res = asyncio.run(loadgen.run_level("http://x", "m", [f"p{i}" for i in range(10)],
                                        concurrency=4))
    assert res["requests"] == 10
    assert res["ok"] == 8 and res["errors"] == 2
    assert res["error_rate"] == 0.2
    assert res["qps"] > 0 and res["tokens_per_s"] > 0
    assert res["ttft_p50"] <= res["ttft_p95"] <= res["ttft_p99"]
    assert res["lat_p50"] <= res["lat_p95"] <= res["lat_p99"]


def test_concurrency_is_actually_capped(monkeypatch):
    """Semaphore 必须真的限流：同时进行的请求数不能超过设定并发。"""
    import asyncio

    state = {"active": 0, "peak": 0}

    async def fake_request(client, model, prompt, *, max_tokens=96, num_ctx=None, temperature=None):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.02)
        state["active"] -= 1
        return loadgen.StreamResult(ok=True, error="", ttft_ms=1.0, total_ms=2.0,
                                    output_tokens=1, decode_tps=1.0, prompt_tokens=1, text="")

    monkeypatch.setattr(loadgen, "one_request", fake_request)
    asyncio.run(loadgen.run_level("http://x", "m", [f"p{i}" for i in range(12)], concurrency=3))
    assert state["peak"] <= 3, f"并发峰值 {state['peak']} 超过了设定值 3"

