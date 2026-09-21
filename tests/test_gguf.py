"""GGUF 元数据解析与 KV cache 理论值测试。

这里的重点不是"能不能读出数字"，而是**读出来的数字能不能当尺子**：
实测斜率要和理论斜率对得上，才能把显存增长归因到 KV cache 上。
所以测试覆盖三件事：解析正确、键名归一化正确、算出来的字节数符合手算结果。
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from inferbench import gguf

# GGUF 元数据类型常量（测试里要手工拼二进制，所以本地再声明一份）
_T_UINT32, _T_STRING, _T_ARRAY = 4, 8, 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", _T_STRING) + _s(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = b"".join(_s(v) for v in values)
    return (_s(key) + struct.pack("<I", _T_ARRAY)
            + struct.pack("<I", _T_STRING) + struct.pack("<Q", len(values)) + body)


def _write_gguf(path: Path, pairs: list[bytes], *, magic: bytes = b"GGUF") -> Path:
    head = magic + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    path.write_bytes(head + b"".join(pairs))
    return path


@pytest.fixture
def qwen3_like(tmp_path: Path) -> Path:
    """一个结构和 Qwen3-1.7B 一致的 GGUF 头（28 层 / 8 KV 头 / head_dim 128）。"""
    return _write_gguf(tmp_path / "model.gguf", [
        _kv_string("general.architecture", "qwen3"),
        _kv_uint32("qwen3.block_count", 28),
        _kv_uint32("qwen3.attention.head_count", 16),
        _kv_uint32("qwen3.attention.head_count_kv", 8),
        _kv_uint32("qwen3.attention.key_length", 128),
        _kv_uint32("qwen3.embedding_length", 2048),
        _kv_uint32("qwen3.context_length", 40960),
        _kv_string_array("tokenizer.ggml.tokens", ["<pad>", "hello", "世界"]),
    ])


def test_reads_metadata_and_skips_arrays(qwen3_like: Path):
    meta = gguf.read_metadata(qwen3_like)
    assert meta["general.architecture"] == "qwen3"
    assert meta["qwen3.block_count"] == 28
    # 数组只记长度，不把词表读进内存（真实词表有十几万项）
    assert meta["tokenizer.ggml.tokens"] == {"__array_len__": 3}
    assert meta["__version__"] == 3


def test_geometry_matches_known_qwen3_shape(qwen3_like: Path):
    geom = gguf.geometry(qwen3_like)
    assert geom["complete"] is True
    assert (geom["block_count"], geom["head_count"], geom["head_count_kv"]) == (28, 16, 8)
    assert geom["key_length"] == 128
    assert geom["context_length"] == 40960


def test_kv_bytes_per_token_matches_hand_calculation(qwen3_like: Path):
    """手算：2 (K+V) × 28 层 × 8 KV 头 × 128 = 57344 个元素 × 2 字节 = 114688 B/token。"""
    geom = gguf.geometry(qwen3_like)
    assert gguf.kv_bytes_per_token(geom) == 114688
    assert gguf.kv_bytes_per_token(geom, bytes_per_elem=1) == 57344


def test_kv_gb_per_1k_is_about_011():
    """0.1094 GB/1K token —— 这个数就是报告里用来和实测斜率对账的尺子。"""
    geom = {"complete": True, "block_count": 28, "head_count_kv": 8, "key_length": 128,
            "value_length": 128, "head_count": 16, "architecture": "qwen3"}
    assert gguf.kv_gb_per_1k_tokens(geom) == pytest.approx(0.1094, abs=0.0005)


def test_kv_for_context_scales_with_parallel_slots():
    """并行度必须乘进去：每个 slot 各占一份独立的 KV cache。"""
    geom = {"complete": True, "block_count": 28, "head_count_kv": 8, "key_length": 128,
            "value_length": 128, "head_count": 16, "architecture": "qwen3"}
    one = gguf.kv_gb_for_context(geom, 4096)
    four = gguf.kv_gb_for_context(geom, 4096, num_parallel=4)
    assert four == pytest.approx(one * 4)
    assert one == pytest.approx(0.4375, abs=0.001)


def test_missing_key_length_falls_back_to_embedding_over_heads(tmp_path: Path):
    """老 GGUFs 没有 key_length，要退回 embedding_length / head_count。"""
    path = _write_gguf(tmp_path / "old.gguf", [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("llama.block_count", 32),
        _kv_uint32("llama.attention.head_count", 32),
        _kv_uint32("llama.attention.head_count_kv", 8),
        _kv_uint32("llama.embedding_length", 4096),
    ])
    geom = gguf.geometry(path)
    assert geom["key_length"] == 128          # 4096 / 32
    assert geom["value_length"] == 128
    assert geom["complete"] is True


def test_head_count_kv_defaults_to_head_count(tmp_path: Path):
    """没有 GQA 的模型不写 head_count_kv，此时 KV 头数等于注意力头数。"""
    path = _write_gguf(tmp_path / "mha.gguf", [
        _kv_uint32("llama.block_count", 12),
        _kv_uint32("llama.attention.head_count", 12),
        _kv_uint32("llama.attention.key_length", 64),
    ])
    geom = gguf.geometry(path)
    assert geom["head_count_kv"] == 12
    assert gguf.kv_bytes_per_token(geom) == 2 * 12 * 12 * 64 * 2


def test_bad_magic_raises(tmp_path: Path):
    path = _write_gguf(tmp_path / "bad.gguf", [_kv_uint32("a.b", 1)], magic=b"NOPE")
    with pytest.raises(gguf.GgufError):
        gguf.read_metadata(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(gguf.GgufError):
        gguf.read_metadata(tmp_path / "nope.gguf")


def test_incomplete_geometry_refuses_to_guess(tmp_path: Path):
    """结构参数不全时必须抛错，不能拿缺省的 0 算出个假数字。

    这条是刻意的：KV 理论值是**尺子**，尺子错了会把实测结论带偏，
    宁可缺一个值，也不要给一个错的值。
    """
    path = _write_gguf(tmp_path / "partial.gguf", [_kv_uint32("llama.block_count", 32)])
    geom = gguf.geometry(path)
    assert geom["complete"] is False
    with pytest.raises(gguf.GgufError):
        gguf.kv_bytes_per_token(geom)
    assert "不完整" in gguf.describe(geom)


def test_truncated_metadata_keeps_what_it_read(tmp_path: Path):
    """头部读到的范围不够时，已解析出来的键仍然可用（不能整体报废）。"""
    full = _write_gguf(tmp_path / "full.gguf", [
        _kv_uint32("llama.block_count", 32),
        _kv_string_array("tokenizer.ggml.tokens", ["a" * 50, "b" * 50]),
    ])
    raw = full.read_bytes()
    cut = tmp_path / "cut.gguf"
    # 切在数组中间：block_count 已读完，词表读不完 → 应当保留前者并优雅停止
    cut.write_bytes(raw[: len(raw) - 40])
    meta = gguf.read_metadata(cut)
    assert meta["llama.block_count"] == 32
    assert "tokenizer.ggml.tokens" not in meta
