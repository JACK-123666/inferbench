"""GGUF 元数据解析：读模型结构参数，用来**算** KV cache 的理论显存。

## 为什么值得为这件事写一个模块

KV cache 的大小是可以从第一性原理算出来的：

    bytes/token = 2 × block_count × head_count_kv × key_length × bytes_per_elem
                  ↑ K 和 V 各一份

这些参数就写在 GGUF 文件头的元数据区里，**不需要加载模型、不需要 GPU**，
读前几十 KB 就能拿到。有了理论值，实测的显存增量就有了一把尺子：

- 实测斜率 ≈ 理论斜率 → 显存增长确实来自 KV cache，归因成立；
- 实测斜率 ≫ 理论斜率 → 是别的东西在吃显存（框架开销、显存碎片、别的进程），
  这时候把账算在 num_ctx 头上就是错的。

**尺子不当结论用**：本模块只负责给理论值，判断留给实验结果。

## GGUF 格式（这里只读需要的那部分）

    magic        4 字节 "GGUF"
    version      uint32
    tensor_count uint64
    kv_count     uint64
    然后 kv_count 组：
      key        string（uint64 长度 + UTF-8 字节）
      value_type uint32
      value      按类型解释（数组是 elem_type + uint64 个数 + 元素）
"""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"GGUF"

# GGUF 元数据类型枚举（见 llama.cpp 的 gguf.h）
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

_SCALAR: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4), _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}

# 各架构的元数据键名不同（llama / qwen3 / qwen2 / gemma3 …），这里做键名归一化。
# 只用「后缀」匹配，前缀交给架构名，避免为每个新架构加一张表。
_KEY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "block_count": ("block_count",),
    "head_count": ("attention.head_count",),
    "head_count_kv": ("attention.head_count_kv",),
    "key_length": ("attention.key_length",),
    "value_length": ("attention.value_length",),
    "embedding_length": ("embedding_length",),
    "context_length": ("context_length",),
}


class GgufError(RuntimeError):
    """GGUF 无法解析（文件不存在、magic 不对、元数据被截断）。"""


def read_metadata(path: str | Path, *, max_pairs: int = 4096) -> dict:
    """读出 GGUF 的元数据键值（只读头部，不加载张量）。

    数组类型的值只保留长度（`{"__array_len__": n}`）——本模块不需要数组内容，
    而且词表数组可能有几十万项，读进来既慢又占内存。
    """
    p = Path(path)
    if not p.is_file():
        raise GgufError(f"文件不存在: {p}")

    # 元数据区通常在前几百 KB 内，先读 4MB 足够覆盖（超长词表也够读到头几个键）
    with p.open("rb") as fh:
        head = fh.read(4)
        if head != MAGIC:
            raise GgufError(f"不是 GGUF 文件（magic={head!r}）: {p}")
        buf = head + fh.read(4 * 1024 * 1024)

    off = 4
    (version,) = struct.unpack_from("<I", buf, off)
    off += 4
    (tensor_count,) = struct.unpack_from("<Q", buf, off)
    off += 8
    (kv_count,) = struct.unpack_from("<Q", buf, off)
    off += 8

    out: dict = {"__version__": version, "__tensor_count__": tensor_count}

    def read_string(pos: int) -> tuple[str, int]:
        (n,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        if pos + n > len(buf):
            raise GgufError(f"元数据被截断（字符串越界）: {p}")
        return buf[pos:pos + n].decode("utf-8", errors="replace"), pos + n

    for _ in range(min(kv_count, max_pairs)):
        try:
            key, off = read_string(off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off += 4
            if vtype in _SCALAR:
                fmt, size = _SCALAR[vtype]
                (value,) = struct.unpack_from(fmt, buf, off)
                off += size
            elif vtype == _STRING:
                value, off = read_string(off)
            elif vtype == _ARRAY:
                (elem_type,) = struct.unpack_from("<I", buf, off)
                off += 4
                (count,) = struct.unpack_from("<Q", buf, off)
                off += 8
                if elem_type in _SCALAR:
                    _, size = _SCALAR[elem_type]
                    off += size * count          # 跳过内容，不读进来
                elif elem_type == _STRING:
                    for _i in range(count):
                        _, off = read_string(off)
                else:
                    raise GgufError(f"数组元素类型不支持: {elem_type}")
                value = {"__array_len__": count}
            else:
                raise GgufError(f"元数据类型不支持: {vtype}")
        except (struct.error, IndexError, GgufError):
            break                                 # 头部读到的范围不够，已读到的键仍可用
        out[key] = value
    return out


def _find(meta: dict, suffix: str, arch: str = "") -> object | None:
    """按键名后缀查找（优先带架构前缀的那个，例如 `qwen3.block_count`）。"""
    for cand in _KEY_SUFFIXES[suffix]:
        if arch:
            hit = meta.get(f"{arch}.{cand}")
            if hit is not None:
                return hit
    for key, val in meta.items():
        if not isinstance(key, str) or not isinstance(val, (int, float)):
            continue
        for cand in _KEY_SUFFIXES[suffix]:
            if key == cand or key.endswith("." + cand):
                return val
    return None


def geometry(path: str | Path) -> dict:
    """提取 KV cache 计算所需的结构参数。

    返回的 `head_dim` 优先用 GGUF 里显式的 key_length；
    没有该键时退回 `embedding_length / head_count`（GQA 下这是标准做法）。
    """
    meta = read_metadata(path)
    arch = str(meta.get("general.architecture") or "")

    def as_int(name: str) -> int | None:
        val = _find(meta, name, arch)
        if val is None or isinstance(val, dict):
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    block_count = as_int("block_count")
    head_count = as_int("head_count")
    head_count_kv = as_int("head_count_kv") or head_count
    embedding_length = as_int("embedding_length")

    key_length = as_int("key_length")
    value_length = as_int("value_length") or key_length
    if key_length is None and embedding_length and head_count:
        key_length = embedding_length // head_count
        value_length = key_length

    geom = {
        "architecture": arch,
        "block_count": block_count,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "key_length": key_length,
        "value_length": value_length,
        "embedding_length": embedding_length,
        "context_length": as_int("context_length"),
        "file": str(path),
    }
    geom["complete"] = all(geom[k] for k in
                           ("block_count", "head_count_kv", "key_length", "value_length"))
    return geom


def kv_bytes_per_token(geom: dict, *, bytes_per_elem: int = 2) -> float:
    """每个 token 的 KV cache 字节数（K 和 V 各一份，所以乘 2）。

    `bytes_per_elem` 默认 2（F16）——这是 Ollama 的默认 KV cache 精度
    （`OLLAMA_KV_CACHE_TYPE` 未设置时）。改成 q8_0 时传 1。
    """
    if not geom.get("complete"):
        raise GgufError(f"结构参数不完整，无法计算 KV 大小: {geom}")
    return (2.0 * int(geom["block_count"]) * int(geom["head_count_kv"])
            * int(geom["key_length"]) * bytes_per_elem)


def kv_gb_per_1k_tokens(geom: dict, *, bytes_per_elem: int = 2) -> float:
    """每 1024 token 上下文的 KV cache 显存（GB）——报告里用的斜率单位。"""
    return kv_bytes_per_token(geom, bytes_per_elem=bytes_per_elem) * 1024 / (1024 ** 3)


def kv_gb_for_context(geom: dict, num_ctx: int, *, bytes_per_elem: int = 2,
                      num_parallel: int = 1) -> float:
    """给定上下文长度（与并行 slot 数）的 KV cache 显存（GB）。

    并行度必须乘进去：**每个 slot 各占一份独立的 KV cache**，
    所以 `num_parallel=4` 的 KV 成本是 `num_parallel=1` 的四倍。
    """
    return kv_bytes_per_token(geom, bytes_per_elem=bytes_per_elem) * num_ctx * num_parallel / (1024 ** 3)


def describe(geom: dict) -> str:
    """一行人类可读的结构摘要（进报告用）。"""
    if not geom.get("complete"):
        return f"结构参数不完整（{geom.get('file')}）"
    return (f"{geom['architecture']} · {geom['block_count']} 层 × "
            f"{geom['head_count_kv']}/{geom['head_count']} KV 头 × "
            f"head_dim {geom['key_length']}（声明上下文 {geom.get('context_length')}）")
