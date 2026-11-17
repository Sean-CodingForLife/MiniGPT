"""纯后端的无 torch 权重文件。

格式（version 1，小端）::

    [4 字节] 头部长度 N (uint32)
    [N 字节] UTF-8 JSON 索引
    [数据区] 按索引顺序拼接的 float32 张量

JSON 索引形如::

    {
      "format": "minigpt-pure-weights",
      "version": 1,
      "dtype": "float32",
      "metadata": {...},
      "tensors": {"name": {"shape": [...], "offset": 0, "bytes": 12}},
      "payload_sha256": "..."
    }

``offset``/``bytes`` 相对数据区起点。文件自包含（无 pickle、无 torch），
因此 bundle 里的权重任何 Python 都能读取。
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT = "minigpt-pure-weights"
VERSION = 1


class WeightFileError(ValueError):
    """权重文件格式非法或校验失败时抛出。"""


def _flatten(state: Mapping[str, tuple[Sequence[int], Sequence[float]]]) -> tuple[list[dict[str, Any]], list[float]]:
    """把 ``name -> (shape, values)`` 摊平成索引条目与连续浮点数据区。

    同时校验每个张量的元素个数与形状一致，不一致直接报错。
    """
    tensors: list[dict[str, Any]] = []
    payload: list[float] = []
    for name, (shape, values) in state.items():
        shape = [int(size) for size in shape]
        expected = 1
        for size in shape:
            expected *= size
        if len(values) != expected:
            raise ValueError(f"{name}: expected {expected} values, got {len(values)}")
        offset = len(payload) * 4
        tensors.append({"name": name, "shape": shape, "offset": offset, "bytes": expected * 4})
        payload.extend(float(value) for value in values)
    return tensors, payload


def save_weights(
    path: str | Path,
    state: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把 ``state``（``name -> (shape, values)``）写成 v1 权重文件，返回索引。"""

    tensors, payload = _flatten(state)
    body = b"".join(struct.pack("<f", value) for value in payload)
    by_name = {item["name"]: item for item in tensors}
    index = {
        "format": FORMAT,
        "version": VERSION,
        "dtype": "float32",
        "metadata": dict(metadata or {}),
        "tensors": {
            name: {"shape": item["shape"], "offset": item["offset"], "bytes": item["bytes"]}
            for name, item in by_name.items()
        },
        "payload_sha256": hashlib.sha256(body).hexdigest(),
    }
    encoded = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(struct.pack("<I", len(encoded)))
        handle.write(encoded)
        handle.write(body)
    return index


def load_weights(
    path: str | Path,
    verify: bool = True,
) -> tuple[dict[str, tuple[tuple[int, ...], list[float]]], dict[str, Any]]:
    """读取权重文件，返回 ``(state, index)``。

    ``verify=True`` 时会重新计算数据区 sha256，并检查每个张量切片都落在
    数据区边界内，防止损坏文件被静默加载。
    """

    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise WeightFileError(f"cannot read weight file {target}: {exc}") from exc
    if len(raw) < 4:
        raise WeightFileError("weight file is truncated")
    (header_length,) = struct.unpack("<I", raw[:4])
    if header_length <= 0 or 4 + header_length > len(raw):
        raise WeightFileError("invalid weight file header length")
    try:
        index = json.loads(raw[4 : 4 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightFileError(f"invalid weight file index: {exc}") from exc
    if index.get("format") != FORMAT:
        raise WeightFileError(f"unexpected format: {index.get('format')!r}")
    if index.get("version") != VERSION:
        raise WeightFileError(f"unsupported version: {index.get('version')!r}")
    if index.get("dtype") != "float32":
        raise WeightFileError(f"unsupported dtype: {index.get('dtype')!r}")
    payload = raw[4 + header_length :]
    if verify:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != index.get("payload_sha256"):
            raise WeightFileError("payload sha256 mismatch")
    tensors = index.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise WeightFileError("weight file has no tensors")
    state: dict[str, tuple[tuple[int, ...], list[float]]] = {}
    for name, item in tensors.items():
        try:
            shape = tuple(int(size) for size in item["shape"])
            offset = int(item["offset"])
            size_bytes = int(item["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeightFileError(f"invalid tensor entry {name!r}") from exc
        count = size_bytes // 4
        if size_bytes % 4 or offset < 0 or offset + size_bytes > len(payload):
            raise WeightFileError(f"tensor {name!r} is outside the payload")
        if count != _product(shape):
            raise WeightFileError(f"tensor {name!r} shape/bytes mismatch")
        values = list(struct.unpack_from(f"<{count}f", payload, offset))
        state[name] = (shape, values)
    if verify and sum(item["bytes"] for item in tensors.values()) != len(payload):
        raise WeightFileError("payload has trailing or missing bytes")
    return state, index


def _product(shape: Sequence[int]) -> int:
    """形状的元素总数（用于核对 bytes 与 shape 是否一致）。"""
    total = 1
    for size in shape:
        total *= int(size)
    return total


__all__ = ["save_weights", "load_weights", "WeightFileError", "FORMAT", "VERSION"]
