"""无 torch 的 checkpoint 格式 ``.mgckpt``（JSON 头 + float64 数据区）。

布局::

    [4B 小端头部长度 N][N B UTF-8 JSON 头][float64 张量数据]

Version 2 用 float64（``<d``）保存张量，使纯后端（float64 训练）的续训无损。
Version 1 文件（float32，``<f``）仍可读取；头部 ``dtype`` 字段（v1 没有）
决定解包格式。

头部保存所有非张量内容（config、tokenizer 状态、指标、RNG 状态、优化器超参）
以及张量索引::

    "tensors": {
      "model.transformer.wte.weight": {"shape": [V, C], "offset": 0, "bytes": ...},
      "optimizer.m.0":              {"shape": [N],      "offset": ..., "bytes": ...},
      ...
    }

``offset``/``bytes`` 相对张量数据区起点。设计上刻意保持简单、自描述：
无 pickle、无 torch，加载时不会执行任何代码。

使用 pure 后端时，Trainer 会写出 ``last.mgckpt`` / ``best.mgckpt``。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT = "minigpt-pure-checkpoint"
VERSION = 2
# 必须与 minigpt.training.checkpoint.CHECKPOINT_VERSION 保持一致，才能共用
# validate_checkpoint() 的校验契约。
FORMAT_VERSION = 1


class PureCheckpointError(ValueError):
    """``.mgckpt`` 格式非法或校验失败时抛出。"""


def _product(shape: Sequence[int]) -> int:
    """形状的元素总数。"""
    total = 1
    for size in shape:
        total *= int(size)
    return total


def _collect_tensors(payload: dict[str, Any]) -> tuple[list[tuple[str, list[int], list[float]]], dict[str, Any]]:
    """把 payload 拆成「张量列表 + 非张量元数据」。

    模型参数命名为 ``model.<name>``，优化器一阶/二阶动量命名为
    ``optimizer.m.<i>`` / ``optimizer.v.<i>``，其余字段留在 header 里。
    任何形状与数值长度不一致都会在这里直接报错。
    """
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, dict):
        raise PureCheckpointError("payload['model'] must be a dict of name -> (shape, values)")
    if optimizer is None:
        optimizer = {}
    if not isinstance(optimizer, dict):
        raise PureCheckpointError("payload['optimizer'] must be a dict")

    tensors: list[tuple[str, list[int], list[float]]] = []
    for name, entry in model.items():
        try:
            shape, values = entry
        except (TypeError, ValueError) as exc:
            raise PureCheckpointError(f"model parameter {name!r} must be (shape, values)") from exc
        shape = [int(size) for size in shape]
        values = [float(value) for value in values]
        if _product(shape) != len(values):
            raise PureCheckpointError(f"model parameter {name!r}: shape/value mismatch")
        tensors.append((f"model.{name}", shape, values))

    moments = optimizer.get("m")
    velocities = optimizer.get("v")
    if moments is not None:
        if not isinstance(moments, list) or not isinstance(velocities, list):
            raise PureCheckpointError("optimizer m/v must be lists")
        if len(moments) != len(velocities):
            raise PureCheckpointError("optimizer m/v length mismatch")
        for index, (m_values, v_values) in enumerate(zip(moments, velocities)):
            tensors.append((f"optimizer.m.{index}", [len(m_values)], [float(value) for value in m_values]))
            tensors.append((f"optimizer.v.{index}", [len(v_values)], [float(value) for value in v_values]))
    meta_optimizer = {key: value for key, value in optimizer.items() if key not in {"m", "v"}}
    meta = {key: value for key, value in payload.items() if key not in {"model", "optimizer"}}
    return tensors, {"payload": meta, "optimizer": meta_optimizer}


def save_pure_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """原子写入 ``.mgckpt``（同目录临时文件 + ``os.replace``），返回 header。"""

    data = dict(payload)
    tensors, parts = _collect_tensors(data)
    header = {
        "format": FORMAT,
        "version": VERSION,
        "format_version": FORMAT_VERSION,
        "dtype": "float64",
        **parts,
        "tensors": {},
    }
    body = bytearray()
    for name, shape, values in tensors:
        encoded = b"".join(struct.pack("<d", value) for value in values)
        header["tensors"][name] = {"shape": shape, "offset": len(body), "bytes": len(encoded)}
        body.extend(encoded)
    header["payload_sha256"] = hashlib.sha256(bytes(body)).hexdigest()
    encoded_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    from minigpt.utils import replace_with_retry

    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<I", len(encoded_header)))
            handle.write(encoded_header)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return header


def load_pure_checkpoint(path: str | Path, verify: bool = True) -> dict[str, Any]:
    """读回 ``.mgckpt``，还原成 Trainer 可直接使用的 payload 字典。

    支持 v1（float32）与 v2（float64）；``verify=True`` 时校验数据区 sha256。
    """

    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise PureCheckpointError(f"cannot read checkpoint {target}: {exc}") from exc
    if len(raw) < 4:
        raise PureCheckpointError("checkpoint is truncated")
    (header_length,) = struct.unpack("<I", raw[:4])
    if header_length <= 0 or 4 + header_length > len(raw):
        raise PureCheckpointError("invalid checkpoint header length")
    try:
        header = json.loads(raw[4 : 4 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PureCheckpointError(f"invalid checkpoint header: {exc}") from exc
    if header.get("format") != FORMAT:
        raise PureCheckpointError(f"unexpected checkpoint format: {header.get('format')!r}")
    version = header.get("version")
    if version not in (1, VERSION):
        raise PureCheckpointError(f"unsupported checkpoint version: {version!r}")
    dtype = header.get("dtype", "float32" if version == 1 else "float64")
    if dtype not in ("float32", "float64"):
        raise PureCheckpointError(f"unsupported checkpoint dtype: {dtype!r}")
    item_code, item_size = ("f", 4) if dtype == "float32" else ("d", 8)

    body = raw[4 + header_length :]
    if verify and hashlib.sha256(body).hexdigest() != header.get("payload_sha256"):
        raise PureCheckpointError("checkpoint payload sha256 mismatch")

    tensors = header.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise PureCheckpointError("checkpoint has no tensors")

    values_by_name: dict[str, list[float]] = {}
    for name, entry in tensors.items():
        try:
            shape = [int(size) for size in entry["shape"]]
            offset = int(entry["offset"])
            size_bytes = int(entry["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PureCheckpointError(f"invalid tensor entry {name!r}") from exc
        if size_bytes % item_size or offset < 0 or offset + size_bytes > len(body):
            raise PureCheckpointError(f"tensor {name!r} is outside the payload")
        count = size_bytes // item_size
        if _product(shape) != count:
            raise PureCheckpointError(f"tensor {name!r} shape/bytes mismatch")
        values_by_name[name] = list(struct.unpack_from(f"<{count}{item_code}", body, offset))

    model: dict[str, tuple[list[int], list[float]]] = {}
    moments: dict[int, list[float]] = {}
    velocities: dict[int, list[float]] = {}
    for name, values in values_by_name.items():
        if name.startswith("model."):
            key = name[len("model.") :]
            model[key] = ([int(size) for size in tensors[name]["shape"]], values)
        elif name.startswith("optimizer.m."):
            moments[int(name.rsplit(".", 1)[1])] = values
        elif name.startswith("optimizer.v."):
            velocities[int(name.rsplit(".", 1)[1])] = values
        else:
            raise PureCheckpointError(f"unknown tensor name: {name!r}")
    optimizer = dict(header.get("optimizer") or {})
    if moments or velocities:
        if sorted(moments) != sorted(velocities):
            raise PureCheckpointError("optimizer m/v index mismatch")
        optimizer["m"] = [moments[index] for index in sorted(moments)]
        optimizer["v"] = [velocities[index] for index in sorted(velocities)]

    payload = dict(header.get("payload") or {})
    payload["model"] = model
    payload["optimizer"] = optimizer
    return payload


def is_pure_checkpoint(path: str | Path) -> bool:
    """快速嗅探：文件头是否属于 ``.mgckpt`` 格式（不加载张量数据）。"""

    target = Path(path)
    try:
        with target.open("rb") as handle:
            prefix = handle.read(4)
            if len(prefix) < 4:
                return False
            (header_length,) = struct.unpack("<I", prefix)
            header = handle.read(header_length)
        return json.loads(header.decode("utf-8")).get("format") == FORMAT
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False


__all__ = [
    "FORMAT",
    "VERSION",
    "PureCheckpointError",
    "save_pure_checkpoint",
    "load_pure_checkpoint",
    "is_pure_checkpoint",
]
