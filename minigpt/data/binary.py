"""数据层二进制与 JSON 文件工具（纯标准库，显式小端）。

约定
----
* token 二进制统一为 **uint16 little-endian**（与 CONSTRUCTION.md §7.3 一致），
  读写使用 ``array.array("H")`` + 按 ``sys.byteorder`` 显式 byteswap，因此在
  大端小端机器上生成的文件字节完全相同。
* 所有写盘操作先写同目录临时文件，再 ``os.replace`` 原子替换，避免中断留下
  半截文件。
* ``sha256_file`` 用于校验下载语料与产出文件的完整性。
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

#: 读取文件哈希时的分块大小（1 MiB）。
_HASH_CHUNK = 1024 * 1024


class Uint16FormatError(ValueError):
    """uint16 二进制文件格式错误。"""


# ----------------------------------------------------------------------
# 原子写
# ----------------------------------------------------------------------
def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """原子写入 bytes（UTF-8/二进制通用）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """原子写入文本（默认 UTF-8，换行不做平台转换）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    """把对象写为 UTF-8 JSON（ensure_ascii=False，便于人工阅读中文）。"""
    payload = json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    return atomic_write_text(path, payload)


def read_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------------------
# uint16 小端 token 文件
# ----------------------------------------------------------------------
def write_uint16(path: str | Path, values: Iterable[int]) -> int:
    """把 token id 序列写为 uint16 小端二进制文件，返回字节数。

    值范围必须是 0..65535；空序列会写出 0 字节文件（合法）。
    """
    buffer = array.array("H")
    try:
        buffer.extend(int(value) for value in values)
    except OverflowError as exc:  # 负数或 >65535
        raise Uint16FormatError(f"token id 超出 uint16 范围：{exc}") from exc
    if sys.byteorder != "little":
        buffer.byteswap()
    data = buffer.tobytes()
    atomic_write_bytes(path, data)
    return len(data)


def read_uint16(path: str | Path) -> list[int]:
    """读取 uint16 小端二进制文件并返回 int 列表。"""
    raw = Path(path).read_bytes()
    if len(raw) % 2 != 0:
        raise Uint16FormatError(f"{path} 字节数 {len(raw)} 不是 2 的倍数，不是合法 uint16 文件")
    buffer = array.array("H")
    buffer.frombytes(raw)
    if sys.byteorder != "little":
        buffer.byteswap()
    return buffer.tolist()


def read_uint16_array(path: str | Path) -> array.array:
    """读取 uint16 小端文件，返回紧凑的 ``array('H')``（大数据集省内存）。"""
    raw = Path(path).read_bytes()
    if len(raw) % 2 != 0:
        raise Uint16FormatError(f"{path} 字节数 {len(raw)} 不是 2 的倍数，不是合法 uint16 文件")
    buffer = array.array("H")
    buffer.frombytes(raw)
    if sys.byteorder != "little":
        buffer.byteswap()
    return buffer


# ----------------------------------------------------------------------
# 哈希
# ----------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """返回 bytes 的 sha256 十六进制小写摘要。"""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = _HASH_CHUNK) -> str:
    """分块计算文件 sha256，返回十六进制小写摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# JSONL（SFT 数据）
# ----------------------------------------------------------------------
def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    """把记录列表写为 UTF-8 JSONL（每行一个 JSON 对象，末尾带换行）。"""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL；空行忽略，非对象行报错并指出行号。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no} 不是 JSON 对象")
            records.append(record)
    return records


__all__ = [
    "Uint16FormatError",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "read_json",
    "write_uint16",
    "read_uint16",
    "read_uint16_array",
    "sha256_bytes",
    "sha256_file",
    "write_jsonl",
    "read_jsonl",
]
