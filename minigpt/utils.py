"""CLI、Trainer 与 Web 后端共用的标准库小工具。

这里集中处理三件容易出错的事：原子写文件（避免读者看到半截 JSON）、
Windows 下的替换重试（看板并发读取时可能出现共享冲突）、以及随机种子/运行
目录命名。整个模块**不依赖 torch**，只有 ``seed_everything`` 会按需惰性导入。
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """返回 UTC 时间戳（ISO 8601，精确到秒），用于各类 created_at 字段。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: str | Path) -> Path:
    """创建目录（含父目录）并返回 ``Path``；目录已存在时不报错。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def replace_with_retry(source: str | Path, target: str | Path, attempts: int = 12, delay: float = 0.05) -> None:
    """Windows 友好的原子替换。

    看板可能在 Trainer 替换 ``status.json`` 的同一瞬间读取该文件，Windows 上
    会抛 ``PermissionError``。由于源文件是私有临时文件、替换本身是原子的，
    重试若干次是安全的；重试耗尽后把最后一次错误抛出。
    """

    last_error: OSError | None = None
    for _ in range(max(1, attempts)):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:  # Windows sharing violation
            last_error = error
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    os.replace(source, target)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """原子写入 bytes：先写到同目录临时文件，fsync 后再 ``os.replace``。

    返回最终目标路径。写入或替换失败时会清理临时文件，避免残留 *.tmp。
    """

    target = Path(path)
    ensure_dir(target.parent)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """原子写入文本（默认 UTF-8，``newline="\\n"`` 避免平台换行差异）。"""

    target = Path(path)
    ensure_dir(target.parent)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_json(path: str | Path, data: Any) -> None:
    """原子写入 JSON（中文不转义、缩进 2、键排序），读者不会看到半截文件。"""

    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, payload)


def read_json(path: str | Path) -> Any:
    """按 UTF-8 读取并解析 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rng_state_to_tuple(value: Any) -> Any:
    """把 JSON 反序列化后的 Python RNG 状态（list）递归转回 tuple。

    random.getstate() 返回嵌套 tuple，经 JSON 存取后变成 list，setstate
    拒绝 list；检查点续训前用本函数还原。
    """
    if isinstance(value, list):
        return tuple(rng_state_to_tuple(item) for item in value)
    return value


def safe_name(value: str, fallback: str = "run") -> str:
    """把任意字符串转成可安全用作目录名的片段。

    只保留字母、数字、``.``、``_``、``-``，其余字符折叠为 ``-``；结果为空
    （或只剩分隔符）时返回 ``fallback``。
    """
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip(".-")
    return text or fallback


def generate_run_id(stage: str, seed: int) -> str:
    """生成形如 ``20260101-120000-pretrain-s1337-abcd`` 的运行 id。

    时间戳便于排序，stage/seed 便于人读，末尾 4 位 UUID 片段避免同秒冲突。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return safe_name(f"{stamp}-{stage}-s{seed}-{uuid.uuid4().hex[:4]}")


_HASH_SEED_WARNED = False


def ensure_hash_seed(seed: int) -> None:
    """检查 PYTHONHASHSEED：未设置时给出一次性提示，不重启进程。

    Python 的哈希随机化在解释器启动时确定，运行时改 os.environ 无效。
    旧实现会带环境变量重启整个进程——作为库函数这过于侵入（子进程、
    退出码透传、Windows 兼容问题），且让调用方无法预期控制流。现在只
    提示用户在启动前设置；需要严格复现哈希顺序时按提示操作即可。
    """

    global _HASH_SEED_WARNED
    if os.environ.get("PYTHONHASHSEED") is not None:
        return
    if _HASH_SEED_WARNED:
        return
    _HASH_SEED_WARNED = True
    print(
        f"[utils] 提示：PYTHONHASHSEED 未设置，哈希顺序不可复现。"
        f"如需严格复现，请在启动前设置 PYTHONHASHSEED={int(seed)}。",
        file=sys.stderr,
    )


def seed_everything(seed: int) -> None:
    """播种 Python 的 ``random``，并在可导入时一并播种 torch / CUDA。

    torch 采用惰性导入，保证只用看板的部署不需要 torch。
    注意：PYTHONHASHSEED 必须在进程启动前设置（见 ensure_hash_seed），
    这里不再假装运行时设置能生效。
    """

    random.seed(seed)
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fmt_seconds(seconds: float) -> str:
    """把秒数格式化成 ``1h02m03s`` / ``2m03s`` / ``45s``；负数按 0 处理。"""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def now_monotonic() -> float:
    """单调时钟（秒），用于计算耗时，不受系统时间调整影响。"""
    return time.monotonic()


__all__ = [
    "utc_now",
    "ensure_dir",
    "replace_with_retry",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "read_json",
    "rng_state_to_tuple",
    "safe_name",
    "generate_run_id",
    "ensure_hash_seed",
    "seed_everything",
    "fmt_seconds",
    "now_monotonic",
]
