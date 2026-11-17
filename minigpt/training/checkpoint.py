"""checkpoint 读写：单个 ``.pt`` 文件内封装模型、优化器、RNG 与 tokenizer 状态。

checkpoint 是**自包含**的：tokenizer 状态内嵌，因此推理侧的 ``chat.py`` / 导出
工具只凭这一个文件即可重建模型与分词器，运行目录可以在机器之间搬迁。

写入一律「同目录临时文件 + fsync + 原子替换」，避免断电/崩溃留下半截文件；
读取走 ``weights_only=True``，不允许 pickle 执行任意代码。字段契约见
``CONSTRUCTION.md`` 的 checkpoint 章节，版本号由 :data:`CHECKPOINT_VERSION`
把关（:func:`validate_checkpoint`）。
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from minigpt.utils import replace_with_retry

# torch 只在真正读写 .pt / 捕获 torch RNG 时才需要（延迟导入）：
# pure 后端与无 torch 环境的看板/训练器导入链不被阻断。

CHECKPOINT_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """分块计算文件 sha256（默认 1 MiB 一块，大文件也不整读进内存）。

    参数 ``path`` 为目标文件；``chunk_size`` 为每次读取的字节数。
    返回十六进制摘要字符串；文件不可读时由 :class:`OSError` 向上抛出。
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """抓取 Python / torch / CUDA 的随机数状态，供续训复现。

    返回 ``{"python": ..., "torch": ...}``，可用 CUDA 时追加 ``"cuda"``
    （每张卡的 RNG 状态列表）。torch 在本函数内延迟导入。
    """

    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """恢复 :func:`capture_rng_state` 抓取的状态；``None``/空 dict 时什么都不做。

    只有 payload 里真的出现 torch/CUDA 字段时才延迟导入 torch；CUDA 状态在
    没有 GPU 的机器上会被静默跳过（跨机器续训不因设备差异失败）。
    """

    if not state:
        return
    if "torch" in state or "cuda" in state:
        import torch
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """把 ``payload`` 写入 ``path``：先写临时文件，再原子替换目标。

    原子性约定（崩溃安全）：
    * 临时文件必须与目标同目录（跨盘 replace 不是原子的）；
    * 写入后 ``fsync``，随后**先关闭句柄再替换**——Windows 拒绝替换仍被打开
      的文件；
    * 任何异常都会尝试删除临时文件后原样抛出，目标文件保持旧版本不变。

    参数 ``payload`` 为待保存字典（模型/优化器/RNG 等，由训练器组装）。
    """

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    closed = False
    try:
        torch.save(payload, temporary)
        # 通过 mkstemp 返回的可写 fd 做 fsync（Windows 上只读句柄无法 flush），
        # 然后在原子替换之前关闭：Windows 不允许替换仍被打开的文件。
        os.fsync(fd)
        os.close(fd)
        closed = True
        replace_with_retry(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        if not closed:
            os.close(fd)


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """读取 checkpoint 字典；文件缺失抛 :class:`FileNotFoundError`。

    参数 ``map_location`` 交给 ``torch.load``（默认 ``"cpu"``，便于无 GPU 环境
    读取 CUDA checkpoint）。返回值必须是 dict，否则抛 :class:`ValueError`
    ——这里只做「是不是字典」的粗校验，字段完整性由
    :func:`validate_checkpoint` 负责。
    """

    import torch

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint not found: {target}")
    # 安全：用 weights_only=True 读取，恶意 .pt 无法执行任意 pickle 代码。
    # 训练器把 ``torch_version`` 存成 ``TorchVersion`` 对象，它不在 torch 默认
    # 允许清单里，因此显式注册（只是一个无害的版本包装）。payload 里其余内容
    # （tensor、dict、int、RNG 元组）本就在允许清单内。
    # 依赖 requirements.txt 中固定的 torch>=2.4；这里刻意不提供旧版本的
    # 不安全回退路径。
    safe_globals = [torch.torch_version.TorchVersion]
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(target, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return payload


def validate_checkpoint(payload: dict[str, Any]) -> None:
    """校验 checkpoint 必需字段与格式版本；不合法抛 :class:`ValueError`。

    只需要读到字段名，不需要 torch，因此看板/导出工具可以在无 torch 环境下
    先验证再决定是否加载权重。``format_version`` 必须**精确等于**
    :data:`CHECKPOINT_VERSION`，不做向前兼容猜测。
    """

    required = {
        "format_version",
        "backend",
        "stage",
        "model",
        "optimizer",
        "iter_num",
        "best_val_loss",
        "config",
        "model_config",
        "tokenizer_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    version = payload.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint format_version={version!r}, expected {CHECKPOINT_VERSION}; "
            "see CONSTRUCTION.md for the current checkpoint contract"
        )


__all__ = [
    "CHECKPOINT_VERSION",
    "sha256_file",
    "capture_rng_state",
    "restore_rng_state",
    "save_checkpoint",
    "load_checkpoint",
    "validate_checkpoint",
]
