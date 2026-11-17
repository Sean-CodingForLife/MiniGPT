"""错误诊断约定：完整诊断留本地日志，对外消息脱敏。

纯标准库模块，供 ``minigpt.web.server`` 与 ``minigpt.web.inference_worker``
共用（因此绝不能 import torch）：

* :func:`public_message` 把异常/stderr 文本收敛成一行对外摘要：去掉堆栈帧、
  绝对路径（项目根、解释器、用户目录、临时目录等）与超长内容；
* :func:`append_diagnostic` 把完整上下文（含 traceback）追加到本地日志文件，
   best-effort，绝不因为日志写失败而影响 API。
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

MAX_PUBLIC_MESSAGE_CHARS = 600
_STDERR_TAIL_LINES = 8

# Windows 盘符绝对路径 / UNC 路径 / 常见 POSIX 文件系统绝对路径。
# POSIX 只匹配真实文件系统前缀，避免误伤 "/api/v1/..." 这类 API 路径。
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
_UNC_PATH_RE = re.compile(r"\\\\[A-Za-z0-9_.-]+\\[^\s'\"<>|]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:home|Users|tmp|var|usr|opt|etc|private|root|mnt|media|srv|proc|dev|run|snap|"
    r"Library|System|Applications)(?:/[^\s'\"<>|]*)?"
)
# 堆栈帧 / traceback 头 / 纯 caret 行（对外摘要里一律丢弃）。
_FRAME_RE = re.compile(
    r'^\s*(?:Traceback \(most recent call last\):'
    r'|File "[^"]*", line \d+'
    r'|During handling of the above exception'
    r'|The above exception was the direct cause'
    r'|[\^~]+\s*$)'
)

PLACEHOLDER = "<path>"


def _known_roots(roots: Iterable[str | Path]) -> list[str]:
    """收集需要脱敏的根路径（调用方给定 + 解释器/用户目录/临时目录）。"""

    collected: list[str] = []
    for root in roots:
        try:
            text = str(Path(root).expanduser())
        except (OSError, ValueError, RuntimeError):
            continue
        if text:
            collected.append(text)
    collected.append(sys.executable)
    try:
        collected.append(str(Path.home()))
    except (OSError, RuntimeError):
        pass
    collected.append(tempfile.gettempdir())
    return sorted(set(collected), key=len, reverse=True)


def sanitize_paths(text: str, roots: Iterable[str | Path] = ()) -> str:
    """把文本中的绝对路径替换成 ``<path>`` 占位符（已知根优先，最长匹配）。"""

    flags = re.IGNORECASE if os.name == "nt" else 0
    for root in _known_roots(roots):
        text = re.compile(re.escape(root), flags).sub(PLACEHOLDER, text)
    text = _WINDOWS_PATH_RE.sub(PLACEHOLDER, text)
    text = _UNC_PATH_RE.sub(PLACEHOLDER, text)
    text = _POSIX_PATH_RE.sub(PLACEHOLDER, text)
    return text


def public_message(
    text: str,
    roots: Iterable[str | Path] = (),
    max_chars: int = MAX_PUBLIC_MESSAGE_CHARS,
) -> str:
    """把异常/stderr 文本收敛成可对外展示的单行脱敏摘要。

    含 traceback 时只保留最后一个异常块（帧与源码回显全部丢弃）；
    普通 stderr 保留最后几行；随后统一脱敏并截断长度。
    """

    text = str(text)
    if "Traceback (most recent call last)" in text:
        kept: list[str] = []
        for line in reversed(text.splitlines()):
            if _FRAME_RE.match(line):
                break
            if line.strip():
                kept.append(line.strip())
        summary = " | ".join(reversed(kept))
        text = summary or text
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = " | ".join(lines[-_STDERR_TAIL_LINES:])
    text = sanitize_paths(text, roots)
    if len(text) > max_chars:
        # 超长输出（如无换行的噪声刷屏）：错误原因通常在末尾，
        # 保留尾部而不是开头，避免真正的失败信息被截掉。
        text = "…" + text[-(max_chars - 1):]
    return text


def append_diagnostic(
    log_path: str | Path,
    context: str,
    detail: str = "",
    exc: BaseException | None = None,
) -> None:
    """把完整诊断追加到本地日志（best-effort，绝不抛错）。"""

    try:
        target = Path(log_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {context}\n")
            if detail:
                handle.write(detail.rstrip() + "\n")
            if exc is not None:
                handle.write(
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                )
    except OSError:
        pass


__all__ = [
    "MAX_PUBLIC_MESSAGE_CHARS",
    "PLACEHOLDER",
    "append_diagnostic",
    "public_message",
    "sanitize_paths",
]