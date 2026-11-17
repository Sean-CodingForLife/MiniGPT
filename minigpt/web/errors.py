"""网页层共享的 HTTP 错误类型。

单独成模块是为了让拆分出的子模块（推理子进程管理、静态文件服务等）都能
引用它，而不必反向导入庞大的 server.py（避免循环依赖）。server.py 仍
re-export ``ApiError``，对外导入路径不变。
"""

from __future__ import annotations


class ApiError(Exception):
    """携带 HTTP 状态码的 API 错误。"""

    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None) -> None:
        """构造带 HTTP 状态码与可选响应头的 API 错误。"""
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers = headers or {}


__all__ = ["ApiError"]