"""静态资源服务：MIME 表与带路径穿越防护的文件读取。

从 server.py 拆出（阶段 4），以 mixin 形式并入 DashboardHandler，公共方法名
与行为保持不变。本模块只依赖标准库。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from minigpt.web.errors import ApiError

STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json; charset=utf-8",
}


class StaticFilesMixin:
    """静态文件服务（并入 DashboardHandler）。

    依赖宿主提供 ``self.server.app``（含 ``web_dir``）与 ``_send_bytes``。
    """

    def _serve_static(self, relative: str) -> None:
        """安全解析相对路径并返回静态资源（拒绝目录穿越）。"""
        app = self.server.app  # type: ignore[attr-defined]
        web_root = app.web_dir.resolve()
        clean = relative.replace("\\", "/").lstrip("/")
        if not clean or "\x00" in clean:
            raise ApiError(404, "静态文件不存在")
        try:
            candidate = (web_root / clean).resolve()
        except OSError as exc:
            raise ApiError(400, "非法静态路径") from exc
        if candidate != web_root and not candidate.is_relative_to(web_root):
            raise ApiError(404, "禁止访问该路径")
        if not candidate.is_file():
            raise ApiError(404, f"静态文件不存在: {clean}")
        try:
            body = candidate.read_bytes()
        except OSError as exc:
            raise ApiError(404, f"静态文件无法读取: {clean}") from exc
        content_type = (
            STATIC_MIME.get(candidate.suffix.lower())
            or "application/octet-stream"
        )
        # 内容哈希 ETag：no-cache 下浏览器必须回源校验，命中 304 省带宽，
        # 内容变化立即生效，杜绝启发式缓存导致的“旧前端”问题。
        etag = f'"{hashlib.sha256(body).hexdigest()[:32]}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self._send_bytes(200, body, content_type, extra_headers={"ETag": etag})


__all__ = ["STATIC_MIME", "StaticFilesMixin"]