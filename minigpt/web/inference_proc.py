"""推理子进程管理：流式 worker 启动/关闭、并发槽位、stderr 采集、批量调用。

从 server.py 拆出（阶段 4），以 mixin 形式并入 DashboardApp，公共方法名与
行为保持不变；测试仍可对 ``app._inference_command`` / ``app._run_inference_worker``
做 monkeypatch。本模块只依赖标准库，不导入 torch。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from minigpt.web.errors import ApiError

INFERENCE_TIMEOUT_S = 180.0
MAX_INFERENCE_CONCURRENCY = 2
MAX_STREAM_LINE_BYTES = 4_000_000
STDERR_CAPTURE_BYTES = 64 * 1024


def _stream_event_type(text: str) -> str | None:
    """Return the ``type`` of one NDJSON stream line, or ``None`` if invalid."""

    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict):
        return str(event.get("type"))
    return None


class _StderrCapture:
    """后台线程持续读取 worker stderr，只保留最后 STDERR_CAPTURE_BYTES 字节。"""

    def __init__(self, stream: Any) -> None:
        """包装真实流并启动后台泵线程。"""
        self._stream = stream
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """持续读取子进程 stderr 并写入有界缓冲区。"""
        try:
            buffer = getattr(self._stream, "buffer", None)
            if buffer is not None:
                # 文本管道：直接读底层二进制缓冲，read1 返回当前可用数据，
                # 不像 read(n) 那样等满 n 字节，尾部内容也能及时采集。
                while True:
                    chunk = buffer.read1(4096)
                    if not chunk:
                        break
                    self._append(chunk)
            else:
                for line in self._stream:
                    data = line.encode("utf-8", errors="replace") if isinstance(line, str) else line
                    self._append(data)
        except (OSError, ValueError):
            pass

    def _append(self, data: bytes) -> None:
        """追加数据并在超过上限时截断。"""
        with self._lock:
            self._buffer.extend(data)
            if len(self._buffer) > STDERR_CAPTURE_BYTES:
                del self._buffer[: len(self._buffer) - STDERR_CAPTURE_BYTES]

    def text(self) -> str:
        """返回当前已捕获的文本（线程安全）。"""
        self._thread.join(timeout=5)
        with self._lock:
            return bytes(self._buffer).decode("utf-8", errors="replace")


class InferenceProcessMixin:
    """推理子进程生命周期管理（并入 DashboardApp）。

    依赖宿主提供：``job_manager``、``project_root``、``log_diagnostic``、
    ``_public``，以及 __init__ 中初始化的 ``_inference_lock`` /
    ``_inference_processes`` / ``_inference_stderr`` / ``_inference_slots``。
    """

    def _inference_command(self) -> list[str]:
        """Fixed worker command; tests may monkeypatch this to a stub worker."""

        return [
            self.job_manager.python_executable,
            "-u",
            "-m",
            "minigpt.web.inference_worker",
        ]

    def _inference_env(self) -> dict[str, str]:
        """构造推理子进程环境变量（编码/无缓冲/根目录）。"""
        env = os.environ.copy()
        env["MINIGPT_ROOT"] = str(self.project_root)
        env["PYTHONIOENCODING"] = "utf-8"
        # worker 以 -m minigpt.web.inference_worker 启动，解释器必须先找到
        # minigpt 包：把服务代码所在根加入 PYTHONPATH（临时项目根场景下
        # cwd 里没有包，仅靠 MINIGPT_ROOT 不够）。
        package_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if package_root not in parts:
            parts.insert(0, package_root)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def start_inference_process(
        self, request: dict[str, Any]
    ) -> subprocess.Popen[str]:
        """Start one streaming inference worker and write its request to stdin."""

        if not self._inference_slots.acquire(blocking=False):
            raise ApiError(429, "推理并发已达上限，请稍后重试")
        command = self._inference_command()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root),
                env=self._inference_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError as exc:
            # acquire 已成功：失败路径必须归还槽位，否则并发额度永久减少。
            self._inference_slots.release()
            self.log_diagnostic("inference interpreter not found", detail=command[0], exc=exc)
            raise ApiError(500, "找不到推理解释器，请检查看板的 Python 配置") from exc
        with self._inference_lock:
            self._inference_processes.add(process)
            self._inference_stderr[process] = _StderrCapture(process.stderr)
        try:
            if process.stdin is None:
                raise OSError("worker stdin 不可用")
            process.stdin.write(json.dumps(request, ensure_ascii=False))
            process.stdin.close()
        except OSError as exc:
            self.close_inference_process(process)
            raise ApiError(500, "无法向推理子进程写入请求") from exc
        return process

    @staticmethod
    def kill_inference_process(process: subprocess.Popen[str]) -> None:
        """Kill a worker that overruns the timeout (also unblocks readline)."""

        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def close_inference_process(self, process: subprocess.Popen[str]) -> None:
        """Terminate a worker and release its pipes; safe to call twice."""

        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        with self._inference_lock:
            # 只有首次关闭才归还槽位：重复调用不得二次 release（BoundedSemaphore
            # 会抛 ValueError），与"safe to call twice"的契约一致。
            first_close = process in self._inference_processes
            self._inference_processes.discard(process)
            self._inference_stderr.pop(process, None)
        if first_close:
            self._inference_slots.release()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass

    def inference_error_tail(self, process: subprocess.Popen[str]) -> str:
        """Best-effort error message from a failed worker (call after exit).

        完整 stderr 留服务端诊断日志；对外只给脱敏摘要。
        """

        capture = self._inference_stderr.get(process)
        raw = capture.text() if capture is not None else ""
        if raw.strip():
            self.log_diagnostic(
                f"inference stream worker exit={process.returncode}", detail=raw
            )
            return self._public(raw)
        return f"推理子进程失败（exit={process.returncode}）"

    def _run_inference_worker(self, request: dict[str, Any]) -> dict[str, Any]:
        """在线程中运行推理 worker 并转发事件。"""
        command = self._inference_command()
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                cwd=str(self.project_root),
                env=self._inference_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=INFERENCE_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            self.log_diagnostic("inference interpreter not found", detail=command[0], exc=exc)
            raise ApiError(500, "找不到推理解释器，请检查看板的 Python 配置") from exc
        except subprocess.TimeoutExpired as exc:
            self.log_diagnostic("inference timeout", detail=json.dumps(request, ensure_ascii=False), exc=exc)
            raise ApiError(504, f"生成超时（>{INFERENCE_TIMEOUT_S:.0f}s）") from exc
        if completed.returncode != 0:
            self.log_diagnostic(
                f"inference worker exit={completed.returncode}",
                detail=f"--- stdout ---\n{completed.stdout or ''}\n--- stderr ---\n{completed.stderr or ''}",
            )
            # worker 的结构化错误（stdout 最后一行 JSON error）优先；否则用 stderr 摘要。
            structured: dict[str, Any] | None = None
            for line in reversed((completed.stdout or "").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                except ValueError:
                    continue
                if isinstance(candidate, dict) and candidate.get("type") == "error":
                    structured = candidate
                break
            if structured is not None:
                code = str(structured.get("code") or "WorkerError")
                detail = self._public(str(structured.get("message") or ""))
                message = f"生成失败（{code}）: {detail}" if detail else f"生成失败（{code}）"
            else:
                raw = completed.stderr or completed.stdout or ""
                detail = self._public(raw)
                message = (
                    f"生成子进程失败: {detail}"
                    if detail
                    else f"生成子进程失败（exit={completed.returncode}）"
                )
            raise ApiError(500, message)
        lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
        if not lines:
            self.log_diagnostic("inference worker returned no output", detail=completed.stderr or "")
            raise ApiError(500, "生成子进程没有返回结果")
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            self.log_diagnostic("inference worker returned invalid JSON", detail=lines[-1], exc=exc)
            raise ApiError(500, "生成子进程返回了非法 JSON（详见服务端日志）") from exc
        if not isinstance(result, dict) or "steps" not in result or "final" not in result:
            self.log_diagnostic("inference worker returned incomplete result", detail=lines[-1][:2000])
            raise ApiError(500, "生成子进程返回结构不完整")
        return result


__all__ = [
    "INFERENCE_TIMEOUT_S",
    "MAX_INFERENCE_CONCURRENCY",
    "MAX_STREAM_LINE_BYTES",
    "STDERR_CAPTURE_BYTES",
    "InferenceProcessMixin",
    "_StderrCapture",
    "_stream_event_type",
]