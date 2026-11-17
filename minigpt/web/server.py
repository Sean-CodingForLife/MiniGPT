"""MiniGPT 控制面板后端：Python 标准库 HTTP 服务 + JSON API + 静态资源。

契约（见 ``CONSTRUCTION.md`` §6.3 / §7.4 / §7.6）：

* 只用标准库；**模块级绝不导入 torch**，因此看板可在没有 PyTorch 的机器上运行。
* API 固定为 ``/api/v1/...``，运行状态读取 ``runs/<id>/`` 下训练进程写的文件：
  ``status.json``、``events.jsonl``、``metrics.csv``、``samples.jsonl``、
  ``training_report.json``、``run_config.json``、``environment.json``、
  ``resolved_config.json``（看板启动任务时冻结的配置）。
* 只暴露 ``configs/``、``runs/``、``web/`` 三类目录；所有名字先做白名单校验，
  再校验解析后的路径仍在目录内，杜绝路径穿越。
* 没有鉴权，默认只监听 ``127.0.0.1``（与 CONSTRUCTION §6.3 一致）。
* 启动任务走 :class:`~minigpt.training.job_manager.JobManager`：固定脚本、
  冻结配置、默认最多 1 个活动任务，服务关闭时停止活动任务。

本模块把「读运行产物」的容错逻辑集中在此：训练进程可能正在写文件，因此所有
读取都容忍不完整的 JSON 行/文件，绝不因为一次坏读让 API 500。
"""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from minigpt import __version__
from minigpt.config import PRESETS, default_project_root
from minigpt.web import stages
from minigpt.web.diagnostics import append_diagnostic, public_message
from minigpt.web.errors import ApiError
from minigpt.web.inference_proc import (
    INFERENCE_TIMEOUT_S,
    MAX_INFERENCE_CONCURRENCY,
    MAX_STREAM_LINE_BYTES,
    STDERR_CAPTURE_BYTES,
    InferenceProcessMixin,
    _StderrCapture,
    _stream_event_type,
)
from minigpt.web.static_files import STATIC_MIME, StaticFilesMixin
from minigpt.training.project_store import ProjectStore
from minigpt.training.job_manager import (
    ACTIVE_STATUSES,
    ConfigError,
    ConfigStore,
    JobLimitExceeded,
    JobManager,
    JobManagerError,
    JobNotFound,
    JobStartError,
    atomic_write_json,
    iso_from_mtime,
    iso_now,
    validate_run_id,
)

MAX_BODY_BYTES = 1_000_000
EVENT_TAIL_DEFAULT = 500
EVENT_TAIL_MAX = 5000
SAMPLE_LIMIT_DEFAULT = 20
SAMPLE_LIMIT_MAX = 500
INFERENCE_TIMEOUT_S = 180.0
MAX_INFERENCE_CONCURRENCY = 2
SOCKET_TIMEOUT_S = 120.0
MAX_PROMPT_CHARS = 4000
RUN_STALE_SECONDS = 60.0
MAX_PREFIX_TOKENS = 512
MAX_STREAM_LINE_BYTES = 4_000_000
MAX_COMPARE_RUNS = 8
MAX_INFERENCE_VARIANTS = 4
COMPARE_SERIES_COLUMNS = ("train_loss", "val_loss", "val_ppl")
COMPARE_REPORT_KEYS = (
    "status",
    "steps",
    "best_val_loss",
    "best_step",
    "stopped_early",
    "initial_loss",
    "final_train_loss",
    "elapsed_s",
)

# 所有运行产物文件名（用于 summary 的 files 面板与 mtime 计算）
RUN_ARTIFACTS = (
    "resolved_config.json",
    "run_config.json",
    "environment.json",
    "status.json",
    "events.jsonl",
    "metrics.csv",
    "samples.jsonl",
    "training_report.json",
    "minirun_report.json",
    "console.log",
)

# ---------------------------------------------------------------------------
# 安全 JSON：把 NaN/Infinity 转成 null，避免浏览器 JSON.parse 失败
# ---------------------------------------------------------------------------
def _sanitize(value: Any) -> Any:
    """递归把非有限浮点数、Path、元组等转换成严格 JSON 可表示的值。"""

    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN 或 ±Inf
            return None
    return value


def _pick_checkpoint(run_dir: Path) -> Path | None:
    """统一检查点探测：best.pt → best.mgckpt → last.pt → last.mgckpt。"""

    for name in ("best.pt", "best.mgckpt", "last.pt", "last.mgckpt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def _relative_display(base: Path, target: Path) -> str:
    """对外展示的目录路径：根内直接子目录给相对名，其余只保留目录名。

    不做多级相对路径拼接，避免把临时目录/用户目录等中间路径名带出去。
    """

    try:
        resolved = target.resolve()
        root = base.resolve()
    except OSError:
        return target.name or "."
    if resolved == root:
        return "."
    if resolved.parent == root:
        return resolved.name
    return resolved.name or "."


# ---------------------------------------------------------------------------
# 容错读取
# ---------------------------------------------------------------------------
def read_json_tolerant(path: str | Path) -> Any | None:
    """读取 JSON；文件不存在/正在写/损坏时返回 ``None``。"""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_jsonl_tolerant(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL，跳过空行和坏行，并给每条记录补 ``index``（原始行号）。"""

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        record.setdefault("index", index)
        events.append(record)
    return events


def _parse_jsonl_window(
    path: Path, after: int, limit: int
) -> list[dict[str, Any]]:
    """全量解析 JSONL 并取 ``[after, after+limit)`` 行窗口（兜底路径）。"""

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    window = text.splitlines()[after : after + limit]
    events: list[dict[str, Any]] = []
    for offset, line in enumerate(window):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        record.setdefault("index", after + offset)
        events.append(record)
    return events


def read_event_window(
    path: str | Path, after: int = 0, limit: int = EVENT_TAIL_DEFAULT
) -> dict[str, Any]:
    """按**原始行号**增量读取事件，返回 ``events`` 与 ``next_after``。"""

    after = max(0, int(after))
    limit = max(1, min(EVENT_TAIL_MAX, int(limit)))
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"events": [], "next_after": after, "total": after, "after": after}
    lines = text.splitlines()
    total = len(lines)
    window = lines[after : after + limit]
    events: list[dict[str, Any]] = []
    for offset, line in enumerate(window):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        record.setdefault("index", after + offset)
        events.append(record)
    return {
        "events": events,
        "after": after,
        "next_after": after + len(window),
        "total": total,
    }


class JsonlTailCache:
    """按文件偏移增量读取 JSONL 尾部记录（线程安全，进程内每 run 一份游标）。

    训练进程持续追加 ``events.jsonl`` / ``samples.jsonl``，前端每秒轮询；
    全量重读是 O(n)/秒。这里只读上次游标之后的新增字节：

    * 文件变小（截断/轮转）→ 游标重置，从头重读；
    * 半行（无换行结尾）留到下次补齐后再解析，行号按原始行计数；
    * 内存只保留最近 ``keep`` 条记录与总行数，不随文件无限增长。
    """

    def __init__(self, keep: int = 512) -> None:
        """设置每个文件保留的尾部行数上限。"""
        self._keep = max(1, int(keep))
        self._lock = threading.Lock()
        self._cursors: dict[str, dict[str, Any]] = {}

    def invalidate(self, path: str | Path) -> None:
        """丢弃某文件的游标（运行目录删除/重建时调用）。"""

        try:
            key = str(Path(path).resolve())
        except OSError:
            key = str(path)
        with self._lock:
            self._cursors.pop(key, None)

    def tail(self, path: str | Path) -> tuple[list[dict[str, Any]], int]:
        """返回 ``(最近记录副本, 总行数)``；文件缺失时返回 ``([], 0)``。"""

        target = Path(path)
        try:
            key = str(target.resolve())
        except OSError:
            key = str(target)
        with self._lock:
            state = self._cursors.get(key)
            if state is None:
                state = {"offset": 0, "line": 0, "records": [], "total": 0}
                self._cursors[key] = state
            try:
                size = target.stat().st_size
            except OSError:
                # 文件可能被删除/轮转；清空游标，下次出现时从头读。
                state.update(offset=0, line=0, total=0)
                state["records"].clear()
                return [], 0
            if size < state["offset"]:
                state.update(offset=0, line=0, total=0)
                state["records"].clear()
            if size > state["offset"]:
                self._read_new(target, state, size)
            return list(state["records"]), state["total"]

    def window(
        self, path: str | Path, after: int, limit: int
    ) -> tuple[list[dict[str, Any]], int]:
        """按原始行号取 ``[after, after+limit)`` 窗口，返回 ``(记录, 总行数)``。

        请求范围落在缓存之外（如翻页回看）时，回退一次性全量解析，
        保证与旧契约一致；常规尾部轮询走增量缓存。
        """

        records, total = self.tail(path)
        after = max(0, int(after))
        limit = max(1, int(limit))
        if records:
            first = records[0].get("index", 0)
            if after >= first:
                start = after - first
                return records[start : start + limit], total
        # 请求范围在缓存之外（回看旧页）：一次性全量解析兜底。
        return _parse_jsonl_window(Path(path), after, limit), total

    def _read_new(self, target: Path, state: dict[str, Any], size: int) -> None:
        """读取 [offset, size) 的新增字节（调用方持有锁）。"""

        records: list[dict[str, Any]] = state["records"]
        try:
            with target.open("rb") as handle:
                handle.seek(state["offset"])
                data = handle.read(size - state["offset"])
        except OSError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return  # 还没有完整行，等下次轮询
        complete = data[: end + 1]
        state["offset"] += len(complete)
        line_number = state["line"]
        for raw in complete.decode("utf-8", errors="replace").splitlines():
            index = line_number
            line_number += 1
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            record.setdefault("index", index)
            records.append(record)
        state["line"] = line_number
        state["total"] = line_number
        if len(records) > self._keep:
            del records[: len(records) - self._keep]


def read_text_tail(path: str | Path, line_count: int = 40) -> str:
    """高效读取文本文件末尾若干行（大文件也最多读 512KB）。"""

    try:
        target = Path(path)
        size = target.stat().st_size
        with target.open("rb") as handle:
            if size > 512 * 1024:
                handle.seek(size - 512 * 1024)
            data = handle.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max(1, int(line_count)) :])
    except OSError:
        return ""


def read_metrics(path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
    """解析 ``metrics.csv``（表头 + 数字行，容忍空值/坏值）。"""

    target = Path(path)
    if not target.is_file():
        return [], []
    try:
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = [str(column).strip() for column in next(reader, [])]
            if not columns:
                return [], []
            rows: list[dict[str, Any]] = []
            for raw in reader:
                if not raw or not any(str(cell).strip() for cell in raw):
                    continue
                row: dict[str, Any] = {}
                for index, column in enumerate(columns):
                    value = raw[index].strip() if index < len(raw) else ""
                    if value == "":
                        row[column] = None
                        continue
                    try:
                        row[column] = int(value)
                        continue
                    except ValueError:
                        pass
                    try:
                        row[column] = float(value)
                    except ValueError:
                        row[column] = value
                rows.append(row)
            return columns, rows
    except (OSError, csv.Error, UnicodeDecodeError):
        return [], []


class MetricsCache:
    """按 (mtime, size) 失效的 metrics.csv 解析缓存（线程安全）。

    训练进程每秒追加 metrics.csv，看板每秒轮询；旧实现每次请求都全量重读
    解析。文件未变化时直接复用上次结果，变化时才重新解析。
    """

    def __init__(self) -> None:
        """初始化以 (路径, mtime, size) 为键的 metrics 缓存。"""
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, int, list[str], list[dict[str, Any]]]] = {}

    def read(self, path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
        """读取并解析 metrics.csv；文件未变化时直接返回缓存。"""
        target = Path(path)
        key = str(target)
        try:
            stat = target.stat()
            signature = (stat.st_mtime, stat.st_size)
        except OSError:
            with self._lock:
                self._entries.pop(key, None)
            return [], []
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached[0] == signature[0] and cached[1] == signature[1]:
                return cached[2], cached[3]
        columns, rows = read_metrics(target)
        with self._lock:
            self._entries[key] = (signature[0], signature[1], columns, rows)
        return columns, rows

    def invalidate(self, path: str | Path) -> None:
        """使某个 metrics 文件的缓存失效。"""
        with self._lock:
            self._entries.pop(str(Path(path)), None)


def _extract_config(resolved: Any, run_config: Any) -> dict[str, Any]:
    """取出完整训练配置：优先 JobManager 冻结的 ``resolved_config.json``。"""

    if isinstance(resolved, dict) and resolved:
        return resolved
    if isinstance(run_config, dict):
        inner = run_config.get("config")
        if isinstance(inner, dict):
            return inner
        return run_config
    return {}


def _interpolate_report(report: Any) -> dict[str, Any] | None:
    """训练报告可能是 ``None`` 或非对象，统一成 dict | None。"""

    if isinstance(report, dict):
        return report
    return None


# ---------------------------------------------------------------------------
# 应用层：把文件系统与 JobManager 组合成 JSON API
# ---------------------------------------------------------------------------
class DashboardApp(InferenceProcessMixin):
    """控制面板应用对象（不含 socket 细节，便于单独测试）。

    推理子进程管理经 InferenceProcessMixin 并入（见 inference_proc.py）。
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        runs_dir: str | Path | None = None,
        web_dir: str | Path | None = None,
        configs_dir: str | Path | None = None,
        projects_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        dist_dir: str | Path | None = None,
        max_jobs: int = 1,
        command_builder: Any = None,
        stop_timeout: float = 5.0,
        python_executable: str | None = None,
        verbose: bool = False,
    ) -> None:
        """装配看板依赖：目录、任务管理器、缓存、项目/配置存储。"""
        root = Path(project_root or default_project_root()).expanduser().resolve()
        self.project_root = root

        def resolve(value: str | Path | None, fallback: str) -> Path:
            candidate = Path(value) if value is not None else Path(fallback)
            if not candidate.is_absolute():
                candidate = root / candidate
            return candidate.expanduser().resolve()

        self.runs_dir = resolve(runs_dir, "runs")
        self.web_dir = resolve(web_dir, "web")
        self.configs_dir = resolve(configs_dir, "configs")
        self.data_dir = resolve(data_dir, "data")
        self.dist_dir = resolve(dist_dir, "dist")
        self.config_store = ConfigStore(self.configs_dir, project_root=root)
        self.projects_dir = resolve(projects_dir, "projects")
        self.project_store = ProjectStore(self.projects_dir, project_root=root)
        self.job_manager = JobManager(
            project_root=root,
            runs_dir=self.runs_dir,
            max_jobs=max_jobs,
            command_builder=command_builder,
            stop_timeout=stop_timeout,
            python_executable=python_executable,
        )
        # N08：评估进程收尾时向该 run 的 events.jsonl 写 eval_end 事件。
        self.job_manager.eval_finish_hook = self._on_eval_finished
        self.verbose = bool(verbose)
        self.started_at = time.time()
        self.started_at_iso = iso_now()
        self._inference_lock = threading.Lock()
        self._inference_processes: set[subprocess.Popen[str]] = set()
        self._inference_stderr: dict[subprocess.Popen[str], _StderrCapture] = {}
        self._inference_slots = threading.BoundedSemaphore(MAX_INFERENCE_CONCURRENCY)
        # F27：轮询增量读取游标（每 run 每文件一份，线程安全）。
        self._events_cache = JsonlTailCache(keep=max(64, EVENT_TAIL_DEFAULT))
        self._samples_cache = JsonlTailCache(keep=max(64, SAMPLE_LIMIT_MAX))
        self._metrics_cache = MetricsCache()

    # -- 诊断日志（完整细节留本地，对外响应脱敏） ---------------------------
    @property
    def diagnostic_log_path(self) -> Path:
        """返回诊断日志文件路径（按需创建目录）。"""
        return self.runs_dir / "dashboard.log"

    def log_diagnostic(
        self, context: str, detail: str = "", exc: BaseException | None = None
    ) -> None:
        """把完整诊断追加到 ``runs/dashboard.log``（best-effort）。"""

        append_diagnostic(self.diagnostic_log_path, context, detail=detail, exc=exc)

    def _public(self, message: str) -> str:
        """对外错误摘要：去掉绝对路径/堆栈并截断长度。"""

        return public_message(message, roots=(self.project_root, self.runs_dir))

    # -- 生命周期 -----------------------------------------------------------
    def close(self) -> None:
        """服务关闭：停止活动训练任务与残留推理子进程，避免孤儿进程。"""

        with self._inference_lock:
            processes = list(self._inference_processes)
        for process in processes:
            self.close_inference_process(process)
        self.job_manager.shutdown()

    # -- 基础信息 -----------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """健康检查：返回状态、版本与关键目录。"""
        return {
            "status": "ok",
            "service": "minigpt-dashboard",
            "version": __version__,
            "time": iso_now(),
            "started_at": self.started_at_iso,
            "uptime_s": round(time.time() - self.started_at, 1),
            "active_jobs": self.job_manager.active_count(),
            "max_jobs": self.job_manager.max_jobs,
            "project_root": _relative_display(self.project_root, self.project_root),
        }

    def presets(self) -> dict[str, Any]:
        """返回可用训练预设及其默认字段。"""
        names = sorted(PRESETS)
        return {
            "names": names,
            "presets": {name: PRESETS[name].to_dict() for name in names},
        }

    def system(self) -> dict[str, Any]:
        """返回系统信息（Python/torch/设备/线程）供页面展示。"""
        torch_available = importlib.util.find_spec("torch") is not None
        torch_version = None
        if torch_available:
            try:
                torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                torch_version = None
            except Exception:  # noqa: BLE001 - 元数据损坏不应让 API 失败
                torch_version = None
        return {
            "time": iso_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_available": torch_available,
            "torch_version": torch_version,
            "backend": "stdlib",
            "project_root": _relative_display(self.project_root, self.project_root),
            "runs_dir": _relative_display(self.project_root, self.runs_dir),
            "web_dir": _relative_display(self.project_root, self.web_dir),
            "configs_dir": _relative_display(self.project_root, self.configs_dir),
            "max_jobs": self.job_manager.max_jobs,
            "active_jobs": self.job_manager.active_count(),
            "preset_count": len(PRESETS),
            "server_started_at": self.started_at_iso,
        }

    # -- 配置 ---------------------------------------------------------------
    def list_configs(self) -> dict[str, Any]:
        """列出 configs/ 下的配置名。"""
        configs = self.config_store.list()
        return {"configs": configs, "configs_dir": str(self.configs_dir), "count": len(configs)}

    def create_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """校验并新建一个配置文件。"""
        if not isinstance(body, dict):
            raise ConfigError("请求体必须是 JSON 对象")
        if "name" not in body:
            raise ConfigError("缺少配置名 name")
        name = body["name"]
        if "config" in body:
            config_data = body["config"]
        else:
            config_data = {key: value for key, value in body.items() if key != "name"}
        saved = self.config_store.save(str(name), config_data)
        saved["valid"] = True
        return saved

    def get_config(self, name: str) -> dict[str, Any]:
        """读取指定配置内容。"""
        return self.config_store.get(name)

    def save_config(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        """覆盖保存指定配置。"""
        if not isinstance(body, dict):
            raise ConfigError("请求体必须是 JSON 对象")
        if "config" in body:
            config_data = body["config"]
        else:
            config_data = body
        saved = self.config_store.save(name, config_data)
        saved["valid"] = True
        saved["updated"] = True
        return saved

    def delete_config(self, name: str) -> dict[str, Any]:
        """删除指定配置。"""
        return self.config_store.delete(name)

    def validate_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """校验一份完整（或部分覆盖预设的）配置；结果用 ``valid`` 表达。"""

        from minigpt.training.job_manager import normalize_config  # 局部导入避免循环

        if not isinstance(body, dict):
            return {"valid": False, "errors": ["请求体必须是 JSON 对象"]}
        preset: str | None = None
        overrides: dict[str, Any] | None = None
        if isinstance(body.get("config"), dict):
            config_data = body["config"]
            preset = body.get("preset") if isinstance(body.get("preset"), str) else None
            overrides = body.get("overrides") if isinstance(body.get("overrides"), dict) else None
        else:
            config_data = dict(body)
            raw_preset = config_data.pop("preset", None)
            preset = raw_preset if isinstance(raw_preset, str) else None
            raw_overrides = config_data.pop("overrides", None)
            overrides = raw_overrides if isinstance(raw_overrides, dict) else None
        try:
            config = normalize_config(
                config_data=config_data,
                preset=preset,
                overrides=overrides,
                project_root=self.project_root,
            )
        except ConfigError as exc:
            return {"valid": False, "errors": [str(exc)], "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - 校验接口必须返回结构化结果
            return {
                "valid": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "error": str(exc),
            }
        return {
            "valid": True,
            "errors": [],
            "config": config.to_dict(portable=True),
            "resolved": config.to_dict(),
        }

    # -- 任务 ---------------------------------------------------------------
    def list_jobs(self) -> dict[str, Any]:
        """列出活动任务与历史任务（按更新时间倒序）。"""
        jobs = [record.to_dict() for record in self.job_manager.list_jobs()]
        return {
            "jobs": jobs,
            "count": len(jobs),
            "active": sum(1 for job in jobs if job.get("active")),
            "max_jobs": self.job_manager.max_jobs,
        }

    def get_job(self, job_id: str) -> dict[str, Any]:
        """返回单个任务详情；不存在时 404。"""
        record = self.job_manager.get_job(job_id)
        if record is not None:
            return {"job": record.to_dict(), "source": "memory"}
        run_dir = self._run_dir(job_id)
        job = self._disk_job_view(job_id, run_dir)
        return {"job": job, "source": "disk"}

    def _parse_run_body(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None, str | None]:
        """Validate a create/start request and merge saved config/overrides."""

        if not isinstance(body, dict):
            raise ConfigError("请求体必须是 JSON 对象")
        allowed = {"preset", "config", "config_name", "overrides", "run_name", "stage"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise ConfigError(f"未知的任务请求字段: {unknown}")

        preset = body.get("preset")
        run_name = body.get("run_name")
        stage = body.get("stage")
        config_name = body.get("config_name")
        config_data = body.get("config")
        overrides = body.get("overrides")

        if preset is not None and not isinstance(preset, str):
            raise ConfigError("preset 必须是字符串")
        if run_name is not None and not isinstance(run_name, str):
            raise ConfigError("run_name 必须是字符串")
        if config_name is not None and not isinstance(config_name, str):
            raise ConfigError("config_name 必须是字符串")
        if isinstance(config_data, str):
            if config_name is None:
                config_name = config_data
            config_data = None
        if config_data is not None and not isinstance(config_data, dict):
            raise ConfigError("config 必须是 JSON 对象或已保存配置名")
        if overrides is not None and not isinstance(overrides, dict):
            raise ConfigError("overrides 必须是 JSON 对象")

        merged: dict[str, Any] = {}
        if config_name:
            merged.update(self.config_store.load(config_name))
        if config_data:
            merged.update(config_data)
        if overrides:
            merged.update(overrides)
        if stage is not None:
            merged["stage"] = stage
        return merged, preset, run_name

    def create_run(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a draft run (frozen config, no process)."""

        merged, preset, run_name = self._parse_run_body(body)
        record = self.job_manager.create_run(config=merged, preset=preset, run_name=run_name)
        return {
            "job": record.to_dict(),
            "message": f"任务已创建（未启动）: {record.run_id}",
        }

    def start_job(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a run and start it immediately (legacy convenience endpoint)."""

        merged, preset, run_name = self._parse_run_body(body)
        record = self.job_manager.start_job(config=merged, preset=preset, run_name=run_name)
        return {
            "job": record.to_dict(),
            "message": f"任务已启动: {record.run_id}",
        }

    def start_run(self, run_id: str) -> dict[str, Any]:
        """Start an existing run (draft / stopped / failed) atomically."""

        record = self.job_manager.start_run(run_id)
        return {
            "job": record.to_dict(),
            "message": f"任务已启动: {record.run_id}",
        }

    def resume_run(self, run_id: str) -> dict[str, Any]:
        """一键断点续训：复用冻结配置，从最新检查点继续。"""

        record = self.job_manager.resume_job(run_id)
        return {
            "job": record.to_dict(),
            "message": f"已从检查点续训: {record.run_id}",
        }

    def stop_job(self, job_id: str) -> dict[str, Any]:
        """请求停止某个任务子进程。"""
        record = self.job_manager.get_job(job_id)
        if record is None:
            # 历史运行（不是本服务启动的）没有可控进程，明确告知。
            self._run_dir(job_id)
            raise JobNotFound(f"运行 {job_id} 不是由当前服务启动的，无法停止")
        was_active = record.is_active
        final = self.job_manager.stop_job(job_id)
        return {
            "job": final.to_dict(),
            "stopped": was_active,
            "message": "已停止任务" if was_active else "任务已结束，无需停止",
        }

    # -- 运行历史 -----------------------------------------------------------
    def list_runs(self) -> dict[str, Any]:
        """扫描 runs/ 并返回可展示的运行列表。"""
        runs: list[dict[str, Any]] = []
        root = self.runs_dir
        if not root.is_dir():
            return {"runs": [], "count": 0, "runs_dir": str(root)}
        entries = [entry for entry in root.iterdir() if entry.is_dir()]
        for entry in entries:
            try:
                validate_run_id(entry.name)
            except ConfigError:
                continue
            try:
                runs.append(self._run_listing(entry))
            except OSError:
                continue
        runs.sort(key=lambda item: item.get("_mtime", 0.0), reverse=True)
        for item in runs:
            item.pop("_mtime", None)
        return {"runs": runs, "count": len(runs), "runs_dir": str(root)}

    def run_summary(self, run_id: str) -> dict[str, Any]:
        """汇总单个 run 的配置、状态、指标与产物。"""
        run_dir = self._run_dir(run_id)
        status = read_json_tolerant(run_dir / "status.json")
        status = status if isinstance(status, dict) else {}
        resolved = read_json_tolerant(run_dir / "resolved_config.json")
        run_config = read_json_tolerant(run_dir / "run_config.json")
        environment = read_json_tolerant(run_dir / "environment.json")
        environment = environment if isinstance(environment, dict) else {}
        report = _interpolate_report(read_json_tolerant(run_dir / "training_report.json"))
        config = _extract_config(resolved, run_config)

        events, event_count = self._events_cache.tail(run_dir / "events.jsonl")
        samples, sample_count = self._samples_cache.tail(run_dir / "samples.jsonl")
        columns, metric_rows = self._metrics_cache.read(run_dir / "metrics.csv")
        record = self.job_manager.get_job(run_id)
        display_status = (
            record.status
            if record is not None
            else str(status.get("status") or "unknown")
        )

        minirun_report = None
        minirun_run_id = None
        mini_dir = self.runs_dir / f"{run_id}-mini"
        if mini_dir.is_dir():
            minirun_run_id = f"{run_id}-mini"
            candidate = read_json_tolerant(mini_dir / "minirun_report.json")
            minirun_report = candidate if isinstance(candidate, dict) else None

        files = {name: (run_dir / name).is_file() for name in RUN_ARTIFACTS}
        updated_at = (
            status.get("updated_at")
            or (run_config or {}).get("created_at")
            or iso_from_mtime(run_dir / "resolved_config.json")
            or iso_from_mtime(run_dir)
        )
        checkpoint = _pick_checkpoint(run_dir)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "project_id": self.project_store.project_of(run_id),
            "parent_run_id": config.get("parent_run_id") or None,
            "best_checkpoint": checkpoint.name if checkpoint is not None else None,
            "display_status": display_status,
            "status": status,
            "job": record.to_dict() if record is not None else None,
            "active": bool(record is not None and record.is_active),
            "config": config,
            "config_source": (
                "resolved_config.json"
                if isinstance(resolved, dict) and resolved
                else ("run_config.json" if isinstance(run_config, dict) else None)
            ),
            "run_config": run_config if isinstance(run_config, dict) else None,
            "environment": environment,
            "report": report,
            "minirun_report": minirun_report,
            "minirun_run_id": minirun_run_id,
            "metrics_columns": columns,
            "metrics": metric_rows[-1] if metric_rows else {},
            "metrics_count": len(metric_rows),
            "events": events[-60:],
            "event_count": event_count,
            "early_stop": [
                event for event in events if event.get("event") == "early_stop"
            ][-5:],
            "samples": samples[-5:],
            "sample_count": sample_count,
            "console_log": read_text_tail(run_dir / "console.log", 40),
            "files": files,
            "has_checkpoint": (run_dir / "last.pt").is_file() or (run_dir / "last.mgckpt").is_file(),
            "updated_at": updated_at,
        }

    def run_events(self, run_id: str, after: int, limit: int) -> dict[str, Any]:
        """按 after/limit 返回增量事件（供轮询）。"""
        run_dir = self._run_dir(run_id)
        after = max(0, int(after))
        limit = max(1, min(EVENT_TAIL_MAX, int(limit)))
        events, total = self._events_cache.window(run_dir / "events.jsonl", after, limit)
        next_after = min(after + limit, total) if total > 0 else after
        return {
            "events": events,
            "after": after,
            "next_after": next_after,
            "total": total,
            "run_id": run_id,
        }

    def run_metrics(self, run_id: str) -> dict[str, Any]:
        """返回训练曲线数据点。"""
        run_dir = self._run_dir(run_id)
        columns, rows = self._metrics_cache.read(run_dir / "metrics.csv")
        return {
            "run_id": run_id,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
        }

    def run_samples(self, run_id: str, limit: int) -> dict[str, Any]:
        """返回训练期间保存的样本片段。"""
        run_dir = self._run_dir(run_id)
        samples, _total = self._samples_cache.tail(run_dir / "samples.jsonl")
        if limit > 0:
            samples = samples[-limit:]
        return {"run_id": run_id, "samples": samples, "count": len(samples)}

    def run_console(self, run_id: str, limit: int) -> dict[str, Any]:
        """返回训练控制台日志尾部。"""
        run_dir = self._run_dir(run_id)
        return {
            "run_id": run_id,
            "console_log": read_text_tail(run_dir / "console.log", limit),
        }

    # -- N07 多 run 对比 ------------------------------------------------------
    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """聚合多个 run 的 metrics.csv 曲线与 training_report.json 摘要。

        * run id 逐个白名单校验（非法 id → 400，不存在 → 404）；
        * 去重但保持请求顺序；最多 ``MAX_COMPARE_RUNS`` 个，防止一次拉爆内存；
        * 曲线只取 step/train_loss/val_loss/val_ppl 四列，坏值/空值保留为 null。
        """

        if not isinstance(run_ids, list) or not run_ids:
            raise ApiError(400, "缺少 runs 参数（逗号分隔的运行 id）")
        cleaned: list[str] = []
        for raw in run_ids:
            run_id = validate_run_id(str(raw).strip())
            if run_id not in cleaned:
                cleaned.append(run_id)
        if not cleaned:
            raise ApiError(400, "缺少 runs 参数（逗号分隔的运行 id）")
        if len(cleaned) > MAX_COMPARE_RUNS:
            raise ApiError(400, f"一次最多对比 {MAX_COMPARE_RUNS} 个运行")
        runs: list[dict[str, Any]] = []
        for run_id in cleaned:
            run_dir = self._run_dir(run_id)
            columns, rows = self._metrics_cache.read(run_dir / "metrics.csv")
            series: dict[str, list[Any]] = {"step": []}
            for name in COMPARE_SERIES_COLUMNS:
                series[name] = []
            for row in rows:
                step = row.get("step")
                if not isinstance(step, (int, float)) or isinstance(step, bool):
                    continue
                series["step"].append(step)
                for name in COMPARE_SERIES_COLUMNS:
                    value = row.get(name)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        value = None
                    series[name].append(value)
            report = _interpolate_report(read_json_tolerant(run_dir / "training_report.json"))
            summary = (
                {key: report.get(key) for key in COMPARE_REPORT_KEYS if key in report}
                if report is not None
                else None
            )
            status = read_json_tolerant(run_dir / "status.json")
            status = status if isinstance(status, dict) else {}
            resolved = read_json_tolerant(run_dir / "resolved_config.json")
            run_config = read_json_tolerant(run_dir / "run_config.json")
            config = _extract_config(resolved, run_config)
            record = self.job_manager.get_job(run_id)
            runs.append(
                {
                    "run_id": run_id,
                    "stage": status.get("stage") or config.get("stage"),
                    "display_status": (
                        record.status
                        if record is not None
                        else str(status.get("status") or "unknown")
                    ),
                    "metrics_columns": columns,
                    "metrics_count": len(rows),
                    "series": series,
                    "report": summary,
                }
            )
        return {"runs": runs, "count": len(runs)}

    @staticmethod
    def _status_is_stale(status: dict[str, Any]) -> bool:
        """True when a ``running`` status has not been refreshed for a minute."""

        updated = status.get("updated_at")
        if not updated:
            return True
        try:
            parsed = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - parsed).total_seconds() > RUN_STALE_SECONDS

    def delete_run(self, run_id: str, force: bool = False) -> dict[str, Any]:
        """Delete a run directory. Destructive: checkpoint/metrics/events go away.

        Active training runs (this server's jobs or a fresh on-disk ``running``
        status) are refused with 409 so a training process never loses files
        under its feet.  ``force=True`` allows deleting a **stale** run whose
        process is gone but ``status.json`` still says ``running``.
        """

        run_dir = self._run_dir(run_id)
        # Hold the manager lock across check + delete so a concurrent start_run
        # cannot launch a process into the directory we are about to remove.
        with self.job_manager._lock:
            job = self.job_manager.get_job(run_dir.name)
            if job is not None and job.is_active:
                raise ApiError(409, '运行 {} 还有活动任务；先停止再删除'.format(run_dir.name))
            status = read_json_tolerant(run_dir / 'status.json') or {}
            status_name = str(status.get('status', '') or '')
            if status_name in ACTIVE_STATUSES:
                stale = self._status_is_stale(status)
                if not force:
                    hint = '；该任务心跳已过期，可用强制删除' if stale else ''
                    raise ApiError(409, '运行 {} 状态为 {}；先停止再删除{}'.format(run_dir.name, status_name, hint))
                if not stale:
                    raise ApiError(409, '运行 {} 仍在更新心跳，拒绝强制删除'.format(run_dir.name))
            try:
                shutil.rmtree(run_dir)
            except OSError as exc:
                raise ApiError(500, '删除运行失败: {}'.format(exc)) from exc
            self._events_cache.invalidate(run_dir / 'events.jsonl')
            self._samples_cache.invalidate(run_dir / 'samples.jsonl')
            self._metrics_cache.invalidate(run_dir / 'metrics.csv')
            try:
                self.job_manager.forget_job(run_dir.name)
            except JobManagerError:
                pass
        return {'deleted': run_dir.name, 'message': '已删除运行 {}'.format(run_dir.name)}

    # -- 只读阶段视图：数据 / 评测 / 导出 ------------------------------------
    def datasets(self) -> dict[str, Any]:
        """列出 ``data/`` 下的语料与 SFT 数据集（只读扫描）。"""

        payload = stages.scan_datasets(self.data_dir)
        payload["project_root"] = str(self.project_root)
        return payload

    def dataset(self, name: str, limit: int = stages.MAX_SAMPLES) -> dict[str, Any]:
        """单个数据集详情：完整 meta + SFT 样例 / 预训练语料预览。"""

        detail = stages.dataset_detail(self.data_dir, name, limit=limit)
        if detail is None:
            raise ApiError(404, f"数据集不存在: {name}")
        return detail

    def evaluations(self) -> dict[str, Any]:
        """扫描 ``runs/<id>/eval_*`` 下的评测报告（只读）。"""

        payload = stages.scan_evaluations(self.runs_dir)
        payload["project_root"] = str(self.project_root)
        return payload

    def evaluation(
        self,
        run_id: str,
        eval_name: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """单个评测详情：报告 + predictions 头部 + 能力边界草稿。"""

        detail = stages.evaluation_detail(
            self.runs_dir, run_id, eval_name=eval_name or None, limit=limit
        )
        if detail is None:
            raise ApiError(404, f"评测不存在: {run_id}")
        return detail

    def exports(self) -> dict[str, Any]:
        """列出 ``dist/`` 下的导出包（以 MANIFEST.json 为准，只读）。"""

        payload = stages.scan_exports(self.dist_dir)
        payload["project_root"] = str(self.project_root)
        return payload

    def export(self, name: str) -> dict[str, Any]:
        """返回某个 dist 模型包的详情。"""
        detail = stages.export_detail(self.dist_dir, name)
        if detail is None:
            raise ApiError(404, f"导出包不存在: {name}")
        source_run = detail.get("run_id")
        detail["run_missing"] = bool(source_run) and not (self.runs_dir / str(source_run)).is_dir()
        return detail

    # -- N08 面板内评估 ------------------------------------------------------
    def evaluation_datasets(self) -> list[str]:
        """data/ 下可用于评估的数据集白名单（有元信息且含 val/test 切分）。"""

        payload = stages.scan_datasets(self.data_dir)
        names: list[str] = []
        for entry in payload.get("datasets", []):
            if entry.get("kind") == "sft":
                ok = bool(entry.get("has_jsonl"))
            else:
                ok = bool(entry.get("has_binary"))
            if ok:
                names.append(str(entry.get("name")))
        return names

    def _evaluate_command(
        self,
        run_dir: Path,
        checkpoint: Path,
        dataset: str,
        split: str,
        output_dir: Path,
        limits: dict[str, int] | None = None,
    ) -> list[str]:
        """固定评估命令：仓库内 evaluate.py；测试可注入假命令。"""

        script = self.project_root / "evaluate.py"
        if not script.is_file():
            raise ApiError(500, "评估脚本 evaluate.py 不存在")
        command = [
            self.job_manager.python_executable,
            "-u",
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--data_dir",
            str(self.data_dir / dataset),
            "--split",
            split,
            "--output_dir",
            str(output_dir),
            "--device",
            "auto",
        ]
        for flag, value in (limits or {}).items():
            command += [flag, str(value)]
        return command

    def _append_eval_event(self, run_dir: Path, run_id: str, event: str, payload: dict[str, Any]) -> None:
        """向该 run 的 events.jsonl 追加评估事件（与训练事件同 schema）。"""

        record = {
            "v": 1,
            "ts": iso_now(),
            "run_id": run_id,
            "event": event,
            "step": 0,
            "payload": payload,
        }
        try:
            with open(run_dir / "events.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        except OSError:
            pass

    def start_evaluation(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /api/v1/evaluate：起 evaluate.py 子进程，进度写该 run 的 eval 事件。"""

        if not isinstance(body, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        run_id = body.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ApiError(400, "run_id 不能为空")
        run_dir = self._run_dir(run_id.strip())
        run_id = run_dir.name
        dataset = body.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ApiError(400, "dataset 不能为空")
        dataset = dataset.strip()
        whitelist = self.evaluation_datasets()
        if dataset not in whitelist:
            raise ApiError(400, f"数据集不在评估白名单内: {dataset}；可选: {', '.join(whitelist) or '（空）'}")
        split = body.get("split", "test")
        if split not in {"val", "test"}:
            raise ApiError(400, "split 必须是 val 或 test")
        checkpoint = _pick_checkpoint(run_dir)
        if checkpoint is None:
            raise ApiError(400, f"运行 {run_id} 没有 best/last 检查点（.pt / .mgckpt），无法评估")
        limits: dict[str, int] = {}
        for key, flag, minimum, maximum in (
            ("max_samples", "--max_samples", 1, 200),
            ("max_new_tokens", "--max_new_tokens", 1, 120),
            ("num_batches", "--num_batches", 1, 50),
        ):
            if key in body:
                value = body[key]
                if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                    raise ApiError(400, f"{key} 必须是 [{minimum}, {maximum}] 内的整数")
                limits[flag] = value
        eval_name = f"eval_{split}"
        output_dir = run_dir / eval_name
        if limits:
            command = self._evaluate_command(run_dir, checkpoint, dataset, split, output_dir, limits)
        else:
            command = self._evaluate_command(run_dir, checkpoint, dataset, split, output_dir)
        try:
            record = self.job_manager.start_eval_job(
                run_id,
                command,
                dataset=dataset,
                split=split,
                eval_dir=eval_name,
                run_dir=run_dir,
                console_log=run_dir / "eval_console.log",
                status_path=run_dir / "eval_status.json",
            )
        except JobLimitExceeded as exc:
            raise ApiError(409, str(exc)) from exc
        self._append_eval_event(
            run_dir,
            run_id,
            "eval_start",
            {
                "dataset": dataset,
                "split": split,
                "eval_dir": eval_name,
                "checkpoint": checkpoint.name,
                "pid": record.pid,
            },
        )
        self._events_cache.invalidate(run_dir / "events.jsonl")
        return record.to_dict()

    def _on_eval_finished(self, record: Any) -> None:
        """JobManager 评估收尾回调：写 eval_end 事件并失效事件缓存。"""

        run_dir = Path(record.run_dir)
        self._append_eval_event(
            run_dir,
            record.run_id,
            "eval_end",
            {
                "dataset": record.dataset,
                "split": record.split,
                "eval_dir": record.eval_dir,
                "status": record.status,
                "returncode": record.returncode,
                "elapsed_s": record.elapsed_s,
                "error": record.error,
            },
        )
        self._events_cache.invalidate(run_dir / "events.jsonl")

    def _eval_result(self, run_dir: Path, eval_name: str | None) -> dict[str, Any] | None:
        """读取并返回某个评估目录的结果摘要。"""
        if not eval_name:
            return None
        report = read_json_tolerant(run_dir / eval_name / "evaluation.json")
        return report if isinstance(report, dict) else None

    def evaluate_status(self, run_id: str) -> dict[str, Any]:
        """GET /api/v1/evaluate/status：内存记录优先，重启后回退状态文件。"""

        run_dir = self._run_dir(str(run_id))
        run_id = run_dir.name
        record = self.job_manager.get_eval_job(run_id)
        if record is not None:
            if record.is_active:
                # 立即收割已退出进程，避免状态查询看到滞后的 running。
                self.job_manager.poll()
                record = self.job_manager.get_eval_job(run_id) or record
            payload = record.to_dict()
            payload["result"] = self._eval_result(run_dir, record.eval_dir)
            return payload
        status = read_json_tolerant(run_dir / "eval_status.json")
        if isinstance(status, dict) and status.get("kind") == "evaluation":
            name = str(status.get("status") or "unknown")
            if name in ACTIVE_STATUSES:
                from minigpt.training.job_manager import _pid_alive

                if not _pid_alive(status.get("pid")):
                    name = "failed"
                    status = dict(status, status=name, error="评估进程已不在运行（服务可能重启过）")
            return {
                "kind": "evaluation",
                "run_id": run_id,
                "dataset": status.get("dataset"),
                "split": status.get("split"),
                "eval_dir": status.get("eval_dir"),
                "status": name,
                "active": name in ACTIVE_STATUSES,
                "pid": status.get("pid"),
                "started_at": status.get("started_at"),
                "ended_at": status.get("ended_at"),
                "returncode": status.get("returncode"),
                "error": status.get("error"),
                "source": "disk",
                "result": self._eval_result(run_dir, status.get("eval_dir")),
            }
        # 从未通过面板评估：若有历史评测报告也一并给出。
        latest = stages.evaluation_detail(self.runs_dir, run_id, limit=1)
        return {
            "kind": "evaluation",
            "run_id": run_id,
            "status": "idle",
            "active": False,
            "dataset": None,
            "split": None,
            "eval_dir": latest.get("eval_dir") if latest else None,
            "result": (latest or {}).get("report"),
            "source": "none",
        }

    # -- 生成可视化 ---------------------------------------------------------
    def build_inference_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """校验浏览器请求并冻结成推理子进程的固定请求体。

        浏览器只能传运行 id（服务端解析为 ``runs/<id>/best.pt``）、生成参数
        和可选的 ``prefix_token_ids``（改选分支用），不能传任意路径。
        """

        def _int(name: str, default: int, minimum: int, maximum: int) -> int:
            try:
                value = int(payload.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ApiError(400, f"{name} 必须是整数") from exc
            if not minimum <= value <= maximum:
                raise ApiError(400, f"{name} 必须在 [{minimum}, {maximum}] 内")
            return value

        def _float(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                value = float(payload.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ApiError(400, f"{name} 必须是数字") from exc
            if not minimum <= value <= maximum:
                raise ApiError(400, f"{name} 必须在 [{minimum}, {maximum}] 内")
            return value

        if not isinstance(payload, dict):
            raise ApiError(400, "请求体必须是 JSON object")
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ApiError(400, "run_id 不能为空")
        run_dir = self._run_dir(run_id.strip())
        checkpoint = _pick_checkpoint(run_dir)
        if checkpoint is None:
            raise ApiError(400, f"运行 {run_dir.name} 没有 best/last 检查点（.pt / .mgckpt），无法生成")
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str):
            raise ApiError(400, "prompt 必须是字符串")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ApiError(400, f"prompt 不能超过 {MAX_PROMPT_CHARS} 字符")
        template = str(payload.get("template", "auto"))
        if template not in {"auto", "qa", "raw"}:
            raise ApiError(400, "template 必须是 auto/qa/raw")
        device = str(payload.get("device", "auto"))
        if device not in {"auto", "cpu", "cuda"}:
            raise ApiError(400, "device 必须是 auto/cpu/cuda")
        prefix = payload.get("prefix_token_ids") or []
        if not isinstance(prefix, list):
            raise ApiError(400, "prefix_token_ids 必须是整数数组")
        if len(prefix) > MAX_PREFIX_TOKENS:
            raise ApiError(400, f"prefix_token_ids 不能超过 {MAX_PREFIX_TOKENS} 个")
        prefix_ids: list[int] = []
        for value in prefix:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ApiError(400, "prefix_token_ids 只能包含整数")
            if value < 0 or value > 1_000_000:
                raise ApiError(400, "prefix_token_ids 超出范围")
            prefix_ids.append(value)
        return {
            "checkpoint": str(checkpoint),
            "prompt": prompt,
            "max_new_tokens": _int("max_new_tokens", 24, 1, 256),
            "temperature": _float("temperature", 0.8, 0.0, 5.0),
            "top_k": _int("top_k", 40, 0, 1000),
            "top_p": _float("top_p", 1.0, 0.0, 1.0),
            "repetition_penalty": _float("repetition_penalty", 1.0, 0.1, 5.0),
            "no_repeat_ngram_size": _int("no_repeat_ngram_size", 0, 0, 8),
            "trace_top_k": _int("trace_top_k", 5, 0, 20),
            "trace_candidates": _int("trace_candidates", 20, 3, 50),
            "max_context": _int("max_context", 32, 1, 128),
            "seed": _int("seed", 1337, 0, 2_147_483_647),
            "template": template,
            "device": device,
            "prefix_token_ids": prefix_ids,
        }

    def inference(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one model generation in a PyTorch-capable subprocess.

        The dashboard process itself stays standard-library-only.  The fixed
        worker command reads one JSON request and writes one JSON trace.
        """

        request = self.build_inference_request(payload)
        if not self._inference_slots.acquire(blocking=False):
            raise ApiError(429, "推理并发已达上限，请稍后重试")
        try:
            result = self._run_inference_worker(request)
        finally:
            self._inference_slots.release()
        result["run_id"] = Path(request["checkpoint"]).parent.name
        result["checkpoint"] = Path(request["checkpoint"]).name
        return result

    # -- 项目级流水线管理 -----------------------------------------------------
    def list_projects(self) -> dict[str, Any]:
        """列出所有项目及其关联 run。"""
        items = self.project_store.list()
        return {"projects": items, "count": len(items)}

    def create_project(self, body: dict[str, Any]) -> dict[str, Any]:
        """新建项目。"""
        if not isinstance(body, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ApiError(400, "name 不能为空")
        note = body.get("note") or ""
        if not isinstance(note, str):
            raise ApiError(400, "note 必须是字符串")
        project_id = body.get("project_id")
        if project_id is not None and not isinstance(project_id, str):
            raise ApiError(400, "project_id 必须是字符串")
        try:
            project = self.project_store.create(name, note=note, project_id=project_id)
        except ConfigError as exc:
            raise ApiError(400, str(exc)) from exc
        return {"project": project, "message": f"项目已创建: {project['project_id']}"}

    def project_detail(self, project_id: str) -> dict[str, Any]:
        """返回单个项目详情与关联 run 摘要。"""
        try:
            project = self.project_store.load(project_id)
        except JobNotFound as exc:
            raise ApiError(404, str(exc)) from exc
        runs: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        fingerprints: dict[str, Any] = {}
        for run_id in project.get("run_ids") or []:
            run_dir = self.runs_dir / run_id
            if not run_dir.is_dir():
                runs.append({"run_id": run_id, "missing": True})
                continue
            runs.append(self._run_listing(run_dir))
            for entry in stages.scan_evaluations(self.runs_dir).get("evaluations", []):
                if entry.get("run_id") == run_id:
                    evaluations.append(entry)
            report = read_json_tolerant(run_dir / "training_report.json")
            if isinstance(report, dict) and report.get("data_fingerprint"):
                fingerprints[run_id] = report["data_fingerprint"]
        member_ids = {run.get("run_id") for run in runs}
        exports = [
            entry
            for entry in stages.scan_exports(self.dist_dir).get("exports", [])
            if entry.get("run_id") in member_ids
        ]
        return {
            "project": project,
            "runs": runs,
            "evaluations": evaluations,
            "exports": exports,
            "data_fingerprints": fingerprints,
        }

    def attach_project_run(self, project_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """把某个 run 关联到项目。"""
        if not isinstance(body, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        run_id = body.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ApiError(400, "run_id 不能为空")
        run_id = run_id.strip()
        if not (self.runs_dir / run_id).is_dir():
            raise ApiError(404, f"运行不存在: {run_id}")
        try:
            project = self.project_store.attach_run(project_id, run_id)
        except JobNotFound as exc:
            raise ApiError(404, str(exc)) from exc
        except ConfigError as exc:
            raise ApiError(400, str(exc)) from exc
        return {"project": project, "message": f"运行 {run_id} 已挂入项目 {project['project_id']}"}

    # -- N09 生成参数对比 -----------------------------------------------------
    def inference_compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """同一 prompt + 多组生成参数（≤4）逐组推理，返回文本与耗时。

        * 组内参数覆盖共享参数；每组独立调用推理 worker（串行，受限并发信号量）；
        * 单组失败只标记该组 ``ok=False``，不拖垮其它组；
        * 请求级错误（run 不存在 / 超限 / 参数非法）仍然整体 4xx。
        """

        if not isinstance(payload, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        variants = payload.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ApiError(400, "variants 必须是非空数组")
        if len(variants) > MAX_INFERENCE_VARIANTS:
            raise ApiError(400, f"一次最多对比 {MAX_INFERENCE_VARIANTS} 组参数")
        allowed_variant_keys = {
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "max_new_tokens",
            "seed",
        }
        base = {key: value for key, value in payload.items() if key != "variants"}
        requests: list[dict[str, Any]] = []
        params_view: list[dict[str, Any]] = []
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise ApiError(400, f"variants[{index}] 必须是 JSON 对象")
            unknown = sorted(set(variant) - allowed_variant_keys)
            if unknown:
                raise ApiError(400, f"variants[{index}] 含未知字段: {unknown}")
            merged = dict(base)
            merged.update(variant)
            try:
                request = self.build_inference_request(merged)
            except ApiError as exc:
                raise ApiError(exc.status, f"variants[{index}]: {exc.message}") from exc
            requests.append(request)
            params_view.append(
                {
                    key: request[key]
                    for key in (
                        "temperature",
                        "top_k",
                        "top_p",
                        "repetition_penalty",
                        "no_repeat_ngram_size",
                        "max_new_tokens",
                        "seed",
                    )
                }
            )
        results: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            entry: dict[str, Any] = {
                "index": index,
                "params": params_view[index],
                "ok": False,
                "text": None,
                "elapsed_ms": None,
                "error": None,
                "result": None,
            }
            started = time.perf_counter()
            try:
                if not self._inference_slots.acquire(blocking=False):
                    raise ApiError(429, "推理并发已达上限，请稍后重试")
                try:
                    result = self._run_inference_worker(request)
                finally:
                    self._inference_slots.release()
            except ApiError as exc:
                entry["error"] = exc.message
                entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                results.append(entry)
                continue
            except Exception as exc:  # noqa: BLE001 - 单组异常不能拖垮整体
                self.log_diagnostic(f"inference compare variant {index} failed", exc=exc)
                entry["error"] = self._public(f"{type(exc).__name__}: {exc}")
                entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                results.append(entry)
                continue
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            final = result.get("final") if isinstance(result.get("final"), dict) else {}
            entry.update(
                ok=True,
                text=str(final.get("continuation") or ""),
                elapsed_ms=elapsed_ms,
                result=result,
            )
            results.append(entry)
        return {
            "run_id": Path(requests[0]["checkpoint"]).parent.name,
            "prompt": requests[0]["prompt"],
            "count": len(results),
            "results": results,
        }

    # 推理子进程管理（_inference_command / start_inference_process /
    # close_inference_process / _run_inference_worker 等）已拆至
    # minigpt.web.inference_proc.InferenceProcessMixin，经类继承并入。

    # -- 内部 ---------------------------------------------------------------
    def _run_dir(self, run_id: str) -> Path:
        """把 run_id 解析为 runs/ 下的目录并校验存在。"""
        run_id = validate_run_id(str(run_id))
        root = self.runs_dir.resolve()
        candidate = (root / run_id).resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise JobNotFound(f"运行不存在: {run_id}")
        if not candidate.is_dir():
            raise JobNotFound(f"运行不存在: {run_id}")
        return candidate

    def _run_mtime(self, run_dir: Path, status: dict[str, Any]) -> float:
        """用于排序的最近活动时间（epoch 秒）。"""

        latest = 0.0
        for name in RUN_ARTIFACTS:
            try:
                latest = max(latest, (run_dir / name).stat().st_mtime)
            except OSError:
                continue
        try:
            latest = max(latest, run_dir.stat().st_mtime)
        except OSError:
            pass
        return latest

    def _disk_job_view(self, run_id: str, run_dir: Path) -> dict[str, Any]:
        """从磁盘状态构造任务视图（服务重启后仍可展示）。"""
        status = read_json_tolerant(run_dir / "status.json")
        status = status if isinstance(status, dict) else {}
        return {
            "job_id": run_id,
            "run_id": run_id,
            "status": str(status.get("status") or "unknown"),
            "active": False,
            "pid": status.get("pid"),
            "stage": status.get("stage"),
            "preset": None,
            "started_at": status.get("started_at"),
            "ended_at": None,
            "elapsed_s": status.get("elapsed_s"),
            "returncode": None,
            "error": status.get("error") or None,
            "run_dir": str(run_dir),
            "config_path": str(run_dir / "resolved_config.json"),
            "console_log": str(run_dir / "console.log"),
            "command": [],
            "max_steps": status.get("max_steps"),
            "stop_requested": False,
            "source": "disk",
        }

    def _run_listing(self, run_dir: Path) -> dict[str, Any]:
        """枚举 run 目录并附带 mtime/状态。"""
        status = read_json_tolerant(run_dir / "status.json")
        status = status if isinstance(status, dict) else {}
        resolved = read_json_tolerant(run_dir / "resolved_config.json")
        run_config = read_json_tolerant(run_dir / "run_config.json")
        config = _extract_config(resolved, run_config)
        record = self.job_manager.get_job(run_dir.name)
        display_status = record.status if record is not None else str(status.get("status") or "unknown")
        updated_at = (
            status.get("updated_at")
            or (run_config or {}).get("created_at")
            or iso_from_mtime(run_dir / "resolved_config.json")
            or iso_from_mtime(run_dir)
        )
        mtime = self._run_mtime(run_dir, status)
        checkpoint = _pick_checkpoint(run_dir)
        return {
            "run_id": run_dir.name,
            "project_id": self.project_store.project_of(run_dir.name),
            "parent_run_id": config.get("parent_run_id") or None,
            "best_checkpoint": checkpoint.name if checkpoint is not None else None,
            "display_status": display_status,
            "status": status.get("status") or ("running" if record and record.is_active else "unknown"),
            "active": bool(record is not None and record.is_active),
            "stage": status.get("stage") or config.get("stage"),
            "step": status.get("step"),
            "max_steps": status.get("max_steps") or config.get("max_steps"),
            "best_val": status.get("best_val"),
            "pid": status.get("pid") or (record.pid if record else None),
            "started_at": status.get("started_at") or (record.started_at if record else None),
            "updated_at": updated_at,
            "error": status.get("error") or (record.error if record else None),
            "has_events": (run_dir / "events.jsonl").is_file(),
            "has_report": (run_dir / "training_report.json").is_file(),
            "has_best": checkpoint is not None,
            "run_dir": str(run_dir),
            "_mtime": mtime,
        }


# ---------------------------------------------------------------------------
# HTTP 处理
# ---------------------------------------------------------------------------
class DashboardHandler(StaticFilesMixin, BaseHTTPRequestHandler):
    """单文件路由：静态资源 + 固定 API 路径。

    静态资源服务经 StaticFilesMixin 并入（见 static_files.py）。
    """

    server_version = f"MiniGPTDashboard/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- 日志（默认安静，--verbose 时才打印） -------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 框架接口
        """访问日志：统一加前缀输出到标准输出。"""
        app = getattr(self.server, "app", None)
        if app is not None and app.verbose:
            sys.stderr.write(
                f"[dashboard] {self.address_string()} {fmt % args}\n"
            )

    def setup(self) -> None:
        """建立连接时设置 socket 超时。"""
        super().setup()
        self.connection.settimeout(SOCKET_TIMEOUT_S)

    # -- HTTP 方法 ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - 框架接口
        """处理 GET 请求。"""
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        """处理 POST 请求。"""
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        """处理 PUT 请求。"""
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        """处理 DELETE 请求。"""
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        """处理 PATCH 请求。"""
        self._dispatch("PATCH")

    def do_HEAD(self) -> None:  # noqa: N802
        """处理 HEAD 请求。"""
        self._dispatch("HEAD")

    # -- 分发 ---------------------------------------------------------------
    # -- 安全：主机 / 来源校验（防 DNS rebinding 与 CSRF） --------------------
    _ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

    @staticmethod
    def _host_name(value: str) -> str:
        """从 Host 头中提取主机名（去掉端口）。"""
        value = value.strip().lower()
        if value.startswith("["):  # IPv6 字面量 [::1]:port
            return value.split("]", 1)[0] + "]"
        return value.rsplit(":", 1)[0] if ":" in value else value

    def _check_request_origin(self) -> None:
        """校验 Origin/Referer，阻止跨站请求伪造。"""
        host = self._host_name(self.headers.get("Host", ""))
        if host not in self._ALLOWED_HOSTS:
            raise ApiError(403, "Host 不被允许（看板仅服务本机回环地址）")
        origin = self.headers.get("Origin")
        if origin:
            parsed = urllib.parse.urlparse(origin)
            if parsed.scheme not in {"http", "https"} or self._host_name(parsed.netloc) not in self._ALLOWED_HOSTS:
                raise ApiError(403, "跨源请求被拒绝")
        referer = self.headers.get("Referer")
        if referer:
            parsed = urllib.parse.urlparse(referer)
            if self._host_name(parsed.netloc) not in self._ALLOWED_HOSTS:
                raise ApiError(403, "跨源请求被拒绝")

    def _dispatch(self, method: str) -> None:
        """统一分发静态资源与 API 请求，并记录异常。"""
        try:
            self._check_request_origin()
            parsed = urllib.parse.urlparse(self.path)
            raw_path = parsed.path or "/"
            path = urllib.parse.unquote(raw_path)
            if "\x00" in path:
                raise ApiError(400, "非法路径")
            path = path.rstrip("/") or "/"
            query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            # HEAD 语义等价于 GET，只是不回写 body（_send_json/_send_bytes 已处理）。
            route_method = "GET" if method == "HEAD" else method
            if path == "/" or path == "/index.html":
                if route_method != "GET":
                    raise ApiError(405, "只支持 GET", {"Allow": "GET"})
                self._serve_static("index.html")
                return
            if path == "/favicon.ico":
                if route_method != "GET":
                    raise ApiError(405, "只支持 GET", {"Allow": "GET"})
                self._send_bytes(204, b"", "image/x-icon")
                return
            if path.startswith("/static/"):
                if route_method != "GET":
                    raise ApiError(405, "只支持 GET", {"Allow": "GET"})
                self._serve_static(path[len("/static/") :])
                return
            if path == "/api/v1" or path.startswith("/api/v1/"):
                self._dispatch_api(route_method, path, query)
                return
            raise ApiError(404, f"未知路径: {path}")
        except ApiError as exc:
            self._send_json(
                exc.status, {"error": self._public_error(exc.message)}, headers=exc.headers
            )
        except ConfigError as exc:
            self._send_json(400, {"error": self._public_error(str(exc)), "type": "config_error"})
        except JobNotFound as exc:
            self._send_json(404, {"error": self._public_error(str(exc)), "type": "not_found"})
        except JobLimitExceeded as exc:
            self._send_json(409, {"error": self._public_error(str(exc)), "type": "job_limit"})
        except JobStartError as exc:
            app = getattr(self.server, "app", None)
            if app is not None:
                app.log_diagnostic("job start failed", exc=exc)
            self._send_json(500, {"error": self._public_error(str(exc)), "type": "job_start_error"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # 客户端断开（关闭标签页/刷新），无需回写
        except Exception as exc:  # noqa: BLE001 - 最后一个兜底，避免连接悬挂
            app = getattr(self.server, "app", None)
            if app is not None:
                app.log_diagnostic(f"internal error on {self.command} {self.path}", exc=exc)
                if app.verbose:
                    sys.stderr.write(f"[dashboard] internal error: {type(exc).__name__}: {exc}\n")
            self._send_json(
                500,
                {"error": "服务器内部错误", "type": "internal_error"},
            )

    def _public_error(self, message: str) -> str:
        """对外错误消息统一脱敏（无 app 上下文时按通用根路径处理）。"""

        app = getattr(self.server, "app", None)
        if app is not None:
            return app._public(message)
        return public_message(message)

    def _dispatch_api(self, method: str, path: str, query: dict[str, list[str]]) -> None:
        """按 method + path 路由到具体的 DashboardApp 方法。"""
        app: DashboardApp = self.server.app  # type: ignore[attr-defined]

        if path == "/api/v1/health":
            self._require(method, {"GET"})
            self._send_json(200, app.health())
            return
        if path == "/api/v1/presets":
            self._require(method, {"GET"})
            self._send_json(200, app.presets())
            return
        if path == "/api/v1/system":
            self._require(method, {"GET"})
            self._send_json(200, app.system())
            return
        if path == "/api/v1/configs":
            if method == "GET":
                self._send_json(200, app.list_configs())
                return
            if method == "POST":
                self._send_json(201, app.create_config(self._read_json_body()))
                return
            self._require(method, {"GET", "POST"})
        if path == "/api/v1/configs/validate":
            self._require(method, {"POST"})
            self._send_json(200, app.validate_config(self._read_json_body()))
            return
        if path == "/api/v1/jobs":
            if method == "GET":
                self._send_json(200, app.list_jobs())
                return
            if method == "POST":
                self._send_json(201, app.start_job(self._read_json_body()))
                return
            self._require(method, {"GET", "POST"})
        if path == "/api/v1/inference":
            self._require(method, {"POST"})
            self._send_json(200, app.inference(self._read_json_body()))
            return
        if path == "/api/v1/inference/stream":
            self._require(method, {"POST"})
            self._serve_inference_stream(app, self._read_json_body())
            return
        if path == "/api/v1/inference/compare":
            self._require(method, {"POST"})
            self._send_json(200, app.inference_compare(self._read_json_body()))
            return
        if path == "/api/v1/compare":
            self._require(method, {"GET"})
            raw_runs = (query.get("runs") or [""])[-1]
            run_ids = [part.strip() for part in raw_runs.split(",") if part.strip()]
            self._send_json(200, app.compare_runs(run_ids))
            return
        if path == "/api/v1/evaluate":
            self._require(method, {"POST"})
            self._send_json(202, app.start_evaluation(self._read_json_body()))
            return
        if path == "/api/v1/evaluate/status":
            self._require(method, {"GET"})
            run_id = (query.get("run_id") or [""])[-1].strip()
            if not run_id:
                raise ApiError(400, "缺少查询参数 run_id")
            self._send_json(200, app.evaluate_status(run_id))
            return
        if path == "/api/v1/datasets":
            self._require(method, {"GET"})
            self._send_json(200, app.datasets())
            return
        if path.startswith("/api/v1/datasets/"):
            name = path[len("/api/v1/datasets/") :]
            if not name or "/" in name:
                raise ApiError(404, "未知数据集路径")
            self._require(method, {"GET"})
            limit = self._query_int(query, "limit", 10, 1, stages.MAX_SAMPLES)
            self._send_json(200, app.dataset(name, limit=limit))
            return
        if path == "/api/v1/evaluations":
            self._require(method, {"GET"})
            self._send_json(200, app.evaluations())
            return
        if path.startswith("/api/v1/evaluations/"):
            rest = path[len("/api/v1/evaluations/") :]
            parts = rest.split("/")
            if not parts[0] or len(parts) > 2:
                raise ApiError(404, "未知评测路径")
            run_id = parts[0]
            eval_name = parts[1] if len(parts) > 1 else (query.get("eval") or [None])[-1]
            self._require(method, {"GET"})
            limit = self._query_int(query, "limit", 50, 1, 500)
            self._send_json(200, app.evaluation(run_id, eval_name=eval_name, limit=limit))
            return
        if path == "/api/v1/exports":
            self._require(method, {"GET"})
            self._send_json(200, app.exports())
            return
        if path.startswith("/api/v1/exports/"):
            name = path[len("/api/v1/exports/") :]
            if not name or "/" in name:
                raise ApiError(404, "未知导出包路径")
            self._require(method, {"GET"})
            self._send_json(200, app.export(name))
            return
        if path == "/api/v1/projects":
            if method == "GET":
                self._send_json(200, app.list_projects())
                return
            if method == "POST":
                self._send_json(201, app.create_project(self._read_json_body()))
                return
            self._require(method, {"GET", "POST"})
        if path.startswith("/api/v1/projects/"):
            rest = path[len("/api/v1/projects/") :]
            parts = rest.split("/")
            if not parts[0] or len(parts) > 2:
                raise ApiError(404, "未知项目路径")
            if len(parts) == 2 and parts[1] == "runs":
                self._require(method, {"POST"})
                self._send_json(200, app.attach_project_run(parts[0], self._read_json_body()))
                return
            if len(parts) == 1:
                self._require(method, {"GET"})
                self._send_json(200, app.project_detail(parts[0]))
                return
            raise ApiError(404, "未知项目路径")
        if path == "/api/v1/runs":
            if method == "GET":
                self._send_json(200, app.list_runs())
                return
            if method == "POST":
                self._send_json(201, app.create_run(self._read_json_body()))
                return
            self._require(method, {"GET", "POST"})
        if path.startswith("/api/v1/configs/"):
            name = path[len("/api/v1/configs/") :]
            if not name or "/" in name:
                raise ApiError(404, "未知配置路径")
            if method == "GET":
                self._send_json(200, app.get_config(name))
                return
            if method == "PUT":
                self._send_json(200, app.save_config(name, self._read_json_body()))
                return
            if method == "DELETE":
                self._send_json(200, app.delete_config(name))
                return
            self._require(method, {"GET", "PUT", "DELETE"})
        if path.startswith("/api/v1/jobs/"):
            rest = path[len("/api/v1/jobs/") :]
            if rest.endswith("/stop") and "/" not in rest[: -len("/stop")]:
                job_id = rest[: -len("/stop")]
                self._require(method, {"POST"})
                self._send_json(200, app.stop_job(job_id))
                return
            if not rest or "/" in rest:
                raise ApiError(404, "未知任务路径")
            self._require(method, {"GET"})
            self._send_json(200, app.get_job(rest))
            return
        if path.startswith("/api/v1/runs/"):
            rest = path[len("/api/v1/runs/") :]
            if "/" not in rest:
                if not rest:
                    raise ApiError(404, "未知运行路径")
                self._require(method, {"DELETE"})
                force = (query.get("force") or ["0"])[-1].lower() in {"1", "true", "yes"}
                self._send_json(200, app.delete_run(rest, force=force))
                return
            run_id, action = rest.rsplit("/", 1)
            if not run_id:
                raise ApiError(404, f"未知运行路径: {path}")
            if action not in {"summary", "events", "metrics", "samples", "console", "start", "resume"}:
                raise ApiError(404, f"未知运行子操作: {action}")
            if action == "start":
                self._require(method, {"POST"})
                self._send_json(200, app.start_run(run_id))
                return
            if action == "resume":
                self._require(method, {"POST"})
                self._send_json(200, app.resume_run(run_id))
                return
            self._require(method, {"GET"})
            if action == "summary":
                self._send_json(200, app.run_summary(run_id))
                return
            if action == "events":
                after = self._query_int(query, "after", 0, minimum=0, maximum=10_000_000)
                limit = self._query_int(
                    query, "limit", EVENT_TAIL_DEFAULT, minimum=1, maximum=EVENT_TAIL_MAX
                )
                self._send_json(200, app.run_events(run_id, after, limit))
                return
            if action == "metrics":
                self._send_json(200, app.run_metrics(run_id))
                return
            if action == "samples":
                limit = self._query_int(
                    query, "limit", SAMPLE_LIMIT_DEFAULT, minimum=1, maximum=SAMPLE_LIMIT_MAX
                )
                self._send_json(200, app.run_samples(run_id, limit))
                return
            if action == "console":
                limit = self._query_int(query, "limit", 80, minimum=1, maximum=2000)
                self._send_json(200, app.run_console(run_id, limit))
                return
            raise ApiError(404, f"未知运行子操作: {action}")
        raise ApiError(404, f"未知 API: {path}")

    # -- 请求 / 响应小工具 --------------------------------------------------
    def _require(self, method: str, allowed: set[str]) -> None:
        """断言请求方法在允许集合内，否则抛 ApiError。"""
        if method not in allowed:
            raise ApiError(
                405,
                f"{method} 不被支持；允许: {', '.join(sorted(allowed))}",
                {"Allow": ", ".join(sorted(allowed))},
            )

    def _query_int(
        self,
        query: dict[str, list[str]],
        key: str,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """解析并夹取整数查询参数。"""
        values = query.get(key)
        if not values:
            return default
        raw = values[-1]
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, f"查询参数 {key} 必须是整数") from exc
        if minimum is not None and value < minimum:
            raise ApiError(400, f"查询参数 {key} 不能小于 {minimum}")
        if maximum is not None and value > maximum:
            raise ApiError(400, f"查询参数 {key} 不能大于 {maximum}")
        return value

    def _read_json_body(self) -> dict[str, Any]:
        """读取并解析 JSON 请求体，限制大小并校验类型。"""
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ApiError(415, "请求体 Content-Type 必须是 application/json")
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ApiError(411, "缺少 Content-Length 头")
        try:
            length = int(length_header)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "Content-Length 非法") from exc
        if length < 0:
            raise ApiError(400, "Content-Length 非法")
        if length > MAX_BODY_BYTES:
            raise ApiError(413, f"请求体过大（>{MAX_BODY_BYTES} 字节）")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ApiError(400, "请求体必须是 UTF-8 编码的 JSON") from exc
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApiError(400, f"JSON 解析失败: {exc.msg} (第 {exc.lineno} 行)") from exc
        if not isinstance(body, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        return body

    def _send_json(
        self,
        status: int,
        payload: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        """发送 JSON 响应。"""
        body = json.dumps(_sanitize(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """发送原始字节响应（静态文件等）。"""
        self.send_response(status)
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    # -- 流式生成（NDJSON over chunked） ------------------------------------
    def _serve_inference_stream(self, app: DashboardApp, payload: dict[str, Any]) -> None:
        """Forward worker NDJSON lines to the browser as they arrive.

        Uses ``Transfer-Encoding: chunked`` (the handler speaks HTTP/1.1) so the
        browser's ``fetch`` reader sees each token immediately.  Client aborts
        kill the worker; a watchdog kills workers that exceed the timeout.
        """

        request = app.build_inference_request(payload)
        request["stream"] = True
        process = app.start_inference_process(request)
        watchdog = threading.Timer(
            INFERENCE_TIMEOUT_S, app.kill_inference_process, args=(process,)
        )
        watchdog.daemon = True
        watchdog.start()
        started = False
        saw_terminal = False
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            started = True
            if process.stdout is None:
                raise ApiError(500, "推理子进程没有 stdout")
            for line in process.stdout:
                text = line.strip()
                if not text:
                    continue
                if len(text) > MAX_STREAM_LINE_BYTES:
                    self._write_chunk(
                        json.dumps(
                            {"type": "error", "message": "推理事件过大"}
                        ).encode("utf-8")
                        + b"\n"
                    )
                    saw_terminal = True
                    break
                if _stream_event_type(text) in {"done", "error"}:
                    saw_terminal = True
                self._write_chunk(text.encode("utf-8") + b"\n")
            returncode = process.wait(timeout=10)
            if returncode != 0 and not saw_terminal:
                message = app.inference_error_tail(process)
                self._write_chunk(
                    json.dumps(
                        {"type": "error", "message": message}, ensure_ascii=False
                    ).encode("utf-8")
                    + b"\n"
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # 浏览器已断开（关闭标签页/点停止），finally 里会杀掉 worker
        except Exception as exc:  # noqa: BLE001 - 响应头已发出，只能在流里报错
            if not started:
                raise
            try:
                self._write_chunk(
                    json.dumps(
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
            except OSError:
                pass
        finally:
            watchdog.cancel()
            app.close_inference_process(process)
            if started:
                try:
                    self._end_chunked()
                except OSError:
                    pass

    def _write_chunk(self, data: bytes) -> None:
        """写一个 chunked 编码分块。"""
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self) -> None:
        """结束 chunked 响应。"""
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # 静态资源服务（_serve_static）已拆至 minigpt.web.static_files.StaticFilesMixin，
    # 经类继承并入。


class DashboardHTTPServer(ThreadingHTTPServer):
    """带 ``app`` 引用的线程化 HTTP 服务；关闭时顺带停止训练任务。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: DashboardApp) -> None:
        """绑定地址并保存应用引用。"""
        self.app = app
        super().__init__(address, DashboardHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """浏览器关闭连接是常态（刷新/停止/关页签），不要刷 traceback。"""

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)

    def server_close(self) -> None:
        """关闭监听并通知应用终止子进程。"""
        try:
            if getattr(self, "app", None) is not None:
                self.app.close()
        finally:
            super().server_close()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    project_root: str | Path | None = None,
    runs_dir: str | Path | None = None,
    web_dir: str | Path | None = None,
    configs_dir: str | Path | None = None,
    projects_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    dist_dir: str | Path | None = None,
    max_jobs: int = 1,
    command_builder: Any = None,
    stop_timeout: float = 5.0,
    python_executable: str | None = None,
    verbose: bool = False,
) -> DashboardHTTPServer:
    """创建（但不启动）HTTP 服务；``port=0`` 时由系统分配端口。"""

    app = DashboardApp(
        project_root=project_root,
        runs_dir=runs_dir,
        web_dir=web_dir,
        configs_dir=configs_dir,
        projects_dir=projects_dir,
        data_dir=data_dir,
        dist_dir=dist_dir,
        max_jobs=max_jobs,
        command_builder=command_builder,
        stop_timeout=stop_timeout,
        python_executable=python_executable,
        verbose=verbose,
    )
    try:
        server = DashboardHTTPServer((host, int(port)), app)
    except Exception:
        app.close()
        raise
    return server


__all__ = [
    "ApiError",
    "DashboardApp",
    "DashboardHandler",
    "DashboardHTTPServer",
    "create_server",
    "read_json_tolerant",
    "read_jsonl_tolerant",
    "read_event_window",
    "read_metrics",
    "read_text_tail",
    "STATIC_MIME",
    "RUN_ARTIFACTS",
]
