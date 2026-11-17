"""任务管理器的基础层：异常、常量、校验工具、配置合并、命令构造与任务记录。

从 job_manager.py 拆出（阶段 4）。job_manager.py 仍 re-export 这里的全部
公共名称，对外导入路径不变。本模块只依赖标准库与 minigpt.config，不导入
torch，也不依赖 JobManager 本体（避免循环）。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from minigpt.config import Config, default_project_root, get_preset

# 活动 / 终态状态集合（与 CONSTRUCTION §7.4 status.json 的取值保持一致）
ACTIVE_STATUSES = frozenset({"starting", "running", "stopping"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped"})

# 运行目录名 / 配置文件名：只允许 ASCII 安全字符，杜绝路径穿越
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CONFIG_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# run_id 冲突后缀的最大尝试次数；超限抛 ConfigError，绝不死循环。
RUN_ID_MAX_ATTEMPTS = 100

STAGE_SCRIPTS = {"pretrain": "pretrain.py", "sft": "sft.py"}

# 命令构造函数：(冻结配置, 项目根, 冻结配置文件路径) -> argv（不含 shell）
CommandBuilder = Callable[[dict[str, Any], Path, Path], list[str]]


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class JobManagerError(RuntimeError):
    """任务管理器异常的基类。"""


class ConfigError(JobManagerError, ValueError):
    """配置不合法（字段未知、类型错误或取值越界）。"""


class JobLimitExceeded(JobManagerError):
    """活动任务数量达到上限。"""


class JobNotFound(JobManagerError):
    """任务 / 运行 / 配置不存在。"""


class JobStartError(JobManagerError):
    """训练进程无法启动。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def iso_now() -> str:
    """返回带本地时区偏移的 ISO-8601 时间字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def iso_from_mtime(path: Path) -> str | None:
    """返回文件修改时间对应的 ISO-8601 字符串；文件不存在时返回 ``None``。"""

    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return None


def validate_run_id(run_id: str) -> str:
    """校验运行 id，返回原值；不合法时抛 :class:`ConfigError`。"""

    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise ConfigError(f"非法的运行 id: {run_id!r}；只允许字母、数字、点、下划线、连字符")
    return run_id


def validate_config_name(name: str) -> str:
    """校验配置名（可带 ``.json`` 后缀），返回去掉后缀的安全名字。"""

    if not isinstance(name, str):
        raise ConfigError("配置名必须是字符串")
    raw = name.strip()
    if raw.lower().endswith(".json"):
        raw = raw[:-5]
    if not raw:
        raise ConfigError("配置名不能为空")
    if not CONFIG_NAME_PATTERN.match(raw):
        raise ConfigError(
            f"非法的配置名: {name!r}；只能包含字母、数字、点、下划线和连字符，且不能以符号开头"
        )
    return raw


def _atomic_write_text(path: Path, text: str) -> None:
    """同目录临时文件 + 原子替换（统一委托 minigpt.utils.atomic_write_text）。"""

    from minigpt.utils import atomic_write_text

    atomic_write_text(path, text)


def atomic_write_json(path: Path, data: Any) -> None:
    """把 JSON 对象原子写入文件（UTF-8、缩进、末尾换行）。"""

    _atomic_write_text(Path(path), json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------
# 配置合并与校验
# ---------------------------------------------------------------------------
def normalize_config(
    config_data: dict[str, Any] | None = None,
    preset: str | None = None,
    run_name: str | None = None,
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """把「预设 + JSON 覆盖 + 额外覆盖」合并成校验过的 :class:`Config`。

    校验包含：

    * 未知字段拒绝（``Config.apply_dict``）；
    * 路径全部解析为绝对路径（相对项目根）；
    * 业务取值校验（``Config.validate``）。

    失败统一抛 :class:`ConfigError`，方便 Web API 映射成 400。
    """

    if config_data is None:
        config_data = {}
    if overrides is None:
        overrides = {}
    if not isinstance(config_data, dict):
        raise ConfigError("config 必须是 JSON 对象")
    if not isinstance(overrides, dict):
        raise ConfigError("overrides 必须是 JSON 对象")

    data: dict[str, Any] = {}
    data.update(config_data)
    data.update(overrides)

    # Security: the dashboard must not be able to load arbitrary local Python
    # files. ``plugins`` is exec_module'd by the training subprocess, so a
    # web-submitted path would be a remote code execution vector (combined with
    # CSRF / DNS rebinding). CLI users can still pass --plugins directly.
    if str(data.get("plugins") or "").strip():
        raise ConfigError("Web 任务不允许配置 plugins（可执行任意本地脚本）；请改用命令行 --plugins")

    try:
        if preset is not None:
            if not isinstance(preset, str):
                raise ConfigError("preset 必须是字符串")
            try:
                config = get_preset(preset)
            except (KeyError, ValueError) as exc:
                raise ConfigError(str(exc)) from exc
        else:
            stage = data.get("stage") or "pretrain"
            config = get_preset("sft_debug_en" if str(stage) == "sft" else "debug_en")

        root = Path(project_root or default_project_root()).expanduser().resolve()
        config.project_root = str(root)
        config.apply_dict(data)  # 未知字段在这里报错
        config.project_root = str(root)  # 冻结到本服务所在项目根，避免跨机路径
        if run_name is not None:
            config.run_name = str(run_name)
        config.resolve_paths()
        config.validate()
    except ConfigError:
        raise
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


# ---------------------------------------------------------------------------
# 命令构造
# ---------------------------------------------------------------------------
def default_command_builder(
    config: dict[str, Any],
    project_root: Path,
    config_file: Path,
    python_executable: str | None = None,
) -> list[str]:
    """默认命令：``python -u pretrain.py|sft.py --config <冻结配置>``。

    只根据 ``stage`` 选择仓库内固定脚本，禁止任意命令。``python_executable``
    默认使用当前解释器，可指向带 PyTorch 的虚拟环境（看板本身不需要 torch）。
    """

    stage = str(config.get("stage", "pretrain"))
    script_name = STAGE_SCRIPTS.get(stage)
    if script_name is None:
        raise ConfigError(f"未知训练阶段: {stage!r}；只支持 pretrain / sft")
    script = Path(project_root) / script_name
    if not script.is_file():
        raise JobStartError(f"训练脚本不存在: {script}")
    return [
        str(python_executable or sys.executable),
        "-u",
        str(script),
        "--config",
        str(Path(config_file)),
    ]


# ---------------------------------------------------------------------------
# 任务记录
# ---------------------------------------------------------------------------
@dataclass
class JobRecord:
    """单个训练任务的内存状态（可序列化为 API JSON）。"""

    run_id: str
    stage: str
    preset: str
    status: str
    run_dir: str
    config_path: str
    console_log: str
    command: list[str]
    project_root: str
    started_at: str
    max_steps: int
    pid: int | None = None
    ended_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    stop_requested: bool = False
    _started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    _ended_monotonic: float | None = field(default=None, repr=False)

    @property
    def job_id(self) -> str:
        """任务 id 与运行 id 相同，便于 ``/api/v1/jobs/<id>`` 与运行目录互查。"""

        return self.run_id

    @property
    def is_active(self) -> bool:
        """进程是否仍在运行（含正在启动/停止）。"""

        return self.status in ACTIVE_STATUSES

    @property
    def elapsed_s(self) -> float:
        """已运行秒数（活动任务实时增长）。"""

        end = self._ended_monotonic if self._ended_monotonic is not None else time.monotonic()
        return round(max(0.0, end - self._started_monotonic), 1)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的任务视图。"""

        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "preset": self.preset,
            "status": self.status,
            "active": self.is_active,
            "pid": self.pid,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_s": self.elapsed_s,
            "returncode": self.returncode,
            "error": self.error,
            "run_dir": self.run_dir,
            "config_path": self.config_path,
            "console_log": self.console_log,
            "command": list(self.command),
            "max_steps": self.max_steps,
            "stop_requested": self.stop_requested,
            "source": "job_manager",
        }


@dataclass
class EvalJobRecord:
    """单个面板内评估任务的内存状态（N08；与训练任务分开计槽）。"""

    run_id: str
    dataset: str
    split: str
    eval_dir: str
    status: str
    run_dir: str
    console_log: str
    command: list[str]
    started_at: str
    pid: int | None = None
    ended_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    _started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    _ended_monotonic: float | None = field(default=None, repr=False)

    @property
    def is_active(self) -> bool:
        """评估任务是否仍在运行。"""
        return self.status in ACTIVE_STATUSES

    @property
    def elapsed_s(self) -> float:
        """评估任务已运行秒数（结束后为总耗时）。"""
        end = self._ended_monotonic if self._ended_monotonic is not None else time.monotonic()
        return round(max(0.0, end - self._started_monotonic), 1)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 API/前端使用的字典。"""
        return {
            "kind": "evaluation",
            "run_id": self.run_id,
            "dataset": self.dataset,
            "split": self.split,
            "eval_dir": self.eval_dir,
            "status": self.status,
            "active": self.is_active,
            "pid": self.pid,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_s": self.elapsed_s,
            "returncode": self.returncode,
            "error": self.error,
            "run_dir": self.run_dir,
            "console_log": self.console_log,
            "command": list(self.command),
            "source": "job_manager",
        }


def _pid_alive(pid: int | None) -> bool:
    """Best-effort check whether a process id is still alive (cross-platform)."""

    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == 259  # STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "RUN_ID_PATTERN",
    "CONFIG_NAME_PATTERN",
    "RUN_ID_MAX_ATTEMPTS",
    "STAGE_SCRIPTS",
    "CommandBuilder",
    "JobManagerError",
    "ConfigError",
    "JobLimitExceeded",
    "JobNotFound",
    "JobStartError",
    "JobRecord",
    "EvalJobRecord",
    "default_command_builder",
    "normalize_config",
    "validate_run_id",
    "validate_config_name",
    "atomic_write_json",
    "iso_now",
    "iso_from_mtime",
    "_pid_alive",
]