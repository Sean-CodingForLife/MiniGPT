"""训练任务管理器：看板后端与训练进程之间的唯一桥梁。

设计约束（见 ``CONSTRUCTION.md`` §4.1 / §7.6 / §9）：

* 只用 Python 标准库，**不在模块顶层导入 torch**；没有安装 PyTorch 的机器
  也能运行控制面板。
* 只启动仓库内固定的 ``pretrain.py`` / ``sft.py``，通过 ``--config`` 传入
  冻结后的完整配置，不经过 shell、不接受任意命令。
* 每个运行目录先写 ``resolved_config.json``（启动时的冻结配置），训练进程的
  stdout/stderr 追加到 ``console.log``；运行产物 ``status.json`` /
  ``events.jsonl`` / ``metrics.csv`` / ``samples.jsonl`` / ``training_report.json``
  由训练进程按 CONSTRUCTION §7.4 的 schema 写入。
* 默认最多 1 个活动任务；停止时先 ``terminate``，超时后 ``kill``；
  服务退出时停止所有活动任务，避免孤儿进程。
* 命令构造函数可注入，单元测试用假的 ``sys.executable -c`` 命令替代真实训练，
  不会真的启动 PyTorch。

本模块同时提供 :class:`ConfigStore`（``configs/`` 目录的读写，只允许安全名字、
原子写入）和 :func:`normalize_config`（把任意 JSON 覆盖合并进 ``Config`` 并校验），
供 ``minigpt/web/server.py`` 复用。
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from minigpt.config import Config, default_project_root

# 基础层（异常/常量/校验/配置合并/命令构造/记录）已拆至 job_records.py，
# 配置目录管理拆至 config_store.py；这里 re-export 保持对外导入路径不变。
from minigpt.training.job_records import (  # noqa: F401
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    RUN_ID_PATTERN,
    CONFIG_NAME_PATTERN,
    RUN_ID_MAX_ATTEMPTS,
    STAGE_SCRIPTS,
    CommandBuilder,
    JobManagerError,
    ConfigError,
    JobLimitExceeded,
    JobNotFound,
    JobStartError,
    JobRecord,
    EvalJobRecord,
    default_command_builder,
    normalize_config,
    validate_run_id,
    validate_config_name,
    atomic_write_json,
    iso_now,
    iso_from_mtime,
    _pid_alive,
)
from minigpt.training.config_store import ConfigStore  # noqa: F401


# ---------------------------------------------------------------------------
# 任务管理器
# ---------------------------------------------------------------------------
class JobManager:
    """固定训练脚本的进程管理器（纯标准库、线程安全、可注入命令构造）。"""

    def __init__(
        self,
        project_root: str | Path | None = None,
        runs_dir: str | Path | None = None,
        max_jobs: int = 1,
        command_builder: CommandBuilder | None = None,
        stop_timeout: float = 5.0,
        kill_timeout: float = 5.0,
        poll_interval: float = 0.2,
        history_limit: int = 100,
        python_executable: str | None = None,
    ) -> None:
        """初始化任务管理器：锁、记录表、监控线程与清理策略。"""
        self.project_root = Path(project_root or default_project_root()).expanduser().resolve()
        runs = Path(runs_dir) if runs_dir is not None else Path("runs")
        if not runs.is_absolute():
            runs = self.project_root / runs
        self.runs_dir = runs.expanduser().resolve()
        try:
            self.max_jobs = int(max_jobs)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"max_jobs 必须是正整数，收到 {max_jobs!r}") from exc
        if self.max_jobs < 1:
            raise ConfigError("max_jobs 必须 >= 1")
        self.python_executable = str(python_executable or sys.executable)
        if command_builder is None:
            self.command_builder: CommandBuilder = functools.partial(
                default_command_builder, python_executable=self.python_executable
            )
        else:
            self.command_builder = command_builder
        self.stop_timeout = float(stop_timeout)
        self.kill_timeout = float(kill_timeout)
        self.poll_interval = max(0.02, float(poll_interval))
        self.history_limit = max(1, int(history_limit))

        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._log_handles: dict[str, Any] = {}
        # 本会话已成功拉起过进程的 run 集合：并发续跑守卫用（进程可能瞬间
        # 退出并被收尾，仅靠 is_active/磁盘 mtime 不足以拒绝第二次 resume）。
        self._launched_runs: set[str] = set()
        # N08 面板内评估：每 run 至多一个活动评估，与训练槽位分开管理。
        self._eval_jobs: dict[str, EvalJobRecord] = {}
        self._eval_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._eval_log_handles: dict[str, Any] = {}
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._closed = False
        # 可选回调：评估进入终态时通知上层（如写 run 事件）；异常不影响收尾。
        self.eval_finish_hook: Callable[[EvalJobRecord], None] | None = None
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # -- 查询 ---------------------------------------------------------------
    def get_job(self, job_id: str) -> JobRecord | None:
        """按 id 取任务记录，不存在返回 ``None``。"""

        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int | None = None) -> list[JobRecord]:
        """返回本服务启动过的任务（新的在前，最多 ``history_limit`` 条）。"""

        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda r: r.started_at, reverse=True)
        limit = self.history_limit if limit is None else max(0, int(limit))
        return jobs[:limit]

    def active_count(self) -> int:
        """当前活动任务数。"""

        with self._lock:
            return sum(1 for record in self._jobs.values() if record.is_active)

    # -- 创建 / 启动 ---------------------------------------------------------
    def _ensure_slot_locked(self) -> None:
        """在持锁状态下检查活动任务数是否已达上限，超限抛 JobLimitExceeded。"""
        if self._closed:
            raise JobStartError("任务管理器已关闭，无法启动新任务")
        active = [record for record in self._jobs.values() if record.is_active]
        if len(active) >= self.max_jobs:
            raise JobLimitExceeded(
                f"最多允许 {self.max_jobs} 个活动任务，当前已有 {len(active)} 个；请先停止正在运行的任务"
            )

    def _make_record(
        self,
        run_id: str,
        frozen: dict[str, Any],
        run_dir: Path,
        config_path: Path,
        console_path: Path,
        preset: str | None,
        status: str,
    ) -> JobRecord:
        """根据冻结配置与 run 目录构造一条内存中的任务记录。"""
        try:
            command = list(self.command_builder(frozen, self.project_root, config_path))
        except JobManagerError:
            raise
        except Exception as exc:  # noqa: BLE001 - 注入的构造器可能抛任意异常
            raise JobStartError(f"构造启动命令失败: {exc}") from exc
        if not command or not all(isinstance(part, str) for part in command):
            raise JobStartError("启动命令必须是非空字符串列表")
        return JobRecord(
            run_id=run_id,
            stage=str(frozen.get("stage", "pretrain")),
            preset=str(preset or ""),
            status=status,
            run_dir=str(run_dir),
            config_path=str(config_path),
            console_log=str(console_path),
            command=command,
            project_root=str(self.project_root),
            started_at=iso_now(),
            max_steps=int(frozen.get("max_steps", 0) or 0),
        )

    def create_run(
        self,
        config: dict[str, Any] | None = None,
        preset: str | None = None,
        run_name: str | None = None,
    ) -> JobRecord:
        """Create a **draft** run: frozen config + status.json, but start no process.

        这是“新建任务”的原子操作；真正启动由 :meth:`start_run` 完成。
        """

        prepared = normalize_config(
            config_data=config,
            preset=preset,
            run_name=run_name,
            project_root=self.project_root,
        )
        # 血缘：父运行只信服务端从 init_from 推导（runs/<id>/…），拒绝客户端直传。
        prepared.parent_run_id = ""
        if prepared.init_from:
            init_path = Path(prepared.init_from).expanduser()
            if not init_path.is_absolute():
                init_path = self.project_root / init_path
            try:
                relative = init_path.resolve().relative_to(self.runs_dir)
            except ValueError:
                relative = None
            if relative is not None and len(relative.parts) >= 2:
                prepared.parent_run_id = relative.parts[0]
        if prepared.project_id:
            from minigpt.training.project_store import ProjectStore

            store = ProjectStore(project_root=self.project_root)
            if not store.exists(prepared.project_id):
                raise ConfigError(f"项目不存在: {prepared.project_id}")
        with self._lock:
            if self._closed:
                raise JobStartError("任务管理器已关闭，无法创建新任务")
            prepared.runs_dir = str(self.runs_dir)
            run_id = self._allocate_run_id(prepared)
            prepared.run_name = run_id
            prepared.project_root = str(self.project_root)
            frozen = prepared.to_dict()
            run_dir = self.runs_dir / run_id
            config_path = run_dir / "resolved_config.json"
            console_path = run_dir / "console.log"

            # 先构造命令再建目录：脚本缺失/构造失败时不会留下空目录。
            record = self._make_record(
                run_id, frozen, run_dir, config_path, console_path, preset, "created"
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            atomic_write_json(config_path, frozen)
            console_path.touch()
            atomic_write_json(
                run_dir / "status.json",
                {
                    "status": "created",
                    "stage": record.stage,
                    "step": 0,
                    "max_steps": record.max_steps,
                    "run_id": run_id,
                    "updated_at": iso_now(),
                },
            )
            self._jobs[run_id] = record
            if prepared.project_id:
                from minigpt.training.project_store import ProjectStore

                ProjectStore(project_root=self.project_root).attach_run(prepared.project_id, run_id)
            return record

    def start_job(
        self,
        config: dict[str, Any] | None = None,
        preset: str | None = None,
        run_name: str | None = None,
    ) -> JobRecord:
        """Create a run and start it immediately (legacy convenience API)."""

        with self._lock:
            self._ensure_slot_locked()
        record = self.create_run(config=config, preset=preset, run_name=run_name)
        return self._launch_record(record)

    def start_run(self, run_id: str, resume: bool | None = None) -> JobRecord:
        """Start an existing run (created draft / stopped / failed) atomically.

        ``resume=None`` auto-detects: a run with ``last.pt`` that is not a fresh
        draft is resumed (``--resume``), otherwise it starts from scratch.
        """

        run_id = validate_run_id(str(run_id))
        with self._lock:
            record = self._jobs.get(run_id)
        if record is None:
            record = self._record_from_disk(run_id)
        if resume is None:
            run_dir = Path(record.run_dir)
            has_checkpoint = (run_dir / "last.pt").is_file() or (run_dir / "last.mgckpt").is_file()
            resume = has_checkpoint and record.status != "created"
        return self._launch_record(record, resume=bool(resume))

    def resume_job(self, run_id: str) -> JobRecord:
        """一键断点续训：复用该 run 的冻结配置，以 ``--resume auto`` 重新拉起。

        * 读取 ``resolved_config.json``（缺失时回退 ``run_config.json`` 的
          ``config`` 段）作为原配置，不重写冻结文件；
        * run 目录必须有 ``last.pt``（pure 后端 ``last.mgckpt``），否则明确报错；
        * 成功拉起后向 ``events.jsonl`` 追加 ``resumed`` 事件（状态流转标记）。
        """

        run_id = validate_run_id(str(run_id))
        attempt_started = time.time()
        with self._lock:
            record = self._jobs.get(run_id)
        if record is None:
            record = self._record_from_disk(run_id)
        run_dir = Path(record.run_dir)
        if not ((run_dir / "last.pt").is_file() or (run_dir / "last.mgckpt").is_file()):
            raise JobStartError(
                f"运行 {run_id} 没有可续训的检查点（run 目录缺少 last.pt / last.mgckpt）"
            )
        frozen = self._load_frozen_config(run_id, run_dir)
        frozen_resume = dict(frozen)
        frozen_resume["resume"] = "auto"
        config_path = run_dir / "resolved_config.json"
        if not config_path.is_file():
            # 历史运行（CLI 训练）只有 run_config.json（含 argv/config 包装），
            # 不能直接当 --config 传入；把提取出的纯配置物化为冻结文件。
            config_path.write_text(
                json.dumps(frozen_resume, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        with self._lock:
            # 锁内复检：两个并发 resume 都可能通过锁外的早期检查，只有
            # 在这里串行化才能保证同一 run 目录只拉起一个续训进程。
            current = self._jobs.get(run_id) or record
            if current.is_active:
                raise JobStartError(f"运行 {run_id} 已经在运行，无法重复续训")
            # 本会话已拉起过该 run 的进程（可能瞬间退出并被收尾）：拒绝并发
            # 的第二次 resume。历史运行（本会话未拉起过）不受影响。
            if run_id in self._launched_runs:
                raise JobStartError(f"运行 {run_id} 的最近一次进程已收尾，无法重复续训")
            # 并发 resume 可能在本线程锁外检查后瞬间拉起并结束进程：
            # 若 status.json 在本次尝试开始之后被写成终态，说明该 run 刚被
            # 另一并发 resume 占用并收尾，拒绝重复续跑。历史运行的终态文件
            # 修改时间早于本次尝试，合法续跑不受影响（用 mtime 避免秒级误伤）。
            status_path = run_dir / "status.json"
            try:
                loaded = json.loads(status_path.read_text(encoding="utf-8"))
                disk_mtime = status_path.stat().st_mtime
            except (OSError, ValueError):
                loaded = None
                disk_mtime = 0.0
            if isinstance(loaded, dict):
                disk_status = str(loaded.get("status") or "")
                if disk_status in TERMINAL_STATUSES and disk_mtime >= attempt_started:
                    raise JobStartError(f"运行 {run_id} 已处于终态（{disk_status}），无法重复续训")
            self._ensure_slot_locked()
            new_record = self._make_record(
                run_id,
                frozen_resume,
                run_dir,
                config_path,
                run_dir / "console.log",
                record.preset or None,
                "created",
            )
            new_record.command = list(new_record.command) + ["--resume", "auto"]
            # 关键：在锁内就把新记录占位为 starting 再登记。若留到 _launch_record
            # 里才标记，两个并发线程会各自持有 created 状态的新记录、都通过
            # is_active 检查，导致双进程写同一 run 目录。
            new_record.status = "starting"
            self._jobs[run_id] = new_record
        launched = self._launch_record(new_record, resume=False, premarked=True)
        with self._lock:
            self._launched_runs.add(run_id)
        self._append_run_event(
            run_dir,
            run_id,
            "resumed",
            {"source": "job_manager", "pid": launched.pid},
        )
        return launched

    def _load_frozen_config(self, run_id: str, run_dir: Path) -> dict[str, Any]:
        """读取 run 的冻结配置：优先 resolved_config.json，回退 run_config.json。"""

        for name in ("resolved_config.json", "run_config.json"):
            path = run_dir / name
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if name == "run_config.json" and isinstance(data, dict):
                data = data.get("config")
            if isinstance(data, dict):
                return data
        raise JobNotFound(f"运行 {run_id} 缺少可用的冻结配置（resolved_config.json / run_config.json）")

    def _append_run_event(self, run_dir: Path, run_id: str, event: str, payload: dict[str, Any]) -> None:
        """向 events.jsonl 追加一条与训练进程同 schema 的事件（尽力而为）。"""

        step = 0
        try:
            status_data = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            if isinstance(status_data, dict):
                step = int(status_data.get("step") or 0)
        except (OSError, ValueError, TypeError):
            step = 0
        record = {
            "v": 1,
            "ts": iso_now(),
            "run_id": run_id,
            "event": event,
            "step": step,
            "payload": payload,
        }
        try:
            with open(run_dir / "events.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        except OSError:
            pass

    def forget_job(self, run_id: str) -> None:
        """Drop a finished job record (used after deleting its run directory)."""

        with self._lock:
            record = self._jobs.get(run_id)
            if record is not None and record.is_active:
                raise JobManagerError(f"任务 {run_id} 仍在运行，不能移除记录")
            self._jobs.pop(run_id, None)

    def _record_from_disk(self, run_id: str) -> JobRecord:
        """Rebuild an in-memory record for a run the server did not start."""

        run_dir = self.runs_dir / run_id
        if not run_dir.is_dir():
            raise JobNotFound(f"运行不存在: {run_id}")
        # 历史运行（CLI 直接训练）只有 run_config.json，没有面板冻结文件；
        # 复用统一的回退读取，让续训/启动对这类运行同样可用。
        frozen = self._load_frozen_config(run_id, run_dir)
        config_path = run_dir / "resolved_config.json"
        if not config_path.is_file():
            config_path = run_dir / "run_config.json"
        status_data: dict[str, Any] = {}
        try:
            loaded = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                status_data = loaded
        except (OSError, ValueError):
            status_data = {}
        status = str(status_data.get("status") or "created")
        if status in ACTIVE_STATUSES:
            # The server restarted; only trust "running" if the recorded pid is
            # still alive, otherwise a stale status would let start_run launch a
            # second --resume process double-writing the same run directory.
            pid = status_data.get("pid")
            status = "running" if _pid_alive(pid) else "stopped"
        record = self._make_record(
            run_id,
            frozen,
            run_dir,
            config_path,
            run_dir / "console.log",
            None,
            status,
        )
        with self._lock:
            self._jobs[run_id] = record
        return record

    def _launch_record(self, record: JobRecord, resume: bool = False, premarked: bool = False) -> JobRecord:
        """启动子进程并把记录登记到活动表；resume 决定是否续训。"""
        with self._lock:
            if self._closed:
                raise JobStartError("任务管理器已关闭，无法启动新任务")
            # premarked=True：调用方已在锁内把记录占位为 starting 并完成
            # is_active/槽位检查（resume_job 的并发串行化）。这里不能重复检查：
            # 占位记录自身已是活动状态，再查会把自己算进槽位/误判为重复运行。
            if not premarked:
                if record.is_active:
                    raise JobStartError(f"运行 {record.run_id} 已经在运行")
                self._ensure_slot_locked()
            command = list(record.command)
            if resume and "--resume" not in command:
                command = command + ["--resume"]
            record.command = command
            record.status = "starting"
            record.started_at = iso_now()
            record.ended_at = None
            record.returncode = None
            record.error = None
            record.stop_requested = False
            record._started_monotonic = time.monotonic()
            record._ended_monotonic = None

            console_path = Path(record.console_log)
            env = self._child_environment()
            log_handle = None
            try:
                log_handle = open(console_path, "ab", buffering=0)  # noqa: SIM115 - 交给监控线程关闭
                process = subprocess.Popen(
                    command,
                    cwd=str(self.project_root),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            except Exception as exc:  # noqa: BLE001 - 启动失败统一转成 JobStartError
                try:
                    if log_handle is not None:
                        log_handle.close()
                except OSError:
                    pass
                message = f"[job_manager] 无法启动训练进程: {exc}\n"
                try:
                    with open(console_path, "a", encoding="utf-8") as handle:
                        handle.write(message)
                except OSError:
                    pass
                record.status = "failed"
                record.error = f"无法启动训练进程: {exc}"
                record.ended_at = iso_now()
                record._ended_monotonic = time.monotonic()
                self._write_terminal_status(record)
                raise JobStartError(record.error) from exc

            record.pid = process.pid
            record.status = "running"
            self._processes[record.run_id] = process
            self._log_handles[record.run_id] = log_handle
            self._ensure_monitor_locked()
            return record

    # -- N08 面板内评估 ------------------------------------------------------
    def get_eval_job(self, run_id: str) -> EvalJobRecord | None:
        """返回某个 run 当前关联的评估任务记录（没有则 None）。"""
        with self._lock:
            return self._eval_jobs.get(run_id)

    def start_eval_job(
        self,
        run_id: str,
        command: list[str],
        *,
        dataset: str,
        split: str,
        eval_dir: str,
        run_dir: str | Path,
        console_log: str | Path,
        status_path: str | Path,
    ) -> EvalJobRecord:
        """启动一个评估子进程（固定命令由调用方构造，不经过 shell）。

        复用训练任务的进程管理模式：同一把锁、console 日志句柄、监控线程、
        原子状态文件；但不占用训练任务槽位（评估与训练可并行，互不挤占）。
        """

        run_id = validate_run_id(str(run_id))
        if not command or not all(isinstance(part, str) for part in command):
            raise JobStartError("评估命令必须是非空字符串列表")
        run_dir = Path(run_dir)
        console_path = Path(console_log)
        status_path = Path(status_path)
        with self._lock:
            if self._closed:
                raise JobStartError("任务管理器已关闭，无法启动评估")
            existing = self._eval_jobs.get(run_id)
            if existing is not None and existing.is_active:
                raise JobLimitExceeded(f"运行 {run_id} 已有评估在进行，请等待完成")
            record = EvalJobRecord(
                run_id=run_id,
                dataset=str(dataset),
                split=str(split),
                eval_dir=str(eval_dir),
                status="starting",
                run_dir=str(run_dir),
                console_log=str(console_path),
                command=list(command),
                started_at=iso_now(),
            )
            env = self._child_environment()
            log_handle = None
            try:
                log_handle = open(console_path, "ab", buffering=0)  # noqa: SIM115
                process = subprocess.Popen(
                    list(command),
                    cwd=str(self.project_root),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            except Exception as exc:  # noqa: BLE001 - 启动失败统一转成 JobStartError
                try:
                    if log_handle is not None:
                        log_handle.close()
                except OSError:
                    pass
                record.status = "failed"
                record.error = f"无法启动评估进程: {exc}"
                record.ended_at = iso_now()
                record._ended_monotonic = time.monotonic()
                self._eval_jobs[run_id] = record
                self._write_eval_status(record, status_path)
                raise JobStartError(record.error) from exc
            record.pid = process.pid
            record.status = "running"
            self._eval_jobs[run_id] = record
            self._eval_processes[run_id] = process
            self._eval_log_handles[run_id] = log_handle
            self._write_eval_status(record, status_path)
            self._ensure_monitor_locked()
            return record

    def _write_eval_status(self, record: EvalJobRecord, status_path: Path) -> None:
        """评估状态文件（与训练 status.json 同风格的原子写）。"""

        try:
            atomic_write_json(
                status_path,
                {
                    "kind": "evaluation",
                    "run_id": record.run_id,
                    "dataset": record.dataset,
                    "split": record.split,
                    "eval_dir": record.eval_dir,
                    "status": record.status,
                    "pid": record.pid,
                    "started_at": record.started_at,
                    "ended_at": record.ended_at,
                    "returncode": record.returncode,
                    "error": record.error,
                    "updated_at": iso_now(),
                },
            )
        except OSError:
            pass

    def _finalize_eval(self, run_id: str, returncode: int | None) -> EvalJobRecord | None:
        """把评估进程退出结果固化为终态（幂等；调用方不持锁）。"""

        changed = False
        with self._lock:
            record = self._eval_jobs.get(run_id)
            if record is None or not record.is_active:
                return record
            record.status = "completed" if returncode == 0 else "failed"
            if returncode != 0:
                record.error = f"评估进程退出码 {returncode}（详见 eval_console.log）"
            record.returncode = returncode
            record.ended_at = iso_now()
            record._ended_monotonic = time.monotonic()
            self._eval_processes.pop(run_id, None)
            handle = self._eval_log_handles.pop(run_id, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            status_path = Path(record.run_dir) / "eval_status.json"
            self._write_eval_status(record, status_path)
            changed = True
        if changed:
            hook = self.eval_finish_hook
            if hook is not None:
                try:
                    hook(record)
                except Exception:  # noqa: BLE001 - 上层回调失败不影响收尾
                    pass
        return record

    def stop_eval_job(self, run_id: str, timeout: float | None = None) -> EvalJobRecord | None:
        """停止一个活动评估：先 terminate，超时后 kill。"""

        with self._lock:
            record = self._eval_jobs.get(run_id)
            process = self._eval_processes.get(run_id)
            if record is None or not record.is_active or process is None:
                return record
        wait_timeout = self.stop_timeout if timeout is None else float(timeout)
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=self.kill_timeout)
            except subprocess.TimeoutExpired:
                pass
        return self._finalize_eval(run_id, process.poll())

    def stop_all_evals(self) -> None:
        """停止所有活动的评估子进程（服务关闭时调用）。"""
        with self._lock:
            active_ids = [
                run_id for run_id, record in self._eval_jobs.items() if record.is_active
            ]
        for run_id in active_ids:
            try:
                self.stop_eval_job(run_id)
            except JobManagerError:
                continue

    # -- 停止 / 收尾 --------------------------------------------------------
    def stop_job(self, job_id: str, timeout: float | None = None) -> JobRecord:
        """停止一个任务：先 terminate，超时后 kill；返回最终记录。"""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise JobNotFound(f"任务不存在: {job_id}")
            process = self._processes.get(job_id)
            if not record.is_active or process is None:
                return record
            record.stop_requested = True

        wait_timeout = self.stop_timeout if timeout is None else float(timeout)
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=self.kill_timeout)
            except subprocess.TimeoutExpired:
                pass
        return self._finalize(job_id, process.poll())

    def stop_all(self) -> list[JobRecord]:
        """停止所有活动任务。"""

        with self._lock:
            active_ids = [record.run_id for record in self._jobs.values() if record.is_active]
        results: list[JobRecord] = []
        for job_id in active_ids:
            try:
                results.append(self.stop_job(job_id))
            except JobManagerError:
                continue
        return results

    def shutdown(self) -> None:
        """服务退出：停止活动任务、结束监控线程、释放日志句柄。"""

        with self._lock:
            self._closed = True
        self.stop_all()
        self.stop_all_evals()
        self._monitor_stop.set()
        monitor = self._monitor
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=max(2.0, self.stop_timeout + 1.0))
        with self._lock:
            for handle in list(self._log_handles.values()) + list(self._eval_log_handles.values()):
                try:
                    handle.close()
                except OSError:
                    pass
            self._log_handles.clear()
            self._processes.clear()
            self._eval_log_handles.clear()
            self._eval_processes.clear()

    close = shutdown

    def __enter__(self) -> "JobManager":
        """支持 with 语法：返回自身。"""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """退出 with 时关闭管理器，终止仍在运行的子进程。"""
        self.shutdown()

    def poll(self) -> None:
        """检查活动进程，刷新终态记录（监控线程每 ``poll_interval`` 秒调用）。"""

        with self._lock:
            candidates = [
                (job_id, self._processes.get(job_id))
                for job_id, record in self._jobs.items()
                if record.is_active
            ]
        for job_id, process in candidates:
            if process is None:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            self._finalize(job_id, returncode)
        with self._lock:
            eval_candidates = [
                (run_id, self._eval_processes.get(run_id))
                for run_id, record in self._eval_jobs.items()
                if record.is_active
            ]
        for run_id, process in eval_candidates:
            if process is None:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            self._finalize_eval(run_id, returncode)

    def wait_for(
        self,
        job_id: str,
        timeout: float = 30.0,
        statuses: set[str] | frozenset[str] | None = None,
    ) -> JobRecord:
        """阻塞等待任务进入给定状态集合（测试与命令行复用）。"""

        wanted = set(statuses) if statuses is not None else set(TERMINAL_STATUSES)
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            record = self.get_job(job_id)
            if record is None:
                raise JobNotFound(f"任务不存在: {job_id}")
            if record.status in wanted:
                return record
            self.poll()
            record = self.get_job(job_id)
            if record is not None and record.status in wanted:
                return record
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待任务 {job_id} 进入 {sorted(wanted)} 超时（当前 {record.status if record else 'missing'}）"
                )
            time.sleep(0.05)

    # -- 内部实现 -----------------------------------------------------------
    def _child_environment(self) -> dict[str, str]:
        """子进程环境：固定 MINIGPT_ROOT，强制无缓冲 UTF-8 输出。"""

        env = dict(os.environ)
        env["MINIGPT_ROOT"] = str(self.project_root)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def _allocate_run_id(self, config: Config) -> str:
        """生成唯一、安全的运行 id（目录名）。

        冲突时追加 ``-2``、``-3``…后缀；后缀**不截断**（截断会让不同序号
        塌缩成同一候选，造成死循环），并设重试上限，超限抛明确错误。
        """

        requested = (config.run_name or "").strip()
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", requested).strip("._-") if requested else ""
        if not base:
            base = f"{config.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
        base = base[:120]
        if not RUN_ID_PATTERN.match(base):
            base = f"{config.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
        candidate = base
        for suffix in range(2, 2 + RUN_ID_MAX_ATTEMPTS):
            if not (self.runs_dir / candidate).exists():
                return candidate
            # 不截断：base<=120 字符 + 短后缀仍满足 RUN_ID_PATTERN 的 128 上限。
            candidate = f"{base}-{suffix}"
        raise ConfigError(
            f"运行名 {base!r} 的冲突候选过多（已尝试 {RUN_ID_MAX_ATTEMPTS} 次），请换一个运行名"
        )

    def _ensure_monitor_locked(self) -> None:
        """惰性启动进程监控线程（调用方需持有锁）。"""

        if self._monitor is not None and self._monitor.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="minigpt-job-monitor", daemon=True
        )
        self._monitor.start()

    def _monitor_loop(self) -> None:
        """后台轮询子进程状态；任何异常都不能让监控线程退出。"""

        while not self._monitor_stop.wait(self.poll_interval):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - 监控线程必须常驻
                continue

    def _write_terminal_status(self, record: JobRecord) -> None:
        """进程退出后把磁盘 status.json 补写为终态（读-合并-写）。

        训练进程存活期间 status.json 由 TelemetryCallback 独写；这里只在
        进程已退出（或根本启动失败）后调用，不存在活写者竞态。保留训练器
        已写的 step/best_val/started_at 等字段；硬 kill 时磁盘不再滞留 running。
        """

        path = Path(record.run_dir) / "status.json"
        data: dict[str, Any] = {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
        data.update(
            {
                "run_id": record.run_id,
                "stage": record.stage,
                "status": record.status,
                "pid": record.pid,
                "error": record.error or "",
                "updated_at": iso_now(),
            }
        )
        data.setdefault("step", 0)
        data.setdefault("max_steps", record.max_steps)
        data.setdefault("best_val", None)
        data.setdefault("started_at", record.started_at)
        try:
            atomic_write_json(path, data)
        except OSError:
            # 磁盘补写失败不应影响内存终态与收尾流程。
            pass

    def _finalize(self, job_id: str, returncode: int | None) -> JobRecord | None:
        """把进程退出结果固化为终态（幂等）。"""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or not record.is_active:
                return record
            if record.stop_requested:
                record.status = "stopped"
                record.error = record.error or None
            elif returncode == 0:
                record.status = "completed"
            else:
                record.status = "failed"
                record.error = f"训练进程退出码 {returncode}（详见 console.log）"
            record.returncode = returncode
            record.ended_at = iso_now()
            record._ended_monotonic = time.monotonic()
            self._processes.pop(job_id, None)
            handle = self._log_handles.pop(job_id, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            self._write_terminal_status(record)
            return record


# ConfigStore（configs/ 目录读写）已拆至 minigpt.training.config_store，
# 经头部 import re-export，对外导入路径不变。


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "RUN_ID_PATTERN",
    "CONFIG_NAME_PATTERN",
    "CommandBuilder",
    "JobManagerError",
    "ConfigError",
    "JobLimitExceeded",
    "JobNotFound",
    "JobStartError",
    "JobRecord",
    "EvalJobRecord",
    "JobManager",
    "ConfigStore",
    "default_command_builder",
    "normalize_config",
    "validate_run_id",
    "validate_config_name",
    "atomic_write_json",
    "iso_now",
    "iso_from_mtime",
]
