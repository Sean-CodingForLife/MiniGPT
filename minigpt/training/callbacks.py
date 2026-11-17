"""训练 callback：telemetry、样本记录与扩展点。

callback 接收 :class:`TrainingState`，由训练器在固定时机调用。单个 callback
抛出的异常只打印堆栈，绝不会中断训练：telemetry 必须不可能毁掉一次实验。

内置实现通过 ``minigpt.registry`` 注册为 ``telemetry`` / ``sample``，由配置里的
``callbacks`` 逗号列表选择；:class:`CallbackList` 负责分发与隔离异常。

events.jsonl schema（B7/N05 结构化阶段事件）
-------------------------------------------
每行一个 JSON 对象，公共字段：``v`` (=1)、``ts`` (ISO-8601)、``run_id``、
``event``、``step`` (int)、``payload`` (object)。事件类型：

* ``run_start``    payload: {stage, max_steps, seed, data_dir}
* ``step``         payload: {lr, train_loss, grad_norm, tokens_per_sec, elapsed}
* ``eval``         payload: {val_loss, train_loss, best_val, lr, grad_norm,
  tokens_per_sec, elapsed}（旧事件，保留）
* ``eval_start``   payload: {split, num_batches}；每次 evaluate 开始时发出
* ``eval_end``     payload: {split, duration_s, val_loss}；evaluate 结束（含
  异常）时发出，``duration_s`` 为耗时秒数，失败时 ``val_loss`` 为 null
* ``checkpoint``   payload: {kind, path}（旧事件，保留）
* ``checkpoint_saved`` payload: {kind, path}；kind ∈ {best, last}，与旧
  ``checkpoint`` 事件同时发出
* ``early_stop``   payload: {best_val, patience}（旧事件，训练结束时发出）
* ``early_stop_triggered`` payload: {best_val, patience, step}；早停判定点发出
* ``run_resumed``  payload: {checkpoint, start_step, mode}；续训成功后在
  train() 开始时发出，mode ∈ {True, "auto"}
* ``resumed``      payload: {source: "job_manager", pid}；看板 resume_job
  重新拉起训练进程时由 job_manager 追加
* ``eval_nonfinite_batches`` payload: {split, count}
* ``nan_window_discarded``   payload: {step}
* ``sample``       payload: {prompt, text}
* ``run_end``      payload: {status, best_val, stopped_early}
* ``error``        payload: {type, message}
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minigpt import registry
from minigpt.utils import atomic_write_json, ensure_dir, utc_now


@dataclass
class TrainingState:
    """训练过程中在 Trainer 与 callback 之间共享的可变状态快照。"""
    run_id: str
    stage: str
    step: int
    max_steps: int
    run_dir: Path
    config: Any
    lr: float = 0.0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    best_val: float = float("inf")
    grad_norm: float = float("nan")
    tokens_per_sec: float = 0.0
    elapsed: float = 0.0
    initial_loss: float = float("nan")
    model: Any = None
    optimizer: Any = None
    backend: Any = None
    tokenizer: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class CallbackBase:
    """No-op base class; override only the hooks you need."""

    def __init__(self, config: Any, run_dir: Path):
        """保存配置与 run 目录，供子类使用。"""
        self.config = config
        self.run_dir = Path(run_dir)

    def on_train_start(self, state: TrainingState) -> None: ...

    def on_step_end(self, state: TrainingState) -> None: ...

    def on_eval_end(self, state: TrainingState) -> None: ...

    def on_checkpoint(self, state: TrainingState, kind: str, path: str) -> None: ...

    def on_train_end(self, state: TrainingState) -> None: ...

    def on_error(self, state: TrainingState, error: BaseException) -> None: ...

    def on_event(self, state: TrainingState, event: str, payload: dict[str, Any] | None = None) -> None: ...

    def close(self) -> None: ...


HEARTBEAT_INTERVAL_S = 1.0


class TelemetryCallback(CallbackBase):
    """Write append-only events plus a small status.json for the dashboard."""

    def __init__(self, config: Any, run_dir: Path):
        """准备 events.jsonl / status.json 的写入器。"""
        super().__init__(config, run_dir)
        ensure_dir(self.run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self.status_path = self.run_dir / "status.json"
        self._status: dict[str, Any] = {
            "run_id": self.run_dir.name,
            "stage": getattr(config, "stage", "pretrain"),
            "status": "starting",
            "step": 0,
            "max_steps": getattr(config, "max_steps", 0),
            "best_val": None,
            "checkpoint": "",
            "pid": os.getpid(),
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "error": "",
        }
        self._last_heartbeat = time.monotonic()

    def write_event(self, event: str, step: int, payload: dict[str, Any] | None = None) -> None:
        """追加一条事件（带 schema 版本与时间戳）。"""
        record = {
            "v": 1,
            "ts": utc_now(),
            "run_id": self._status.get("run_id", self.run_dir.name),
            "event": event,
            "step": int(step),
            "payload": payload or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def update_status(self, **values: Any) -> None:
        """把当前状态原子写入 status.json。"""
        self._status.update(values)
        self._status["updated_at"] = utc_now()
        atomic_write_json(self.status_path, self._status)

    def on_train_start(self, state: TrainingState) -> None:
        """写入运行开始事件与初始状态。"""
        self._status.update(
            {
                "run_id": state.run_id,
                "stage": state.stage,
                "status": "running",
                "step": state.step,
                "max_steps": state.max_steps,
                "started_at": utc_now(),
                "pid": os.getpid(),
            }
        )
        self.write_event(
            "run_start",
            state.step,
            {
                "stage": state.stage,
                "max_steps": state.max_steps,
                "seed": getattr(state.config, "seed", None),
                "data_dir": getattr(state.config, "data_dir", ""),
            },
        )
        self.update_status()

    def on_step_end(self, state: TrainingState) -> None:
        # 心跳每步刷新（限流每秒一次）：慢模型（单步 >60s）不再被误标
        # 「心跳过期」；日志/指标事件仍按 log_interval 批量输出。
        """写入 step 事件（loss/lr/梯度/tok-s）。"""
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL_S:
            self._last_heartbeat = now
            self.update_status(step=state.step)
        log_interval = max(1, int(getattr(self.config, "log_interval", 1)))
        if state.step % log_interval == 0:
            self._last_heartbeat = time.monotonic()
            payload = {
                "lr": state.lr,
                "train_loss": state.train_loss,
                "grad_norm": state.grad_norm,
                "tokens_per_sec": state.tokens_per_sec,
                "elapsed": state.elapsed,
            }
            self.write_event("step", state.step, payload)
            self.update_status(
                step=state.step,
                lr=state.lr,
                train_loss=state.train_loss,
                grad_norm=state.grad_norm,
                tokens_per_sec=state.tokens_per_sec,
                elapsed=state.elapsed,
            )

    def on_eval_end(self, state: TrainingState) -> None:
        """写入 eval 事件与最佳指标。"""
        self.write_event(
            "eval",
            state.step,
            {
                "val_loss": state.val_loss,
                "train_loss": state.train_loss,
                "best_val": state.best_val,
                "lr": state.lr,
                "grad_norm": state.grad_norm,
                "tokens_per_sec": state.tokens_per_sec,
                "elapsed": state.elapsed,
            },
        )
        self.update_status(
            step=state.step,
            val_loss=state.val_loss,
            best_val=state.best_val,
            lr=state.lr,
        )

    def on_checkpoint(self, state: TrainingState, kind: str, path: str) -> None:
        """写入 checkpoint 事件。"""
        self.write_event("checkpoint", state.step, {"kind": kind, "path": path})
        self.update_status(checkpoint=path, best_val=state.best_val)

    def on_train_end(self, state: TrainingState) -> None:
        """写入完成/停止状态与最终报告。"""
        status = state.extra.get("status", "completed")
        if state.extra.get("stopped_early"):
            self.write_event(
                "early_stop",
                state.step,
                {
                    "best_val": state.best_val,
                    "patience": getattr(self.config, "patience", None),
                },
            )
        self.write_event(
            "run_end",
            state.step,
            {
                "status": status,
                "best_val": state.best_val,
                "stopped_early": state.extra.get("stopped_early", False),
            },
        )
        self.update_status(status=status, step=state.step, best_val=state.best_val)

    def on_error(self, state: TrainingState, error: BaseException) -> None:
        """写入失败状态与错误信息。"""
        self.write_event("error", state.step, {"type": type(error).__name__, "message": str(error)})
        self.update_status(status="failed", error=f"{type(error).__name__}: {error}")

    def on_event(self, state: TrainingState, event: str, payload: dict[str, Any] | None = None) -> None:
        """转发自定义事件。"""
        self.write_event(event, state.step, payload)

    def close(self) -> None:
        """刷新并关闭文件句柄。"""
        pass


class SampleLoggerCallback(CallbackBase):
    """Generate a short sample at each evaluation and append it to samples.jsonl."""

    def __init__(self, config: Any, run_dir: Path):
        """准备 samples.jsonl 写入器。"""
        super().__init__(config, run_dir)
        self.path = Path(run_dir) / "samples.jsonl"
        self.events_path = Path(run_dir) / "events.jsonl"

    def on_eval_end(self, state: TrainingState) -> None:
        """按配置采样并记录模型输出样本。"""
        prompt = getattr(self.config, "sample_prompt", "")
        if not prompt or state.model is None or state.backend is None or state.tokenizer is None:
            return
        # pure 后端逐 token 全窗口重算（约 1-2s/token），80 token 会让每次评估
        # 额外耗时数分钟；按后端自适应缩减样本长度，torch 保持 80 不变。
        cap = 16 if getattr(state.backend, "name", "") == "pure" else 80
        max_new = min(int(getattr(self.config, "max_new_tokens", cap)), cap)
        if max_new <= 0:
            return
        prompt_ids = state.tokenizer.encode(prompt)
        eos_id = getattr(state.tokenizer, "eos_id", None)
        generated = state.backend.generate(
            state.model,
            prompt_ids,
            max_new_tokens=max_new,
            temperature=float(getattr(self.config, "temperature", 0.8)),
            top_k=int(getattr(self.config, "top_k", 40)),
            top_p=float(getattr(self.config, "top_p", 1.0)),
            eos_id=eos_id,
        )
        text = state.tokenizer.decode(generated)
        record = {
            "v": 1,
            "ts": utc_now(),
            "run_id": state.run_id,
            "step": state.step,
            "prompt": prompt,
            "text": text,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        event = {
            "v": 1,
            "ts": utc_now(),
            "run_id": state.run_id,
            "event": "sample",
            "step": state.step,
            "payload": {"prompt": prompt, "text": text},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()


class CallbackList:
    def __init__(self, callbacks: list[Any]):
        """保存回调列表。"""
        self.callbacks = callbacks

    @classmethod
    def from_config(cls, config: Any, run_dir: Path) -> "CallbackList":
        """按 callbacks 配置构建回调组合。"""
        names = [name.strip() for name in str(getattr(config, "callbacks", "")).split(",") if name.strip()]
        callbacks = []
        for name in names:
            callback = registry.CALLBACKS.create(name, config=config, run_dir=Path(run_dir))
            callbacks.append(callback)
        return cls(callbacks)

    def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> None:
        """向所有回调分发同一个钩子。"""
        for callback in self.callbacks:
            function = getattr(callback, method, None)
            if function is None:
                continue
            try:
                function(*args, **kwargs)
            except Exception:
                print(f"[callback] {type(callback).__name__}.{method} failed:")
                traceback.print_exc()

    def on_train_start(self, state: TrainingState) -> None:
        """分发训练开始钩子。"""
        self._dispatch("on_train_start", state)

    def on_step_end(self, state: TrainingState) -> None:
        """分发步结束钩子。"""
        self._dispatch("on_step_end", state)

    def on_eval_end(self, state: TrainingState) -> None:
        """分发评估结束钩子。"""
        self._dispatch("on_eval_end", state)

    def on_checkpoint(self, state: TrainingState, kind: str, path: str) -> None:
        """分发 checkpoint 钩子。"""
        self._dispatch("on_checkpoint", state, kind, path)

    def on_train_end(self, state: TrainingState) -> None:
        """分发训练结束钩子。"""
        self._dispatch("on_train_end", state)

    def on_error(self, state: TrainingState, error: BaseException) -> None:
        """分发异常钩子。"""
        self._dispatch("on_error", state, error)

    def on_event(self, state: TrainingState, event: str, payload: dict[str, Any] | None = None) -> None:
        """分发自定义事件钩子。"""
        self._dispatch("on_event", state, event, payload)

    def close(self) -> None:
        """关闭全部回调。"""
        for callback in self.callbacks:
            try:
                callback.close()
            except Exception:
                pass


registry.CALLBACKS.register("telemetry")(TelemetryCallback)
registry.CALLBACKS.register("sample")(SampleLoggerCallback)

__all__ = [
    "TrainingState",
    "CallbackBase",
    "TelemetryCallback",
    "SampleLoggerCallback",
    "CallbackList",
]
