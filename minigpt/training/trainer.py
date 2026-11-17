"""与框架无关的训练循环。

训练器只依赖 :class:`ModelBackend`（torch 或 pure 后端）与数据源，独占了
「实验协议」：学习率调度、梯度累积、验证、best/last checkpoint、早停、metrics
CSV、events、status 与训练报告。

这里是**唯一**决定「何时评估/保存/停止」的地方；将来把 PyTorch 换成纯 Python
后端也不会复制出一份协议逻辑。

调用时序（单个训练进程内）::

    Trainer(...)            校验配置/运行目录 → 载入并校验 checkpoint（续训）
                            → 写 run_config.json/environment.json/status.json
    train()                 on_train_start → 循环[前向/反向/评估/保存]
                            → 写 training_report.json + 终态 status
    finally                 callbacks.close()

状态迁移：``starting`` → ``running`` →（``completed`` | ``stopped`` | ``failed``）。
文件写入时机：每一步只刷新 status 心跳；评估步写 metrics.csv；保存步写
``best.<suffix>`` 或 ``last.<suffix>``；结束时写 ``training_report.json``。
失败/中断时报告与 status 仍会落盘，磁盘上的 last checkpoint 保持可续训。
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from minigpt.training.callbacks import CallbackList, TelemetryCallback, TrainingState
from minigpt.training.checkpoint import CHECKPOINT_VERSION, validate_checkpoint
from minigpt.training.early_stop import EarlyStopping
from minigpt.utils import (
    atomic_write_json,
    ensure_dir,
    fmt_seconds,
    generate_run_id,
    seed_everything,
    utc_now,
)

METRIC_FIELDS = [
    "step",
    "train_loss",
    "val_loss",
    "val_ppl",
    "val_bpc",
    "lr",
    "grad_norm",
    "tokens_per_sec",
    "elapsed_s",
    "best_val",
]

# 生命周期契约：只有「真实训练产物」才代表一个已有运行。
# 看板 create_run 预写的 resolved_config.json / status.json / console.log
# 属于草稿启动文件，不计入；否则看板启动的 run 会被自己的预写文件拒跑。
TRAINING_ARTIFACTS = (
    "metrics.csv",
    "samples.jsonl",
    "events.jsonl",
    "training_report.json",
    "run_config.json",
    "environment.json",
    "last.pt",
    "best.pt",
    "last.mgckpt",
    "best.mgckpt",
)


def cosine_lr(step: int, max_steps: int, warmup: int, lr: float, min_lr: float) -> float:
    """线性 warmup 之后按余弦曲线衰减到 ``min_lr``。

    ``step`` 为 0-based（即将执行的那一步）。公式：warmup 段 ``lr*(step+1)/warmup``；
    之后 ``min_lr + 0.5*(1+cos(pi*ratio))*(lr-min_lr)``，``ratio`` 是「warmup 之后
    已走的比例」并被夹到 [0, 1]。超过调度视野（或 ``max_steps<=warmup``）一律
    返回 ``min_lr``。
    """

    if warmup > 0 and step < warmup:
        return lr * float(step + 1) / float(warmup)
    if max_steps <= warmup:
        return min_lr
    ratio = (step - warmup) / float(max_steps - warmup)
    ratio = min(1.0, max(0.0, ratio))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (lr - min_lr)


def _warmup_lr(step: int, warmup: int, lr: float) -> float | None:
    """warmup 段内的学习率；不在 warmup 段返回 None。"""

    if warmup > 0 and step < warmup:
        return lr * float(step + 1) / float(warmup)
    return None


def wsd_lr(
    step: int,
    max_steps: int,
    warmup: int,
    lr: float,
    min_lr: float,
    decay_fraction: float = 0.2,
) -> float:
    """Warmup-Stable-Decay：warmup 后恒定 ``lr``，最后 ``decay_fraction`` 比例
    （相对 warmup 之后的步数）线性降到 ``min_lr``。

    ``step`` 为 0-based；超过 horizon 返回 ``min_lr``。``decay_fraction=0``
    表示没有衰减段（warmup 后一直恒定）。``decay_fraction`` 会被夹到 [0, 1]；
    若衰减起点不早于 ``max_steps``（例如 warmup 已占满全程）则保持恒定 ``lr``。
    """

    warm = _warmup_lr(step, warmup, lr)
    if warm is not None:
        return warm
    if max_steps <= warmup:
        return min_lr
    decay_fraction = min(1.0, max(0.0, float(decay_fraction)))
    decay_start = warmup + (1.0 - decay_fraction) * (max_steps - warmup)
    if step < decay_start or decay_start >= max_steps:
        return lr
    ratio = (step - decay_start) / float(max_steps - decay_start)
    ratio = min(1.0, max(0.0, ratio))
    return lr + ratio * (min_lr - lr)


def linear_lr(step: int, max_steps: int, warmup: int, lr: float, min_lr: float) -> float:
    """线性 warmup 之后线性衰减，在 ``max_steps`` 处降到 ``min_lr``。

    ``ratio`` 同样夹到 [0, 1]，因此超出 horizon 后稳定返回 ``min_lr``。
    """

    warm = _warmup_lr(step, warmup, lr)
    if warm is not None:
        return warm
    if max_steps <= warmup:
        return min_lr
    ratio = (step - warmup) / float(max_steps - warmup)
    ratio = min(1.0, max(0.0, ratio))
    return lr + ratio * (min_lr - lr)


# 调度名 → 函数。配置项 lr_schedule 的合法取值即这里的键（另有 wsd 的特例分支）。
LR_SCHEDULES = {"cosine": cosine_lr, "wsd": wsd_lr, "linear": linear_lr}


def lr_for_step(config: Any, step: int) -> float:
    """按 ``config.lr_schedule`` 选择调度函数（默认 cosine，旧行为不变）。

    ``wsd`` 单独分支是因为它多一个 ``decay_fraction`` 参数；其余调度走
    :data:`LR_SCHEDULES` 表。未知调度名抛 :class:`ValueError`（配置校验阶段
    已经拦过一遍，这里是运行时兜底）。
    """

    name = str(getattr(config, "lr_schedule", "cosine") or "cosine")
    if name == "wsd":
        return wsd_lr(
            step,
            config.max_steps,
            config.warmup_steps,
            config.lr,
            config.min_lr,
            decay_fraction=float(getattr(config, "decay_fraction", 0.2)),
        )
    schedule = LR_SCHEDULES.get(name)
    if schedule is None:
        raise ValueError(f"unknown lr_schedule {name!r}; expected cosine/wsd/linear")
    return schedule(step, config.max_steps, config.warmup_steps, config.lr, config.min_lr)


def check_mini_run(
    result: dict[str, Any],
    vocab_size: int,
    loss_tolerance: float = 0.5,
    min_drop: float = 0.2,
) -> dict[str, Any]:
    """CONSTRUCTION.md 里的 mini-run 三项门禁，返回逐项判定报告。

    三个判据（全部为真才算通过）：

    1. **初始 loss** 接近理论值 ``ln(vocab_size)``（均匀分布交叉熵），
       允许 ``loss_tolerance`` 的绝对误差——用来抓「权重初始化/标签错位」；
    2. **梯度范数** 至少有一个有限值、全部非负且存在正值——全 0 说明图断开；
    3. **loss 下降** ``initial - final_train_loss >= min_drop``。

    参数 ``result`` 是训练器 ``train()`` 返回的结果字典（读取
    ``initial_loss`` / ``grad_norms`` / ``final_train_loss``）。返回值只用于
    报告，不参与训练决策。
    """

    expected = math.log(max(2, vocab_size))
    initial = float(result.get("initial_loss", float("nan")))
    initial_ok = math.isfinite(initial) and abs(initial - expected) <= loss_tolerance
    # 只统计有限值：NaN/Inf 梯度说明本步已失败，不计入门禁。
    grads = [g for g in result.get("grad_norms", []) if math.isfinite(g)]
    grad_ok = bool(grads) and all(g >= 0 for g in grads) and any(g > 0 for g in grads)
    final_train = float(result.get("final_train_loss", float("nan")))
    # 任一值为 NaN 时 drop 也置 NaN，由下面 isfinite 统一判失败。
    drop = initial - final_train if math.isfinite(initial) and math.isfinite(final_train) else float("nan")
    drop_ok = math.isfinite(drop) and drop >= min_drop
    report = {
        "passed": bool(initial_ok and grad_ok and drop_ok),
        "vocab_size": vocab_size,
        "expected_initial_loss": expected,
        "initial_loss": initial,
        "initial_loss_abs_diff": abs(initial - expected) if math.isfinite(initial) else None,
        "initial_loss_ok": initial_ok,
        "grad_norms_count": len(grads),
        "grad_norm_min": min(grads) if grads else None,
        "grad_norm_max": max(grads) if grads else None,
        "gradient_ok": grad_ok,
        "final_train_loss": final_train,
        "loss_drop": drop,
        "loss_drop_ok": drop_ok,
        "loss_tolerance": loss_tolerance,
        "min_drop": min_drop,
    }
    return report


def _rng_tuple(value: Any) -> Any:
    """把 JSON 解码后的 Python RNG 状态（list）递归还原成 tuple。

    ``random.setstate`` 只接受 tuple 结构（内部元素也是 tuple），而 JSON 只有
    数组，因此恢复前必须转换。
    """

    if isinstance(value, list):
        return tuple(_rng_tuple(item) for item in value)
    return value


def _fingerprint_one(meta: Any, suffix: str) -> dict[str, str]:
    """从单个数据源 metadata 提取数据文件 sha256（不记录绝对路径）。"""
    if not isinstance(meta, dict):
        return {}
    result: dict[str, str] = {}
    splits = meta.get("splits")
    if isinstance(splits, dict):
        for split, info in splits.items():
            if isinstance(info, dict) and isinstance(info.get("sha256"), str) and info["sha256"]:
                result[f"{split}{suffix}"] = info["sha256"]
    source_sha = meta.get("source_sha256")
    if isinstance(source_sha, str) and source_sha:
        result["source"] = source_sha
    source_info = meta.get("source_info")
    if isinstance(source_info, dict) and isinstance(source_info.get("sha256"), str) and source_info["sha256"]:
        result["source_file"] = source_info["sha256"]
    return result


def _data_fingerprint(meta: Any) -> dict[str, str]:
    """数据文件 sha256 指纹；metadata 无哈希信息时返回空 dict。

    兼容 MixedSource 的 ``{"pretrain": ..., "sft": ...}`` 嵌套结构：
    预训练切分记为 ``pretrain/<split>.bin``，SFT 切分记为 ``sft/<split>.jsonl``。
    """
    if not isinstance(meta, dict):
        return {}
    if isinstance(meta.get("sft"), dict) or isinstance(meta.get("pretrain"), dict):
        result: dict[str, str] = {}
        for key, digest in _fingerprint_one(meta.get("pretrain"), ".bin").items():
            result[f"pretrain/{key}"] = digest
        for key, digest in _fingerprint_one(meta.get("sft"), ".jsonl").items():
            result[f"sft/{key}"] = digest
        return result
    stage = meta.get("stage")
    suffix = ".jsonl" if stage == "sft" else ".bin"
    return _fingerprint_one(meta, suffix)


def _capability_boundary(stage: str) -> str:
    """与训练阶段对应的能力边界说明（写入报告，供模型卡引用）。"""
    if stage == "sft":
        return (
            "SFT（监督微调）阶段：在预训练权重基础上，用问答格式数据教模型遵循 "
            "Q:/A: 对话模板。字符级小模型只学习局部字符结构与输出格式，"
            "不保证回答语义正确，可能输出空回答或重复文本；不是通用助手，"
            "不应作为事实知识库或生产服务。"
        )
    return (
        "预训练阶段：模型在原始语料上学习字符级局部统计结构（续写能力），"
        "尚未学习问答格式，不能直接作为对话助手使用；生成内容是语法上可能像、"
        "语义上不可靠的文本，不应作为事实知识库或生产服务。"
    )


class Trainer:
    """训练协议的执行者：调度 LR、累积梯度、评估、保存、早停、写报告。

    构造即产生「运行」：校验运行目录（非续训时拒绝覆盖已有产物）、可选地
    载入 checkpoint、写 ``run_config.json`` / ``environment.json`` /
    ``status.json``。此后调用 :meth:`train` 执行主循环，结束后由
    ``try/finally`` 关闭 callbacks。

    线程/进程语义：对象属于单个训练子进程，不加锁；并发安全靠「看板只允许
    一个活动任务 + 运行目录的唯一 run_id」保证。副作用集中在 ``self.run_dir``。
    """

    def __init__(
        self,
        config: Any,
        backend: Any,
        model: Any,
        optimizer: Any,
        tokenizer: Any,
        data_source: Any,
        callbacks: CallbackList | None = None,
        resume: bool | str = False,
    ):
        """组装训练器并写入运行元数据。

        参数 ``backend`` 提供设备/保存/随机数等后端能力（torch 或 pure）；
        ``callbacks`` 为空时按 ``config.callbacks`` 从 registry 构建；
        ``resume`` 可为 ``True`` / ``"auto"`` / ``False``（见
        ``minigpt.config.normalize_resume``）。

        异常：``block_size`` 超出模型能力、非续训但运行目录已有训练产物、
        续训 checkpoint 校验失败时抛 :class:`ValueError` /
        :class:`FileNotFoundError`；此时**不会**改写已有 run 元数据。
        """

        from minigpt.config import normalize_resume, resume_enabled

        resume = normalize_resume(resume)
        self.resume_mode = resume
        resume_on = resume_enabled(resume)
        self.config = config
        self.backend = backend
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.data_source = data_source

        if getattr(model, "config", None) is not None and getattr(model.config, "block_size", 0) < config.block_size:
            raise ValueError(
                f"config.block_size={config.block_size} exceeds model block_size={model.config.block_size}"
            )

        if not config.run_name:
            if resume_on:
                raise ValueError("--resume requires --run_name (the existing run directory)")
            config.run_name = generate_run_id(config.stage, config.seed)
        self.run_id = config.run_name
        run_dir = Path(config.runs_dir) / self.run_id
        existing = [name for name in TRAINING_ARTIFACTS if (run_dir / name).exists()]
        if not resume_on and existing:
            raise ValueError(
                f"run directory already holds training artifacts ({', '.join(existing)}): {run_dir}. "
                "Use --resume to continue it, or choose a different --run_name "
                "(refusing to silently overwrite a previous run)."
            )
        self.run_dir = ensure_dir(run_dir)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.status_path = self.run_dir / "status.json"
        self.early = EarlyStopping(config.patience, config.min_delta)
        self.start_iter = 0
        self.best_val = float("inf")
        self.tokens_seen = 0
        self._resume_info: dict[str, Any] | None = None
        self._status: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": config.stage,
            "status": "starting",
            "step": 0,
            "max_steps": config.max_steps,
            "best_val": None,
            "checkpoint": "",
            "pid": os.getpid(),
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "error": "",
        }

        # callbacks 必须在第一次写 status 之前建好：telemetry 是 status.json 的
        # 唯一写者（见 _write_status）。
        if callbacks is None:
            callbacks = CallbackList.from_config(config, self.run_dir)
        self.callbacks = callbacks
        self._telemetry = next(
            (cb for cb in self.callbacks.callbacks if isinstance(cb, TelemetryCallback)), None
        )

        # 先校验/载入 checkpoint，再重写运行元数据：续训失败不能把原始的
        # run_config/environment 覆盖掉。
        if resume_on:
            self._load_for_resume()

        self._write_run_metadata()
        self._write_status()

        self.state = TrainingState(
            run_id=self.run_id,
            stage=config.stage,
            step=self.start_iter,
            max_steps=config.max_steps,
            run_dir=self.run_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            backend=backend,
            tokenizer=tokenizer,
            best_val=self.best_val,
        )

    # ------------------------------------------------------------------
    def _write_run_metadata(self) -> None:
        """写 ``run_config.json``（可移植配置 + argv）与 ``environment.json``。

        两个文件都用原子写；``environment.json`` 合并后端 ``device_info()``
        （torch/CUDA 版本等）与解释器、平台、项目根，供复现与报告引用。
        """

        portable = self.config.to_dict(portable=True)
        atomic_write_json(
            self.run_dir / "run_config.json",
            {
                "run_id": self.run_id,
                "created_at": utc_now(),
                "argv": sys.argv,
                "config": portable,
            },
        )
        info = self.backend.device_info()
        info.update(
            {
                "run_id": self.run_id,
                "python": sys.version,
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "project_root": self.config.project_root,
                "created_at": utc_now(),
                "parameters": self.model.num_parameters() if hasattr(self.model, "num_parameters") else None,
            }
        )
        atomic_write_json(self.run_dir / "environment.json", info)

    def _write_status(self, **values: Any) -> None:
        """合并字段后刷新 ``status.json``（telemetry 存在时由它单写）。

        参数按关键字传入要更新的字段；``updated_at`` 由实际写盘者补。
        """

        # status.json 的唯一写入口：telemetry 存在时由它独占文件并合并每次更新，
        # 训练器与 callback 的写入因此不会互相覆盖字段；没有 telemetry 时训练器
        # 自己维护完整的 status 字典。
        self._status.update(values)
        if self._telemetry is not None:
            self._telemetry.update_status(**self._status)
        else:
            self._status["updated_at"] = utc_now()
            atomic_write_json(self.status_path, self._status)

    def _heartbeat(self, step: int) -> None:
        """每步刷新心跳（限流每秒一次），避免慢模型被误标「心跳过期」。

        有 telemetry 时由其 on_step_end 负责心跳，这里不重复写盘。
        """

        if self._telemetry is not None:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_heartbeat", 0.0) < 1.0:
            return
        self._last_heartbeat = now
        self._write_status(step=step)

    def _write_metrics(self, row: dict[str, Any]) -> None:
        """向 ``metrics.csv`` 追加一行；文件不存在时先写表头。

        只在评估步调用，列顺序固定为 :data:`METRIC_FIELDS`。
        """

        is_new = not self.metrics_path.exists()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    def _resume_checkpoint_path(self) -> Path:
        """续训读取的 checkpoint 路径：``last<suffix>``（torch 为 .pt，pure 为 .mgckpt）。"""

        return self.run_dir / f"last{self.backend.checkpoint_suffix}"

    def _load_for_resume(self) -> None:
        """载入 ``last`` checkpoint 并恢复模型/优化器/早停/随机数状态。

        ``resume_mode == "auto"`` 时 checkpoint 必须存在，否则抛
        :class:`FileNotFoundError`。stage 与 tokenizer 状态必须与当前配置一致，
        防止「用 A 数据的分词器续训 B 数据」这种静默错误。

        副作用：原地修改 ``self.start_iter`` / ``self.best_val`` /
        ``self.early`` / ``self._resume_info``，并把权重写回 model/optimizer。
        """

        path = self._resume_checkpoint_path()
        if self.resume_mode == "auto" and not path.is_file():
            raise FileNotFoundError(
                f"--resume auto: no checkpoint found in run directory {self.run_dir} "
                f"(expected {path.name}); nothing to resume"
            )
        # 检查点格式与随机数恢复都下沉到后端：训练器不再按 backend.name 分支。
        payload = self.backend.load_checkpoint(path)
        validate_checkpoint(payload)
        if payload.get("stage") != self.config.stage:
            raise ValueError(f"checkpoint stage={payload.get('stage')!r}, expected {self.config.stage!r}")
        if payload.get("tokenizer_state") != self.tokenizer.state_dict():
            raise ValueError("checkpoint tokenizer does not match the current data tokenizer")
        self.backend.load_state_dict(self.model, payload["model"])
        self.backend.load_optimizer_state_dict(self.optimizer, payload["optimizer"])
        self.start_iter = int(payload["iter_num"])
        self.best_val = float(payload["best_val_loss"])
        if payload.get("early_stopping"):
            self.early.load_state_dict(payload["early_stopping"])
        self.backend.restore_rng_state(payload.get("rng"))
        backend_rng = getattr(self.backend, "rng", None)
        backend_state = (payload.get("rng") or {}).get("backend")
        if backend_rng is not None and backend_state:
            backend_rng.setstate(_rng_tuple(backend_state))
        scaler = getattr(self.backend, "scaler", None)
        if scaler is not None and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        print(f"[resume] {path} at step {self.start_iter}, best_val={self.best_val:.4f}")
        self._resume_info = {"checkpoint": str(path), "start_step": self.start_iter, "mode": self.resume_mode}

    def _checkpoint_payload(self, iter_num: int) -> dict[str, Any]:
        """组装完整 checkpoint payload（模型/优化器/RNG/tokenizer/早停/元数据）。"""

        model_config = self.model.config.to_dict() if hasattr(self.model, "config") else {}
        data_meta = getattr(self.data_source, "metadata", {}) or {}
        info = self.backend.device_info()
        rng_state = self.backend.capture_rng_state()
        backend_rng = getattr(self.backend, "rng", None)
        if backend_rng is not None:
            rng_state["backend"] = backend_rng.getstate()
        scaler = getattr(self.backend, "scaler", None)
        scaler_state = scaler.state_dict() if scaler is not None else None
        return {
            "format_version": CHECKPOINT_VERSION,
            "backend": self.backend.name,
            "backend_version": self.backend.version,
            "stage": self.config.stage,
            "model": self.backend.state_dict(self.model),
            "optimizer": self.backend.optimizer_state_dict(self.optimizer),
            "iter_num": int(iter_num),
            "best_val_loss": float(self.best_val),
            "config": self.config.to_dict(portable=True),
            "model_config": model_config,
            "tokenizer_state": self.tokenizer.state_dict(),
            "data_meta": data_meta,
            "rng": rng_state,
            "scaler": scaler_state,
            "early_stopping": self.early.state_dict(),
            "python_version": platform.python_version(),
            "torch_version": info.get("torch", ""),
            "created_at": utc_now(),
            "run_id": self.run_id,
        }

    def save(self, kind: str, iter_num: int) -> str:
        """保存 ``best`` / ``last`` checkpoint，并刷新 status 与事件。

        参数 ``kind`` 决定文件名（``<kind><suffix>``），``iter_num`` 写入
        payload 的 ``iter_num``。返回保存后的绝对路径字符串。走
        ``backend.save_checkpoint``，即后端的原子写实现。
        """

        path = self.run_dir / f"{kind}{self.backend.checkpoint_suffix}"
        self.backend.save_checkpoint(path, self._checkpoint_payload(iter_num))
        self._write_status(checkpoint=str(path), best_val=self.best_val)
        self.callbacks.on_checkpoint(self.state, kind, str(path))
        self.callbacks.on_event(self.state, "checkpoint_saved", {"kind": kind, "path": str(path)})
        return str(path)

    # ------------------------------------------------------------------
    def _set_lr(self, lr: float) -> None:
        """把学习率写入优化器（torch 的 param_groups 或 pure 的单一 ``lr``）。"""

        if hasattr(self.optimizer, "param_groups"):
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        else:
            # pure 后端的 AdamW 只有单一学习率字段。
            self.optimizer.lr = lr

    def evaluate(self, split: str, num_batches: int) -> float:
        """在 ``split`` 上跑 ``num_batches`` 个批次，返回**按 token 加权**的平均 loss。

        非有限 loss 的批次会被丢弃（计数并记 ``eval_nonfinite_batches`` 事件），
        全部批次都无效时抛 :class:`RuntimeError`。无论成功失败都会发
        ``eval_start`` / ``eval_end`` 事件（``eval_end`` 里带耗时与 val_loss）。
        """

        self.callbacks.on_event(self.state, "eval_start", {"split": split, "num_batches": int(num_batches)})
        started = time.monotonic()
        val_loss: float | None = None
        try:
            batches = self.data_source.eval_batches(
                split=split,
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                num_batches=num_batches,
                seed=self.config.eval_seed,
            )
            total_loss = 0.0
            total_tokens = 0
            skipped = 0
            for batch in batches:
                result = self.backend.eval_step(self.model, batch)
                if not math.isfinite(result.loss):
                    skipped += 1
                    continue
                total_loss += result.loss * result.tokens
                total_tokens += result.tokens
            if total_tokens == 0:
                raise RuntimeError(f"no validation tokens for split {split!r}")
            if skipped:
                print(f"[warn] eval split={split}: skipped {skipped} non-finite loss batch(es)")
                self.callbacks.on_event(self.state, "eval_nonfinite_batches", {"split": split, "count": skipped})
            val_loss = total_loss / total_tokens
            return val_loss
        finally:
            self.callbacks.on_event(
                self.state,
                "eval_end",
                {"split": split, "duration_s": time.monotonic() - started, "val_loss": val_loss},
            )

    def _quick_loss(self, split: str) -> float:
        """单批次快速 loss（用于训练前的初始 loss 冒烟检查）。"""

        return self.evaluate(split, 1)

    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        """执行主训练循环，返回结果字典（status/best_val/steps/elapsed_s 等）。

        循环体顺序固定：算 LR → 取 K 个 micro-batch → 反向累积 → 优化器步 →
        更新状态/心跳 → 按 ``log_interval`` 打印 → 按 ``eval_interval`` 评估、
        早停判定、写 metrics → 按 ``save_interval`` 存 ``last``。

        ``KeyboardInterrupt`` 视为正常停止（保留磁盘上的 last checkpoint）；
        其它异常写 ``failed`` 报告与状态后原样抛出。``finally`` 总会
        ``callbacks.close()``。
        """

        cfg = self.config
        state = self.state
        self.callbacks.on_train_start(state)
        if self._resume_info is not None:
            self.callbacks.on_event(state, "run_resumed", self._resume_info)
        self._write_status(status="running", step=self.start_iter)
        started = time.monotonic()
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": "running",
            "initial_loss": float("nan"),
            "final_train_loss": float("nan"),
            "best_val": float("inf"),
            "best_step": -1,
            "stopped_early": False,
            "steps": 0,
            # 有界缓冲：长训练不再无限累积梯度/损失。门禁与末段均值只需最近
            # 窗口，2048 远大于 mini-run 步数，统计语义不变。
            "grad_norms": deque(maxlen=2048),
            "elapsed_s": 0.0,
        }
        try:
            initial_loss = self._quick_loss("train")
            result["initial_loss"] = initial_loss
            state.initial_loss = initial_loss
            expected = math.log(max(2, self.tokenizer.vocab_size))
            print(
                f"[sanity] vocab={self.tokenizer.vocab_size} "
                f"initial_loss={initial_loss:.4f} expected~{expected:.4f}"
            )
            params = self.model.num_parameters() if hasattr(self.model, "num_parameters") else "?"
            print(
                f"[run] {self.run_id} stage={cfg.stage} params={params} "
                f"device={getattr(self.backend, 'device', '?')} max_steps={cfg.max_steps}"
            )

            train_losses: deque[float] = deque(maxlen=2048)
            step_index = self.start_iter
            last_completed = self.start_iter
            while step_index < cfg.max_steps:
                lr = lr_for_step(cfg, step_index)
                self._set_lr(lr)

                # 按 token 加权累积：先取完整窗口（K 很小，1-8），每个 micro-batch
                # 的 backward 权重为 tokens_i / total_tokens。SFT 各 micro-batch 的
                # 有效 token 数不等，固定 1/K 会让它们被高估或低估。
                window = [
                    self.data_source.train_batch(
                        batch_size=cfg.batch_size,
                        block_size=cfg.block_size,
                        rng=None,
                    )
                    for _ in range(cfg.grad_accum_steps)
                ]
                window_tokens = [
                    sum(1 for row in batch["y"] for value in row if int(value) != -100)
                    for batch in window
                ]
                total_window_tokens = sum(window_tokens)
                if total_window_tokens == 0:
                    raise RuntimeError("no training tokens in the accumulated batch")

                micro_loss_sum = 0.0
                micro_tokens = 0
                last_loss = float("nan")
                window_poisoned = False
                for batch, tokens in zip(window, window_tokens):
                    partial = self.backend.train_step(
                        self.model,
                        self.optimizer,
                        batch,
                        grad_clip=cfg.grad_clip,
                        do_step=False,
                        loss_scale=tokens / total_window_tokens,
                    )
                    if math.isfinite(partial.loss):
                        micro_loss_sum += partial.loss * partial.tokens
                        micro_tokens += partial.tokens
                        last_loss = partial.loss
                    else:
                        window_poisoned = True

                if window_poisoned:
                    # 整个累积窗口作废：只用幸存 micro-batch 的梯度做 step 会
                    # 静默污染更新。在窗口边界清一次梯度并跳过本步。
                    self.backend.zero_grad(self.optimizer)
                    print(f"[warn] step {step_index + 1}: non-finite micro-batch loss; window discarded")
                    self.callbacks.on_event(state, "nan_window_discarded", {"step": step_index + 1})
                    step_index += 1
                    continue

                if micro_tokens == 0:
                    raise RuntimeError("no training tokens in the accumulated batch")
                optimizer_result = self.backend.optimizer_step(
                    self.model,
                    self.optimizer,
                    grad_clip=cfg.grad_clip,
                    loss_value=last_loss,
                    tokens=micro_tokens,
                )
                grad_norm = float(optimizer_result.grad_norm)
                if math.isfinite(grad_norm):
                    result["grad_norms"].append(grad_norm)

                completed = step_index + 1
                last_completed = completed
                train_loss = micro_loss_sum / micro_tokens
                train_losses.append(train_loss)
                # 末段均值取最近几个步：单个噪声 micro-window 不应左右 mini-run 门禁。
                tail = list(train_losses)[-min(5, len(train_losses)):]
                result["final_train_loss"] = sum(tail) / len(tail)
                self.tokens_seen += micro_tokens
                elapsed = time.monotonic() - started
                tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0.0

                state.step = completed
                state.lr = lr
                state.train_loss = train_loss
                state.grad_norm = grad_norm
                state.tokens_per_sec = tokens_per_sec
                state.elapsed = elapsed
                state.best_val = self.best_val
                self._heartbeat(completed)
                self.callbacks.on_step_end(state)

                if completed % cfg.log_interval == 0 or completed == 1:
                    print(
                        f"[{completed}/{cfg.max_steps}] loss={train_loss:.4f} "
                        f"lr={lr:.2e} grad={grad_norm:.3f} tok/s={tokens_per_sec:.0f} "
                        f"elapsed={fmt_seconds(elapsed)}"
                    )

                # 评估：按间隔，且最后一步必须评估（保证报告里有终态 val）。
                if completed % cfg.eval_interval == 0 or completed == cfg.max_steps:
                    val_loss = self.evaluate("val", cfg.eval_iters)
                    state.val_loss = val_loss
                    improved = self.early.update(val_loss, completed)
                    if improved:
                        self.best_val = val_loss
                    state.best_val = self.best_val
                    result["best_val"] = self.best_val
                    result["best_step"] = self.early.best_step
                    if improved:
                        self.save("best", completed)
                    self._write_metrics(
                        {
                            "step": completed,
                            "train_loss": f"{train_loss:.6f}",
                            "val_loss": f"{val_loss:.6f}",
                            # ppl/bpc 由 val_loss 推导；ppl 用 exp(min(loss, 50)) 防溢出。
                            "val_ppl": f"{math.exp(min(val_loss, 50.0)):.4f}",
                            "val_bpc": f"{val_loss / math.log(2):.6f}",
                            "lr": f"{lr:.8f}",
                            "grad_norm": f"{grad_norm:.6f}" if math.isfinite(grad_norm) else "nan",
                            "tokens_per_sec": f"{tokens_per_sec:.2f}",
                            "elapsed_s": f"{elapsed:.2f}",
                            "best_val": f"{self.best_val:.6f}",
                        }
                    )
                    self._write_status(step=completed, val_loss=val_loss, best_val=self.best_val)
                    self.callbacks.on_eval_end(state)
                    print(
                        f"[eval] step={completed} val_loss={val_loss:.4f} "
                        f"ppl={math.exp(min(val_loss, 50.0)):.2f} best={self.best_val:.4f}"
                    )
                    if self.early.should_stop:
                        result["stopped_early"] = True
                        state.extra["stopped_early"] = True
                        print(f"[early-stop] no improvement for {cfg.patience} evaluations")
                        self.callbacks.on_event(
                            state,
                            "early_stop_triggered",
                            {"best_val": self.best_val, "patience": cfg.patience, "step": completed},
                        )
                        self.save("last", completed)
                        break

                if completed % cfg.save_interval == 0:
                    self.save("last", completed)

                step_index += 1

            self.save("last", last_completed)
            result["steps"] = int(last_completed)
            result["elapsed_s"] = time.monotonic() - started
            result["status"] = "completed" if not result["stopped_early"] else "stopped"
            state.extra["status"] = result["status"]
            self._write_training_report(result)
            self._write_status(
                status=result["status"],
                step=int(last_completed),
                best_val=self.best_val,
                elapsed_s=result["elapsed_s"],
            )
            self.callbacks.on_train_end(state)
            print(
                f"[done] best_val={self.best_val:.4f} best_step={self.early.best_step} "
                f"elapsed={fmt_seconds(result['elapsed_s'])}"
            )
            return result
        except KeyboardInterrupt:
            result["status"] = "stopped"
            state.extra["status"] = "stopped"
            self._write_training_report(result)
            self._write_status(status="stopped", step=state.step, best_val=self.best_val)
            self.callbacks.on_train_end(state)
            print("[interrupted] saved status/report; last checkpoint remains on disk")
            return result
        except Exception as error:
            result["status"] = "failed"
            state.extra["status"] = "failed"
            self._write_training_report(result, error=f"{type(error).__name__}: {error}")
            self._write_status(status="failed", error=f"{type(error).__name__}: {error}")
            self.callbacks.on_error(state, error)
            raise
        finally:
            self.callbacks.close()

    def _write_training_report(self, result: dict[str, Any], error: str = "") -> None:
        """原子写 ``training_report.json``（成功、停止、失败三条路径共用）。

        参数 ``result`` 是主循环的累计结果，``error`` 仅失败路径传非空字符串。
        报告额外携带可移植超参、数据指纹、指标汇总与「能力边界」说明
        （N12 报告自动化），供模型卡与看板直接引用。
        """

        best_val = result.get("best_val", self.best_val)
        report = {
            "run_id": self.run_id,
            "stage": self.config.stage,
            "status": result.get("status", "unknown"),
            "steps": result.get("steps", 0),
            "best_val_loss": best_val,
            "best_step": result.get("best_step", self.early.best_step),
            "stopped_early": bool(result.get("stopped_early", False)),
            "patience": self.config.patience,
            "initial_loss": result.get("initial_loss"),
            "final_train_loss": result.get("final_train_loss"),
            "final_train_loss_note": "末段均值：最后 min(5, 完成步数) 个训练步 loss 的算术平均",
            "elapsed_s": result.get("elapsed_s", 0.0),
            "error": error,
            "created_at": utc_now(),
            # N12 报告自动化增强：可移植超参、数据指纹、指标汇总、能力边界。
            "hyperparameters": self.config.to_dict(portable=True),
            "data_fingerprint": _data_fingerprint(getattr(self.data_source, "metadata", None)),
            "metrics": {
                "initial_loss": result.get("initial_loss"),
                "final_train_loss": result.get("final_train_loss"),
                "best_val_loss": best_val,
                "best_step": result.get("best_step", self.early.best_step),
                "steps": result.get("steps", 0),
                "elapsed_s": result.get("elapsed_s", 0.0),
            },
            "capability_boundary": _capability_boundary(self.config.stage),
        }
        atomic_write_json(self.run_dir / "training_report.json", report)


__all__ = [
    "Trainer",
    "cosine_lr",
    "wsd_lr",
    "linear_lr",
    "lr_for_step",
    "LR_SCHEDULES",
    "check_mini_run",
    "METRIC_FIELDS",
]
