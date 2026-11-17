""":class:`ModelBackend` 的 PyTorch 实现（Stage 1）。

后端把训练循环与框架解耦：Trainer 只传 Python list、只收 Python float。
将来用纯自动求导后端替换本文件时，Trainer、数据、checkpoint 与看板都不需要改。

设计要点
--------
* CPU 默认 float32；CUDA 优先 bfloat16，不支持时用 float16 + 梯度缩放器。
* 梯度累积通过多次 ``train_step(..., do_step=False)`` 后调用一次
  ``optimizer_step`` 完成。
* 返回的梯度范数是**裁剪前**测量值；出现非有限梯度时跳过优化器更新并返回
  ``stepped=False``。
"""

from __future__ import annotations

import math
from typing import Any

import torch

from minigpt import registry
from minigpt.backends.base import EvalResult, StepResult
from minigpt.models.gpt import GPT, GPTConfig


class TorchBackend:
    """基于 PyTorch 的训练/推理后端，注册名为 ``torch``。"""

    name = "torch"
    version = "1"
    checkpoint_suffix = ".pt"

    def __init__(self, config: Any):
        """解析设备/精度，配置线程数与（必要时）CUDA 梯度缩放器。

        参数:
            config: 训练配置，读取 ``device``、``dtype``、``num_threads``。
        """
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype, self.device)
        self.ptdtype = self._torch_dtype(self.dtype)
        self.use_amp = self.dtype != "float32"
        self.use_scaler = self.use_amp and self.dtype == "float16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler) if self.use_scaler else None
        if config.num_threads and config.num_threads > 0:
            torch.set_num_threads(config.num_threads)
        self._last_tokens = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        """把 ``auto/cpu/cuda`` 解析为具体设备；显式要 CUDA 但不可用时直接报错。"""
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device(requested)

    @staticmethod
    def _resolve_dtype(requested: str, device: torch.device) -> str:
        """解析精度：显式值原样校验，``auto`` 按设备能力选择。"""
        if requested != "auto":
            if requested == "float16" and device.type == "cpu":
                raise ValueError("float16 autocast is not supported on CPU; use bfloat16 or float32")
            return requested
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
            return "float16"
        return "float32"

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        """字符串精度名 -> ``torch.dtype``。"""
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]

    def _autocast(self):
        """返回 autocast 上下文；float32 时显式 ``enabled=False``。"""
        if not self.use_amp:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.ptdtype, enabled=True)

    # ------------------------------------------------------------------
    def build_model(self, model_config: Any, vocab_size: int) -> GPT:
        """按扁平化配置与词表大小构建 GPT，并移到目标设备。"""
        gpt_config = GPTConfig.from_flat_config(model_config, vocab_size)
        model = GPT(gpt_config).to(self.device)
        return model

    def build_optimizer(self, model: GPT, train_config: Any):
        """构建 AdamW：二维及以上权重做 weight decay，偏置/LayerNorm 不做。

        CUDA 上优先尝试 ``fused=True``，不被支持时自动回退到普通实现。
        """
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": train_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {"lr": train_config.lr, "betas": (train_config.beta1, train_config.beta2)}
        if self.device.type == "cuda":
            try:
                return torch.optim.AdamW(groups, fused=True, **kwargs)
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(groups, **kwargs)

    # ------------------------------------------------------------------
    def _batch_tensors(self, batch: dict[str, list[list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        """把数据源的 Python list 批次转成设备上的 int64 张量。"""
        x = torch.tensor(batch["x"], dtype=torch.long, device=self.device)
        y = torch.tensor(batch["y"], dtype=torch.long, device=self.device)
        return x, y

    @staticmethod
    def _valid_tokens(y: torch.Tensor) -> int:
        """统计参与 loss 的 token 数（``-100`` 为忽略位）。"""
        return int((y != -100).sum().item())

    def train_step(
        self,
        model: GPT,
        optimizer: Any,
        batch: dict[str, list[list[int]]],
        grad_clip: float,
        do_step: bool = True,
        loss_scale: float = 1.0,
    ) -> StepResult:
        """执行一次前向 + 反向；``do_step=False`` 时只累积梯度。

        参数:
            batch: 含 ``x``/``y`` 的 token 批次。
            grad_clip: 梯度范数裁剪阈值。
            do_step: 是否在本 micro-batch 后立刻更新参数。
            loss_scale: 梯度累积的缩放系数，通常为 ``1/累积步数``。
        返回:
            ``StepResult``；loss 非有限时返回 NaN 且 ``stepped=False``。
        """
        model.train()
        x, y = self._batch_tensors(batch)
        tokens = self._valid_tokens(y)
        with self._autocast():
            _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            # 这里不能清零梯度：累积期间清零会抹掉此前健康的 micro-batch。
            # Trainer 会在整个累积窗口结束时丢弃窗口并统一清梯度。
            return StepResult(loss=float("nan"), grad_norm=float("nan"), tokens=tokens, stepped=False)

        # loss_scale 用于归一化梯度累积：K 个 micro-batch 的平均 loss 先除以 K
        # 再反传，使累积结果接近一个大 batch，而不是 K 倍梯度。
        scaled_loss = loss * loss_scale
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if not do_step:
            return StepResult(loss=float(loss.item()), grad_norm=0.0, tokens=tokens, stepped=False)

        return self.optimizer_step(model, optimizer, grad_clip, loss_value=float(loss.item()), tokens=tokens)

    def zero_grad(self, optimizer: Any) -> None:
        """清空梯度（``set_to_none=True``，省内存且语义更干净）。"""
        optimizer.zero_grad(set_to_none=True)

    def optimizer_step(
        self,
        model: GPT,
        optimizer: Any,
        grad_clip: float,
        loss_value: float = float("nan"),
        tokens: int = 0,
    ) -> StepResult:
        """反缩放、裁剪梯度并更新参数；梯度非有限时跳过本次更新。

        返回的 ``grad_norm`` 是裁剪前的值，便于监控训练稳定性。
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = float(grad_norm_tensor.item())
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()
            return StepResult(loss=loss_value, grad_norm=grad_norm, tokens=tokens, stepped=False)

        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return StepResult(loss=loss_value, grad_norm=grad_norm, tokens=tokens, stepped=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_step(self, model: GPT, batch: dict[str, list[list[int]]]) -> EvalResult:
        """在 ``no_grad`` 下计算一个 batch 的 loss，并返回有效 token 数。"""
        model.eval()
        x, y = self._batch_tensors(batch)
        with self._autocast():
            _, loss = model(x, y)
        tokens = self._valid_tokens(y)
        if loss is None:
            return EvalResult(loss=float("nan"), tokens=tokens)
        return EvalResult(loss=float(loss.item()), tokens=tokens)

    # ------------------------------------------------------------------
    def state_dict(self, model: GPT) -> dict[str, Any]:
        """导出模型权重（脱离计算图并放到 CPU），用于 checkpoint/导出。"""
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}

    def load_state_dict(self, model: GPT, state: dict[str, Any]) -> None:
        """把（可能在 CPU 上的）权重加载回模型所在设备。"""
        model.load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def optimizer_state_dict(self, optimizer: Any) -> dict[str, Any]:
        """导出优化器状态（AdamW 的动量等）。"""
        return optimizer.state_dict()

    def load_optimizer_state_dict(self, optimizer: Any, state: dict[str, Any]) -> None:
        """恢复优化器状态。"""
        optimizer.load_state_dict(state)

    # ------------------------------------------------------------------
    def generate(
        self,
        model: GPT,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_id: int | None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_sequences: list[list[int]] | None = None,
    ) -> list[int]:
        """非流式生成，返回 prompt + 续写的完整 token id 列表。"""
        model.eval()
        return model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stop_sequences=stop_sequences,
        )

    def generate_stream(
        self,
        model: GPT,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_id: int | None,
        trace_top_k: int = 0,
        trace_attention: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_sequences: list[list[int]] | None = None,
    ):
        """逐 token 流式生成，产出字典序列（字段见 ``GPT.generate_stream``）。"""

        model.eval()
        return model.generate_stream(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            trace_top_k=trace_top_k,
            trace_attention=trace_attention,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stop_sequences=stop_sequences,
        )

    # -- 检查点格式 / 随机数状态（ModelBackend 协议） ----------------------
    def save_checkpoint(self, path: Any, payload: dict[str, Any]) -> None:
        """按 torch 格式保存 checkpoint（委托给 training.checkpoint）。"""
        from minigpt.training.checkpoint import save_checkpoint

        save_checkpoint(path, payload)

    def load_checkpoint(self, path: Any) -> dict[str, Any]:
        """加载 torch checkpoint，并映射到当前设备（委托给 training.checkpoint）。"""
        from minigpt.training.checkpoint import load_checkpoint

        return load_checkpoint(path, map_location=self.device)

    def capture_rng_state(self) -> dict[str, Any]:
        """采集 Python / torch / CUDA 的随机数状态，写入 checkpoint 以保证续训可复现。"""
        from minigpt.training.checkpoint import capture_rng_state

        return capture_rng_state()

    def restore_rng_state(self, state: dict[str, Any] | None) -> None:
        """恢复之前采集的随机数状态；``None`` 表示无状态可恢复（静默跳过）。"""
        from minigpt.training.checkpoint import restore_rng_state

        restore_rng_state(state)

    def device_info(self) -> dict[str, Any]:
        """返回设备/精度/线程数的可序列化快照，写入 environment.json 与看板。"""
        info = {
            "backend": self.name,
            "device": str(self.device),
            "dtype": self.dtype,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
        }
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return info


registry.BACKENDS.register("torch")(TorchBackend)

__all__ = ["TorchBackend"]
