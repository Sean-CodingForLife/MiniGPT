""":class:`ModelBackend` 协议的纯 Python 实现。

注册名 ``"pure"``，用 ``--backend_type pure`` 选择。它保持 Trainer 契约不变：
Trainer 只传 Python list、只收 ``StepResult``/``EvalResult``，因此数据加载、
遥测、checkpoint 与看板都无需改动。

与 :class:`TorchBackend` 的差异是刻意保留并有文档说明的：

* 设备固定 CPU，精度为 float64（即 Python float）；
* 优化器对**所有**参数施加 weight decay（torch 后端会排除一维 LayerNorm 权重），
  这是一处已记录的小偏差；
* 每次前向都重算整个上下文（无 KV cache），因此速度较慢。
"""

from __future__ import annotations

import math
import random
from typing import Any

from minigpt import registry
from minigpt.backends.base import EvalResult, StepResult
from minigpt.pure.adamw import AdamW, clip_grad_norm
from minigpt.pure import tensor as T
from minigpt.pure.gpt import PureGPT, PureGPTConfig


class PureBackend:
    """纯 Python 训练/推理后端（CPU + float64），注册名为 ``pure``。"""

    name = "pure"
    version = "1"
    checkpoint_suffix = ".mgckpt"

    def __init__(self, config: Any) -> None:
        """固定 CPU/float64，并按配置 seed 初始化后端私有随机源。"""
        self.config = config
        self.device = "cpu"
        self.dtype = "float64"
        seed = getattr(config, "seed", None)
        self.seed = 1337 if seed is None else int(seed)
        self.rng = random.Random(self.seed)

    # -- construction ---------------------------------------------------
    def build_model(self, model_config: Any, vocab_size: int) -> PureGPT:
        """按扁平化配置构建纯 Python GPT，并注入后端随机种子。"""
        config = PureGPTConfig.from_flat_config(model_config, vocab_size)
        return PureGPT(config, seed=self.seed)

    def build_optimizer(self, model: PureGPT, train_config: Any) -> AdamW:
        """构建 AdamW（eps 固定 1e-8，weight decay 作用于全部参数）。"""
        return AdamW(
            model.parameters(),
            lr=float(train_config.lr),
            betas=(float(train_config.beta1), float(train_config.beta2)),
            eps=1e-8,
            weight_decay=float(train_config.weight_decay),
        )

    # -- training -------------------------------------------------------
    @staticmethod
    def _tokens(batch: dict[str, list[list[int]]]) -> int:
        """统计参与 loss 的 token 数（排除 ``-100``）。"""
        return sum(1 for row in batch["y"] for value in row if int(value) != -100)

    def train_step(
        self,
        model: PureGPT,
        optimizer: AdamW,
        batch: dict[str, list[list[int]]],
        grad_clip: float,
        do_step: bool = True,
        loss_scale: float = 1.0,
    ) -> StepResult:
        tokens = self._tokens(batch)
        logits, loss = model.forward(batch["x"], batch["y"])
        if loss is None:
            return StepResult(loss=float("nan"), grad_norm=float("nan"), tokens=tokens, stepped=False)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            # 累积窗口中途不要清梯度：Trainer 会在窗口结束时整体丢弃并清理
            # （与 TorchBackend 的处理保持一致）。
            return StepResult(loss=float("nan"), grad_norm=float("nan"), tokens=tokens, stepped=False)
        # loss_scale 用于归一化梯度累积（见 TorchBackend）。
        # backward() 会把标量 loss 的初始梯度设为 self.grad（默认 1.0）。
        if loss_scale != 1.0:
            loss.grad = T.Tensor([loss_scale], ())
        loss.backward()
        if not do_step:
            return StepResult(loss=loss_value, grad_norm=0.0, tokens=tokens, stepped=False)
        return self.optimizer_step(model, optimizer, grad_clip, loss_value=loss_value, tokens=tokens)

    def zero_grad(self, optimizer: AdamW) -> None:
        """清空所有参数的梯度。"""
        optimizer.zero_grad()

    def optimizer_step(
        self,
        model: PureGPT,
        optimizer: AdamW,
        grad_clip: float,
        loss_value: float = float("nan"),
        tokens: int = 0,
    ) -> StepResult:
        """裁剪梯度并更新参数；梯度非有限时跳过更新并返回 ``stepped=False``。"""
        grad_norm = clip_grad_norm(model.parameters(), grad_clip)
        if not math.isfinite(grad_norm):
            optimizer.zero_grad()
            return StepResult(loss=loss_value, grad_norm=grad_norm, tokens=tokens, stepped=False)
        optimizer.step()
        optimizer.zero_grad()
        return StepResult(loss=loss_value, grad_norm=grad_norm, tokens=tokens, stepped=True)

    def eval_step(self, model: PureGPT, batch: dict[str, list[list[int]]]) -> EvalResult:
        """在 ``no_grad`` 下计算一个 batch 的 loss 与有效 token 数。"""
        tokens = self._tokens(batch)
        with T.no_grad():
            _logits, loss = model.forward(batch["x"], batch["y"])
        if loss is None:
            return EvalResult(loss=float("nan"), tokens=tokens)
        return EvalResult(loss=float(loss.item()), tokens=tokens)

    # -- state ----------------------------------------------------------
    def state_dict(self, model: PureGPT) -> dict[str, tuple[tuple[int, ...], list[float]]]:
        """导出 ``{名称: (形状, 扁平数据)}``，供 checkpoints/权重文件使用。"""
        return model.state_dict()

    def load_state_dict(
        self,
        model: PureGPT,
        state: dict[str, tuple[tuple[int, ...], list[float]]],
    ) -> None:
        # strict=True 会拒绝缺失参数（截断的 checkpoint 不能静默保留随机权重）
        # 与未知键；绑定的 lm_head 及 bias 键仍按 PureGPT 的设计跳过。
        model.load_state_dict(state, strict=True)

    def optimizer_state_dict(self, optimizer: AdamW) -> dict[str, Any]:
        """导出优化器状态（m/v、步数、超参），用于断点续训。"""
        return optimizer.state_dict()

    def load_optimizer_state_dict(self, optimizer: AdamW, state: dict[str, Any]) -> None:
        """恢复优化器状态。"""
        optimizer.load_state_dict(state)

    # -- inference ------------------------------------------------------
    def generate(
        self,
        model: PureGPT,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_id: int | None,
    ) -> list[int]:
        """非流式生成，返回 prompt + 续写的 token id 列表。"""
        return model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            rng=self.rng,
        )

    def generate_stream(
        self,
        model: PureGPT,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_id: int | None,
        trace_top_k: int = 0,
        trace_attention: bool = False,
        trace_sampling: bool = False,
        trace_candidates: int = 20,
    ):
        """逐 token 流式生成；trace_* 参数控制可视化所需的附加信息。"""
        return model.generate_stream(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            trace_top_k=trace_top_k,
            rng=self.rng,
            trace_attention=trace_attention,
            trace_sampling=trace_sampling,
            trace_candidates=trace_candidates,
        )

    # -- 检查点格式 / 随机数状态（ModelBackend 协议） ----------------------
    def save_checkpoint(self, path: Any, payload: dict[str, Any]) -> None:
        """按 ``.mgckpt`` 格式保存 checkpoint。"""
        from minigpt.pure.checkpoint import save_pure_checkpoint

        save_pure_checkpoint(path, payload)

    def load_checkpoint(self, path: Any) -> dict[str, Any]:
        """读取 ``.mgckpt``（必要时校验哈希），返回统一 payload。"""
        from minigpt.pure.checkpoint import load_pure_checkpoint

        return load_pure_checkpoint(path)

    def capture_rng_state(self) -> dict[str, Any]:
        """捕获 Python 全局 RNG 状态（数据采样/混合同样依赖它）。"""
        # pure 只依赖 Python 全局 RNG（数据采样/混合）；后端自有 RNG 由训练器
        # 单独写入 payload["rng"]["backend"]，这里不重复捕获。
        return {"python": random.getstate()}

    def restore_rng_state(self, state: dict[str, Any] | None) -> None:
        """恢复 Python 全局 RNG；``None``/空字典时静默跳过。"""
        from minigpt.utils import rng_state_to_tuple

        if not state:
            return
        python_state = state.get("python")
        if python_state:
            random.setstate(rng_state_to_tuple(python_state))

    def device_info(self) -> dict[str, Any]:
        """返回固定为 CPU/float64 的设备信息（``torch`` 字段留空表示未使用）。"""
        return {
            "backend": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "torch": "",
            "threads": 1,
        }


registry.BACKENDS.register("pure")(PureBackend)

__all__ = ["PureBackend"]
