"""后端协议：PyTorch 与纯 Python 后端之间的可替换接缝。

Stage 1 只提供 ``TorchBackend``。纯 Python 后端 ``PureBackend``（自写张量与
反向传播）实现同一组方法，因此训练器、数据管线、checkpoint、遥测与 dashboard
都无需改动。

一个 ``batch`` 是数据层产出的普通字典::

    {"x": [[token_id, ...], ...], "y": [[token_id or -100, ...], ...]}

其中 ``y`` 里的 ``-100`` 是被忽略的位置（通常是 prompt 部分），不对 loss 有贡献。

所有框架相关细节（张量转换、device、autocast、优化器状态）都归后端所有，
训练器只看到 Python 数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class StepResult:
    """一次 ``train_step`` / ``optimizer_step`` 的统计结果。

    字段:
        loss: 本批次平均交叉熵（若启用梯度累积则为累积窗口内的均值）。
        grad_norm: 裁剪前的梯度范数；未做反向时可能为 0.0。
        tokens: 本批次计入 loss 的目标 token 数（已排除 ``-100`` 位置）。
        stepped: 是否真正执行了优化器更新（梯度累积未满一轮时为 False）。
        extra: 后端自定义的附加数值（如学习率、显存峰值），供遥测直接合并。
    """

    loss: float
    grad_norm: float
    tokens: int
    stepped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """一次 ``eval_step`` 的结果。

    字段:
        loss: 该批次平均交叉熵。
        tokens: 该批次计入 loss 的 token 数，用于跨批次加权平均。
    """

    loss: float
    tokens: int


@runtime_checkable
class ModelBackend(Protocol):
    """训练后端必须实现的全部方法（结构类型，无需继承）。

    实现方在模块导入时通过 ``@BACKENDS.register("名字")`` 登记，训练器只按
    ``config.backend_type`` 取实例，从不 ``isinstance`` 判断具体后端。
    """

    name: str
    version: str
    #: 检查点文件后缀（torch=".pt"，pure=".mgckpt"）。训练器据此命名 last/best，
    #: 不再按 backend.name 硬编码分支。
    checkpoint_suffix: str

    #: 用 ``model_config``（``GPTConfig``）与词表大小构造未训练的模型。
    def build_model(self, model_config: Any, vocab_size: int) -> Any: ...

    #: 按 ``train_config`` 的超参（学习率、weight decay、betas 等）构造优化器。
    def build_optimizer(self, model: Any, train_config: Any) -> Any: ...

    #: 前向 + 反向得到梯度；``do_step=False`` 时只累积梯度（梯度累积）。
    #: ``loss_scale`` 用于混合精度下的 loss 放大，后端负责对应缩小梯度。
    def train_step(
        self,
        model: Any,
        optimizer: Any,
        batch: dict[str, list[list[int]]],
        grad_clip: float,
        do_step: bool = True,
        loss_scale: float = 1.0,
    ) -> StepResult: ...

    #: 执行裁剪与参数更新（供梯度累积窗口末尾或手动分步调用）。
    def optimizer_step(self, model: Any, optimizer: Any, grad_clip: float) -> StepResult: ...

    #: 清空梯度缓存；部分后端用 ``set_to_none`` 语义以节省显存。
    def zero_grad(self, optimizer: Any) -> None: ...

    #: 无梯度的前向评估，返回平均 loss 与 token 数（用于加权汇总）。
    def eval_step(self, model: Any, batch: dict[str, list[list[int]]]) -> EvalResult: ...

    #: 导出模型权重为可序列化字典（键名即 checkpoint 中的 ``model`` 段）。
    def state_dict(self, model: Any) -> dict[str, Any]: ...

    #: 把 ``state`` 载入模型；后端负责 device/dtype 转换与键名匹配。
    def load_state_dict(self, model: Any, state: dict[str, Any]) -> None: ...

    #: 导出优化器状态（含步数），用于中断后精确续训。
    def optimizer_state_dict(self, optimizer: Any) -> dict[str, Any]: ...

    #: 恢复优化器状态；``state`` 为空时应能安全跳过。
    def load_optimizer_state_dict(self, optimizer: Any, state: dict[str, Any]) -> None: ...

    #: 自回归生成；返回完整序列（含 ``prompt_ids`` 前缀）。
    #: ``eos_id`` 为 None 表示不因结束符提前停止；``stop_sequences`` 命中任一
    #: token 序列即截断。采样规则由 :mod:`minigpt.sampling` 统一决定。
    def generate(
        self,
        model: Any,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_id: int | None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_sequences: list[list[int]] | None = None,
    ) -> list[int]: ...

    # -- 检查点格式（下沉到后端，训练器不再按 name 分支） -------------------
    def save_checkpoint(self, path: Any, payload: dict[str, Any]) -> None: ...

    def load_checkpoint(self, path: Any) -> dict[str, Any]: ...

    # -- 随机数状态（python 全局 + 后端自有 RNG） ---------------------------
    def capture_rng_state(self) -> dict[str, Any]: ...

    def restore_rng_state(self, state: dict[str, Any] | None) -> None: ...

    def device_info(self) -> dict[str, Any]: ...


__all__ = ["ModelBackend", "StepResult", "EvalResult"]
