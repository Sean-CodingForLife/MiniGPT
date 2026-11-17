"""不启动训练循环、直接加载 checkpoint 做推理的入口。

被 ``chat.py``、``evaluate.py`` 与 ``observe.py`` 复用。checkpoint 内嵌了
tokenizer 状态，因此 checkpoint 文件可以单独拷贝到另一台机器，无需原始
data 目录。

架构位置：位于 ``training.checkpoint``（torch 检查点读写）与
``pure.checkpoint``/``pure.inference``（纯标准库检查点）之上，对上层暴露
统一的 :class:`CheckpointBundle`。两种文件格式由文件头自动区分，调用方
不必关心后缀。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minigpt import registry

try:  # torch 可选：pure 检查点分派不需要 torch
    import minigpt.backends.torch_backend  # noqa: F401  (registers the torch backend)
except ImportError:
    pass
import minigpt.tokenization as _tokenization  # noqa: F401  (registers char tokenizer)
from minigpt.config import Config
from minigpt.training.checkpoint import load_checkpoint, validate_checkpoint


@dataclass
class CheckpointBundle:
    """推理所需的一组已就绪对象（torch 与 pure 两条路径共用此形状）。

    字段:
        payload: 原始 checkpoint 字典，含 ``config``、``model_config``、
            ``model``、``tokenizer_state``、``stage`` 等键；调用方可读取
            历史指标等附加信息。
        config: 已 ``resolve_paths()`` 且 ``validate()`` 过的运行时配置。
        backend: 计算后端实例，提供 ``device`` 与 ``load_state_dict``。
        model: 已加载权重并置为 ``eval()`` 的 GPT 模型。
        tokenizer: 由 checkpoint 内状态重建的 tokenizer，与训练时一致。
    """

    payload: dict[str, Any]
    config: Config
    backend: Any
    model: Any
    tokenizer: Any


def load_bundle(
    checkpoint_path: str | Path,
    device: str = "auto",
    dtype: str = "auto",
    data_dir: str | None = None,
) -> CheckpointBundle:
    """加载 checkpoint 并返回可直接推理的 :class:`CheckpointBundle`。

    参数:
        checkpoint_path: checkpoint 文件路径；按文件头判断是纯 Python
            格式（``.mgckpt``）还是 torch 格式（``.pt``）。
        device: 目标设备，``"auto"`` 表示交给后端自行选择；纯格式路径下
            该参数只作为记录，不涉及张量搬运。
        dtype: 推理精度（如 ``"float32"``/``"float16"``），``"auto"`` 由
            配置与后端决定；纯格式路径忽略此参数。
        data_dir: 覆盖 checkpoint 中记录的数据目录；``None`` 表示沿用原值。
            仅影响 ``config.data_dir``，tokenizer 已内嵌无需数据文件。
    返回:
        填充完毕的 :class:`CheckpointBundle`。
    异常:
        FileNotFoundError: checkpoint 不存在。
        ValueError: checkpoint 结构校验失败（由 ``validate_checkpoint`` 抛出）。
    """
    # 统一分派：按文件头识别 .mgckpt（pure）与 .pt（torch），调用方不再各自
    # 复制后缀判断。pure 分支懒加载，避免纯推理场景导入 torch。
    from minigpt.pure.checkpoint import is_pure_checkpoint

    if is_pure_checkpoint(checkpoint_path):
        from minigpt.pure.inference import load_pure_bundle

        return load_pure_bundle(checkpoint_path, device=device, data_dir=data_dir)
    # torch 路径：先以 CPU 反序列化，避免在没有 GPU 的机器上恢复 CUDA 张量失败。
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint(payload)

    config = Config()
    config.apply_dict(payload["config"])
    # stage 优先取 checkpoint 顶层值（旧格式可能只在顶层记录阶段）。
    config.stage = payload.get("stage", config.stage)
    config.device = device
    config.dtype = dtype
    if data_dir:
        config.data_dir = data_dir
    config.resolve_paths()
    config.validate()

    from minigpt.models.gpt import GPT, GPTConfig

    backend = registry.BACKENDS.get(config.backend_type)(config)
    model_config = GPTConfig.from_dict(payload["model_config"])
    model = GPT(model_config).to(backend.device)
    backend.load_state_dict(model, payload["model"])
    model.eval()

    # tokenizer 类型记录在 config 子字典里，缺失时回落到 char（最早期的 checkpoint）。
    tokenizer_cls = registry.TOKENIZERS.get(payload["config"].get("tokenizer_type", "char"))
    tokenizer = tokenizer_cls.from_state(payload["tokenizer_state"])
    return CheckpointBundle(payload, config, backend, model, tokenizer)


__all__ = ["CheckpointBundle", "load_bundle"]
