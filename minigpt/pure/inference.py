"""在**不依赖 PyTorch** 的情况下加载 ``.mgckpt`` checkpoint。

返回结构与 :class:`minigpt.inference.CheckpointBundle` 对齐，使 ``chat.py`` /
``evaluate.py`` 可以共用同一套生成与评估代码::

    payload, config, backend, model, tokenizer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minigpt import registry
from minigpt.config import Config
from minigpt.pure.backend import PureBackend
from minigpt.pure.checkpoint import load_pure_checkpoint
from minigpt.pure.gpt import PureGPT, PureGPTConfig
import minigpt.tokenization  # noqa: F401  （注册 char tokenizer）


@dataclass
class PureCheckpointBundle:
    """一次加载得到的纯后端推理上下文（payload/config/backend/model/tokenizer）。"""

    payload: dict[str, Any]
    config: Config
    backend: PureBackend
    model: PureGPT
    tokenizer: Any


def load_pure_bundle(
    checkpoint_path: str | Path,
    device: str = "cpu",
    data_dir: str | None = None,
) -> PureCheckpointBundle:
    """加载 ``.mgckpt``，恢复配置、权重与 tokenizer，并构建可推理的 bundle。

    参数:
        checkpoint_path: ``.mgckpt`` 文件路径。
        device: 仅为接口对齐保留；pure 引擎始终在 CPU 上运行。
        data_dir: 覆盖 checkpoint 内记录的数据目录（通常用于评估/观察）。
    """
    payload = load_pure_checkpoint(checkpoint_path)

    config = Config()
    config.apply_dict(payload["config"])
    config.stage = payload.get("stage", config.stage)
    config.device = "cpu"
    # dtype 是 torch 侧的 autocast 选项，pure 引擎始终用 Python float（float64）
    # 计算、不受其影响；这里沿用合法默认值，实际精度以 device_info 汇报为准。
    config.dtype = "float32"
    if data_dir:
        config.data_dir = data_dir
    config.resolve_paths()
    config.validate()

    backend = PureBackend(config)
    model = PureGPT(PureGPTConfig.from_dict(payload["model_config"]))
    backend.load_state_dict(model, payload["model"])
    tokenizer_cls = registry.TOKENIZERS.get(payload["config"].get("tokenizer_type", "char"))
    tokenizer = tokenizer_cls.from_state(payload["tokenizer_state"])
    return PureCheckpointBundle(payload, config, backend, model, tokenizer)


__all__ = ["PureCheckpointBundle", "load_pure_bundle"]
