"""训练后端（backend）。

导入本包不引入 PyTorch：具体后端模块在被导入时自行注册：

    from minigpt.backends.torch_backend import TorchBackend  # registers "torch"
    from minigpt.pure.backend import PureBackend            # registers "pure"

正因如此，本 ``__init__`` 只再导出与框架无关的协议与结果类型，
dashboard 或纯标准库工具可以 ``import minigpt.backends.base`` 而不牵入
PyTorch。
"""

from minigpt.backends.base import EvalResult, ModelBackend, StepResult

__all__ = ["ModelBackend", "StepResult", "EvalResult"]
