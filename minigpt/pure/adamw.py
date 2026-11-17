"""纯后端使用的 AdamW（解耦 weight decay），仅标准库。

更新公式与 ``torch.optim.AdamW`` 一致，weight decay 与自适应项解耦::

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

全部用 Python float；优化器自己持有状态，因此未来的 pure checkpoint 可以把
``m``/``v`` 与参数一起序列化。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from minigpt.pure.tensor import Tensor


class AdamW:
    """作用于 :class:`~minigpt.pure.tensor.Tensor` 参数列表的最小 AdamW 实现。"""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """保存超参并为一阶/二阶动量分配与参数同形状的扁平缓冲。

        只有 ``requires_grad=True`` 的参数会参与更新。
        """
        self.params = [parameter for parameter in params if parameter.requires_grad]
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.m = [[0.0] * parameter.numel for parameter in self.params]
        self.v = [[0.0] * parameter.numel for parameter in self.params]

    def zero_grad(self) -> None:
        """清空所有参数的梯度。"""
        for parameter in self.params:
            parameter.zero_grad()

    def step(self) -> None:
        """执行一次参数更新（含偏差修正与解耦 weight decay）。"""
        self.step_count += 1
        bias1 = 1.0 - self.beta1 ** self.step_count
        bias2 = 1.0 - self.beta2 ** self.step_count
        for index, parameter in enumerate(self.params):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.data
            moments = self.m[index]
            velocities = self.v[index]
            for position in range(parameter.numel):
                grad = gradient[position]
                moments[position] = self.beta1 * moments[position] + (1 - self.beta1) * grad
                velocities[position] = self.beta2 * velocities[position] + (1 - self.beta2) * grad * grad
                m_hat = moments[position] / bias1
                v_hat = velocities[position] / bias2
                parameter.data[position] -= self.lr * (
                    m_hat / (math.sqrt(v_hat) + self.eps)
                    + self.weight_decay * parameter.data[position]
                )

    def state_dict(self) -> dict[str, object]:
        """导出超参、步数与 m/v 动量（深拷贝为普通 list，便于 JSON 序列化）。"""
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
            "m": [list(values) for values in self.m],
            "v": [list(values) for values in self.v],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """从 ``state_dict`` 恢复超参与动量（同时恢复 step_count 以保证偏差修正正确）。"""
        self.lr = float(state["lr"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.step_count = int(state["step_count"])
        self.m = [[float(value) for value in values] for values in state["m"]]  # type: ignore[union-attr]
        self.v = [[float(value) for value in values] for values in state["v"]]  # type: ignore[union-attr]


def clip_grad_norm(params: Iterable[Tensor], max_norm: float) -> float:
    """按全局 L2 范数就地裁剪梯度，返回**裁剪前**的范数。

    参数:
        params: 参数可迭代对象；无梯度的参数会被跳过。
        max_norm: 范数上限，``<= 0`` 表示只测量不裁剪。
    """
    total = 0.0
    for parameter in params:
        if parameter.grad is None:
            continue
        total += sum(value * value for value in parameter.grad.data)
    norm = math.sqrt(total)
    if max_norm > 0 and norm > max_norm:
        factor = max_norm / (norm + 1e-12)
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.data = [value * factor for value in parameter.grad.data]
    return norm


__all__ = ["AdamW", "clip_grad_norm"]
