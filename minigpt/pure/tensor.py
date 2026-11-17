"""自写张量 + 反向模式自动求导（纯 Python 标准库）。

这是 Stage 2 的基础：只实现项目真正需要的那一小撮算子，目标是可读，而不是
替代 NumPy。

数据表示
--------
``Tensor`` 内部是行主序的一维 ``list[float]`` 加一个 ``shape`` 元组。每次运算
返回新的 ``Tensor``，并记录「如何把梯度回传给父节点」。``backward`` 按逆拓扑序
遍历计算图，把梯度累加进 ``.grad``（同样是 ``Tensor``）。

支持的算子（GPT 前向/反向恰好只需要这些）
----------------------------------------
* 逐元素 ``add`` / ``mul``（右对齐广播）、``scale``
* ``matmul``（二维，以及带广播 batch 维的高维）
* ``transpose_last``、``permute``、``reshape``、``slice_last``
* ``softmax``（最后一维）、``layer_norm``、``gelu``（tanh 近似）
* ``embedding`` 查表、``cross_entropy``（支持 ``-100`` 忽略索引）
* ``sum_all``：把 loss 变成标量

数值使用 Python float（双精度）；未来若换后端，可把存储改成 ``array('f')``
换取速度，代价是精度。

示例::

    from minigpt.pure.tensor import Tensor, matmul, softmax

    x = Tensor([1.0, 2.0], (2,), requires_grad=True)
    loss = sum_all(mul(x, x))
    loss.backward()
    print(x.grad.data)  # [2.0, 4.0]
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Sequence

_GLOBAL_GRAD_ENABLED = True


class no_grad:
    """禁用计算图构建的上下文管理器（推理时省内存、提速）。"""

    def __enter__(self) -> "no_grad":
        """保存进入前的开关状态并关闭梯度记录。"""
        global _GLOBAL_GRAD_ENABLED
        self._previous = _GLOBAL_GRAD_ENABLED
        _GLOBAL_GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """退出时恢复进入前的开关状态（异常路径也会恢复）。"""
        global _GLOBAL_GRAD_ENABLED
        _GLOBAL_GRAD_ENABLED = self._previous


def _numel(shape: Sequence[int]) -> int:
    """形状对应的元素总数（空形状的标量为 1）。"""
    total = 1
    for size in shape:
        total *= int(size)
    return total


def _strides(shape: Sequence[int]) -> list[int]:
    """计算行主序（C 顺序）下每一维的跨步，用于把多维索引压平。"""
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * int(shape[index + 1])
    return stride


def _flat(shape: Sequence[int], index: Sequence[int]) -> int:
    """把多维索引换算成一维偏移，并做边界检查。"""
    offset = 0
    for size, position, stride in zip(shape, index, _strides(shape)):
        if position < 0 or position >= size:
            raise IndexError(f"index {position} out of range for dimension {size}")
        offset += position * stride
    return offset


class Tensor:
    """稠密浮点张量，可选地参与梯度追踪。"""

    __slots__ = ("data", "shape", "requires_grad", "grad", "_prev", "_backward", "_op")

    def __init__(
        self,
        data: Iterable[float] | float,
        shape: Sequence[int] | None = None,
        requires_grad: bool = False,
    ) -> None:
        """从扁平数据与形状构造张量。

        参数:
            data: 扁平 float 序列；传单个数字时视为 0 维标量（形状必须为 ``()``）。
            shape: 目标形状；省略时按一维处理。
            requires_grad: 是否追踪梯度（叶节点参数通常为 True）。
        异常:
            ValueError: 元素个数与形状不符，或标量形状不合法。
        """
        if isinstance(data, (int, float)):
            values = [float(data)]
            if shape is None or tuple(shape) != ():
                raise ValueError("scalar Tensor must have shape ()")
        else:
            values = [float(value) for value in data]
        if shape is None:
            shape = (len(values),)
        self.shape = tuple(int(size) for size in shape)
        if _numel(self.shape) != len(values):
            raise ValueError(f"shape {self.shape} does not match {len(values)} values")
        self.data = values
        self.requires_grad = bool(requires_grad)
        self.grad: Tensor | None = None
        self._prev: tuple[Tensor, ...] = ()
        self._backward = None
        self._op = "leaf"

    # -- basics ---------------------------------------------------------
    @property
    def numel(self) -> int:
        """元素总数。"""
        return len(self.data)

    def item(self) -> float:
        """取出单元素张量的 Python 浮点值。"""
        if self.numel != 1:
            raise ValueError("item() requires a single-element Tensor")
        return self.data[0]

    def detach(self) -> "Tensor":
        """返回共享数据、但不参与梯度的新张量（类似 ``torch.detach``）。"""
        return Tensor(self.data, self.shape)

    def zero_grad(self) -> None:
        """清空梯度。"""
        self.grad = None

    def tolist(self) -> list[float]:
        """以 Python list 返回扁平数据（副本）。"""
        return list(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data[:4]}{'...' if len(self.data) > 4 else ''})"

    # -- autograd -------------------------------------------------------
    def backward(self) -> None:
        """从当前张量（通常是标量 loss）反向传播梯度。

        实现方式：显式栈做拓扑排序，再按逆序调用每个节点的 ``_backward``；
        每个闭包只执行一次（执行后置 None），因此计算图是「一次性」的。
        """

        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))
        if self.grad is None:
            self.grad = Tensor([1.0], ())
        for node in reversed(order):
            if node._backward is None or node.grad is None:
                continue
            node._backward(node.grad.data)
            node._backward = None  # free closures; graph is single-use

    def _accumulate(self, values: Sequence[float]) -> None:
        """把一批梯度累加到 ``self.grad``；首次写入时按自身形状创建。"""
        if self.grad is None:
            self.grad = Tensor(values, self.shape)
        else:
            for index, value in enumerate(values):
                self.grad.data[index] += value


def _wrap(
    data: list[float],
    shape: Sequence[int],
    parents: Sequence[Tensor],
    backward,
    op: str,
) -> Tensor:
    """构造运算结果张量：仅在需要时挂上父节点与反向闭包。

    ``no_grad`` 期间或所有父节点都不需要梯度时，结果为「叶子」，不保留计算图。
    """
    requires = _GLOBAL_GRAD_ENABLED and any(parent.requires_grad for parent in parents)
    out = Tensor.__new__(Tensor)
    out.data = data
    out.shape = tuple(int(size) for size in shape)
    out.requires_grad = requires
    out.grad = None
    out._prev = tuple(parents) if requires else ()
    out._backward = backward if requires else None
    out._op = op
    return out


# ---------------------------------------------------------------------------
# broadcasting helpers
# ---------------------------------------------------------------------------
def broadcast_shape(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """按右对齐规则计算广播后的形状；无法广播时抛 ``ValueError``。

    规则与 NumPy 一致：从最右维开始比较，相等或其中一方为 1 即可广播。
    """
    n = max(len(a), len(b))
    left = [1] * (n - len(a)) + list(a)
    right = [1] * (n - len(b)) + list(b)
    out: list[int] = []
    for x, y in zip(left, right):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError(f"shapes {tuple(a)} and {tuple(b)} are not broadcastable")
    return tuple(out)


def _right_index(index: tuple[int, ...], out_shape: Sequence[int], in_shape: Sequence[int]) -> tuple[int, ...]:
    """把输出索引映射回某个输入张量的索引（右对齐 + 广播维取 0）。"""
    pad = len(out_shape) - len(in_shape)
    return tuple(index[pad + i] if in_shape[i] != 1 else 0 for i in range(len(in_shape)))


def _all_indices(shape: Sequence[int]):
    """生成形状内所有多维索引（笛卡尔积）。"""
    return itertools.product(*(range(size) for size in shape))


# ---------------------------------------------------------------------------
# 逐元素运算
# ---------------------------------------------------------------------------
def add(a: Tensor, b: Tensor | float) -> Tensor:
    """逐元素加法，支持标量与右对齐广播；反向时对广播维求和回原形状。"""
    b = b if isinstance(b, Tensor) else Tensor([float(b)], ())
    out_shape = broadcast_shape(a.shape, b.shape)
    data = [0.0] * _numel(out_shape)
    for index in _all_indices(out_shape):
        left = a.data[_flat(a.shape, _right_index(index, out_shape, a.shape))]
        right = b.data[_flat(b.shape, _right_index(index, out_shape, b.shape))]
        data[_flat(out_shape, index)] = left + right

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        grad_b = [0.0] * b.numel
        for index in _all_indices(out_shape):
            value = grad[_flat(out_shape, index)]
            grad_a[_flat(a.shape, _right_index(index, out_shape, a.shape))] += value
            grad_b[_flat(b.shape, _right_index(index, out_shape, b.shape))] += value
        if a.requires_grad:
            a._accumulate(grad_a)
        if b.requires_grad:
            b._accumulate(grad_b)

    return _wrap(data, out_shape, (a, b), backward, "add")


def mul(a: Tensor, b: Tensor | float) -> Tensor:
    """逐元素乘法；反向分别是「乘对方的值」，广播维同样累加回原形状。"""
    b = b if isinstance(b, Tensor) else Tensor([float(b)], ())
    out_shape = broadcast_shape(a.shape, b.shape)
    data = [0.0] * _numel(out_shape)
    for index in _all_indices(out_shape):
        flat_a = _flat(a.shape, _right_index(index, out_shape, a.shape))
        flat_b = _flat(b.shape, _right_index(index, out_shape, b.shape))
        data[_flat(out_shape, index)] = a.data[flat_a] * b.data[flat_b]

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        grad_b = [0.0] * b.numel
        for index in _all_indices(out_shape):
            value = grad[_flat(out_shape, index)]
            flat_a = _flat(a.shape, _right_index(index, out_shape, a.shape))
            flat_b = _flat(b.shape, _right_index(index, out_shape, b.shape))
            if a.requires_grad:
                grad_a[flat_a] += value * b.data[flat_b]
            if b.requires_grad:
                grad_b[flat_b] += value * a.data[flat_a]
        if a.requires_grad:
            a._accumulate(grad_a)
        if b.requires_grad:
            b._accumulate(grad_b)

    return _wrap(data, out_shape, (a, b), backward, "mul")


def scale(a: Tensor, factor: float) -> Tensor:
    """乘以标量；反向也乘以同一标量。"""
    factor = float(factor)
    data = [value * factor for value in a.data]

    def backward(grad: list[float]) -> None:
        a._accumulate([value * factor for value in grad])

    return _wrap(data, a.shape, (a,), backward, "scale")


def sum_all(a: Tensor) -> Tensor:
    """对所有元素求和，返回 0 维标量；反向把同一个梯度复制给每个元素。"""
    total = sum(a.data)

    def backward(grad: list[float]) -> None:
        a._accumulate([grad[0]] * a.numel)

    return _wrap([total], (), (a,), backward, "sum_all")


# ---------------------------------------------------------------------------
# 形状变换
# ---------------------------------------------------------------------------
def reshape(a: Tensor, shape: Sequence[int]) -> Tensor:
    """改变形状但保持元素顺序（行主序），元素总数必须不变。"""
    shape = tuple(int(size) for size in shape)
    if _numel(shape) != a.numel:
        raise ValueError(f"cannot reshape {a.shape} to {shape}")

    def backward(grad: list[float]) -> None:
        a._accumulate(grad)

    return _wrap(list(a.data), shape, (a,), backward, "reshape")


def transpose_last(a: Tensor) -> Tensor:
    """交换最后两维（批量矩阵转置），前面的维度保持不变。"""
    if len(a.shape) < 2:
        raise ValueError("transpose_last needs at least 2 dimensions")
    shape = a.shape[:-2] + (a.shape[-1], a.shape[-2])
    batch = _numel(a.shape[:-2])
    rows, cols = a.shape[-2], a.shape[-1]
    data = [0.0] * a.numel
    for batch_index in range(batch):
        base_in = batch_index * rows * cols
        base_out = batch_index * rows * cols
        for i in range(rows):
            for j in range(cols):
                data[base_out + j * rows + i] = a.data[base_in + i * cols + j]

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        for batch_index in range(batch):
            base = batch_index * rows * cols
            for i in range(rows):
                for j in range(cols):
                    grad_a[base + i * cols + j] = grad[base + j * rows + i]
        a._accumulate(grad_a)

    return _wrap(data, shape, (a,), backward, "transpose_last")


def permute(a: Tensor, dims: Sequence[int]) -> Tensor:
    """按 ``dims`` 重排维度；反向用逆置换把梯度搬回原顺序。"""
    dims = tuple(int(dim) for dim in dims)
    if sorted(dims) != list(range(len(a.shape))):
        raise ValueError(f"invalid permutation {dims} for shape {a.shape}")
    shape = tuple(a.shape[dim] for dim in dims)
    data = [0.0] * a.numel
    for index in _all_indices(shape):
        source = tuple(index[dims.index(dim)] for dim in range(len(a.shape)))
        data[_flat(shape, index)] = a.data[_flat(a.shape, source)]

    inverse = tuple(dims.index(dim) for dim in range(len(dims)))

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        for index in _all_indices(shape):
            source = tuple(index[dims.index(dim)] for dim in range(len(a.shape)))
            grad_a[_flat(a.shape, source)] = grad[_flat(shape, index)]
        a._accumulate(grad_a)

    return _wrap(data, shape, (a,), backward, "permute")


def slice_last(a: Tensor, start: int, end: int) -> Tensor:
    """截取最后一维的 ``[start, end)`` 区间；其余维度原样保留。"""
    if not a.shape:
        raise ValueError("slice_last needs at least 1 dimension")
    width = a.shape[-1]
    if not 0 <= start < end <= width:
        raise ValueError(f"invalid slice [{start}, {end}) for width {width}")
    rows = a.numel // width
    keep = end - start
    shape = a.shape[:-1] + (keep,)
    data = [0.0] * (rows * keep)
    for row in range(rows):
        base = row * width
        target = row * keep
        for offset in range(keep):
            data[target + offset] = a.data[base + start + offset]

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        for row in range(rows):
            base = row * width
            target = row * keep
            for offset in range(keep):
                grad_a[base + start + offset] = grad[target + offset]
        a._accumulate(grad_a)

    return _wrap(data, shape, (a,), backward, "slice_last")


# ---------------------------------------------------------------------------
# 线性代数
# ---------------------------------------------------------------------------
def _batch_indices(batch_shape: Sequence[int]):
    """生成 batch 维上的所有索引组合。"""
    return itertools.product(*(range(size) for size in batch_shape))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """批量矩阵乘法 ``[..., M, K] @ [..., K, N] -> [..., M, N]``。

    batch 维按广播规则对齐；反向分别用 ``grad @ B^T`` 与 ``A^T @ grad``。
    """
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("matmul needs at least 2 dimensions on both operands")
    rows, inner = a.shape[-2:]
    inner_b, cols = b.shape[-2:]
    if inner != inner_b:
        raise ValueError(f"matmul mismatch: {a.shape} @ {b.shape}")
    batch = broadcast_shape(a.shape[:-2], b.shape[:-2])
    out_shape = batch + (rows, cols)
    data = [0.0] * _numel(out_shape)
    # 预转置 B：把内层对 k 的跨步访问变成连续行访问，配合生成器 sum 在 C 层
    # 迭代，显著减少纯 Python 逐元素开销。求和顺序仍是 k=0..inner-1，数值不变。
    b_t = [0.0] * b.numel
    b_batch = _numel(b.shape[:-2])
    for bi in range(b_batch):
        base_in = bi * inner * cols
        base_out = bi * inner * cols
        for k in range(inner):
            for j in range(cols):
                b_t[base_out + j * inner + k] = b.data[base_in + k * cols + j]
    a_data = a.data
    for batch_index in _batch_indices(batch):
        index_a = _right_index(batch_index, batch, a.shape[:-2]) if a.shape[:-2] else ()
        index_b = _right_index(batch_index, batch, b.shape[:-2]) if b.shape[:-2] else ()
        offset_a = (_flat(a.shape[:-2], index_a) if a.shape[:-2] else 0) * rows * inner
        offset_b = (_flat(b.shape[:-2], index_b) if b.shape[:-2] else 0) * inner * cols
        base_out = _flat(batch, batch_index) * rows * cols
        for i in range(rows):
            a_row_base = offset_a + i * inner
            out_row_base = base_out + i * cols
            for j in range(cols):
                b_col_base = offset_b + j * inner
                data[out_row_base + j] = sum(
                    a_data[a_row_base + k] * b_t[b_col_base + k] for k in range(inner)
                )

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        grad_b = [0.0] * b.numel
        for batch_index in _batch_indices(batch):
            index_a = _right_index(batch_index, batch, a.shape[:-2]) if a.shape[:-2] else ()
            index_b = _right_index(batch_index, batch, b.shape[:-2]) if b.shape[:-2] else ()
            offset_a = (_flat(a.shape[:-2], index_a) if a.shape[:-2] else 0) * rows * inner
            offset_b = (_flat(b.shape[:-2], index_b) if b.shape[:-2] else 0) * inner * cols
            base_out = _flat(batch, batch_index) * rows * cols
            for i in range(rows):
                for j in range(cols):
                    value = grad[base_out + i * cols + j]
                    if value == 0.0:
                        continue
                    for k in range(inner):
                        if a.requires_grad:
                            grad_a[offset_a + i * inner + k] += value * b.data[offset_b + k * cols + j]
                        if b.requires_grad:
                            grad_b[offset_b + k * cols + j] += value * a.data[offset_a + i * inner + k]
        if a.requires_grad:
            a._accumulate(grad_a)
        if b.requires_grad:
            b._accumulate(grad_b)

    return _wrap(data, out_shape, (a, b), backward, "matmul")


# ---------------------------------------------------------------------------
# 神经网络基础算子
# ---------------------------------------------------------------------------
def softmax(a: Tensor) -> Tensor:
    """对最后一维做数值稳定的 softmax；反向用 ``y * (g - dot(g, y))``。"""
    if not a.shape:
        raise ValueError("softmax needs at least 1 dimension")
    width = a.shape[-1]
    rows = a.numel // width
    data = [0.0] * a.numel
    for row in range(rows):
        base = row * width
        maximum = max(a.data[base + index] for index in range(width))
        total = 0.0
        for index in range(width):
            value = math.exp(a.data[base + index] - maximum)
            data[base + index] = value
            total += value
        if total == 0.0:
            raise ValueError("softmax row sums to zero (all -inf?)")
        for index in range(width):
            data[base + index] /= total

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        for row in range(rows):
            base = row * width
            dot = 0.0
            for index in range(width):
                dot += grad[base + index] * data[base + index]
            for index in range(width):
                grad_a[base + index] = data[base + index] * (grad[base + index] - dot)
        a._accumulate(grad_a)

    return _wrap(data, a.shape, (a,), backward, "softmax")


def layer_norm(a: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """对最后一维做 LayerNorm（无偏置），再乘可学习的 ``weight``。

    反向使用标准的三项闭式梯度（含均值/方差带来的耦合项）。
    """
    width = a.shape[-1]
    rows = a.numel // width
    data = [0.0] * a.numel
    means = [0.0] * rows
    inv_std = [0.0] * rows
    for row in range(rows):
        base = row * width
        mean = sum(a.data[base : base + width]) / width
        variance = sum((a.data[base + index] - mean) ** 2 for index in range(width)) / width
        std = math.sqrt(variance + eps)
        means[row] = mean
        inv_std[row] = 1.0 / std
        for index in range(width):
            data[base + index] = (a.data[base + index] - mean) / std * weight.data[index]

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        grad_w = [0.0] * width
        for row in range(rows):
            base = row * width
            mean = means[row]
            inv = inv_std[row]
            normalized = [(a.data[base + index] - mean) * inv for index in range(width)]
            sum_grad = 0.0
            sum_grad_norm = 0.0
            for index in range(width):
                grad_w[index] += grad[base + index] * normalized[index]
                scaled = grad[base + index] * weight.data[index]
                sum_grad += scaled
                sum_grad_norm += scaled * normalized[index]
            for index in range(width):
                scaled = grad[base + index] * weight.data[index]
                grad_a[base + index] = inv * (
                    scaled - sum_grad / width - normalized[index] * sum_grad_norm / width
                )
        a._accumulate(grad_a)
        weight._accumulate(grad_w)

    return _wrap(data, a.shape, (a, weight), backward, "layer_norm")


def gelu(a: Tensor) -> Tensor:
    """GELU 的 tanh 近似（与 torch 的 ``approximate="tanh"`` 对齐）。"""

    constant = math.sqrt(2.0 / math.pi)
    data = [0.0] * a.numel
    for index, value in enumerate(a.data):
        inner = constant * (value + 0.044715 * value ** 3)
        data[index] = 0.5 * value * (1.0 + math.tanh(inner))

    def backward(grad: list[float]) -> None:
        grad_a = [0.0] * a.numel
        for index, value in enumerate(a.data):
            inner = constant * (value + 0.044715 * value ** 3)
            tanh_inner = math.tanh(inner)
            derivative = 0.5 * (1.0 + tanh_inner) + 0.5 * value * (1.0 - tanh_inner ** 2) * constant * (
                1.0 + 3.0 * 0.044715 * value ** 2
            )
            grad_a[index] = grad[index] * derivative
        a._accumulate(grad_a)

    return _wrap(data, a.shape, (a,), backward, "gelu")


def embedding(weight: Tensor, ids: Sequence[int]) -> Tensor:
    """按 token id 查表，返回 ``(len(ids), width)``；反向把梯度散射回对应行。"""
    vocab, width = weight.shape
    count = len(ids)
    data = [0.0] * (count * width)
    for row, token_id in enumerate(ids):
        if not 0 <= token_id < vocab:
            raise ValueError(f"token id {token_id} out of range [0, {vocab})")
        data[row * width : (row + 1) * width] = weight.data[token_id * width : (token_id + 1) * width]

    def backward(grad: list[float]) -> None:
        grad_w = [0.0] * weight.numel
        for row, token_id in enumerate(ids):
            base = token_id * width
            source = row * width
            for index in range(width):
                grad_w[base + index] += grad[source + index]
        weight._accumulate(grad_w)

    return _wrap(data, (count, width), (weight,), backward, "embedding")


def cross_entropy(logits: Tensor, targets: Sequence[int], ignore_index: int = -100) -> Tensor:
    """对非忽略目标求平均交叉熵（与 torch 默认 reduction 一致）。"""

    count_rows, vocab = logits.shape
    if len(targets) != count_rows:
        raise ValueError("targets length must match logits rows")
    per_row = [0.0] * count_rows
    probabilities = [0.0] * logits.numel
    valid = 0
    for row in range(count_rows):
        base = row * vocab
        maximum = max(logits.data[base : base + vocab])
        total = 0.0
        for index in range(vocab):
            value = math.exp(logits.data[base + index] - maximum)
            probabilities[base + index] = value
            total += value
        for index in range(vocab):
            probabilities[base + index] /= total
        target = int(targets[row])
        if target == ignore_index:
            continue
        if not 0 <= target < vocab:
            raise ValueError(f"target {target} out of range [0, {vocab})")
        per_row[row] = -math.log(max(probabilities[base + target], 1e-300))
        valid += 1
    if valid == 0:
        raise ValueError("all targets are ignored")
    loss = sum(per_row) / valid

    def backward(grad: list[float]) -> None:
        factor = grad[0] / valid
        grad_logits = [0.0] * logits.numel
        for row in range(count_rows):
            target = int(targets[row])
            if target == ignore_index:
                continue
            base = row * vocab
            for index in range(vocab):
                grad_logits[base + index] = probabilities[base + index] * factor
            grad_logits[base + target] -= factor
        logits._accumulate(grad_logits)

    return _wrap([loss], (), (logits,), backward, "cross_entropy")


__all__ = [
    "Tensor",
    "no_grad",
    "add",
    "mul",
    "scale",
    "sum_all",
    "reshape",
    "transpose_last",
    "permute",
    "slice_last",
    "matmul",
    "softmax",
    "layer_norm",
    "gelu",
    "embedding",
    "cross_entropy",
    "broadcast_shape",
]
