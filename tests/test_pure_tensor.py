"""Pure tensor/autograd tests: forward values, broadcasting, gradients."""

from __future__ import annotations

import math
import unittest

from minigpt.pure import tensor as T


def _values(count: int, seed: int = 0) -> list[float]:
    # deterministic, non-trivial values in [-1, 1)
    return [((seed + index * 37) % 29) / 14.0 - 1.0 for index in range(count)]


def numeric_grad(fn, tensors, epsilon: float = 1e-6) -> list[list[float]]:
    """Central differences for a scalar-valued function of the given tensors."""

    result = []
    for tensor in tensors:
        flat = []
        for index in range(tensor.numel):
            original = tensor.data[index]
            tensor.data[index] = original + epsilon
            plus = fn().item()
            tensor.data[index] = original - epsilon
            minus = fn().item()
            tensor.data[index] = original
            flat.append((plus - minus) / (2 * epsilon))
        result.append(flat)
    return result


class PureTensorTests(unittest.TestCase):
    """覆盖 PureTensorTests 相关行为。"""
    def assert_gradclose(self, analytic: list[float], numeric: list[float], tolerance: float = 1e-5) -> None:
        self.assertEqual(len(analytic), len(numeric))
        for left, right in zip(analytic, numeric):
            self.assertLessEqual(abs(left - right), tolerance * max(1.0, abs(right)), (left, right))

    def test_add_broadcast_forward(self) -> None:
        """验证 `test_add_broadcast_forward` 对应的行为约定。"""
        a = T.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
        b = T.Tensor([10.0, 20.0, 30.0], (3,))
        out = T.add(a, b)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.data, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
        with self.assertRaises(ValueError):
            T.add(T.Tensor([1.0, 2.0], (2,)), T.Tensor([1.0, 2.0, 3.0], (3,)))

    def test_add_broadcast_grad(self) -> None:
        """验证 梯度 相关行为。"""
        a = T.Tensor(_values(6), (2, 3), requires_grad=True)
        b = T.Tensor(_values(3, seed=5), (3,), requires_grad=True)
        T.sum_all(T.add(a, b)).backward()
        self.assertEqual(a.grad.data, [1.0] * 6)
        self.assertEqual(b.grad.data, [2.0, 2.0, 2.0])

    def test_matmul_batched_and_grad(self) -> None:
        """验证 梯度 相关行为。"""
        a = T.Tensor(_values(12), (2, 2, 3), requires_grad=True)
        b = T.Tensor(_values(24, seed=3), (2, 3, 4), requires_grad=True)

        def build():
            return T.matmul(a, b)

        out = build()
        self.assertEqual(out.shape, (2, 2, 4))

        def loss():
            return T.sum_all(build())

        loss().backward()
        numeric = numeric_grad(loss, [a, b])
        self.assert_gradclose(a.grad.data, numeric[0])
        self.assert_gradclose(b.grad.data, numeric[1])

    def test_softmax_matches_manual(self) -> None:
        """验证 MANUAL 相关行为。"""
        values = T.Tensor([1.0, 2.0, 3.0], (1, 3), requires_grad=True)
        out = T.softmax(values)
        total = sum(math.exp(value) for value in [1.0, 2.0, 3.0])
        expected = [math.exp(value) / total for value in [1.0, 2.0, 3.0]]
        for left, right in zip(out.data, expected):
            self.assertAlmostEqual(left, right, places=12)

    def test_softmax_grad(self) -> None:
        """验证 梯度 相关行为。"""
        values = T.Tensor(_values(6), (2, 3), requires_grad=True)
        coefficients = T.Tensor([1.0, 2.0, -1.0, 0.5, 0.0, 3.0], (2, 3))

        def loss():
            return T.sum_all(T.mul(T.softmax(values), coefficients))

        loss().backward()
        numeric = numeric_grad(loss, [values])
        self.assert_gradclose(values.grad.data, numeric[0])

    def test_layer_norm_matches_manual(self) -> None:
        """验证 MANUAL 相关行为。"""
        x = T.Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        weight = T.Tensor([2.0, 0.5], (2,))
        out = T.layer_norm(x, weight, eps=1e-5)
        row0 = [1.0, 2.0]
        mean0 = sum(row0) / 2
        var0 = sum((value - mean0) ** 2 for value in row0) / 2
        expected0 = [(value - mean0) / math.sqrt(var0 + 1e-5) * weight.data[index] for index, value in enumerate(row0)]
        for left, right in zip(out.data[:2], expected0):
            self.assertAlmostEqual(left, right, places=12)

    def test_layer_norm_grad(self) -> None:
        """验证 梯度 相关行为。"""
        x = T.Tensor(_values(8), (2, 4), requires_grad=True)
        weight = T.Tensor(_values(4, seed=7), (4,), requires_grad=True)
        out = T.layer_norm(x, weight)

        def loss():
            return T.sum_all(T.mul(T.layer_norm(x, weight), T.Tensor(_values(8, seed=11), (2, 4))))

        loss().backward()
        numeric = numeric_grad(loss, [x, weight])
        self.assert_gradclose(x.grad.data, numeric[0])
        self.assert_gradclose(weight.grad.data, numeric[1])

    def test_gelu_grad(self) -> None:
        """验证 梯度 相关行为。"""
        x = T.Tensor(_values(5), (5,), requires_grad=True)
        out = T.gelu(x)

        def loss():
            return T.sum_all(T.mul(T.gelu(x), T.Tensor([1.0, -1.0, 2.0, 0.5, -2.0], (5,))))

        loss().backward()
        numeric = numeric_grad(loss, [x])
        self.assert_gradclose(x.grad.data, numeric[0])

    def test_cross_entropy_ignore_index(self) -> None:
        """验证 `test_cross_entropy_ignore_index` 对应的行为约定。"""
        logits = T.Tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], (2, 3), requires_grad=True)
        loss = T.cross_entropy(logits, [2, -100])
        probabilities = T.softmax(T.Tensor([0.0, 1.0, 2.0], (1, 3))).data
        expected = -math.log(probabilities[2])
        self.assertAlmostEqual(loss.item(), expected, places=12)

        def scalar_loss():
            return T.cross_entropy(logits, [2, -100])

        scalar_loss().backward()
        numeric = numeric_grad(scalar_loss, [logits])
        self.assert_gradclose(logits.grad.data, numeric[0])

    def test_embedding_grad_accumulates_repeats(self) -> None:
        """验证 梯度 相关行为。"""
        weight = T.Tensor(_values(12), (4, 3), requires_grad=True)
        out = T.embedding(weight, [0, 2, 0])
        self.assertEqual(out.shape, (3, 3))
        T.sum_all(out).backward()
        self.assertEqual(weight.grad.data[:3], [2.0, 2.0, 2.0])
        self.assertEqual(weight.grad.data[3:6], [0.0, 0.0, 0.0])
        self.assertEqual(weight.grad.data[6:9], [1.0, 1.0, 1.0])

    def test_reshape_permute_slice_roundtrip(self) -> None:
        """验证 `test_reshape_permute_slice_roundtrip` 对应的行为约定。"""
        original = _values(24)
        x = T.Tensor(original, (2, 3, 4))
        permuted = T.permute(x, (2, 0, 1))
        self.assertEqual(permuted.shape, (4, 2, 3))
        back = T.permute(permuted, (1, 2, 0))
        self.assertEqual(back.shape, (2, 3, 4))
        self.assertEqual(back.data, original)

        reshaped = T.reshape(x, (6, 4))
        self.assertEqual(reshaped.data, original)
        sliced = T.slice_last(reshaped, 1, 3)
        self.assertEqual(sliced.shape, (6, 2))
        self.assertEqual(sliced.data[:2], original[1:3])

    def test_transpose_grad(self) -> None:
        """验证 梯度 相关行为。"""
        x = T.Tensor(_values(6), (2, 3), requires_grad=True)
        out = T.transpose_last(x)

        def loss():
            return T.sum_all(T.mul(T.transpose_last(x), T.Tensor(_values(6, seed=2), (3, 2))))

        loss().backward()
        numeric = numeric_grad(loss, [x])
        self.assert_gradclose(x.grad.data, numeric[0])

    def test_no_grad_disables_graph(self) -> None:
        """验证 梯度 相关行为。"""
        x = T.Tensor(_values(4), (4,), requires_grad=True)
        with T.no_grad():
            out = T.mul(x, 2.0)
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out._backward)

    def test_shape_validation(self) -> None:
        """验证 校验 相关行为。"""
        with self.assertRaises(ValueError):
            T.Tensor([1.0, 2.0], (3,))
        with self.assertRaises(ValueError):
            T.matmul(T.Tensor([1.0, 2.0], (1, 2)), T.Tensor([1.0, 2.0, 3.0], (1, 3)))
        with self.assertRaises(ValueError):
            T.reshape(T.Tensor([1.0, 2.0], (2,)), (3,))

    def test_adamw_minimizes_quadratic(self) -> None:
        """验证 `test_adamw_minimizes_quadratic` 对应的行为约定。"""
        from minigpt.pure.adamw import AdamW

        parameter = T.Tensor([0.0], (1,), requires_grad=True)
        optimizer = AdamW([parameter], lr=0.1, weight_decay=0.0)
        target = 3.0
        for _ in range(250):
            optimizer.zero_grad()
            difference = T.add(parameter, -target)
            T.sum_all(T.mul(difference, difference)).backward()
            optimizer.step()
        self.assertAlmostEqual(parameter.item(), target, delta=0.05)

    def test_clip_grad_norm(self) -> None:
        """验证 裁剪、梯度 相关行为。"""
        from minigpt.pure.adamw import clip_grad_norm

        parameter = T.Tensor([0.0, 0.0], (2,), requires_grad=True)
        T.sum_all(T.mul(parameter, 1.0)).backward()
        norm = clip_grad_norm([parameter], 1.0)
        self.assertAlmostEqual(norm, math.sqrt(2.0), places=12)
        self.assertAlmostEqual(parameter.grad.data[0], 1.0 / math.sqrt(2.0), places=9)
        self.assertAlmostEqual(parameter.grad.data[1], 1.0 / math.sqrt(2.0), places=9)


if __name__ == "__main__":
    unittest.main()
