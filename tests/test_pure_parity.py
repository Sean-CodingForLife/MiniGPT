"""Pure backend parity: same weights, same logits/loss/gradients as PyTorch."""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.models.gpt import GPT, GPTConfig
from minigpt.pure import tensor as T
from minigpt.pure.gpt import PureGPT, PureGPTConfig

ROOT = Path(__file__).resolve().parents[1]
REAL_CHECKPOINT = ROOT / "runs" / "demo-en-sft" / "best.pt"


def torch_state(model: torch.nn.Module) -> dict[str, tuple[tuple[int, ...], list[float]]]:
    return {
        name: (tuple(value.shape), value.detach().reshape(-1).tolist())
        for name, value in model.state_dict().items()
    }


class PureParityTests(unittest.TestCase):
    """覆盖 PureParityTests 相关行为。"""
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = GPTConfig(
            block_size=8, vocab_size=13, n_layer=2, n_head=2, n_embd=8, dropout=0.0
        )
        self.torch_model = GPT(self.config).double().eval()
        self.pure_model = PureGPT(PureGPTConfig.from_dict(self.config.to_dict()))
        skipped = self.pure_model.load_state_dict(torch_state(self.torch_model))
        self.assertEqual(skipped, ["lm_head.weight"])
        self.idx = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 0]]
        self.targets = [[2, 3, 4, 5, 6], [4, 3, 2, 1, 0]]

    def test_parameter_count_matches(self) -> None:
        """验证 `test_parameter_count_matches` 对应的行为约定。"""
        self.assertEqual(
            self.pure_model.num_parameters(),
            sum(parameter.numel() for parameter in self.torch_model.parameters()),
        )

    def test_logits_and_loss_match(self) -> None:
        """验证 logits、loss 相关行为。"""
        logits, loss = self.pure_model.forward(self.idx, self.targets)
        with torch.no_grad():
            torch_logits, torch_loss = self.torch_model(torch.tensor(self.idx), torch.tensor(self.targets))
        expected = torch_logits.reshape(-1).tolist()
        worst = max(abs(left - right) for left, right in zip(logits.data, expected))
        self.assertLess(worst, 1e-10)
        self.assertAlmostEqual(loss.item(), float(torch_loss), places=12)

    def test_gradients_match(self) -> None:
        """验证 `test_gradients_match` 对应的行为约定。"""
        weights = [((index * 37) % 13) / 13.0 - 0.5 for index in range(2 * 5 * self.config.vocab_size)]
        loss = T.sum_all(T.mul(self.pure_model.forward(self.idx)[0], T.Tensor(weights, (2, 5, self.config.vocab_size))))
        loss.backward()

        torch_logits, _ = self.torch_model(torch.tensor(self.idx))
        torch_weights = torch.tensor(weights, dtype=torch.float64).reshape(torch_logits.shape)
        (torch_logits * torch_weights).sum().backward()

        torch_params = dict(self.torch_model.named_parameters())
        worst = 0.0
        for name, parameter in self.pure_model.params.items():
            expected = torch_params[name].grad.reshape(-1).tolist()
            worst = max(worst, max(abs(left - right) for left, right in zip(parameter.grad.data, expected)))
        self.assertLess(worst, 1e-10)

    def test_loss_gradient_matches(self) -> None:
        """验证 loss 相关行为。"""
        loss = self.pure_model.forward(self.idx, self.targets)[1]
        loss.backward()
        _, torch_loss = self.torch_model(torch.tensor(self.idx), torch.tensor(self.targets))
        torch_loss.backward()
        torch_params = dict(self.torch_model.named_parameters())
        worst = max(
            max(abs(left - right) for left, right in zip(parameter.grad.data, torch_params[name].grad.reshape(-1).tolist()))
            for name, parameter in self.pure_model.params.items()
        )
        self.assertLess(worst, 1e-10)

    def test_greedy_generation_matches(self) -> None:
        # 双后端逐 token 对齐用全窗口路径（use_cache=False）：
        # 缓存路径有 ~1e-7 float32 噪声，近平局 argmax 可能翻转；
        # 缓存自身的正确性由 test_inference_enhanced 的一致性测试保证。
        """验证 生成 相关行为。"""
        prompt = [3, 1, 4, 1, 5]
        pure_ids = self.pure_model.generate(prompt, max_new_tokens=5, temperature=0, top_k=40, top_p=1.0)
        with torch.no_grad():
            torch_ids = self.torch_model.generate(
                prompt, max_new_tokens=5, temperature=0, top_k=40, top_p=1.0, use_cache=False
            )
        self.assertEqual(pure_ids, torch_ids)

    @unittest.skipUnless(REAL_CHECKPOINT.is_file(), "runs/demo-en-sft/best.pt is not available")
    def test_real_checkpoint_logits_match(self) -> None:
        """验证 checkpoint、logits 相关行为。"""
        from minigpt.inference import load_bundle

        bundle = load_bundle(REAL_CHECKPOINT, device="cpu")
        model = bundle.model
        pure = PureGPT(PureGPTConfig.from_dict(model.config.to_dict()))
        skipped = pure.load_state_dict(torch_state(model))
        self.assertIn("lm_head.weight", skipped)
        prompt = bundle.tokenizer.encode("Q: Hello\nA: ")[:12]
        logits, _ = pure.forward([prompt])
        with torch.no_grad():
            torch_logits, _ = model(torch.tensor([prompt]))
        expected = torch_logits.reshape(-1).tolist()
        worst = max(abs(left - right) for left, right in zip(logits.data, expected))
        self.assertLess(worst, 1e-4)
        self.assertEqual(pure.config.vocab_size, bundle.tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
