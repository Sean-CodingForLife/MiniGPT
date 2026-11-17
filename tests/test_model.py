"""针对 `test_model.py` 的单元测试与回归保护。"""
from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.models.gpt import GPT, GPTConfig


class ModelTests(unittest.TestCase):
    """覆盖 ModelTests 相关行为。"""
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.config = GPTConfig(
            block_size=16,
            vocab_size=68,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            bias=False,
            tie_weights=True,
        )
        self.model = GPT(self.config).eval()

    def test_shapes_and_loss(self) -> None:
        """验证 loss 相关行为。"""
        idx = torch.randint(0, self.config.vocab_size, (2, 8))
        targets = torch.randint(0, self.config.vocab_size, (2, 8))
        logits, loss = self.model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 8, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(float(loss.detach())))

    def test_causal_property(self) -> None:
        """验证 `test_causal_property` 对应的行为约定。"""
        idx = torch.randint(0, self.config.vocab_size, (2, 8))
        changed = idx.clone()
        changed[:, -1] = (changed[:, -1] + 1) % self.config.vocab_size
        with torch.no_grad():
            logits_a, _ = self.model(idx)
            logits_b, _ = self.model(changed)
        self.assertTrue(torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6))

    def test_attention_is_causal_and_normalized(self) -> None:
        """验证 注意力 相关行为。"""
        idx = torch.randint(0, self.config.vocab_size, (1, 6))
        with torch.no_grad():
            _, _, attns = self.model(idx, return_attns=True)
        self.assertEqual(len(attns), self.config.n_layer)
        for attn in attns:
            self.assertEqual(tuple(attn.shape), (1, self.config.n_head, 6, 6))
            upper = torch.triu(attn[0, 0], diagonal=1)
            self.assertLess(float(upper.abs().max()), 1e-6)
            self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5))

    def test_weight_tying_and_no_bias(self) -> None:
        """验证 `test_weight_tying_and_no_bias` 对应的行为约定。"""
        self.assertIs(self.model.lm_head.weight, self.model.transformer["wte"].weight)
        for name, parameter in self.model.named_parameters():
            self.assertNotIn(".bias", name)
            self.assertIsNotNone(parameter)

    def test_initial_loss_close_to_log_vocab(self) -> None:
        """验证 loss、词表 相关行为。"""
        idx = torch.randint(0, self.config.vocab_size, (8, 16))
        targets = torch.randint(0, self.config.vocab_size, (8, 16))
        with torch.no_grad():
            _, loss = self.model(idx, targets)
        expected = math.log(self.config.vocab_size)
        self.assertAlmostEqual(float(loss), expected, delta=0.3)

    def test_generate_length(self) -> None:
        """验证 生成 相关行为。"""
        prompt = [1, 2, 3]
        generated = self.model.generate(prompt, max_new_tokens=5, temperature=0)
        self.assertEqual(len(generated), len(prompt) + 5)

    def test_generate_empty_prompt_raises(self) -> None:
        """验证 生成 相关行为。"""
        with self.assertRaises(ValueError):
            self.model.generate([], max_new_tokens=4)

    def test_generate_stream_empty_prompt_raises(self) -> None:
        # generate_stream 必须与 generate 一样抛 ValueError，而不是 IndexError。
        """验证 生成、流式输出 相关行为。"""
        with self.assertRaises(ValueError):
            list(self.model.generate_stream([], max_new_tokens=4))

    def test_generate_stream_reports_probabilities_and_attention(self) -> None:
        """验证 生成、流式输出、注意力 相关行为。"""
        prompt = [1, 2, 3]
        steps = list(
            self.model.generate_stream(
                prompt,
                max_new_tokens=3,
                temperature=0,
                eos_id=None,
                trace_top_k=3,
                trace_attention=True,
            )
        )
        self.assertEqual(len(steps), 3)
        for index, item in enumerate(steps, start=1):
            self.assertEqual(item["step"], index)
            self.assertGreaterEqual(item["prob"], 0.0)
            self.assertLessEqual(item["prob"], 1.0)
            self.assertEqual(len(item["topk"]), 3)
            probs = [candidate["prob"] for candidate in item["topk"]]
            self.assertEqual(probs, sorted(probs, reverse=True))
            context_len = len(prompt) + index - 1
            self.assertEqual(item["context_len"], context_len)
            self.assertEqual(len(item["attention"]), self.config.n_layer)
            for layer in item["attention"]:
                self.assertEqual(len(layer), self.config.n_head)
                for matrix in layer:
                    self.assertEqual(len(matrix), context_len)
                    self.assertEqual(len(matrix[0]), context_len)
            self.assertEqual(len(item["hidden"]), self.config.n_embd)

    def test_generate_stream_matches_generate(self) -> None:
        """验证 生成、流式输出 相关行为。"""
        prompt = [4, 5, 6]
        torch.manual_seed(7)
        generated = self.model.generate(prompt, max_new_tokens=4, temperature=0)
        torch.manual_seed(7)
        steps = list(
            self.model.generate_stream(
                prompt,
                max_new_tokens=4,
                temperature=0,
                eos_id=None,
            )
        )
        self.assertEqual(generated, prompt + [item["token_id"] for item in steps])

    def test_parameter_count_matches_contract(self) -> None:
        """验证 `test_parameter_count_matches_contract` 对应的行为约定。"""
        large = GPTConfig(block_size=128, vocab_size=68, n_layer=4, n_head=4, n_embd=128, dropout=0.1)
        model = GPT(large)
        self.assertEqual(model.num_parameters(), 812672)
        entry = GPTConfig(block_size=256, vocab_size=68, n_layer=6, n_head=6, n_embd=384, dropout=0.1)
        self.assertEqual(GPT(entry).num_parameters(), 10746240)


if __name__ == "__main__":
    unittest.main()
