"""pure 引擎生成过程追踪：注意力/隐层/采样契约 + 随机流不变 + iter_trace 端到端。

背景：生成可视化页曾仅支持 torch 检查点；pure 前向本就计算注意力权重，
补齐暴露后 .mgckpt 也能走 iter_trace（Web NDJSON 与静态导出共用）。
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from minigpt import sampling
from minigpt.pure.backend import PureBackend
from minigpt.pure.gpt import PureGPT, PureGPTConfig
from minigpt.render import iter_trace
from minigpt.tokenization.char import CharTokenizer


def make_model() -> PureGPT:
    config = PureGPTConfig(block_size=16, vocab_size=20, n_layer=2, n_head=2, n_embd=16)
    return PureGPT(config, seed=7)


class PureStreamTraceTests(unittest.TestCase):
    """覆盖 PureStreamTraceTests 相关行为。"""
    def setUp(self):
        self.model = make_model()

    def test_attention_and_hidden_contract(self):
        """验证 注意力、隐藏状态 相关行为。"""
        items = list(self.model.generate_stream(
            [1, 2, 3], max_new_tokens=2, temperature=0.0,
            trace_attention=True, trace_top_k=3,
        ))
        self.assertEqual(len(items), 2)
        for step, item in enumerate(items, start=1):
            attention = item["attention"]
            self.assertEqual(len(attention), 2)  # n_layer
            for layer in attention:
                self.assertEqual(len(layer), 2)  # n_head
                t = 3 + step - 1
                self.assertEqual(len(layer[0]), t)
                for row_index in range(t):
                    row = layer[0][row_index]
                    self.assertEqual(len(row), t)
                    self.assertAlmostEqual(sum(row), 1.0, places=5)
                    for col_index, value in enumerate(row):
                        if col_index > row_index:
                            self.assertEqual(value, 0.0)  # 因果遮罩
            self.assertEqual(len(item["hidden"]), 16)  # n_embd
            self.assertEqual(len(item["topk"]), 3)
            self.assertEqual(item["context_len"], 3 + step - 1)

    def test_sampling_trace_contract(self):
        """验证 采样、追踪 相关行为。"""
        items = list(self.model.generate_stream(
            [1, 2, 3], max_new_tokens=2, temperature=0.8, top_k=5, top_p=0.9,
            trace_sampling=True, trace_candidates=8, rng=random.Random(3),
        ))
        for item in items:
            trace = item["sampling"]
            for key in (
                "mode", "filters_applied", "temperature", "top_k", "top_p",
                "vocab_size", "trace_candidates", "top_k_threshold_logit",
                "nucleus_mass", "nucleus_count", "nucleus_segments", "entropy_bits",
                "top1_prob", "u", "chosen_id", "chosen_prob", "chosen_cum_low",
                "chosen_cum_high", "candidates",
            ):
                self.assertIn(key, trace)
            self.assertEqual(trace["mode"], "sample")
            self.assertEqual(trace["vocab_size"], 20)
            self.assertEqual(trace["chosen_id"], item["token_id"])
            self.assertTrue(any(candidate["chosen"] for candidate in trace["candidates"]))
            self.assertGreaterEqual(trace["chosen_cum_high"], trace["chosen_cum_low"])
            # 与 torch 同语义：nucleus_mass 是重归一前的保留质量；
            # 段概率是重归一后的最终分布，和为 1。
            self.assertLessEqual(trace["nucleus_mass"], 1.0 + 1e-6)
            self.assertAlmostEqual(sum(
                segment["prob"] for segment in trace["nucleus_segments"]
            ), 1.0, places=4)

    def test_greedy_trace_has_no_draw(self):
        """验证 追踪 相关行为。"""
        items = list(self.model.generate_stream(
            [1, 2, 3], max_new_tokens=1, temperature=0.0,
            trace_sampling=True, trace_candidates=5,
        ))
        trace = items[0]["sampling"]
        self.assertEqual(trace["mode"], "greedy")
        self.assertIsNone(trace["u"])
        self.assertFalse(trace["filters_applied"])

    def test_trace_does_not_change_token_stream(self):
        """验证 追踪、token、流式输出 相关行为。"""
        kwargs = dict(max_new_tokens=6, temperature=0.8, top_k=5, top_p=0.9, eos_id=None)
        plain = [item["token_id"] for item in self.model.generate_stream(
            [1, 2, 3], rng=random.Random(11), **kwargs)]
        traced = [item["token_id"] for item in self.model.generate_stream(
            [1, 2, 3], rng=random.Random(11), trace_attention=True,
            trace_sampling=True, trace_candidates=8, **kwargs)]
        self.assertEqual(plain, traced)

    def test_no_trace_keeps_legacy_fields(self):
        """验证 追踪 相关行为。"""
        items = list(self.model.generate_stream([1, 2, 3], max_new_tokens=2, temperature=0.0))
        for item in items:
            self.assertNotIn("attention", item)
            self.assertNotIn("hidden", item)
            self.assertNotIn("sampling", item)


class SampleWithTraceKernelTests(unittest.TestCase):
    """覆盖 SampleWithTraceKernelTests 相关行为。"""
    def test_matches_sample_from_logits(self):
        """验证 采样、logits 相关行为。"""
        logits = [0.3, 1.2, -0.5, 2.0, 0.7, -1.1, 0.1, 1.5]
        for temperature, top_k, top_p in ((0.8, 3, 0.9), (1.2, 0, 1.0), (0.5, 4, 0.7)):
            draw = 0.37
            token_a, probs_a = sampling.sample_from_logits(logits, temperature, top_k, top_p, draw)
            token_b, probs_b, trace = sampling.sample_with_trace(logits, temperature, top_k, top_p, draw, trace_candidates=5)
            self.assertEqual(token_a, token_b)
            self.assertEqual(probs_a, probs_b)
            self.assertIsNotNone(trace)
        greedy_a, _ = sampling.sample_from_logits(logits, 0.0, 3, 0.9, 0.0)
        greedy_b, _, trace = sampling.sample_with_trace(logits, 0.0, 3, 0.9, None, trace_candidates=5)
        self.assertEqual(greedy_a, greedy_b)
        self.assertEqual(trace["mode"], "greedy")

    def test_zero_candidates_skips_trace(self):
        """验证 追踪 相关行为。"""
        token, probs, trace = sampling.sample_with_trace([0.1, 0.9], 0.8, 0, 1.0, 0.5, trace_candidates=0)
        self.assertIsNone(trace)
        self.assertEqual(len(probs), 2)


class PureIterTraceTests(unittest.TestCase):
    """覆盖 PureIterTraceTests 相关行为。"""
    def test_iter_trace_end_to_end(self):
        """验证 追踪 相关行为。"""
        model = make_model()
        # 词表规模与模型 vocab_size 一致（含特殊 token），避免候选越界。
        tokenizer = CharTokenizer.build("abcdefghijklmnopq")
        bundle = SimpleNamespace(
            backend=PureBackend(SimpleNamespace(seed=7)),
            model=model,
            tokenizer=tokenizer,
            payload={"stage": "pretrain"},
        )
        events = list(iter_trace(
            bundle, prompt="abc", max_new_tokens=3, temperature=0.8,
            top_k=5, top_p=0.9, trace_top_k=4, max_context=8, template="raw",
        ))
        types = [event["type"] for event in events]
        self.assertEqual(types[0], "meta")
        self.assertEqual(types[-1], "final")
        steps = [event["data"] for event in events if event["type"] == "step"]
        self.assertEqual(len(steps), 3)
        for record in steps:
            self.assertEqual(len(record["attn_last"]), 2)
            self.assertEqual(len(record["attn_last"][0]), 2)
            self.assertIn("norm", record["hidden"])
            self.assertTrue(record["sampling"]["candidates"])
            self.assertTrue(all("token" in candidate for candidate in record["sampling"]["candidates"]))
            self.assertTrue(record["topk"])
        final = events[-1]["data"]
        self.assertEqual(len(final["attention"]), 2)
        self.assertEqual(len(final["attention"][0]), 2)
        self.assertEqual(final["steps"], 3)
        self.assertTrue(final["context"])


if __name__ == "__main__":
    unittest.main()
