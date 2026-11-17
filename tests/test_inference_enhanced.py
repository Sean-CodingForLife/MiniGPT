"""N01 KV cache / N02 采样控制增强 的行为测试。

正确性准绳：缓存增量解码必须与朴素全窗口前向逐 token 一致（greedy），
采样路径在固定种子下与参照实现完全一致。
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.models.gpt import GPT, GPTConfig


def make_model(seed: int = 7) -> GPT:
    torch.manual_seed(seed)
    config = GPTConfig(
        block_size=32, vocab_size=24, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    model = GPT(config)
    model.eval()
    return model


def naive_generate(model: GPT, prompt: list[int], n: int) -> list[int]:
    """测试内参照实现：每步重算全窗口（旧行为），greedy。"""
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            cond = torch.tensor([ids[-model.config.block_size :]], dtype=torch.long)
            logits, _ = model(cond)
            ids.append(int(logits[0, -1].argmax().item()))
    return ids


class KVCacheTests(unittest.TestCase):
    """覆盖 KVCacheTests 相关行为。"""
    def test_greedy_matches_naive_reference(self) -> None:
        """验证 `test_greedy_matches_naive_reference` 对应的行为约定。"""
        model = make_model()
        prompt = [3, 1, 4, 1, 5, 9, 2, 6]
        expected = naive_generate(model, prompt, 20)
        actual = model.generate(prompt, max_new_tokens=20, temperature=0.0)
        self.assertEqual(actual, expected)

    def test_greedy_longer_than_block_size(self) -> None:
        """验证 `test_greedy_longer_than_block_size` 对应的行为约定。"""
        model = make_model()
        prompt = [i % 20 for i in range(10)]
        expected = naive_generate(model, prompt, 40)  # 超出 block_size=32，触发滑窗
        actual = model.generate(prompt, max_new_tokens=40, temperature=0.0)
        self.assertEqual(actual, expected)

    def test_sampling_distribution_matches_naive(self) -> None:
        # 缓存路径与全窗口路径存在 ~1e-7 浮点差，贪心 argmax 在近平局时可能翻转，
        # 因此采样正确性以"每步概率分布一致"为准绳，而非逐 token 序列相等。
        """验证 采样 相关行为。"""
        model = make_model()
        prompt = [2, 7, 1]
        ids = torch.tensor([prompt], dtype=torch.long)
        with torch.no_grad():
            full_logits, _ = model(ids)
            _, _, cache = model.forward_with_cache(ids[:, :-1])
            inc_logits, _, _ = model.forward_with_cache(ids[:, -1:], past_kv=cache)
        self.assertTrue(
            torch.allclose(full_logits[:, -1, :], inc_logits[:, -1, :], atol=1e-5)
        )

    def test_greedy_cache_and_no_cache_agree_on_clear_margins(self) -> None:
        """验证 缓存 相关行为。"""
        model = make_model()
        prompt = [3, 1, 4, 1, 5, 9, 2, 6]
        a = model.generate(prompt, max_new_tokens=20, temperature=0.0, use_cache=True)
        b = model.generate(prompt, max_new_tokens=20, temperature=0.0, use_cache=False)
        self.assertEqual(a, b)

    def test_stream_matches_generate(self) -> None:
        """验证 流式输出、生成 相关行为。"""
        model = make_model()
        prompt = [5, 5, 2]
        torch.manual_seed(3)
        full = model.generate(prompt, max_new_tokens=10, temperature=0.7, top_k=5)
        torch.manual_seed(3)
        streamed = list(prompt)
        for item in model.generate_stream(prompt, max_new_tokens=10, temperature=0.7, top_k=5):
            if item["finished"]:
                break
            streamed.append(item["token_id"])
        self.assertEqual(streamed, full)

    def test_forward_with_cache_returns_consistent_logits(self) -> None:
        """验证 缓存、logits 相关行为。"""
        model = make_model()
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            full_logits, _ = model(ids)
            # 先喂前 3 个 token 建缓存，再增量喂后 2 个。
            logits_a, _, cache = model.forward_with_cache(ids[:, :3])
            logits_b, _, cache2 = model.forward_with_cache(ids[:, 3:], past_kv=cache)
        self.assertTrue(torch.allclose(full_logits[:, 3:, :], logits_b, atol=1e-5))
        self.assertEqual(len(cache2), model.config.n_layer)


class SamplingControlTests(unittest.TestCase):
    """覆盖 SamplingControlTests 相关行为。"""
    def test_repetition_penalty_changes_distribution(self) -> None:
        """验证 `test_repetition_penalty_changes_distribution` 对应的行为约定。"""
        model = make_model()
        prompt = [3, 3, 3, 3]
        base = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        penalized = model.generate(
            prompt, max_new_tokens=8, temperature=0.0, repetition_penalty=2.0
        )
        # 强惩罚下不应继续无限重复同一 token。
        self.assertNotEqual(base, penalized)

    def test_no_repeat_ngram_blocks_repeats(self) -> None:
        """验证 `test_no_repeat_ngram_blocks_repeats` 对应的行为约定。"""
        model = make_model()
        prompt = [1, 2, 1, 2, 1, 2]
        out = model.generate(
            prompt, max_new_tokens=12, temperature=0.0, no_repeat_ngram_size=2
        )
        # prompt 自身的重复不受控；只要求"涉及新生成 token"的 n-gram 不重复。
        grams = [tuple(out[i : i + 2]) for i in range(len(prompt) - 1, len(out) - 1)]
        self.assertEqual(len(grams), len(set(grams)), f"生成部分出现重复 2-gram: {out}")

    def test_stop_sequence_truncates(self) -> None:
        """验证 停止 相关行为。"""
        model = make_model()
        prompt = [4, 2]
        # 先无停止生成一段，取其中出现的二元组合作为停止序列。
        long_out = model.generate(prompt, max_new_tokens=20, temperature=0.0)
        stop = long_out[len(prompt) + 2 : len(prompt) + 4]
        self.assertEqual(len(stop), 2)
        out = model.generate(
            prompt, max_new_tokens=20, temperature=0.0, stop_sequences=[stop]
        )
        tail = out[len(prompt) :]
        # 停止序列本身不应完整出现在输出里（在其完成前截断）。
        found = any(tuple(tail[i : i + 2]) == tuple(stop) for i in range(len(tail) - 1))
        self.assertFalse(found, f"停止序列未被截断: {out}")
        self.assertLess(len(out), len(long_out))

    def test_defaults_preserve_old_behavior(self) -> None:
        """验证 `test_defaults_preserve_old_behavior` 对应的行为约定。"""
        model = make_model()
        prompt = [6, 1, 8]
        torch.manual_seed(11)
        a = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=6)
        torch.manual_seed(11)
        b = model.generate(
            prompt, max_new_tokens=10, temperature=0.8, top_k=6,
            repetition_penalty=1.0, no_repeat_ngram_size=0, stop_sequences=None,
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()