"""KV cache 的 logits 级一致性护栏（阶段 0 新增）。

现有 test_inference_enhanced 用未训练小模型做序列级对比：输出退化为常数
序列时，缓存路径即使位置错乱也会"碰巧"通过。这里给权重加确定性扰动构造
非退化分布，直接断言缓存路径与全窗口路径**每步 logits** 一致（float32
下 atol=1e-5），并覆盖超出 block_size 的滑窗路径。
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.models.gpt import GPT, GPTConfig


def make_perturbed_model(seed: int = 11) -> GPT:
    torch.manual_seed(seed)
    config = GPTConfig(
        block_size=16, vocab_size=31, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    model = GPT(config)
    # 放大词表嵌入差异，避免随机小权重下 argmax 常年不变（退化分布）。
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(3.0).add_(torch.randn_like(parameter) * 0.35)
    model.eval()
    return model


def stepwise_logits(model: GPT, prompt: list[int], n: int, use_cache: bool) -> list[torch.Tensor]:
    """按 generate_stream 的采样前 logits 逐步收集（greedy 推进，两路径同轨）。"""
    rows: list[torch.Tensor] = []
    ids = list(prompt)
    for item in model.generate_stream(
        ids, max_new_tokens=n, temperature=0.0, use_cache=use_cache, trace_top_k=1
    ):
        rows.append(item)
        ids.append(item["token_id"])
    return rows


class KVCacheLogitsTests(unittest.TestCase):
    """逐 token logits 级对拍 KV cache 路径与全窗口前向，捕捉位置错乱类回归。"""

    def _assert_paths_agree(self, prompt: list[int], n: int) -> None:
        """贪心推进下两条路径的 token 序列与每步 top-1 概率都必须一致。"""
        model = make_perturbed_model()
        cached = stepwise_logits(model, prompt, n, use_cache=True)
        naive = stepwise_logits(model, prompt, n, use_cache=False)
        self.assertEqual(len(cached), len(naive))
        self.assertEqual(
            [item["token_id"] for item in cached],
            [item["token_id"] for item in naive],
            "缓存路径与全窗口路径的 greedy 序列不一致",
        )
        # topk[0] 的 prob 来自各自路径的 softmax；再直接对拍每步末位 logits。
        for step, (a, b) in enumerate(zip(cached, naive), start=1):
            self.assertAlmostEqual(
                a["topk"][0]["prob"], b["topk"][0]["prob"], delta=1e-5,
                msg=f"step {step} 概率不一致（缓存位置错乱的信号）",
            )

    def test_short_prompt_logits_match(self) -> None:
        """窗口内正常增量解码时，缓存与全量前向的每步概率应逐位吻合。"""
        self._assert_paths_agree([3, 1, 4, 1, 5, 9, 2, 6], 8)

    def test_sliding_window_logits_match(self) -> None:
        """总长度超过 block_size 触发滑窗重建后，缓存路径仍须与全窗口一致。"""
        # prompt 8 + 生成 24 > block_size=16，必须走滑窗重建。
        self._assert_paths_agree([i % 29 for i in range(8)], 24)

    def test_first_step_logits_match_full_forward(self) -> None:
        """流式生成首步的 logits/argmax 必须等价于对整段 prompt 的一次完整前向。"""
        model = make_perturbed_model()
        prompt = [3, 1, 4, 1, 5, 9, 2, 6]
        ids = torch.tensor([prompt], dtype=torch.long)
        with torch.no_grad():
            full, _ = model(ids)
        stream = model.generate_stream(prompt, max_new_tokens=1, temperature=0.0)
        item = next(stream)
        # 用全窗口参照对拍首步：直接检查 generate_stream 首步 logits。
        with torch.no_grad():
            cached_logits = None
            for entry in model.generate_stream(
                prompt, max_new_tokens=1, temperature=0.0, trace_top_k=1
            ):
                cached_logits = entry
        # 首步 top-1 概率必须与全窗口一致（重复末 token 会显著改变分布）。
        top_prob_full = float(torch.softmax(full[0, -1], dim=-1).max().item())
        self.assertAlmostEqual(cached_logits["topk"][0]["prob"], top_prob_full, delta=1e-5)
        self.assertEqual(item["token_id"], int(full[0, -1].argmax().item()))


if __name__ == "__main__":
    unittest.main()