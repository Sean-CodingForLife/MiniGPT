"""交互式可视化脚本（visualize.py）的追踪数据与 HTML 产物测试。

守住三件事：逐步生成时 steps/final 的上下文长度与注意力层数正确；
导出的 HTML 完全自包含（不含任何 http(s) 外部资源），可离线打开；
隐藏状态摘要里的范数按原始数值计算，不被展示格式化影响。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

import visualize
from minigpt.models.gpt import GPT, GPTConfig
from minigpt.tokenization.char import CharTokenizer
from visualize import _hidden_summary, build_html, collect_trace


class VisualizationTests(unittest.TestCase):
    """用最小 GPT 模型验证追踪采集与自包含页面渲染。"""

    def setUp(self) -> None:
        # 固定种子 + dropout=0 的极小模型：结果确定，跑得快，注意力形状足够断言。
        torch.manual_seed(11)
        self.tokenizer = CharTokenizer.build("hello mini gpt visualization\n" * 4)
        self.model = GPT(
            GPTConfig(
                block_size=12,
                vocab_size=self.tokenizer.vocab_size,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
            )
        ).eval()
        # 用 SimpleNamespace 冒充推理 bundle，避免依赖权重文件或真实 backend。
        backend = SimpleNamespace(device_info=lambda: {"device": "cpu"})
        self.bundle = SimpleNamespace(
            model=self.model,
            tokenizer=self.tokenizer,
            backend=backend,
            payload={"stage": "pretrain"},
        )

    def test_collect_trace_and_render_self_contained_html(self) -> None:
        """生成 2 个 token 后上下文应增长到 3，页面内联数据且不引用任何外链资源。"""
        data = collect_trace(
            self.bundle,
            prompt="he",
            max_new_tokens=2,
            temperature=0,
            top_k=5,
            top_p=1.0,
            trace_top_k=3,
            max_context=8,
            template="raw",
        )
        self.assertEqual(len(data["steps"]), 2)
        self.assertEqual(data["final"]["step"], 2)
        self.assertEqual(len(data["final"]["context"]), 3)
        self.assertEqual(len(data["final"]["attention"]), 1)
        self.assertEqual(len(data["final"]["attention"][0]), 2)
        page = build_html(data)
        self.assertIn("<!doctype html>", page)
        self.assertIn("id=\"heatmap\"", page)
        self.assertIn("trace-data", page)
        self.assertIn("逐 token 概率", page)
        self.assertIn("注意力热图", page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertIn(data["prompt"], page)
        self.assertIn("he", page)
        hidden = _hidden_summary([1.0, -2.0, 0.5])
        self.assertAlmostEqual(hidden["norm"], (1.0 + 4.0 + 0.25) ** 0.5)


if __name__ == "__main__":
    unittest.main()
