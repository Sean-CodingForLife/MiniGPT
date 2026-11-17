"""评测指标函数（归一化、精确匹配、字符 F1、重复率、汇总统计）的单元测试。

这些指标是 SFT/评测报告的口径来源，一旦口径变化，历史对比结果就不可比，
因此本文件锁定各指标的边界行为（空串、全重复、按类别汇总）。
"""

from __future__ import annotations

import unittest

from minigpt.eval.metrics import (
    char_f1,
    exact_match,
    normalize_text,
    repetition_rate,
    summarize_predictions,
)


class MetricTests(unittest.TestCase):
    """验证评测指标在正常样本与空/退化样本上的取值口径。"""

    def test_normalization_and_exact_match(self) -> None:
        """文本归一化需去除空白与标点并统一大小写，精确匹配基于归一化结果而非原始串。"""
        self.assertEqual(normalize_text(" Hello,  World! "), "helloworld")
        self.assertEqual(exact_match("Hello!", " hello "), 1.0)
        self.assertEqual(exact_match("cat", "dog"), 0.0)

    def test_char_f1(self) -> None:
        """字符级 F1 在完全相同为 1、部分重合按位置计分、空预测为 0，防止除零。"""
        self.assertAlmostEqual(char_f1("abc", "abc"), 1.0)
        self.assertAlmostEqual(char_f1("abc", "abd"), 2 / 3)
        self.assertEqual(char_f1("", "abc"), 0.0)

    def test_repetition(self) -> None:
        """重复率需能识别长串单字符退化输出，同时不误报正常的短文本。"""
        self.assertGreater(repetition_rate("aaaaaaaaaa"), 0.5)
        self.assertEqual(repetition_rate("abcd"), 0.0)

    def test_summary_categories(self) -> None:
        """汇总结果需给出样本数、空回答比例，并按 category 分桶以便定位弱项。"""
        records = [
            {"prediction": "北京", "reference": "北京", "category": "in"},
            {"prediction": "", "reference": "上海", "category": "edge"},
        ]
        summary = summarize_predictions(records)
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["empty_rate"], 0.5)
        self.assertIn("in", summary["by_category"])
        self.assertIn("edge", summary["by_category"])


if __name__ == "__main__":
    unittest.main()
