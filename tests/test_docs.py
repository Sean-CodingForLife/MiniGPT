"""文档体系守卫：README 保持精简、MANUAL 是唯一长手册、核心模块都有 docstring。

文档漂移是这类教学仓库最常见的腐化点（占位符残留、长文散落到 docs/、
模块改名后文档没跟上、示例指标与真实 demo 不符），因此用静态断言锁定。
"""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    """校验文档结构、必备章节、命名约定与核心模块文档完整性。"""

    def test_readme_is_short_and_links_docs(self) -> None:
        """README 只允许做入口页（≤120 行），并必须链接到 MANUAL 与 CONSTRUCTION。"""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 120)
        self.assertIn("MANUAL.md", text)
        self.assertIn("CONSTRUCTION.md", text)

    def test_manual_is_the_single_long_manual(self) -> None:
        """MANUAL.md 需覆盖从安装到故障排查的全部章节，且不能残留未填写的占位符。"""
        text = (ROOT / "MANUAL.md").read_text(encoding="utf-8")
        required = [
            "完全手册",
            "零基础",
            "安装",
            "快速开始",
            "预训练",
            "SFT",
            "评估",
            "对话",
            "Web",
            "checkpoint",
            "可扩展",
            "故障排查",
            "术语",
            "附录",
            "vocab_extra",
            "observe.py",
            "当前能力",
            "模型打包",
            "MiniGPT Bundle",
            "生成过程可视化",
            "SPA",
        ]
        missing = [word for word in required if word not in text]
        self.assertEqual(missing, [], f"MANUAL.md missing sections: {missing}")
        self.assertNotIn("LEAD-FILL", text)
        self.assertIn("screenshots/dashboard.png", text)

    def test_readme_has_no_placeholders(self) -> None:
        """README 中不能出现 LEAD-FILL 之类的模板占位标记。"""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("LEAD-FILL", text)

    def test_no_scattered_long_docs_folder(self) -> None:
        """长文档统一放 MANUAL.md，docs/ 下不应再堆积 Markdown 副本。"""
        docs = ROOT / "docs"
        if docs.is_dir():
            self.assertEqual(list(docs.rglob("*.md")), [])

    def test_observe_renamed_to_avoid_stdlib_shadowing(self) -> None:
        """脚本必须叫 observe.py；叫 inspect.py 会遮蔽标准库同名模块。"""
        self.assertTrue((ROOT / "observe.py").is_file())
        self.assertFalse((ROOT / "inspect.py").exists())

    def test_core_modules_have_docstrings(self) -> None:
        """核心模块必须有模块级 docstring，便于 help() 与文档生成。"""
        modules = [
            "minigpt.config",
            "minigpt.registry",
            "minigpt.models.gpt",
            "minigpt.training.trainer",
            "minigpt.training.checkpoint",
            "minigpt.training.callbacks",
            "minigpt.backends.base",
            "minigpt.backends.torch_backend",
            "minigpt.inference",
            "minigpt.eval.metrics",
        ]
        for name in modules:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError:
                # 无 torch 环境：torch 专属模块改查源码 docstring。
                source = (ROOT / (name.replace(".", "/") + ".py")).read_text(encoding="utf-8")
                doc = ast.get_docstring(ast.parse(source))
                self.assertTrue((doc or "").strip(), f"{name} has no module docstring")
                continue
            self.assertTrue((module.__doc__ or "").strip(), f"{name} has no module docstring")

    def test_manual_records_actual_demo_metrics(self) -> None:
        """手册引用的 demo 指标必须来自真实运行结果，防止示例数字过期或臆造。"""
        text = (ROOT / "MANUAL.md").read_text(encoding="utf-8")
        for value in ("1.9539", "5.0775", "1,003,854"):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
