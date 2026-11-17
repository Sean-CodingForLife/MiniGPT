"""B5 收尾：F30 字形匹配 / F31 演示数据 / F32 哈希种子 / F33 config 路径 / F34 README。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable


class VisualizeGlyphTests(unittest.TestCase):
    """覆盖 VisualizeGlyphTests 相关行为。"""
    def test_glyph_match_uses_bare_chars(self) -> None:
        """验证 `test_glyph_match_uses_bare_chars` 对应的行为约定。"""
        text = (ROOT / "visualize.py").read_text(encoding="utf-8")
        # token_repr 返回裸字符（" "、"\n"），JS 匹配不得带多余引号。
        self.assertNotIn('=== "\' \'"', text)
        self.assertNotIn("=== \"'\\\\n'\"", text)
        self.assertIn('=== " "', text)


def _load_demo_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_demo_data", ROOT / "scripts" / "build_demo_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoDataTests(unittest.TestCase):
    """覆盖 DemoDataTests 相关行为。"""
    def test_quantifiers_have_no_duplicates(self) -> None:
        """验证 `test_quantifiers_have_no_duplicates` 对应的行为约定。"""
        mod = _load_demo_module()
        pairs = mod.ZH_QUANTIFIERS
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_eval_in_distribution_covers_multiple_categories(self) -> None:
        """验证 评估 相关行为。"""
        mod = _load_demo_module()
        for language in ("english", "chinese"):
            records = mod.build_eval(language)
            in_dist = [r for r in records if r["category"] == "in_distribution"]
            self.assertEqual(len(in_dist), 10)
            # 问题文本应来自多个类别：用答案多样性粗检（加法题答案全是数字）。
            numeric = sum(1 for r in in_dist if r["a"].strip().isdigit())
            self.assertLess(numeric, len(in_dist), f"{language} in_distribution 几乎全是算术题")


class HashSeedTests(unittest.TestCase):
    """覆盖 HashSeedTests 相关行为。"""
    def test_ensure_hash_seed_warns_without_restart(self) -> None:
        """新契约：未设置时只提示，不重启进程、不修改环境变量。"""
        code = (
            "import os, sys; sys.path.insert(0, r'" + str(ROOT) + "');"
            "from minigpt.utils import ensure_hash_seed;"
            "ensure_hash_seed(123);"
            "print('alive', os.environ.get('PYTHONHASHSEED', 'unset'))"
        )
        env = os.environ.copy()
        env.pop("PYTHONHASHSEED", None)
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [PYTHON, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=60, env=env
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # 进程没有被重启替换：同一进程继续执行并打印 alive。
        self.assertEqual(result.stdout.strip(), "alive unset")
        self.assertIn("PYTHONHASHSEED", result.stderr)

    def test_ensure_hash_seed_silent_when_already_set(self) -> None:
        """验证 随机种子 相关行为。"""
        code = (
            "import os, sys; sys.path.insert(0, r'" + str(ROOT) + "');"
            "from minigpt.utils import ensure_hash_seed;"
            "ensure_hash_seed(123);"
            "print('ok', os.environ.get('PYTHONHASHSEED'))"
        )
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "7"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [PYTHON, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=60, env=env
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # 已设置时不提示、不覆盖用户显式值。
        self.assertEqual(result.stdout.strip(), "ok 7")
        self.assertNotIn("PYTHONHASHSEED", result.stderr)

    def test_seed_everything_does_not_fake_hash_seed(self) -> None:
        """验证 随机种子 相关行为。"""
        import ast

        text = (ROOT / "minigpt" / "utils.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "seed_everything")
        # 函数体（不含 docstring）内不得再假装运行时设置 PYTHONHASHSEED 能生效。
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                continue
        source_lines = text.splitlines()[fn.lineno - 1 : fn.end_lineno]
        code_only = "\n".join(
            line for line in source_lines if "PYTHONHASHSEED" not in line or line.strip().startswith(("os.environ", "env["))
        )
        self.assertNotIn('os.environ["PYTHONHASHSEED"]', code_only)


class ConfigPathTests(unittest.TestCase):
    """覆盖 ConfigPathTests 相关行为。"""
    def test_relative_config_resolved_against_project_root(self) -> None:
        """验证 配置、项目 相关行为。"""
        from minigpt.config import parse_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cfg.json").write_text(json.dumps({"lr": 0.0005}), encoding="utf-8")
            other = root / "elsewhere"
            other.mkdir()
            old = os.getcwd()
            try:
                os.chdir(other)
                config = parse_config(
                    "pretrain",
                    ["--preset", "debug_en", "--project_root", str(root), "--config", "cfg.json"],
                )
            finally:
                os.chdir(old)
            self.assertAlmostEqual(config.lr, 0.0005)


class ReadmeTests(unittest.TestCase):
    """覆盖 ReadmeTests 相关行为。"""
    def test_parameter_count_accurate(self) -> None:
        """验证 `test_parameter_count_accurate` 对应的行为约定。"""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("默认只训练 **0.81M**", text)
        self.assertIn("0.81M–1.5M", text)


if __name__ == "__main__":
    unittest.main()