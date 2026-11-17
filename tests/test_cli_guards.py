"""命令行入口守卫（阶段 0 新增）。

1. chat.Conversation.build_prompt 丢完历史后仍可能返回超 block_size 的
   prompt（单问超长不截断），后端行为未定义；
2. parse_config 的 --config 相对路径应按项目根解析（注释承诺），实际
   在 project_root 为空时落到 cwd；
3. prepare.py 的 vocab_extra 传入 SFT val/test 文件属评估泄漏，目前只有
   文档警告，无运行时提示。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.config import parse_config  # noqa: E402
from minigpt.tokenization import CharTokenizer  # noqa: E402


class ChatPromptGuardTests(unittest.TestCase):
    """单条提问本身超过 block_size 时也必须被截断。"""

    def test_single_overlong_question_is_truncated(self) -> None:
        """历史全部丢光后仍超窗的极端输入不能原样送进模型，必须裁到 block_size 以内。"""
        from chat import Conversation

        tokenizer = CharTokenizer.build("abcdefghijklmnopqrstuvwxyz :?\n")
        conv = Conversation(tokenizer, block_size=24)
        prompt = conv.build_prompt("a" * 200)
        self.assertLessEqual(
            len(tokenizer.encode(prompt)), 24, "单问超长时也必须截断到 block_size 内"
        )


class ConfigPathGuardTests(unittest.TestCase):
    """--config 的相对路径应以 project_root 为基准而非进程 cwd。"""

    def test_relative_config_resolved_against_project_root(self) -> None:
        """在其它工作目录下启动时，conf/my.json 仍要能按项目根解析到并读到 n_layer。"""
        with tempfile.TemporaryDirectory(prefix="minigpt-cfg-") as tmp:
            root = Path(tmp)
            (root / "conf").mkdir()
            (root / "conf" / "my.json").write_text(
                json.dumps({"n_layer": 6}), encoding="utf-8"
            )
            cwd = os.getcwd()
            other = root / "elsewhere"
            other.mkdir()
            try:
                # 故意切到与配置文件无关的目录，暴露「按 cwd 解析」的回归。
                os.chdir(other)
                config = parse_config(
                    "pretrain",
                    ["--project_root", str(root), "--config", "conf/my.json"],
                )
            finally:
                os.chdir(cwd)
            self.assertEqual(config.n_layer, 6)


class VocabExtraLeakGuardTests(unittest.TestCase):
    """vocab_extra 指向 SFT val/test 文件时属于评估泄漏，至少要在 stderr 告警。"""

    def test_val_test_vocab_extra_warns(self) -> None:
        """构造 data/sft_demo/val.jsonl 作为词表来源，应输出含「泄漏」的提示。"""
        import prepare

        with tempfile.TemporaryDirectory(prefix="minigpt-leak-") as tmp:
            root = Path(tmp)
            sft_dir = root / "data" / "sft_demo"
            sft_dir.mkdir(parents=True)
            leak_file = sft_dir / "val.jsonl"
            leak_file.write_text('{"q": "hi", "a": "hello"}\n', encoding="utf-8")
            # 用轻量匿名对象顶替完整配置，只提供 _load_vocab_extra 需要的两个字段。
            config = type("C", (), {"project_root": str(root), "vocab_extra": str(leak_file)})()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                prepare._load_vocab_extra(config)
            self.assertIn(
                "泄漏",
                stderr.getvalue(),
                "vocab_extra 命中 val/test 文件时必须给出评估泄漏警告",
            )


if __name__ == "__main__":
    unittest.main()