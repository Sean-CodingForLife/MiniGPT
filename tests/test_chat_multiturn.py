"""N03 多轮对话 CLI：历史拼接、按 block_size 截断、/命令解析。

对话历史一旦拼接错位或截断策略反了（丢掉最新问题），模型回答就会文不对题；
本文件用字符级 tokenizer 精确控制 token 数来锁定这些边界。
"""

from __future__ import annotations

import unittest

from minigpt.data.formatters import format_sft_prompt
from minigpt.tokenization.char import CharTokenizer


def make_tokenizer() -> CharTokenizer:
    """构造字符表固定的 tokenizer，使 prompt 字符数≈token 数，便于精确断言截断。"""
    return CharTokenizer.build("abcdefghijklmnopqrstuvwxyzQ:A \n0123456789?")


class ConversationTests(unittest.TestCase):
    """验证多轮上下文拼接、截断顺序与系统提示词位置。"""

    def test_history_builds_context_with_past_turns(self) -> None:
        """历史轮次需按 Q/A 格式前置，当前提问以标准 SFT prompt 形式结尾。"""
        from chat import Conversation

        tokenizer = make_tokenizer()
        conv = Conversation(tokenizer, block_size=256)
        conv.add_turn("hello", "hi there")
        prompt = conv.build_prompt("how are you")
        self.assertIn("Q: hello", prompt)
        self.assertIn("A: hi there", prompt)
        self.assertTrue(prompt.endswith(format_sft_prompt("how are you")))

    def test_empty_history_equals_single_turn(self) -> None:
        """没有历史时构造结果必须与单轮格式化完全一致，避免多出分隔符。"""
        from chat import Conversation

        tokenizer = make_tokenizer()
        conv = Conversation(tokenizer, block_size=256)
        self.assertEqual(conv.build_prompt("hello"), format_sft_prompt("hello"))

    def test_truncation_drops_oldest_turns_first(self) -> None:
        """超长历史截断必须优先丢弃最旧轮次，同时保证总长度不超 block_size。"""
        from chat import Conversation

        tokenizer = make_tokenizer()
        conv = Conversation(tokenizer, block_size=64)
        # 塞入远超窗口的历史，制造必须截断的局面。
        for i in range(20):
            conv.add_turn(f"question number {i}", f"answer number {i}")
        prompt = conv.build_prompt("latest question")
        ids = tokenizer.encode(prompt)
        self.assertLessEqual(len(ids), 64)
        # 最新问题必须保留；最旧的历史应被丢弃。
        self.assertIn("latest question", prompt)
        self.assertNotIn("question number 0", prompt)

    def test_reset_clears_history(self) -> None:
        """/reset 之后历史需清空到与全新会话相同的状态。"""
        from chat import Conversation

        tokenizer = make_tokenizer()
        conv = Conversation(tokenizer, block_size=256)
        conv.add_turn("hello", "hi")
        conv.reset()
        self.assertEqual(conv.build_prompt("x"), format_sft_prompt("x"))

    def test_system_prompt_prepended(self) -> None:
        """系统提示词需位于整段上下文最前面，且不影响后续 SFT prompt 拼接。"""
        from chat import Conversation

        tokenizer = make_tokenizer()
        conv = Conversation(tokenizer, block_size=256, system="be brief")
        prompt = conv.build_prompt("hello")
        self.assertTrue(prompt.startswith("be brief"))
        self.assertIn(format_sft_prompt("hello"), prompt)


class CommandTests(unittest.TestCase):
    """验证 CLI 斜杠命令的识别与参数切分，未知命令不应被误当普通提问。"""

    def test_parse_commands(self) -> None:
        """已知命令返回 (名称, 参数)，普通文本与未知命令返回 None。"""
        from chat import parse_command

        self.assertEqual(parse_command("/reset"), ("reset", ""))
        self.assertEqual(parse_command("/help"), ("help", ""))
        self.assertEqual(parse_command("/system be nice"), ("system", "be nice"))
        self.assertIsNone(parse_command("normal question"))
        self.assertIsNone(parse_command("/unknown"))


if __name__ == "__main__":
    unittest.main()