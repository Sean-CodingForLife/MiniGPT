"""chat.py 推理 prompt 归一化测试（F19）。

训练侧 formatters.format_sft_prompt 会先 normalize_question（折叠空白、统一
换行、去空行、trim）；推理侧必须得到完全一致的 prompt，否则训推不一致。
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chat
from minigpt.data.formatters import format_sft_prompt
from minigpt.tokenization.char import CharTokenizer

VOCAB_TEXT = "Question Answer hello world ok abcdefghijklmnopqrstuvwxyz0123456789?:. \n"


class ChatPromptNormalizationTests(unittest.TestCase):
    """确认 CLI 推理侧与训练侧对同一问题生成逐字节相同的 prompt。"""

    def setUp(self) -> None:
        # 用假 backend 截获真正送给模型的 prompt_ids，绕开真实权重与前向计算。
        self.tokenizer = CharTokenizer.build(VOCAB_TEXT)
        self.captured: dict = {}
        tokenizer = self.tokenizer
        captured = self.captured

        def generate(model, prompt_ids, **kwargs):
            captured["prompt_ids"] = list(prompt_ids)
            return list(prompt_ids) + tokenizer.encode("ok")

        def generate_stream(model, prompt_ids, **kwargs):
            captured["prompt_ids"] = list(prompt_ids)
            for index, token_id in enumerate(tokenizer.encode("ok"), start=1):
                yield {
                    "step": index,
                    "token_id": token_id,
                    "prob": 1.0,
                    "finished": False,
                    "topk": [],
                }

        self.bundle = SimpleNamespace(
            tokenizer=tokenizer,
            backend=SimpleNamespace(generate=generate, generate_stream=generate_stream),
            model=None,
            payload={"stage": "sft"},
        )

    def _prompt_text(self) -> str:
        """把截获的 prompt token 解回文本，便于与 formatters 的期望串直接比较。"""
        return self.tokenizer.decode(self.captured["prompt_ids"], skip_special=False)

    def test_answer_normalizes_whitespace_like_training(self) -> None:
        """连续空白与首尾空白必须折叠成单个空格，最终 prompt 与训练格式完全一致。"""
        chat._answer(
            self.bundle, "  hello   world \n\n ", max_new_tokens=4, temperature=0, top_k=5, top_p=1.0
        )
        self.assertEqual(self._prompt_text(), format_sft_prompt("hello world"))
        self.assertEqual(self._prompt_text(), "Q: hello world\nA: ")

    def test_answer_unifies_newlines_like_training(self) -> None:
        """CRLF/孤立 CR 都要先统一成 LF，否则同一问题会编码出不同 token 序列。"""
        chat._answer(
            self.bundle, "hello\r\nworld\r", max_new_tokens=4, temperature=0, top_k=5, top_p=1.0
        )
        self.assertEqual(self._prompt_text(), format_sft_prompt("hello\nworld"))

    def test_answer_messy_and_clean_inputs_encode_identically(self) -> None:
        """脏输入（换行/制表符/多余空格）与干净输入的 token id 序列必须完全相同。"""
        chat._answer(
            self.bundle, "hello world", max_new_tokens=4, temperature=0, top_k=5, top_p=1.0
        )
        clean_ids = list(self.captured["prompt_ids"])
        chat._answer(
            self.bundle, "\n hello\t world  \n", max_new_tokens=4, temperature=0, top_k=5, top_p=1.0
        )
        self.assertEqual(self.captured["prompt_ids"], clean_ids)

    def test_stream_answer_normalizes_whitespace_like_training(self) -> None:
        """流式回答走的是另一条代码路径，同样必须复用训练侧的归一化逻辑。"""
        with redirect_stdout(io.StringIO()):
            chat._stream_answer(
                self.bundle,
                "  hello   world \n\n ",
                max_new_tokens=4,
                temperature=0,
                top_k=5,
                top_p=1.0,
                trace_top_k=0,
            )
        self.assertEqual(self._prompt_text(), format_sft_prompt("hello world"))


if __name__ == "__main__":
    unittest.main()
