"""Tokenizer 实现（Stage 1：字符级）。

导入本包即完成 ``char`` 注册（``TOKENIZERS.get("char")``），
训练器/CLI 通过 ``minigpt.config.Config.tokenizer_type`` 选择实现。

本包只依赖标准库；新增 tokenizer（如 BPE）应放在同目录并使用
``@TOKENIZERS.register("名字")`` 自行登记，无需修改本文件。
"""

from .base import Tokenizer
from .char import MAX_VOCAB_SIZE, SPECIALS, SPECIALS_IDS, STATE_VERSION, CharTokenizer

__all__ = [
    "Tokenizer",
    "CharTokenizer",
    "SPECIALS",
    "SPECIALS_IDS",
    "MAX_VOCAB_SIZE",
    "STATE_VERSION",
]
