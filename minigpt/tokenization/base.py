"""Tokenizer 协议。

Stage 1 使用字符级 tokenizer（不依赖外部 tokenizer 包）。未来的 tokenizer
（如 BPE）只需实现下面的接口，就能被训练器、数据管线与推理路径直接复用。

约定：
* ``vocab_size`` 是词表大小，模型最后一层输出维度必须与之相等；
* ``encode``/``decode`` 互为逆操作（特殊 token 除外）；
* ``state_dict``/``from_state`` 必须保证 id 稳定，否则旧 checkpoint 的权重
  会与词表错位。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """tokenizer 需要提供的最小接口（结构类型，无需继承）。"""

    #: 词表大小，等于模型输出 logits 的最后一维。
    vocab_size: int

    #: 文本 -> id 列表；``add_eos=True`` 时在末尾追加 ``<eos>``。
    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...

    #: id 列表 -> 文本；``skip_special=True`` 时丢弃特殊 token。
    def decode(self, ids: list[int], skip_special: bool = True) -> str: ...

    #: 可 JSON 序列化的状态（词表 + 特殊 token id 映射）。
    def state_dict(self) -> dict: ...

    #: 从 ``state_dict`` 的结果重建，保证 id 与原先一致。
    @classmethod
    def from_state(cls, state: dict) -> "Tokenizer": ...

    #: 从目录读取持久化状态；具体文件名由实现方决定（char 版为 tokenizer.json）。
    @classmethod
    def load(cls, directory: str | Path) -> "Tokenizer": ...

    #: 把状态写入目录，供跨进程/跨机器复用。
    def save(self, directory: str | Path) -> None: ...

    #: 单个 token 的可读表示（日志与人工检查）。
    def token_repr(self, token_id: int) -> str: ...


__all__ = ["Tokenizer"]
