"""字符级 tokenizer（Stage 1 唯一实现）。

设计要点
--------
* 词表 = 3 个特殊 token + 训练集出现的字符；字符 id 按 Unicode 码点升序排列，
  因此同一份训练文本每次构建的结果完全一致（可复现）。
* 特殊 token 固定为 ``<unk>/<eos>/<pad>``，id 固定为 0/1/2；id 随
  ``state_dict()`` 一起写入 ``tokenizer.json``，load 后保持稳定。
* 词表只允许用 **训练集** 文本构建；验证集/测试集里的未登录字符编码为
  ``<unk>``，由数据准备脚本统计 unk rate（见 ``prepare.py``）。
* 只依赖 Python 标准库，模块顶层不导入 torch/numpy。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..registry import TOKENIZERS

#: 特殊 token，顺序即 id 顺序：``<unk>=0, <eos>=1, <pad>=2``。
SPECIALS: tuple[str, ...] = ("<unk>", "<eos>", "<pad>")

#: 词表状态格式版本，写入 ``tokenizer.json``。
STATE_VERSION = 1

#: 字符级 tokenizer 的 id 上限（uint16 二进制格式只能存 0..65535）。
MAX_VOCAB_SIZE = 65536


class CharTokenizer:
    """字符级 tokenizer：一个字符对应一个 token id。

    用法::

        tok = CharTokenizer.build(train_text)      # 只用训练集构建
        ids = tok.encode("hello", add_eos=True)    # -> [..., eos_id]
        text = tok.decode(ids)                     # 跳过特殊 token
        tok.save("data/shakespeare_en")            # 写 tokenizer.json
        tok2 = CharTokenizer.load("data/shakespeare_en")

    构造参数一般由 :meth:`build` / :meth:`from_state` 使用，不需要手工调用。
    """

    def __init__(self, chars: Iterable[str], specials: Mapping[str, int] | None = None):
        """由普通字符序列与特殊 token 映射构造 tokenizer。

        参数:
            chars: 训练集中出现的字符，每个元素必须是长度为 1 的 str；顺序即
                其 id 顺序（id 从 ``len(SPECIALS)`` 开始连续分配），不允许重复，
                也不允许与特殊 token 同名。
            specials: 特殊 token 名到 id 的映射；``None`` 时使用模块级
                ``SPECIALS_IDS``。只接受规范映射（``<unk>=0/<eos>=1/<pad>=2``），
                任何置换都会被拒绝，以保证跨 checkpoint 的 id 稳定。
        异常:
            ValueError: 特殊 token 集合或 id 不合规、字符非法/重复、
                词表规模超过 ``MAX_VOCAB_SIZE``。
        """
        # ---- 特殊 token：强制规范映射 unk=0/eos=1/pad=2（置换直接拒绝）----
        special_map = dict(SPECIALS_IDS if specials is None else specials)
        if set(special_map) != set(SPECIALS):
            raise ValueError(f"specials 必须是 {SPECIALS}，收到 {tuple(special_map)}")
        if any(special_map[name] != index for index, name in enumerate(SPECIALS)):
            raise ValueError(
                f"specials 必须是规范映射 {SPECIALS_IDS}（不允许置换），收到 {special_map}"
            )

        # ---- 普通字符：去重、保序（build 已排序；from_state 保持写入顺序）----
        seen: set[str] = set()
        clean_chars: list[str] = []
        for ch in chars:
            if not isinstance(ch, str) or len(ch) != 1:
                raise ValueError(f"词表元素必须是单个字符，收到 {ch!r}")
            if ch in special_map:
                raise ValueError(f"普通字符不能与特殊 token 同名：{ch!r}")
            if ch in seen:
                raise ValueError(f"词表字符重复：{ch!r}")
            seen.add(ch)
            clean_chars.append(ch)

        if len(SPECIALS) + len(clean_chars) > MAX_VOCAB_SIZE:
            raise ValueError(f"词表超过 uint16 上限 {MAX_VOCAB_SIZE}")

        self.specials: dict[str, int] = {name: special_map[name] for name in SPECIALS}
        self._id_to_special: dict[int, str] = {token_id: name for name, token_id in self.specials.items()}
        self.chars: list[str] = clean_chars
        self._char_to_id: dict[str, int] = {
            ch: len(SPECIALS) + index for index, ch in enumerate(clean_chars)
        }
        self._special_ids: frozenset[int] = frozenset(self.specials.values())

    # ------------------------------------------------------------------
    # 构建 / 基本属性
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        """用给定文本（只应是训练集）构建词表；字符按码点升序排列。"""
        if not isinstance(text, str):
            raise TypeError("CharTokenizer.build 需要 str")
        chars = sorted(set(text), key=ord)
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        """词表大小：特殊 token 数 + 普通字符数。"""
        return len(SPECIALS) + len(self.chars)

    @property
    def unk_id(self) -> int:
        """未登录字符的 token id（默认 0）。"""
        return self.specials["<unk>"]

    @property
    def eos_id(self) -> int:
        """结束符 token id（默认 1），生成时作为停止条件。"""
        return self.specials["<eos>"]

    @property
    def pad_id(self) -> int:
        """填充 token id（默认 2）。"""
        return self.specials["<pad>"]

    def __len__(self) -> int:
        """``len(tokenizer)`` 等价于 ``vocab_size``。"""
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        """调试用表示，不包含完整词表内容。"""
        return f"<CharTokenizer vocab_size={self.vocab_size} chars={len(self.chars)}>"

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """把文本编码为 id 列表；未登录字符映射为 ``<unk>``。"""
        if not isinstance(text, str):
            raise TypeError("CharTokenizer.encode 需要 str")
        mapping = self._char_to_id
        unk = self.unk_id
        ids = [mapping.get(ch, unk) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """把 id 列表解码为文本。

        ``skip_special=True`` 时丢掉 ``<unk>/<eos>/<pad>``；``False`` 时把它们
        还原成字面量（如 ``"<eos>"``），便于记录/调试。

        参数:
            ids: token id 序列，元素会被 ``int()`` 转换（兼容 numpy/torch 标量）。
            skip_special: 是否过滤特殊 token。
        返回:
            拼接后的字符串；越界 id 会抛 ``ValueError`` 而不是静默替换。
        """
        n = len(self.chars)
        parts: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._special_ids:
                if skip_special:
                    continue
                parts.append(self.token_repr(token_id))
            elif 0 <= token_id < self.vocab_size:
                parts.append(self.chars[token_id - len(SPECIALS)])
            else:
                raise ValueError(f"token id 越界：{token_id}（vocab_size={self.vocab_size}）")
        return "".join(parts)

    def token_repr(self, token_id: int) -> str:
        """单个 token 的可读表示，供日志/检查工具使用。"""
        token_id = int(token_id)
        if token_id in self._special_ids:
            return self._id_to_special[token_id]
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token id 越界：{token_id}（vocab_size={self.vocab_size}）")
        return _char_repr(self.chars[token_id - len(SPECIALS)])

    # ------------------------------------------------------------------
    # 状态序列化
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的稳定状态（含特殊 token 的 id）。"""
        return {
            "type": "char",
            "version": STATE_VERSION,
            "specials": dict(self.specials),
            "chars": list(self.chars),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CharTokenizer":
        """从 :meth:`state_dict` 的结果恢复 tokenizer（id 保持不变）。

        参数:
            state: 含 ``chars`` 列表，可选 ``type``（缺省视为 ``"char"``，
                兼容最早期的 checkpoint）、``version``、``specials``。
        异常:
            ValueError: 类型不符、版本不是 ``STATE_VERSION`` 或缺 ``chars``。
            版本不匹配时直接失败而不做迁移，避免静默错位。
        """
        if not isinstance(state, Mapping):
            raise ValueError("tokenizer state 必须是 dict")
        if state.get("type") not in (None, "char"):
            raise ValueError(f"不是字符级 tokenizer 状态：type={state.get('type')!r}")
        version = int(state.get("version", STATE_VERSION))
        if version != STATE_VERSION:
            raise ValueError(f"不支持的 tokenizer state 版本：{version}")
        chars = state.get("chars")
        if not isinstance(chars, list):
            raise ValueError("tokenizer state 缺少 chars 列表")
        specials = state.get("specials")
        if specials is None:
            specials = SPECIALS_IDS
        return cls(chars, specials=specials)

    # ------------------------------------------------------------------
    # 磁盘读写
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path) -> "CharTokenizer":
        """从目录（读 ``tokenizer.json``）或直接 json 文件加载 tokenizer。"""
        path = _tokenizer_path(Path(directory))
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 tokenizer 文件：{path}。请先运行 prepare.py 生成 tokenizer.json。"
            )
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        return cls.from_state(state)

    def save(self, directory: str | Path) -> Path:
        """把 tokenizer 保存为 ``<directory>/tokenizer.json``（UTF-8、原子写）。"""
        path = _tokenizer_path(Path(directory))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.state_dict(), ensure_ascii=False, indent=2) + "\n"
        _atomic_write_text(path, payload)
        return path


def _char_repr(ch: str) -> str:
    """普通字符的可读表示：可打印字符原样返回，控制字符转义。"""
    if ch.isprintable():
        return ch
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    return f"\\u{ord(ch):04x}"


def _tokenizer_path(path: Path) -> Path:
    """目录 -> ``<dir>/tokenizer.json``；已指向 json 文件则原样返回。"""
    if path.suffix.lower() == ".json":
        return path
    return path / "tokenizer.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """UTF-8 文本原子写（统一委托 minigpt.utils.atomic_write_text）。"""
    from ..utils import atomic_write_text

    atomic_write_text(path, text)


# 特殊 token 的默认 id 映射（模块级常量，便于 from_state 补齐）。
SPECIALS_IDS: dict[str, int] = {name: index for index, name in enumerate(SPECIALS)}

# 注册到全局 registry：config.tokenizer_type = "char" 时由 CLI/训练器取用。
TOKENIZERS.register("char")(CharTokenizer)

__all__ = ["CharTokenizer", "SPECIALS", "SPECIALS_IDS", "MAX_VOCAB_SIZE", "STATE_VERSION"]
