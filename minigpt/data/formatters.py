"""SFT 文本格式化与样本编码（仅标准库）。

固定格式（CONSTRUCTION.md §5）::

    Q: {q}
    A: {a}<eos>

* 特殊 token ``<eos>`` 不写成字面量，而是使用 tokenizer 的 ``eos_id``；
* 训练标签（label）只对 answer + EOS 位置计 loss，question 与 padding 位置为
  ``-100``（PyTorch ``cross_entropy(ignore_index=-100)`` 约定）；
* 每条样本被裁/填充成固定 ``block_size``，方便 backend 直接堆成 tensor。

本模块不导入 torch/numpy，也不依赖具体 tokenizer 实现（鸭子类型）。
"""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping

#: 允许的最长 question / answer 字符数（超长样本在数据准备阶段丢弃并统计）。
MAX_QUESTION_CHARS = 4000
MAX_ANSWER_CHARS = 16000

#: 格式化模板，写入 SFT 元数据供文档/检查工具引用。
SFT_FORMAT_TEMPLATE = "Q: {q}\nA: {a}<eos>"


def normalize_question(text: Any) -> str:
    """规范化 question：统一换行、去除行尾空白、丢弃空行、trim 首尾。"""
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in value.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def normalize_answer(text: Any) -> str:
    """规范化 answer：保留内部换行结构，只去掉行尾空白与首尾空白。"""
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    return "\n".join(lines).strip()


def normalize_group(text: Any) -> str:
    """group 归一化：NFKC + casefold + 空白折叠，用作“同组不跨集合”的分组键。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = " ".join(value.split())
    return value.casefold()


def format_sft_prompt(question: Any) -> str:
    """构造 SFT prompt（不含 answer）：``Q: {q}\\nA: ``。"""
    q = normalize_question(question)
    if not q:
        raise ValueError("question 不能为空")
    return f"Q: {q}\nA: "


def format_sft_full_text(question: Any, answer: Any) -> str:
    """人类可读的完整 SFT 文本（``<eos>`` 只是字面量，不参与编码）。"""
    q = normalize_question(question)
    a = normalize_answer(answer)
    if not q or not a:
        raise ValueError("question/answer 不能为空")
    return f"Q: {q}\nA: {a}<eos>"


def _tokenizer_attrs(tokenizer: Any) -> tuple[int, int]:
    """取出 (eos_id, pad_id)，兼容 CharTokenizer 与任何实现同协议的对象。"""
    eos_id = getattr(tokenizer, "eos_id", None)
    pad_id = getattr(tokenizer, "pad_id", None)
    if eos_id is None or pad_id is None:
        specials = getattr(tokenizer, "specials", {})
        eos_id = specials.get("<eos>")
        pad_id = specials.get("<pad>")
    if eos_id is None or pad_id is None:
        raise TypeError("tokenizer 缺少 eos_id/pad_id（或 specials 映射）")
    return int(eos_id), int(pad_id)


def encode_sft_example(
    tokenizer: Any,
    question: Any,
    answer: Any,
    block_size: int,
) -> tuple[list[int], list[int]]:
    """把一条 Q&A 编码成固定长度的 ``(x, y)``。

    * ``x``：输入 token id，长度恒为 ``block_size``，不足用 ``pad_id`` 填充；
    * ``y``：下一个 token 标签，question/pad 位置为 ``-100``，仅 answer+EOS 计
      loss；
    * 样本过长时保留 question 前缀与 ``\nA: `` 边界，answer 至少留 2 个 token，末尾仍然补 ``<eos>``。
    """
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError(f"block_size 至少为 2，收到 {block_size}")
    q = normalize_question(question)
    a = normalize_answer(answer)
    if not q or not a:
        raise ValueError("question/answer 不能为空")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"question 过长（{len(q)} > {MAX_QUESTION_CHARS} 字符）")
    if len(a) > MAX_ANSWER_CHARS:
        raise ValueError(f"answer 过长（{len(a)} > {MAX_ANSWER_CHARS} 字符）")

    eos_id, pad_id = _tokenizer_attrs(tokenizer)
    prompt_ids: list[int] = list(tokenizer.encode(format_sft_prompt(q)))
    answer_ids: list[int] = list(tokenizer.encode(a))
    if not prompt_ids or not answer_ids:
        raise ValueError("tokenizer 编码结果为空")

    # 序列总长（含 eos）最多 block_size + 1，这样 x/y 恰好 block_size。
    max_seq = block_size + 1
    if len(prompt_ids) + len(answer_ids) + 1 > max_seq:
        # 截断时尽量保留 "\nA: " 边界，避免模型学到“问题中途直接接答案”。
        # 至少保留 2 个 answer token（+EOS），使监督信号不退化为单字符。
        suffix_ids = list(tokenizer.encode("\nA: "))
        keep_answer = min(len(answer_ids), max(2, max_seq // 3))
        keep_prompt = max_seq - 1 - keep_answer
        if len(suffix_ids) < keep_prompt:
            head = prompt_ids[: keep_prompt - len(suffix_ids)]
            prompt_ids = head + suffix_ids
        else:
            prompt_ids = prompt_ids[:keep_prompt]
        answer_ids = answer_ids[:keep_answer]

    sequence = prompt_ids + answer_ids + [eos_id]
    sequence = sequence[:max_seq]
    x = sequence[:-1]
    y = [-100] * len(x)
    first_answer_pos = max(0, len(prompt_ids) - 1)
    for position in range(first_answer_pos, len(x)):
        y[position] = sequence[position + 1]

    # 填充到固定长度：输入补 pad，标签补 -100。
    missing = block_size - len(x)
    if missing > 0:
        x.extend([pad_id] * missing)
        y.extend([-100] * missing)
    return x, y


def validate_record(record: Mapping[str, Any], default_source: str = "") -> dict[str, str]:
    """校验并规范化一条 SFT 记录，返回字段固定的新字典。

    必填 ``q/a``（也接受 ``question/answer`` 别名）；``group`` 缺省为归一化后的
    问题文本，``category/source`` 可选。校验失败抛 ``ValueError``（附原因），
    由数据准备脚本统计跳过数量。
    """
    if not isinstance(record, Mapping):
        raise ValueError("记录必须是 JSON 对象")
    question = record.get("q", record.get("question"))
    answer = record.get("a", record.get("answer"))
    q = normalize_question(question)
    a = normalize_answer(answer)
    if not q:
        raise ValueError("question 为空")
    if not a:
        raise ValueError("answer 为空")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"question 过长（{len(q)} > {MAX_QUESTION_CHARS}）")
    if len(a) > MAX_ANSWER_CHARS:
        raise ValueError(f"answer 过长（{len(a)} > {MAX_ANSWER_CHARS}）")
    group_raw = record.get("group")
    group = normalize_group(group_raw) if group_raw else normalize_group(q)
    return {
        "q": q,
        "a": a,
        "group": group,
        "category": str(record.get("category") or ""),
        "source": str(record.get("source") or default_source or ""),
    }


def record_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """去重键：归一化问题 + 归一化回答。"""
    return (
        normalize_group(record.get("q", "")),
        normalize_answer(record.get("a", "")),
    )


__all__ = [
    "MAX_QUESTION_CHARS",
    "MAX_ANSWER_CHARS",
    "SFT_FORMAT_TEMPLATE",
    "normalize_question",
    "normalize_answer",
    "normalize_group",
    "format_sft_prompt",
    "format_sft_full_text",
    "encode_sft_example",
    "validate_record",
    "record_identity",
]
