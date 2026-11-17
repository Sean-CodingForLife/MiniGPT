"""轻量、可解释的评估指标（不依赖任何外部 NLP 库）。

指标刻意做得简单：用于发现空输出、重复退化、明显跑偏格式的生成结果；
它们**不能**替代人工评审（参见 MANUAL.md 的评估章节）。

约定：除 :func:`summarize_predictions` 外，所有指标先经 :func:`normalize_text`
归一化（NFKC、小写、去空白、去标点），因此对大小写与空格差异不敏感。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


def normalize_text(text: str) -> str:
    """归一化文本用于指标比较：NFKC、去首尾空白、小写、去空白与标点。

    参数:
        text: 任意文本；``None`` 会按空串处理。
    返回:
        仅保留字母/数字/下划线与 CJK 统一表意文字（``\\u4e00``-``\\u9fff``）
        的字符串，长度即参与比较的字符数。
    """
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text


def exact_match(prediction: str, reference: str) -> float:
    """归一化后完全一致返回 1.0，否则 0.0（便于直接求均值）。"""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def char_f1(prediction: str, reference: str) -> float:
    """字符级 F1：按字符多重集合计算精确率与召回率的调和平均。

    参数:
        prediction: 模型生成文本。
        reference: 参考答案。
    返回:
        取值 [0, 1]；任一为空（归一化后）时返回 0.0。
    """
    pred = Counter(normalize_text(prediction))
    ref = Counter(normalize_text(reference))
    if not pred or not ref:
        return 0.0
    # Counter 的交集按元素取较小计数，values() 之和即多集合重叠大小。
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def repetition_rate(text: str, n: int = 4) -> float:
    """重复 n-gram 占比；越高说明文本越退化（例如同一短语循环）。

    参数:
        text: 待检测文本，先经 :func:`normalize_text`。
        n: n-gram 长度，默认 4 个字符。
    返回:
        ``1 - 唯一 n-gram 数 / 全部 n-gram 数``；文本短于 ``n`` 时为 0.0。
    """

    normalized = normalize_text(text)
    if len(normalized) < n:
        return 0.0
    grams = [normalized[i : i + n] for i in range(len(normalized) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def eos_rate(stop_reason: list[bool]) -> float:
    """统计「以 EOS 正常结束」的样本比例。

    参数:
        stop_reason: 每个样本一个布尔值，True 表示由 EOS 停止（而非长度上限）。
    返回:
        命中比例；空列表按分母 1 处理，返回 0.0。
    """
    return sum(1 for x in stop_reason if x) / max(1, len(stop_reason))


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总逐样本记录，得到总体与按类别的指标。

    参数:
        records: 每项至少含键 ``prediction``、``reference``，可选 ``category``
            （缺省归入 ``"uncategorized"``）。
    返回:
        含 ``count``、``empty_rate``、``exact_match``、``char_f1``、
        ``repetition_rate`` 的字典；有记录时另含 ``by_category``，其值为
        去掉 ``empty_rate`` 的同结构小字典。记录为空时返回全零指标（无
        ``by_category``）。
    """

    if not records:
        return {
            "count": 0,
            "empty_rate": 0.0,
            "exact_match": 0.0,
            "char_f1": 0.0,
            "repetition_rate": 0.0,
        }
    # 逐条指标只算一次，按类别聚合时复用，避免 O(类别数 × 样本数) 的重复计算。
    exact = [exact_match(r.get("prediction", ""), r.get("reference", "")) for r in records]
    f1 = [char_f1(r.get("prediction", ""), r.get("reference", "")) for r in records]
    repetition = [repetition_rate(r.get("prediction", "")) for r in records]
    empty = [1.0 if not str(r.get("prediction", "")).strip() else 0.0 for r in records]
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record.get("category", "uncategorized")), []).append(index)
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted(groups):
        indices = groups[category]
        count = len(indices)
        by_category[category] = {
            "count": count,
            "exact_match": sum(exact[i] for i in indices) / count,
            "char_f1": sum(f1[i] for i in indices) / count,
            "repetition_rate": sum(repetition[i] for i in indices) / count,
        }
    return {
        "count": len(records),
        "empty_rate": sum(empty) / len(empty),
        "exact_match": sum(exact) / len(exact),
        "char_f1": sum(f1) / len(f1),
        "repetition_rate": sum(repetition) / len(repetition),
        "by_category": by_category,
    }


__all__ = [
    "normalize_text",
    "exact_match",
    "char_f1",
    "repetition_rate",
    "eos_rate",
    "summarize_predictions",
]
