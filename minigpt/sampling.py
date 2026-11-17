"""共享采样规则内核（纯标准库，torch / pure 双后端共用）。

设计取舍
--------
采样"规则"（温度缩放、top-k 阈值、nucleus 累积边界、逆 CDF 选词、重复惩罚、
n-gram 封禁）只在此处实现一份；数值计算（softmax/cumsum）由各后端用自己的
张量库完成——torch 保持 float32  GPU/CPU 一致性与既有随机数流，pure 用
Python float。内核函数只接收普通 list/标量，不导入 torch，因此可被 pure 引擎
直接调用，也可被打包进无 torch 的推理包。

调用方式
--------
* torch：``temperature_scale`` -> 自身 softmax -> ``top_k_threshold`` ->
  自身 masked softmax -> ``nucleus_keep`` -> 自身 renorm/cumsum ->
  ``inverse_cdf_token``。
* pure：``sample_from_logits`` 一次性完成上述全部步骤（内部用本模块的
  ``softmax``）。
"""

from __future__ import annotations

import math
from typing import Sequence


def softmax(values: Sequence[float]) -> list[float]:
    """数值稳定的 softmax（减去最大值防溢出）。"""
    maximum = max(values)
    if not math.isfinite(maximum):
        return [1.0 / len(values)] * len(values)
    exponentials = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in values]
    total = sum(exponentials)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exponentials]


def temperature_scale(logits: Sequence[float], temperature: float) -> list[float]:
    """温度缩放；temperature<=0 表示 greedy，原样返回（由调用方走 argmax）。"""
    if temperature <= 0:
        return list(logits)
    return [value / temperature for value in logits]


def top_k_threshold(scaled: Sequence[float], top_k: int) -> float | None:
    """top-k 截断的 logit 阈值（第 k 大的缩放值）；top_k<=0 返回 None（不截断）。

    保留所有 >= 阈值的 token（与 torch ``scaled >= threshold`` 一致，含并列）。
    """
    if not top_k or top_k <= 0:
        return None
    k = min(int(top_k), len(scaled))
    return sorted(scaled, reverse=True)[k - 1]


def nucleus_keep(stage_probs: Sequence[float], top_p: float) -> list[bool]:
    """nucleus（top-p）保留掩码：按概率降序累积，保留"之前累积 <= top_p"的 token。

    与 torch 实现一致：``remove = (cumulative - sorted_probs) > top_p``，即每个
    token 看它**之前**的累积是否已超 top_p；第一个跨越边界的 token 仍保留。
    top_p>=1 时全部保留。
    """
    if top_p >= 1.0:
        return [True] * len(stage_probs)
    order = sorted(range(len(stage_probs)), key=lambda index: stage_probs[index], reverse=True)
    keep = [False] * len(stage_probs)
    cumulative = 0.0
    for index in order:
        if cumulative > top_p:
            break
        keep[index] = True
        cumulative += stage_probs[index]
    return keep


def inverse_cdf_token(cumulative: Sequence[float], draw: float) -> int:
    """逆 CDF 选词：第一个累积 >= draw 的下标（等价 torch.searchsorted 左插入）。

    draw 落在 [0,1)；浮点末尾累积可能略小于 1，越界时钳到最后一个 token。
    """
    for index, value in enumerate(cumulative):
        if draw <= value:
            return index
    return len(cumulative) - 1


def sample_from_logits(
    logits: Sequence[float],
    temperature: float,
    top_k: int,
    top_p: float,
    draw: float,
) -> tuple[int, list[float]]:
    """pure 引擎用：一次性完成温度/top-k/top-p/逆 CDF，返回 (token_id, 最终概率)。

    greedy（temperature<=0）时返回 argmax 与原始 softmax 概率，不消费 draw。
    """
    if temperature <= 0:
        probs = softmax(logits)
        token_id = max(range(len(probs)), key=lambda index: probs[index])
        return token_id, probs

    scaled = temperature_scale(logits, temperature)
    probs = softmax(scaled)
    threshold = top_k_threshold(scaled, top_k)
    if threshold is not None:
        probs = [value if value >= threshold else 0.0 for value in probs]
        total = sum(probs)
        # 数值下溢/并列截断可能使保留概率全零：回退均匀分布而非除零崩溃。
        probs = [value / total for value in probs] if total > 0 else [1.0 / len(probs)] * len(probs)
    keep = nucleus_keep(probs, top_p)
    if not all(keep):
        probs = [value if kept else 0.0 for value, kept in zip(probs, keep)]
        total = sum(probs)
        probs = [value / total for value in probs] if total > 0 else [1.0 / len(probs)] * len(probs)
    cumulative = 0.0
    cumulative_list: list[float] = []
    for value in probs:
        cumulative += value
        cumulative_list.append(cumulative)
    token_id = inverse_cdf_token(cumulative_list, draw)
    return token_id, probs


def sample_with_trace(
    logits: Sequence[float],
    temperature: float,
    top_k: int,
    top_p: float,
    draw: float | None,
    trace_candidates: int = 20,
) -> tuple[int, list[float], dict | None]:
    """采样并可选记录完整管线（pure 引擎用；与 torch 追踪同一份字段契约）。

    规则与 :func:`sample_from_logits` 完全一致（同一内核判定），随机数流不变：
    greedy 不消费 draw；采样路径由调用方先取 draw 再传入。
    ``trace_candidates<=0`` 时不构建追踪（训练/评估热路径不付费）。
    """

    greedy = temperature <= 0
    if greedy:
        probs = softmax(logits)
        token_id = max(range(len(probs)), key=lambda index: probs[index])
        trace = None
        if trace_candidates > 0:
            trace = build_sample_trace(
                logits=logits, raw_probs=probs, stage_probs=probs, final_probs=probs,
                chosen=token_id, u=None, mode="greedy", temperature=0.0,
                top_k=top_k, top_p=top_p, top_k_threshold=None, nucleus_mass=1.0,
                trace_candidates=trace_candidates, filters_applied=False,
            )
        return token_id, probs, trace

    scaled = temperature_scale(logits, temperature)
    raw_probs = softmax(scaled)
    threshold = top_k_threshold(scaled, top_k)
    if threshold is not None:
        stage_probs = [value if value >= threshold else 0.0 for value in raw_probs]
        total = sum(stage_probs)
        stage_probs = (
            [value / total for value in stage_probs]
            if total > 0
            else [1.0 / len(stage_probs)] * len(stage_probs)
        )
    else:
        stage_probs = raw_probs
    final_probs = stage_probs
    nucleus_mass = 1.0
    if top_p < 1.0:
        keep = nucleus_keep(stage_probs, top_p)
        if not all(keep):
            kept = [value if kept_flag else 0.0 for value, kept_flag in zip(stage_probs, keep)]
            nucleus_mass = sum(kept)
            total = nucleus_mass
            final_probs = (
                [value / total for value in kept]
                if total > 0
                else [1.0 / len(kept)] * len(kept)
            )
    cumulative: list[float] = []
    running = 0.0
    for value in final_probs:
        running += value
        cumulative.append(running)
    if draw is None:
        raise ValueError("sampling mode requires an explicit draw")
    token_id = inverse_cdf_token(cumulative, draw)
    trace = None
    if trace_candidates > 0:
        trace = build_sample_trace(
            logits=logits, raw_probs=raw_probs, stage_probs=stage_probs, final_probs=final_probs,
            chosen=token_id, u=draw, mode="sample", temperature=float(temperature),
            top_k=int(top_k), top_p=float(top_p), top_k_threshold=threshold,
            nucleus_mass=nucleus_mass, trace_candidates=trace_candidates, filters_applied=True,
        )
    return token_id, final_probs, trace


def build_sample_trace(
    *,
    logits: Sequence[float],
    raw_probs: Sequence[float],
    stage_probs: Sequence[float],
    final_probs: Sequence[float],
    chosen: int,
    u: float | None,
    mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    top_k_threshold: float | None,
    nucleus_mass: float,
    trace_candidates: int,
    filters_applied: bool,
) -> dict:
    """构建单 token 的 JSON 可序列化采样追踪（字段与 torch 版逐一对齐）。"""

    vocab_size = len(raw_probs)
    count = max(1, min(int(trace_candidates), vocab_size))
    indices = sorted(range(vocab_size), key=lambda index: raw_probs[index], reverse=True)[:count]
    if chosen not in indices:
        # nucleus 重归一后选中 token 可能跌出原始 top-N：始终保留它。
        indices[-1] = chosen
    t1_probs = softmax(logits)
    cumulative: list[float] = []
    running = 0.0
    for value in final_probs:
        running += value
        cumulative.append(running)
    candidates: list[dict] = []
    for token_id in indices:
        candidates.append(
            {
                "token_id": token_id,
                "logit": round(float(logits[token_id]), 6),
                "prob_t1": round(float(t1_probs[token_id]), 6),
                "prob_raw": round(float(raw_probs[token_id]), 6),
                "prob_top_k": round(float(stage_probs[token_id]), 6),
                "prob_final": round(float(final_probs[token_id]), 6),
                "in_top_k": bool(stage_probs[token_id] > 0.0),
                "in_top_p": bool(final_probs[token_id] > 0.0),
                "cum_low": round(float(cumulative[token_id - 1]) if token_id > 0 else 0.0, 6),
                "cum_high": round(float(cumulative[token_id]), 6),
                "chosen": token_id == chosen,
            }
        )
    candidates.sort(key=lambda candidate: candidate["prob_raw"], reverse=True)
    entropy = -sum(
        value * math.log2(value) for value in final_probs if value > 0.0
    )
    keep_ids = [index for index, value in enumerate(final_probs) if value > 0.0]
    segments: list[dict] = []
    if len(keep_ids) <= 64:
        segments = [
            {
                "token_id": token_id,
                "prob": round(float(final_probs[token_id]), 6),
                "cum_low": round(float(cumulative[token_id - 1]) if token_id > 0 else 0.0, 6),
                "cum_high": round(float(cumulative[token_id]), 6),
                "chosen": token_id == chosen,
            }
            for token_id in keep_ids
        ]
    return {
        "mode": mode,
        "filters_applied": bool(filters_applied),
        "temperature": round(float(temperature), 4),
        "top_k": int(top_k),
        "top_p": round(float(top_p), 4),
        "vocab_size": vocab_size,
        "trace_candidates": count,
        "top_k_threshold_logit": (
            None if top_k_threshold is None else round(float(top_k_threshold), 6)
        ),
        "nucleus_mass": round(float(nucleus_mass), 6),
        "nucleus_count": len(keep_ids),
        "nucleus_segments": segments,
        "entropy_bits": round(entropy, 6),
        "top1_prob": round(float(max(final_probs)), 6),
        "u": None if u is None else round(float(u), 6),
        "chosen_id": int(chosen),
        "chosen_prob": round(float(final_probs[chosen]), 6),
        "chosen_cum_low": round(float(cumulative[chosen - 1]) if chosen > 0 else 0.0, 6),
        "chosen_cum_high": round(float(cumulative[chosen]), 6),
        "candidates": candidates,
    }


def repetition_penalized_logits(
    logits: Sequence[float], history: Sequence[int], penalty: float
) -> list[float]:
    """重复惩罚（HF 约定）：正 logit 除以 penalty，负 logit 乘以 penalty。

    penalty==1 或无历史时原样返回。只作用于历史中出现过的 token。
    """
    if penalty == 1.0 or not history:
        return list(logits)
    result = list(logits)
    for token_id in set(history):
        if 0 <= token_id < len(result):
            score = result[token_id]
            result[token_id] = score / penalty if score > 0 else score * penalty
    return result


def ngram_banned_tokens(history: Sequence[int], n: int) -> set[int]:
    """no_repeat_ngram：返回应封禁的下一 token 集合。

    若历史末尾的 (n-1)-gram 曾在历史中出现过，则其后继 token 全部封禁。
    n<=0 或历史不足 n 时返回空集。
    """
    if n <= 0 or len(history) < n:
        return set()
    prefix = tuple(history[-(n - 1) :]) if n > 1 else ()
    banned: set[int] = set()
    for i in range(len(history) - n + 1):
        if tuple(history[i : i + n - 1]) == prefix:
            banned.add(history[i + n - 1])
    return banned


__all__ = [
    "softmax",
    "temperature_scale",
    "top_k_threshold",
    "nucleus_keep",
    "inverse_cdf_token",
    "sample_from_logits",
    "repetition_penalized_logits",
    "ngram_banned_tokens",
]