"""生成追踪（generation trace）采集，供 CLI 可视化与 Web worker 共用。

这些辅助函数原先放在根目录 ``visualize.py`` 里，导致
``minigpt.web.inference_worker`` 必须导入仓库脚本（破坏包的可移植性）。
现在下沉到包内；``visualize.py`` 只是薄 CLI 包装并重新导出它们。
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterator


def _prompt_for(stage: str, prompt: str, template: str) -> str:
    """按模板选择最终 prompt：``raw`` 原样，``qa``/SFT 的 auto 套 Q/A 模板。"""
    if template == "raw":
        return prompt
    if template == "qa" or (template == "auto" and stage == "sft"):
        return f"Q: {prompt}\nA: "
    return prompt


def _hidden_summary(values: list[float], top_n: int = 8) -> dict[str, Any]:
    """汇总一步隐藏状态：L2 范数、均值、标准差，以及绝对值最大的若干维度。"""
    if not values:
        return {"norm": 0.0, "mean": 0.0, "std": 0.0, "top": []}
    norm = math.sqrt(sum(value * value for value in values))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    ranked = sorted(range(len(values)), key=lambda index: abs(values[index]), reverse=True)[:top_n]
    return {
        "norm": norm,
        "mean": mean,
        "std": math.sqrt(variance),
        "top": [{"dim": index, "value": values[index]} for index in ranked],
    }


def _sampling_with_tokens(
    tokenizer, sampling: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Add the printable token text to every sampling-trace candidate/segment."""

    if not sampling:
        return None
    enriched = dict(sampling)
    enriched["candidates"] = [
        {**candidate, "token": tokenizer.token_repr(candidate["token_id"])}
        for candidate in sampling.get("candidates", [])
    ]
    if sampling.get("nucleus_segments"):
        enriched["nucleus_segments"] = [
            {**segment, "token": tokenizer.token_repr(segment["token_id"])}
            for segment in sampling["nucleus_segments"]
        ]
    else:
        enriched["nucleus_segments"] = []
    return enriched


def iter_trace(
    bundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    trace_top_k: int,
    max_context: int,
    template: str,
    forced_ids: list[int] | None = None,
    trace_candidates: int = 20,
) -> Iterator[dict[str, Any]]:
    """Yield ``meta``, one ``step`` per generated token, then ``final``.

    ``collect_trace`` assembles these events for the static HTML exporter; the
    Web backend forwards them as NDJSON so the browser can render each token as
    it is produced.
    """

    tokenizer = bundle.tokenizer
    stage = str(bundle.payload.get("stage", "pretrain"))
    prompt_text = _prompt_for(stage, prompt, template)
    prompt_ids = tokenizer.encode(prompt_text)
    if forced_ids:
        prompt_ids = prompt_ids + [int(token_id) for token_id in forced_ids]
    eos_id = getattr(tokenizer, "eos_id", None)
    context_ids = list(prompt_ids)
    generation = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "eos_id": eos_id,
        "trace_candidates": trace_candidates,
    }
    yield {
        "type": "meta",
        "stage": stage,
        "device": bundle.backend.device_info().get("device", "?"),
        "prompt": prompt_text,
        "prompt_tokens": [tokenizer.token_repr(token_id) for token_id in prompt_ids],
        "forced_tokens": [
            tokenizer.token_repr(token_id) for token_id in (forced_ids or [])
        ],
        "model": bundle.model.config.to_dict(),
        "parameters": bundle.model.num_parameters(),
        "generation": generation,
    }

    # 注意力/采样追踪需要模型流式接口支持 trace 关键字（torch 与 pure 均已
    # 实现）；对不满足契约的后端提前给出明确错误，而不是抛难懂的 TypeError。
    import inspect

    try:
        stream_params = inspect.signature(bundle.model.generate_stream).parameters
    except (TypeError, ValueError):
        stream_params = {}
    if not {"trace_attention", "trace_sampling", "trace_candidates"} <= set(stream_params):
        raise RuntimeError(
            "该检查点的后端不支持生成过程可视化（注意力/采样追踪）；"
            "请用 chat.py / evaluate.py 查看生成结果"
        )

    records: list[dict[str, Any]] = []
    generated_ids: list[int] = []
    last_context: list[str] = []
    last_attention: list[list[list[list[float]]]] = []
    last_step: int | None = None
    reason = "max_tokens"
    started = time.perf_counter()
    for item in bundle.model.generate_stream(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
        trace_top_k=trace_top_k,
        trace_attention=True,
        trace_sampling=True,
        trace_candidates=trace_candidates,
    ):
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        forward_context = context_ids[-item["context_len"] :]
        keep = min(max_context, len(forward_context))
        context = forward_context[-keep:]
        attention = item["attention"]
        n_layer = len(attention)
        n_head = len(attention[0]) if n_layer else 0
        attn_last = [
            [
                [round(float(value), 6) for value in attention[layer][head][-1][-keep:]]
                for head in range(n_head)
            ]
            for layer in range(n_layer)
        ]
        record: dict[str, Any] = {
            "step": item["step"],
            "token_id": item["token_id"],
            "token": tokenizer.token_repr(item["token_id"]),
            "prob": item["prob"],
            "finished": item["finished"],
            "elapsed_ms": round(elapsed_ms, 2),
            "context": [tokenizer.token_repr(token_id) for token_id in context],
            "attn_last": attn_last,
            "hidden": _hidden_summary(item["hidden"]),
            "topk": [
                {
                    "token": tokenizer.token_repr(candidate["token_id"]),
                    "prob": candidate["prob"],
                    "chosen": candidate["token_id"] == item["token_id"],
                }
                for candidate in item["topk"]
            ],
            "sampling": _sampling_with_tokens(tokenizer, item.get("sampling")),
        }
        records.append(record)
        trimmed_attention = [
            [
                [
                    [round(float(value), 6) for value in row[-keep:]]
                    for row in attention[layer][head][-keep:]
                ]
                for head in range(n_head)
            ]
            for layer in range(n_layer)
        ]
        last_context = record["context"]
        last_attention = trimmed_attention
        last_step = item["step"]
        yield {"type": "step", "data": record}
        if item["finished"]:
            reason = "eos"
            break
        generated_ids.append(item["token_id"])
        context_ids.append(item["token_id"])

    if not records:
        raise ValueError("no token was generated; raise --max_new_tokens")
    continuation = tokenizer.decode(generated_ids, skip_special=True).strip()
    yield {
        "type": "final",
        "data": {
            "step": last_step,
            "context": last_context,
            "attention": last_attention,
            "continuation": continuation,
            "steps": len(records),
            "reason": reason,
        },
    }


def collect_trace(
    bundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    trace_top_k: int,
    max_context: int,
    template: str,
    forced_ids: list[int] | None = None,
    trace_candidates: int = 20,
) -> dict[str, Any]:
    """Run one generation and return the complete trace (static export API)."""

    meta: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for event in iter_trace(
        bundle,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        trace_top_k=trace_top_k,
        max_context=max_context,
        template=template,
        forced_ids=forced_ids,
        trace_candidates=trace_candidates,
    ):
        if event["type"] == "meta":
            meta = event
        elif event["type"] == "step":
            steps.append(event["data"])
        elif event["type"] == "final":
            final = event["data"]
    if meta is None or final is None:
        raise ValueError("no token was generated; raise --max_new_tokens")
    return {
        "stage": meta["stage"],
        "device": meta["device"],
        "prompt": meta["prompt"],
        "prompt_tokens": meta["prompt_tokens"],
        "continuation": final["continuation"],
        "model": meta["model"],
        "parameters": meta["parameters"],
        "generation": meta["generation"],
        "steps": steps,
        "final": {
            "step": final["step"],
            "context": final["context"],
            "attention": final["attention"],
        },
    }


__all__ = ["_prompt_for", "_hidden_summary", "_sampling_with_tokens", "iter_trace", "collect_trace"]