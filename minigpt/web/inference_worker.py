"""SPA 生成视图的推理 worker（支持 torch 与 pure 两种 checkpoint）。

两种模式，由请求字段 ``stream`` 选择：

* batch（默认）：执行一次生成，向 stdout 写一份完整 JSON trace（兼容旧的
  ``POST /api/v1/inference`` 端点）；
* stream：写 NDJSON 事件流（``status`` -> ``meta`` -> ``step`` ×N ->
  ``final`` -> ``done``，或 ``error``），每行立即 flush，看板便可把每个
  生成的 token 实时转发给浏览器。

请求由看板构造：run id 在这里被解析为 ``runs/<id>/`` 下由
``_pick_checkpoint`` 选出的权重（``best.pt`` → ``best.mgckpt`` → ``last.pt``
→ ``last.mgckpt``，两种格式都由 ``minigpt.inference.load_bundle`` 自动识别），
浏览器不能传任意路径。``prefix_token_ids``（可选）固定已决定的 token，让 UI
能从更早的步骤分叉。

这个模块刻意作为子进程边界：导入它或看板后端都不会导入 torch。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minigpt.render import _prompt_for, collect_trace, iter_trace  # noqa: E402
from minigpt.inference import load_bundle  # noqa: E402
from minigpt.utils import seed_everything  # noqa: E402
from minigpt.web.diagnostics import public_message  # noqa: E402

MAX_PREFIX_TOKENS = 512


def _emit(event: dict) -> None:
    """Write one NDJSON event and flush so the parent sees it immediately."""

    sys.stdout.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _validate_prefix(bundle, raw) -> list[int]:
    """Validate optional forced prefix token ids against the bundle vocabulary."""

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("prefix_token_ids 必须是整数数组")
    if len(raw) > MAX_PREFIX_TOKENS:
        raise ValueError(f"prefix_token_ids 不能超过 {MAX_PREFIX_TOKENS} 个")
    vocab_size = int(getattr(bundle.tokenizer, "vocab_size", 0) or 0)
    prefix: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("prefix_token_ids 只能包含整数")
        if value < 0 or (vocab_size and value >= vocab_size):
            raise ValueError(f"prefix_token_ids 超出词表范围: {value}")
        prefix.append(value)
    return prefix


def _generation_options(request: dict, prefix: list[int]) -> dict:
    """从请求中解析并夹取生成参数。"""
    return {
        "prompt": str(request.get("prompt", "")),
        "max_new_tokens": int(request.get("max_new_tokens", 24)),
        "temperature": float(request.get("temperature", 0.8)),
        "top_k": int(request.get("top_k", 40)),
        "top_p": float(request.get("top_p", 1.0)),
        "trace_top_k": int(request.get("trace_top_k", 5)),
        "max_context": int(request.get("max_context", 32)),
        "template": str(request.get("template", "auto")),
        "forced_ids": prefix or None,
        "trace_candidates": int(request.get("trace_candidates", 20)),
    }


def _repetition_controls(request: dict) -> tuple[float, int]:
    """N09：重复惩罚 / n-gram 封禁参数（1.0 / 0 = 关闭，保持旧行为）。"""

    repetition_penalty = float(request.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(request.get("no_repeat_ngram_size", 0))
    if not 0.1 <= repetition_penalty <= 5.0:
        raise ValueError("repetition_penalty 必须在 [0.1, 5.0] 内")
    if not 0 <= no_repeat_ngram_size <= 8:
        raise ValueError("no_repeat_ngram_size 必须在 [0, 8] 内")
    return repetition_penalty, no_repeat_ngram_size


def _generate_with_controls(bundle, request: dict, prefix: list[int]) -> dict:
    """启用重复控制时的生成路径：backend.generate 支持惩罚参数。

    注意力/采样逐步跟踪（iter_trace）不支持这两个参数，因此该路径只产出
    最终文本（steps 为空、attention 为空），协议字段与 batch 模式保持一致。
    """

    repetition_penalty, no_repeat_ngram_size = _repetition_controls(request)
    tokenizer = bundle.tokenizer
    stage = str(bundle.payload.get("stage", "pretrain"))
    prompt_text = _prompt_for(stage, str(request.get("prompt", "")), str(request.get("template", "auto")))
    prompt_ids = tokenizer.encode(prompt_text)
    if prefix:
        prompt_ids = prompt_ids + [int(token_id) for token_id in prefix]
    max_new_tokens = int(request.get("max_new_tokens", 24))
    eos_id = getattr(tokenizer, "eos_id", None)
    started = time.perf_counter()
    generated = bundle.backend.generate(
        bundle.model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=float(request.get("temperature", 0.8)),
        top_k=int(request.get("top_k", 40)),
        top_p=float(request.get("top_p", 1.0)),
        eos_id=eos_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    continuation_ids = generated[len(prompt_ids) :]
    continuation = tokenizer.decode(continuation_ids, skip_special=True).strip()
    reason = "eos" if eos_id is not None and continuation_ids and continuation_ids[-1] == eos_id else "max_tokens"
    return {
        "steps": [],
        "final": {
            "step": len(continuation_ids),
            "context": [tokenizer.token_repr(token_id) for token_id in generated[-32:]],
            "attention": [],
            "continuation": continuation,
            "steps": len(continuation_ids),
            "reason": reason,
            "elapsed_ms": round(elapsed_ms, 2),
        },
        "meta": {
            "stage": stage,
            "prompt": prompt_text,
            "generation": {
                "max_new_tokens": max_new_tokens,
                "temperature": float(request.get("temperature", 0.8)),
                "top_k": int(request.get("top_k", 40)),
                "top_p": float(request.get("top_p", 1.0)),
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            },
        },
    }


def main() -> int:
    """推理 worker 入口：加载 checkpoint、流式发出事件、处理异常。"""
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("推理请求必须是 JSON 对象")
        checkpoint = Path(request["checkpoint"]).expanduser().resolve()
        stream = bool(request.get("stream"))
        seed = int(request.get("seed", 1337))
        repetition_penalty, no_repeat_ngram_size = _repetition_controls(request)
        controls_active = repetition_penalty != 1.0 or no_repeat_ngram_size > 0
        if stream:
            _emit({"type": "status", "phase": "loading"})
        seed_everything(seed)
        bundle = load_bundle(checkpoint, device=str(request.get("device", "auto")))
        prefix = _validate_prefix(bundle, request.get("prefix_token_ids"))
        options = _generation_options(request, prefix)
        if controls_active:
            # 重复控制路径：逐 token 跟踪不支持惩罚参数，直接产出最终结果。
            data = _generate_with_controls(bundle, request, prefix)
            data["checkpoint"] = checkpoint.name
            data["seed"] = seed
            if stream:
                _emit({"type": "status", "phase": "generating"})
                _emit({"type": "meta", **data["meta"], "checkpoint": checkpoint.name, "seed": seed})
                _emit({"type": "final", "data": data["final"]})
                _emit({"type": "done"})
            else:
                json.dump(data, sys.stdout, ensure_ascii=False, separators=(",", ":"))
                sys.stdout.write("\n")
                sys.stdout.flush()
            return 0
        if stream:
            _emit({"type": "status", "phase": "generating"})
            for event in iter_trace(bundle, **options):
                if event["type"] == "meta":
                    event["checkpoint"] = checkpoint.name
                    event["seed"] = seed
                _emit(event)
            _emit({"type": "done"})
            return 0
        data = collect_trace(bundle, **options)
        data["checkpoint"] = checkpoint.name
        data["seed"] = seed
        json.dump(data, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:  # noqa: BLE001 - 两种模式都必须输出结构化错误
        # 完整诊断留本地 stderr；stdout 协议只走脱敏后的 JSON error。
        traceback.print_exc(file=sys.stderr)
        try:
            _emit(
                {
                    "type": "error",
                    "code": type(exc).__name__,
                    "message": public_message(str(exc), roots=(ROOT,)),
                }
            )
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
