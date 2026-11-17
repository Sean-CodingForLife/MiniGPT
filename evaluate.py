"""在留出切分（val / test）上评估 checkpoint。

* pretrain：交叉熵 / 困惑度（PPL）/ 每字符比特数（BPC），并附一段采样。
* SFT：仅答案部分的 loss + 生成指标（空回答率、完全匹配、字符 F1、
  重复率），以及按 category 的汇总。

测试集只应触碰一次：开发阶段用 ``--split val``，最终报告再用 ``--split test``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  （把项目根目录加入 sys.path）
from minigpt.config import default_project_root, overlay_config
from minigpt.data.formatters import format_sft_prompt
from minigpt.data.sources import PretrainSource, SFTJsonlSource
from minigpt.eval.metrics import summarize_predictions
from minigpt.utils import atomic_write_json, ensure_dir, seed_everything, utc_now


def build_parser() -> argparse.ArgumentParser:
    """构造评估命令行参数（公共配置项由 overlay_config 补齐）。"""
    parser = argparse.ArgumentParser(description="Evaluate a MiniGPT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=["pretrain", "sft"], default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _load_bundle(checkpoint: str, device: str):
    """统一加载分派（minigpt.inference.load_bundle 按文件头识别 pure/torch）。"""

    from minigpt.inference import load_bundle

    return load_bundle(checkpoint, device=device)


def _num_parameters(bundle) -> int:
    """统计模型参数量，兼容 pure 的 ``num_parameters()`` 与 torch 的 ``parameters()``。"""
    model = bundle.model
    if hasattr(model, "num_parameters"):
        return int(model.num_parameters())
    return sum(parameter.numel() for parameter in model.parameters())


def _loss_for_batches(bundle, batches) -> tuple[float, int]:
    """在若干 batch 上按 token 数加权平均 loss。

    参数:
        bundle: 加载好的模型/后端/tokenizer 组合。
        batches: 可迭代的 batch 序列。
    返回:
        ``(平均 loss, 参与统计的 token 总数)``；非有限的 loss（NaN/Inf）会被
        跳过，避免单个异常 batch 污染整体指标。
    异常:
        RuntimeError: 所有 batch 都没有有效 token。
    """
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        result = bundle.backend.eval_step(bundle.model, batch)
        if not math.isfinite(result.loss):
            continue
        total_loss += result.loss * result.tokens
        total_tokens += result.tokens
    if total_tokens == 0:
        raise RuntimeError("no evaluation tokens")
    return total_loss / total_tokens, total_tokens


def _sample(bundle, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> str:
    """从给定 prompt 采样一段续写，仅用于在报告中展示模型当前输出风格。"""
    prompt_ids = bundle.tokenizer.encode(prompt)
    eos_id = getattr(bundle.tokenizer, "eos_id", None)
    generated = bundle.backend.generate(
        bundle.model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
    )
    return bundle.tokenizer.decode(generated[len(prompt_ids) :], skip_special=True)


def _read_jsonl(path: Path, limit: int) -> list[dict]:
    """读取 JSONL 记录，``limit`` 为 0 或负数时读取全部。"""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def _capability_boundary(path: Path, bundle, stage: str, split: str, report: dict) -> None:
    """写出能力边界说明（自动生成，指标来自本次评估）。

    该文件刻意保留「需人工复核」的措辞：自动指标只能说明分布拟合情况，
    不能替代对模型能力的定性判断。
    """
    lines = [
        "# Capability boundary（自动生成，需人工复核）",
        "",
        f"- 生成时间：{utc_now()}",
        f"- 阶段：{stage}",
        f"- 评估切分：{split}",
        f"- 模型：{bundle.payload.get('model_config')}",
        f"- 参数量：{_num_parameters(bundle)}",
        f"- 训练数据目录：{bundle.config.data_dir}",
        f"- 词表大小：{bundle.tokenizer.vocab_size}",
        "",
        "## 自动指标",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 预期边界（demo，需按实际观察修改）",
        "",
        "- 这是字符级小模型，不能替代通用助手，也不保证事实正确。",
        "- 能力严格受限于预训练语料与 SFT 数据覆盖范围。",
        "- 测试集只用于最终报告；调参请使用验证集。",
        "- 详细方法见 MANUAL.md 的“评估与能力边界”章节。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_path(value: str) -> Path:
    """相对路径按项目根解析（与 config.resolve_paths 的约定一致）。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = default_project_root() / path
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    """评估入口：按 stage 选择预训练或 SFT 指标，写出 evaluation.json 等产物。

    输出目录默认是 ``<checkpoint 所在目录>/eval_<split>/``，包含
    ``evaluation.json``、``capability_boundary.md``，SFT 还会额外写
    ``predictions.jsonl``。
    """
    args = overlay_config(build_parser(), argv, stage="sft")
    seed_everything(args.seed)
    checkpoint_path = _resolve_path(args.checkpoint)
    bundle = _load_bundle(str(checkpoint_path), args.device)
    if args.data_dir:
        bundle.config.data_dir = str(_resolve_path(args.data_dir))
    # stage 优先取显式参数；否则用 checkpoint 内记录的阶段；最后退回 pretrain。
    stage = args.stage or bundle.payload.get("stage", "pretrain")
    data_dir = Path(bundle.config.data_dir)
    out_dir = ensure_dir(
        str(_resolve_path(args.output_dir)) if args.output_dir else checkpoint_path.parent / f"eval_{args.split}"
    )

    if stage == "pretrain":
        source = PretrainSource(data_dir)
        # tokenizer 必须与训练时完全一致，否则 loss 没有可比性。
        if source.tokenizer.state_dict() != bundle.tokenizer.state_dict():
            raise ValueError("checkpoint tokenizer does not match the evaluation data")
        batches = source.eval_batches(
            split=args.split,
            batch_size=bundle.config.batch_size,
            block_size=bundle.config.block_size,
            num_batches=args.num_batches,
            seed=args.seed,
        )
        loss, tokens = _loss_for_batches(bundle, batches)
        sample_prompt = bundle.config.sample_prompt or "\n"
        sample = _sample(bundle, sample_prompt, 80, args.temperature, args.top_k, args.top_p)
        report = {
            "stage": stage,
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_dir": str(data_dir),
            "tokens": tokens,
            "loss": loss,
            "perplexity": math.exp(min(loss, 50.0)),  # 截断上限防止溢出
            "bpc": loss / math.log(2),
            "sample_prompt": sample_prompt,
            "sample": sample,
            "created_at": utc_now(),
        }
        atomic_write_json(out_dir / "evaluation.json", report)
        _capability_boundary(out_dir / "capability_boundary.md", bundle, stage, args.split, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # SFT：先算仅答案部分的 loss，再逐条生成并汇总生成指标。
    source = SFTJsonlSource(data_dir, tokenizer=bundle.tokenizer, block_size=bundle.config.block_size)
    if source.tokenizer.state_dict() != bundle.tokenizer.state_dict():
        raise ValueError("checkpoint tokenizer does not match the SFT data")
    batches = source.eval_batches(
        split=args.split,
        batch_size=bundle.config.batch_size,
        block_size=bundle.config.block_size,
        num_batches=args.num_batches,
        seed=args.seed,
    )
    answer_loss, tokens = _loss_for_batches(bundle, batches)

    split_file = data_dir / f"{args.split}.jsonl"
    records = _read_jsonl(split_file, args.max_samples)
    predictions = []
    eos_id = getattr(bundle.tokenizer, "eos_id", None)
    for record in records:
        question = str(record.get("q", ""))
        reference = str(record.get("a", ""))
        prompt = format_sft_prompt(question)
        prompt_ids = bundle.tokenizer.encode(prompt)
        # 用流式接口拿结束原因：EOS 会置 finished=True；跑满 max_new_tokens
        # 属于长度截断，不是 EOS。旧实现把"没跑满"一律当 EOS，口径错误。
        continuation: list[int] = []
        stopped_by_eos = False
        for item in bundle.backend.generate_stream(
            bundle.model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=eos_id,
        ):
            if item.get("finished"):
                stopped_by_eos = eos_id is not None and item["token_id"] == eos_id
                break
            continuation.append(int(item["token_id"]))
        prediction = bundle.tokenizer.decode(continuation, skip_special=True).strip()
        predictions.append(
            {
                "q": question,
                "reference": reference,
                "prediction": prediction,
                "category": record.get("category", "uncategorized"),
                "stopped_early": len(continuation) < args.max_new_tokens,
                "stopped_by_eos": stopped_by_eos,
                "truncated": len(continuation) >= args.max_new_tokens,
            }
        )
    metrics = summarize_predictions(predictions)
    # 两个比率单独计算并并回指标：它们描述的是「生成为何结束」，不是答案质量。
    metrics["eos_rate"] = sum(1 for record in predictions if record.get("stopped_by_eos")) / max(1, len(predictions))
    metrics["truncated_rate"] = sum(1 for record in predictions if record.get("truncated")) / max(1, len(predictions))
    report = {
        "stage": stage,
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(data_dir),
        "tokens": tokens,
        "answer_loss": answer_loss,
        "perplexity": math.exp(min(answer_loss, 50.0)),
        "metrics": metrics,
        "created_at": utc_now(),
    }
    atomic_write_json(out_dir / "evaluation.json", report)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _capability_boundary(out_dir / "capability_boundary.md", bundle, stage, args.split, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
