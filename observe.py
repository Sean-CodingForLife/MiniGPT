"""观察一个 MiniGPT 模型：参数量、loss 合理性、下一个 token 概率与注意力。

用于训练前/后的手动检查：既可以直接看随机初始化模型的初始 loss 是否接近
``ln(vocab_size)``，也可以加载 checkpoint 看真实采样分布与注意力热力图。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import _bootstrap  # noqa: F401  （把项目根目录加入 sys.path）
from minigpt import registry
try:  # torch 可选：无 torch 环境仍可观察 .mgckpt（pure 引擎）
    from minigpt.backends.torch_backend import TorchBackend  # noqa: F401  （注册 "torch" 后端）
except ImportError:
    pass
from minigpt.pure.backend import PureBackend  # noqa: F401  （注册 "pure" 后端）
from minigpt.config import get_preset, overlay_config
from minigpt.data.sources import PretrainSource
from minigpt.utils import seed_everything


def _attention_text(values: list[float]) -> str:
    """把一行 [0,1] 的注意力权重映射成 ASCII 灰度字符，供终端可视化。"""
    chars = " .:-=+*#%@"
    return "".join(chars[min(len(chars) - 1, int(max(0.0, min(1.0, v)) * (len(chars) - 1)))] for v in values)


def _load_bundle(checkpoint: str, device: str, data_dir: str | None):
    """统一加载分派（minigpt.inference.load_bundle 按文件头识别 pure/torch）。

    这里先按后缀做一次显式校验，让「传错文件」的报错比加载器内部更直白。
    """

    suffix = Path(checkpoint).suffix.lower()
    if suffix not in {".pt", ".mgckpt"}:
        raise ValueError(
            f"无法识别的检查点类型：{checkpoint!r}（后缀 {suffix or '<无>'}）；支持 .pt（torch）与 .mgckpt（pure）"
        )
    from minigpt.inference import load_bundle

    return load_bundle(checkpoint, device=device, data_dir=data_dir)


def _parameter_counts(model, backend_name: str) -> tuple[int, int]:
    """返回 ``(总参数量, 词表相关 embedding 参数量)``，兼容 pure / torch 两种模型对象。

    embedding 统计 ``wte`` + ``wpe``；由于输出头与 wte 绑定（tie_weights），
    总参数量不会重复计入 lm_head。
    """

    if backend_name == "pure":
        total = model.num_parameters()
        embedding = model.params["transformer.wte.weight"].numel + model.params["transformer.wpe.weight"].numel
        return total, embedding
    total = sum(p.numel() for p in model.parameters())
    embedding = model.transformer["wte"].weight.numel() + model.transformer["wpe"].weight.numel()
    return total, embedding


def _next_token_trace(model, backend, ids: list[int], top_k: int):
    """计算最后一个位置上的 top-k 概率，并尽可能返回注意力张量。

    返回:
        ``(top_probs, top_ids, attention_or_None)``。torch 路径返回每层注意力
        权重（可用于 ASCII 可视化）；pure 引擎不记录注意力，返回 ``None``。
    """

    if backend.name == "pure":
        logits, _ = model.forward([list(ids)])
        width = model.config.vocab_size
        start = (len(ids) - 1) * width
        row = logits.data[start : start + width]
        # 手写数值稳定 softmax：先减最大值再取指数，避免上溢。
        maximum = max(row)
        exps = [math.exp(value - maximum) for value in row]
        total = sum(exps)
        probs = [value / total for value in exps]
        order = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)
        order = order[: max(0, min(int(top_k), len(order)))]
        return [probs[index] for index in order], order, None
    import torch

    x = torch.tensor([ids], dtype=torch.long, device=torch.device(getattr(backend, "device", "cpu")))
    with torch.no_grad():
        logits, _, attns = model(x, return_attns=True)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    values, indices = torch.topk(probs, min(top_k, probs.numel()))
    return values.tolist(), indices.tolist(), attns


def main(argv: list[str] | None = None) -> int:
    """观察入口：加载 checkpoint（或按预设随机初始化）并打印诊断信息。"""
    parser = argparse.ArgumentParser(description="Inspect a MiniGPT model")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--preset", default="debug_en")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    args = overlay_config(parser, argv, stage="pretrain", field_map={"prompt": "sample_prompt"})
    seed_everything(args.seed)

    if args.checkpoint:
        # 已有 checkpoint：结构与 tokenizer 全部从文件恢复，stage 用于展示来源。
        bundle = _load_bundle(args.checkpoint, device=args.device, data_dir=args.data_dir)
        model, backend, tokenizer, config = bundle.model, bundle.backend, bundle.tokenizer, bundle.config
        stage = bundle.payload.get("stage")
    else:
        # 没有 checkpoint：按预设建一个随机初始化的模型，用来核对初始 loss。
        config = get_preset(args.preset)
        if args.data_dir:
            config.data_dir = args.data_dir
        config.resolve_paths()
        source = PretrainSource(config.data_dir)
        tokenizer = source.tokenizer
        if config.backend_type == "pure":
            import minigpt.pure.backend  # noqa: F401  （注册 pure 后端）
        backend = registry.BACKENDS.get(config.backend_type)(config)
        model = backend.build_model(config, tokenizer.vocab_size)
        stage = "random"

    total, embedding = _parameter_counts(model, backend.name)
    print(f"stage={stage} params={total:,} embedding_params={embedding:,} non_embedding={total - embedding:,}")
    print(f"config={model.config.to_dict()} device={backend.device_info()}")
    # 均匀分布下的理论初始 loss 是 ln(V)；实测值应与之接近（字符级小词表尤其明显）。
    expected = math.log(max(2, tokenizer.vocab_size))
    print(f"vocab={tokenizer.vocab_size} initial_loss_expected~{expected:.4f}")

    if args.data_dir:
        # 可选：抽一个训练 batch 计算真实 loss，用于判断数据/权重是否匹配。
        try:
            source = PretrainSource(Path(args.data_dir))
            batch = next(iter(source.eval_batches("train", 2, min(32, model.config.block_size), 1, seed=args.seed)))
            result = backend.eval_step(model, batch)
            print(f"sample_batch_loss={result.loss:.4f} tokens={result.tokens}")
        except Exception as error:
            # 数据检查是附加信息，失败不应中断主流程。
            print(f"[data check skipped] {type(error).__name__}: {error}")

    prompt = args.prompt if args.prompt is not None else (config.sample_prompt or "ROMEO:")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt 编码为空，无法观察下一个 token")
    if len(ids) > model.config.block_size:
        # 只保留尾部，保证不超过模型上下文长度。
        ids = ids[-model.config.block_size :]
    values, indices, attns = _next_token_trace(model, backend, ids, args.top_k)
    print("next-token top-k:")
    for value, index in zip(values, indices):
        print(f"  {tokenizer.token_repr(index)!r}: {value:.4f}")

    if args.attention:
        if attns is None:
            raise RuntimeError(
                "pure 引擎不记录注意力权重，--attention 仅支持 torch 检查点（.pt）"
            )
        import torch

        print("attention entropy (nats):")
        for layer_index, attn in enumerate(attns):
            # 张量形状 (B, H, T, T)：对 query 维取平均，得到每个 head 的熵。
            per_head = []
            for head_index in range(attn.shape[1]):
                matrix = attn[0, head_index]
                entropy = float(-(matrix * torch.log(matrix.clamp_min(1e-9))).sum(dim=-1).mean().item())
                per_head.append(entropy)
            print(f"  layer {layer_index}: " + ", ".join(f"h{h}={e:.3f}" for h, e in enumerate(per_head)))
        # 用户指定的 layer/head 越界时收敛到合法范围；矩阵过大只取最后 32 个 token。
        layer = max(0, min(args.layer, len(attns) - 1))
        head = max(0, min(args.head, attns[layer].shape[1] - 1))
        matrix = attns[layer][0, head].detach().cpu()
        limit = min(32, matrix.shape[0])
        print(f"ASCII attention layer={layer} head={head} (last {limit} tokens):")
        for row in matrix[-limit:, -limit:].tolist():
            print("  " + _attention_text(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
