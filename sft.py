"""在 Q/A 形式的 JSONL 数据上做监督微调（SFT）。

样本模板固定为::

    Q: {question}
    A: {answer}<eos>

只有「答案 + EOS」参与 loss，问题与 padding 位置使用 ``-100`` 忽略索引。
配置 ``pretrain_mix_ratio`` 后会在每个 batch 中按比例混入预训练数据，用于
缓解灾难性遗忘。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401  （把项目根目录加入 sys.path）
from minigpt import cli_utils, registry
try:  # torch 可选：无 torch 环境仍可用 pure 后端做 SFT
    from minigpt.backends.torch_backend import TorchBackend  # noqa: F401  （注册 "torch" 后端）
except ImportError:
    pass
from minigpt.pure.backend import PureBackend  # noqa: F401  （注册 "pure" 后端）
from minigpt.config import parse_config, resume_enabled
from minigpt.data.sources import MixedSource, PretrainSource, SFTJsonlSource
from minigpt.training.checkpoint import load_checkpoint, validate_checkpoint
from minigpt.training.trainer import Trainer
from minigpt.utils import atomic_write_json, ensure_hash_seed, generate_run_id, safe_name, seed_everything


def _load_plugins(config) -> None:
    """加载 ``--plugins`` 指定的扩展模块（tokenizer / callback / backend 注册）。"""
    cli_utils.load_plugins(config)


def _tokenizer_from(payload):
    """从预训练 checkpoint 中恢复当时使用的 tokenizer。

    tokenizer 类型记录在 checkpoint 的 ``config.tokenizer_type`` 中，缺省为
    ``char``；必须恢复同一个 tokenizer，否则 SFT 数据的词表映射会错位。
    """
    tokenizer_cls = registry.TOKENIZERS.get(payload["config"].get("tokenizer_type", "char"))
    return tokenizer_cls.from_state(payload["tokenizer_state"])


def _build_trainer(config, resume: bool | str):
    """装配 SFT Trainer：恢复模型结构与 tokenizer、准备数据源与优化器。

    参数:
        config: 已解析的 SFT 配置。
        resume: 续训开关；为真时从 ``runs/<run_name>/last.*`` 恢复。
    返回:
        就绪的 :class:`~minigpt.training.trainer.Trainer`。
    异常:
        ValueError: 既没有 ``--init_from`` 也不是合法续训，或混入的预训练数据
        tokenizer 与 SFT 数据不一致。
    """
    if not config.init_from and not resume_enabled(resume):
        raise ValueError("SFT needs --init_from <pretrain checkpoint> (or --resume with --run_name)")

    # 1) 读取初始化权重：pure 的 .mgckpt 与 torch 的 .pt 走不同加载器。
    init_payload = None
    if config.init_from:
        from minigpt.pure.checkpoint import is_pure_checkpoint, load_pure_checkpoint

        if is_pure_checkpoint(config.init_from):
            init_payload = load_pure_checkpoint(config.init_from)
        else:
            init_payload = load_checkpoint(config.init_from, map_location="cpu")
    if init_payload is not None:
        validate_checkpoint(init_payload)
    tokenizer = _tokenizer_from(init_payload) if init_payload is not None else None

    # 2) 按后端分别恢复模型结构。模型超参以 checkpoint 为准，命令行只允许覆盖
    #    dropout 等与训练稳定性相关的项。
    backend = registry.BACKENDS.get(config.backend_type)(config)
    if init_payload is not None and backend.name == "pure":
        from minigpt.pure.gpt import PureGPT, PureGPTConfig

        model_config = PureGPTConfig.from_dict(init_payload["model_config"])
        model_config.dropout = config.dropout
        config.block_size = min(config.block_size, model_config.block_size)
        model = PureGPT(model_config)
        backend.load_state_dict(model, init_payload["model"])
        config.n_layer = model_config.n_layer
        config.n_head = model_config.n_head
        config.n_embd = model_config.n_embd
    elif init_payload is not None:
        from minigpt.models.gpt import GPT, GPTConfig

        model_config = GPTConfig.from_dict(init_payload["model_config"])
        model_config.dropout = config.dropout
        # SFT 可以使用比预训练更短的 block，但不能更长（位置编码表长度固定）。
        config.block_size = min(config.block_size, model_config.block_size)
        model = GPT(model_config).to(backend.device)
        backend.load_state_dict(model, init_payload["model"])
        # 让 run_config 记录真实使用的结构，而不是命令行默认值。
        config.n_layer = model_config.n_layer
        config.n_head = model_config.n_head
        config.n_embd = model_config.n_embd
    else:
        # 续训分支：run 目录里的 checkpoint 结构一致，等真实 tokenizer 就绪后
        # 再在下面统一建模型。
        tokenizer = None

    # 3) 构建 SFT 数据源；tokenizer 以 SFT 数据为准（init_from 时两者本应一致）。
    sft_source = SFTJsonlSource(config.data_dir, tokenizer=tokenizer, block_size=config.block_size)
    tokenizer = sft_source.tokenizer
    if init_payload is None:
        model = backend.build_model(config, tokenizer.vocab_size)
    data_source = sft_source
    if config.pretrain_mix_ratio > 0:
        pretrain_source = PretrainSource(config.pretrain_data_dir)
        if pretrain_source.tokenizer.state_dict() != tokenizer.state_dict():
            raise ValueError("pretrain mix data uses a different tokenizer than the SFT data")
        # 混采样使用全局 `random` 模块（不引入私有 RNG），这样 Trainer 写入
        # checkpoint 的 Python RNG 状态也能复现 SFT/预训练混合序列。
        data_source = MixedSource(
            sft_source,
            pretrain_source,
            pretrain_ratio=config.pretrain_mix_ratio,
        )

    seed_everything(config.seed)
    optimizer = backend.build_optimizer(model, config)
    trainer = Trainer(
        config=config,
        backend=backend,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        data_source=data_source,
        resume=resume,
    )
    return trainer


def main(argv: list[str] | None = None) -> int:
    """SFT 入口：解析配置 → 可选 mini-run → 构建 Trainer → 训练。

    返回:
        进程退出码：0 训练完成/正常停止或 mini-run 通过，1 训练失败或
        mini-run 未通过，2 参数组合非法或文件缺失。
    """
    config = parse_config("sft", argv)
    ensure_hash_seed(config.seed)
    _load_plugins(config)
    if resume_enabled(config.resume) and not config.run_name:
        print("--resume requires --run_name (the existing run directory)", file=sys.stderr)
        return 2
    if resume_enabled(config.resume) and config.mini_run:
        print("--resume cannot be combined with --mini_run", file=sys.stderr)
        return 2

    if config.mini_run:
        # mini-run 只验证「能过拟合一小段数据」：步数少、评估快、不存 best。
        base = config.run_name or generate_run_id("sft-mini", config.seed)
        mini = replace(
            config,
            run_name=safe_name(f"{base}-mini"),
            max_steps=min(60, config.max_steps),
            eval_interval=20,
            eval_iters=min(10, config.eval_iters),
            save_interval=10_000,
            patience=0,
            resume=False,
            mini_run=False,
        )
        trainer = _build_trainer(mini, resume=False)
        result = trainer.train()
        # ``final_train_loss`` 是最后不超过 5 个训练 loss 的尾部均值（由 Trainer
        # 计算），比单看最后一个 batch 更稳定。
        report = {
            "run_id": trainer.run_id,
            "run_dir": str(trainer.run_dir),
            "passed": result["final_train_loss"] < result["initial_loss"] - 0.05,
            "initial_loss": result["initial_loss"],
            "final_train_loss": result["final_train_loss"],
            "best_val": result["best_val"],
            "steps": result["steps"],
        }
        atomic_write_json(trainer.run_dir / "minirun_report.json", report)
        print("[sft-mini-run] " + ("PASS" if report["passed"] else "FAIL"))
        return 0 if report["passed"] else 1

    try:
        trainer = _build_trainer(config, resume=config.resume)
    except FileNotFoundError as error:
        # 典型场景：SFT 数据目录、init_from 文件或续训目标不存在。
        print(f"[sft] {error}", file=sys.stderr)
        return 2
    result = trainer.train()
    print(f"[sft] {result['status']} run_dir={result['run_dir']}")
    return 0 if result["status"] in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
