"""预训练字符级 MiniGPT。

示例
----
    python pretrain.py --preset debug_en --mini_run
    python pretrain.py --preset debug_en --gate
    python pretrain.py --preset debug_en --run_name my-run
    python pretrain.py --preset debug_zh --gate

``--mini_run``：只跑缩减版 200 步门槛检查（mini-run）并退出。
``--gate``：先跑门槛检查，通过后才用完整配置正式训练。
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401  （把项目根目录加入 sys.path）
from minigpt import cli_utils, registry
try:  # torch 可选：无 torch 环境仍可用 pure 后端训练
    from minigpt.backends.torch_backend import TorchBackend  # noqa: F401  （注册 "torch" 后端）
except ImportError:
    pass
from minigpt.pure.backend import PureBackend  # noqa: F401  （注册 "pure" 后端）
from minigpt.config import parse_config, resume_enabled
from minigpt.data.sources import PretrainSource
from minigpt.training.trainer import Trainer, check_mini_run
from minigpt.utils import atomic_write_json, ensure_hash_seed, generate_run_id, safe_name, seed_everything, utc_now


def _load_plugins(config) -> None:
    """加载 ``--plugins`` 指定的扩展模块（tokenizer / callback / backend 注册）。"""
    cli_utils.load_plugins(config)


def _build(config, resume: bool | str = False):
    """按配置装配数据源、后端、模型、优化器与 Trainer。

    参数:
        config: 已解析的预训练配置（``minigpt.config.Config``）。
        resume: 传给 Trainer 的续训开关，语义见 ``resume_enabled``。
    返回:
        可直接调用 ``train()`` 的 :class:`~minigpt.training.trainer.Trainer`。
    """
    source = PretrainSource(config.data_dir)
    backend = registry.BACKENDS.get(config.backend_type)(config)
    seed_everything(config.seed)
    model = backend.build_model(config, source.tokenizer.vocab_size)
    optimizer = backend.build_optimizer(model, config)
    trainer = Trainer(
        config=config,
        backend=backend,
        model=model,
        optimizer=optimizer,
        tokenizer=source.tokenizer,
        data_source=source,
        resume=resume,
    )
    return trainer


def _mini_config(config):
    """派生 mini-run 配置：缩小模型输入与步数，关闭早停、续训和 gate。

    目的是用几十秒验证「初始化 loss 合理、梯度有限、loss 在下降」，而不是
    评估模型质量，因此这里只保证规模足够小且不污染用户的原配置。
    """
    base_name = config.run_name or generate_run_id("pretrain-mini", config.seed)
    mini = replace(
        config,
        run_name=safe_name(f"{base_name}-mini"),
        block_size=max(8, config.block_size // 2),
        batch_size=max(1, config.batch_size // 2),
        grad_accum_steps=1,
        # 迷你运行不得反超用户设定的 max_steps（旧实现 max(50,…) 会放大）。
        max_steps=min(config.max_steps, min(200, max(50, config.max_steps))),
        eval_interval=20,
        eval_iters=min(10, config.eval_iters),
        log_interval=10,
        save_interval=10_000,
        warmup_steps=min(20, config.warmup_steps),
        patience=0,
        resume=False,
        mini_run=False,
        gate=False,
    )
    return mini


def _run_mini(config) -> tuple[bool, dict, str]:
    """执行一次 mini-run 门槛检查并落盘结果。

    参数:
        config: 用户原始配置；内部会派生出 mini 配置。
    返回:
        ``(是否通过, 检查报告, minirun_report.json 路径)``。
    副作用:
        在 run 目录写入 ``minirun_report.json``，并向 ``events.jsonl`` 追加一条
        ``minirun`` 事件，供 Web 端与日志观察。
    """
    mini = _mini_config(config)
    trainer = _build(mini, resume=False)
    result = trainer.train()
    report = check_mini_run(result, trainer.tokenizer.vocab_size)
    report["run_id"] = trainer.run_id
    report["run_dir"] = str(trainer.run_dir)
    report["config"] = mini.to_dict(portable=True)
    path = trainer.run_dir / "minirun_report.json"
    atomic_write_json(path, report)
    # 事件 schema 与训练期事件保持一致：v/ts/run_id/event/step/payload。
    event = {
        "v": 1,
        "ts": utc_now(),
        "run_id": trainer.run_id,
        "event": "minirun",
        "step": mini.max_steps,
        "payload": {
            "passed": report["passed"],
            "initial_loss": report["initial_loss"],
            "expected_initial_loss": report["expected_initial_loss"],
            "loss_drop": report["loss_drop"],
            "grad_norm_min": report["grad_norm_min"],
            "grad_norm_max": report["grad_norm_max"],
            "vocab_size": report["vocab_size"],
        },
    }
    with (trainer.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
    print("[mini-run] " + ("PASS" if report["passed"] else "FAIL"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(report["passed"]), report, str(path)


def main(argv: list[str] | None = None) -> int:
    """预训练入口：解析配置 → 可选 mini-run/gate → 构建 Trainer → 训练。

    返回:
        进程退出码：0 训练完成或被正常提前停止，1 门槛失败/训练失败，
        2 参数组合非法或数据缺失。
    """
    config = parse_config("pretrain", argv)
    ensure_hash_seed(config.seed)
    _load_plugins(config)
    # 这几个参数组合在语义上互斥，早失败比跑一半再报错更清晰。
    if config.mini_run and config.gate:
        print("--mini_run and --gate are mutually exclusive", file=sys.stderr)
        return 2
    if resume_enabled(config.resume) and (config.mini_run or config.gate):
        print("--resume cannot be combined with --mini_run/--gate", file=sys.stderr)
        return 2

    if config.mini_run or config.gate:
        passed, _report, _path = _run_mini(config)
        if config.mini_run or not passed:
            # 纯 mini_run：无论成败都到此结束；gate 失败也直接退出。
            return 0 if passed else 1
        # Gate 通过：清掉门槛标记，用用户原本的完整配置开一次全新的正式训练。
        config = replace(config, gate=False, mini_run=False)
        print("[gate] passed; starting full pretraining")

    try:
        trainer = _build(config, resume=config.resume)
    except FileNotFoundError as error:
        # 典型场景：数据目录或 resume 目标不存在。
        print(f"[pretrain] {error}", file=sys.stderr)
        return 2
    result = trainer.train()
    print(f"[pretrain] {result['status']} run_dir={result['run_dir']}")
    return 0 if result["status"] in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
