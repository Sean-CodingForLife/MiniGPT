"""MiniGPT Bundle v1: a portable, HF-style *self-contained* inference package.

Layering
--------
1. training checkpoint (``runs/<id>/best.pt``): model + optimizer + RNG;
2. bundle directory (this module): config / weights / tokenizer / model card /
   standalone runner / stdlib HTTP server, loadable without the repository;
3. ``.zip`` of that directory (``export_bundle(..., create_zip=True)``).

The bundle uses only Python standard library + PyTorch.  It is **not** loadable
by ``transformers.AutoModel``/``AutoTokenizer``; the custom character tokenizer
and model layout are MiniGPT-specific.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minigpt.pure.checkpoint import is_pure_checkpoint, load_pure_checkpoint
from minigpt.pure.weights import load_weights, save_weights
from minigpt.tokenization.char import CharTokenizer
from minigpt.training.checkpoint import load_checkpoint, validate_checkpoint

BUNDLE_FORMAT = "minigpt-bundle-v1"
BUNDLE_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]

RUNTIME_FILES = (
    "minigpt/__init__.py",
    "minigpt/registry.py",
    "minigpt/utils.py",
    "minigpt/sampling.py",
    "minigpt/models/__init__.py",
    "minigpt/models/gpt.py",
    "minigpt/tokenization/__init__.py",
    "minigpt/tokenization/base.py",
    "minigpt/tokenization/char.py",
    "minigpt/pure/__init__.py",
    "minigpt/pure/tensor.py",
    "minigpt/pure/gpt.py",
    "minigpt/pure/weights.py",
    "minigpt/pure/adamw.py",
)

TEMPLATES = ("run.py", "serve.py")


def _count_parameters(model_state: dict[str, Any], model_config: dict[str, Any]) -> int:
    """从权重形状统计参数量（ tied lm_head 与 embedding 共享，只计一次）。

    只为计数实例化整个 GPT 会白白分配一遍权重；形状乘积即可得到同一数字。
    """
    total = 0
    for key, value in model_state.items():
        if model_config.get("tie_weights", True) and key.startswith("lm_head."):
            continue
        shape = value.shape if hasattr(value, "shape") else value[0]
        count = 1
        for size in shape:
            count *= int(size)
        total += count
    return total


def _load_export_payload(checkpoint_path: Path) -> tuple[dict[str, Any], bool]:
    """按文件头分派检查点格式：``.mgckpt`` -> pure，其余 -> torch。"""
    if is_pure_checkpoint(checkpoint_path):
        return load_pure_checkpoint(checkpoint_path), True
    return load_checkpoint(checkpoint_path, map_location="cpu"), False


def _sha256(path: Path) -> str:
    """分块计算文件 sha256（每块 1 MiB），用于 MANIFEST 与校验。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    """返回 UTC ISO 8601 时间戳（秒级），写入导出产物的元数据。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dangerous_output_dirs() -> set[Path]:
    """返回绝不能被用作导出目标（以免被清空）的目录集合。"""
    home = Path.home().resolve()
    dirs = {ROOT, home, ROOT.parent}
    for name in ("Desktop", "Documents", "Downloads", "Music", "Pictures", "Videos"):
        dirs.add(home / name)
    return dirs


def _reject_dangerous_output(output: Path) -> None:
    """拒绝把导出目标设成文件系统根、用户主目录或项目根的上层目录。

    导出在 ``force=True`` 时会先删除目标目录，因此必须把「误删工作区」
    这类风险挡在写入之前。
    """
    if output.parent == output:  # 盘符根 / 文件系统根
        raise ValueError(f"refusing to export over a filesystem root: {output}")
    dangerous = _dangerous_output_dirs()
    if output in dangerous:
        raise ValueError(f"refusing to export over a protected directory: {output}")
    # 同时拒绝「项目根 / 用户主目录」的任一祖先目录（例如包含工作区的桌面文件夹）。
    for protected in (ROOT, Path.home().resolve()):
        if protected != output and protected.is_relative_to(output):
            raise ValueError(f"refusing to export over an ancestor of {protected}: {output}")


def _write_json(path: Path, data: Any) -> None:
    """按统一风格写 JSON（中文不转义、缩进 2、键排序、末尾换行）。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluation_for(run_dir: Path) -> dict[str, Any]:
    """从 run 目录里挑选一份评估结果，抽取关键指标写进模型卡。

    查找顺序：``eval_test/`` > ``eval_val/`` > run 根目录；文件损坏时跳过。
    只保留与读者相关的少量字段，避免把整份 report 塞进 README。
    """
    for name in ("eval_test/evaluation.json", "eval_val/evaluation.json", "evaluation.json"):
        path = run_dir / name
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            result: dict[str, Any] = {}
            for key in ("loss", "answer_loss", "perplexity", "bpc"):
                if key in data:
                    result[key] = data[key]
            if isinstance(data.get("metrics"), dict):
                metrics = data["metrics"]
                for key in ("count", "empty_rate", "exact_match", "char_f1", "repetition_rate", "eos_rate"):
                    if key in metrics:
                        result[key] = metrics[key]
            return result
    return {}


def _prompt_template(stage: str) -> str:
    """按阶段返回 prompt 模板：SFT 用 Q/A 模板，预训练直接用原文。"""
    return "Q: {prompt}\nA: " if stage == "sft" else "{prompt}"


def _data_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """从数据 meta 中挑出可公开的事实，绝不写绝对源路径。"""

    if not isinstance(meta, dict):
        return {}
    if isinstance(meta.get("sft"), dict):
        sft = _data_summary(meta["sft"])
        sft["pretrain"] = _data_summary(meta.get("pretrain") or {})
        sft["pretrain_mix_ratio"] = meta.get("pretrain_ratio")
        return {key: value for key, value in sft.items() if value is not None}

    source = meta.get("source") or {}
    summary: dict[str, Any] = {
        "dataset": meta.get("dataset"),
        "stage": meta.get("stage"),
        "tokenizer_type": meta.get("tokenizer_type"),
        "vocab_size": meta.get("vocab_size"),
        "val_fraction": meta.get("val_fraction"),
        "test_fraction": meta.get("test_fraction"),
        "unk_rate": meta.get("unk_rate"),
        "license": meta.get("license"),
        "format": meta.get("format"),
    }
    if isinstance(source, str):
        summary["source"] = source
    elif isinstance(source, dict):
        summary["source"] = {
            key: source.get(key)
            for key in ("kind", "name", "sha256", "book_ids", "chars", "cjk_ratio")
            if source.get(key) is not None
        }
    source_info = meta.get("source_info")
    if isinstance(source_info, dict):
        summary["source_info"] = {
            key: source_info.get(key)
            for key in ("kind", "name", "sha256", "bytes", "records_total", "license", "url")
            if source_info.get(key) is not None
        }
    counts = meta.get("counts")
    if isinstance(counts, dict):
        summary["counts"] = {
            key: counts.get(key)
            for key in ("input", "valid", "kept", "invalid", "duplicates")
            if counts.get(key) is not None
        }
    splits = meta.get("splits")
    if isinstance(splits, dict):
        summary["splits"] = {
            split: {
                key: value.get(key)
                for key in ("tokens", "chars", "bytes", "sha256", "examples")
                if isinstance(value, dict) and value.get(key) is not None
            }
            for split, value in splits.items()
        }
    group_split = meta.get("group_split")
    if isinstance(group_split, dict):
        summary["group_split"] = {
            key: group_split.get(key)
            for key in ("rule", "groups_total", "group_counts", "example_counts", "group_integrity")
            if group_split.get(key) is not None
        }
    extras = meta.get("vocab_extra")
    if isinstance(extras, dict):
        summary["vocab_extra"] = {
            key: extras.get(key) for key in ("enabled", "chars", "novel_chars", "note") if extras.get(key) is not None
        }
    return {key: value for key, value in summary.items() if value is not None}


def _readme_text(name: str, config: dict[str, Any]) -> str:
    """生成 bundle 的 README.md 文本（说明两种引擎、运行方式与文件清单）。"""
    params = config.get("source", {}).get("parameters")
    param_label = f"{params / 1e6:.2f}M" if isinstance(params, (int, float)) else "小型"
    return f"""# MiniGPT Bundle: {name}

这是 `{BUNDLE_FORMAT}` 推理包。它**不需要本仓库**：
- 默认用 PyTorch 引擎（需要安装 torch）；
- 也可以用 `--engine pure` 走纯 Python 自研张量引擎，**不需要 PyTorch**（权重来自 `weights.f32`）。

它不是 `transformers` 的 `AutoModel` 格式；当前不提供 Hugging Face `AutoModel`/`AutoTokenizer` 兼容。

## 快速运行

```bash
# 有 torch：默认引擎
python run.py --prompt "Hello"
# 没有 torch：纯 Python 引擎
python run.py --engine pure --prompt "Hello"
# 自动选择：有 torch 用 torch，否则用 pure
python run.py --engine auto --prompt "Hello"
python run.py --interactive
python serve.py --port 8000
```

然后：

```bash
curl -X POST http://127.0.0.1:8000/api/generate -H "Content-Type: application/json" -d "{{\\"prompt\\":\\"Hello\\",\\"max_new_tokens\\":40}}"
```

## 文件

| 文件 | 作用 |
|---|---|
| `config.json` | 模型结构、tokenizer 摘要、prompt 模板、生成参数、来源 |
| `pytorch_model.bin` | PyTorch 权重 state_dict（CPU） |
| `weights.f32` | torch-free 权重（float32 + JSON 索引 + sha256），供 pure 引擎使用 |
| `tokenizer.json` | 字符级 tokenizer 状态（本项目的 char tokenizer） |
| `special_tokens_map.json` | `<unk>/<eos>/<pad>` 映射 |
| `generation_config.json` | temperature/top-k/top-p/max_new_tokens |
| `run.py` / `serve.py` | 单文件 CLI / 标准库 HTTP 服务 |
| `minigpt/` | 运行所需的最小源码子集（含 `pure/` 引擎） |
| `MODEL_CARD.md` | 模型卡、数据、指标、限制、许可 |
| `MANIFEST.json` | 所有文件的 sha256/字节数 |

## 引擎差异

| 引擎 | 依赖 | 速度 | 采样 |
|---|---|---|---|
| `torch` | PyTorch | 快 | torch RNG |
| `pure` | 纯标准库 | 慢（0.81M 模型约 1.4–1.9 s/token） | Python `random.Random` |

两者用**同一份权重**；`temperature 0`（贪心）时输出逐 token 完全一致。

## 限制

- 示例权重通常来自 **{param_label} 字符级 debug demo**，不是可用助手；SFT 输出可能为空或语义错误。
- 推理包只含权重，不含优化器/RNG；训练恢复使用项目 `runs/<id>/best.pt`。
- `serve.py` 当前只支持 torch 引擎。
- 权重来源：{config.get("source", {}).get("run_id", "unknown")}。
"""


def _training_summary_for(run_dir: Path) -> dict[str, Any]:
    """读取训练 run 的 training_report.json 摘要（N12）；缺失/损坏时返回 {}。"""
    path = run_dir / "training_report.json"
    if not path.is_file():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(report, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("run_id", "stage", "status", "hyperparameters", "data_fingerprint", "metrics", "capability_boundary"):
        if key in report:
            summary[key] = report[key]
    return summary


#: 模型卡超参摘要优先展示的训练关键项（其余超参见 config.json）。
_SUMMARY_HYPER_KEYS = (
    "stage", "backend_type", "n_layer", "n_head", "n_embd", "block_size",
    "batch_size", "grad_accum_steps", "max_steps", "lr", "min_lr",
    "warmup_steps", "weight_decay", "grad_clip", "dropout", "seed",
    "patience", "min_delta", "eval_interval", "eval_iters",
)


def _training_summary_section(summary: dict[str, Any]) -> str:
    """把训练报告摘要渲染成模型卡段落；无摘要时返回空字符串。"""
    if not summary:
        return ""
    lines = ["## 训练摘要（来自 training_report.json）", ""]
    hyper = summary.get("hyperparameters")
    if isinstance(hyper, dict) and hyper:
        lines.append("### 超参数摘要")
        lines.append("")
        shown = [key for key in _SUMMARY_HYPER_KEYS if key in hyper]
        for key in shown:
            lines.append(f"- `{key}`: {hyper[key]}")
        lines.append(f"- 完整可移植超参见包内 `config.json` 的 `training_summary.hyperparameters`（共 {len(hyper)} 项）。")
        lines.append("")
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics:
        lines.append("### 训练指标汇总")
        lines.append("")
        for key in ("initial_loss", "final_train_loss", "best_val_loss", "best_step", "steps", "elapsed_s"):
            if key in metrics:
                lines.append(f"- `{key}`: {metrics[key]}")
        lines.append("")
    fingerprint = summary.get("data_fingerprint")
    if isinstance(fingerprint, dict) and fingerprint:
        lines.append("### 数据指纹（sha256）")
        lines.append("")
        for key, value in fingerprint.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    boundary = summary.get("capability_boundary")
    if isinstance(boundary, str) and boundary:
        lines.append("### 能力边界")
        lines.append("")
        lines.append(boundary)
        lines.append("")
    return "\n".join(lines) + "\n"


def _model_card_text(name: str, config: dict[str, Any]) -> str:
    """生成 MODEL_CARD.md：结构、参数量、训练数据摘要、评估指标与能力边界。"""
    model = config["model_config"]
    source = config.get("source", {})
    evaluation = config.get("evaluation", {})
    data_blob = json.dumps(config.get("data", {}), ensure_ascii=False).lower()
    uses_nc = "by-nc" in data_blob or "alpaca" in data_blob
    if uses_nc:
        data_origin_note = (
            "本包训练数据含公开指令数据（如 alpaca-zh，CC BY-NC 4.0），权重受其许可约束，"
            "仅限研究/非商业用途；详见下方许可与训练数据字段。"
        )
    else:
        data_origin_note = "当前示例包只使用合成/公版数据。"
    params = source.get("parameters")
    param_label = f"{params / 1e6:.2f}M" if isinstance(params, (int, float)) else "小型"
    metrics = "\n".join(f"- `{key}`: {value}" for key, value in evaluation.items()) or "- 未检测到 evaluation.json；本包未附带测试指标。"
    return f"""# Model Card: {name}

## 模型信息

- 包格式：`{BUNDLE_FORMAT}` v{config.get("format_version")}
- 来源 run：`{source.get("run_id")}`
- 阶段：`{source.get("stage")}`
- 结构：{model.get("n_layer")} 层 / {model.get("n_head")} 头 / {model.get("n_embd")} 维 / block {model.get("block_size")}
- 词表：{model.get("vocab_size")}（字符级 + 特殊 token）
- 参数量：{source.get("parameters", "见权重文件")}
- 生成模板：`{config.get("prompt_template")!r}`

## 训练数据

{json.dumps(config.get("data", {}), ensure_ascii=False, indent=2)[:4000]}

## 评估（来源训练 run）

{metrics}

{_training_summary_section(config.get("training_summary") or {})}
## 预期用途与限制

- 这是**方法/教学 demo 包**，不是通用助手，不应作为事实知识库或生产服务。
- 字符级 {param_label} 模型只能学习局部字符结构；SFT 只教格式，不能保证回答正确。
- 输入超过 `block_size` 会被截断；生成最多 `max_new_tokens` 个 token。
- 评估指标不能外推到其他数据；请在自己的验证集上重新评估。
- 公开数据（如 alpaca-zh）训练的权重受其许可证约束；{data_origin_note}

## 许可

- 代码：MIT（随包附 `LICENSE`）。
- 由合成数据/公版文本生成的 demo 权重：CC BY 4.0；不含保证。
- Alpaca-ZH 权重受 CC BY-NC 4.0（仅研究、非商业）约束；使用该数据训练的包须在模型卡中注明。

## 文件与完整性

- 所有文件 sha256/字节数见 `MANIFEST.json`；运行时不会联网。
"""


def export_bundle(
    checkpoint: str | Path,
    output_dir: str | Path,
    create_zip: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Export a training checkpoint to a portable bundle directory (+ zip)."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    # Guard FIRST: never delete anything before the destination is proven safe.
    _reject_dangerous_output(output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if output.exists():
        if not force:
            raise FileExistsError(f"output already exists: {output} (use --force)")
        shutil.rmtree(output)

    payload, from_pure = _load_export_payload(checkpoint_path)
    validate_checkpoint(payload)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None and not from_pure:
        raise ValueError("exporting a torch checkpoint requires PyTorch; only pure (.mgckpt) checkpoints can export without it")
    model_config = dict(payload["model_config"])
    # The torch-free weights.f32 copy drops lm_head/bias; exporting a bias=True
    # or untied model would silently corrupt the pure engine. Refuse instead.
    if model_config.get("bias") or not model_config.get("tie_weights", True):
        raise ValueError(
            "cannot export a bias=True or tie_weights=False model: the pure engine "
            "does not support these; export a default (bias=False, tied) checkpoint"
        )
    tokenizer_state = payload["tokenizer_state"]
    stage = str(payload.get("stage", "pretrain"))
    run_id = str(payload.get("run_id") or checkpoint_path.parent.name)
    name = output.name
    parameter_count = _count_parameters(payload["model"], model_config)
    config = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "model_type": "minigpt",
        "architectures": ["MiniGPT"],
        "model_config": model_config,
        "tokenizer": {
            "type": tokenizer_state.get("type", "char"),
            "vocab_size": tokenizer_state.get("vocab_size", model_config.get("vocab_size")),
            "specials": tokenizer_state.get("specials", {}),
        },
        "prompt_template": _prompt_template(stage),
        "generation_config": {
            "max_new_tokens": int(payload.get("config", {}).get("max_new_tokens", 200)),
            "temperature": float(payload.get("config", {}).get("temperature", 0.8)),
            "top_k": int(payload.get("config", {}).get("top_k", 40)),
            "top_p": float(payload.get("config", {}).get("top_p", 1.0)),
            "eos_token": "<eos>",
        },
        "source": {
            "run_id": run_id,
            "stage": stage,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_file": checkpoint_path.name,
            "parameters": parameter_count,
            "created_at": _utc_now(),
            "torch_version": torch.__version__ if torch is not None else None,
        },
        "data": _data_summary(payload.get("data_meta") or {}),
        "evaluation": _evaluation_for(checkpoint_path.parent),
        "limitations": [
            "Character-level demo model; not a general assistant.",
            "SFT output may be empty or semantically wrong.",
            "Not directly loadable by transformers.AutoModel without an adapter.",
        ],
    }
    # N12: 附带来源 run 的训练报告摘要（超参/数据指纹/指标/能力边界）。
    training_summary = _training_summary_for(checkpoint_path.parent)
    if training_summary:
        config["training_summary"] = training_summary

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    _write_json(output / "tokenizer.json", tokenizer_state)
    _write_json(
        output / "special_tokens_map.json",
        {
            "unk_token": "<unk>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "specials": tokenizer_state.get("specials", {}),
        },
    )
    _write_json(output / "generation_config.json", config["generation_config"])
    if from_pure:
        # pure 检查点已是 (shape, values) 列表：补回 tied lm_head 后转 torch 张量。
        pure_state = {
            key: (tuple(int(size) for size in shape), [float(value) for value in values])
            for key, (shape, values) in payload["model"].items()
        }
        if torch is not None:
            state = {
                key: torch.tensor(values, dtype=torch.float32).reshape(shape)
                for key, (shape, values) in pure_state.items()
            }
            if model_config.get("tie_weights", True) and "transformer.wte.weight" in state:
                state["lm_head.weight"] = state["transformer.wte.weight"]
    else:
        state = {key: value.detach().cpu() for key, value in payload["model"].items()}
        # Torch-free copy for the pure engine: skip the tied LM head and (absent) biases.
        pure_state = {}
        for key, value in state.items():
            if key.startswith("lm_head.") or key.endswith(".bias"):
                continue
            pure_state[key] = (tuple(value.shape), value.reshape(-1).tolist())
    if torch is not None:
        torch.save(state, output / "pytorch_model.bin")
    save_weights(
        output / "weights.f32",
        pure_state,
        {
            "run_id": run_id,
            "stage": stage,
            "model_config": model_config,
            "torch_version": torch.__version__ if torch is not None else None,
        },
    )

    for relative in RUNTIME_FILES:
        source = ROOT / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for template in TEMPLATES:
        shutil.copy2(ROOT / "bundle_templates" / template, output / template)
    shutil.copy2(ROOT / "requirements.txt", output / "requirements.txt")
    if (ROOT / "LICENSE").is_file():
        shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(_readme_text(name, config), encoding="utf-8")
    (output / "MODEL_CARD.md").write_text(_model_card_text(name, config), encoding="utf-8")

    files = {}
    for path in sorted(output.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_file() and path.name != "MANIFEST.json":
            files[str(path.relative_to(output)).replace(os.sep, "/")] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "created_at": _utc_now(),
        "source": config["source"],
        "files": files,
    }
    _write_json(output / "MANIFEST.json", manifest)

    zip_path = None
    if create_zip:
        zip_path = output.parent / f"{output.name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                if path.is_file():
                    relative = str(path.relative_to(output)).replace(os.sep, "/")
                    archive.write(path, arcname=f"{output.name}/{relative}")

    return {
        "bundle": str(output),
        "zip": str(zip_path) if zip_path else None,
        "manifest": manifest,
        "config": config,
    }


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """校验 MANIFEST 哈希/字节数，并确认模型与 tokenizer 能真正加载。

    * 先逐文件核对 sha256 与字节数，再拒绝任何未登记在 MANIFEST 中的文件
      （防止往包内塞入会被 `run.py` 导入的同名影子模块）；
    * 有 `pytorch_model.bin` 时用 torch 加载，并（若存在 `weights.f32`）比对
      两者数值一致；纯引擎包只有 `weights.f32` 时用 pure 模型加载校验。
    """

    directory = Path(bundle_dir).expanduser().resolve()
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing MANIFEST.json: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format: {manifest.get('format')!r}")
    errors = []
    for relative, info in manifest.get("files", {}).items():
        path = directory / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if _sha256(path) != info.get("sha256"):
            errors.append(f"hash mismatch: {relative}")
        if path.stat().st_size != info.get("bytes"):
            errors.append(f"size mismatch: {relative}")
    # Reject files that are not registered in the manifest. run.py inserts the
    # bundle root at sys.path[0], so an unregistered shadow module (e.g. a
    # planted torch.py) would otherwise pass verification and get imported.
    registered = set(manifest.get("files", {}))
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(directory)).replace(os.sep, "/")
        if relative == "MANIFEST.json" or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if relative not in registered:
            errors.append(f"unregistered file: {relative}")
    if errors:
        raise ValueError("bundle verification failed:\n" + "\n".join(errors))

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(directory / "tokenizer.json")
    weights_path = directory / "weights.f32"
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        # 无 torch：仅 pure 权重的包用 pure 引擎核对可加载性。
        if not weights_path.is_file():
            raise ValueError("bundle has no weights.f32 and PyTorch is unavailable; cannot verify")
        from minigpt.pure.gpt import PureGPT, PureGPTConfig

        pure_state, _ = load_weights(weights_path)
        model = PureGPT(PureGPTConfig.from_dict(config["model_config"]))
        model.load_state_dict(pure_state)
        return {"bundle": str(directory), "config": config, "tokenizer": tokenizer, "model": model}
    from minigpt.models.gpt import GPT, GPTConfig

    if not (directory / "pytorch_model.bin").is_file():
        # 纯引擎包（例如 dist/pure-sft）只有 weights.f32：即使本机装了 torch，
        # 也应改用 pure 模型核对可加载性，而不是要求不存在的 pytorch_model.bin。
        if not weights_path.is_file():
            raise ValueError("bundle has neither pytorch_model.bin nor weights.f32; cannot verify")
        from minigpt.pure.gpt import PureGPT, PureGPTConfig

        pure_state, _ = load_weights(weights_path)
        model = PureGPT(PureGPTConfig.from_dict(config["model_config"]))
        model.load_state_dict(pure_state)
        return {"bundle": str(directory), "config": config, "tokenizer": tokenizer, "model": model}

    model = GPT(GPTConfig.from_dict(config["model_config"]))
    state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if weights_path.is_file():
        pure_state, _ = load_weights(weights_path)
        torch_state = {
            key: value
            for key, value in state.items()
            if not key.startswith("lm_head.") and not key.endswith(".bias")
        }
        if set(pure_state) != set(torch_state):
            raise ValueError("weights.f32 tensor names do not match pytorch_model.bin")
        worst = 0.0
        for key, tensor in torch_state.items():
            expected = tensor.reshape(-1).tolist()
            worst = max(worst, max(abs(left - right) for left, right in zip(pure_state[key][1], expected)))
        if worst > 1e-4:
            raise ValueError(f"weights.f32 differs from pytorch_model.bin (max |diff| {worst:.3g})")
    return {"bundle": str(directory), "config": config, "tokenizer": tokenizer, "model": model}


__all__ = ["BUNDLE_FORMAT", "BUNDLE_VERSION", "export_bundle", "verify_bundle"]
