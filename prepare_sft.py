"""SFT 数据准备 CLI。

数据来源
--------
``--source/--sft_source`` 可选：

* ``synthetic_en`` / ``synthetic_zh`` —— 仓库内 ``data/demo_sft_{en,zh}.jsonl``
  （由 ``scripts/build_demo_data.py`` 确定性生成，各 500 条）；
* ``alpaca_zh`` —— 公开数据集 ``shibing624/alpaca-zh``（hf-mirror，约 35MB，
  CC BY-NC 4.0，仅研究；**只缓存不提交**），按 seed 确定性采样 ``max_samples`` 条；
* ``jsonl`` —— ``--input/--sft_input`` 指定的本地 JSONL/JSON。

处理流程
--------
字段规范：``q``、``a`` 必填；``group/category/source`` 可选，``group`` 缺省为
归一化后的问题文本。校验空值/超长/重复后，按 **group** 做 90/5/5 切分（同一
group 绝不跨 train/val/test），写出 ``train.jsonl``、``val.jsonl``、
``test.jsonl``、``sft_meta.json``、``split_report.json``；若 ``--tokenizer_dir``
（或 ``pretrain_data_dir/tokenizer.json``）可用，额外报告 SFT 文本的 tokenizer
覆盖率。

示例::

    python prepare_sft.py --source synthetic_en
    python prepare_sft.py --source synthetic_zh --data_dir data/sft_demo_zh
    python prepare_sft.py --source alpaca_zh --max_samples 800
    python prepare_sft.py --source jsonl --input my_sft.jsonl --data_dir data/sft_custom

全程只用 Python 标准库；不需要 torch。
"""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  确保任意 cwd 下都能 import minigpt

from minigpt import cli_utils
from minigpt.config import parse_config
from minigpt.data.binary import atomic_write_json, read_json, read_jsonl, sha256_file, write_jsonl
from minigpt.data.download import (
    ALPACA_ZH_LICENSE,
    ALPACA_ZH_URL,
    DownloadError,
    download_alpaca_zh,
)
from minigpt.data.formatters import (
    SFT_FORMAT_TEMPLATE,
    format_sft_prompt,
    record_identity,
    validate_record,
)
from minigpt.tokenization import CharTokenizer

#: 支持的 SFT 数据源。
SOURCES = ("synthetic_en", "synthetic_zh", "alpaca_zh", "jsonl")

#: preset -> 未显式指定 source 时的默认来源。
_PRESET_DEFAULT_SOURCE = {
    "sft_debug_en": "synthetic_en",
    "sft_debug_zh": "synthetic_zh",
    "sft_10m": "alpaca_zh",
}

#: 各来源默认的数据目录名（用于未显式指定 --data_dir 时推导）。
_SOURCE_DATA_DIR = {
    "synthetic_en": "sft_demo_en",
    "synthetic_zh": "sft_demo_zh",
    "alpaca_zh": "sft_alpaca_zh",
}
_DEFAULT_SFT_DATA_DIR_NAMES = {"sft_demo_en", "sft_demo_zh", "sft_alpaca_zh", "sft_10m"}

#: 常用 CLI 别名 -> Config 字段名（``--source`` 与 CONSTRUCTION 的 sft_source 对应）。
_ARGT_ALIASES = {
    "--source": "--sft_source",
    "--input": "--sft_input",
}


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _translate_argv(argv: list[str]) -> list[str]:
    """把友好别名（--source/--input）翻译成 Config 字段参数。"""
    return cli_utils.translate_argv(argv, _ARGT_ALIASES)


def _has_flag(argv: list[str], flag: str) -> bool:
    """判断命令行里是否显式出现过某个参数。"""
    return cli_utils.has_flag(argv, flag)


def _now_iso() -> str:
    """返回带时区的 ISO 8601 时间戳，写入 meta/report 的 created_at。"""
    return cli_utils.now_iso()


def _preset_name(argv: list[str], default: str = "sft_debug_en") -> str:
    """从原始 argv 中提取 ``--preset``（兼容 ``--preset=x``），用于推导默认来源。

    这里刻意不读最终 config：来源选择要在解析前决定，而别名翻译可能改变参数形态。
    """
    for index, arg in enumerate(argv):
        if arg == "--preset" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--preset="):
            return arg.split("=", 1)[1]
    return default


def _resolve_input_path(config: Any) -> Path:
    """sft_input 已在 config.resolve_paths() 中解析为绝对路径；这里做兜底。"""
    return cli_utils.resolve_input_path(config.sft_input, config.project_root)


def _derive_data_dir(config: Any, source: str, argv: list[str]) -> bool:
    """未显式指定 data_dir 时按来源推导默认目录（与 preset 目录风格一致）。"""
    if _has_flag(argv, "--data_dir") or "MINIGPT_DATA_DIR" in os.environ:
        return False
    root = Path(config.project_root)
    current = Path(config.data_dir)
    if current.parent != root / "data" or current.name not in _DEFAULT_SFT_DATA_DIR_NAMES:
        return False
    if source in _SOURCE_DATA_DIR:
        candidate = root / "data" / _SOURCE_DATA_DIR[source]
    elif source == "jsonl":
        stem = Path(config.sft_input).stem if config.sft_input else "jsonl"
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._")[:48] or "jsonl"
        candidate = root / "data" / f"sft_{safe}"
    else:
        return False
    if candidate == current:
        return False
    config.data_dir = str(candidate)
    return True


# ----------------------------------------------------------------------
# 来源加载
# ----------------------------------------------------------------------
def _load_source_records(config: Any, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回 (原始记录列表, 来源元数据)。"""
    project_data = Path(config.project_root) / "data"
    if config.sft_input:
        path = _resolve_input_path(config)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 SFT 输入文件：{path}")
        return _load_records_file(path, source)

    if source in ("synthetic_en", "synthetic_zh"):
        name = "demo_sft_en.jsonl" if source == "synthetic_en" else "demo_sft_zh.jsonl"
        path = project_data / name
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到合成数据 {path}；请先运行 scripts/build_demo_data.py 生成演示数据。"
            )
        return _load_records_file(path, source)

    if source == "jsonl":
        raise ValueError("jsonl 来源需要 --input/--sft_input 指定本地 JSONL/JSON 文件")

    if source == "alpaca_zh":
        print(f"[prepare_sft] 获取 alpaca-zh（{ALPACA_ZH_LICENSE}，只缓存不提交）")
        cache = download_alpaca_zh(config.download_dir, url=str(config.source_url or ALPACA_ZH_URL), verbose=True)
        try:
            payload = read_json(cache)
        except (OSError, ValueError) as exc:
            raise ValueError(f"alpaca-zh 缓存不是合法 JSON：{cache}（{exc}）") from exc
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"alpaca-zh 数据不是非空 JSON 数组：{cache}")
        records: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("instruction") or "").strip()
            extra_input = str(item.get("input") or "").strip()
            output = str(item.get("output") or "").strip()
            question = instruction if not extra_input else f"{instruction}\n{extra_input}"
            records.append(
                {
                    "q": question,
                    "a": output,
                    "category": "alpaca_zh",
                    "source": "alpaca_zh",
                }
            )
        info = {
            "kind": "public_dataset",
            "name": "alpaca_zh",
            "path": str(cache),
            "sha256": sha256_file(cache),
            "bytes": cache.stat().st_size,
            "records_total": len(payload),
            "license": ALPACA_ZH_LICENSE,
            "url": str(config.source_url or ALPACA_ZH_URL),
        }
        return records, info

    raise ValueError(f"未知 source {source!r}；可选：{', '.join(SOURCES)}")


def _load_records_file(path: Path, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取本地 JSON（数组）或 JSONL 文件，统一返回记录列表。"""
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"{path} 不是 JSON 数组")
        records = [item for item in payload if isinstance(item, dict)]
        skipped = len(payload) - len(records)
        if skipped:
            print(f"[prepare_sft] 警告：{path} 有 {skipped} 条非对象记录被忽略")
    else:
        records = read_jsonl(path)
    info = {
        "kind": "local_file",
        "name": source,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "records_total": len(records),
    }
    return records, info


# ----------------------------------------------------------------------
# 校验 / 采样 / 切分
# ----------------------------------------------------------------------
def _validate_and_dedupe(
    raw_records: list[dict[str, Any]], default_source: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """校验 + 去重，返回 (有效记录, 统计信息)。"""
    valid: list[dict[str, str]] = []
    invalid_examples: list[str] = []
    seen: set[tuple[str, str]] = set()
    invalid_count = 0
    duplicate_count = 0
    for index, raw in enumerate(raw_records):
        try:
            record = validate_record(raw, default_source=default_source)
        except ValueError as exc:
            invalid_count += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(f"#{index}: {exc}")
            continue
        identity = record_identity(record)
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        valid.append(record)
    stats = {
        "input": len(raw_records),
        "invalid": invalid_count,
        "duplicates": duplicate_count,
        "valid": len(valid),
        "invalid_examples": invalid_examples,
    }
    return valid, stats


def _cap_samples(records: list[dict[str, str]], max_samples: int, seed: int) -> tuple[list[dict[str, str]], int]:
    """超过 max_samples 时按 seed 确定性采样（保留原顺序）。"""
    max_samples = int(max_samples)
    if max_samples <= 0 or len(records) <= max_samples:
        return records, len(records)
    rng = random.Random(f"sample:{seed}")
    indices = sorted(rng.sample(range(len(records)), max_samples))
    return [records[index] for index in indices], max_samples


def _split_counts(group_count: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    """按 group 数量计算 val/test 组数，保证 train 至少 1 组。"""
    if group_count <= 1:
        return 0, 0
    n_val = int(group_count * float(val_fraction))
    n_test = int(group_count * float(test_fraction))
    if val_fraction > 0 and n_val == 0 and group_count >= 3:
        n_val = 1
    if test_fraction > 0 and n_test == 0 and group_count - n_val >= 2:
        n_test = 1
    while group_count - n_val - n_test < 1 and (n_val or n_test):
        if n_test:
            n_test -= 1
        elif n_val:
            n_val -= 1
    return n_val, n_test


def _split_by_group(
    records: list[dict[str, str]], val_fraction: float, test_fraction: float, seed: int
) -> tuple[dict[str, list[dict[str, str]]], dict[str, set[str]], dict[str, Any]]:
    """按 group 做 90/5/5 切分，返回 (切分记录, 切分 group 集合, 报告)。"""
    groups = sorted({record["group"] for record in records})
    rng = random.Random(f"split:{seed}")
    rng.shuffle(groups)
    n_val, n_test = _split_counts(len(groups), val_fraction, test_fraction)
    n_train = len(groups) - n_val - n_test
    group_sets = {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train:n_train + n_val]),
        "test": set(groups[n_train + n_val:]),
    }
    split_records: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for record in records:
        for split, members in group_sets.items():
            if record["group"] in members:
                split_records[split].append(record)
                break
    overlap = (group_sets["train"] & group_sets["val"]) | (group_sets["train"] & group_sets["test"]) | (
        group_sets["val"] & group_sets["test"]
    )
    report = {
        "rule": "split by group (same group never crosses splits); 90/5/5 by default",
        "groups_total": len(groups),
        "group_counts": {split: len(members) for split, members in group_sets.items()},
        "example_counts": {split: len(items) for split, items in split_records.items()},
        "group_overlap": sorted(overlap)[:20],
        "group_integrity": not overlap,
        "example_groups": {
            split: sorted(members)[:5] for split, members in group_sets.items()
        },
    }
    return split_records, group_sets, report


# ----------------------------------------------------------------------
# tokenizer 覆盖率
# ----------------------------------------------------------------------
def _check_tokenizer_coverage(config: Any, records: list[dict[str, str]]) -> dict[str, Any] | None:
    """若找到预训练 tokenizer，统计 SFT 文本的 unk rate / 字符覆盖率。"""
    candidates: list[Path] = []
    if config.tokenizer_dir:
        candidates.append(Path(config.tokenizer_dir))
    if config.pretrain_data_dir:
        candidates.append(Path(config.pretrain_data_dir))
    tokenizer = None
    used_dir: Path | None = None
    for candidate in candidates:
        path = candidate if candidate.is_file() else candidate / "tokenizer.json"
        if path.is_file():
            tokenizer = CharTokenizer.load(path)
            used_dir = candidate
            break
    if tokenizer is None:
        return None

    total = 0
    unknown = 0
    for record in records:
        text = format_sft_prompt(record["q"]) + record["a"]
        ids = tokenizer.encode(text, add_eos=True)
        total += len(ids)
        unknown += ids.count(tokenizer.unk_id)
    unk_rate = (unknown / total) if total else 0.0
    info = {
        "tokenizer_dir": str(used_dir),
        "vocab_size": tokenizer.vocab_size,
        "checked_examples": len(records),
        "tokens": total,
        "unk_tokens": unknown,
        "unk_rate": unk_rate,
        "coverage": 1.0 - unk_rate,
    }
    if unk_rate > 0.05:
        print(
            f"[prepare_sft] 警告：SFT 文本 unk rate = {unk_rate:.2%}（>{5}%），"
            "词表与 SFT 数据可能不匹配（例如简体/繁体差异）",
            file=sys.stderr,
        )
    return info


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def prepare_sft(config: Any, source: str) -> dict[str, Any]:
    """执行完整 SFT 数据准备，返回汇总信息（主要字段写入 sft_meta.json）。"""
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_records, source_info = _load_source_records(config, source)
    if not raw_records:
        raise ValueError(f"来源 {source!r} 没有可用记录")

    valid, stats = _validate_and_dedupe(raw_records, default_source=source)
    if not valid:
        raise ValueError(
            "所有 SFT 记录都无效（检查 q/a 字段是否为空）："
            + "; ".join(stats["invalid_examples"])
        )
    valid, kept = _cap_samples(valid, config.max_samples, config.seed)
    stats["kept"] = kept
    print(
        f"[prepare_sft] 来源={source} 输入={stats['input']} 条，"
        f"无效={stats['invalid']}，重复={stats['duplicates']}，保留={len(valid)}"
    )

    split_records, _group_sets, group_report = _split_by_group(
        valid, config.val_fraction, config.test_fraction, config.seed
    )
    allow_empty_eval = os.environ.get("MINIGPT_ALLOW_EMPTY_EVAL", "").lower() in {"1", "true", "yes"}
    for split in ("train", "val", "test"):
        if not split_records[split]:
            if split == "train" or not allow_empty_eval:
                raise ValueError(
                    f"{split} 切分为空（数据量太小）。请增大 --val_fraction/--test_fraction 或增加数据；"
                    "确需空评估集可设 MINIGPT_ALLOW_EMPTY_EVAL=1。"
                )
            print(f"[prepare_sft] 警告：{split} 切分为空（已显式允许）")

    tokenizer_info = _check_tokenizer_coverage(config, valid)

    # 写出 JSONL 与元数据。
    split_meta: dict[str, dict[str, Any]] = {}
    for split, items in split_records.items():
        path = data_dir / f"{split}.jsonl"
        write_jsonl(path, items)
        split_meta[split] = {
            "examples": len(items),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    created_at = _now_iso()
    categories: dict[str, int] = {}
    for record in valid:
        key = record.get("category") or "(none)"
        categories[key] = categories.get(key, 0) + 1

    sft_meta = {
        "stage": "sft",
        "source": source,
        "source_info": source_info,
        "format": SFT_FORMAT_TEMPLATE,
        "fields": ["q", "a", "group", "category", "source"],
        "counts": {
            **stats,
            "splits": {split: len(items) for split, items in split_records.items()},
        },
        "splits": split_meta,
        "group_split": group_report,
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "seed": int(config.seed),
        "max_samples": int(config.max_samples),
        "categories": categories,
        "tokenizer": tokenizer_info,
        "created_at": created_at,
    }
    atomic_write_json(data_dir / "sft_meta.json", sft_meta)

    split_report = {
        "stage": "sft",
        "source": source,
        "group_rule": group_report["rule"],
        "group_rule_detail": group_report,
        "splits": split_meta,
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "seed": int(config.seed),
        "group_integrity": group_report["group_integrity"],
        "created_at": created_at,
    }
    atomic_write_json(data_dir / "split_report.json", split_report)

    print(
        "[prepare_sft] 切分："
        f"train={split_meta['train']['examples']}  "
        f"val={split_meta['val']['examples']}  "
        f"test={split_meta['test']['examples']}  "
        f"(groups: train={group_report['group_counts']['train']} "
        f"val={group_report['group_counts']['val']} test={group_report['group_counts']['test']})"
    )
    if tokenizer_info is not None:
        print(
            f"[prepare_sft] tokenizer 覆盖率：{tokenizer_info['coverage']:.2%}"
            f"（unk {tokenizer_info['unk_rate']:.2%}，词表 {tokenizer_info['vocab_size']}）"
        )
    else:
        print("[prepare_sft] 未找到预训练 tokenizer.json，跳过覆盖率检查")
    print(f"[prepare_sft] 输出目录：{data_dir}")
    print("[prepare_sft] 已写出：train.jsonl val.jsonl test.jsonl sft_meta.json split_report.json")
    return {"data_dir": str(data_dir), "sft_meta": sft_meta, "split_report": split_report}


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；返回进程退出码。"""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    argv = _translate_argv(raw_argv)
    try:
        config = parse_config("sft", argv)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[prepare_sft] 配置错误：{exc}", file=sys.stderr)
        return 2

    preset = _preset_name(raw_argv)
    source = str(config.sft_source or "").strip() or _PRESET_DEFAULT_SOURCE.get(preset, "synthetic_en")
    if source not in SOURCES:
        print(f"[prepare_sft] 未知 source {source!r}；可选：{', '.join(SOURCES)}", file=sys.stderr)
        return 2
    if _derive_data_dir(config, source, raw_argv):
        print(f"[prepare_sft] 按 source 自动选择数据目录：{config.data_dir}")

    try:
        prepare_sft(config, source)
    except (DownloadError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"[prepare_sft] 失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
