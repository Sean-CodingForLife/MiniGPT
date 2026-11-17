"""Read-only stage scans for the dashboard: datasets / evaluations / exports.

The Web backend is a *view* over files produced by the CLIs, so every function
here only **reads** files and never starts work:

* :func:`scan_datasets` / :func:`dataset_detail` — ``data/`` 下的语料与 SFT 数据集
  （``meta.json``、``sft_meta.json``、``split_report.json``、``tokenizer.json``）；
* :func:`scan_evaluations` / :func:`evaluation_detail` — ``runs/<id>/eval_*/`` 下的
  ``evaluation.json``、``predictions.jsonl``、``capability_boundary.md``；
* :func:`scan_exports` / :func:`export_detail` — ``dist/`` 下的 Bundle v1
  （``MANIFEST.json``、``config.json``、``MODEL_CARD.md``）。

所有读取都容忍半截文件/坏 JSON（返回 ``None`` 或跳过），不会因为一个坏文件让
接口 500；所有名字先做白名单校验（只允许目录里真实存在的条目），再做路径包含
校验，杜绝路径穿越。本模块只用标准库，**不导入 torch**。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_PREVIEW_CHARS = 600
MAX_SAMPLES = 20
MAX_BOUNDARY_CHARS = 4000

DATA_META_FILES = ("meta.json", "sft_meta.json")
EVAL_SCRIPTS = ("eval_validate", "eval_train", "eval_test")
EXPORT_ESSENTIALS = ("config.json", "pytorch_model.bin", "tokenizer.json", "MODEL_CARD.md", "run.py", "serve.py")


# ---------------------------------------------------------------------------
# 容错读取
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    """读 JSON；文件不存在/半截/坏 JSON 时返回 ``None``（绝不抛出）。"""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl_head(path: Path, limit: int) -> list[dict[str, Any]]:
    """读 JSONL 的前 ``limit`` 条合法记录（跳过坏行）。"""

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except OSError:
        return records
    return records


def _read_text_head(path: Path, limit: int) -> str:
    """读取文本文件前若干行，用于控制台/卡片预览。"""
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return ""


def _mtime_iso(path: Path) -> str | None:
    """把文件修改时间转成 ISO 字符串。"""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _child_dir(root: Path, name: str) -> Path | None:
    """把名字解析成 ``root`` 下的真实目录（白名单 + 越界检查）。"""

    name = str(name or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        candidate = (root / name).resolve()
    except OSError:
        return None
    if candidate == root.resolve() or not candidate.is_relative_to(root.resolve()):
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _as_int(value: Any) -> int | None:
    """尽力把值转成 int，失败返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    """尽力把值转成 float，失败返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def _dataset_entry(directory: Path) -> dict[str, Any] | None:
    """把一个 ``data/<name>/`` 目录整理成一条数据集记录。"""

    meta_path = None
    for candidate in DATA_META_FILES:
        path = directory / candidate
        if path.is_file():
            meta_path = path
            break
    if meta_path is None:
        return None
    meta = _read_json(meta_path)
    if not isinstance(meta, dict):
        return None
    kind = "sft" if meta_path.name == "sft_meta.json" or str(meta.get("stage")) == "sft" else "pretrain"
    files = sorted(item.name for item in directory.iterdir() if item.is_file())
    source = meta.get("source_info") or meta.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    splits = meta.get("splits") if isinstance(meta.get("splits"), dict) else {}
    unk_rate = meta.get("unk_rate") if isinstance(meta.get("unk_rate"), dict) else None
    tokenizer_meta = meta.get("tokenizer") if isinstance(meta.get("tokenizer"), dict) else None
    vocab_size = _as_int(meta.get("vocab_size"))
    if vocab_size is None and tokenizer_meta:
        vocab_size = _as_int(tokenizer_meta.get("vocab_size"))
    group = meta.get("group_split") if isinstance(meta.get("group_split"), dict) else {}
    counts = meta.get("counts") if isinstance(meta.get("counts"), dict) else {}
    entry: dict[str, Any] = {
        "name": directory.name,
        "kind": kind,
        "meta_file": meta_path.name,
        "created_at": meta.get("created_at"),
        "source": {
            "kind": source.get("kind"),
            "name": source.get("name") or meta.get("dataset") or meta.get("source"),
            "license": source.get("license"),
            "sha256": source.get("sha256") or meta.get("source_sha256"),
            "url": source.get("url"),
            "records_total": _as_int(source.get("records_total")),
        },
        "vocab_size": vocab_size,
        "splits": splits,
        "unk_rate": unk_rate,
        "coverage": _as_float((tokenizer_meta or {}).get("coverage")),
        "group_integrity": group.get("group_integrity"),
        "group_counts": group.get("group_counts"),
        "vocab_extra": meta.get("vocab_extra") if isinstance(meta.get("vocab_extra"), dict) else None,
        "counts": {
            "input": _as_int(counts.get("input")),
            "valid": _as_int(counts.get("valid")),
            "kept": _as_int(counts.get("kept")),
            "invalid": _as_int(counts.get("invalid")),
            "duplicates": _as_int(counts.get("duplicates")),
        } if counts else None,
        "fields": meta.get("fields"),
        "format": meta.get("format"),
        "seed": meta.get("seed"),
        "has_tokenizer": (directory / "tokenizer.json").is_file(),
        "has_binary": all((directory / f"{split}.bin").is_file() for split in ("train", "val", "test")),
        "has_jsonl": all((directory / f"{split}.jsonl").is_file() for split in ("train", "val", "test")),
        "files": files,
        "bytes": sum(item.stat().st_size for item in directory.iterdir() if item.is_file()),
    }
    return entry


def scan_datasets(data_dir: str | Path) -> dict[str, Any]:
    """列出 ``data/`` 下的数据集目录（跳过 ``raw/`` 等没有元信息的目录）。"""

    root = Path(data_dir)
    datasets = [
        entry
        for entry in (
            _dataset_entry(directory)
            for directory in sorted(root.iterdir()) if directory.is_dir()
        )
        if entry is not None
    ] if root.is_dir() else []
    raw_dir = root / "raw"
    raw_files = (
        [
            {"name": item.name, "bytes": item.stat().st_size}
            for item in sorted(raw_dir.iterdir())
            if item.is_file()
        ]
        if raw_dir.is_dir()
        else []
    )
    pretrain = [item for item in datasets if item["kind"] == "pretrain"]
    sft = [item for item in datasets if item["kind"] == "sft"]
    return {
        "data_dir": str(root.resolve()) if root.is_dir() else str(root),
        "exists": root.is_dir(),
        "count": len(datasets),
        "pretrain_count": len(pretrain),
        "sft_count": len(sft),
        "datasets": datasets,
        "raw_files": raw_files,
    }


def _decode_preview(tokens: list[int], tokenizer_state: dict[str, Any] | None) -> str:
    """把少量 token id 解码成可读文本（用于语料预览，跳过特殊 token）。"""

    if tokenizer_state is None:
        return ""
    chars = tokenizer_state.get("chars") or []
    specials = tokenizer_state.get("specials") or {}
    offset = len(specials)
    parts: list[str] = []
    for token_id in tokens:
        if token_id in specials.values():
            continue
        if 0 <= token_id < len(chars) + offset:
            parts.append(str(chars[token_id - offset]))
    return "".join(parts)


def dataset_detail(data_dir: str | Path, name: str, limit: int = MAX_SAMPLES) -> dict[str, Any] | None:
    """单个数据集的详情：完整 meta + SFT 样例 / 预训练语料预览。"""

    root = Path(data_dir)
    directory = _child_dir(root, name)
    if directory is None:
        return None
    entry = _dataset_entry(directory)
    if entry is None:
        return None
    limit = max(1, min(int(limit), MAX_SAMPLES))
    meta = _read_json(directory / entry["meta_file"]) or {}
    tokenizer_state = _read_json(directory / "tokenizer.json") if entry["has_tokenizer"] else None
    detail: dict[str, Any] = {
        **entry,
        "meta": meta,
        "split_report": _read_json(directory / "split_report.json"),
        "tokenizer": (
            {
                "vocab_size": len(tokenizer_state.get("chars") or []) + len(tokenizer_state.get("specials") or {}),
                "specials": tokenizer_state.get("specials"),
                "type": tokenizer_state.get("type"),
                "version": tokenizer_state.get("version"),
            }
            if isinstance(tokenizer_state, dict)
            else None
        ),
        "samples": [],
        "preview": None,
    }
    if entry["kind"] == "sft":
        detail["samples"] = _read_jsonl_head(directory / "train.jsonl", limit)
        detail["sample_files"] = {split: (directory / f"{split}.jsonl").is_file() for split in ("train", "val", "test")}
    elif entry["has_binary"]:
        tokens: list[int] = []
        try:
            with (directory / "train.bin").open("rb") as handle:
                raw = handle.read(limit * 8 * 2)
            values = list(int.from_bytes(raw[index : index + 2], "little") for index in range(0, len(raw) - 1, 2))
            tokens = values[: limit * 8]
        except OSError:
            tokens = []
        text = _decode_preview(tokens, tokenizer_state if isinstance(tokenizer_state, dict) else None)
        detail["preview"] = {
            "tokens": len(tokens),
            "text": text[:MAX_PREVIEW_CHARS],
        }
    return detail


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def _normalize_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """把 pretrain / sft 两种报告指标压成同一组键，供前端表格使用。"""

    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    flat: dict[str, Any] = {}
    for key in ("loss", "perplexity", "bpc", "answer_loss", "exact_match", "char_f1", "empty_rate", "repetition_rate", "count", "eos_rate"):
        value = report.get(key)
        if value is None:
            value = metrics.get(key)
        number = _as_float(value)
        if number is not None:
            flat[key] = number
    return flat


def _eval_entry(run_id: str, eval_dir: Path) -> dict[str, Any] | None:
    """把单个评估目录整理成列表项（含指标与产物）。"""
    report_path = eval_dir / "evaluation.json"
    if not report_path.is_file():
        return None
    report = _read_json(report_path)
    if not isinstance(report, dict):
        return None
    predictions = eval_dir / "predictions.jsonl"
    boundary = eval_dir / "capability_boundary.md"
    lines = 0
    if predictions.is_file():
        try:
            with predictions.open("r", encoding="utf-8") as handle:
                lines = sum(1 for line in handle if line.strip())
        except OSError:
            lines = 0
    return {
        "run_id": run_id,
        "eval_dir": eval_dir.name,
        "stage": report.get("stage"),
        "split": report.get("split"),
        "created_at": report.get("created_at"),
        "checkpoint_name": Path(str(report.get("checkpoint", ""))).name or "best.pt",
        "metrics": _normalize_metrics(report),
        "tokens": _as_int(report.get("tokens")),
        "sample": report.get("sample") if isinstance(report.get("sample"), str) else None,
        "sample_prompt": report.get("sample_prompt"),
        "has_predictions": lines > 0,
        "prediction_count": lines,
        "has_boundary": boundary.is_file(),
        "mtime": _mtime_iso(report_path),
        "files": sorted(item.name for item in eval_dir.iterdir() if item.is_file()),
    }


def scan_evaluations(runs_dir: str | Path) -> dict[str, Any]:
    """扫描所有 ``runs/<id>/eval_*`` 目录里的评测报告。"""

    root = Path(runs_dir)
    entries: list[dict[str, Any]] = []
    if root.is_dir():
        for run_dir in sorted(item for item in root.iterdir() if item.is_dir()):
            for eval_dir in sorted(item for item in run_dir.iterdir() if item.is_dir() and item.name.startswith("eval")):
                entry = _eval_entry(run_dir.name, eval_dir)
                if entry is not None:
                    entries.append(entry)
    entries.sort(key=lambda item: item.get("mtime") or "", reverse=True)
    return {
        "runs_dir": str(root.resolve()) if root.is_dir() else str(root),
        "exists": root.is_dir(),
        "count": len(entries),
        "evaluations": entries,
    }


def _find_eval_dir(run_root: Path, eval_name: str | None) -> Path | None:
    """在 run 目录下定位指定的评估子目录。"""
    if eval_name:
        directory = _child_dir(run_root, eval_name)
        return directory if directory is not None else None
    candidates = sorted(
        (item for item in run_root.iterdir() if item.is_dir() and item.name.startswith("eval")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def evaluation_detail(
    runs_dir: str | Path,
    run_id: str,
    eval_name: str | None = None,
    limit: int = 50,
    boundary_chars: int = MAX_BOUNDARY_CHARS,
) -> dict[str, Any] | None:
    """单个评测详情：完整报告 + predictions 头部 + 能力边界草稿。"""

    root = Path(runs_dir)
    run_root = _child_dir(root, run_id)
    if run_root is None:
        return None
    eval_dir = _find_eval_dir(run_root, eval_name)
    if eval_dir is None:
        return None
    entry = _eval_entry(run_root.name, eval_dir)
    if entry is None:
        return None
    limit = max(1, min(int(limit), 500))
    return {
        **entry,
        "report": _read_json(eval_dir / "evaluation.json"),
        "predictions": _read_jsonl_head(eval_dir / "predictions.jsonl", limit),
        "boundary": _read_text_head(eval_dir / "capability_boundary.md", boundary_chars),
    }


# ---------------------------------------------------------------------------
# 导出包
# ---------------------------------------------------------------------------
def _export_entry(directory: Path) -> dict[str, Any] | None:
    """把 dist 模型包整理成列表项（读取 MANIFEST/配置）。"""
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        return None
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    archive = directory.with_suffix(".zip")
    if not archive.is_file():
        archive = directory.parent / f"{directory.name}.zip"
    present = sorted(item.name for item in directory.iterdir() if item.is_file())
    entry = {
        "name": directory.name,
        "format": manifest.get("format"),
        "format_version": manifest.get("format_version"),
        "created_at": manifest.get("created_at"),
        "run_id": source.get("run_id"),
        "parameters": _as_int(source.get("parameters")),
        "checkpoint_name": Path(str(source.get("checkpoint_file", ""))).name or None,
        "file_count": len(files) or len(present),
        "total_bytes": sum(_as_int(value.get("bytes")) or 0 for value in files.values()) if files else sum(
            item.stat().st_size for item in directory.iterdir() if item.is_file()
        ),
        "missing_files": sorted(item for item in EXPORT_ESSENTIALS if item not in present),
        "has_zip": archive.is_file(),
        "zip_bytes": archive.stat().st_size if archive.is_file() else None,
        "model_card_bytes": (directory / "MODEL_CARD.md").stat().st_size if (directory / "MODEL_CARD.md").is_file() else None,
        "config": _read_json(directory / "config.json"),
        "files": present,
        "mtime": _mtime_iso(manifest_path),
    }
    return entry


def scan_exports(dist_dir: str | Path) -> dict[str, Any]:
    """列出 ``dist/`` 下的导出包（以 ``MANIFEST.json`` 为准）。"""

    root = Path(dist_dir)
    exports: list[dict[str, Any]] = []
    if root.is_dir():
        for directory in sorted(item for item in root.iterdir() if item.is_dir()):
            entry = _export_entry(directory)
            if entry is not None:
                exports.append(entry)
    exports.sort(key=lambda item: item.get("mtime") or "", reverse=True)
    zips = [
        {"name": item.name, "bytes": item.stat().st_size}
        for item in sorted(root.iterdir())
        if root.is_dir() and item.is_file() and item.suffix.lower() == ".zip"
    ] if root.is_dir() else []
    return {
        "dist_dir": str(root.resolve()) if root.is_dir() else str(root),
        "exists": root.is_dir(),
        "count": len(exports),
        "exports": exports,
        "zips": zips,
    }


def export_detail(
    dist_dir: str | Path,
    name: str,
    model_card_chars: int = 4000,
) -> dict[str, Any] | None:
    """单个导出包详情：manifest 文件清单（含 sha256）+ 模型卡开头。"""

    root = Path(dist_dir)
    directory = _child_dir(root, name)
    if directory is None:
        return None
    entry = _export_entry(directory)
    if entry is None:
        return None
    manifest = _read_json(directory / "MANIFEST.json") or {}
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    return {
        **entry,
        "manifest": manifest,
        "manifest_files": [
            {"name": key, "bytes": _as_int(value.get("bytes")), "sha256": value.get("sha256")}
            for key, value in sorted(files.items())
        ],
        "model_card": _read_text_head(directory / "MODEL_CARD.md", model_card_chars),
        "readme": _read_text_head(directory / "README.md", 1200),
    }


__all__ = [
    "scan_datasets",
    "dataset_detail",
    "scan_evaluations",
    "evaluation_detail",
    "scan_exports",
    "export_detail",
]
