"""预训练数据准备 CLI。

流程
----
1. ``minigpt.config.parse_config("pretrain", argv)`` 解析配置（默认值 < config.json
   < MINIGPT_* 环境变量 < 命令行；未知字段会被拒绝）；
2. 按 ``--dataset`` 获取语料：``shakespeare_en`` / ``zh_gutenberg`` 自动下载并
   校验，``input`` 或 ``--input`` 使用本地文本（离线兜底）；
3. 连续切分 90/5/5（``--val_fraction``/``--test_fraction`` 可改），
   **只用 train 文本构建字符级 tokenizer**（可选 ``--vocab_extra`` 只读若干
   ``.jsonl/.txt`` 文件用于扩展词表：SFT 的 q/a 文本可以补字，但**绝不进入预训练
   token 流**；SFT 的 val/test 文本禁止传入，否则评估泄漏）；
4. 写出 ``train.bin/val.bin/test.bin``（uint16 小端）、``tokenizer.json``、
   ``meta.json``、``split_report.json``，并报告 val/test 的 unk rate。

示例::

    python prepare.py --dataset shakespeare_en
    python prepare.py --dataset zh_gutenberg --data_dir data/zh_gutenberg \
        --vocab_extra data/sft_demo_zh/train.jsonl,data/sft_alpaca_zh/train.jsonl
    python prepare.py --dataset input --input my_corpus.txt --data_dir data/custom

``--vocab_extra`` 路径相对项目根或绝对路径，逗号分隔；``.jsonl`` 只抽取 ``q/a``
字符串用于建词表，``.txt`` 按 UTF-8 文本读取。路径缺失/扩展名不支持时直接报错。

全程只用 Python 标准库；不需要 torch。
"""

from __future__ import annotations

import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  确保任意 cwd 下都能 import minigpt

from minigpt import cli_utils
from minigpt.config import parse_config
from minigpt.data.binary import atomic_write_json, read_jsonl, sha256_file, write_uint16
from minigpt.data.download import (
    SHAKESPEARE_SHA256,
    SHAKESPEARE_URLS,
    DownloadError,
    clean_corpus_text,
    download_shakespeare,
    download_zh_gutenberg,
    load_local_text,
)
from minigpt.tokenization import SPECIALS, CharTokenizer

#: 支持的语料名。
DATASETS = ("shakespeare_en", "zh_gutenberg", "input")

#: 数据目录默认名（用于在 --dataset 变化时推导 data_dir，纯便利功能）。
_DEFAULT_DATA_DIR_NAMES = {"shakespeare_en", "zh_gutenberg"}

#: 常用 CLI 别名 -> Config 字段名。
_ARGT_ALIASES = {
    "--input": "--input_path",
    "--url": "--source_url",
    "--books": "--zh_books",
}


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _translate_argv(argv: list[str]) -> list[str]:
    """把友好别名（--input/--url/--books）翻译成 Config 字段参数。"""
    return cli_utils.translate_argv(argv, _ARGT_ALIASES)


def _has_flag(argv: list[str], flag: str) -> bool:
    """判断命令行里是否显式出现过某个参数（用于「用户是否覆盖了默认值」）。"""
    return cli_utils.has_flag(argv, flag)


def _now_iso() -> str:
    """返回带时区的 ISO 8601 时间戳，写入 meta/report 的 created_at。"""
    return cli_utils.now_iso()


def _parse_book_ids(raw: str) -> list[str]:
    """解析 ``--zh_books``（逗号分隔的 Gutenberg 书号）。"""
    ids = [item.strip() for item in str(raw).split(",")]
    ids = [item for item in ids if item]
    if not ids:
        raise ValueError("--zh_books 不能为空，应为逗号分隔的书号，例如 23950,23962")
    bad = [item for item in ids if not item.isdigit()]
    if bad:
        raise ValueError(f"--zh_books 含非法书号：{bad}")
    return ids


def _resolve_input_path(config: Any) -> Path:
    """把 config.input_path 解析为绝对路径（相对路径按项目根解释）。"""
    return cli_utils.resolve_input_path(config.input_path, config.project_root)


def _load_vocab_extra(config: Any) -> tuple[list[dict[str, Any]], str, list[str]]:
    """读取 ``--vocab_extra``：仅用于构建词表，绝不进入预训练 token 流。

    返回 ``(文件元数据列表, 拼接后的额外文本, 逐文件文本列表)``。

    * ``.jsonl``：抽取每条记录的 ``q/a``（兼容 ``question/answer``）字符串；
    * ``.txt``：按 UTF-8 文本读取；
    * 路径可以是绝对路径，或相对项目根的路径；缺失/扩展名不支持直接报错。

    警告：这里只应传入 SFT 的 **train** 数据（或词表补充文本）。SFT 的 val/test
    文本如果传入，会让词表“见过”评估字符，属于评估泄漏，必须避免。
    """
    raw = str(getattr(config, "vocab_extra", "") or "").strip()
    if not raw:
        return [], "", []
    root = Path(config.project_root)
    infos: list[dict[str, Any]] = []
    texts: list[str] = []
    seen: set[Path] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"--vocab_extra 文件不存在：{path}")
        if path.stem in {"val", "test"}:
            print(
                f"[prepare] 警告：--vocab_extra 传入了 {path.name}（验证/测试切分）。"
                "词表将“见过”评估字符，属于评估泄漏；请只传 SFT 的 train 数据。",
                file=sys.stderr,
            )
        suffix = path.suffix.lower()
        detail: dict[str, Any] = {}
        if suffix == ".jsonl":
            records = read_jsonl(path)
            chunks: list[str] = []
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(f"{path} 第 {index + 1} 行不是 JSON 对象")
                question = record.get("q", record.get("question"))
                answer = record.get("a", record.get("answer"))
                if not isinstance(question, str) or not isinstance(answer, str):
                    raise ValueError(f"{path} 第 {index + 1} 行缺少字符串字段 q/a")
                chunks.append(question)
                chunks.append(answer)
            text = "\n".join(chunks)
            detail["records"] = len(records)
            kind = "jsonl"
        elif suffix == ".txt":
            text = load_local_text(path)
            kind = "txt"
            detail["lines"] = text.count("\n") + (0 if text.endswith("\n") else 1)
        else:
            raise ValueError(f"--vocab_extra 只支持 .jsonl 或 .txt 文件：{path}")
        text = clean_corpus_text(text).strip("\n")
        texts.append(text)
        infos.append(
            {
                "path": str(path),
                "kind": kind,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "chars": len(text),
                **detail,
                "novel_chars": 0,  # 稍后相对预训练 train 文本统计
            }
        )
    return infos, "\n".join(texts), texts


def _derive_data_dir(config: Any, argv: list[str]) -> bool:
    """若用户没有显式指定 data_dir，按 dataset 推导合理默认目录。

    只在当前 data_dir 看起来仍是 preset 默认的 ``data/<名字>`` 时生效，
    避免覆盖 config.json 里自定义的目录。返回是否修改过。
    """
    if _has_flag(argv, "--data_dir") or "MINIGPT_DATA_DIR" in os.environ:
        return False
    root = Path(config.project_root)
    current = Path(config.data_dir)
    if current.parent != root / "data" or current.name not in _DEFAULT_DATA_DIR_NAMES:
        return False
    dataset = str(config.dataset)
    if dataset in _DEFAULT_DATA_DIR_NAMES:
        candidate = root / "data" / dataset
    elif dataset == "input":
        if not config.input_path:
            return False
        stem = Path(config.input_path).expanduser().stem or "input"
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._")[:48] or "input"
        candidate = root / "data" / f"input_{safe}"
    else:
        return False
    if candidate == current:
        return False
    config.data_dir = str(candidate)
    return True


# ----------------------------------------------------------------------
# 语料获取
# ----------------------------------------------------------------------
def _resolve_corpus(config: Any) -> tuple[str, dict[str, Any]]:
    """返回 (原始文本, 来源元数据)。来源一律先做统一换行清洗。"""
    if config.input_path:
        path = _resolve_input_path(config)
        text = load_local_text(path)
        source = {
            "kind": "local",
            "name": path.name,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        return text, source

    dataset = str(config.dataset)
    if dataset == "shakespeare_en":
        # Tiny Shakespeare 的身份由固定 sha256 定义：即使是回退镜像/自定义 URL，
        # 也要求字节完全一致，保证可复现。
        print(f"[prepare] 获取 Tiny Shakespeare（sha256={SHAKESPEARE_SHA256[:12]}...）")
        urls = (str(config.source_url),) if config.source_url else None
        path = download_shakespeare(
            config.download_dir,
            urls=urls,
            expected_sha256=SHAKESPEARE_SHA256,
            verbose=True,
        )
        text = load_local_text(path)
        digest = sha256_file(path)
        source = {
            "kind": "download",
            "name": "tiny_shakespeare",
            "url": (urls[0] if urls else SHAKESPEARE_URLS[0]),
            "files": [{"path": str(path), "sha256": digest, "bytes": path.stat().st_size}],
            "sha256": digest,
        }
        return text, source

    if dataset == "zh_gutenberg":
        template = None
        if config.source_url:
            if "{id}" not in str(config.source_url):
                raise ValueError(
                    "zh_gutenberg 的 --source_url 需要包含 {id} 占位符，例如 "
                    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
                )
            template = str(config.source_url)
        book_ids = _parse_book_ids(config.zh_books)
        print(f"[prepare] 下载 Project Gutenberg 中文书单：{','.join(book_ids)}")
        result = download_zh_gutenberg(
            config.download_dir,
            book_ids=book_ids,
            template=template,
            verbose=True,
        )
        text = load_local_text(result["path"])
        source = {
            "kind": "download",
            "name": "gutenberg_zh",
            "path": result["path"],
            "sha256": result["sha256"],
            "chars": result["chars"],
            "cjk_ratio": result["cjk_ratio"],
            "book_ids": result["book_ids"],
            "books": result["books"],
        }
        return text, source

    if dataset == "input":
        raise ValueError("dataset=input 需要同时用 --input/--input_path 指定本地文本文件")

    raise ValueError(f"未知 dataset {dataset!r}；可选：{', '.join(DATASETS)}")


# ----------------------------------------------------------------------
# 切分 / 统计 / 产出
# ----------------------------------------------------------------------
def _split_text(text: str, val_fraction: float, test_fraction: float) -> dict[str, str]:
    """按字符偏移连续切分（与 CONSTRUCTION.md §5 一致：90/5/5 连续切分）。

    尺寸约定（与验收数字对齐）：``train = floor(N * (1 - val - test))``、
    ``val = floor(N * val_fraction)``，**test 吸收取整余数**。例如 Tiny Shakespeare
    N=1,115,394 时得到 1,003,854 / 55,769 / 55,771。分数用 :class:`fractions.Fraction`
    精确计算，避免浮点误差改变切分边界。
    """
    total = len(text)
    if total < 20:
        raise ValueError(f"语料过短（{total} 字符），无法切分")
    try:
        frac_val = Fraction(str(float(val_fraction))).limit_denominator(10_000)
        frac_test = Fraction(str(float(test_fraction))).limit_denominator(10_000)
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover - config.validate 已拦截
        raise ValueError(f"非法的切分比例：{exc}") from exc
    frac_train = Fraction(1) - frac_val - frac_test
    if frac_train < 0:
        raise ValueError("val_fraction + test_fraction 必须 < 1")
    n_train = int(total * frac_train)
    n_val = int(total * frac_val)
    if frac_val > 0 and n_val == 0:
        n_val = 1
        n_train = total - n_val - int(total * frac_test)
    n_test = total - n_train - n_val
    if n_test <= 0 and frac_test > 0:
        # 极小语料兜底：保证 test 至少 1 个字符（从 train 借 1 个）。
        n_test = 1
        n_train = total - n_val - n_test
    if n_train < 1 or n_test < 0:
        raise ValueError(
            f"val/test 比例过大：语料 {total} 字符，train 只剩 {n_train} 字符；"
            "请调小 --val_fraction/--test_fraction"
        )
    return {
        "train": text[:n_train],
        "val": text[n_train:n_train + n_val],
        "test": text[n_train + n_val:],
    }


def _unknown_chars(texts: dict[str, str], tokenizer: CharTokenizer, limit: int = 200) -> dict[str, list[str]]:
    """统计 val/test 中未进入词表的字符样例（最多 ``limit`` 个/切分）。

    返回的字符在编码时都会变成 ``<unk>``；把它们写进 split_report.json，
    便于判断是否需要补语料或调整 vocab_extra。
    """
    known = set(tokenizer.chars)
    result: dict[str, list[str]] = {}
    for split in ("val", "test"):
        unknown = sorted(set(texts[split]) - known)
        result[split] = unknown[:limit]
    return result


def prepare(config: Any) -> dict[str, Any]:
    """执行完整的数据准备流程，返回汇总信息（也写入了 meta.json）。"""
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_text, source = _resolve_corpus(config)
    text = clean_corpus_text(raw_text)
    if len(text) < 20:
        raise ValueError(f"清洗后的语料只有 {len(text)} 字符，过短")

    texts = _split_text(text, config.val_fraction, config.test_fraction)

    # 关键约束：词表只用 train 构建，val/test 的未登录字符走 <unk>。
    # 可选 --vocab_extra 只读额外文件补字（例如 SFT 训练集），仅用于建词表，
    # 绝不把额外文本加入预训练 token 流；SFT 的 val/test 文本禁止传入。
    extra_infos, extra_text, extra_texts = _load_vocab_extra(config)
    novel_chars: set[str] = set()
    train_vocab_text = texts["train"]
    if extra_text:
        train_chars_set = set(texts["train"])
        for info, extra_part in zip(extra_infos, extra_texts):
            new_chars = set(extra_part) - train_chars_set
            info["novel_chars"] = len(new_chars)
            novel_chars.update(new_chars)
        train_vocab_text = texts["train"] + "\n" + extra_text
    tokenizer = CharTokenizer.build(train_vocab_text)
    vocab_extra_meta = {
        "enabled": bool(extra_infos),
        "files": extra_infos,
        "chars": len(extra_text),
        "novel_chars": len(novel_chars),
        "note": "仅用于构建词表，不加入预训练 token 流；禁止传入 SFT 的 val/test 文本",
    }
    token_ids = {split: tokenizer.encode(texts[split]) for split in ("train", "val", "test")}

    # 写出二进制 token 文件。
    split_meta: dict[str, dict[str, Any]] = {}
    unk_counts: dict[str, int] = {}
    unk_rate: dict[str, float] = {}
    for split in ("train", "val", "test"):
        ids = token_ids[split]
        bin_path = data_dir / f"{split}.bin"
        byte_size = write_uint16(bin_path, ids)
        split_meta[split] = {
            "tokens": len(ids),
            "bytes": byte_size,
            "sha256": sha256_file(bin_path),
            "chars": len(texts[split]),
        }
        count = ids.count(tokenizer.unk_id)
        unk_counts[split] = count
        unk_rate[split] = (count / len(ids)) if ids else 0.0

    tokenizer_path = tokenizer.save(data_dir)
    created_at = _now_iso()

    meta = {
        "dataset": str(config.dataset),
        "source": source,
        "source_sha256": source.get("sha256", ""),
        "vocab_size": tokenizer.vocab_size,
        "specials": dict(tokenizer.specials),
        "splits": split_meta,
        "unk_rate": unk_rate,
        "unk_counts": unk_counts,
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "tokenizer_built_on": ("train+vocab_extra" if extra_infos else "train"),
        "vocab_extra": vocab_extra_meta,
        "seed": int(config.seed),
        "created_at": created_at,
    }
    meta_path = atomic_write_json(data_dir / "meta.json", meta)

    # 连续切分的偏移量、每份字符/token 数、未登录字符样例都记入报告。
    train_chars = len(texts["train"])
    val_chars = len(texts["val"])
    split_report = {
        "dataset": str(config.dataset),
        "total_chars": len(text),
        "total_tokens": sum(len(ids) for ids in token_ids.values()),
        "split_rule": "contiguous by character offset (no shuffle); tokenizer built on train text (+ optional vocab_extra training text)",
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "bounds": {
            "train": [0, train_chars],
            "val": [train_chars, train_chars + val_chars],
            "test": [train_chars + val_chars, len(text)],
        },
        "char_counts": {split: len(texts[split]) for split in texts},
        "token_counts": {split: len(token_ids[split]) for split in token_ids},
        "unk_counts": unk_counts,
        "unk_rate": unk_rate,
        "unknown_chars": _unknown_chars(texts, tokenizer),
        "sha256": {split: split_meta[split]["sha256"] for split in split_meta},
        "tokenizer": {
            "path": str(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "specials": list(SPECIALS),
            "chars": len(tokenizer.chars),
            "built_on": ("train+vocab_extra" if extra_infos else "train"),
        },
        "vocab_extra": {
            "enabled": vocab_extra_meta["enabled"],
            "files": vocab_extra_meta["files"],
            "chars": vocab_extra_meta["chars"],
            "novel_chars": vocab_extra_meta["novel_chars"],
            "note": vocab_extra_meta["note"],
        },
        "created_at": created_at,
    }
    report_path = atomic_write_json(data_dir / "split_report.json", split_report)

    print(f"[prepare] dataset={config.dataset}  来源={source.get('kind')}")
    print(f"[prepare] 语料：{len(text):,} 字符 -> {split_report['total_tokens']:,} tokens")
    print(f"[prepare] 词表：{tokenizer.vocab_size}（{len(tokenizer.chars)} 字符 + {len(SPECIALS)} 特殊 token）")
    if extra_infos:
        print(
            f"[prepare] vocab_extra：{len(extra_infos)} 个文件，"
            f"{vocab_extra_meta['chars']:,} 字符（新增 {vocab_extra_meta['novel_chars']} 个词表字符）"
            " —— 仅用于建词表，不进入预训练 token 流"
        )
    print(
        "[prepare] 切分："
        f"train={split_meta['train']['tokens']:,}  "
        f"val={split_meta['val']['tokens']:,}  "
        f"test={split_meta['test']['tokens']:,}"
    )
    print(
        "[prepare] unk rate："
        f"train={unk_rate['train']:.4%}  val={unk_rate['val']:.4%}  test={unk_rate['test']:.4%}"
    )
    print(f"[prepare] 输出目录：{data_dir}")
    print(
        "[prepare] 已写出：train.bin val.bin test.bin tokenizer.json meta.json split_report.json"
        f"（meta: {meta_path.name}，report: {report_path.name}）"
    )
    return {
        "data_dir": str(data_dir),
        "meta": meta,
        "split_report": split_report,
        "tokenizer": tokenizer,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；返回进程退出码。"""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    argv = _translate_argv(raw_argv)
    try:
        config = parse_config("pretrain", argv)
    except SystemExit:
        raise
    except Exception as exc:  # 配置错误：给出可读信息而不是 traceback
        print(f"[prepare] 配置错误：{exc}", file=sys.stderr)
        return 2

    if _derive_data_dir(config, raw_argv):
        print(f"[prepare] 按 dataset 自动选择数据目录：{config.data_dir}")

    try:
        prepare(config)
    except (DownloadError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"[prepare] 失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
