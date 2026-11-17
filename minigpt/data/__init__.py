"""数据层：下载、二进制 I/O、SFT 格式化与批量数据源。

训练器只需要::

    from minigpt.data import PretrainSource, SFTJsonlSource, MixedSource
    from minigpt.data import format_sft_prompt, load_tokenizer

本包只依赖标准库（不导入 torch/numpy）。
"""

from .binary import (
    Uint16FormatError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    read_uint16,
    read_uint16_array,
    sha256_bytes,
    sha256_file,
    write_jsonl,
    write_uint16,
)
from .download import (
    ALPACA_ZH_FILENAME,
    ALPACA_ZH_LICENSE,
    ALPACA_ZH_URL,
    GUTENBERG_BOOK_IDS,
    SHAKESPEARE_SHA256,
    SHAKESPEARE_SIZE,
    SHAKESPEARE_URLS,
    DownloadError,
    cjk_ratio,
    clean_corpus_text,
    decode_text,
    download_alpaca_zh,
    download_shakespeare,
    download_zh_gutenberg,
    fetch_bytes,
    fetch_zh_book,
    load_local_text,
    strip_gutenberg_markers,
)
from .formatters import (
    SFT_FORMAT_TEMPLATE,
    encode_sft_example,
    format_sft_full_text,
    format_sft_prompt,
    normalize_answer,
    normalize_group,
    normalize_question,
    record_identity,
    validate_record,
)
from .sources import (
    EVAL_SPLITS,
    IGNORE_INDEX,
    SPLITS,
    MixedSource,
    PretrainSource,
    SFTJsonlSource,
    load_tokenizer,
)

__all__ = [
    # binary
    "Uint16FormatError",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "read_json",
    "read_jsonl",
    "read_uint16",
    "read_uint16_array",
    "sha256_bytes",
    "sha256_file",
    "write_jsonl",
    "write_uint16",
    # download
    "ALPACA_ZH_FILENAME",
    "ALPACA_ZH_LICENSE",
    "ALPACA_ZH_URL",
    "GUTENBERG_BOOK_IDS",
    "SHAKESPEARE_SHA256",
    "SHAKESPEARE_SIZE",
    "SHAKESPEARE_URLS",
    "DownloadError",
    "cjk_ratio",
    "clean_corpus_text",
    "decode_text",
    "download_alpaca_zh",
    "download_shakespeare",
    "download_zh_gutenberg",
    "fetch_bytes",
    "fetch_zh_book",
    "load_local_text",
    "strip_gutenberg_markers",
    # formatters
    "SFT_FORMAT_TEMPLATE",
    "encode_sft_example",
    "format_sft_full_text",
    "format_sft_prompt",
    "normalize_answer",
    "normalize_group",
    "normalize_question",
    "record_identity",
    "validate_record",
    # sources
    "EVAL_SPLITS",
    "IGNORE_INDEX",
    "SPLITS",
    "MixedSource",
    "PretrainSource",
    "SFTJsonlSource",
    "load_tokenizer",
]
