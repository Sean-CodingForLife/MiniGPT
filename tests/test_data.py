"""数据层单元测试：二进制 I/O、下载/清洗（mock，无网络）、SFT 格式化、数据源。

全部使用 unittest + 临时目录；下载路径一律 mock ``minigpt.data.download._urlopen``，
不发起真实网络请求，也不导入 torch/numpy/pytest。
"""

from __future__ import annotations

import io
import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minigpt.data import binary, download, formatters, sources
from minigpt.tokenization import CharTokenizer

FIXTURE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. 0123456789\n"
) * 20

ASCII_HINT = "Question Answer abcdefghijklmnopqrstuvwxyz0123456789?:. "


# ======================================================================
# 辅助
# ======================================================================
class FakeResponse:
    """模拟 ``urllib.request.urlopen`` 返回的上下文管理器。"""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, amount: int = -1) -> bytes:
        # 与真实 http.client 响应一致：支持限长读取（fetch_bytes 用其防超大响应）。
        if amount is None or amount < 0:
            return self._data
        return self._data[:amount]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class FakeUrlOpen:
    """按 handler(url) 返回内容，同时记录调用 URL。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.calls.append(url)
        result = self.handler(url)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


def make_pretrain_dir(
    root: Path,
    text: str = FIXTURE_TEXT,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> CharTokenizer:
    """构造一个最小可用的预训练数据目录（与 prepare.py 产物同构）。"""
    text = download.clean_corpus_text(text)
    total = len(text)
    n_val = int(total * val_fraction)
    n_test = int(total * test_fraction)
    n_train = total - n_val - n_test
    parts = {
        "train": text[:n_train],
        "val": text[n_train:n_train + n_val],
        "test": text[n_train + n_val:],
    }
    tokenizer = CharTokenizer.build(parts["train"])
    root.mkdir(parents=True, exist_ok=True)
    split_meta = {}
    for split, part in parts.items():
        ids = tokenizer.encode(part)
        path = root / f"{split}.bin"
        binary.write_uint16(path, ids)
        split_meta[split] = {"tokens": len(ids), "sha256": binary.sha256_file(path), "bytes": len(ids) * 2}
    tokenizer.save(root)
    meta = {
        "dataset": "fixture",
        "source": {"kind": "inline"},
        "source_sha256": binary.sha256_bytes(text.encode("utf-8")),
        "vocab_size": tokenizer.vocab_size,
        "specials": tokenizer.specials,
        "splits": split_meta,
        "unk_rate": {split: 0.0 for split in parts},
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    binary.atomic_write_json(root / "meta.json", meta)
    return tokenizer


def make_sft_dir(root: Path, counts=(8, 3, 2)) -> Path:
    """构造最小 SFT 目录（不含 tokenizer.json，由调用方注入 tokenizer）。"""
    root.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test")
    split_meta = {}
    for split, count in zip(splits, counts):
        records = [
            {"q": f"what is number {index}?", "a": f"answer number {index}"}
            for index in range(count)
        ]
        path = root / f"{split}.jsonl"
        binary.write_jsonl(path, records)
        split_meta[split] = {"examples": count, "sha256": binary.sha256_file(path), "bytes": path.stat().st_size}
    meta = {"stage": "sft", "format": formatters.SFT_FORMAT_TEMPLATE, "splits": split_meta}
    binary.atomic_write_json(root / "sft_meta.json", meta)
    binary.atomic_write_json(root / "split_report.json", {"stage": "sft", "splits": split_meta})
    return root


# ======================================================================
# 二进制 I/O
# ======================================================================
class BinaryTest(unittest.TestCase):
    """覆盖 BinaryTest 相关行为。"""
    def test_uint16_roundtrip_and_explicit_little_endian(self) -> None:
        """验证 `test_uint16_roundtrip_and_explicit_little_endian` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.bin"
            size = binary.write_uint16(path, [0, 1, 2, 255, 256, 65535])
            raw = path.read_bytes()
            self.assertEqual(size, 12)
            self.assertEqual(len(raw), 12)
            # 显式小端：0 -> 00 00，1 -> 01 00，256 -> 00 01，65535 -> ff ff
            self.assertEqual(raw[0:2], b"\x00\x00")
            self.assertEqual(raw[2:4], b"\x01\x00")
            self.assertEqual(raw[8:10], b"\x00\x01")
            self.assertEqual(raw[10:12], b"\xff\xff")
            self.assertEqual(binary.read_uint16(path), [0, 1, 2, 255, 256, 65535])
            self.assertEqual(list(binary.read_uint16_array(path)), [0, 1, 2, 255, 256, 65535])

    def test_uint16_rejects_out_of_range(self) -> None:
        """验证 `test_uint16_rejects_out_of_range` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bin"
            with self.assertRaises(binary.Uint16FormatError):
                binary.write_uint16(path, [70000])
            with self.assertRaises(binary.Uint16FormatError):
                binary.write_uint16(path, [-1])

    def test_read_rejects_odd_bytes(self) -> None:
        """验证 `test_read_rejects_odd_bytes` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "odd.bin"
            path.write_bytes(b"\x01\x02\x03")
            with self.assertRaises(binary.Uint16FormatError):
                binary.read_uint16(path)

    def test_empty_file_is_valid(self) -> None:
        """验证 `test_empty_file_is_valid` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.bin"
            binary.write_uint16(path, [])
            self.assertEqual(path.read_bytes(), b"")
            self.assertEqual(binary.read_uint16(path), [])

    def test_sha256_helpers(self) -> None:
        """验证 `test_sha256_helpers` 对应的行为约定。"""
        import hashlib

        data = "中文abc".encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(data)
            self.assertEqual(binary.sha256_file(path), hashlib.sha256(data).hexdigest())
            self.assertEqual(binary.sha256_bytes(b""), hashlib.sha256(b"").hexdigest())

    def test_json_helpers_roundtrip_and_atomic(self) -> None:
        """验证 `test_json_helpers_roundtrip_and_atomic` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            payload = {"name": "中文", "nested": {"items": [1, 2, 3]}}
            binary.atomic_write_json(path, payload)
            self.assertEqual(binary.read_json(path), payload)
            self.assertEqual([item.name for item in Path(tmp).iterdir()], ["meta.json"])
            # 覆盖写不会残留临时文件
            binary.atomic_write_json(path, {"name": "覆盖"})
            self.assertEqual(binary.read_json(path), {"name": "覆盖"})
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_jsonl_helpers(self) -> None:
        """验证 `test_jsonl_helpers` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            records = [{"q": "问题1", "a": "回答1"}, {"q": "q2", "a": "a2"}]
            binary.write_jsonl(path, records)
            self.assertEqual(binary.read_jsonl(path), records)
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertIn("问题1", text)  # ensure_ascii=False
            # 空文件 -> 空列表；坏行 -> ValueError
            binary.write_jsonl(path, [])
            self.assertEqual(binary.read_jsonl(path), [])
            path.write_text('{"q": 1}\nnot json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                binary.read_jsonl(path)


# ======================================================================
# 下载 / 清洗（全部 mock 网络）
# ======================================================================
class DownloadTest(unittest.TestCase):
    """覆盖 DownloadTest 相关行为。"""
    def test_fetch_bytes_retries_then_succeeds(self) -> None:
        """验证 `test_fetch_bytes_retries_then_succeeds` 对应的行为约定。"""
        fake = mock.Mock(side_effect=[urllib.error.URLError("boom"), FakeResponse(b"ok")])
        with mock.patch.object(download, "_urlopen", fake):
            data = download.fetch_bytes("http://example/x", retries=2, backoff=0.0)
        self.assertEqual(data, b"ok")
        self.assertEqual(fake.call_count, 2)

    def test_fetch_bytes_all_fail(self) -> None:
        """验证 失败路径 相关行为。"""
        fake = mock.Mock(side_effect=urllib.error.URLError("down"))
        with mock.patch.object(download, "_urlopen", fake):
            with self.assertRaises(download.DownloadError):
                download.fetch_bytes("http://example/x", retries=2, backoff=0.0)
        self.assertEqual(fake.call_count, 2)

    def test_fetch_bytes_sha256_mismatch(self) -> None:
        """验证 `test_fetch_bytes_sha256_mismatch` 对应的行为约定。"""
        fake = FakeUrlOpen(lambda url: b"content")
        with mock.patch.object(download, "_urlopen", fake):
            with self.assertRaises(download.DownloadError):
                download.fetch_bytes("http://example/x", retries=1, expected_sha256="0" * 64)
        self.assertEqual(len(fake.calls), 1)

    def test_fetch_bytes_detects_truncated_response(self) -> None:
        # Content-Length 声明 100 字节但实际只返回 10 字节 -> 必须判为截断。
        # 真实 http.client 响应读完后 length 归零，检查必须在读取前记录声明长度。
        """验证 `test_fetch_bytes_detects_truncated_response` 对应的行为约定。"""
        class TruncatedResponse(FakeResponse):
            """覆盖 TruncatedResponse 相关行为。"""
            def __init__(self) -> None:
                super().__init__(b"0123456789")
                self.length = 100

            def read(self, amount: int = -1) -> bytes:
                data = super().read(amount)
                self.length = 0  # 模拟读完后剩余长度归零
                return data

        fake = mock.Mock(return_value=TruncatedResponse())
        with mock.patch.object(download, "_urlopen", fake):
            with self.assertRaises(download.DownloadError) as ctx:
                download.fetch_bytes("http://example/x", retries=1, backoff=0.0)
        self.assertIn("截断", str(ctx.exception))

    def test_fetch_bytes_accepts_complete_response_with_length(self) -> None:
        """验证 `test_fetch_bytes_accepts_complete_response_with_length` 对应的行为约定。"""
        class CompleteResponse(FakeResponse):
            """覆盖 CompleteResponse 相关行为。"""
            def __init__(self) -> None:
                super().__init__(b"0123456789")
                self.length = 10

            def read(self, amount: int = -1) -> bytes:
                data = super().read(amount)
                self.length = 0
                return data

        fake = mock.Mock(return_value=CompleteResponse())
        with mock.patch.object(download, "_urlopen", fake):
            data = download.fetch_bytes("http://example/x", retries=1, backoff=0.0)
        self.assertEqual(data, b"0123456789")

    def test_download_shakespeare_cache_and_fallback(self) -> None:
        """验证 缓存 相关行为。"""
        content = b"tiny shakespeare bytes"
        digest = binary.sha256_bytes(content)
        fake = FakeUrlOpen(lambda url: b"first source down" if "A" in url else content)
        # 第一个 URL 用 URLError，第二个成功
        def handler(url: str):
            if "A" in url:
                raise urllib.error.URLError("down")
            return content

        fake = FakeUrlOpen(handler)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(download, "_urlopen", fake):
                path = download.download_shakespeare(
                    tmp,
                    urls=["http://A/x", "http://B/x"],
                    expected_sha256=digest,
                    retries=1,
                )
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(len(fake.calls), 2)
                # 第二次：命中缓存，不再联网
                again = download.download_shakespeare(
                    tmp, urls=["http://A/x", "http://B/x"], expected_sha256=digest
                )
                self.assertEqual(again, path)
                self.assertEqual(len(fake.calls), 2)  # 未新增请求

    def test_download_shakespeare_hash_mismatch_raises(self) -> None:
        """验证 `test_download_shakespeare_hash_mismatch_raises` 对应的行为约定。"""
        fake = FakeUrlOpen(lambda url: b"wrong bytes")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(download, "_urlopen", fake):
                with self.assertRaises(download.DownloadError):
                    download.download_shakespeare(tmp, urls=["http://A/x"], expected_sha256="0" * 64, retries=1)
            self.assertFalse((Path(tmp) / "shakespeare_en.txt").exists())

    def test_strip_gutenberg_markers(self) -> None:
        """验证 `test_strip_gutenberg_markers` 对应的行为约定。"""
        raw = (
            "Header info\r\n*** START OF THE PROJECT GUTENBERG EBOOK 示例 ***\r\n\r\n"
            "第一段正文。\r\n第二段正文。\r\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK 示例 ***\r\nFooter info\r\n"
        )
        body = download.strip_gutenberg_markers(raw)
        self.assertEqual(body, "第一段正文。\n第二段正文。")
        with self.assertRaises(download.DownloadError):
            download.strip_gutenberg_markers("没有 START 标记的文本")
        with self.assertRaises(download.DownloadError):
            download.strip_gutenberg_markers(
                "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n正文但没有结束标记"
            )

    def test_zh_gutenberg_download_and_cache(self) -> None:
        """验证 缓存 相关行为。"""
        def handler(url: str) -> bytes:
            for book_id in ("111", "222"):
                if f"pg{book_id}.txt" in url:
                    body = f"这是第{book_id}本书的正文内容，用于测试。" * 10
                    text = (
                        f"Header\n*** START OF THE PROJECT GUTENBERG EBOOK BOOK {book_id} ***\n"
                        f"{body}\n*** END OF THE PROJECT GUTENBERG EBOOK BOOK {book_id} ***\nFooter"
                    )
                    return text.encode("utf-8")
            raise urllib.error.URLError(f"unexpected url {url}")

        fake = FakeUrlOpen(handler)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(download, "_urlopen", fake):
                result = download.download_zh_gutenberg(
                    tmp, book_ids=["111", "222"], retries=1, min_chars_per_book=20
                )
            self.assertTrue(Path(result["path"]).is_file())
            self.assertEqual(result["book_ids"], ["111", "222"])
            self.assertEqual(len(result["books"]), 2)
            self.assertGreater(result["cjk_ratio"], 0.5)
            self.assertEqual(result["sha256"], binary.sha256_file(result["path"]))
            combined = Path(result["path"]).read_text(encoding="utf-8")
            self.assertNotIn("START OF", combined)
            self.assertIn("第111本书", combined)
            calls_before = len(fake.calls)
            # 二次调用：per-book 缓存命中，不再联网
            with mock.patch.object(download, "_urlopen", fake):
                again = download.download_zh_gutenberg(
                    tmp, book_ids=["111", "222"], retries=1, min_chars_per_book=20
                )
            self.assertEqual(again["sha256"], result["sha256"])
            self.assertEqual(len(fake.calls), calls_before)

    def test_truncated_book_cache_is_redownloaded(self) -> None:
        """验证 缓存 相关行为。"""
        raw_dir = Path(tempfile.mkdtemp())
        try:
            # 缓存里放一个缺 END 标记的截断文件
            (raw_dir / "gutenberg_333.txt").write_text("截断内容，没有标记", encoding="utf-8")
            body = "这本书的正文完整。" * 20
            payload = (
                f"*** START OF THE PROJECT GUTENBERG EBOOK 333 ***\n{body}\n"
                f"*** END OF THE PROJECT GUTENBERG EBOOK 333 ***"
            ).encode("utf-8")
            fake = FakeUrlOpen(lambda url: payload)
            with mock.patch.object(download, "_urlopen", fake):
                stats = download.fetch_zh_book("333", raw_dir, retries=1, min_chars=20)
            self.assertIn("这本书的正文完整", stats["text"])
            self.assertTrue((raw_dir / "gutenberg_333.txt").read_text(encoding="utf-8").startswith("*** START"))
        finally:
            for item in raw_dir.iterdir():
                item.unlink()
            raw_dir.rmdir()

    def test_decode_text_and_load_local_text(self) -> None:
        """验证 `test_decode_text_and_load_local_text` 对应的行为约定。"""
        self.assertEqual(download.decode_text("中文".encode("gb18030")), "中文")
        self.assertEqual(download.decode_text("\ufeff中文".encode("utf-8")), "中文")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.txt"
            path.write_bytes("第一行\r\n第二行\r\n".encode("utf-8"))
            self.assertEqual(download.load_local_text(path), "第一行\n第二行\n")
            with self.assertRaises(FileNotFoundError):
                download.load_local_text(Path(tmp) / "nope.txt")

    def test_clean_corpus_text(self) -> None:
        """验证 `test_clean_corpus_text` 对应的行为约定。"""
        self.assertEqual(download.clean_corpus_text("\ufeffa\r\nb\rc\x00"), "a\nb\nc")

    def test_download_alpaca_cache_hit(self) -> None:
        """验证 缓存 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / download.ALPACA_ZH_FILENAME
            records = [{"instruction": "问题", "input": "", "output": "回答"}]
            cache.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            fake = FakeUrlOpen(lambda url: b"should not be used")
            with mock.patch.object(download, "_urlopen", fake):
                path = download.download_alpaca_zh(tmp)
            self.assertEqual(path, cache)
            self.assertEqual(fake.calls, [])


# ======================================================================
# SFT 格式化
# ======================================================================
class FormatterTest(unittest.TestCase):
    """覆盖 FormatterTest 相关行为。"""
    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.build(ASCII_HINT)

    def test_format_sft_prompt(self) -> None:
        """验证 SFT 相关行为。"""
        self.assertEqual(formatters.format_sft_prompt("hello"), "Q: hello\nA: ")
        self.assertEqual(formatters.format_sft_prompt("  hello  "), "Q: hello\nA: ")
        with self.assertRaises(ValueError):
            formatters.format_sft_prompt("   ")

    def test_encode_sft_example_labels(self) -> None:
        """验证 SFT 相关行为。"""
        block_size = 32
        x, y = formatters.encode_sft_example(self.tokenizer, "hello", "world", block_size)
        self.assertEqual(len(x), block_size)
        self.assertEqual(len(y), block_size)
        prompt_ids = self.tokenizer.encode(formatters.format_sft_prompt("hello"))
        answer_ids = self.tokenizer.encode("world")
        self.assertEqual(x[: len(prompt_ids)], prompt_ids)
        self.assertEqual(x[len(prompt_ids): len(prompt_ids) + len(answer_ids)], answer_ids)
        # x = sequence[:-1]，所以 EOS 只作为标签出现，不在输入里。
        pad_start = len(prompt_ids) + len(answer_ids)
        self.assertEqual(x[pad_start:], [self.tokenizer.pad_id] * (block_size - pad_start))
        # 仅 answer+EOS 计 loss：prompt 之前全是 -100
        self.assertEqual(y[: len(prompt_ids) - 1], [-100] * (len(prompt_ids) - 1))
        self.assertEqual(
            y[len(prompt_ids) - 1: len(prompt_ids) + len(answer_ids)],
            answer_ids + [self.tokenizer.eos_id],
        )
        self.assertEqual(y[pad_start:], [-100] * (block_size - pad_start))
        # 因果对齐：y[i] = sequence[i+1]；最后一个有效标签预测的是序列外的 EOS，
        # 它只出现在标签里（x 中对应位置已是 padding）。
        for index, label in enumerate(y):
            if label == -100:
                continue
            if index + 1 < pad_start:
                self.assertEqual(x[index + 1], label)
            else:
                self.assertEqual(label, self.tokenizer.eos_id)

    def test_encode_sft_example_truncation(self) -> None:
        """验证 SFT 相关行为。"""
        block_size = 10
        x, y = formatters.encode_sft_example(self.tokenizer, "hello", "x" * 500, block_size)
        self.assertEqual(len(x), block_size)
        self.assertEqual(len(y), block_size)
        enabled = [label for label in y if label != -100]
        self.assertGreaterEqual(len(enabled), 2)  # 至少 answer + EOS
        self.assertEqual(enabled[-1], self.tokenizer.eos_id)
        self.assertEqual(x[0], self.tokenizer.encode(formatters.format_sft_prompt("hello"))[0])

    def test_encode_sft_example_validation(self) -> None:
        """验证 SFT、校验 相关行为。"""
        with self.assertRaises(ValueError):
            formatters.encode_sft_example(self.tokenizer, "hello", "world", 1)
        with self.assertRaises(ValueError):
            formatters.encode_sft_example(self.tokenizer, "", "world", 16)
        with self.assertRaises(ValueError):
            formatters.encode_sft_example(self.tokenizer, "hello", "", 16)

    def test_validate_record(self) -> None:
        """验证 校验 相关行为。"""
        record = formatters.validate_record(
            {"q": "  What is 1 + 1? ", "a": " 2 ", "category": "math"}, default_source="demo"
        )
        self.assertEqual(record["q"], "What is 1 + 1?")
        self.assertEqual(record["a"], "2")
        self.assertEqual(record["group"], "what is 1 + 1?")
        self.assertEqual(record["category"], "math")
        self.assertEqual(record["source"], "demo")
        # 别名 + 显式 group
        alias = formatters.validate_record({"question": "Q?", "answer": "A", "group": "G"})
        self.assertEqual((alias["q"], alias["a"], alias["group"]), ("Q?", "A", "g"))
        for bad in ({}, {"q": "", "a": "x"}, {"q": "x", "a": ""}, {"q": "x" * 5000, "a": "y"}):
            with self.assertRaises(ValueError):
                formatters.validate_record(bad)

    def test_normalize_group_and_full_text(self) -> None:
        """验证 `test_normalize_group_and_full_text` 对应的行为约定。"""
        self.assertEqual(formatters.normalize_group("  What   IS 1+1？ "), "what is 1+1?")
        self.assertEqual(
            formatters.format_sft_full_text("hello", "world"), "Q: hello\nA: world<eos>"
        )


# ======================================================================
# 数据源
# ======================================================================
class SourceTest(unittest.TestCase):
    """覆盖 SourceTest 相关行为。"""
    def test_pretrain_source_batches(self) -> None:
        """验证 预训练 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = make_pretrain_dir(Path(tmp))
            source = sources.PretrainSource(Path(tmp))
            try:
                self.assertEqual(source.vocab_size, tokenizer.vocab_size)
                self.assertEqual(source.metadata["dataset"], "fixture")
                self.assertGreater(source.split_tokens("train"), 0)
                batch = source.train_batch(4, 16, random.Random(0))
                self.assertEqual(set(batch), {"x", "y"})
                self.assertEqual(len(batch["x"]), 4)
                for x, y in zip(batch["x"], batch["y"]):
                    self.assertEqual(len(x), 16)
                    self.assertEqual(len(y), 16)
                    self.assertTrue(all(isinstance(value, int) for value in x + y))
                    self.assertEqual(y[:-1], x[1:])  # 预训练标签 = 下一个 token
                    self.assertTrue(all(0 <= value < tokenizer.vocab_size for value in x + y))
                with self.assertRaises(ValueError):
                    source.train_batch(4, 10 ** 6, random.Random(0))
                with self.assertRaises(ValueError):
                    source.eval_batches("nope", 2, 8, 1)
            finally:
                del source

    def test_pretrain_eval_batches_are_deterministic(self) -> None:
        """验证 预训练、评估 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            make_pretrain_dir(Path(tmp))
            source = sources.PretrainSource(Path(tmp))
            first = source.eval_batches("val", 2, 8, 3, seed=1234)
            second = source.eval_batches("val", 2, 8, 3, seed=1234)
            other = source.eval_batches("val", 2, 8, 3, seed=999)
            self.assertEqual(first, second)
            self.assertNotEqual(first, other)
            self.assertEqual(len(first), 3)
            self.assertEqual(source.eval_batches("val", 2, 8, 0, seed=1), [])
            for batch in first:
                for x, y in zip(batch["x"], batch["y"]):
                    self.assertEqual(y[:-1], x[1:])

    def test_pretrain_missing_files(self) -> None:
        """验证 预训练 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                sources.PretrainSource(Path(tmp) / "nope")
            (Path(tmp) / "train.bin").write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                sources.PretrainSource(Path(tmp))

    def test_train_too_short_raises(self) -> None:
        """验证 训练 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            make_pretrain_dir(Path(tmp), text="abcdef", val_fraction=0.0, test_fraction=0.0)
            source = sources.PretrainSource(Path(tmp))
            with self.assertRaises(ValueError):
                source.train_batch(1, 8, random.Random(0))

    def test_sft_source_batches_and_alignment(self) -> None:
        """验证 SFT 相关行为。"""
        tokenizer = CharTokenizer.build(ASCII_HINT)
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_sft_dir(Path(tmp))
            source = sources.SFTJsonlSource(directory, tokenizer=tokenizer, block_size=12)
            self.assertEqual(source.split_examples("train"), 8)
            self.assertEqual(source.metadata["stage"], "sft")
            batch = source.train_batch(3, 12, random.Random(1))
            for x, y in zip(batch["x"], batch["y"]):
                self.assertEqual(len(x), 12)
                self.assertEqual(len(y), 12)
                self.assertIn(-100, y)  # question/pad 不计 loss
                enabled = [(i, label) for i, label in enumerate(y) if label != -100]
                self.assertTrue(enabled)
                for index, label in enabled:
                    if index < len(x) - 1:
                        if x[index + 1] == tokenizer.pad_id:
                            # 序列外只可能是末尾 EOS（作为标签出现，不在输入里）
                            self.assertEqual(label, tokenizer.eos_id)
                        else:
                            self.assertEqual(x[index + 1], label)  # 因果对齐
            first = source.eval_batches("val", 2, 12, 2, seed=42)
            self.assertEqual(first, source.eval_batches("val", 2, 12, 2, seed=42))
            self.assertEqual(len(first), 2)

    def test_sft_source_tokenizer_fallback(self) -> None:
        """验证 SFT、tokenizer 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_sft_dir(Path(tmp))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                source = sources.SFTJsonlSource(directory)
            self.assertGreater(source.vocab_size, 3)
            self.assertIn("tokenizer", stderr.getvalue())
            self.assertTrue(source.train_batch(1, 8, random.Random(0))["x"])

    def test_mixed_source_ratio_extremes_and_statistics(self) -> None:
        """验证 `test_mixed_source_ratio_extremes_and_statistics` 对应的行为约定。"""
        tokenizer = CharTokenizer.build(ASCII_HINT)
        with tempfile.TemporaryDirectory() as tmp:
            pretrain_dir = Path(tmp) / "pre"
            sft_dir = Path(tmp) / "sft"
            make_pretrain_dir(pretrain_dir)
            make_sft_dir(sft_dir)
            pretrain = sources.PretrainSource(pretrain_dir)
            sft_source = sources.SFTJsonlSource(sft_dir, tokenizer=tokenizer, block_size=12)

            only_sft = sources.MixedSource(sft_source, pretrain, pretrain_ratio=0.0)
            batch = only_sft.train_batch(4, 12, random.Random(0))
            self.assertTrue(all(-100 in row for row in batch["y"]))

            only_pre = sources.MixedSource(sft_source, pretrain, pretrain_ratio=1.0)
            batch = only_pre.train_batch(4, 12, random.Random(0))
            self.assertTrue(all(-100 not in row for row in batch["y"]))

            mixed = sources.MixedSource(sft_source, pretrain, pretrain_ratio=0.15)
            batch = mixed.train_batch(400, 12, random.Random(7))
            sft_rows = sum(1 for row in batch["y"] if -100 in row)
            self.assertGreater(len(batch["x"]), 0)
            # pretrain_ratio=0.15 时约 85% 的行来自 SFT（含 -100 标签）。
            self.assertGreaterEqual(sft_rows, 300)
            self.assertLessEqual(sft_rows, 380)
            meta = mixed.data_meta
            self.assertIn("pretrain", meta)
            self.assertIn("sft", meta)
            self.assertEqual(meta["pretrain_ratio"], 0.15)
            # 评估委托给 SFT 源
            self.assertEqual(
                mixed.eval_batches("val", 2, 12, 1, seed=1),
                sft_source.eval_batches("val", 2, 12, 1, seed=1),
            )
            with self.assertRaises(ValueError):
                sources.MixedSource(sft_source, pretrain, pretrain_ratio=1.5)

    def test_mixed_source_samples_only_needed_rows(self) -> None:
        # F25：不再各抽一个完整 batch 再逐行挑选；按比例只抽所需行数。
        """验证 `test_mixed_source_samples_only_needed_rows` 对应的行为约定。"""
        class CountingSource:
            """覆盖 CountingSource 相关行为。"""
            def __init__(self, marker: int) -> None:
                self.marker = marker
                self.requests: list[int] = []

            def train_batch(self, batch_size, block_size, rng=None):
                self.requests.append(int(batch_size))
                return {
                    "x": [[self.marker] * int(block_size) for _ in range(int(batch_size))],
                    "y": [[self.marker] * int(block_size) for _ in range(int(batch_size))],
                }

        sft = CountingSource(1)
        pre = CountingSource(2)
        mixed = sources.MixedSource(sft, pre, pretrain_ratio=0.25)
        batch = mixed.train_batch(40, 4, random.Random(7))
        self.assertEqual(len(batch["x"]), 40)
        # 抽样开销：两侧合计只抽 batch_size 行（旧实现是 2*batch_size）。
        self.assertEqual(sum(sft.requests) + sum(pre.requests), 40)
        self.assertLessEqual(sum(sft.requests), 40)
        self.assertLessEqual(sum(pre.requests), 40)
        # 行内容必须来自正确的数据源
        for x, y in zip(batch["x"], batch["y"]):
            self.assertEqual(x, y)
            self.assertIn(x[0], (1, 2))

    def test_mixed_source_ratio_distribution(self) -> None:
        # 统计断言：大样本下预训练行占比接近配置比例（随机性与分布不变）。
        """验证 `test_mixed_source_ratio_distribution` 对应的行为约定。"""
        class CountingSource:
            """覆盖 CountingSource 相关行为。"""
            def __init__(self, marker: int) -> None:
                self.marker = marker

            def train_batch(self, batch_size, block_size, rng=None):
                return {
                    "x": [[self.marker] * int(block_size) for _ in range(int(batch_size))],
                    "y": [[self.marker] * int(block_size) for _ in range(int(batch_size))],
                }

        for ratio in (0.15, 0.5):
            mixed = sources.MixedSource(CountingSource(1), CountingSource(2), pretrain_ratio=ratio)
            batch = mixed.train_batch(2000, 4, random.Random(11))
            pre_rows = sum(1 for row in batch["x"] if row[0] == 2)
            self.assertAlmostEqual(pre_rows / 2000, ratio, delta=0.04)

    def test_mixed_source_unifies_mismatched_tokenizers(self) -> None:
        """验证 `test_mixed_source_unifies_mismatched_tokenizers` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            pretrain_dir = Path(tmp) / "pre"
            sft_dir = Path(tmp) / "sft"
            make_pretrain_dir(pretrain_dir)
            make_sft_dir(sft_dir)
            pretrain = sources.PretrainSource(pretrain_dir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                sft_source = sources.SFTJsonlSource(sft_dir)  # 用 SFT 文本兜底构建，词表不同
                mixed = sources.MixedSource(sft_source, pretrain, pretrain_ratio=0.1)
            self.assertIn("tokenizer", stderr.getvalue())
            self.assertIs(mixed.sft_source.tokenizer, pretrain.tokenizer)
            batch = mixed.train_batch(2, 8, random.Random(0))
            self.assertEqual(len(batch["x"]), 2)

    def test_load_tokenizer_helper(self) -> None:
        """验证 tokenizer 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = make_pretrain_dir(Path(tmp))
            loaded = sources.load_tokenizer(Path(tmp))
            self.assertEqual(loaded.state_dict(), tokenizer.state_dict())
            self.assertEqual(sources.load_tokenizer(Path(tmp) / "tokenizer.json").vocab_size, tokenizer.vocab_size)


# ======================================================================
# 依赖守卫：数据层只允许标准库
# ======================================================================
class ImportGuardTest(unittest.TestCase):
    """覆盖 ImportGuardTest 相关行为。"""
    FORBIDDEN_MODULES = ("torch", "numpy", "pytest", "transformers", "datasets", "tqdm", "matplotlib", "pandas")
    FORBIDDEN_IMPORT = re.compile(
        r"^\s*(?:import|from)\s+(?:"
        + "|".join(FORBIDDEN_MODULES)
        + r")\b",
        re.MULTILINE,
    )

    def test_no_forbidden_imports_in_source(self) -> None:
        """验证 `test_no_forbidden_imports_in_source` 对应的行为约定。"""
        files = sorted((ROOT / "minigpt" / "data").glob("*.py"))
        files += sorted((ROOT / "minigpt" / "tokenization").glob("*.py"))
        files += [ROOT / "prepare.py", ROOT / "prepare_sft.py", ROOT / "scripts" / "build_demo_data.py"]
        self.assertTrue(files)
        for path in files:
            text = path.read_text(encoding="utf-8")
            match = self.FORBIDDEN_IMPORT.search(text)
            self.assertIsNone(match, f"{path} 导入了禁止的第三方库：{match.group(0) if match else ''}")

    def test_import_does_not_load_third_party(self) -> None:
        """验证 导入 相关行为。"""
        code = (
            "import minigpt.data, minigpt.tokenization\n"
            "import sys\n"
            "bad = sorted(m for m in "
            f"{self.FORBIDDEN_MODULES!r} if m in sys.modules)\n"
            "print(bad)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
