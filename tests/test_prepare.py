"""``prepare.py`` 集成测试（subprocess + 临时目录；本地 ``--input`` 路径，无网络）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minigpt.data.binary import read_json, read_uint16, sha256_file
from minigpt.tokenization import CharTokenizer

#: 前 90% 只含普通字符，后 10% 含 train 中不存在的稀有字符 -> val/test 有 unk。
CORPUS = ("The quick brown fox jumps over the lazy dog. " * 200) + ("Ω≈ç√∫€£¥" * 100)
CORPUS_LEN = len(CORPUS)  # 9,800 字符，字符级 token 数相同


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """构造隔离的 subprocess 环境（去掉外部 MINIGPT_* 变量）。"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("MINIGPT_")}
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def _run_prepare(args: list[str], cwd: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "prepare.py"), *args],
        cwd=str(cwd),
        env=_clean_env(env_extra),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


class PrepareTest(unittest.TestCase):
    """覆盖 PrepareTest 相关行为。"""
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.corpus = self.tmp / "corpus.txt"
        self.corpus.write_text(CORPUS, encoding="utf-8", newline="\n")
        self.raw = self.tmp / "raw"
        self.out = self.tmp / "out"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _run_input(self, *extra: str, data_dir: Path | None = None) -> subprocess.CompletedProcess:
        args = [
            "--dataset", "input",
            "--input_path", str(self.corpus),
            "--data_dir", str(data_dir or self.out),
            "--download_dir", str(self.raw),
            *extra,
        ]
        return _run_prepare(args, cwd=self.tmp)

    # ------------------------------------------------------------------
    def test_shakespeare_split_sizes_match_construction(self) -> None:
        """验收数字：Tiny Shakespeare 90/5/5 -> 1,003,854 / 55,769 / 55,771。"""
        import prepare

        parts = prepare._split_text("x" * 1_115_394, 0.05, 0.05)
        self.assertEqual(len(parts["train"]), 1_003_854)
        self.assertEqual(len(parts["val"]), 55_769)
        self.assertEqual(len(parts["test"]), 55_771)
        self.assertEqual(sum(len(part) for part in parts.values()), 1_115_394)

    def test_prepare_input_creates_artifacts(self) -> None:
        """验证 数据准备 相关行为。"""
        result = self._run_input()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("已写出", result.stdout)
        for name in ("train.bin", "val.bin", "test.bin", "tokenizer.json", "meta.json", "split_report.json"):
            self.assertTrue((self.out / name).is_file(), f"缺少产物 {name}")

        meta = read_json(self.out / "meta.json")
        report = read_json(self.out / "split_report.json")
        self.assertEqual(meta["dataset"], "input")
        self.assertEqual(meta["source"]["kind"], "local")
        self.assertEqual(meta["source_sha256"], sha256_file(self.corpus))
        self.assertEqual(meta["tokenizer_built_on"], "train")
        self.assertIn("created_at", meta)
        self.assertEqual(meta["val_fraction"], 0.05)
        self.assertEqual(meta["test_fraction"], 0.05)
        self.assertFalse(meta["vocab_extra"]["enabled"])
        self.assertEqual(meta["vocab_extra"]["files"], [])

        tokenizer = CharTokenizer.load(self.out)
        meta_tokenizer = meta["specials"]
        self.assertEqual(meta_tokenizer, {"<unk>": 0, "<eos>": 1, "<pad>": 2})
        self.assertEqual(meta["vocab_size"], tokenizer.vocab_size)

        # 字符级：token 数 == 字符数；split 尺寸按约定
        counts = {split: len(read_uint16(self.out / f"{split}.bin")) for split in ("train", "val", "test")}
        self.assertEqual(counts["train"], 8820)
        self.assertEqual(counts["val"], 490)
        self.assertEqual(counts["test"], 490)
        self.assertEqual(sum(counts.values()), CORPUS_LEN)
        for split in ("train", "val", "test"):
            self.assertEqual(meta["splits"][split]["tokens"], counts[split])
            self.assertEqual(meta["splits"][split]["sha256"], sha256_file(self.out / f"{split}.bin"))

        # 词表只用 train 构建：稀有字符进 <unk>
        train_text = CORPUS[:8820]
        self.assertEqual(tokenizer.vocab_size, 3 + len(set(train_text)))
        self.assertNotIn("Ω", tokenizer.chars)
        self.assertEqual(tokenizer.encode("Ω"), [tokenizer.unk_id])
        self.assertEqual(meta["unk_rate"]["train"], 0.0)
        self.assertGreater(meta["unk_rate"]["val"], 0.0)
        self.assertGreater(meta["unk_rate"]["test"], 0.0)
        # val = 前 180 个普通字符 + 后 310 个稀有字符（全部未登录）
        self.assertEqual(meta["unk_counts"]["val"], 310)
        self.assertEqual(meta["unk_counts"]["test"], 490)

        # split_report 记录偏移/unk/未登录字符
        self.assertEqual(report["total_chars"], CORPUS_LEN)
        self.assertEqual(report["bounds"]["train"], [0, 8820])
        self.assertEqual(report["bounds"]["val"], [8820, 9310])
        self.assertEqual(report["bounds"]["test"], [9310, 9310 + 490])
        self.assertIn("Ω", report["unknown_chars"]["test"])
        self.assertEqual(report["token_counts"], counts)
        for split in ("train", "val", "test"):
            self.assertEqual(report["sha256"][split], sha256_file(self.out / f"{split}.bin"))

    def test_prepare_is_deterministic(self) -> None:
        """验证 数据准备 相关行为。"""
        out2 = self.tmp / "out2"
        first = self._run_input()
        second = self._run_input(data_dir=out2)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        for name in ("train.bin", "val.bin", "test.bin", "tokenizer.json"):
            self.assertEqual((self.out / name).read_bytes(), (out2 / name).read_bytes(), name)

    def test_preset_with_local_input_avoids_network(self) -> None:
        """验证 预设 相关行为。"""
        result = _run_prepare(
            [
                "--preset", "debug_en",
                "--input_path", str(self.corpus),
                "--data_dir", str(self.out),
                "--download_dir", str(self.raw),
            ],
            cwd=self.tmp,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = read_json(self.out / "meta.json")
        self.assertEqual(meta["dataset"], "shakespeare_en")
        self.assertEqual(meta["source"]["kind"], "local")
        self.assertFalse((self.raw / "shakespeare_en.txt").exists())

    def test_prepare_input_requires_input_path(self) -> None:
        """验证 数据准备、路径 相关行为。"""
        result = _run_prepare(
            ["--dataset", "input", "--data_dir", str(self.out), "--download_dir", str(self.raw)],
            cwd=self.tmp,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("input", result.stderr)

    def test_prepare_unknown_dataset_fails(self) -> None:
        """验证 数据准备、数据集 相关行为。"""
        result = _run_prepare(
            ["--dataset", "bogus", "--data_dir", str(self.out), "--download_dir", str(self.raw)],
            cwd=self.tmp,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("未知 dataset", result.stderr)

    def test_prepare_tiny_corpus_fails(self) -> None:
        """验证 数据准备 相关行为。"""
        tiny = self.tmp / "tiny.txt"
        tiny.write_text("abc", encoding="utf-8")
        result = _run_prepare(
            ["--dataset", "input", "--input_path", str(tiny), "--data_dir", str(self.out),
             "--download_dir", str(self.raw)],
            cwd=self.tmp,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("过短", result.stderr)

    def test_fraction_cli_overrides(self) -> None:
        """验证 命令行 相关行为。"""
        result = self._run_input("--val_fraction", "0.1", "--test_fraction", "0.2")
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = read_json(self.out / "meta.json")
        self.assertEqual(meta["val_fraction"], 0.1)
        self.assertEqual(meta["test_fraction"], 0.2)
        counts = {split: len(read_uint16(self.out / f"{split}.bin")) for split in ("train", "val", "test")}
        self.assertEqual(counts["train"], 6860)  # 9800 * 0.7
        self.assertEqual(counts["val"], 980)
        self.assertEqual(counts["test"], 1960)
        self.assertEqual(sum(counts.values()), CORPUS_LEN)

    def test_data_dir_derivation_under_minigpt_root(self) -> None:
        """验证 数据 相关行为。"""
        fake_root = self.tmp / "fake_root"
        fake_root.mkdir()
        result = _run_prepare(
            [
                "--dataset", "input",
                "--input_path", str(self.corpus),
                "--download_dir", str(self.raw),
            ],
            cwd=self.tmp,
            env_extra={"MINIGPT_ROOT": str(fake_root)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((fake_root / "data" / "input_corpus" / "meta.json").is_file())

    # ------------------------------------------------------------------
    # --vocab_extra：只扩充词表，不改变预训练 token 流
    # ------------------------------------------------------------------
    def test_vocab_extra_extends_vocab_without_changing_stream(self) -> None:
        """验证 词表、流式输出 相关行为。"""
        extra = self.tmp / "vocab_extra.txt"
        extra_chars = "Ω≈ç√∫€£¥♞🔥"
        extra.write_text(extra_chars + "\n", encoding="utf-8", newline="\n")
        result = self._run_input("--vocab_extra", str(extra))
        self.assertEqual(result.returncode, 0, result.stderr)

        tokenizer = CharTokenizer.load(self.out)
        for ch in extra_chars:
            self.assertIn(ch, tokenizer.chars, f"vocab_extra 字符未进入词表：{ch!r}")
        ids = tokenizer.encode("Ω♞🔥")
        self.assertNotIn(tokenizer.unk_id, ids)

        meta = read_json(self.out / "meta.json")
        extra_meta = meta["vocab_extra"]
        self.assertTrue(extra_meta["enabled"])
        self.assertEqual(len(extra_meta["files"]), 1)
        self.assertEqual(extra_meta["files"][0]["path"], str(extra.resolve()))
        self.assertEqual(extra_meta["files"][0]["sha256"], sha256_file(extra))
        self.assertEqual(extra_meta["files"][0]["kind"], "txt")
        self.assertEqual(extra_meta["novel_chars"], len(extra_chars))
        self.assertEqual(meta["tokenizer_built_on"], "train+vocab_extra")
        # 稀有字符（原本只在 val/test）现在已知：unk 归零，但 token 流长度不变、
        # 解码回原文，说明额外文本没有进入预训练 token 流。
        self.assertEqual(meta["unk_rate"]["val"], 0.0)
        self.assertEqual(meta["unk_rate"]["test"], 0.0)
        counts = {split: len(read_uint16(self.out / f"{split}.bin")) for split in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 8820, "val": 490, "test": 490})
        self.assertEqual(sum(counts.values()), CORPUS_LEN)
        bounds = read_json(self.out / "split_report.json")["bounds"]
        for split in ("train", "val", "test"):
            start, end = bounds[split]
            self.assertEqual(tokenizer.decode(read_uint16(self.out / f"{split}.bin")), CORPUS[start:end])
        report = read_json(self.out / "split_report.json")
        self.assertTrue(report["vocab_extra"]["enabled"])
        self.assertEqual(report["tokenizer"]["built_on"], "train+vocab_extra")

    def test_vocab_extra_reads_jsonl_q_and_a(self) -> None:
        """验证 词表 相关行为。"""
        extra = self.tmp / "extra.jsonl"
        records = [
            {"q": "问题☂", "a": "答案☂", "category": "x"},
            {"q": "另一个问题", "a": "另一个回答"},
        ]
        extra.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        result = self._run_input("--vocab_extra", str(extra))
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = read_json(self.out / "meta.json")
        info = meta["vocab_extra"]["files"][0]
        self.assertEqual(info["kind"], "jsonl")
        self.assertEqual(info["records"], 2)
        self.assertEqual(info["sha256"], sha256_file(extra))
        tokenizer = CharTokenizer.load(self.out)
        self.assertIn("☂", tokenizer.chars)
        ids = tokenizer.encode("☂")
        self.assertEqual(len(ids), 1)
        self.assertNotEqual(ids[0], tokenizer.unk_id)

    def test_vocab_extra_does_not_auto_include_other_sft_splits(self) -> None:
        """只显式传入 train.jsonl 时，val.jsonl 的字符不得自动进入词表。"""
        sft_dir = self.tmp / "sft"
        sft_dir.mkdir()
        train_path = sft_dir / "train.jsonl"
        val_path = sft_dir / "val.jsonl"
        train_path.write_text(
            json.dumps({"q": "训练问题Ω", "a": "训练回答Ω"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        val_path.write_text(
            json.dumps({"q": "验证问题☂", "a": "验证回答☂"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result = self._run_input("--vocab_extra", str(train_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        tokenizer = CharTokenizer.load(self.out)
        self.assertIn("Ω", tokenizer.chars)
        self.assertNotIn("☂", tokenizer.chars)  # 未传入的 val 文本不参与建词表

    def test_vocab_extra_missing_file_rejected(self) -> None:
        """验证 词表 相关行为。"""
        result = self._run_input("--vocab_extra", str(self.tmp / "does_not_exist.txt"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("vocab_extra", result.stderr)

    def test_vocab_extra_unsupported_extension_rejected(self) -> None:
        """验证 词表 相关行为。"""
        extra = self.tmp / "extra.json"
        extra.write_text("[]", encoding="utf-8")
        result = self._run_input("--vocab_extra", str(extra))
        self.assertEqual(result.returncode, 1)
        self.assertIn("vocab_extra", result.stderr)

    def test_list_presets(self) -> None:
        """验证 `test_list_presets` 对应的行为约定。"""
        result = _run_prepare(["--list_presets"], cwd=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("debug_en", result.stdout)
        self.assertIn("debug_zh", result.stdout)


if __name__ == "__main__":
    unittest.main()
