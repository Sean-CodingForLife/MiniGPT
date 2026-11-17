"""``prepare_sft.py`` 与 ``scripts/build_demo_data.py`` 集成测试。

* 演示数据：500/500 条 SFT、30/30 条评估（in_distribution/edge/unrelated 各 10）；
  生成脚本确定性，重新生成必须与仓库内提交文件逐字节一致。
* SFT 准备：group 90/5/5 切分不跨集合、字段校验/去重/采样、meta/split_report、
  tokenizer 覆盖率、alpaca-zh 缓存路径（用本地缓存 fixture，不联网）。
"""

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

from minigpt.data.binary import read_json, read_jsonl, sha256_file
from minigpt.data.formatters import SFT_FORMAT_TEMPLATE, normalize_group

DATA_DIR = ROOT / "data"

DEMO_FILES = {
    "demo_sft_en.jsonl": 500,
    "demo_sft_zh.jsonl": 500,
    "demo_eval_en.jsonl": 30,
    "demo_eval_zh.jsonl": 30,
}


def _clean_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("MINIGPT_")}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_script(script: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


class DemoDataTest(unittest.TestCase):
    """覆盖 DemoDataTest 相关行为。"""
    def test_committed_demo_files(self) -> None:
        """验证 `test_committed_demo_files` 对应的行为约定。"""
        for name, expected_count in DEMO_FILES.items():
            path = DATA_DIR / name
            self.assertTrue(path.is_file(), f"缺少演示数据文件 {path}")
            records = read_jsonl(path)
            self.assertEqual(len(records), expected_count, name)
            for record in records:
                self.assertIsInstance(record.get("q"), str)
                self.assertIsInstance(record.get("a"), str)
                self.assertTrue(record["q"].strip())
                self.assertTrue(record["a"].strip())
                self.assertEqual(record["group"], normalize_group(record["q"]))
                self.assertIn("category", record)
                self.assertIn("source", record)
            self.assertEqual(len({record["q"] for record in records}), len(records), f"{name} 有重复问题")

        eval_en = read_jsonl(DATA_DIR / "demo_eval_en.jsonl")
        eval_zh = read_jsonl(DATA_DIR / "demo_eval_zh.jsonl")
        for eval_records in (eval_en, eval_zh):
            categories = {}
            for record in eval_records:
                categories[record["category"]] = categories.get(record["category"], 0) + 1
            self.assertEqual(
                categories, {"in_distribution": 10, "edge": 10, "unrelated": 10}
            )
        # 评估 in_distribution 问题不应出现在 SFT 训练集中
        sft_questions = {record["q"] for record in read_jsonl(DATA_DIR / "demo_sft_en.jsonl")}
        for record in eval_en:
            if record["category"] == "in_distribution":
                self.assertNotIn(record["q"], sft_questions)

    def test_build_script_is_deterministic_and_matches_committed(self) -> None:
        """验证 构建 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "demo"
            result = _run_script(
                ROOT / "scripts" / "build_demo_data.py",
                ["--out-dir", str(out_dir)],
                cwd=Path(tmp),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in DEMO_FILES:
                generated = out_dir / name
                committed = DATA_DIR / name
                self.assertTrue(generated.is_file(), name)
                self.assertEqual(sha256_file(generated), sha256_file(committed), f"{name} 与提交版本不一致")
            # 再跑一次仍一致
            second = _run_script(
                ROOT / "scripts" / "build_demo_data.py",
                ["--out-dir", str(out_dir)],
                cwd=Path(tmp),
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(sha256_file(out_dir / "demo_sft_zh.jsonl"), sha256_file(DATA_DIR / "demo_sft_zh.jsonl"))


class SftPrepareTest(unittest.TestCase):
    """覆盖 SftPrepareTest 相关行为。"""
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.raw = self.tmp / "raw"
        self.out = self.tmp / "out"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _run(self, args: list[str], out: Path | None = None) -> subprocess.CompletedProcess:
        base = [
            "--data_dir", str(out or self.out),
            "--download_dir", str(self.raw),
            "--pretrain_data_dir", str(self.tmp / "no_pretrain"),
            *args,
        ]
        return _run_script(ROOT / "prepare_sft.py", base, cwd=self.tmp)

    def _split_groups(self, out: Path) -> dict[str, set[str]]:
        return {
            split: {record["group"] for record in read_jsonl(out / f"{split}.jsonl")}
            for split in ("train", "val", "test")
        }

    # ------------------------------------------------------------------
    def test_synthetic_en(self) -> None:
        """验证 `test_synthetic_en` 对应的行为约定。"""
        result = self._run(["--source", "synthetic_en"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "sft_meta.json", "split_report.json"):
            self.assertTrue((self.out / name).is_file(), f"缺少产物 {name}")

        splits = {split: read_jsonl(self.out / f"{split}.jsonl") for split in ("train", "val", "test")}
        counts = {split: len(records) for split, records in splits.items()}
        self.assertEqual(sum(counts.values()), 500)
        self.assertEqual(counts, {"train": 450, "val": 25, "test": 25})
        groups = self._split_groups(self.out)
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        for records in splits.values():
            for record in records:
                for key in ("q", "a", "group", "category", "source"):
                    self.assertIn(key, record)
                self.assertEqual(record["source"], "demo_sft_en")

        meta = read_json(self.out / "sft_meta.json")
        self.assertEqual(meta["stage"], "sft")
        self.assertEqual(meta["source"], "synthetic_en")
        self.assertEqual(meta["format"], SFT_FORMAT_TEMPLATE)
        self.assertEqual(meta["counts"]["splits"], counts)
        self.assertEqual(meta["group_split"]["group_integrity"], True)
        self.assertIsNone(meta["tokenizer"])
        for split in ("train", "val", "test"):
            self.assertEqual(meta["splits"][split]["examples"], counts[split])
            self.assertEqual(meta["splits"][split]["sha256"], sha256_file(self.out / f"{split}.jsonl"))
        report = read_json(self.out / "split_report.json")
        self.assertEqual(report["stage"], "sft")
        self.assertEqual(report["group_integrity"], True)

    def test_synthetic_zh_and_sft_source_field(self) -> None:
        """验证 SFT 相关行为。"""
        out = self.tmp / "out_zh"
        result = self._run(["--sft_source", "synthetic_zh"], out=out)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = read_jsonl(out / "train.jsonl")
        self.assertEqual(len(records), 450)
        for record in records:
            self.assertEqual(record["source"], "demo_sft_zh")
            self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in record["q"]))
        meta = read_json(out / "sft_meta.json")
        self.assertEqual(meta["source"], "synthetic_zh")

    def test_max_samples_cap(self) -> None:
        """验证 `test_max_samples_cap` 对应的行为约定。"""
        result = self._run(["--source", "synthetic_en", "--max_samples", "100"])
        self.assertEqual(result.returncode, 0, result.stderr)
        counts = {
            split: len(read_jsonl(self.out / f"{split}.jsonl")) for split in ("train", "val", "test")
        }
        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(counts["val"] + counts["test"], 10)  # int(100*0.05)*2
        meta = read_json(self.out / "sft_meta.json")
        self.assertEqual(meta["counts"]["kept"], 100)
        self.assertEqual(meta["max_samples"], 100)

    def test_custom_jsonl_group_split_and_validation(self) -> None:
        """验证 切分、校验 相关行为。"""
        input_path = self.tmp / "custom.jsonl"
        rows: list[dict] = []
        # 8 个 group，条数不同（group 成员必须整体落在同一 split）
        for group, size in (("alpha", 4), ("beta", 3), ("gamma", 3), ("delta", 2), ("epsilon", 2), ("zeta", 2), ("eta", 2), ("theta", 2)):
            for index in range(size):
                rows.append({"q": f"question {group} {index}", "a": f"answer {group} {index}", "group": group, "category": "custom"})
        rows.append(dict(rows[0]))  # 完全重复
        rows.append({"q": "", "a": "empty question"})
        rows.append({"q": "empty answer", "a": ""})
        input_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
        )
        result = self._run(
            [
                "--source", "jsonl",
                "--input", str(input_path),
                "--val_fraction", "0.25",
                "--test_fraction", "0.25",
                "--max_samples", "1000",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = read_json(self.out / "sft_meta.json")
        self.assertEqual(meta["counts"]["input"], 23)
        self.assertEqual(meta["counts"]["invalid"], 2)
        self.assertEqual(meta["counts"]["duplicates"], 1)
        self.assertEqual(meta["counts"]["kept"], 20)
        splits = {split: read_jsonl(self.out / f"{split}.jsonl") for split in ("train", "val", "test")}
        self.assertEqual(sum(len(records) for records in splits.values()), 20)
        groups = self._split_groups(self.out)
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        # 8 组，25%/25% -> val 2 组、test 2 组、train 4 组
        self.assertEqual(meta["group_split"]["group_counts"], {"train": 4, "val": 2, "test": 2})
        self.assertEqual(meta["group_split"]["example_counts"]["val"], len(splits["val"]))
        for split, records in splits.items():
            other_groups = set().union(*(groups[name] for name in groups if name != split))
            for record in records:
                self.assertIn(record["group"], groups[split])
                self.assertNotIn(record["group"], other_groups)

    def test_jsonl_source_requires_input(self) -> None:
        """验证 `test_jsonl_source_requires_input` 对应的行为约定。"""
        result = self._run(["--source", "jsonl"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("jsonl", result.stderr)

    def test_unknown_source_fails(self) -> None:
        """验证 `test_unknown_source_fails` 对应的行为约定。"""
        result = self._run(["--source", "bogus"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("未知 source", result.stderr)

    def test_tokenizer_coverage_check(self) -> None:
        # 1) 先用本地文本生成预训练数据（含 tokenizer.json）
        """验证 tokenizer 相关行为。"""
        corpus = self.tmp / "corpus.txt"
        corpus.write_text(
            "The quick brown fox jumps over the lazy dog. 0123456789 Q: A: \n" * 100,
            encoding="utf-8",
        )
        pretrain_out = self.tmp / "pretrain"
        prep = _run_script(
            ROOT / "prepare.py",
            [
                "--dataset", "input",
                "--input_path", str(corpus),
                "--data_dir", str(pretrain_out),
                "--download_dir", str(self.raw),
            ],
            cwd=self.tmp,
        )
        self.assertEqual(prep.returncode, 0, prep.stderr)
        # 2) SFT 准备时显式做覆盖率检查
        out = self.tmp / "sft_cov"
        result = self._run(
            ["--source", "synthetic_en", "--tokenizer_dir", str(pretrain_out)],
            out=out,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = read_json(out / "sft_meta.json")
        tokenizer_info = meta["tokenizer"]
        self.assertIsNotNone(tokenizer_info)
        # Windows 下 TEMP 可能返回 8.3 短路径（ADMINI~1），统一 realpath + normcase 后比较。
        self.assertEqual(
            os.path.normcase(os.path.realpath(tokenizer_info["tokenizer_dir"])),
            os.path.normcase(os.path.realpath(pretrain_out)),
        )
        # 预训练语料是简单英文文本，演示 SFT 的大多数字符已覆盖；
        # 少量标点（如 ?、'）未登录，覆盖率应在合理区间且与 unk_rate 互补。
        self.assertGreater(tokenizer_info["coverage"], 0.9)
        self.assertLess(tokenizer_info["unk_rate"], 0.1)
        self.assertAlmostEqual(
            tokenizer_info["coverage"], 1.0 - tokenizer_info["unk_rate"], places=9
        )
        self.assertGreater(tokenizer_info["vocab_size"], 3)

    def test_alpaca_zh_uses_cache_and_samples_deterministically(self) -> None:
        """验证 缓存 相关行为。"""
        cache = self.raw / "alpaca_gpt4_data_zh.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "instruction": f"这是第{index}个问题",
                "input": f"补充信息{index}" if index % 5 == 0 else "",
                "output": f"这是第{index}个回答",
            }
            for index in range(40)
        ]
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        out = self.tmp / "alpaca"
        result = self._run(["--source", "alpaca_zh", "--max_samples", "20", "--seed", "7"], out=out)
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = read_json(out / "sft_meta.json")
        self.assertEqual(meta["source"], "alpaca_zh")
        self.assertEqual(meta["counts"]["kept"], 20)
        self.assertIn("CC BY-NC", meta["source_info"]["license"])
        records = [
            record
            for split in ("train", "val", "test")
            for record in read_jsonl(out / f"{split}.jsonl")
        ]
        self.assertEqual(len(records), 20)
        for record in records:
            self.assertEqual(record["category"], "alpaca_zh")
            self.assertEqual(record["source"], "alpaca_zh")
        # instruction + input 合并进 q
        self.assertTrue(any("\n补充信息" in record["q"] for record in records))

        # 相同 seed 再跑一次，文件逐字节一致
        out2 = self.tmp / "alpaca2"
        again = self._run(["--source", "alpaca_zh", "--max_samples", "20", "--seed", "7"], out=out2)
        self.assertEqual(again.returncode, 0, again.stderr)
        for split in ("train", "val", "test"):
            self.assertEqual(
                (out / f"{split}.jsonl").read_bytes(), (out2 / f"{split}.jsonl").read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
