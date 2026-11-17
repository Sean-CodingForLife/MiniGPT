"""``minigpt.web.stages`` + 只读阶段接口的测试（全部用临时目录，不碰真实仓库）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from minigpt.web import stages
from minigpt.web.server import ApiError, DashboardApp

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_uint16(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(int(value).to_bytes(2, "little") for value in values)
    path.write_bytes(data)


def make_pretrain_dataset(directory: Path, name: str = "corpus") -> None:
    write_json(
        directory / "meta.json",
        {
            "dataset": name,
            "source": {"kind": "download", "name": "tiny", "sha256": "a" * 64, "url": "file://local"},
            "source_sha256": "a" * 64,
            "vocab_size": 5,
            "specials": {"<unk>": 0, "<eos>": 1, "<pad>": 2},
            "splits": {
                "train": {"tokens": 8, "bytes": 16, "sha256": "b" * 64},
                "val": {"tokens": 2, "bytes": 4, "sha256": "c" * 64},
                "test": {"tokens": 2, "bytes": 4, "sha256": "d" * 64},
            },
            "unk_rate": {"train": 0.0, "val": 0.0, "test": 0.0},
            "val_fraction": 0.05,
            "test_fraction": 0.05,
            "tokenizer_built_on": "train",
            "vocab_extra": {"enabled": False, "files": [], "chars": 0, "novel_chars": 0},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    write_json(
        directory / "tokenizer.json",
        {"type": "char", "version": 1, "specials": {"<unk>": 0, "<eos>": 1, "<pad>": 2}, "chars": ["a", "b"]},
    )
    # token id 3 -> "a", 4 -> "b"，特殊 token 会被预览跳过
    write_uint16(directory / "train.bin", [3, 4, 3, 4, 1, 3])
    write_uint16(directory / "val.bin", [3])
    write_uint16(directory / "test.bin", [4])
    write_json(directory / "split_report.json", {"split": {"train": 8}})


def make_sft_dataset(directory: Path, name: str = "sft") -> None:
    write_json(
        directory / "sft_meta.json",
        {
            "stage": "sft",
            "source": name,
            "source_info": {
                "kind": "public_dataset",
                "name": name,
                "license": "CC BY-NC 4.0",
                "sha256": "e" * 64,
                "records_total": 20,
                "url": "https://example.invalid/data",
            },
            "format": "Q: {q}\nA: {a}<eos>",
            "fields": ["q", "a", "group"],
            "counts": {"input": 20, "invalid": 0, "duplicates": 0, "valid": 20, "kept": 6, "splits": {"train": 4, "val": 1, "test": 1}},
            "splits": {
                "train": {"examples": 4, "bytes": 40, "sha256": "f" * 64},
                "val": {"examples": 1, "bytes": 10, "sha256": "1" * 64},
                "test": {"examples": 1, "bytes": 10, "sha256": "2" * 64},
            },
            "group_split": {"group_integrity": True, "group_counts": {"train": 4, "val": 1, "test": 1}},
            "tokenizer": {"vocab_size": 5, "coverage": 0.999, "unk_rate": 0.0001},
            "seed": 1337,
            "val_fraction": 0.05,
            "test_fraction": 0.05,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    for split, count in (("train", 4), ("val", 1), ("test", 1)):
        lines = [
            json.dumps({"q": f"问题{i}", "a": f"答案{i}", "group": f"g{i}"}, ensure_ascii=False)
            for i in range(count)
        ]
        (directory / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_evaluation(run_dir: Path, eval_name: str = "eval_test") -> None:
    eval_dir = run_dir / eval_name
    write_json(
        eval_dir / "evaluation.json",
        {
            "stage": "sft",
            "split": "test",
            "checkpoint": str(run_dir / "best.pt"),
            "created_at": "2026-01-02T00:00:00+00:00",
            "tokens": 100,
            "answer_loss": 2.5,
            "perplexity": 12.0,
            "metrics": {
                "exact_match": 0.0,
                "char_f1": 0.25,
                "empty_rate": 0.1,
                "count": 4,
                "by_category": {"greetings": {"count": 2, "char_f1": 0.4, "exact_match": 0.0}},
            },
            "sample": "hello there",
        },
    )
    predictions = [{"q": "hi", "reference": "hello", "prediction": "helo", "category": "greetings"}]
    (eval_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n", encoding="utf-8"
    )
    (eval_dir / "capability_boundary.md").write_text("# boundary\n\n- 阶段：sft\n", encoding="utf-8")


def make_export(directory: Path, name: str = "pkg") -> None:
    write_json(
        directory / "MANIFEST.json",
        {
            "format": "minigpt-bundle-v1",
            "format_version": 1,
            "created_at": "2026-01-03T00:00:00+00:00",
            "source": {"run_id": "demo", "parameters": 1234, "checkpoint_file": "best.pt", "checkpoint_sha256": "3" * 64},
            "files": {
                "config.json": {"bytes": 10, "sha256": "4" * 64},
                "pytorch_model.bin": {"bytes": 20, "sha256": "5" * 64},
            },
        },
    )
    write_json(directory / "config.json", {"n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 64, "vocab_size": 9})
    (directory / "MODEL_CARD.md").write_text("# model card\n", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "pytorch_model.bin").write_bytes(b"x" * 20)


class DatasetScanTests(unittest.TestCase):
    """覆盖 DatasetScanTests 相关行为。"""
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "data"
        make_pretrain_dataset(self.data_dir / "corpus")
        make_sft_dataset(self.data_dir / "sft")
        (self.data_dir / "raw").mkdir()
        (self.data_dir / "raw" / "input.txt").write_text("raw", encoding="utf-8")
        (self.data_dir / "empty_dir").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scan_lists_both_kinds_and_skips_no_meta(self) -> None:
        """验证 `test_scan_lists_both_kinds_and_skips_no_meta` 对应的行为约定。"""
        payload = stages.scan_datasets(self.data_dir)
        self.assertEqual(payload["exists"], True)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["pretrain_count"], 1)
        self.assertEqual(payload["sft_count"], 1)
        names = sorted(item["name"] for item in payload["datasets"])
        self.assertEqual(names, ["corpus", "sft"])
        self.assertEqual([item["name"] for item in payload["raw_files"]], ["input.txt"])
        sft = next(item for item in payload["datasets"] if item["name"] == "sft")
        self.assertEqual(sft["kind"], "sft")
        self.assertEqual(sft["counts"]["kept"], 6)
        self.assertEqual(sft["vocab_size"], 5)

    def test_dataset_detail_returns_samples_and_preview(self) -> None:
        """验证 数据集 相关行为。"""
        detail = stages.dataset_detail(self.data_dir, "sft", limit=2)
        self.assertEqual(len(detail["samples"]), 2)
        self.assertEqual(detail["samples"][0]["q"], "问题0")
        self.assertEqual(detail["coverage"], 0.999)
        self.assertIsNone(detail["tokenizer"])  # SFT 目录只有 meta 里的 tokenizer 统计
        pretrain = stages.dataset_detail(self.data_dir, "corpus")
        self.assertEqual(pretrain["preview"]["text"], "ababa")  # <eos> 被跳过
        self.assertEqual(pretrain["splits"]["train"]["tokens"], 8)
        self.assertEqual(pretrain["tokenizer"]["vocab_size"], 5)

    def test_dataset_names_cannot_escape_root(self) -> None:
        """验证 数据集、转义 相关行为。"""
        for bad in ("../runs", "corpus/..", "..", "./corpus", ""):
            self.assertIsNone(stages.dataset_detail(self.data_dir, bad))
        self.assertIsNone(stages.scan_datasets(self.root / "missing")["datasets"] or None)


class EvaluationScanTests(unittest.TestCase):
    """覆盖 EvaluationScanTests 相关行为。"""
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs_dir = self.root / "runs"
        make_evaluation(self.runs_dir / "demo")
        (self.runs_dir / "no-eval").mkdir()
        (self.runs_dir / "demo" / "train_only").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scan_and_detail(self) -> None:
        """验证 `test_scan_and_detail` 对应的行为约定。"""
        payload = stages.scan_evaluations(self.runs_dir)
        self.assertEqual(payload["count"], 1)
        entry = payload["evaluations"][0]
        self.assertEqual(entry["run_id"], "demo")
        self.assertEqual(entry["eval_dir"], "eval_test")
        self.assertEqual(entry["checkpoint_name"], "best.pt")
        self.assertEqual(entry["prediction_count"], 1)
        self.assertTrue(entry["has_boundary"])
        self.assertEqual(entry["metrics"]["perplexity"], 12.0)
        detail = stages.evaluation_detail(self.runs_dir, "demo")
        self.assertEqual(detail["report"]["split"], "test")
        self.assertEqual(len(detail["predictions"]), 1)
        self.assertIn("阶段：sft", detail["boundary"])
        self.assertEqual(detail["report"]["metrics"]["by_category"]["greetings"]["count"], 2)

    def test_detail_rejects_unknown_or_escaping_names(self) -> None:
        """验证 `test_detail_rejects_unknown_or_escaping_names` 对应的行为约定。"""
        self.assertIsNone(stages.evaluation_detail(self.runs_dir, "../data"))
        self.assertIsNone(stages.evaluation_detail(self.runs_dir, "demo", eval_name="../eval_test"))
        self.assertIsNone(stages.evaluation_detail(self.runs_dir, "missing"))


class ExportScanTests(unittest.TestCase):
    """覆盖 ExportScanTests 相关行为。"""
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dist_dir = self.root / "dist"
        make_export(self.dist_dir / "pkg")
        (self.dist_dir / "pkg.zip").write_bytes(b"z" * 8)
        (self.dist_dir / "not-a-bundle").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scan_and_detail(self) -> None:
        """验证 `test_scan_and_detail` 对应的行为约定。"""
        payload = stages.scan_exports(self.dist_dir)
        self.assertEqual(payload["count"], 1)
        entry = payload["exports"][0]
        self.assertEqual(entry["format"], "minigpt-bundle-v1")
        self.assertEqual(entry["parameters"], 1234)
        self.assertEqual(entry["file_count"], 2)
        self.assertTrue(entry["has_zip"])
        self.assertEqual(entry["missing_files"], ["run.py", "serve.py"])
        self.assertEqual([item["name"] for item in payload["zips"]], ["pkg.zip"])
        detail = stages.export_detail(self.dist_dir, "pkg")
        self.assertEqual(len(detail["manifest_files"]), 2)
        self.assertEqual(detail["manifest_files"][0]["sha256"], "4" * 64)
        self.assertIn("model card", detail["model_card"])
        self.assertEqual(detail["config"]["n_layer"], 2)

    def test_name_validation(self) -> None:
        """验证 校验 相关行为。"""
        self.assertIsNone(stages.export_detail(self.dist_dir, "../runs"))
        self.assertIsNone(stages.export_detail(self.dist_dir, "not-a-bundle"))


class StageAppTests(unittest.TestCase):
    """覆盖 StageAppTests 相关行为。"""
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "data"
        self.runs_dir = self.root / "runs"
        self.dist_dir = self.root / "dist"
        make_pretrain_dataset(self.data_dir / "corpus")
        make_evaluation(self.runs_dir / "demo")
        make_export(self.dist_dir / "pkg")
        self.app = DashboardApp(
            project_root=self.root,
            runs_dir=self.runs_dir,
            web_dir=ROOT / "web",
            configs_dir=self.root / "configs",
            data_dir=self.data_dir,
            dist_dir=self.dist_dir,
            python_executable="python",
        )

    def tearDown(self) -> None:
        self.app.close()
        self.tmp.cleanup()

    def test_app_exposes_stage_payloads(self) -> None:
        """验证 阶段页 相关行为。"""
        self.assertEqual(self.app.datasets()["count"], 1)
        self.assertEqual(self.app.dataset("corpus")["name"], "corpus")
        self.assertEqual(self.app.evaluations()["count"], 1)
        self.assertEqual(self.app.evaluation("demo")["run_id"], "demo")
        self.assertEqual(self.app.exports()["count"], 1)
        self.assertEqual(self.app.export("pkg")["name"], "pkg")

    def test_app_returns_404_for_bad_names(self) -> None:
        """验证 `test_app_returns_404_for_bad_names` 对应的行为约定。"""
        for call in (
            lambda: self.app.dataset("../runs"),
            lambda: self.app.evaluation("../data"),
            lambda: self.app.export("../configs"),
        ):
            with self.assertRaises(ApiError) as caught:
                call()
            self.assertEqual(caught.exception.status, 404)

    def test_default_dirs_are_project_relative(self) -> None:
        """验证 项目 相关行为。"""
        app = DashboardApp(
            project_root=self.root,
            runs_dir=self.runs_dir,
            web_dir=ROOT / "web",
            configs_dir=self.root / "configs",
        )
        try:
            self.assertEqual(app.data_dir, self.root / "data")
            self.assertEqual(app.dist_dir, self.root / "dist")
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
