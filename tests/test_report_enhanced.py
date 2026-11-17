"""N12: 训练报告新结构 + 导出模型卡自动附带训练摘要。

报告新增：hyperparameters（portable 配置）、data_fingerprint（数据 sha256）、
metrics 汇总（initial/final/best/steps/elapsed）、capability_boundary（阶段说明）。
旧字段语义保留（顶层仍可读取）。导出 bundle 的 MODEL_CARD.md 附带这些摘要。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.bundle import export_bundle

from tests.test_bundle import make_checkpoint
from tests.test_trainer_failures import FakeSource, make_config, make_trainer


class FingerprintSource(FakeSource):
    """带真实数据文件哈希元信息的数据源（模拟 prepare.py 的 meta.json）。"""

    def __init__(self):
        super().__init__()
        self.metadata = {
            "dataset": "unit",
            "source_sha256": "a" * 64,
            "splits": {
                "train": {"tokens": 16, "sha256": "b" * 64},
                "val": {"tokens": 8, "sha256": "c" * 64},
            },
        }


class TrainingReportV2Tests(unittest.TestCase):
    """覆盖 TrainingReportV2Tests 相关行为。"""
    def _train(self, tmp: str, source=None, **overrides):
        config = make_config(tmp, max_steps=3, eval_interval=1, **overrides)
        trainer = make_trainer(config)
        if source is not None:
            trainer.data_source = source
        trainer.train()
        report = json.loads((trainer.run_dir / "training_report.json").read_text(encoding="utf-8"))
        return trainer, report

    def test_report_has_new_sections(self) -> None:
        """验证 报告 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            trainer, report = self._train(tmp, source=FingerprintSource())
            for key in ("hyperparameters", "data_fingerprint", "metrics", "capability_boundary"):
                self.assertIn(key, report, key)
            # hyperparameters = config.to_dict(portable=True)
            self.assertEqual(report["hyperparameters"], trainer.config.to_dict(portable=True))
            self.assertEqual(report["hyperparameters"]["max_steps"], 3)
            # data_fingerprint 来自 data_source.metadata 的 sha256
            fingerprint = report["data_fingerprint"]
            self.assertEqual(fingerprint["train.bin"], "b" * 64)
            self.assertEqual(fingerprint["val.bin"], "c" * 64)
            self.assertEqual(fingerprint["source"], "a" * 64)
            # metrics 汇总
            metrics = report["metrics"]
            for key in ("initial_loss", "final_train_loss", "best_val_loss", "best_step", "steps", "elapsed_s"):
                self.assertIn(key, metrics, key)
            self.assertEqual(metrics["steps"], 3)
            self.assertAlmostEqual(metrics["initial_loss"], report["initial_loss"])
            self.assertAlmostEqual(metrics["best_val_loss"], report["best_val_loss"])
            self.assertGreater(metrics["elapsed_s"], 0.0)
            # capability_boundary 与阶段相关
            self.assertIn("预训练", report["capability_boundary"])
            # 旧字段语义保留
            self.assertEqual(report["run_id"], trainer.run_id)
            self.assertEqual(report["status"], "completed")
            self.assertIn("均值", report["final_train_loss_note"])

    def test_report_sft_boundary_and_empty_fingerprint(self) -> None:
        """验证 报告、SFT 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._train(tmp, stage="sft")
            self.assertIn("SFT", report["capability_boundary"])
            # FakeSource.metadata 没有 sha256 → 空 dict
            self.assertEqual(report["data_fingerprint"], {})

    def test_report_fingerprint_mixed_source(self) -> None:
        """验证 报告 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            source = FingerprintSource()
            source.metadata = {
                "pretrain": {"splits": {"train": {"sha256": "d" * 64}}},
                "sft": {"splits": {"train": {"sha256": "e" * 64}}},
                "pretrain_ratio": 0.15,
            }
            _, report = self._train(tmp, source=source, stage="sft")
            fingerprint = report["data_fingerprint"]
            self.assertEqual(fingerprint["pretrain/train.bin"], "d" * 64)
            self.assertEqual(fingerprint["sft/train.jsonl"], "e" * 64)


class ModelCardSummaryTests(unittest.TestCase):
    """覆盖 ModelCardSummaryTests 相关行为。"""
    def test_model_card_contains_training_summary(self) -> None:
        """验证 模型、训练、摘要 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "run"
            run_dir.mkdir()
            checkpoint = make_checkpoint(run_dir)
            summary = {
                "run_id": "bundle-unit",
                "stage": "sft",
                "status": "completed",
                "hyperparameters": {"lr": 0.001, "max_steps": 3, "batch_size": 2, "n_layer": 1},
                "data_fingerprint": {"train.jsonl": "f" * 64},
                "metrics": {
                    "initial_loss": 3.0,
                    "final_train_loss": 2.5,
                    "best_val_loss": 2.0,
                    "best_step": 2,
                    "steps": 3,
                    "elapsed_s": 1.5,
                },
                "capability_boundary": "SFT 只教问答格式，不保证回答正确。",
            }
            (run_dir / "training_report.json").write_text(
                json.dumps(summary, ensure_ascii=False), encoding="utf-8"
            )
            bundle_dir = base / "bundle"
            export_bundle(checkpoint, bundle_dir, create_zip=False, force=True)
            card = (bundle_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
            self.assertIn("训练摘要", card)
            self.assertIn("超参数", card)
            self.assertIn("0.001", card)  # lr
            self.assertIn("数据指纹", card)
            self.assertIn("f" * 64, card)
            self.assertIn("能力边界", card)
            self.assertIn("SFT 只教问答格式", card)
            self.assertIn("2.0", card)  # best_val_loss

    def test_model_card_without_report_still_exports(self) -> None:
        """验证 模型、报告 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            checkpoint = make_checkpoint(base)
            bundle_dir = base / "bundle"
            export_bundle(checkpoint, bundle_dir, create_zip=False, force=True)
            card = (bundle_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
            self.assertIn("Model Card", card)
            config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
            self.assertNotIn("training_summary", config)


if __name__ == "__main__":
    unittest.main()