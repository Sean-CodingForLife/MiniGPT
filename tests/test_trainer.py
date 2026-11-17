"""针对 `test_trainer.py` 的单元测试与回归保护。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from minigpt.backends.torch_backend import TorchBackend
except ImportError:
    raise unittest.SkipTest("torch not installed") from None
from minigpt.config import Config
from minigpt.training.checkpoint import load_checkpoint, validate_checkpoint
from minigpt.training.trainer import Trainer, check_mini_run, cosine_lr


class FakeTokenizer:
    """覆盖 FakeTokenizer 相关行为。"""
    vocab_size = 20
    eos_id = 1
    pad_id = 2

    def state_dict(self):
        return {"type": "fake", "vocab_size": self.vocab_size, "specials": {"<eos>": 1, "<pad>": 2}}


class FakeSource:
    """覆盖 FakeSource 相关行为。"""
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.metadata = {"fake": True}
        self.step = 0

    def _batch(self):
        self.step += 1
        x = [[(self.step + i + j) % 20 for j in range(8)] for i in range(2)]
        y = [[(value + 1) % 20 for value in row] for row in x]
        return {"x": x, "y": y}

    def train_batch(self, batch_size, block_size, rng=None):
        return self._batch()

    def eval_batches(self, split, batch_size, block_size, num_batches, seed=1234):
        return [self._batch() for _ in range(max(1, num_batches))]


class TrainerTests(unittest.TestCase):
    """覆盖 TrainerTests 相关行为。"""
    def _config(self, tmp, **overrides):
        base = dict(
            project_root=tmp,
            data_dir=tmp,
            runs_dir=str(Path(tmp) / "runs"),
            run_name="unit-run",
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            dropout=0.0,
            batch_size=2,
            max_steps=3,
            warmup_steps=1,
            lr=1e-3,
            min_lr=1e-4,
            eval_interval=2,
            eval_iters=1,
            log_interval=1,
            save_interval=2,
            patience=5,
            callbacks="telemetry",
            seed=123,
        )
        base.update(overrides)
        return Config(**base)

    def _make(self, config, resume=False):
        backend = TorchBackend(config)
        model = backend.build_model(config, 20)
        optimizer = backend.build_optimizer(model, config)
        source = FakeSource()
        return Trainer(
            config=config,
            backend=backend,
            model=model,
            optimizer=optimizer,
            tokenizer=source.tokenizer,
            data_source=source,
            resume=resume,
        )

    def test_training_writes_artifacts_and_resumes(self) -> None:
        """验证 训练 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp)
            trainer = self._make(config)
            result = trainer.train()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps"], 3)
            run_dir = trainer.run_dir
            for name in (
                "run_config.json",
                "environment.json",
                "metrics.csv",
                "events.jsonl",
                "status.json",
                "training_report.json",
                "best.pt",
                "last.pt",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            payload = load_checkpoint(run_dir / "last.pt")
            validate_checkpoint(payload)
            self.assertEqual(payload["iter_num"], 3)

            config2 = self._config(tmp, max_steps=5, resume=True)
            trainer2 = self._make(config2, resume=True)
            result2 = trainer2.train()
            self.assertEqual(result2["status"], "completed")
            self.assertEqual(result2["steps"], 5)
            self.assertEqual(load_checkpoint(trainer2.run_dir / "last.pt")["iter_num"], 5)

    def test_mini_gate_report_logic(self) -> None:
        """验证 报告 相关行为。"""
        good = {
            "initial_loss": 4.2,
            "final_train_loss": 3.9,
            "grad_norms": [0.5, 0.4, 0.6],
        }
        report = check_mini_run(good, vocab_size=68, loss_tolerance=0.5, min_drop=0.2)
        self.assertTrue(report["passed"])
        bad = {"initial_loss": 1.0, "final_train_loss": 0.9, "grad_norms": []}
        self.assertFalse(check_mini_run(bad, vocab_size=68)["passed"])

    def test_cosine_schedule(self) -> None:
        """验证 学习率调度 相关行为。"""
        self.assertAlmostEqual(cosine_lr(0, 100, 10, 1e-3, 1e-4), 1e-4)
        self.assertAlmostEqual(cosine_lr(9, 100, 10, 1e-3, 1e-4), 1e-3)
        self.assertAlmostEqual(cosine_lr(100, 100, 10, 1e-3, 1e-4), 1e-4)


if __name__ == "__main__":
    unittest.main()
