"""Failure-path coverage: NaN circuit breaker, early stop, interrupt, exception.

Also covers gradient-accumulation poisoning: a non-finite micro-batch must not
silently clear already-accumulated gradients and then step with the remainder.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch

    from minigpt.backends.torch_backend import TorchBackend
except ImportError:  # 无 torch 环境：torch 用例由类装饰器跳过，pure 用例仍可运行
    torch = None
    TorchBackend = None
from minigpt.config import Config
from minigpt.training.trainer import Trainer


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


def make_config(tmp, **overrides):
    base = dict(
        project_root=tmp,
        data_dir=tmp,
        runs_dir=str(Path(tmp) / "runs"),
        run_name="fail-run",
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


def make_trainer(config, resume=False):
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


@unittest.skipIf(torch is None, "torch not installed")
class GradAccumPoisoningTests(unittest.TestCase):
    """覆盖 GradAccumPoisoningTests 相关行为。"""
    def test_nonfinite_microbatch_does_not_step_with_partial_gradients(self) -> None:
        """Window = [good, NaN]: the whole window must be discarded, no optimizer step."""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, grad_accum_steps=2, max_steps=1)
            trainer = make_trainer(config)
            backend = trainer.backend
            real_train_step = backend.train_step
            calls = {"n": 0}

            def flaky_train_step(model, optimizer, batch, **kwargs):
                calls["n"] += 1
                if calls["n"] == 2:
                    # Simulate an fp16 overflow on the second micro-batch.
                    from minigpt.backends.base import StepResult

                    return StepResult(loss=float("nan"), grad_norm=float("nan"), tokens=16, stepped=False)
                return real_train_step(model, optimizer, batch, **kwargs)

            params_before = [p.detach().clone() for p in trainer.model.parameters()]
            step_results = []
            real_optimizer_step = backend.optimizer_step

            def spy_optimizer_step(*args, **kwargs):
                result = real_optimizer_step(*args, **kwargs)
                step_results.append(result)
                return result

            with mock.patch.object(backend, "train_step", side_effect=flaky_train_step):
                with mock.patch.object(backend, "optimizer_step", side_effect=spy_optimizer_step):
                    trainer.train()
            # The window contained a NaN micro-batch: no parameter update may happen.
            for result in step_results:
                self.assertFalse(result.stepped, "stepped with poisoned gradients")
            params_after = list(trainer.model.parameters())
            for before, after in zip(params_before, params_after):
                self.assertTrue(torch.equal(before, after.detach()), "parameters changed despite NaN window")

    def test_all_good_window_still_steps(self) -> None:
        """验证 `test_all_good_window_still_steps` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, grad_accum_steps=2, max_steps=1)
            trainer = make_trainer(config)
            params_before = [p.detach().clone() for p in trainer.model.parameters()]
            trainer.train()
            changed = any(
                not torch.equal(before, after.detach())
                for before, after in zip(params_before, trainer.model.parameters())
            )
            self.assertTrue(changed, "healthy window should update parameters")


@unittest.skipIf(torch is None, "torch not installed")
class TrainFailurePathTests(unittest.TestCase):
    """覆盖 TrainFailurePathTests 相关行为。"""
    def _status(self, trainer):
        return json.loads((trainer.run_dir / "status.json").read_text(encoding="utf-8"))

    def test_keyboard_interrupt_marks_stopped(self) -> None:
        """验证 `test_keyboard_interrupt_marks_stopped` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            with mock.patch.object(trainer.data_source, "train_batch", side_effect=KeyboardInterrupt):
                result = trainer.train()
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(self._status(trainer)["status"], "stopped")

    def test_exception_marks_failed_and_reraises(self) -> None:
        """验证 `test_exception_marks_failed_and_reraises` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            with mock.patch.object(
                trainer.data_source, "train_batch", side_effect=ValueError("boom")
            ):
                with self.assertRaises(ValueError):
                    trainer.train()
            status = self._status(trainer)
            self.assertEqual(status["status"], "failed")
            self.assertIn("boom", status.get("error", ""))

    def test_early_stop_marks_stopped_and_saves(self) -> None:
        """验证 早停、停止 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, max_steps=10, eval_interval=1, patience=1, min_delta=0.0)
            trainer = make_trainer(config)
            losses = iter([2.0, 2.5, 2.6, 2.7, 2.8, 2.9])
            with mock.patch.object(trainer, "evaluate", side_effect=lambda *a, **k: next(losses)):
                result = trainer.train()
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["status"], "stopped")
            self.assertTrue((trainer.run_dir / "last.pt").is_file())

    def test_nonfinite_eval_loss_raises_instead_of_silent_skip(self) -> None:
        """验证 评估、loss 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            from minigpt.backends.base import EvalResult

            with mock.patch.object(
                trainer.backend, "eval_step", return_value=EvalResult(loss=float("nan"), tokens=16)
            ):
                with self.assertRaises(RuntimeError):
                    trainer.evaluate("val", 2)


if __name__ == "__main__":
    unittest.main()