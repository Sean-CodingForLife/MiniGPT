"""Batch B3 regression tests: trainer protocol fixes F11-F17.

F11 early-stop status step off-by-one
F12 non-finite eval batches warn + event
F13 resume validates the checkpoint before rewriting run metadata
F14 early-stopping resume keeps the current patience/min_delta config
F15 status.json has a single writer (no field clobbering)
F16 mini-run gate uses the tail mean of train losses
F17 gradient accumulation weights micro-batches by valid tokens
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:  # 无 torch 环境：torch 用例跳过，pure 用例仍运行
    torch = None

from minigpt.backends.base import EvalResult, StepResult
from minigpt.config import Config
from minigpt.pure.backend import PureBackend
from minigpt.training.trainer import Trainer, check_mini_run

from tests.test_trainer_failures import FakeSource, make_config, make_trainer


def read_status(trainer) -> dict:
    return json.loads((trainer.run_dir / "status.json").read_text(encoding="utf-8"))


def read_events(trainer) -> list[dict]:
    path = trainer.run_dir / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class UnequalTokenSource(FakeSource):
    """Alternating micro-batches with 4 vs 12 valid tokens (SFT-style masking)."""

    def _batch(self):
        self.step += 1
        if self.step % 2 == 1:
            x = [[1, 2, 3, 4]]
            y = [[2, 3, 4, 5]]
        else:
            x = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]]
            y = [[-100, -100, -100, -100, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 10]]
        return {"x": x, "y": y}

    def eval_batches(self, split, batch_size, block_size, num_batches, seed=1234):
        # Fixed eval batches: evaluation must not shift the train-window counter.
        return [{"x": [[1, 2, 3, 4]], "y": [[2, 3, 4, 5]]} for _ in range(max(1, num_batches))]


class ZeroTokenSource(FakeSource):
    """覆盖 ZeroTokenSource 相关行为。"""
    def _batch(self):
        self.step += 1
        return {"x": [[1, 2, 3, 4]], "y": [[-100, -100, -100, -100]]}


def make_pure_trainer(config):
    backend = PureBackend(config)
    model = backend.build_model(config, 20)
    optimizer = backend.build_optimizer(model, config)
    source = UnequalTokenSource()
    return Trainer(
        config=config,
        backend=backend,
        model=model,
        optimizer=optimizer,
        tokenizer=source.tokenizer,
        data_source=source,
    )


@unittest.skipIf(torch is None, "torch not installed")
class F11EarlyStopStatusStepTests(unittest.TestCase):
    """覆盖 F11EarlyStopStatusStepTests 相关行为。"""
    def test_status_step_equals_completed_steps_on_early_stop(self) -> None:
        """验证 状态文件、早停、停止 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, max_steps=10, eval_interval=1, patience=2, min_delta=0.0, callbacks="")
            trainer = make_trainer(config)
            losses = iter([2.0, 2.5, 2.6, 2.7, 2.8])
            with mock.patch.object(trainer, "evaluate", side_effect=lambda *a, **k: next(losses)):
                result = trainer.train()
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["steps"], 3)
            self.assertEqual(read_status(trainer)["step"], result["steps"])


@unittest.skipIf(torch is None, "torch not installed")
class F12EvalNonFiniteTests(unittest.TestCase):
    """覆盖 F12EvalNonFiniteTests 相关行为。"""
    def test_partial_nonfinite_eval_warns_and_emits_event(self) -> None:
        """验证 评估、事件 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            results = [
                EvalResult(loss=float("nan"), tokens=16),
                EvalResult(loss=2.0, tokens=16),
                EvalResult(loss=float("inf"), tokens=16),
                EvalResult(loss=2.0, tokens=16),
            ]
            buffer = io.StringIO()
            with mock.patch.object(trainer.backend, "eval_step", side_effect=results):
                with contextlib.redirect_stdout(buffer):
                    value = trainer.evaluate("val", 4)
            self.assertAlmostEqual(value, 2.0)
            self.assertIn("non-finite", buffer.getvalue())
            events = [e for e in read_events(trainer) if e["event"] == "eval_nonfinite_batches"]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["payload"], {"split": "val", "count": 2})

    def test_all_nonfinite_eval_still_raises(self) -> None:
        """验证 评估 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            with mock.patch.object(
                trainer.backend, "eval_step", return_value=EvalResult(loss=float("nan"), tokens=16)
            ):
                with self.assertRaises(RuntimeError):
                    trainer.evaluate("val", 2)


@unittest.skipIf(torch is None, "torch not installed")
class F13ResumeMetadataTests(unittest.TestCase):
    """覆盖 F13ResumeMetadataTests 相关行为。"""
    def test_corrupt_checkpoint_resume_keeps_original_metadata(self) -> None:
        """验证 checkpoint、断点续训 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, max_steps=1)
            trainer = make_trainer(config)
            trainer.train()
            run_dir = trainer.run_dir
            original_config = (run_dir / "run_config.json").read_bytes()
            original_env = (run_dir / "environment.json").read_bytes()
            (run_dir / "last.pt").write_bytes(b"this is not a checkpoint")
            config2 = make_config(tmp, max_steps=5)
            with self.assertRaises(Exception):
                make_trainer(config2, resume=True)
            self.assertEqual((run_dir / "run_config.json").read_bytes(), original_config)
            self.assertEqual((run_dir / "environment.json").read_bytes(), original_env)


@unittest.skipIf(torch is None, "torch not installed")
class F14EarlyStopResumeConfigTests(unittest.TestCase):
    """覆盖 F14EarlyStopResumeConfigTests 相关行为。"""
    def test_resume_training_uses_new_patience(self) -> None:
        """验证 断点续训、训练、耐心值 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, max_steps=2, eval_interval=1, patience=1, min_delta=0.0)
            trainer = make_trainer(config)
            losses = iter([3.5, 2.0, 2.5])
            with mock.patch.object(trainer, "evaluate", side_effect=lambda *a, **k: next(losses)):
                first = trainer.train()
            self.assertTrue(first["stopped_early"])

            config2 = make_config(tmp, max_steps=6, eval_interval=1, patience=3, min_delta=0.0)
            trainer2 = make_trainer(config2, resume=True)
            self.assertEqual(trainer2.early.patience, 3)
            losses2 = iter([2.2, 2.3, 1.9, 2.0, 2.1])
            with mock.patch.object(trainer2, "evaluate", side_effect=lambda *a, **k: next(losses2)):
                second = trainer2.train()
            self.assertFalse(second["stopped_early"])
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["steps"], 6)


@unittest.skipIf(torch is None, "torch not installed")
class F15StatusSingleWriterTests(unittest.TestCase):
    """覆盖 F15StatusSingleWriterTests 相关行为。"""
    def test_status_fields_survive_full_training(self) -> None:
        """验证 状态文件、训练 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            initial = read_status(trainer)
            trainer.train()
            final = read_status(trainer)
            self.assertIn("elapsed_s", final)
            self.assertEqual(final["started_at"], initial["started_at"])
            self.assertEqual(final["pid"], initial["pid"])
            self.assertTrue(final["started_at"])
            self.assertGreater(final["pid"], 0)
            self.assertEqual(final["step"], 3)
            self.assertEqual(final["status"], "completed")
            for key in ("run_id", "stage", "max_steps", "best_val", "checkpoint", "updated_at", "error"):
                self.assertIn(key, final)

    def test_status_complete_without_telemetry(self) -> None:
        """验证 状态文件 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, callbacks="")
            trainer = make_trainer(config)
            trainer.train()
            status = read_status(trainer)
            for key in (
                "run_id", "stage", "status", "step", "max_steps", "best_val", "checkpoint",
                "pid", "started_at", "updated_at", "error", "val_loss", "elapsed_s",
            ):
                self.assertIn(key, status)
            self.assertEqual(status["step"], 3)
            self.assertEqual(status["status"], "completed")
            self.assertTrue(status["started_at"])
            self.assertGreater(status["pid"], 0)
            self.assertTrue(status["checkpoint"])


@unittest.skipIf(torch is None, "torch not installed")
class F16TailMeanGateTests(unittest.TestCase):
    """覆盖 F16TailMeanGateTests 相关行为。"""
    def _noisy_trainer(self, tmp):
        config = make_config(tmp, max_steps=5, eval_interval=100, save_interval=100, log_interval=100)
        trainer = make_trainer(config)
        scripted = [3.0, 2.8, 2.6, 2.4, 5.0]
        real_train_step = trainer.backend.train_step
        calls = {"n": 0}

        def fake_train_step(model, optimizer, batch, **kwargs):
            index = calls["n"]
            calls["n"] += 1
            real_train_step(model, optimizer, batch, **kwargs)
            return StepResult(loss=scripted[index], grad_norm=0.0, tokens=16, stepped=False)

        return trainer, fake_train_step

    def test_noisy_last_batch_still_passes_gate(self) -> None:
        """验证 `test_noisy_last_batch_still_passes_gate` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            trainer, fake_train_step = self._noisy_trainer(tmp)
            with mock.patch.object(trainer, "evaluate", return_value=3.5):
                with mock.patch.object(trainer.backend, "train_step", side_effect=fake_train_step):
                    result = trainer.train()
            tail = [3.0, 2.8, 2.6, 2.4, 5.0]
            self.assertAlmostEqual(result["final_train_loss"], sum(tail) / len(tail))
            report = check_mini_run(result, vocab_size=20, loss_tolerance=1.0, min_drop=0.2)
            self.assertTrue(report["passed"])

    def test_training_report_notes_tail_mean_semantics(self) -> None:
        """验证 训练、报告 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            trainer, fake_train_step = self._noisy_trainer(tmp)
            with mock.patch.object(trainer, "evaluate", return_value=3.5):
                with mock.patch.object(trainer.backend, "train_step", side_effect=fake_train_step):
                    trainer.train()
            report = json.loads((trainer.run_dir / "training_report.json").read_text(encoding="utf-8"))
            self.assertIn("final_train_loss_note", report)
            self.assertIn("均值", report["final_train_loss_note"])


class F17TokenWeightedAccumTests(unittest.TestCase):
    """覆盖 F17TokenWeightedAccumTests 相关行为。"""

    @unittest.skipIf(torch is None, "torch not installed")
    def test_micro_batch_loss_scales_are_token_weighted(self) -> None:
        """验证 loss、token 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, grad_accum_steps=2, max_steps=1, eval_interval=100, save_interval=100)
            trainer = make_trainer(config)
            trainer.data_source = UnequalTokenSource()
            scales: list[float] = []
            real_train_step = trainer.backend.train_step

            def spy(model, optimizer, batch, **kwargs):
                scales.append(kwargs.get("loss_scale"))
                return real_train_step(model, optimizer, batch, **kwargs)

            with mock.patch.object(trainer.backend, "train_step", side_effect=spy):
                trainer.train()
            self.assertEqual(len(scales), 2)
            self.assertAlmostEqual(scales[0], 4 / 16)
            self.assertAlmostEqual(scales[1], 12 / 16)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_torch_gradients_equal_global_token_weighted_mean(self) -> None:
        """验证 token 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, grad_accum_steps=2, max_steps=1, eval_interval=100, save_interval=100)
            trainer = make_trainer(config)
            source = UnequalTokenSource()
            trainer.data_source = source
            batches = [source.train_batch(2, 8), source.train_batch(2, 8)]
            source.step = 0  # trainer consumes the identical window again
            before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
            captured: dict[str, torch.Tensor] = {}
            real_optimizer_step = trainer.backend.optimizer_step

            def spy_optimizer_step(model, optimizer, **kwargs):
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        captured[name] = param.grad.detach().clone()
                return real_optimizer_step(model, optimizer, **kwargs)

            with mock.patch.object(trainer.backend, "optimizer_step", side_effect=spy_optimizer_step):
                trainer.train()
            self.assertTrue(captured, "optimizer_step was never reached")
            with torch.no_grad():
                for name, param in trainer.model.named_parameters():
                    param.copy_(before[name])
            trainer.backend.zero_grad(trainer.optimizer)
            for batch, weight in zip(batches, (4 / 16, 12 / 16)):
                trainer.backend.train_step(
                    trainer.model, trainer.optimizer, batch,
                    grad_clip=config.grad_clip, do_step=False, loss_scale=weight,
                )
            for name, param in trainer.model.named_parameters():
                if param.grad is None:
                    continue
                self.assertTrue(
                    torch.allclose(param.grad, captured[name], atol=1e-6, rtol=1e-5),
                    f"gradient mismatch for {name}",
                )

    def test_pure_gradients_equal_global_token_weighted_mean(self) -> None:
        """验证 纯后端、token 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp, backend_type="pure", grad_accum_steps=2, max_steps=1,
                eval_interval=100, save_interval=100, n_embd=16,
            )
            trainer = make_pure_trainer(config)
            source = trainer.data_source
            batches = [source.train_batch(2, 8), source.train_batch(2, 8)]
            source.step = 0  # trainer consumes the identical window again
            before = {name: list(p.data) for name, p in trainer.model.params.items()}
            captured: dict[str, list[float]] = {}
            real_optimizer_step = trainer.backend.optimizer_step

            def spy_optimizer_step(model, optimizer, **kwargs):
                for name, param in model.params.items():
                    if param.grad is not None:
                        captured[name] = list(param.grad.data)
                return real_optimizer_step(model, optimizer, **kwargs)

            with mock.patch.object(trainer.backend, "optimizer_step", side_effect=spy_optimizer_step):
                trainer.train()
            self.assertTrue(captured, "optimizer_step was never reached")
            for name, param in trainer.model.params.items():
                param.data = list(before[name])
            trainer.backend.zero_grad(trainer.optimizer)
            for batch, weight in zip(batches, (4 / 16, 12 / 16)):
                trainer.backend.train_step(
                    trainer.model, trainer.optimizer, batch,
                    grad_clip=config.grad_clip, do_step=False, loss_scale=weight,
                )
            for name, param in trainer.model.params.items():
                if param.grad is None:
                    continue
                for got, want in zip(param.grad.data, captured[name]):
                    self.assertAlmostEqual(got, want, places=6, msg=f"gradient mismatch for {name}")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_zero_token_window_raises_before_backward(self) -> None:
        """验证 token 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, grad_accum_steps=2, max_steps=1)
            trainer = make_trainer(config)
            trainer.data_source = ZeroTokenSource()
            with mock.patch.object(trainer.backend, "train_step") as step:
                with self.assertRaises(RuntimeError):
                    trainer.train()
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
