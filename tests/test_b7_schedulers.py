"""B7/N06: 学习率调度器扩展（cosine / wsd / linear）。

关键点数值断言：warmup 起点/终点、stable 段、衰减终点；
以及 trainer 按 config.lr_schedule 选择调度函数。
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.config import Config, parse_config  # noqa: E402
from minigpt.training.trainer import cosine_lr, linear_lr, lr_for_step, wsd_lr  # noqa: E402
from tests.test_trainer_failures import make_config, make_trainer, torch  # noqa: E402

LR = 1e-3
MIN_LR = 1e-4


class ScheduleDefaultsTests(unittest.TestCase):
    """覆盖 ScheduleDefaultsTests 相关行为。"""
    def test_defaults_keep_cosine(self) -> None:
        """验证 `test_defaults_keep_cosine` 对应的行为约定。"""
        config = Config()
        self.assertEqual(config.lr_schedule, "cosine")
        self.assertAlmostEqual(config.decay_fraction, 0.2)
        config.validate()

    def test_cli_schedule_fields(self) -> None:
        """验证 命令行、学习率调度 相关行为。"""
        config = parse_config("pretrain", ["--lr_schedule", "wsd", "--decay_fraction", "0.3"])
        self.assertEqual(config.lr_schedule, "wsd")
        self.assertAlmostEqual(config.decay_fraction, 0.3)

    def test_invalid_schedule_rejected(self) -> None:
        """验证 非法输入、学习率调度 相关行为。"""
        with self.assertRaises(ValueError):
            Config(lr_schedule="polynomial").validate()
        for fraction in (-0.1, 1.5):
            with self.assertRaises(ValueError, msg=str(fraction)):
                Config(decay_fraction=fraction).validate()


class CosineKeyPointsTests(unittest.TestCase):
    """覆盖 CosineKeyPointsTests 相关行为。"""
    def test_warmup_and_end(self) -> None:
        """验证 预热 相关行为。"""
        self.assertAlmostEqual(cosine_lr(0, 100, 10, LR, MIN_LR), LR / 10)
        self.assertAlmostEqual(cosine_lr(9, 100, 10, LR, MIN_LR), LR)
        self.assertAlmostEqual(cosine_lr(100, 100, 10, LR, MIN_LR), MIN_LR)
        self.assertAlmostEqual(cosine_lr(150, 100, 10, LR, MIN_LR), MIN_LR)


class WsdKeyPointsTests(unittest.TestCase):
    """warmup-stable-decay：max_steps=100, warmup=10, decay_fraction=0.2。

    decay_start = warmup + (1 - 0.2) * (max_steps - warmup) = 10 + 72 = 82。
    """

    def test_warmup_start_and_end(self) -> None:
        """验证 预热、启动 相关行为。"""
        self.assertAlmostEqual(wsd_lr(0, 100, 10, LR, MIN_LR), LR / 10)
        self.assertAlmostEqual(wsd_lr(9, 100, 10, LR, MIN_LR), LR)

    def test_stable_segment_is_constant(self) -> None:
        """验证 `test_stable_segment_is_constant` 对应的行为约定。"""
        for step in (10, 40, 81):
            self.assertAlmostEqual(wsd_lr(step, 100, 10, LR, MIN_LR), LR, msg=f"step={step}")

    def test_decay_midpoint_and_end(self) -> None:
        """验证 `test_decay_midpoint_and_end` 对应的行为约定。"""
        self.assertAlmostEqual(wsd_lr(91, 100, 10, LR, MIN_LR), (LR + MIN_LR) / 2)
        self.assertAlmostEqual(wsd_lr(100, 100, 10, LR, MIN_LR), MIN_LR)
        self.assertAlmostEqual(wsd_lr(150, 100, 10, LR, MIN_LR), MIN_LR)

    def test_decay_fraction_configurable(self) -> None:
        # decay_fraction=0.5 -> decay_start = 10 + 45 = 55
        """验证 `test_decay_fraction_configurable` 对应的行为约定。"""
        self.assertAlmostEqual(wsd_lr(54, 100, 10, LR, MIN_LR, decay_fraction=0.5), LR)
        self.assertAlmostEqual(wsd_lr(55, 100, 10, LR, MIN_LR, decay_fraction=0.5), LR)
        self.assertAlmostEqual(wsd_lr(100, 100, 10, LR, MIN_LR, decay_fraction=0.5), MIN_LR)
        # decay_fraction=0 -> 全程 stable，无衰减段
        self.assertAlmostEqual(wsd_lr(99, 100, 10, LR, MIN_LR, decay_fraction=0.0), LR)


class LinearKeyPointsTests(unittest.TestCase):
    """覆盖 LinearKeyPointsTests 相关行为。"""
    def test_warmup_start_and_end(self) -> None:
        """验证 预热、启动 相关行为。"""
        self.assertAlmostEqual(linear_lr(0, 100, 10, LR, MIN_LR), LR / 10)
        self.assertAlmostEqual(linear_lr(9, 100, 10, LR, MIN_LR), LR)

    def test_linear_midpoint_and_end(self) -> None:
        """验证 `test_linear_midpoint_and_end` 对应的行为约定。"""
        self.assertAlmostEqual(linear_lr(55, 100, 10, LR, MIN_LR), (LR + MIN_LR) / 2)
        self.assertAlmostEqual(linear_lr(100, 100, 10, LR, MIN_LR), MIN_LR)
        self.assertAlmostEqual(linear_lr(150, 100, 10, LR, MIN_LR), MIN_LR)


class DispatcherTests(unittest.TestCase):
    """覆盖 DispatcherTests 相关行为。"""
    def test_lr_for_step_dispatch(self) -> None:
        """验证 `test_lr_for_step_dispatch` 对应的行为约定。"""
        config = Config(lr_schedule="cosine", warmup_steps=10, max_steps=100, lr=LR, min_lr=MIN_LR)
        self.assertAlmostEqual(lr_for_step(config, 55), cosine_lr(55, 100, 10, LR, MIN_LR))
        config.lr_schedule = "wsd"
        self.assertAlmostEqual(lr_for_step(config, 55), LR)
        config.lr_schedule = "linear"
        self.assertAlmostEqual(lr_for_step(config, 55), (LR + MIN_LR) / 2)

    def test_lr_for_step_unknown_schedule_raises(self) -> None:
        """验证 学习率调度 相关行为。"""
        config = Config(lr_schedule="nope")
        with self.assertRaises(ValueError):
            lr_for_step(config, 0)


@unittest.skipIf(torch is None, "torch not installed")
class TrainerUsesConfiguredScheduleTests(unittest.TestCase):
    """覆盖 TrainerUsesConfiguredScheduleTests 相关行为。"""
    def test_trainer_wsd_keeps_lr_constant_in_stable_phase(self) -> None:
        """验证 Trainer 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp,
                max_steps=11,
                warmup_steps=1,
                eval_interval=100,
                save_interval=100,
                lr_schedule="wsd",
                decay_fraction=0.2,
                callbacks="",
            )
            trainer = make_trainer(config)
            seen: list[float] = []
            original = trainer._set_lr

            def spy(lr: float) -> None:
                seen.append(lr)
                original(lr)

            trainer._set_lr = spy
            with contextlib.redirect_stdout(io.StringIO()):
                result = trainer.train()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(seen), 11)
            # decay_start = 1 + 0.8 * 10 = 9：step 0 warmup 终点，1..8 stable，
            # 9..10 线性衰减。
            self.assertAlmostEqual(seen[0], config.lr)
            for lr in seen[1:9]:
                self.assertAlmostEqual(lr, config.lr)
            # step 9 是衰减段起点（ratio=0，仍为峰值），step 10 已下降。
            self.assertAlmostEqual(seen[9], wsd_lr(9, 11, 1, config.lr, config.min_lr, 0.2))
            self.assertAlmostEqual(seen[9], config.lr)
            self.assertAlmostEqual(seen[10], wsd_lr(10, 11, 1, config.lr, config.min_lr, 0.2))
            self.assertLess(seen[10], seen[9])

    def test_trainer_linear_matches_formula(self) -> None:
        """验证 Trainer 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp,
                max_steps=3,
                warmup_steps=1,
                eval_interval=100,
                save_interval=100,
                lr_schedule="linear",
                callbacks="",
            )
            trainer = make_trainer(config)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train()
            expected = linear_lr(2, 3, 1, config.lr, config.min_lr)
            self.assertAlmostEqual(trainer.state.lr, expected)


if __name__ == "__main__":
    unittest.main()