"""早停（EarlyStopping）的耐心计数与断点续训语义测试。

守住两件事：连续 N 次无改进后才触发停止；从 checkpoint 恢复时只恢复
「最优指标/步数/坏评估数」等训练状态，当前配置文件里的 patience/min_delta
仍以本次运行为准（对应 CONSTRUCTION.md F14）。
"""

from __future__ import annotations

import unittest

from minigpt.training.early_stop import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    """验证早停的触发时机、状态序列化以及恢复后的配置优先级。"""

    def test_patience_and_state(self) -> None:
        """连续无改进次数达到 patience 才置位，state_dict 往返需完整保留最优值与坏评估计数。"""
        stopper = EarlyStopping(patience=2, min_delta=0.0)
        self.assertTrue(stopper.update(3.0, 1))
        self.assertFalse(stopper.update(3.1, 2))
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update(3.2, 3))
        self.assertTrue(stopper.should_stop)
        state = stopper.state_dict()
        restored = EarlyStopping(patience=99)
        restored.load_state_dict(state)
        self.assertEqual(restored.best, 3.0)
        self.assertEqual(restored.best_step, 1)
        self.assertEqual(restored.num_bad_evals, 2)

    def test_restore_keeps_current_patience_and_min_delta(self) -> None:
        """恢复 checkpoint 时不得用旧 patience/min_delta 覆盖当前配置，且新阈值立即生效。"""
        old = EarlyStopping(patience=1, min_delta=0.5)
        old.update(2.0, 1)
        old.update(2.1, 2)
        self.assertTrue(old.should_stop)
        state = old.state_dict()
        resumed = EarlyStopping(patience=5, min_delta=0.01)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.patience, 5)
        self.assertEqual(resumed.min_delta, 0.01)
        self.assertEqual(resumed.best, 2.0)
        self.assertEqual(resumed.best_step, 1)
        self.assertEqual(resumed.num_bad_evals, 1)
        self.assertFalse(resumed.should_stop)
        # 恢复后改用新的 min_delta 判定改进，1.98 相对 2.0 的下降已足够。
        self.assertTrue(resumed.update(1.98, 3))

    def test_improvement_resets(self) -> None:
        """出现足够幅度的改进后坏评估计数必须清零，避免历史耐心累积导致误停。"""
        stopper = EarlyStopping(patience=2)
        stopper.update(2.0, 1)
        stopper.update(2.1, 2)
        self.assertTrue(stopper.update(1.9, 3))
        self.assertEqual(stopper.num_bad_evals, 0)
        self.assertFalse(stopper.should_stop)


if __name__ == "__main__":
    unittest.main()
