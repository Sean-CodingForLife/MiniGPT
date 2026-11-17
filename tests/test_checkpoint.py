"""checkpoint 的落盘/加载、结构校验、恶意 pickle 拒绝与 RNG 状态恢复测试。

断点续训的正确性依赖三点：文件可无损往返、缺失/版本不符的旧文件被明确
拒绝、反序列化不执行任意代码；另外 RNG 状态必须完整捕获，否则续训后
采样与 dropout 会与不中断训练分叉。
"""

from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.training.checkpoint import (
    CHECKPOINT_VERSION,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    """覆盖 checkpoint 序列化、格式守卫、安全加载与随机数流恢复。"""

    def test_roundtrip_and_validation(self) -> None:
        """合法 payload 保存后能原样读回（含张量数值），并通过结构校验。"""
        # 手工拼最小合法结构，避免依赖真实训练产物。
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "backend": "torch",
            "stage": "pretrain",
            "model": {"w": torch.arange(3)},
            "optimizer": {},
            "iter_num": 7,
            "best_val_loss": 1.5,
            "config": {},
            "model_config": {},
            "tokenizer_state": {"chars": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, payload)
            self.assertTrue(path.is_file())
            loaded = load_checkpoint(path)
            validate_checkpoint(loaded)
            self.assertEqual(loaded["iter_num"], 7)
            self.assertTrue(torch.equal(loaded["model"]["w"], torch.arange(3)))

    def test_missing_field_rejected(self) -> None:
        """缺少必需字段的 checkpoint 必须在校验阶段报 ValueError，而非后续 KeyError。"""
        with self.assertRaises(ValueError):
            validate_checkpoint({"format_version": CHECKPOINT_VERSION})

    def test_wrong_version_rejected(self) -> None:
        """格式版本不认识时必须拒绝加载，防止按新结构误读旧数据。"""
        payload = {
            "format_version": 999,
            "backend": "torch",
            "stage": "pretrain",
            "model": {},
            "optimizer": {},
            "iter_num": 1,
            "best_val_loss": 1.0,
            "config": {},
            "model_config": {},
            "tokenizer_state": {},
        }
        with self.assertRaises(ValueError):
            validate_checkpoint(payload)

    def test_malicious_pickle_rejected(self) -> None:
        """含 __reduce__ 副作用的 pickle 必须被拒绝且绝不执行，用临时文件作探针验证。"""
        import pickle

        marker = Path(tempfile.gettempdir()) / "minigpt_pwned_marker.txt"
        if marker.exists():
            marker.unlink()

        class Evil:
            """覆盖 Evil 相关行为。"""
            def __reduce__(self):
                # 若加载端使用不安全的 torch.load，这里会在反序列化时创建探针文件。
                return (marker.write_text, ("pwned",))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.pt"
            torch.save({"payload": Evil()}, path)
            with self.assertRaises(Exception):
                load_checkpoint(path)
            self.assertFalse(marker.exists(), "malicious pickle was executed during load")

    def test_rng_roundtrip(self) -> None:
        """恢复 RNG 状态后 Python random 与 torch 都必须复现出相同的下一批随机数。"""
        random.seed(42)
        torch.manual_seed(42)
        state = capture_rng_state()
        expected_python = random.random()
        expected_torch = torch.rand(3)

        # 故意打乱两条随机流，再恢复，确认恢复覆盖了真实状态而非仅记录种子。
        random.seed(1)
        torch.manual_seed(1)
        restore_rng_state(state)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.allclose(torch.rand(3), expected_torch))


if __name__ == "__main__":
    unittest.main()
