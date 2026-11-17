"""B7/N05: 训练可观测性——events.jsonl 结构化阶段事件。

事件 schema（v=1，见 minigpt/training/callbacks.py 模块 docstring）：

* ``eval_start``  payload: {split, num_batches}
* ``eval_end``    payload: {split, duration_s, val_loss}（val_loss 可为 null）
* ``checkpoint_saved`` payload: {kind, path}（旧 ``checkpoint`` 事件保留）
* ``early_stop_triggered`` payload: {best_val, patience, step}
* ``run_resumed`` payload: {checkpoint, start_step, mode}（N04 已实现）
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_trainer_failures import make_config, make_trainer, torch  # noqa: E402


def read_events(run_dir: Path) -> list[dict]:
    """读取 events.jsonl 并忽略空行；文件不存在时返回空列表便于断言「没写事件」。"""
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_types(events: list[dict]) -> list[str]:
    """抽出事件名序列，用于顺序断言。"""
    return [e["event"] for e in events]


@unittest.skipIf(torch is None, "torch not installed")
class StructuredPhaseEventTests(unittest.TestCase):
    """验证训练各阶段写出符合 schema v1 的结构化事件，且新旧事件并存。"""

    def test_short_training_emits_phase_events(self) -> None:
        """一次 4 步训练需产出配对的 eval_start/eval_end、checkpoint_saved 与 run 边界事件。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, max_steps=4, eval_interval=2, save_interval=2, callbacks="telemetry")
            trainer = make_trainer(config)
            with contextlib.redirect_stdout(io.StringIO()):
                result = trainer.train()
            self.assertEqual(result["status"], "completed")
            events = read_events(trainer.run_dir)
            types = event_types(events)

            for required in ("eval_start", "eval_end", "checkpoint_saved", "run_start", "run_end"):
                self.assertIn(required, types, required)

            # eval_start/eval_end 成对且含 split 与耗时。
            starts = [e for e in events if e["event"] == "eval_start"]
            ends = [e for e in events if e["event"] == "eval_end"]
            self.assertGreaterEqual(len(starts), 2)  # train sanity + val
            self.assertEqual(len(starts), len(ends))
            self.assertEqual({e["payload"]["split"] for e in starts}, {"train", "val"})
            for end in ends:
                self.assertIn("duration_s", end["payload"])
                self.assertGreaterEqual(end["payload"]["duration_s"], 0.0)
                self.assertIn("split", end["payload"])
            val_ends = [e for e in ends if e["payload"]["split"] == "val"]
            self.assertTrue(all(e["payload"].get("val_loss") is not None for e in val_ends))

            # checkpoint_saved 含 kind/path；旧 checkpoint 事件保留。
            saved = [e for e in events if e["event"] == "checkpoint_saved"]
            self.assertTrue(saved)
            self.assertEqual({e["payload"]["kind"] for e in saved}, {"best", "last"})
            for e in saved:
                self.assertTrue(Path(e["payload"]["path"]).is_file())
            self.assertIn("checkpoint", types)

            # 顺序：先 eval_start 后 eval_end；先 checkpoint_saved 后 run_end。
            self.assertLess(types.index("eval_start"), types.index("eval_end"))
            self.assertLess(types.index("checkpoint_saved"), types.index("run_end"))

    def test_early_stop_emits_triggered_event(self) -> None:
        """连续无改进触发早停时，事件需只发一次并带上 patience/best_val 与真实步数。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp, max_steps=10, eval_interval=1, patience=2, min_delta=0.0, callbacks="telemetry"
            )
            trainer = make_trainer(config)
            # 固定递减失败的验证损失序列，把早停触发时机变成确定的。
            losses = iter([2.0, 2.5, 2.6, 2.7, 2.8])
            with mock.patch.object(trainer, "evaluate", side_effect=lambda *a, **k: next(losses)):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = trainer.train()
            self.assertTrue(result["stopped_early"])
            events = read_events(trainer.run_dir)
            triggered = [e for e in events if e["event"] == "early_stop_triggered"]
            self.assertEqual(len(triggered), 1)
            payload = triggered[0]["payload"]
            self.assertEqual(payload["patience"], 2)
            self.assertIn("best_val", payload)
            self.assertEqual(triggered[0]["step"], result["steps"])
            # 旧 early_stop 事件保留。
            self.assertIn("early_stop", event_types(events))

    def test_evaluate_direct_call_emits_eval_pair(self) -> None:
        """绕过训练循环直接调用 evaluate 时也必须成对写出 eval_start/eval_end。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_trainer(config)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.evaluate("val", 1)
            events = read_events(trainer.run_dir)
            types = event_types(events)
            self.assertIn("eval_start", types)
            self.assertIn("eval_end", types)
            self.assertLess(types.index("eval_start"), types.index("eval_end"))


if __name__ == "__main__":
    unittest.main()