"""生命周期契约回归：草稿目录可启动、真实产物拒覆盖、磁盘终态补写。

背景：看板 create_run 会预写 resolved_config.json / status.json / console.log，
旧版 Trainer「目录非空即拒跑」会把看板自己启动的 run 拒掉；进程硬 kill 后
磁盘 status.json 也会滞留 running。本文件固化修复后的契约。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from minigpt.config import Config
from minigpt.pure.backend import PureBackend
from minigpt.training.job_manager import JobManager, JobStartError

PYTHON = sys.executable


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
        run_name="lifecycle-run",
        backend_type="pure",
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        dropout=0.0,
        batch_size=2,
        max_steps=2,
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


def make_pure_trainer(config, resume=False):
    # 延迟导入：trainer 经 checkpoint 模块依赖 torch，无 torch 环境只跳过用例。
    from minigpt.training.trainer import Trainer

    backend = PureBackend(config)
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


def write_draft(run_dir: Path) -> None:
    """模拟看板 create_run 的预写启动文件（草稿）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"status": "created", "run_id": run_dir.name, "step": 0}), encoding="utf-8"
    )
    (run_dir / "console.log").touch()


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # 原子写替换瞬间文件可能短暂缺失：返回空状态继续轮询。
        return {}


class DraftStartContractTests(unittest.TestCase):
    """Trainer 保护只认真实训练产物，草稿启动文件放行。"""

    def test_draft_dir_starts_and_completes(self):
        """验证 `test_draft_dir_starts_and_completes` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            write_draft(Path(config.runs_dir) / config.run_name)
            trainer = make_pure_trainer(config)
            result = trainer.train()
            self.assertEqual(result["status"], "completed")
            self.assertTrue((trainer.run_dir / "metrics.csv").is_file())

    def test_real_artifacts_refuse_overwrite(self):
        """验证 `test_real_artifacts_refuse_overwrite` 对应的行为约定。"""
        for artifact in ("metrics.csv", "last.mgckpt", "training_report.json", "run_config.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp:
                config = make_config(tmp)
                run_dir = Path(config.runs_dir) / config.run_name
                write_draft(run_dir)
                (run_dir / artifact).write_text("x", encoding="utf-8")
                with self.assertRaises(ValueError):
                    make_pure_trainer(config)

    def test_resume_still_allowed_with_artifacts(self):
        """验证 断点续训 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            trainer = make_pure_trainer(config)
            trainer.train()
            config2 = make_config(tmp, max_steps=4)
            trainer2 = make_pure_trainer(config2, resume=True)
            result = trainer2.train()
            self.assertEqual(result["status"], "completed")


class DiskTerminalStatusTests(unittest.TestCase):
    """进程退出后磁盘 status.json 必须落终态（含硬 kill 与启动失败）。"""

    def _manager(self, tmp, command):
        return JobManager(
            project_root=tmp,
            runs_dir=Path(tmp) / "runs",
            command_builder=lambda config, root, cfg: list(command),
            poll_interval=0.05,
        )

    def test_completed_writes_disk_terminal(self):
        """验证 `test_completed_writes_disk_terminal` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp, [PYTHON, "-u", "-c", "print('ok', flush=True)"])
            try:
                record = manager.start_job(config={"stage": "pretrain"}, run_name="disk-ok")
                run_dir = Path(record.run_dir)
                self.assertTrue(wait_until(lambda: read_status(run_dir).get("status") == "completed"))
                self.assertEqual(manager.get_job(record.run_id).status, "completed")
            finally:
                manager.shutdown()

    def test_hard_kill_writes_disk_terminal(self):
        """验证 `test_hard_kill_writes_disk_terminal` 对应的行为约定。"""
        with tempfile.TemporaryDirectory() as tmp:
            command = [PYTHON, "-u", "-c", "import time; print('up', flush=True); time.sleep(60)"]
            manager = self._manager(tmp, command)
            try:
                record = manager.start_job(config={"stage": "pretrain"}, run_name="disk-kill")
                run_dir = Path(record.run_dir)
                self.assertTrue(wait_until(lambda: manager.get_job(record.run_id).status == "running"))
                manager._processes[record.run_id].kill()
                self.assertTrue(wait_until(lambda: read_status(run_dir).get("status") == "failed"))
                self.assertIn("退出码", read_status(run_dir).get("error", ""))
            finally:
                manager.shutdown()

    def test_spawn_failure_writes_disk_failed(self):
        """验证 失败路径 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "no-such-python.exe")
            manager = self._manager(tmp, [missing, "-c", "pass"])
            try:
                with self.assertRaises(JobStartError):
                    manager.start_job(config={"stage": "pretrain"}, run_name="disk-spawn")
                run_dir = Path(tmp) / "runs" / "disk-spawn"
                status = read_status(run_dir)
                self.assertEqual(status["status"], "failed")
                self.assertIn("无法启动训练进程", status["error"])
            finally:
                manager.shutdown()


class HeartbeatTests(unittest.TestCase):
    """慢模型（单步 >60s 或 log_interval 很大）心跳仍每步刷新。"""

    def _state(self, step):
        from types import SimpleNamespace

        return SimpleNamespace(
            step=step, lr=1e-3, train_loss=2.0, grad_norm=1.0,
            tokens_per_sec=10.0, elapsed=1.0,
        )

    def test_slow_steps_keep_heartbeat_fresh(self):
        """验证 `test_slow_steps_keep_heartbeat_fresh` 对应的行为约定。"""
        from minigpt.training.callbacks import TelemetryCallback

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, log_interval=100, callbacks="telemetry")
            run_dir = Path(tmp) / "runs" / config.run_name
            run_dir.mkdir(parents=True)
            cb = TelemetryCallback(config, run_dir)
            cb.update_status(status="running", step=0)  # 训练开始的全量写盘
            cb._last_heartbeat = time.monotonic() - 2  # 模拟首步耗时 >1s
            cb.on_step_end(self._state(1))
            first = read_status(run_dir)
            time.sleep(1.1)
            cb.on_step_end(self._state(2))
            second = read_status(run_dir)
            self.assertEqual(second["step"], 2)  # 非 log_interval 步也刷新
            self.assertNotEqual(first["updated_at"], second["updated_at"])
            events = (run_dir / "events.jsonl")
            lines = events.read_text(encoding="utf-8").splitlines() if events.is_file() else []
            self.assertFalse(any('"event": "step"' in line for line in lines))  # 日志仍批量

    def test_heartbeat_throttled_within_one_second(self):
        """验证 `test_heartbeat_throttled_within_one_second` 对应的行为约定。"""
        from minigpt.training.callbacks import TelemetryCallback

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, log_interval=100, callbacks="telemetry")
            run_dir = Path(tmp) / "runs" / config.run_name
            run_dir.mkdir(parents=True)
            cb = TelemetryCallback(config, run_dir)
            cb.update_status(status="running", step=0)
            cb._last_heartbeat = time.monotonic() - 2
            cb.on_step_end(self._state(1))
            cb.on_step_end(self._state(2))  # 同一秒内：限流跳过
            self.assertEqual(read_status(run_dir)["step"], 1)

    def test_trainer_without_telemetry_heartbeats(self):
        """验证 Trainer 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, callbacks="")
            trainer = make_pure_trainer(config)
            trainer._heartbeat(3)
            self.assertEqual(read_status(trainer.run_dir)["step"], 3)
            time.sleep(1.1)
            trainer._heartbeat(4)
            self.assertEqual(read_status(trainer.run_dir)["step"], 4)


if __name__ == "__main__":
    unittest.main()
