"""B7/N04: 一键断点续训。

覆盖：
* ``--resume`` 保持旧布尔用法（裸开关 / true / false），新增 ``auto``；
* ``Config.resume`` 兼容 bool 与 str（"auto"/"true"/"false"）；
* ``--resume auto`` 在 run 目录自动找 last.pt（pure 后端 last.mgckpt），
  找不到时明确报错；
* ``JobManager.resume_job(run_id)`` 复用 run 目录冻结配置重新拉起训练，
  并向 events.jsonl 追加 ``resumed`` 事件。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pretrain  # noqa: E402
import sft  # noqa: E402
from minigpt.config import Config, parse_config  # noqa: E402
from minigpt.training.job_manager import JobManager, JobNotFound  # noqa: E402
from tests.test_trainer_failures import make_config, make_trainer, torch  # noqa: E402

PYTHON = sys.executable

# 入口脚本的 ensure_hash_seed 在未设置 PYTHONHASHSEED 时会重启整个解释器；
# 测试进程内直接调用 main() 前先设置，避免把 unittest 进程重跑一遍。
os.environ.setdefault("PYTHONHASHSEED", "0")


def _read_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_config(tmp: str, config: Config, tag: str) -> str:
    path = Path(tmp) / f"cfg-{tag}.json"
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8")
    return str(path)


class ResumeParsingTests(unittest.TestCase):
    """--resume 的解析与 Config 兼容。"""

    def test_bare_resume_flag_stays_true(self) -> None:
        """验证 断点续训 相关行为。"""
        config = parse_config("pretrain", ["--resume"])
        self.assertTrue(config.resume)
        self.assertTrue(Config.resume_enabled(config.resume))

    def test_resume_auto_parses_to_auto(self) -> None:
        """验证 断点续训 相关行为。"""
        config = parse_config("pretrain", ["--resume", "auto"])
        self.assertEqual(config.resume, "auto")
        self.assertTrue(Config.resume_enabled(config.resume))

    def test_resume_true_false_strings_parse_to_bool(self) -> None:
        """验证 断点续训 相关行为。"""
        self.assertIs(parse_config("pretrain", ["--resume", "true"]).resume, True)
        self.assertIs(parse_config("pretrain", ["--resume", "false"]).resume, False)
        self.assertFalse(Config.resume_enabled("false"))
        self.assertFalse(Config.resume_enabled(False))

    def test_resume_invalid_value_rejected(self) -> None:
        """验证 断点续训、非法输入 相关行为。"""
        with self.assertRaises(SystemExit):
            parse_config("pretrain", ["--resume", "sometimes"])

    def test_resume_json_bool_and_auto_apply(self) -> None:
        """验证 断点续训 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"resume": "auto"}), encoding="utf-8")
            config = parse_config("pretrain", ["--config", str(path)])
            self.assertEqual(config.resume, "auto")
            path.write_text(json.dumps({"resume": True}), encoding="utf-8")
            config = parse_config("pretrain", ["--config", str(path)])
            self.assertIs(config.resume, True)

    def test_resume_env_auto(self) -> None:
        """验证 断点续训、环境变量 相关行为。"""
        old = os.environ.get("MINIGPT_RESUME")
        os.environ["MINIGPT_RESUME"] = "auto"
        try:
            config = parse_config("pretrain", [])
        finally:
            if old is None:
                os.environ.pop("MINIGPT_RESUME", None)
            else:
                os.environ["MINIGPT_RESUME"] = old
        self.assertEqual(config.resume, "auto")

    def test_validate_rejects_bad_resume_value(self) -> None:
        """验证 校验、断点续训 相关行为。"""
        with self.assertRaises(ValueError):
            Config(resume="sometimes").validate()
        Config(resume="auto").validate()
        Config(resume=True).validate()
        Config(resume=False).validate()


class ResumeAutoTrainerTests(unittest.TestCase):
    """--resume auto 的端到端续训（真实小训练）。"""

    @unittest.skipIf(torch is None, "torch not installed")
    def test_pretrain_resume_auto_continues_from_last_checkpoint(self) -> None:
        """验证 预训练、断点续训、checkpoint 相关行为。"""
        data_dir = str(PROJECT_ROOT / "data" / "shakespeare_en")
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, max_steps=2, run_name="auto-run", data_dir=data_dir)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = pretrain.main(["--config", _write_config(tmp, config, "first")])
            self.assertEqual(code, 0)

            config2 = make_config(tmp, max_steps=4, run_name="auto-run", resume="auto", data_dir=data_dir)
            config_path = _write_config(tmp, config2, "resume")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = pretrain.main(["--config", config_path, "--resume", "auto"])
            self.assertEqual(code, 0)
            from minigpt.training.checkpoint import load_checkpoint

            payload = load_checkpoint(Path(tmp) / "runs" / "auto-run" / "last.pt")
            self.assertEqual(payload["iter_num"], 4)
            events = _read_events(Path(tmp) / "runs" / "auto-run")
            self.assertTrue(any(e["event"] == "run_resumed" for e in events))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_pretrain_resume_auto_without_checkpoint_errors(self) -> None:
        """验证 预训练、断点续训、checkpoint 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "runs" / "empty-run").mkdir(parents=True)
            config = make_config(
                tmp,
                run_name="empty-run",
                resume="auto",
                data_dir=str(PROJECT_ROOT / "data" / "shakespeare_en"),
            )
            config_path = _write_config(tmp, config, "empty")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = pretrain.main(["--config", config_path, "--resume", "auto"])
            self.assertEqual(code, 2)
            self.assertIn("last.pt", err.getvalue())

    def test_sft_resume_auto_requires_run_name(self) -> None:
        """验证 SFT、断点续训 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, stage="sft", run_name="", resume="auto")
            config_path = _write_config(tmp, config, "sft")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = sft.main(["--config", config_path, "--resume", "auto"])
            self.assertEqual(code, 2)

    def test_pure_resume_auto_finds_mgckpt(self) -> None:
        """验证 纯后端、断点续训、mgckpt 格式 相关行为。"""
        from minigpt.pure.backend import PureBackend
        from minigpt.training.trainer import Trainer
        from tests.test_trainer_failures import FakeSource

        def pure_config(tmp: str, **overrides) -> Config:
            return make_config(
                tmp,
                backend_type="pure",
                device="cpu",
                dtype="float32",
                n_layer=1,
                n_head=2,
                n_embd=8,
                block_size=8,
                batch_size=2,
                warmup_steps=0,
                eval_interval=1,
                eval_iters=1,
                save_interval=1,
                run_name="pure-auto",
                num_threads=1,
                **overrides,
            )

        def build(config: Config, resume) -> Trainer:
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

        with tempfile.TemporaryDirectory() as tmp:
            trainer = build(pure_config(tmp, max_steps=1), resume=False)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train()
            self.assertTrue((trainer.run_dir / "last.mgckpt").is_file())

            config2 = pure_config(tmp, max_steps=2, resume="auto")
            trainer2 = build(config2, resume="auto")
            self.assertEqual(trainer2.start_iter, 1)
            with contextlib.redirect_stdout(io.StringIO()):
                result = trainer2.train()
            self.assertEqual(result["steps"], 2)

    def test_pure_resume_auto_without_checkpoint_errors(self) -> None:
        """验证 纯后端、断点续训、checkpoint 相关行为。"""
        from minigpt.pure.backend import PureBackend
        from minigpt.training.trainer import Trainer
        from tests.test_trainer_failures import FakeSource

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp,
                backend_type="pure",
                device="cpu",
                dtype="float32",
                n_layer=1,
                n_head=2,
                n_embd=8,
                block_size=8,
                batch_size=2,
                max_steps=1,
                run_name="pure-empty",
                num_threads=1,
                resume="auto",
            )
            backend = PureBackend(config)
            model = backend.build_model(config, 20)
            optimizer = backend.build_optimizer(model, config)
            source = FakeSource()
            with self.assertRaises(FileNotFoundError) as ctx:
                Trainer(
                    config=config,
                    backend=backend,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=source.tokenizer,
                    data_source=source,
                    resume="auto",
                )
            self.assertIn("last.mgckpt", str(ctx.exception))


class ResumeJobManagerTests(unittest.TestCase):
    """JobManager.resume_job：复用冻结配置 + resumed 事件。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b7-resume-")
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()
        self.runs_dir = self.project / "runs"
        self.captured: list[dict] = []

        def builder(config, project_root, config_file):
            self.captured.append(dict(config))
            return [PYTHON, "-u", "-c", "print('dummy-done', flush=True)"]

        self.manager = JobManager(
            project_root=self.project,
            runs_dir=self.runs_dir,
            command_builder=builder,
            poll_interval=0.05,
        )

    def tearDown(self):
        self.manager.shutdown()
        self._tmp.cleanup()

    def test_resume_job_reuses_frozen_config(self) -> None:
        """验证 断点续训、任务、配置 相关行为。"""
        record = self.manager.start_job(preset="debug_en", run_name="resume-me")
        final = self.manager.wait_for(record.run_id, timeout=20)
        self.assertEqual(final.status, "completed")
        frozen_path = Path(record.run_dir) / "resolved_config.json"
        frozen_before = frozen_path.read_bytes()
        (Path(record.run_dir) / "last.pt").write_bytes(b"fake-checkpoint")

        resumed = self.manager.resume_job(record.run_id)
        try:
            self.assertEqual(resumed.run_id, record.run_id)
            self.assertEqual(resumed.status, "running")
            self.assertIn("--resume", resumed.command)
            self.assertIn("auto", resumed.command)
            # 冻结配置被复用：除 resume 字段外与磁盘原配置一致。
            self.assertEqual(len(self.captured), 2)
            original = json.loads(frozen_before.decode("utf-8"))
            reused = self.captured[-1]
            for key, value in original.items():
                if key == "resume":
                    continue
                self.assertEqual(reused[key], value, key)
            self.assertTrue(Config.resume_enabled(reused["resume"]))
            self.assertEqual(frozen_path.read_bytes(), frozen_before)
            events = _read_events(record.run_dir)
            resumed_events = [e for e in events if e["event"] == "resumed"]
            self.assertEqual(len(resumed_events), 1)
            self.assertEqual(resumed_events[0]["run_id"], record.run_id)
            self.assertEqual(resumed_events[0]["payload"]["source"], "job_manager")
        finally:
            self.manager.stop_job(record.run_id)

    def test_resume_job_missing_run_raises(self) -> None:
        """验证 断点续训、任务 相关行为。"""
        with self.assertRaises(JobNotFound):
            self.manager.resume_job("no-such-run")

    def test_resume_job_without_checkpoint_raises(self) -> None:
        """验证 断点续训、任务、checkpoint 相关行为。"""
        record = self.manager.start_job(preset="debug_en", run_name="no-ckpt")
        self.manager.wait_for(record.run_id, timeout=20)
        with self.assertRaises(Exception) as ctx:
            self.manager.resume_job(record.run_id)
        self.assertIn("last.pt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()