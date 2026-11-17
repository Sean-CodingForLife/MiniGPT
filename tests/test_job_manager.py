"""``minigpt.training.job_manager`` 的单元测试（纯标准库、临时目录、注入假命令）。

覆盖：

* 配置合并/校验（预设、覆盖、未知字段、非法取值、run_name 路径安全）；
* ``configs/`` 读写（安全名字、校验、原子写、删除）；
* 默认命令构造（固定脚本 + ``--config``，可指定训练解释器）；
* 任务启动/停止/完成/失败/并发上限/服务退出清理，全部用 ``sys.executable -c``
  假进程，不启动真实 PyTorch 训练；
* 在 ``torch`` 被禁用的子进程里导入看板后端，证明看板不依赖 PyTorch。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.config import Config  # noqa: E402
from minigpt.training.job_manager import (  # noqa: E402
    CONFIG_NAME_PATTERN,
    RUN_ID_PATTERN,
    ConfigError,
    ConfigStore,
    JobLimitExceeded,
    JobManager,
    JobNotFound,
    JobStartError,
    default_command_builder,
    normalize_config,
)

PYTHON = sys.executable

LONG_COMMAND = [
    PYTHON,
    "-u",
    "-c",
    "import time; print('dummy-started', flush=True); time.sleep(60)",
]
QUICK_COMMAND = [PYTHON, "-u", "-c", "print('dummy-done', flush=True)"]
FAILING_COMMAND = [
    PYTHON,
    "-u",
    "-c",
    "import sys; print('boom-stderr', file=sys.stderr, flush=True); sys.exit(3)",
]


def long_builder(config, project_root, config_file):
    """注入的命令构造器：长睡假任务，用于测启动/停止。"""

    return list(LONG_COMMAND)


def quick_builder(config, project_root, config_file):
    """注入的命令构造器：立即成功退出。"""

    return list(QUICK_COMMAND)


def failing_builder(config, project_root, config_file):
    """注入的命令构造器：非零退出码。"""

    return list(FAILING_COMMAND)


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """轮询等待条件成立。"""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class TestNormalizeConfig(unittest.TestCase):
    """配置合并与校验。"""

    def test_preset_defaults_resolve_paths(self):
        """验证 预设 相关行为。"""
        config = normalize_config(preset="debug_en", project_root=PROJECT_ROOT)
        self.assertEqual(config.stage, "pretrain")
        self.assertEqual(config.dataset, "shakespeare_en")
        self.assertTrue(Path(config.runs_dir).is_absolute())
        self.assertEqual(Path(config.runs_dir), (PROJECT_ROOT / "runs").resolve())
        self.assertEqual(Path(config.data_dir), (PROJECT_ROOT / "data/shakespeare_en").resolve())

    def test_stage_sft_uses_sft_defaults(self):
        """验证 阶段页、SFT 相关行为。"""
        config = normalize_config({"stage": "sft"}, project_root=PROJECT_ROOT)
        self.assertEqual(config.stage, "sft")
        self.assertEqual(Path(config.data_dir).name, "sft_demo_en")

    def test_overrides_win(self):
        """验证 `test_overrides_win` 对应的行为约定。"""
        overrides = {"n_layer": 2, "n_head": 2, "n_embd": 64, "max_steps": 7}
        config = normalize_config(overrides, preset="debug_en", project_root=PROJECT_ROOT)
        self.assertEqual((config.n_layer, config.n_head, config.n_embd), (2, 2, 64))
        self.assertEqual(config.max_steps, 7)

    def test_unknown_field_rejected(self):
        """验证 `test_unknown_field_rejected` 对应的行为约定。"""
        with self.assertRaises(ConfigError):
            normalize_config({"unknown_field": 1}, preset="debug_en", project_root=PROJECT_ROOT)

    def test_invalid_values_rejected(self):
        """验证 非法输入 相关行为。"""
        with self.assertRaises(ConfigError):
            normalize_config({"n_embd": 7, "n_head": 4}, preset="debug_en", project_root=PROJECT_ROOT)
        with self.assertRaises(ConfigError):
            normalize_config({"lr": -1.0}, preset="debug_en", project_root=PROJECT_ROOT)

    def test_unknown_preset_rejected(self):
        """验证 预设 相关行为。"""
        with self.assertRaises(ConfigError):
            normalize_config(preset="does-not-exist", project_root=PROJECT_ROOT)

    def test_run_name_traversal_rejected(self):
        """验证 目录穿越 相关行为。"""
        with self.assertRaises(ConfigError):
            normalize_config(preset="debug_en", run_name="../evil", project_root=PROJECT_ROOT)
        with self.assertRaises(ConfigError):
            normalize_config(preset="debug_en", run_name="a/b", project_root=PROJECT_ROOT)

    def test_run_name_accepted(self):
        """验证 `test_run_name_accepted` 对应的行为约定。"""
        config = normalize_config(preset="debug_en", run_name="my-run.1", project_root=PROJECT_ROOT)
        self.assertEqual(config.run_name, "my-run.1")

    def test_project_root_is_forced(self):
        """验证 项目 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = normalize_config(
                {"project_root": "C:/somewhere/else"},
                preset="debug_en",
                project_root=root,
            )
            self.assertEqual(Path(config.project_root), root)
            self.assertEqual(Path(config.runs_dir), root / "runs")


class TestConfigStore(unittest.TestCase):
    """configs/ 目录读写。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-configs-")
        self.configs_dir = Path(self._tmp.name) / "configs"
        self.store = ConfigStore(self.configs_dir, project_root=PROJECT_ROOT)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_get_list_delete(self):
        """验证 删除 相关行为。"""
        saved = self.store.save("demo", {"stage": "pretrain", "dataset": "shakespeare_en"})
        self.assertEqual(saved["name"], "demo")
        path = self.configs_dir / "demo.json"
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["stage"], "pretrain")
        listed = self.store.list()
        self.assertEqual([item["name"] for item in listed], ["demo"])
        self.assertEqual(self.store.get("demo")["config"]["dataset"], "shakespeare_en")
        self.assertTrue(self.store.exists("demo"))
        deleted = self.store.delete("demo")
        self.assertTrue(deleted["deleted"])
        self.assertFalse(self.store.exists("demo"))
        with self.assertRaises(JobNotFound):
            self.store.get("demo")
        with self.assertRaises(JobNotFound):
            self.store.delete("demo")

    def test_invalid_config_not_saved(self):
        """验证 非法输入、配置 相关行为。"""
        with self.assertRaises(ConfigError):
            self.store.save("bad", {"n_embd": 7, "n_head": 4})
        self.assertFalse((self.configs_dir / "bad.json").exists())

    def test_invalid_names_rejected(self):
        """验证 非法输入 相关行为。"""
        for name in ["../evil", "a/b", "", ".hidden", "a b", "-leading"]:
            with self.assertRaises(ConfigError, msg=f"name={name!r} should be rejected"):
                self.store.save(name, {"stage": "pretrain"})
        for path in self.configs_dir.glob("*.json"):
            self.assertFalse(path.name.startswith("."))

    def test_json_suffix_accepted(self):
        """验证 `test_json_suffix_accepted` 对应的行为约定。"""
        self.store.save("demo.json", {"stage": "pretrain"})
        self.assertTrue((self.configs_dir / "demo.json").is_file())
        self.assertEqual(self.store.get("demo")["name"], "demo")

    def test_no_temp_files_left(self):
        """验证 `test_no_temp_files_left` 对应的行为约定。"""
        self.store.save("atomic", {"stage": "pretrain"})
        names = [path.name for path in self.configs_dir.iterdir()]
        self.assertEqual(names, ["atomic.json"])


class TestDefaultCommandBuilder(unittest.TestCase):
    """默认命令只允许固定脚本 + 冻结配置。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-cmd-")
        self.root = Path(self._tmp.name)
        (self.root / "pretrain.py").write_text("print('pretrain')\n", encoding="utf-8")
        (self.root / "sft.py").write_text("print('sft')\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_pretrain_command(self):
        """验证 预训练 相关行为。"""
        config_path = self.root / "runs" / "x" / "resolved_config.json"
        command = default_command_builder({"stage": "pretrain"}, self.root, config_path)
        self.assertEqual(
            command,
            [PYTHON, "-u", str(self.root / "pretrain.py"), "--config", str(config_path)],
        )

    def test_sft_command_and_custom_python(self):
        """验证 SFT 相关行为。"""
        config_path = self.root / "resolved_config.json"
        command = default_command_builder(
            {"stage": "sft"}, self.root, config_path, python_executable="/opt/py/python"
        )
        self.assertEqual(command[0], "/opt/py/python")
        self.assertIn(str(self.root / "sft.py"), command)
        self.assertEqual(command[-2:], ["--config", str(config_path)])

    def test_missing_script(self):
        """验证 `test_missing_script` 对应的行为约定。"""
        (self.root / "pretrain.py").unlink()
        with self.assertRaises(JobStartError):
            default_command_builder({"stage": "pretrain"}, self.root, self.root / "c.json")

    def test_unknown_stage(self):
        """验证 阶段页 相关行为。"""
        with self.assertRaises(ConfigError):
            default_command_builder({"stage": "eval"}, self.root, self.root / "c.json")


class TestJobManager(unittest.TestCase):
    """任务管理器的进程生命周期（全部使用假命令）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-jobs-")
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()
        self.runs_dir = self.project / "runs"
        self.manager = JobManager(
            project_root=self.project,
            runs_dir=self.runs_dir,
            command_builder=long_builder,
            max_jobs=1,
            stop_timeout=2.0,
            kill_timeout=2.0,
            poll_interval=0.05,
        )

    def tearDown(self):
        self.manager.shutdown()
        self._tmp.cleanup()

    def _start(self, **kwargs):
        return self.manager.start_job(preset="debug_en", **kwargs)

    def test_start_writes_run_artifacts(self):
        """验证 启动 相关行为。"""
        record = self.manager.start_job(preset="debug_en")
        try:
            self.assertEqual(record.status, "running")
            self.assertIsNotNone(record.pid)
            self.assertGreater(record.pid, 0)
            self.assertTrue(RUN_ID_PATTERN.match(record.run_id))
            run_dir = Path(record.run_dir)
            self.assertTrue(run_dir.is_dir())
            resolved_path = run_dir / "resolved_config.json"
            self.assertTrue(resolved_path.is_file())
            self.assertTrue((run_dir / "console.log").is_file())
            frozen = json.loads(resolved_path.read_text(encoding="utf-8"))
            self.assertEqual(frozen["run_name"], record.run_id)
            self.assertEqual(Path(frozen["runs_dir"]), self.runs_dir.resolve())
            self.assertEqual(frozen["stage"], "pretrain")
            self.assertEqual(frozen["dataset"], "shakespeare_en")
            self.assertTrue(
                wait_until(
                    lambda: "dummy-started" in read_text(run_dir / "console.log"), timeout=15
                ),
                "子进程 stdout 应写入 console.log",
            )
            self.assertEqual(self.manager.active_count(), 1)
            self.assertEqual(len(self.manager.list_jobs()), 1)
            listed = self.manager.list_jobs()[0].to_dict()
            json.dumps(listed)  # 必须可 JSON 序列化
            self.assertTrue(listed["active"])
        finally:
            self.manager.stop_job(record.run_id)

    def test_create_run_then_start_separately(self):
        """验证 创建、启动 相关行为。"""
        record = self.manager.create_run(preset="debug_en", run_name="draft-run")
        self.assertEqual(record.status, "created")
        self.assertFalse(record.is_active)
        self.assertIsNone(record.pid)
        run_dir = Path(record.run_dir)
        status_doc = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status_doc["status"], "created")
        self.assertFalse(
            wait_until(
                lambda: "dummy-started" in read_text(run_dir / "console.log"), timeout=0.5
            )
        )

        started = self.manager.start_run("draft-run")
        try:
            self.assertEqual(started.status, "running")
            self.assertTrue(
                wait_until(
                    lambda: "dummy-started" in read_text(run_dir / "console.log"), timeout=15
                )
            )
        finally:
            self.manager.stop_job("draft-run")

    def test_forget_finished_job_record(self):
        """验证 任务 相关行为。"""
        record = self.manager.start_job(preset="debug_en")
        self.manager.stop_job(record.run_id)
        self.manager.forget_job(record.run_id)
        self.assertIsNone(self.manager.get_job(record.run_id))

    def test_stop_job_terminates_process(self):
        """验证 停止、任务、子进程 相关行为。"""
        record = self.manager.start_job(preset="debug_en")
        try:
            self.assertTrue(
                wait_until(lambda: "dummy-started" in read_text(Path(record.run_dir) / "console.log"))
            )
            final = self.manager.stop_job(record.run_id)
            self.assertEqual(final.status, "stopped")
            self.assertFalse(final.is_active)
            self.assertIsNotNone(final.ended_at)
            self.assertTrue(final.stop_requested)
            self.assertEqual(self.manager.active_count(), 0)
        finally:
            self.manager.stop_job(record.run_id)

    def test_max_jobs_limit(self):
        """验证 任务 相关行为。"""
        first = self.manager.start_job(preset="debug_en")
        try:
            with self.assertRaises(JobLimitExceeded):
                self.manager.start_job(preset="debug_en")
        finally:
            self.manager.stop_job(first.run_id)
        second = self.manager.start_job(preset="debug_en")
        try:
            self.assertEqual(second.status, "running")
        finally:
            self.manager.stop_job(second.run_id)

    def test_completed_job(self):
        """验证 任务 相关行为。"""
        self.manager.command_builder = quick_builder
        record = self.manager.start_job(preset="debug_en")
        final = self.manager.wait_for(record.run_id, timeout=20)
        self.assertEqual(final.status, "completed")
        self.assertEqual(final.returncode, 0)
        self.assertFalse(final.is_active)
        self.assertIn("dummy-done", read_text(Path(record.run_dir) / "console.log"))

    def test_failed_job(self):
        """验证 任务 相关行为。"""
        self.manager.command_builder = failing_builder
        record = self.manager.start_job(preset="debug_en")
        final = self.manager.wait_for(record.run_id, timeout=20)
        self.assertEqual(final.status, "failed")
        self.assertEqual(final.returncode, 3)
        self.assertIn("退出码", final.error or "")
        self.assertIn("boom-stderr", read_text(Path(record.run_dir) / "console.log"))

    def test_spawn_failure_is_reported(self):
        """验证 失败路径 相关行为。"""
        missing = str(self.project / "does-not-exist-python")
        self.manager.command_builder = lambda config, root, cfg: [missing, "-c", "pass"]
        with self.assertRaises(JobStartError):
            self.manager.start_job(preset="debug_en")
        jobs = self.manager.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, "failed")
        self.assertIn("无法启动训练进程", jobs[0].error or "")

    def test_invalid_config_does_not_create_run_dir(self):
        """验证 非法输入、配置、创建 相关行为。"""
        with self.assertRaises(ConfigError):
            self.manager.start_job(config={"n_embd": 7, "n_head": 4}, preset="debug_en")
        self.assertEqual(list(self.runs_dir.iterdir()), [])

    def test_run_name_sanitized_and_duplicates_suffixed(self):
        """验证 `test_run_name_sanitized_and_duplicates_suffixed` 对应的行为约定。"""
        first = self.manager.start_job(preset="debug_en", run_name="My Run!")
        try:
            self.assertEqual(first.run_id, "My-Run")
            frozen = json.loads(
                (Path(first.run_dir) / "resolved_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frozen["run_name"], "My-Run")
        finally:
            self.manager.stop_job(first.run_id)
        second = self.manager.start_job(preset="debug_en", run_name="My Run!")
        try:
            self.assertEqual(second.run_id, "My-Run-2")
        finally:
            self.manager.stop_job(second.run_id)

    def test_unknown_job(self):
        """验证 任务 相关行为。"""
        self.assertIsNone(self.manager.get_job("nope"))
        with self.assertRaises(JobNotFound):
            self.manager.stop_job("nope")
        with self.assertRaises(JobNotFound):
            self.manager.wait_for("nope", timeout=0.2)

    def test_shutdown_stops_active_jobs(self):
        """验证 任务 相关行为。"""
        record = self.manager.start_job(preset="debug_en")
        self.assertTrue(
            wait_until(lambda: "dummy-started" in read_text(Path(record.run_dir) / "console.log"))
        )
        self.manager.shutdown()
        final = self.manager.get_job(record.run_id)
        self.assertIsNotNone(final)
        self.assertEqual(final.status, "stopped")
        self.assertFalse(final.is_active)

    def test_default_builder_end_to_end(self):
        """真实走「固定脚本 + --config 冻结配置」路径（脚本是假的，无需 torch）。"""

        script = self.project / "pretrain.py"
        script.write_text(
            "import sys\nprint('fake-pretrain-argv', sys.argv[1:], flush=True)\n",
            encoding="utf-8",
        )
        manager = JobManager(
            project_root=self.project,
            runs_dir=self.runs_dir,
            poll_interval=0.05,
        )
        try:
            record = manager.start_job(preset="debug_en")
            final = manager.wait_for(record.run_id, timeout=20)
            self.assertEqual(final.status, "completed")
            self.assertEqual(Path(final.command[2]), script.resolve())
            self.assertEqual(final.command[3], "--config")
            console = read_text(Path(record.run_dir) / "console.log")
            self.assertIn("fake-pretrain-argv", console)
            self.assertIn("resolved_config.json", console)
        finally:
            manager.shutdown()

    def test_stop_finished_job_is_noop(self):
        """验证 停止、任务 相关行为。"""
        self.manager.command_builder = quick_builder
        record = self.manager.start_job(preset="debug_en")
        final = self.manager.wait_for(record.run_id, timeout=20)
        again = self.manager.stop_job(record.run_id)
        self.assertEqual(again.status, final.status)

    def test_frozen_config_is_loadable(self):
        """验证 配置 相关行为。"""
        record = self.manager.start_job(preset="debug_en")
        try:
            frozen = json.loads(
                (Path(record.run_dir) / "resolved_config.json").read_text(encoding="utf-8")
            )
            config = Config()
            config.apply_dict(frozen)  # 字段名必须全部合法
            self.assertEqual(Path(config.runs_dir), self.runs_dir.resolve())
        finally:
            self.manager.stop_job(record.run_id)


class TestBackendImportsWithoutTorch(unittest.TestCase):
    """看板后端不得依赖 torch：在禁用 torch 的子进程里导入。"""

    def test_import_with_torch_blocked(self):
        """验证 导入 相关行为。"""
        code = "\n".join(
            [
                "import sys",
                "class Blocker:",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname == 'torch' or fullname.startswith('torch.'):",
                "            raise ImportError('torch is blocked in this test')",
                "sys.meta_path.insert(0, Blocker())",
                "import minigpt.training.job_manager",
                "import minigpt.web.server",
                "import dashboard",
                "print('IMPORT-OK')",
            ]
        )
        completed = subprocess.run(
            [PYTHON, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("IMPORT-OK", completed.stdout)
        self.assertNotIn("torch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
