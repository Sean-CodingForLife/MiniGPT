"""JobManager 续训加固护栏（阶段 0 新增）。

1. start_run 自动续训只查 last.pt，不查 pure 后端的 last.mgckpt；
2. resume_job 的 is_active 检查在锁外，并发双 resume 可双进程写同一
   run 目录（start_run 的 _launch_record 有锁内复检，resume 没有）。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.training.job_manager import JobManager, JobStartError  # noqa: E402

PYTHON = sys.executable
LONG_COMMAND = [PYTHON, "-u", "-c", "import time; print('up', flush=True); time.sleep(60)"]


class JobHardeningTests(unittest.TestCase):
    """覆盖自动续训的检查点识别与并发 resume 的单进程保证。"""

    def setUp(self) -> None:
        # 独立临时 project_root，避免碰到仓库里真实的 runs/ 与状态文件。
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-jobhard-")
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()
        self.runs_dir = self.project / "runs"
        self.launches: list[list[str]] = []
        self._launch_lock = threading.Lock()

        def builder(config, project_root, config_file):
            # 用长睡眠子进程代替真实训练命令，便于观察并发进程数。
            with self._launch_lock:
                self.launches.append(list(LONG_COMMAND))
            return list(LONG_COMMAND)

        self.manager = JobManager(
            project_root=self.project,
            runs_dir=self.runs_dir,
            command_builder=builder,
            max_jobs=4,
            poll_interval=0.05,
        )

    def tearDown(self) -> None:
        self.manager.shutdown()
        # Windows 上子进程日志句柄释放后目录可能短暂占用：容错重试清理。
        for _ in range(5):
            try:
                self._tmp.cleanup()
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            # TemporaryDirectory.cleanup() 在 Python 3.14 不接受 ignore_errors；
            # 用 shutil.rmtree 兜底，保持「重试全失败也不让 tearDown 抛错」的语义。
            shutil.rmtree(self._tmp.name, ignore_errors=True)

    def _finished_run(self, run_name: str, checkpoint: str) -> str:
        """造出一个「已停止且已有 checkpoint」的历史 run，用于触发自动续训判定。"""
        record = self.manager.create_run(preset="debug_en", run_name=run_name)
        run_dir = Path(record.run_dir)
        (run_dir / checkpoint).write_bytes(b"fake-checkpoint")
        (run_dir / "status.json").write_text(
            json.dumps({"status": "stopped", "step": 3}), encoding="utf-8"
        )
        # 逐出内存记录：模拟服务重启后从磁盘恢复的历史运行（CLI 训练产物），
        # 否则内存里的 created 草稿状态按设计不触发自动续训。
        with self.manager._lock:
            self.manager._jobs.pop(record.run_id, None)
        return record.run_id

    def test_start_run_autodetects_pure_checkpoint(self) -> None:
        """只有 last.mgckpt（pure 后端）时也必须自动附加 --resume，不能漏续训。"""
        run_id = self._finished_run("pure-run", "last.mgckpt")
        started = self.manager.start_run(run_id)
        try:
            self.assertIn("--resume", started.command, "pure 检查点应触发自动续训")
        finally:
            self.manager.stop_job(run_id)

    def test_concurrent_resume_launches_single_process(self) -> None:
        """两个线程同时 resume 同一 run，必须一成一败且只存在一个子进程。"""
        run_id = self._finished_run("race-run", "last.pt")
        barrier = threading.Barrier(2, timeout=10)
        outcomes: list[str] = []
        outcomes_lock = threading.Lock()

        def attempt() -> None:
            # 用 Barrier 把两次 resume 卡在同一时刻，最大化竞态窗口。
            barrier.wait()
            try:
                self.manager.resume_job(run_id)
                result = "launched"
            except JobStartError:
                result = "rejected"
            with outcomes_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        try:
            self.assertEqual(sorted(outcomes), ["launched", "rejected"], f"实际结果: {outcomes}")
            # command_builder 在创建草稿时也会被调用，不能当进程数；
            # 以真实拉起的子进程为准。
            time.sleep(0.2)
            self.assertEqual(len(self.manager._processes), 1, "同一 run 只允许拉起一个续训进程")
        finally:
            self.manager.stop_job(run_id)


if __name__ == "__main__":
    unittest.main()