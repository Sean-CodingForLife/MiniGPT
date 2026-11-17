"""推理子进程并发槽位的资源护栏（阶段 0 新增）。

两个已确认缺陷：
1. start_inference_process 先 acquire 信号量，Popen 抛 FileNotFoundError
   时直接 raise，槽位永久泄漏；
2. close_inference_process 注释称"可重复调用"，但每次都 release
   BoundedSemaphore，二次调用抛 ValueError。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.web.server import MAX_INFERENCE_CONCURRENCY, ApiError, DashboardApp  # noqa: E402

PYTHON = sys.executable


class InferenceSlotTests(unittest.TestCase):
    """确保推理子进程在启动失败或重复关闭时都不会泄漏并发槽位。"""

    def setUp(self) -> None:
        # 用临时目录充当 project_root，避免测试读写真实 runs/data 目录。
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-slots-")
        self.root = Path(self._tmp.name)
        (self.root / "runs").mkdir()
        (self.root / "data").mkdir()
        self.app = DashboardApp(
            project_root=self.root,
            runs_dir=self.root / "runs",
            web_dir=PROJECT_ROOT / "web",
            configs_dir=self.root / "configs",
            data_dir=self.root / "data",
            python_executable=PYTHON,
        )

    def tearDown(self) -> None:
        self.app.close()
        self._tmp.cleanup()

    def _drain_slots(self) -> int:
        """非阻塞地占满信号量再全部释放，返回当前可用槽位数，用于检测是否泄漏。"""
        acquired = 0
        while self.app._inference_slots.acquire(blocking=False):
            acquired += 1
        for _ in range(acquired):
            self.app._inference_slots.release()
        return acquired

    def test_spawn_failure_does_not_leak_slot(self) -> None:
        """Popen 抛 FileNotFoundError 时已获取的信号量必须回滚，否则并发能力逐次下降。"""
        missing = str(self.root / "no-such-python")
        self.app._inference_command = lambda: [missing, "-u", "-c", "pass"]  # type: ignore[method-assign]
        with self.assertRaises(ApiError):
            self.app.start_inference_process({"prompt": "hi"})
        self.assertEqual(
            self._drain_slots(),
            MAX_INFERENCE_CONCURRENCY,
            "Popen 失败后信号量必须完整归还",
        )

    def test_close_is_idempotent(self) -> None:
        """重复关闭同一推理进程不得二次 release BoundedSemaphore 而抛 ValueError。"""
        # 用长睡眠子进程保证关闭时它仍存活，从而真正走终止+归位路径。
        self.app._inference_command = lambda: [  # type: ignore[method-assign]
            PYTHON, "-u", "-c", "import time; time.sleep(30)",
        ]
        process = self.app.start_inference_process({"prompt": "hi"})
        try:
            self.app.close_inference_process(process)
            # 文档承诺 safe to call twice：第二次不得抛 ValueError。
            self.app.close_inference_process(process)
            self.assertEqual(self._drain_slots(), MAX_INFERENCE_CONCURRENCY)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()