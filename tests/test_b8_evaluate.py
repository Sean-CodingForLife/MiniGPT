"""B8-N08 面板内评估：POST /api/v1/evaluate 子进程 + 状态流转 + 结果读取。

用注入的假评估命令（python -c）替代真实 evaluate.py：不启动 PyTorch，
但完整走 JobManager 的锁/pid/状态文件/监控线程路径。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.web.server import ApiError, DashboardApp  # noqa: E402

ROOT = PROJECT_ROOT
PYTHON = sys.executable

# 假评估进程：解析 --output_dir / --split，稍睡后写 evaluation.json（模拟耗时评估）。
FAKE_EVAL_SNIPPET = (
    "import json, sys, time\n"
    "from pathlib import Path\n"
    "argv = sys.argv\n"
    "out = Path(argv[argv.index('--output_dir') + 1])\n"
    "split = argv[argv.index('--split') + 1]\n"
    "time.sleep(0.3)\n"
    "out.mkdir(parents=True, exist_ok=True)\n"
    "report = {'stage': 'pretrain', 'split': split, 'loss': 1.5, "
    "'perplexity': 4.4817, 'tokens': 1024, 'sample': 'hello eval'}\n"
    "(out / 'evaluation.json').write_text(json.dumps(report), encoding='utf-8')\n"
    "print('fake-eval-done', flush=True)\n"
)

FAKE_EVAL_FAIL_SNIPPET = "import sys; sys.stderr.write('boom-eval\\n'); sys.exit(3)\n"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_dataset(root: Path, name: str, kind: str = "pretrain") -> Path:
    directory = root / "data" / name
    directory.mkdir(parents=True, exist_ok=True)
    meta_name = "sft_meta.json" if kind == "sft" else "meta.json"
    write_json(directory / meta_name, {"stage": kind, "vocab_size": 65})
    for split in ("train", "val", "test"):
        if kind == "sft":
            (directory / f"{split}.jsonl").write_text('{"q": "hi", "a": "hello"}\n', encoding="utf-8")
        else:
            (directory / f"{split}.bin").write_bytes(b"\x01\x00" * 64)
    return directory


def make_run(root: Path, run_id: str, *, checkpoint: str = "best.pt", stage: str = "pretrain") -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        (run_dir / checkpoint).write_bytes(b"fake-checkpoint")
    write_json(run_dir / "resolved_config.json", {"stage": stage, "max_steps": 10})
    write_json(run_dir / "status.json", {"status": "completed", "stage": stage, "step": 10})
    return run_dir


class EvaluateAppTests(unittest.TestCase):
    """覆盖 EvaluateAppTests 相关行为。"""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-eval-")
        self.root = Path(self._tmp.name)
        make_dataset(self.root, "demo_pretrain")
        make_dataset(self.root, "demo_sft", kind="sft")
        self.run_dir = make_run(self.root, "run-x")
        self.app = DashboardApp(
            project_root=self.root,
            runs_dir=self.root / "runs",
            web_dir=ROOT / "web",
            configs_dir=self.root / "configs",
            data_dir=self.root / "data",
            python_executable=PYTHON,
        )
        # 注入假评估命令：保留真实 evaluate.py 的参数形态。
        self.app._evaluate_command = lambda run_dir, checkpoint, dataset, split, output_dir: [  # type: ignore[method-assign]
            PYTHON,
            "-u",
            "-c",
            FAKE_EVAL_SNIPPET,
            "--checkpoint",
            str(checkpoint),
            "--split",
            split,
            "--output_dir",
            str(output_dir),
        ]

    def tearDown(self):
        self.app.close()
        self._tmp.cleanup()

    def wait_status(self, run_id: str, statuses: set[str], timeout: float = 15.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload = self.app.evaluate_status(run_id)
            if payload.get("status") in statuses:
                return payload
            time.sleep(0.05)
        self.fail(f"评估未在 {timeout}s 内进入 {statuses}: {payload}")

    def test_start_status_result_flow(self):
        """验证 启动、状态文件 相关行为。"""
        started = self.app.start_evaluation({"run_id": "run-x", "dataset": "demo_pretrain"})
        self.assertEqual(started["run_id"], "run-x")
        self.assertEqual(started["dataset"], "demo_pretrain")
        self.assertEqual(started["split"], "test")
        self.assertEqual(started["eval_dir"], "eval_test")
        self.assertTrue(started["pid"] > 0)
        self.assertIn(started["status"], {"starting", "running"})
        # 状态文件已落盘（复用训练任务的状态文件契约）。
        status_file = json.loads((self.run_dir / "eval_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status_file["run_id"], "run-x")
        self.assertEqual(status_file["kind"], "evaluation")

        final = self.wait_status("run-x", {"completed", "failed"})
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["returncode"], 0)
        self.assertIsNotNone(final["result"])
        self.assertAlmostEqual(final["result"]["perplexity"], 4.4817, places=3)
        self.assertEqual(final["result"]["split"], "test")

        # 事件写入该 run 的 events.jsonl。
        events = [
            json.loads(line)
            for line in (self.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kinds = [event.get("event") for event in events]
        self.assertIn("eval_start", kinds)
        self.assertIn("eval_end", kinds)
        end_event = [event for event in events if event.get("event") == "eval_end"][-1]
        self.assertEqual(end_event["payload"]["status"], "completed")
        # 评估控制台日志独立于训练 console.log。
        self.assertTrue((self.run_dir / "eval_console.log").is_file())

    def test_status_idle_for_run_without_evaluation(self):
        """验证 状态文件 相关行为。"""
        payload = self.app.evaluate_status("run-x")
        self.assertEqual(payload["status"], "idle")
        self.assertIsNone(payload["result"])

    def test_duplicate_start_rejected_while_active(self):
        """验证 启动 相关行为。"""
        self.app.start_evaluation({"run_id": "run-x", "dataset": "demo_pretrain"})
        with self.assertRaises(ApiError) as caught:
            self.app.start_evaluation({"run_id": "run-x", "dataset": "demo_pretrain"})
        self.assertEqual(caught.exception.status, 409)
        self.wait_status("run-x", {"completed"})
        # 结束后可以再次评估。
        again = self.app.start_evaluation({"run_id": "run-x", "dataset": "demo_pretrain"})
        self.assertIn(again["status"], {"starting", "running"})
        self.wait_status("run-x", {"completed"})

    def test_invalid_dataset_rejected(self):
        """验证 非法输入、数据集 相关行为。"""
        for dataset in ("no-such-set", "../evil", "a/b", "", None):
            with self.assertRaises(ApiError) as caught:
                self.app.start_evaluation({"run_id": "run-x", "dataset": dataset})
            self.assertIn(caught.exception.status, {400, 404}, str(dataset))

    def test_invalid_split_rejected(self):
        """验证 非法输入、切分 相关行为。"""
        with self.assertRaises(ApiError) as caught:
            self.app.start_evaluation({"run_id": "run-x", "dataset": "demo_pretrain", "split": "train"})
        self.assertEqual(caught.exception.status, 400)

    def test_unknown_run_rejected(self):
        """验证 `test_unknown_run_rejected` 对应的行为约定。"""
        from minigpt.training.job_manager import JobNotFound

        # 应用层抛 JobNotFound，HTTP 层统一映射为 404（见 EvaluateHttpTests）。
        with self.assertRaises(JobNotFound):
            self.app.start_evaluation({"run_id": "missing", "dataset": "demo_pretrain"})

    def test_run_without_checkpoint_rejected(self):
        """验证 checkpoint 相关行为。"""
        make_run(self.root, "run-nockpt", checkpoint="")
        with self.assertRaises(ApiError) as caught:
            self.app.start_evaluation({"run_id": "run-nockpt", "dataset": "demo_pretrain"})
        self.assertEqual(caught.exception.status, 400)

    def test_failed_evaluation_reported(self):
        """验证 `test_failed_evaluation_reported` 对应的行为约定。"""
        self.app._evaluate_command = lambda run_dir, checkpoint, dataset, split, output_dir: [  # type: ignore[method-assign]
            PYTHON, "-u", "-c", FAKE_EVAL_FAIL_SNIPPET,
        ]
        self.app.start_evaluation({"run_id": "run-x", "dataset": "demo_pretrain"})
        final = self.wait_status("run-x", {"failed", "completed"})
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["returncode"], 3)
        self.assertIsNone(final["result"])
        self.assertTrue(final.get("error"))

    def test_dataset_whitelist_listing(self):
        """验证 数据集 相关行为。"""
        names = self.app.evaluation_datasets()
        self.assertIn("demo_pretrain", names)
        self.assertIn("demo_sft", names)


class EvaluateHttpTests(unittest.TestCase):
    """路由层：POST /api/v1/evaluate 与 GET /api/v1/evaluate/status。"""

    def setUp(self):
        import threading

        from minigpt.web.server import create_server

        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-eval-http-")
        self.root = Path(self._tmp.name)
        make_dataset(self.root, "demo_pretrain")
        make_run(self.root, "run-x")
        self.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=self.root,
            runs_dir=self.root / "runs",
            web_dir=ROOT / "web",
            configs_dir=self.root / "configs",
            data_dir=self.root / "data",
            python_executable=PYTHON,
        )
        self.server.app._evaluate_command = lambda run_dir, checkpoint, dataset, split, output_dir: [  # type: ignore[method-assign]
            PYTHON, "-u", "-c", FAKE_EVAL_SNIPPET,
            "--checkpoint", str(checkpoint), "--split", split, "--output_dir", str(output_dir),
        ]
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=10)
        self.server.server_close()
        self._tmp.cleanup()

    def request(self, method: str, path: str, body=None):
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            data = response.read()
            status = response.status
        finally:
            connection.close()
        try:
            parsed = json.loads(data.decode("utf-8"))
        except ValueError:
            parsed = None
        return status, parsed

    def test_post_evaluate_then_status_then_result(self):
        """验证 评估、状态文件 相关行为。"""
        status, payload = self.request("POST", "/api/v1/evaluate", {"run_id": "run-x", "dataset": "demo_pretrain"})
        self.assertEqual(status, 202, payload)
        self.assertEqual(payload["run_id"], "run-x")
        deadline = time.monotonic() + 15
        final = None
        while time.monotonic() < deadline:
            status, final = self.request("GET", "/api/v1/evaluate/status?run_id=run-x")
            self.assertEqual(status, 200)
            if final["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(final["status"], "completed")
        self.assertAlmostEqual(final["result"]["loss"], 1.5, places=3)

    def test_post_evaluate_bad_dataset_400(self):
        """验证 评估、数据集 相关行为。"""
        status, payload = self.request("POST", "/api/v1/evaluate", {"run_id": "run-x", "dataset": "nope"})
        self.assertIn(status, {400, 404})
        self.assertIn("error", payload)

    def test_status_requires_run_id(self):
        """验证 状态文件 相关行为。"""
        status, payload = self.request("GET", "/api/v1/evaluate/status")
        self.assertEqual(status, 400)

    def test_unknown_run_404(self):
        """验证 `test_unknown_run_404` 对应的行为约定。"""
        status, payload = self.request("POST", "/api/v1/evaluate", {"run_id": "missing", "dataset": "demo_pretrain"})
        self.assertEqual(status, 404)
        status, payload = self.request("GET", "/api/v1/evaluate/status?run_id=missing")
        self.assertEqual(status, 404)

    def test_method_not_allowed(self):
        """验证 `test_method_not_allowed` 对应的行为约定。"""
        status, _ = self.request("GET", "/api/v1/evaluate")
        self.assertEqual(status, 405)
        status, _ = self.request("POST", "/api/v1/evaluate/status?run_id=run-x", {})
        self.assertEqual(status, 405)


class EvaluateDomTests(unittest.TestCase):
    """覆盖 EvaluateDomTests 相关行为。"""
    def test_eval_launch_form_wired(self):
        """验证 评估、表单 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        evals = (ROOT / "web" / "js" / "eval.js").read_text(encoding="utf-8")
        for marker in (
            'id="eval-launch-form"',
            'id="eval-launch-run"',
            'id="eval-launch-dataset"',
            'id="eval-launch-split"',
            'id="eval-launch-btn"',
            'id="eval-launch-status"',
        ):
            self.assertIn(marker, index)
        for marker in (
            "/api/v1/evaluate",
            "/api/v1/evaluate/status",
            "eval-launch-btn",
            "setInterval",
        ):
            self.assertIn(marker, evals)
        # 现有只读契约不被破坏。
        self.assertIn("/api/v1/evaluations", evals)
        self.assertIn('id="eval-list"', index)
        self.assertIn('id="eval-detail"', index)


if __name__ == "__main__":
    unittest.main()