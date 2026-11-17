"""批次 B4 加固回归测试：F05/F06/F07/F27/F28/F29。

* F05 推理 batch/stream 错误结构化 + 对外脱敏（worker 子进程 + server 转发）；
* F06 run_id 冲突后缀不再截断出重复候选（有限步内成功或明确报错）；
* F07 system()/错误响应不回显绝对路径、解释器路径与堆栈；
* F27 events/samples 轮询增量读取（游标缓存、截断/轮转重置、API 契约不变）;
* F28 流式推理 stderr 后台有界采集，失败响应带脱敏原因；
* F29 共享 trace 逻辑下沉到 minigpt.render，包内模块不依赖根脚本。
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.training.job_manager import (  # noqa: E402
    RUN_ID_PATTERN,
    ConfigError,
    JobManager,
    normalize_config,
)
from minigpt.web.server import ApiError, DashboardApp  # noqa: E402

PYTHON = sys.executable


def run_worker(request_text: str) -> subprocess.CompletedProcess:
    """以真实子进程运行推理 worker（不加载真实 checkpoint）。"""

    return subprocess.run(
        [PYTHON, "-u", "-m", "minigpt.web.inference_worker"],
        input=request_text,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def parse_stdout_events(stdout: str) -> list[dict]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))  # 非法 JSON 会直接让测试失败
    return events


class TempAppMixin(unittest.TestCase):
    """临时项目根 + runs/demo/best.pt 的 DashboardApp 夹具。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b4-")
        self.root = Path(self._tmp.name).resolve()
        self.runs_dir = self.root / "runs"
        (self.runs_dir / "demo").mkdir(parents=True)
        (self.runs_dir / "demo" / "best.pt").write_bytes(b"fake-checkpoint")
        self.app = DashboardApp(
            project_root=self.root,
            runs_dir=self.runs_dir,
            web_dir=PROJECT_ROOT / "web",
            configs_dir=self.root / "configs",
            python_executable=PYTHON,
        )

    def tearDown(self) -> None:
        self.app.close()
        self._tmp.cleanup()

    def assert_no_leak(self, text: str) -> None:
        """对外文本不得包含项目根路径、解释器路径或堆栈。"""

        self.assertNotIn(str(self.root), text)
        self.assertNotIn(str(PROJECT_ROOT), text)
        self.assertNotIn(sys.executable, text)
        self.assertNotIn("Traceback (most recent call last)", text)
        self.assertNotIn('File "', text)


# ---------------------------------------------------------------------------
# F05 推理错误结构化 + 脱敏
# ---------------------------------------------------------------------------
class WorkerErrorProtocolTests(unittest.TestCase):
    """覆盖 WorkerErrorProtocolTests 相关行为。"""
    def test_batch_bad_checkpoint_emits_json_error(self) -> None:
        """验证 checkpoint、错误处理 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "missing.pt"
            request = json.dumps({"checkpoint": str(bad), "prompt": "hi"})
            result = run_worker(request)
        self.assertEqual(result.returncode, 1)
        events = parse_stdout_events(result.stdout)
        self.assertEqual(len(events), 1)
        error = events[0]
        self.assertEqual(error["type"], "error")
        self.assertTrue(error.get("code"))
        self.assertTrue(error.get("message"))
        self.assertNotIn(str(bad), error["message"])
        self.assertNotIn("Traceback (most recent call last)", result.stdout)
        self.assertNotIn('File "', error["message"])
        # 完整诊断必须留在本地 stderr，而不是被吞掉。
        self.assertIn("Traceback (most recent call last)", result.stderr)

    def test_batch_bad_request_emits_json_error(self) -> None:
        """验证 错误处理 相关行为。"""
        result = run_worker('{"prompt": "hi"}')  # 缺 checkpoint 字段
        self.assertEqual(result.returncode, 1)
        events = parse_stdout_events(result.stdout)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertTrue(events[0].get("code"))
        self.assertIn("Traceback (most recent call last)", result.stderr)

    def test_stream_bad_checkpoint_emits_json_error_with_code(self) -> None:
        """验证 流式输出、checkpoint、错误处理 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "missing.pt"
            request = json.dumps(
                {"checkpoint": str(bad), "prompt": "hi", "stream": True}
            )
            result = run_worker(request)
        self.assertEqual(result.returncode, 1)
        events = parse_stdout_events(result.stdout)
        self.assertEqual(events[-1]["type"], "error")
        error = events[-1]
        self.assertTrue(error.get("code"))
        self.assertNotIn(str(bad), error["message"])
        self.assertNotIn("Traceback (most recent call last)", result.stdout)

    def test_batch_invalid_json_request_emits_json_error(self) -> None:
        """验证 非法输入、错误处理 相关行为。"""
        result = run_worker("{not json")
        self.assertEqual(result.returncode, 1)
        events = parse_stdout_events(result.stdout)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertTrue(events[0].get("code"))


class ServerBatchErrorSanitizationTests(TempAppMixin):
    """覆盖 ServerBatchErrorSanitizationTests 相关行为。"""
    def test_worker_json_error_is_forwarded_sanitized(self) -> None:
        """验证 推理 worker、错误处理 相关行为。"""
        worker = (
            "import json, sys\n"
            "path = " + json.dumps(str(self.root)) + "\n"
            "print(json.dumps({'type': 'error', 'code': 'RuntimeError', "
            "'message': 'boom at ' + path}), flush=True)\n"
            "sys.exit(1)\n"
        )
        self.app._inference_command = lambda: [PYTHON, "-u", "-c", worker]  # type: ignore[method-assign]
        with self.assertRaises(ApiError) as caught:
            self.app.inference({"run_id": "demo", "prompt": "hi"})
        self.assertEqual(caught.exception.status, 500)
        self.assertIn("boom", caught.exception.message)
        self.assert_no_leak(caught.exception.message)

    def test_worker_stderr_traceback_is_sanitized(self) -> None:
        """验证 推理 worker 相关行为。"""
        worker = (
            "import sys\n"
            "path = " + json.dumps(str(self.root)) + "\n"
            "sys.stderr.write('Traceback (most recent call last):\\n  File \"' + path "
            "+ '\", line 1, in <module>\\nRuntimeError: boom-detail\\n')\n"
            "sys.exit(1)\n"
        )
        self.app._inference_command = lambda: [PYTHON, "-u", "-c", worker]  # type: ignore[method-assign]
        with self.assertRaises(ApiError) as caught:
            self.app.inference({"run_id": "demo", "prompt": "hi"})
        self.assertEqual(caught.exception.status, 500)
        self.assertIn("boom-detail", caught.exception.message)
        self.assert_no_leak(caught.exception.message)

    def test_invalid_json_response_is_sanitized(self) -> None:
        """验证 非法输入 相关行为。"""
        worker = (
            "path = " + json.dumps(str(self.root)) + "\n"
            "print('garbage ' + path + ' garbage')\n"
        )
        self.app._inference_command = lambda: [PYTHON, "-u", "-c", worker]  # type: ignore[method-assign]
        with self.assertRaises(ApiError) as caught:
            self.app.inference({"run_id": "demo", "prompt": "hi"})
        self.assertEqual(caught.exception.status, 500)
        self.assert_no_leak(caught.exception.message)

    def test_missing_interpreter_message_has_no_path(self) -> None:
        """验证 路径 相关行为。"""
        self.app._inference_command = lambda: [  # type: ignore[method-assign]
            str(self.root / "no-such-python.exe"),
            "-c",
            "pass",
        ]
        with self.assertRaises(ApiError) as caught:
            self.app.inference({"run_id": "demo", "prompt": "hi"})
        self.assertEqual(caught.exception.status, 500)
        self.assert_no_leak(caught.exception.message)


# ---------------------------------------------------------------------------
# F06 run_id 冲突后缀：不截断 + 重试上限
# ---------------------------------------------------------------------------
class RunIdAllocationTests(unittest.TestCase):
    """覆盖 RunIdAllocationTests 相关行为。"""
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b4-runid-")
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()
        self.runs_dir = self.project / "runs"
        self.runs_dir.mkdir()
        self.manager = JobManager(
            project_root=self.project,
            runs_dir=self.runs_dir,
            command_builder=lambda config, root, cfg: [PYTHON, "-c", "pass"],
        )

    def tearDown(self) -> None:
        self.manager.shutdown()
        self._tmp.cleanup()

    def _config(self, run_name: str):
        return normalize_config(preset="debug_en", run_name=run_name, project_root=self.project)

    def test_long_name_conflicts_do_not_loop_forever(self) -> None:
        """120 字符长名 + 大量同名冲突：旧实现截断出重复候选会死循环。"""

        base = "a" * 120
        # 预占 base 与所有截断后会重复的候选（旧实现 f"{base}-{n}"[:128] 恒等于 base+"-2"）。
        (self.runs_dir / base).mkdir()
        (self.runs_dir / (base + "-2")).mkdir()
        (self.runs_dir / (base + "-3")).mkdir()

        import threading

        outcome: dict = {}

        def allocate() -> None:
            try:
                outcome["run_id"] = self.manager._allocate_run_id(self._config(base))
            except Exception as exc:  # noqa: BLE001 - 记录明确报错也算通过
                outcome["error"] = exc

        worker = threading.Thread(target=allocate, daemon=True)
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(), "run_id 分配陷入死循环")
        if "run_id" in outcome:
            run_id = outcome["run_id"]
            self.assertTrue(RUN_ID_PATTERN.match(run_id))
            self.assertFalse((self.runs_dir / run_id).exists())
            self.assertNotIn(run_id, {base, base + "-2", base + "-3"})
        else:
            self.assertIsInstance(outcome.get("error"), ConfigError)

    def test_retry_limit_raises_clear_error(self) -> None:
        """冲突候选超过重试上限时必须抛明确错误，而不是无限循环。"""

        config = self._config("busy")
        with mock.patch.object(Path, "exists", autospec=True, return_value=True):
            with self.assertRaises(ConfigError) as caught:
                self.manager._allocate_run_id(config)
        self.assertIn("busy", str(caught.exception))

    def test_normal_suffix_still_works(self) -> None:
        """验证 `test_normal_suffix_still_works` 对应的行为约定。"""
        (self.runs_dir / "demo").mkdir()
        (self.runs_dir / "demo-2").mkdir()
        run_id = self.manager._allocate_run_id(self._config("demo"))
        self.assertEqual(run_id, "demo-3")


# ---------------------------------------------------------------------------
# F07 system()/错误响应信息收敛
# ---------------------------------------------------------------------------
class SystemInfoSanitizationTests(TempAppMixin):
    """覆盖 SystemInfoSanitizationTests 相关行为。"""
    def test_system_has_no_absolute_paths(self) -> None:
        """验证 `test_system_has_no_absolute_paths` 对应的行为约定。"""
        payload = self.app.system()
        text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn(sys.executable, text)
        self.assertTrue(payload["python"])  # 版本号保留
        self.assertEqual(payload["project_root"], ".")
        self.assertEqual(payload["runs_dir"], "runs")
        self.assertEqual(payload["configs_dir"], "configs")
        self.assertEqual(payload["web_dir"], "web")

    def test_system_keeps_frontend_fields(self) -> None:
        """验证 `test_system_keeps_frontend_fields` 对应的行为约定。"""
        payload = self.app.system()
        for key in ("python", "torch_available", "cpu_count", "backend", "max_jobs"):
            self.assertIn(key, payload)


class ErrorResponseSanitizationTests(TempAppMixin):
    """覆盖 ErrorResponseSanitizationTests 相关行为。"""
    def setUp(self) -> None:
        super().setUp()
        import threading

        from minigpt.web.server import create_server

        self.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=self.root,
            runs_dir=self.runs_dir,
            web_dir=PROJECT_ROOT / "web",
            configs_dir=self.root / "configs",
            python_executable=PYTHON,
        )
        self.http_thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.http_thread.start()
        self.host, self.port = self.server.server_address[0], self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.http_thread.join(timeout=5)
        self.server.server_close()
        super().tearDown()

    def _get(self, path: str) -> tuple[int, str]:
        import http.client

        connection = http.client.HTTPConnection(self.host, self.port, timeout=15)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8", "replace")
        finally:
            connection.close()

    def _post(self, path: str, body: dict) -> tuple[int, str]:
        import http.client

        connection = http.client.HTTPConnection(self.host, self.port, timeout=60)
        try:
            connection.request(
                "POST",
                path,
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8", "replace")
        finally:
            connection.close()

    def test_system_endpoint_has_no_absolute_paths(self) -> None:
        """验证 `test_system_endpoint_has_no_absolute_paths` 对应的行为约定。"""
        status, text = self._get("/api/v1/system")
        self.assertEqual(status, 200)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn(sys.executable, text)

    def test_health_endpoint_has_no_absolute_paths(self) -> None:
        """验证 `test_health_endpoint_has_no_absolute_paths` 对应的行为约定。"""
        status, text = self._get("/api/v1/health")
        self.assertEqual(status, 200)
        self.assertNotIn(str(self.root), text)

    def test_batch_error_response_has_no_traceback_or_paths(self) -> None:
        """验证 错误处理 相关行为。"""
        worker = (
            "import sys\n"
            "path = " + json.dumps(str(self.root)) + "\n"
            "sys.stderr.write('Traceback (most recent call last):\\n  File \"' + path "
            "+ '\", line 9, in main\\nValueError: leak-check-detail\\n')\n"
            "sys.exit(1)\n"
        )
        self.server.app._inference_command = lambda: [PYTHON, "-u", "-c", worker]  # type: ignore[method-assign]
        status, text = self._post("/api/v1/inference", {"run_id": "demo", "prompt": "hi"})
        self.assertEqual(status, 500)
        self.assertIn("leak-check-detail", text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("Traceback (most recent call last)", text)
        self.assertNotIn("File ", text)
        # 完整诊断必须留在服务端日志里。
        log = self.server.app.diagnostic_log_path
        self.assertTrue(log.is_file())
        self.assertIn("leak-check-detail", log.read_text(encoding="utf-8"))

    def test_job_start_error_response_has_no_paths(self) -> None:
        """验证 任务、启动、错误处理 相关行为。"""
        missing = str(self.root / "no-such-python.exe")
        self.server.app.job_manager.command_builder = (
            lambda config, root, cfg: [missing, "-c", "pass"]
        )
        status, text = self._post(
            "/api/v1/jobs", {"preset": "debug_en", "run_name": "f07-start-error"}
        )
        self.assertEqual(status, 500)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("Traceback", text)


# ---------------------------------------------------------------------------
# F27 events/samples 轮询增量读取
# ---------------------------------------------------------------------------
class IncrementalReadTests(TempAppMixin):
    """覆盖 IncrementalReadTests 相关行为。"""
    def setUp(self) -> None:
        super().setUp()
        self.run_dir = self.runs_dir / "demo"
        self.events_path = self.run_dir / "events.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"

    def _event(self, step: int) -> str:
        return json.dumps({"event": "step", "step": step}, ensure_ascii=False) + "\n"

    def _append(self, path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def test_summary_counts_and_tails_grow_incrementally(self) -> None:
        """验证 摘要 相关行为。"""
        self._append(self.events_path, "".join(self._event(i) for i in range(3)))
        self._append(self.samples_path, json.dumps({"step": 1, "text": "a"}) + "\n")
        first = self.app.run_summary("demo")
        self.assertEqual(first["event_count"], 3)
        self.assertEqual(first["sample_count"], 1)
        self.assertEqual(len(first["events"]), 3)

        self._append(self.events_path, "".join(self._event(i) for i in range(3, 7)))
        self._append(self.samples_path, json.dumps({"step": 2, "text": "b"}) + "\n")
        second = self.app.run_summary("demo")
        self.assertEqual(second["event_count"], 7)
        self.assertEqual(second["sample_count"], 2)
        # 契约：summary 只带最后 60 条事件。
        self.assertEqual([e["step"] for e in second["events"]], list(range(7)))
        self.assertEqual([s["step"] for s in second["samples"]], [1, 2])

    def test_second_poll_only_reads_appended_bytes(self) -> None:
        """验证 `test_second_poll_only_reads_appended_bytes` 对应的行为约定。"""
        self._append(self.events_path, "".join(self._event(i) for i in range(50)))
        self._append(self.samples_path, json.dumps({"step": 1, "text": "a"}) + "\n")
        self.app.run_summary("demo")  # 首次全量

        import builtins

        read_sizes: list[int] = []
        real_open = builtins.open

        class CountingReader:
            """覆盖 CountingReader 相关行为。"""
            def __init__(self, handle, path: Path) -> None:
                self._handle = handle
                self._path = path

            def read(self, *args):
                data = self._handle.read(*args)
                read_sizes.append(len(data))
                return data

            def __getattr__(self, name):
                return getattr(self._handle, name)

            def __enter__(self):
                self._handle.__enter__()
                return self

            def __exit__(self, *exc):
                return self._handle.__exit__(*exc)

        def counting_open(file, mode="r", *args, **kwargs):
            handle = real_open(file, mode, *args, **kwargs)
            try:
                same = Path(file).resolve() in (
                    self.events_path.resolve(),
                    self.samples_path.resolve(),
                )
            except (TypeError, ValueError, OSError):
                same = False
            if same and "r" in mode and "b" in mode:
                return CountingReader(handle, Path(file))
            return handle

        self._append(self.events_path, self._event(50))
        with mock.patch.object(builtins, "open", counting_open), mock.patch.object(
            io, "open", counting_open
        ):
            payload = self.app.run_summary("demo")
        self.assertEqual(payload["event_count"], 51)
        self.assertTrue(read_sizes, "应发生文件读取")
        self.assertLess(max(read_sizes), 200, f"二次轮询不应全量重读: {read_sizes}")

    def test_truncated_file_is_reread_from_start(self) -> None:
        """验证 启动 相关行为。"""
        self._append(self.events_path, "".join(self._event(i) for i in range(5)))
        self.assertEqual(self.app.run_summary("demo")["event_count"], 5)
        # 轮转：文件变小且内容全新。
        self.events_path.write_text(self._event(100) + self._event(101), encoding="utf-8")
        payload = self.app.run_summary("demo")
        self.assertEqual(payload["event_count"], 2)
        self.assertEqual([e["step"] for e in payload["events"]], [100, 101])

    def test_run_events_window_contract_unchanged(self) -> None:
        """验证 事件流 相关行为。"""
        self._append(self.events_path, "".join(self._event(i) for i in range(6)))
        self.app.run_summary("demo")  # 预热游标
        payload = self.app.run_events("demo", after=2, limit=2)
        self.assertEqual(payload["after"], 2)
        self.assertEqual(payload["next_after"], 4)
        self.assertEqual(payload["total"], 6)
        self.assertEqual([e["step"] for e in payload["events"]], [2, 3])
        self.assertEqual(payload["events"][0]["index"], 2)
        self.assertEqual(payload["run_id"], "demo")
        # 追加后 total 与增量窗口仍然正确。
        self._append(self.events_path, self._event(6))
        payload = self.app.run_events("demo", after=5, limit=10)
        self.assertEqual(payload["total"], 7)
        self.assertEqual([e["step"] for e in payload["events"]], [5, 6])
        self.assertEqual(payload["next_after"], 7)

    def test_run_samples_incremental_and_limit(self) -> None:
        """验证 `test_run_samples_incremental_and_limit` 对应的行为约定。"""
        for step in range(4):
            self._append(self.samples_path, json.dumps({"step": step, "text": f"s{step}"}) + "\n")
        payload = self.app.run_samples("demo", limit=2)
        self.assertEqual(payload["count"], 2)
        self.assertEqual([s["step"] for s in payload["samples"]], [2, 3])
        self._append(self.samples_path, json.dumps({"step": 4, "text": "s4"}) + "\n")
        payload = self.app.run_samples("demo", limit=2)
        self.assertEqual([s["step"] for s in payload["samples"]], [3, 4])
        self.assertEqual(payload["count"], 2)

    def test_partial_line_is_not_parsed_until_complete(self) -> None:
        """验证 `test_partial_line_is_not_parsed_until_complete` 对应的行为约定。"""
        self._append(self.events_path, self._event(0))
        self.assertEqual(self.app.run_summary("demo")["event_count"], 1)
        half = json.dumps({"event": "step", "step": 1}, ensure_ascii=False)
        self._append(self.events_path, half[:10])  # 无换行的半行
        payload = self.app.run_summary("demo")
        self.assertEqual(payload["event_count"], 1)
        self._append(self.events_path, half[10:] + "\n")
        payload = self.app.run_summary("demo")
        self.assertEqual(payload["event_count"], 2)

    def test_concurrent_readers_are_consistent(self) -> None:
        """验证 `test_concurrent_readers_are_consistent` 对应的行为约定。"""
        import threading

        self._append(self.events_path, "".join(self._event(i) for i in range(20)))
        errors: list[Exception] = []
        results: list[int] = []
        lock = threading.Lock()

        def poll() -> None:
            try:
                for _ in range(10):
                    payload = self.app.run_summary("demo")
                    with lock:
                        results.append(payload["event_count"])
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=poll) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertTrue(all(count == 20 for count in results), results[:10])


# ---------------------------------------------------------------------------
# F28 流式推理 stderr 后台采集
# ---------------------------------------------------------------------------
class StreamStderrCaptureTests(unittest.TestCase):
    """覆盖 StreamStderrCaptureTests 相关行为。"""
    def setUp(self) -> None:
        import threading

        from minigpt.web.server import create_server

        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b4-stream-")
        self.root = Path(self._tmp.name).resolve()
        (self.root / "runs" / "demo").mkdir(parents=True)
        (self.root / "runs" / "demo" / "best.pt").write_bytes(b"fake")
        self.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=self.root,
            runs_dir=self.root / "runs",
            web_dir=PROJECT_ROOT / "web",
            configs_dir=self.root / "configs",
            python_executable=PYTHON,
        )
        self.app = self.server.app
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()
        self.host, self.port = self.server.server_address[0], self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self._tmp.cleanup()

    def _stream_events(self, worker: str) -> list[dict]:
        import http.client

        self.app._inference_command = lambda: [PYTHON, "-u", "-c", worker]  # type: ignore[method-assign]
        connection = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            connection.request(
                "POST",
                "/api/v1/inference/stream",
                body=json.dumps({"run_id": "demo", "prompt": "hi"}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            events = []
            for line in response:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
            return events
        finally:
            connection.close()

    def test_stream_failure_includes_sanitized_stderr_reason(self) -> None:
        """验证 流式输出、失败路径 相关行为。"""
        worker = (
            "import sys\n"
            "path = " + json.dumps(str(self.root)) + "\n"
            "sys.stderr.write('Traceback (most recent call last):\\n  File \"' + path "
            "+ '\", line 3, in main\\nRuntimeError: stream-boom-detail\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(3)\n"
        )
        events = self._stream_events(worker)
        self.assertTrue(events, "失败时必须有 error 事件")
        error = events[-1]
        self.assertEqual(error["type"], "error")
        message = error["message"]
        self.assertIn("stream-boom-detail", message)
        self.assertNotIn(str(self.root), message)
        self.assertNotIn("Traceback (most recent call last)", message)
        self.assertNotIn("UnsupportedOperation", message)
        self.assertNotIn("seek", message)
        # 完整诊断留服务端日志。
        log = self.app.diagnostic_log_path
        self.assertTrue(log.is_file())
        self.assertIn("stream-boom-detail", log.read_text(encoding="utf-8"))

    def test_stderr_capture_is_bounded(self) -> None:
        """验证 `test_stderr_capture_is_bounded` 对应的行为约定。"""
        from minigpt.web.server import STDERR_CAPTURE_BYTES

        self.assertLessEqual(STDERR_CAPTURE_BYTES, 64 * 1024)
        worker = (
            "import sys\n"
            "sys.stderr.write('x' * 300000 + '\\nfinal-reason-marker\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(4)\n"
        )
        events = self._stream_events(worker)
        error = events[-1]
        self.assertEqual(error["type"], "error")
        self.assertIn("final-reason-marker", error["message"])
        self.assertLessEqual(len(error["message"]), 700)

    def test_stream_success_unaffected(self) -> None:
        """验证 流式输出 相关行为。"""
        worker = (
            "import json, sys\n"
            "sys.stderr.write('warning: harmless noise\\n')\n"
            "print(json.dumps({'type': 'step', 'data': {'step': 1}}), flush=True)\n"
            "print(json.dumps({'type': 'done'}), flush=True)\n"
        )
        events = self._stream_events(worker)
        self.assertEqual([e["type"] for e in events], ["step", "done"])


if __name__ == "__main__":
    unittest.main()