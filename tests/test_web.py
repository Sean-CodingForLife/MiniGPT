"""``minigpt.web.server`` 的集成测试：真实 HTTP 服务（端口 0）+ 临时目录。

覆盖 CONSTRUCTION §7.6 的全部 API：health / presets / configs CRUD+validate /
jobs 启动停止（注入假命令）/ runs summary·events·metrics·samples / system，
以及静态资源、路径穿越防护、无外部 CDN、无 torch 后端、dashboard.py CLI。
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.training.job_manager import RUN_ID_PATTERN, validate_config_name  # noqa: E402
from minigpt.web.server import create_server  # noqa: E402

PYTHON = sys.executable
FIXED_TS = "2026-11-11T10:00:00+00:00"
FIXED_TS_END = "2026-11-11T10:05:00+00:00"

# 注入的假训练命令：长睡，用于测启动/停止，不启动真实 PyTorch。
LONG_COMMAND = [
    PYTHON,
    "-u",
    "-c",
    "import time; print('web-dummy-started', flush=True); time.sleep(60)",
]
METRIC_HEADER = "step,train_loss,val_loss,val_ppl,val_bpc,lr,grad_norm,tokens_per_sec,elapsed_s,best_val"


def long_builder(config, project_root, config_file):
    return list(LONG_COMMAND)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_run(
    runs_dir: Path,
    run_id: str,
    *,
    status: str = "completed",
    step: int = 3000,
    with_status: bool = True,
    with_report: bool = True,
    minirun: bool = False,
) -> Path:
    """构造一个看起来像训练进程产物的运行目录。"""

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "pretrain",
        "dataset": "shakespeare_en",
        "data_dir": "data/shakespeare_en",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 128,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "max_steps": 3000,
        "lr": 0.001,
        "device": "cpu",
        "dtype": "float32",
        "backend_type": "torch",
        "seed": 1337,
        "run_name": run_id,
    }
    write_json(run_dir / "resolved_config.json", dict(config, runs_dir=str(runs_dir)))
    write_json(
        run_dir / "run_config.json",
        {
            "run_id": run_id,
            "created_at": FIXED_TS,
            "argv": ["pretrain.py", "--config", "resolved_config.json"],
            "config": config,
        },
    )
    write_json(
        run_dir / "environment.json",
        {
            "run_id": run_id,
            "python": "3.14.5 (main)",
            "python_version": "3.14.5",
            "torch": "2.15.0",
            "torch_version": "2.15.0",
            "device": "cpu",
            "cuda_available": False,
            "platform": "Windows-11-10.0.26100",
            "parameters": 812000,
            "project_root": str(PROJECT_ROOT),
        },
    )
    if with_status:
        write_json(
            run_dir / "status.json",
            {
                "run_id": run_id,
                "stage": "pretrain",
                "status": status,
                "step": step,
                "max_steps": 3000,
                "best_val": 1.87,
                "checkpoint": "best.pt",
                "pid": 4321,
                "started_at": FIXED_TS,
                "updated_at": FIXED_TS_END,
                "error": "",
                "lr": 0.0003,
                "train_loss": 2.01,
                "val_loss": 1.95,
                "grad_norm": 0.9,
                "tokens_per_sec": 1234.5,
            },
        )
    (run_dir / "metrics.csv").write_text(
        METRIC_HEADER
        + "\n"
        + "200,3.900000,2.900000,18.1741,4.1836,0.00100000,0.950000,1100.00,12.50,2.900000\n"
        + "3000,2.010000,1.950000,7.0297,2.8133,0.00030000,0.900000,1400.00,120.00,1.870000\n",
        encoding="utf-8",
    )
    events = [
        {
            "v": 1,
            "ts": FIXED_TS,
            "run_id": run_id,
            "event": "run_start",
            "step": 0,
            "payload": {"stage": "pretrain", "max_steps": 3000, "seed": 1337},
        },
        {
            "v": 1,
            "ts": FIXED_TS,
            "run_id": run_id,
            "event": "step",
            "step": 20,
            "payload": {"train_loss": 3.95, "lr": 0.0009, "grad_norm": 1.2, "tokens_per_sec": 1000.0},
        },
        {
            "v": 1,
            "ts": FIXED_TS,
            "run_id": run_id,
            "event": "eval",
            "step": 200,
            "payload": {"val_loss": 2.9, "best_val": 2.9},
        },
        {
            "v": 1,
            "ts": FIXED_TS,
            "run_id": run_id,
            "event": "checkpoint",
            "step": 200,
            "payload": {"kind": "best", "path": "best.pt"},
        },
        {
            "v": 1,
            "ts": FIXED_TS,
            "run_id": run_id,
            "event": "sample",
            "step": 200,
            "payload": {"prompt": "ROMEO:", "text": "ROMEO: hello there"},
        },
        {
            "v": 1,
            "ts": FIXED_TS_END,
            "run_id": run_id,
            "event": "run_end",
            "step": step,
            "payload": {"status": "completed", "best_val": 1.87},
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    samples = [
        {"v": 1, "ts": FIXED_TS, "run_id": run_id, "step": 200, "prompt": "ROMEO:", "text": "ROMEO: hello there"},
        {"v": 1, "ts": FIXED_TS_END, "run_id": run_id, "step": 3000, "prompt": "ROMEO:", "text": "ROMEO: good morrow"},
    ]
    (run_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    (run_dir / "console.log").write_text(
        "console line one\n[eval] step=200 val_loss=2.9000\n", encoding="utf-8"
    )
    if with_report:
        write_json(
            run_dir / "training_report.json",
            {
                "run_id": run_id,
                "stage": "pretrain",
                "status": "completed",
                "steps": step,
                "best_val_loss": 1.87,
                "best_step": 2800,
                "stopped_early": False,
                "patience": 10,
                "initial_loss": 4.21,
                "final_train_loss": 1.98,
                "elapsed_s": 120.0,
                "error": "",
                "created_at": FIXED_TS_END,
            },
        )
    if minirun:
        mini_dir = runs_dir / f"{run_id}-mini"
        write_json(
            mini_dir / "minirun_report.json",
            {
                "passed": True,
                "vocab_size": 68,
                "expected_initial_loss": 4.2195,
                "initial_loss": 4.21,
                "initial_loss_abs_diff": 0.0095,
                "initial_loss_ok": True,
                "gradient_ok": True,
                "final_train_loss": 3.9,
                "loss_drop": 0.31,
                "min_drop": 0.2,
                "run_id": f"{run_id}-mini",
            },
        )
    return run_dir


class DashboardServerTest(unittest.TestCase):
    """真实 HTTP 服务集成测试（端口 0，类级共享，临时目录隔离）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="minigpt-web-")
        root = Path(cls._tmp.name)
        cls.runs_dir = root / "runs"
        cls.runs_dir.mkdir(parents=True)
        cls.configs_dir = root / "configs"
        cls.configs_dir.mkdir(parents=True)

        make_run(cls.runs_dir, "run-completed", status="completed", step=3000, minirun=True)
        make_run(cls.runs_dir, "run-live", status="running", step=200, with_report=False)
        make_run(cls.runs_dir, "run-minimal", with_status=False, with_report=False)
        bad_dir = cls.runs_dir / "bad run"
        bad_dir.mkdir()
        write_json(bad_dir / "status.json", {"status": "running"})

        cls.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=PROJECT_ROOT,
            runs_dir=cls.runs_dir,
            web_dir=PROJECT_ROOT / "web",
            configs_dir=cls.configs_dir,
            max_jobs=1,
            command_builder=long_builder,
            stop_timeout=2.0,
        )
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=10)
        cls.server.server_close()
        cls._tmp.cleanup()

    def tearDown(self):
        # 任何测试遗留的活动任务都要停掉，避免影响其它用例（max_jobs=1）。
        try:
            status, payload = self.json_request("GET", "/api/v1/jobs")
            for job in (payload or {}).get("jobs", []):
                if job.get("active"):
                    self.json_request("POST", "/api/v1/jobs/%s/stop" % job["run_id"], {})
        except Exception:  # noqa: BLE001 - 清理失败不应掩盖测试结果
            pass

    # -- HTTP 小工具 -------------------------------------------------------
    def request(self, method: str, path: str, body=None, raw_body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        request_headers = dict(headers or {})
        payload = None
        if raw_body is not None:
            payload = raw_body.encode("utf-8") if isinstance(raw_body, str) else raw_body
            request_headers.setdefault("Content-Type", "application/json")
        elif body is not None:
            payload = json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            data = response.read()
            status = response.status
            response_headers = dict(response.getheaders())
        finally:
            connection.close()
        return status, data, response_headers

    def json_request(self, method: str, path: str, body=None):
        status, data, headers = self.request(method, path, body)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        return status, payload

    # -- 基础 API ----------------------------------------------------------
    def test_health(self):
        """验证 `test_health` 对应的行为约定。"""
        status, payload = self.json_request("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("version", payload)
        self.assertEqual(payload["max_jobs"], 1)
        self.assertIn("active_jobs", payload)
        self.assertIn("uptime_s", payload)

    def test_presets(self):
        """验证 `test_presets` 对应的行为约定。"""
        status, payload = self.json_request("GET", "/api/v1/presets")
        self.assertEqual(status, 200)
        names = payload["names"]
        self.assertIn("debug_en", names)
        self.assertIn("debug_zh", names)
        self.assertIn("sft_debug_en", names)
        self.assertEqual(payload["presets"]["debug_en"]["stage"], "pretrain")
        self.assertEqual(payload["presets"]["sft_debug_en"]["stage"], "sft")
        self.assertEqual(payload["presets"]["debug_en"]["max_steps"], 3000)

    def test_system(self):
        """验证 `test_system` 对应的行为约定。"""
        status, payload = self.json_request("GET", "/api/v1/system")
        self.assertEqual(status, 200)
        self.assertIsInstance(payload["torch_available"], bool)
        self.assertTrue(payload["python"])
        # F07：对外只给相对/脱敏信息，不回显绝对路径与解释器路径。
        self.assertEqual(payload["project_root"], ".")
        self.assertEqual(payload["runs_dir"], self.runs_dir.name)
        self.assertEqual(payload["configs_dir"], self.configs_dir.name)
        self.assertNotIn("python_executable", payload)
        text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(str(PROJECT_ROOT), text)
        self.assertNotIn(sys.executable, text)
        self.assertEqual(payload["backend"], "stdlib")

    # -- 静态资源与安全 ----------------------------------------------------
    def test_index_page(self):
        """验证 `test_index_page` 对应的行为约定。"""
        status, data, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        text = data.decode("utf-8")
        self.assertIn("MiniGPT", text)
        self.assertIn("/static/app.js", text)
        self.assertIn("/static/style.css", text)
        self.assertIn("/static/vendor/chart.umd.min.js", text)

    def test_static_assets(self):
        """验证 `test_static_assets` 对应的行为约定。"""
        status, data, headers = self.request("GET", "/static/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers.get("Content-Type", ""))
        self.assertIn(b"MiniGPT", data)

        status, data, headers = self.request("GET", "/static/style.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", headers.get("Content-Type", ""))

        status, data, headers = self.request("GET", "/static/vendor/chart.umd.min.js")
        self.assertEqual(status, 200)
        self.assertGreater(len(data), 100_000)
        self.assertIn(b"Chart.js v4.4.7", data)

        status, data, headers = self.request("GET", "/static/vendor/CHARTJS_LICENSE.md")
        self.assertEqual(status, 200)
        self.assertIn(b"MIT", data)

    def test_frontend_has_no_external_urls(self):
        """验证 `test_frontend_has_no_external_urls` 对应的行为约定。"""
        for name in ("index.html", "app.js", "style.css"):
            text = (PROJECT_ROOT / "web" / name).read_text(encoding="utf-8").lower()
            self.assertNotIn("http://", text, f"{name} 不应包含外部 URL")
            self.assertNotIn("https://", text, f"{name} 不应包含外部 URL")
            self.assertNotIn("//cdn.", text, f"{name} 不应引用 CDN")
        index_text = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("/static/vendor/chart.umd.min.js", index_text)

    def test_static_path_traversal_blocked(self):
        """验证 路径、目录穿越 相关行为。"""
        paths = [
            "/static/../_bootstrap.py",
            "/static/%2e%2e/%2e%2e/minigpt/config.py",
            "/static/..%5C..%5C_bootstrap.py",
            "/static/%2e%2e/README.md",
            "/static/raw%00name",
        ]
        for path in paths:
            status, data, _headers = self.request("GET", path)
            self.assertIn(status, (400, 404), f"{path} -> {status}")
            text = data.decode("utf-8", "replace")
            self.assertNotIn("PROJECT_ROOT", text)
            self.assertNotIn("def default_project_root", text)

    def test_favicon(self):
        """验证 `test_favicon` 对应的行为约定。"""
        status, data, _headers = self.request("GET", "/favicon.ico")
        self.assertEqual(status, 204)
        self.assertEqual(data, b"")

    # -- 配置 CRUD / 校验 --------------------------------------------------
    def test_config_crud(self):
        """验证 配置 相关行为。"""
        body = {
            "name": "ui-demo",
            "config": {
                "stage": "pretrain",
                "dataset": "shakespeare_en",
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
            },
        }
        status, payload = self.json_request("POST", "/api/v1/configs", body)
        self.assertEqual(status, 201)
        self.assertEqual(payload["name"], "ui-demo")
        self.assertTrue((self.configs_dir / "ui-demo.json").is_file())

        status, payload = self.json_request("GET", "/api/v1/configs")
        self.assertEqual(status, 200)
        self.assertIn("ui-demo", [item["name"] for item in payload["configs"]])

        status, payload = self.json_request("GET", "/api/v1/configs/ui-demo")
        self.assertEqual(status, 200)
        self.assertEqual(payload["name"], "ui-demo")
        self.assertEqual(payload["config"]["stage"], "pretrain")

        status, payload = self.json_request(
            "PUT",
            "/api/v1/configs/ui-demo",
            {"config": {"stage": "sft", "n_layer": 2, "n_head": 2, "n_embd": 64}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["updated"])
        status, payload = self.json_request("GET", "/api/v1/configs/ui-demo")
        self.assertEqual(payload["config"]["stage"], "sft")

        status, payload = self.json_request("DELETE", "/api/v1/configs/ui-demo")
        self.assertEqual(status, 200)
        self.assertTrue(payload["deleted"])
        status, payload = self.json_request("GET", "/api/v1/configs/ui-demo")
        self.assertEqual(status, 404)
        status, payload = self.json_request("DELETE", "/api/v1/configs/ui-demo")
        self.assertEqual(status, 404)

    def test_config_rejects_invalid(self):
        """验证 配置、非法输入 相关行为。"""
        status, payload = self.json_request(
            "POST", "/api/v1/configs", {"name": "bad-config", "config": {"n_embd": 7, "n_head": 4}}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertFalse((self.configs_dir / "bad-config.json").exists())

        for name in ("../evil", "a/b", ".hidden", ""):
            status, payload = self.json_request(
                "POST", "/api/v1/configs", {"name": name, "config": {"stage": "pretrain"}}
            )
            self.assertEqual(status, 400, f"name={name!r} 应被拒绝")

    def test_config_path_traversal_blocked(self):
        """验证 配置、路径、目录穿越 相关行为。"""
        for path in (
            "/api/v1/configs/..%2F..%2Fsecret",
            "/api/v1/configs/%2e%2e",
            "/api/v1/configs/%2e%2e%2f%2e%2e%2fsecret",
        ):
            status, payload = self.json_request("GET", path)
            self.assertIn(status, (400, 404), f"{path} -> {status}")

    def test_validate_endpoint(self):
        """验证 校验 相关行为。"""
        status, payload = self.json_request(
            "POST",
            "/api/v1/configs/validate",
            {
                "preset": "debug_en",
                "config": {
                    "stage": "pretrain",
                    "n_layer": 2,
                    "n_head": 2,
                    "n_embd": 64,
                    "max_steps": 10,
                    "eval_interval": 5,
                    "eval_iters": 2,
                    "log_interval": 1,
                    "save_interval": 5,
                    "batch_size": 1,
                    "block_size": 16,
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["errors"], [])
        self.assertTrue(Path(payload["resolved"]["runs_dir"]).is_absolute())

        status, payload = self.json_request(
            "POST", "/api/v1/configs/validate", {"config": {"n_embd": 7, "n_head": 4}}
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["errors"])

        status, payload = self.json_request(
            "POST", "/api/v1/configs/validate", {"config": {"bogus_field": 1}}
        )
        self.assertFalse(payload["valid"])
        self.assertIn("unknown", " ".join(payload["errors"]).lower())

        # 直接提交完整配置（不带 config 包装）也应可用。
        status, payload = self.json_request(
            "POST", "/api/v1/configs/validate", {"preset": "debug_en"}
        )
        self.assertTrue(payload["valid"])

    # -- 任务 --------------------------------------------------------------
    def test_jobs_start_list_get_stop(self):
        """验证 任务、启动、停止 相关行为。"""
        status, payload = self.json_request("POST", "/api/v1/jobs", {"preset": "debug_en"})
        self.assertEqual(status, 201)
        job = payload["job"]
        run_id = job["run_id"]
        self.assertRegex(run_id, RUN_ID_PATTERN)
        self.assertTrue(job["active"])
        self.assertEqual(job["status"], "running")
        frozen_path = self.runs_dir / run_id / "resolved_config.json"
        self.assertTrue(frozen_path.is_file())
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["run_name"], run_id)
        self.assertEqual(Path(frozen["runs_dir"]), self.runs_dir.resolve())

        status, payload = self.json_request("GET", "/api/v1/jobs")
        self.assertEqual(status, 200)
        self.assertTrue(any(job["run_id"] == run_id for job in payload["jobs"]))
        self.assertEqual(payload["active"], 1)
        self.assertEqual(payload["max_jobs"], 1)

        status, payload = self.json_request("GET", "/api/v1/jobs/%s" % run_id)
        self.assertEqual(status, 200)
        self.assertEqual(payload["job"]["run_id"], run_id)
        self.assertEqual(payload["source"], "memory")

        # 默认最多 1 个活动任务
        status, payload = self.json_request("POST", "/api/v1/jobs", {"preset": "debug_en"})
        self.assertEqual(status, 409)
        self.assertIn("error", payload)

        status, payload = self.json_request("POST", "/api/v1/jobs/%s/stop" % run_id)
        self.assertEqual(status, 200)
        self.assertTrue(payload["stopped"])
        self.assertFalse(payload["job"]["active"])
        self.assertEqual(payload["job"]["status"], "stopped")

        status, payload = self.json_request("GET", "/api/v1/jobs")
        self.assertEqual(payload["active"], 0)
        status, payload = self.json_request("POST", "/api/v1/jobs/%s/stop" % run_id)
        self.assertEqual(status, 200)
        self.assertFalse(payload["stopped"])

    def test_jobs_from_saved_config(self):
        """验证 任务、配置 相关行为。"""
        save_body = {
            "name": "job-cfg",
            "config": {
                "stage": "pretrain",
                "run_name": "from-config",
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
            },
        }
        status, _payload = self.json_request("POST", "/api/v1/configs", save_body)
        self.assertEqual(status, 201)
        try:
            status, payload = self.json_request("POST", "/api/v1/jobs", {"config_name": "job-cfg"})
            self.assertEqual(status, 201)
            self.assertEqual(payload["job"]["run_id"], "from-config")
            self.json_request("POST", "/api/v1/jobs/from-config/stop")
        finally:
            self.json_request("DELETE", "/api/v1/configs/job-cfg")

    def test_jobs_validation_errors(self):
        """验证 任务、校验 相关行为。"""
        status, payload = self.json_request("POST", "/api/v1/jobs", {"preset": "no-such-preset"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

        status, payload = self.json_request("POST", "/api/v1/jobs", {"bogus": 1})
        self.assertEqual(status, 400)

        status, _payload, _headers = self.request("POST", "/api/v1/jobs", raw_body="[,]")
        self.assertEqual(status, 400)

        status, _payload, _headers = self.request("POST", "/api/v1/jobs", raw_body="{not json")
        self.assertEqual(status, 400)

        status, payload = self.json_request("GET", "/api/v1/jobs/does-not-exist")
        self.assertEqual(status, 404)

    def test_jobs_disk_view(self):
        """验证 任务 相关行为。"""
        status, payload = self.json_request("GET", "/api/v1/jobs/run-completed")
        self.assertEqual(status, 200)
        self.assertEqual(payload["source"], "disk")
        self.assertEqual(payload["job"]["status"], "completed")
        self.assertFalse(payload["job"]["active"])

    # -- 运行历史 ----------------------------------------------------------
    def test_runs_list(self):
        """验证 `test_runs_list` 对应的行为约定。"""
        status, payload = self.json_request("GET", "/api/v1/runs")
        self.assertEqual(status, 200)
        ids = [run["run_id"] for run in payload["runs"]]
        self.assertIn("run-completed", ids)
        self.assertIn("run-live", ids)
        self.assertIn("run-minimal", ids)
        self.assertNotIn("bad run", ids)
        completed = next(run for run in payload["runs"] if run["run_id"] == "run-completed")
        self.assertEqual(completed["display_status"], "completed")
        self.assertEqual(completed["step"], 3000)
        self.assertEqual(completed["best_val"], 1.87)
        live = next(run for run in payload["runs"] if run["run_id"] == "run-live")
        self.assertEqual(live["display_status"], "running")
        minimal = next(run for run in payload["runs"] if run["run_id"] == "run-minimal")
        self.assertEqual(minimal["display_status"], "unknown")

    def test_run_summary(self):
        """验证 摘要 相关行为。"""
        status, payload = self.json_request("GET", "/api/v1/runs/run-completed/summary")
        self.assertEqual(status, 200)
        self.assertEqual(payload["run_id"], "run-completed")
        self.assertEqual(payload["display_status"], "completed")
        self.assertEqual(payload["status"]["step"], 3000)
        self.assertEqual(payload["config"]["dataset"], "shakespeare_en")
        self.assertEqual(payload["config_source"], "resolved_config.json")
        self.assertEqual(payload["environment"]["torch_version"], "2.15.0")
        self.assertEqual(payload["report"]["best_val_loss"], 1.87)
        self.assertTrue(payload["minirun_report"]["passed"])
        self.assertEqual(payload["minirun_run_id"], "run-completed-mini")
        self.assertEqual(payload["metrics"]["train_loss"], 2.01)
        self.assertEqual(payload["metrics_count"], 2)
        self.assertEqual(len(payload["samples"]), 2)
        self.assertEqual(payload["sample_count"], 2)
        self.assertGreater(len(payload["events"]), 0)
        self.assertIn("console line one", payload["console_log"])
        self.assertTrue(payload["files"]["status.json"])
        self.assertTrue(payload["files"]["events.jsonl"])
        self.assertTrue(payload["files"]["metrics.csv"])
        self.assertFalse(payload["active"])
        self.assertIsNone(payload["job"] or None)
        json.dumps(payload)  # 必须可严格序列化（不出现 NaN）

    def test_run_metrics(self):
        """验证 指标 相关行为。"""
        status, payload = self.json_request("GET", "/api/v1/runs/run-completed/metrics")
        self.assertEqual(status, 200)
        self.assertIn("val_loss", payload["columns"])
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["rows"][0]["step"], 200)
        self.assertAlmostEqual(payload["rows"][1]["val_ppl"], 7.0297)

    def test_run_events_pagination(self):
        """验证 事件流 相关行为。"""
        status, payload = self.json_request("GET", "/api/v1/runs/run-completed/events?after=2&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["after"], 2)
        self.assertEqual(payload["next_after"], 4)
        self.assertEqual(payload["total"], 6)
        self.assertEqual(len(payload["events"]), 2)
        self.assertEqual(payload["events"][0]["event"], "eval")

        status, payload = self.json_request("GET", "/api/v1/runs/run-completed/events?after=abc")
        self.assertEqual(status, 400)

    def test_run_samples(self):
        """验证 `test_run_samples` 对应的行为约定。"""
        status, payload = self.json_request("GET", "/api/v1/runs/run-completed/samples?limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 1)
        self.assertIn("ROMEO", payload["samples"][0]["text"])

    def test_run_unknown_and_traversal(self):
        """验证 目录穿越 相关行为。"""
        status, payload = self.json_request("GET", "/api/v1/runs/no-such-run/summary")
        self.assertEqual(status, 404)

        for path in (
            "/api/v1/runs/..%2F..%2Fsecret/summary",
            "/api/v1/runs/%2e%2e%2fsummary",
        ):
            status, payload = self.json_request("GET", path)
            self.assertIn(status, (400, 404), f"{path} -> {status}")

        status, payload = self.json_request("GET", "/api/v1/runs/run-completed/unknown-action")
        self.assertEqual(status, 404)

        # 裸运行路径只支持 DELETE（删除运行），GET 属于方法不允许。
        status, _data, headers = self.request("GET", "/api/v1/runs/run-completed")
        self.assertEqual(status, 405)
        self.assertIn("DELETE", headers.get("Allow", ""))

    def test_run_minimal_summary(self):
        """验证 摘要 相关行为。"""
        status, payload = self.json_request("GET", "/api/v1/runs/run-minimal/summary")
        self.assertEqual(status, 200)
        self.assertEqual(payload["display_status"], "unknown")
        self.assertEqual(payload["config"]["stage"], "pretrain")
        self.assertTrue(payload["files"]["resolved_config.json"])
        self.assertFalse(payload["files"]["status.json"])

    # -- 任务生命周期：新建 / 启动 / 删除 ---------------------------------
    def test_create_start_delete_task_lifecycle(self):
        """验证 创建、启动、删除、生命周期 相关行为。"""
        status, payload = self.json_request(
            "POST",
            "/api/v1/runs",
            {"preset": "debug_en", "run_name": "task-lifecycle", "stage": "pretrain"},
        )
        self.assertEqual(status, 201, payload)
        run_id = payload["job"]["run_id"]
        run_dir = self.runs_dir / run_id
        self.assertTrue((run_dir / "resolved_config.json").is_file())
        created = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(created["status"], "created")
        self.assertFalse(payload["job"]["active"])

        status, payload = self.json_request("POST", f"/api/v1/runs/{run_id}/start")
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["job"]["active"])

        # 活动任务不能删；停止之后可以删。
        status, payload = self.json_request("DELETE", f"/api/v1/runs/{run_id}")
        self.assertEqual(status, 409)
        self.assertIn("活动", payload["error"])

        status, payload = self.json_request("POST", f"/api/v1/jobs/{run_id}/stop")
        self.assertEqual(status, 200, payload)

        status, payload = self.json_request("DELETE", f"/api/v1/runs/{run_id}")
        self.assertEqual(status, 200, payload)
        self.assertFalse(run_dir.exists())
        status, payload = self.json_request("GET", f"/api/v1/jobs/{run_id}")
        self.assertEqual(status, 404)

    def test_force_delete_stale_running_run(self):
        # 进程已死但 status.json 还写着 running：普通删除 409，force 可删。
        """验证 删除 相关行为。"""
        stale = self.runs_dir / "run-stale"
        stale.mkdir(parents=True, exist_ok=True)
        write_json(stale / "status.json", {"status": "running", "updated_at": "2020-01-01T00:00:00+00:00"})
        status, payload = self.json_request("DELETE", "/api/v1/runs/run-stale")
        self.assertEqual(status, 409)
        self.assertIn("心跳", payload["error"])

        status, payload = self.json_request("DELETE", "/api/v1/runs/run-stale?force=1")
        self.assertEqual(status, 200, payload)
        self.assertFalse(stale.exists())

    def test_force_delete_recent_running_refused(self):
        """验证 删除 相关行为。"""
        from minigpt.web.server import iso_now

        fresh = self.runs_dir / "run-fresh"
        fresh.mkdir(parents=True, exist_ok=True)
        write_json(fresh / "status.json", {"status": "running", "updated_at": iso_now()})
        status, payload = self.json_request("DELETE", "/api/v1/runs/run-fresh?force=1")
        self.assertEqual(status, 409)
        self.assertTrue(fresh.is_dir())

    def test_delete_run_refuses_active_status_and_missing(self):
        """验证 删除、状态文件 相关行为。"""
        status, payload = self.json_request("DELETE", "/api/v1/runs/run-live")
        self.assertEqual(status, 409, payload)

        status, payload = self.json_request("DELETE", "/api/v1/runs/no-such-run")
        self.assertEqual(status, 404)

        status, payload = self.json_request("DELETE", "/api/v1/runs/..%2F..%2Fsecret")
        self.assertIn(status, (400, 404))

    def test_delete_finished_run_removes_directory(self):
        """验证 删除 相关行为。"""
        run_dir = self.runs_dir / "run-deletable"
        make_run(self.runs_dir, "run-deletable", status="stopped")
        self.assertTrue(run_dir.is_dir())
        status, payload = self.json_request("DELETE", "/api/v1/runs/run-deletable")
        self.assertEqual(status, 200, payload)
        self.assertFalse(run_dir.exists())

    # -- 路由 / 方法 / 错误 ------------------------------------------------
    def test_unknown_paths(self):
        """验证 `test_unknown_paths` 对应的行为约定。"""
        status, payload = self.json_request("GET", "/api/v1/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

        status, _data, _headers = self.request("GET", "/nope")
        self.assertEqual(status, 404)

    def test_method_not_allowed(self):
        """验证 `test_method_not_allowed` 对应的行为约定。"""
        status, _data, headers = self.request("DELETE", "/api/v1/health")
        self.assertEqual(status, 405)
        self.assertIn("GET", headers.get("Allow", ""))

        status, _data, _headers = self.request("PUT", "/api/v1/runs")
        self.assertEqual(status, 405)

        # 已知路由 + 未允许的方法必须 405（不是 501）
        status, _data, headers = self.request("PATCH", "/api/v1/jobs")
        self.assertEqual(status, 405)
        self.assertIn("GET", headers.get("Allow", ""))
        self.assertIn("POST", headers.get("Allow", ""))

        status, _data, _headers = self.request("PATCH", "/api/v1/health")
        self.assertEqual(status, 405)

    def test_no_nan_in_json(self):
        """训练文件里可能有 nan；任何 API 响应都必须是严格 JSON。"""

        nan_dir = self.runs_dir / "run-nan"
        nan_dir.mkdir(parents=True, exist_ok=True)
        write_json(nan_dir / "resolved_config.json", {"stage": "pretrain", "run_name": "run-nan"})
        write_json(nan_dir / "status.json", {"status": "running", "step": 1, "train_loss": float("nan")})
        (nan_dir / "metrics.csv").write_text(
            METRIC_HEADER + "\n1,nan,nan,nan,nan,nan,nan,nan,nan,nan\n", encoding="utf-8"
        )
        status, data, _headers = self.request("GET", "/api/v1/runs/run-nan/summary")
        self.assertEqual(status, 200)
        text = data.decode("utf-8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        json.loads(text)  # 严格 JSON 解析

        status, data, _headers = self.request("GET", "/api/v1/runs/run-nan/metrics")
        self.assertEqual(status, 200)
        self.assertNotIn("NaN", data.decode("utf-8"))
        shutil.rmtree(nan_dir, ignore_errors=True)


class DashboardCliTest(unittest.TestCase):
    """dashboard.py 命令行冒烟测试（端口 0，不占用固定端口）。"""

    CHILD_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    def test_help(self):
        """验证 帮助信息 相关行为。"""
        completed = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "dashboard.py"), "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=self.CHILD_ENV,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage", (completed.stdout or "").lower())
        for flag in ("--host", "--port", "--runs-dir", "--web-dir", "--configs-dir", "--project-root"):
            self.assertIn(flag, completed.stdout or "")

    def test_missing_python_returns_error(self):
        """验证 错误处理 相关行为。"""
        completed = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "dashboard.py"), "--port", "0", "--python", "no-such-python"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=self.CHILD_ENV,
        )
        self.assertEqual(completed.returncode, 2)

    def test_port_zero_end_to_end(self):
        """验证 `test_port_zero_end_to_end` 对应的行为约定。"""
        with tempfile.TemporaryDirectory(prefix="minigpt-cli-") as tmp:
            root = Path(tmp)
            process = subprocess.Popen(
                [
                    PYTHON,
                    "-u",
                    str(PROJECT_ROOT / "dashboard.py"),
                    "--port",
                    "0",
                    "--runs-dir",
                    str(root / "runs"),
                    "--configs-dir",
                    str(root / "configs"),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.CHILD_ENV,
            )
            try:
                output_lines = _read_until_url(process, timeout=40)
                url_line = next(
                    (line for line in output_lines if re.search(r"http://127\.0\.0\.1:(\d+)/", line)),
                    "",
                )
                self.assertTrue(url_line, "dashboard.py 应打印服务 URL；输出: " + "".join(output_lines))
                match = re.search(r"http://127\.0\.0\.1:(\d+)/", url_line)
                port = int(match.group(1))
                self.assertGreater(port, 0)
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/health", timeout=15
                ) as response:
                    health = json.loads(response.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=15) as response:
                    html = response.read().decode("utf-8", "replace")
                self.assertIn("MiniGPT", html)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                if process.stdout is not None:
                    process.stdout.close()


def _read_until_url(process: subprocess.Popen, timeout: float = 30.0) -> list:
    """在后台线程逐行读取子进程 stdout，直到出现服务 URL 或超时。"""

    lines: list = []
    result: "queue.Queue" = queue.Queue()

    def reader() -> None:
        try:
            if process.stdout is None:
                result.put(None)
                return
            for line in process.stdout:
                result.put(line)
        except Exception:  # noqa: BLE001
            result.put(None)
        finally:
            try:
                result.put(None)
            except Exception:  # noqa: BLE001
                pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = result.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            break
        if line is None:
            break
        lines.append(line)
        if re.search(r"http://127\.0\.0\.1:(\d+)/", line):
            break
    return lines


if __name__ == "__main__":
    unittest.main()
