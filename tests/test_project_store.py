"""项目级管理：ProjectStore 并发/校验 + projects API 聚合。

成员关系唯一权威在 projects/<id>/project.json；run 摘要的 project_id
反查存储；详情聚合成员运行、评测、导出与数据指纹。
"""

from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.training.job_manager import ConfigError, JobNotFound  # noqa: E402
from minigpt.training.project_store import ProjectStore  # noqa: E402
from minigpt.web.server import create_server  # noqa: E402

from tests.test_web import make_run, write_json  # noqa: E402


class ProjectStoreTests(unittest.TestCase):
    """覆盖 ProjectStoreTests 相关行为。"""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-proj-")
        self.root = Path(self._tmp.name)
        self.store = ProjectStore(self.root / "projects", project_root=self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_load_list(self):
        """验证 创建 相关行为。"""
        created = self.store.create("英文流水线", note="demo")
        self.assertEqual(created["name"], "英文流水线")
        self.assertEqual(created["run_ids"], [])
        loaded = self.store.load(created["project_id"])
        self.assertEqual(loaded["note"], "demo")
        items = self.store.list()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["project_id"], created["project_id"])

    def test_duplicate_and_invalid_rejected(self):
        """验证 非法输入 相关行为。"""
        self.store.create("p1", project_id="proj-a")
        with self.assertRaises(ConfigError):
            self.store.create("p2", project_id="proj-a")
        with self.assertRaises(ConfigError):
            self.store.create("p3", project_id="../escape")
        with self.assertRaises(ConfigError):
            self.store.create("   ")
        with self.assertRaises(JobNotFound):
            self.store.load("missing")

    def test_attach_idempotent_and_ordered(self):
        """验证 `test_attach_idempotent_and_ordered` 对应的行为约定。"""
        self.store.create("p", project_id="proj-b")
        self.store.attach_run("proj-b", "run-1")
        self.store.attach_run("proj-b", "run-2")
        self.store.attach_run("proj-b", "run-1")
        self.assertEqual(self.store.load("proj-b")["run_ids"], ["run-1", "run-2"])
        self.assertEqual(self.store.project_of("run-2"), "proj-b")
        self.assertIsNone(self.store.project_of("run-x"))

    def test_concurrent_attach_keeps_all_members(self):
        """验证 `test_concurrent_attach_keeps_all_members` 对应的行为约定。"""
        self.store.create("p", project_id="proj-c")
        barrier = threading.Barrier(8, timeout=10)

        def attach(index: int) -> None:
            barrier.wait()
            self.store.attach_run("proj-c", f"run-{index}")

        threads = [threading.Thread(target=attach, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        runs = self.store.load("proj-c")["run_ids"]
        self.assertEqual(sorted(runs), [f"run-{i}" for i in range(8)])


class ProjectsApiTests(unittest.TestCase):
    """覆盖 ProjectsApiTests 相关行为。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="minigpt-projapi-")
        root = Path(cls._tmp.name)
        cls.runs_dir = root / "runs"
        cls.runs_dir.mkdir(parents=True)
        cls.dist_dir = root / "dist"
        cls.dist_dir.mkdir(parents=True)
        make_run(cls.runs_dir, "pre-1", status="completed")
        make_run(cls.runs_dir, "sft-1", status="completed")
        write_json(cls.runs_dir / "pre-1" / "training_report.json", {"data_fingerprint": {"train.bin": "abc"}})
        write_json(cls.runs_dir / "pre-1" / "eval_test" / "evaluation.json", {"perplexity": 9.0})
        bundle = cls.dist_dir / "bundle-1"
        bundle.mkdir(parents=True)
        write_json(bundle / "MANIFEST.json", {"source": {"run_id": "sft-1"}, "files": {}})
        cls.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=root,
            runs_dir=cls.runs_dir,
            web_dir=PROJECT_ROOT / "web",
            configs_dir=root / "configs",
            projects_dir=root / "projects",
            dist_dir=cls.dist_dir,
        )
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=10)
        cls.server.server_close()
        cls._tmp.cleanup()

    def json_request(self, method: str, path: str, body=None):
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
        except (ValueError, UnicodeDecodeError):
            parsed = None
        return status, parsed

    def test_project_lifecycle_and_aggregation(self):
        """验证 项目、生命周期 相关行为。"""
        status, payload = self.json_request("POST", "/api/v1/projects", {"name": "英文流水线", "project_id": "proj-en"})
        self.assertEqual(status, 201, payload)
        status, payload = self.json_request("GET", "/api/v1/projects")
        self.assertEqual(status, 200)
        self.assertIn("proj-en", [item["project_id"] for item in payload["projects"]])
        self.assertEqual(payload["count"], len(payload["projects"]))

        for run_id in ("pre-1", "sft-1"):
            status, payload = self.json_request("POST", "/api/v1/projects/proj-en/runs", {"run_id": run_id})
            self.assertEqual(status, 200, payload)

        status, payload = self.json_request("GET", "/api/v1/projects/proj-en")
        self.assertEqual(status, 200)
        self.assertEqual([run["run_id"] for run in payload["runs"]], ["pre-1", "sft-1"])
        self.assertEqual(payload["data_fingerprints"]["pre-1"]["train.bin"], "abc")
        self.assertTrue(any(entry["run_id"] == "pre-1" for entry in payload["evaluations"]))
        self.assertEqual([entry["name"] for entry in payload["exports"]], ["bundle-1"])

        status, payload = self.json_request("GET", "/api/v1/runs")
        by_id = {run["run_id"]: run for run in payload["runs"]}
        self.assertEqual(by_id["pre-1"]["project_id"], "proj-en")
        self.assertEqual(by_id["sft-1"]["project_id"], "proj-en")

    def test_error_paths(self):
        """验证 错误处理 相关行为。"""
        status, _ = self.json_request("GET", "/api/v1/projects/missing")
        self.assertEqual(status, 404)
        status, _ = self.json_request("POST", "/api/v1/projects", {"name": ""})
        self.assertEqual(status, 400)
        self.json_request("POST", "/api/v1/projects", {"name": "p", "project_id": "proj-err"})
        status, _ = self.json_request("POST", "/api/v1/projects/proj-err/runs", {"run_id": "no-such-run"})
        self.assertEqual(status, 404)
        status, _ = self.json_request("POST", "/api/v1/projects/proj-err/runs", {})
        self.assertEqual(status, 400)


class LineageTests(unittest.TestCase):
    """血缘字段：服务端推导 parent_run_id、项目校验与自动挂载。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-lineage-")
        self.root = Path(self._tmp.name)
        (self.root / "pretrain.py").write_text("# stub\n", encoding="utf-8")
        (self.root / "sft.py").write_text("# stub\n", encoding="utf-8")
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir()
        (self.runs_dir / "parent-run").mkdir()
        ((self.runs_dir / "parent-run") / "best.mgckpt").write_bytes(b"x")
        from minigpt.training.job_manager import JobManager

        self.manager = JobManager(
            project_root=self.root,
            runs_dir=self.runs_dir,
            command_builder=lambda config, root, cfg: [sys.executable, "-c", "pass"],
        )
        self.store = ProjectStore(self.root / "projects", project_root=self.root)

    def tearDown(self):
        self.manager.shutdown()
        self._tmp.cleanup()

    def test_parent_derived_from_init_from(self):
        """验证 `test_parent_derived_from_init_from` 对应的行为约定。"""
        record = self.manager.create_run(
            config={"stage": "sft", "init_from": "runs/parent-run/best.mgckpt", "parent_run_id": "spoofed"},
            run_name="child-run",
        )
        frozen = json.loads((Path(record.run_dir) / "resolved_config.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen["parent_run_id"], "parent-run")

    def test_unknown_project_rejected(self):
        """验证 项目 相关行为。"""
        from minigpt.training.job_manager import ConfigError

        with self.assertRaises(ConfigError):
            self.manager.create_run(config={"stage": "pretrain", "project_id": "no-such"}, run_name="x-run")

    def test_project_auto_attach_and_listing_fields(self):
        """验证 项目 相关行为。"""
        from minigpt.web.server import DashboardApp

        self.store.create("p", project_id="proj-line")
        app = DashboardApp(
            project_root=self.root,
            runs_dir=self.runs_dir,
            web_dir=PROJECT_ROOT / "web",
            projects_dir=self.root / "projects",
            command_builder=lambda config, root, cfg: [sys.executable, "-c", "pass"],
        )
        try:
            app.create_run({"config": {"stage": "pretrain", "project_id": "proj-line"}, "run_name": "member-run"})
            self.assertEqual(self.store.load("proj-line")["run_ids"], ["member-run"])
            listing = app._run_listing(self.runs_dir / "member-run")
            self.assertEqual(listing["project_id"], "proj-line")
            self.assertIsNone(listing["parent_run_id"])
            self.assertIsNone(listing["best_checkpoint"])
            parent_listing = app._run_listing(self.runs_dir / "parent-run")
            self.assertEqual(parent_listing["best_checkpoint"], "best.mgckpt")
            self.assertTrue(parent_listing["has_best"])
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
