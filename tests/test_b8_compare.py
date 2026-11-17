"""B8-N07 多 run 对比：GET /api/v1/compare 聚合 + 前端纯函数（node）+ DOM 契约。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.web.server import create_server  # noqa: E402
from tests.test_web import make_run  # noqa: E402

ROOT = PROJECT_ROOT


class CompareApiTests(unittest.TestCase):
    """真实 HTTP 服务（端口 0）+ 临时 runs 目录。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-compare-")
        root = Path(cls._tmp.name)
        cls.runs_dir = root / "runs"
        cls.runs_dir.mkdir(parents=True)
        make_run(cls.runs_dir, "run-a", status="completed", step=3000)
        make_run(cls.runs_dir, "run-b", status="completed", step=3000)
        make_run(cls.runs_dir, "run-noreport", with_report=False)
        cls.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=PROJECT_ROOT,
            runs_dir=cls.runs_dir,
            web_dir=ROOT / "web",
            configs_dir=root / "configs",
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

    def get(self, path: str):
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            data = response.read()
            status = response.status
        finally:
            connection.close()
        try:
            payload = json.loads(data.decode("utf-8"))
        except ValueError:
            payload = None
        return status, payload

    def test_compare_aggregates_metrics_and_report(self):
        """验证 对比、指标、报告 相关行为。"""
        status, payload = self.get("/api/v1/compare?runs=run-a,run-b")
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 2)
        self.assertEqual([item["run_id"] for item in payload["runs"]], ["run-a", "run-b"])
        first = payload["runs"][0]
        self.assertEqual(first["series"]["step"], [200, 3000])
        self.assertEqual(first["series"]["train_loss"], [3.9, 2.01])
        self.assertEqual(first["series"]["val_loss"], [2.9, 1.95])
        self.assertAlmostEqual(first["series"]["val_ppl"][1], 7.0297, places=4)
        self.assertEqual(first["metrics_count"], 2)
        self.assertEqual(first["stage"], "pretrain")
        self.assertEqual(first["display_status"], "completed")
        self.assertEqual(first["report"]["best_val_loss"], 1.87)
        self.assertEqual(first["report"]["best_step"], 2800)
        self.assertEqual(first["report"]["steps"], 3000)

    def test_compare_dedupes_and_keeps_order(self):
        """验证 对比 相关行为。"""
        status, payload = self.get("/api/v1/compare?runs=run-b,run-a,run-b")
        self.assertEqual(status, 200)
        self.assertEqual([item["run_id"] for item in payload["runs"]], ["run-b", "run-a"])

    def test_compare_run_without_report(self):
        """验证 对比、报告 相关行为。"""
        status, payload = self.get("/api/v1/compare?runs=run-noreport")
        self.assertEqual(status, 200)
        self.assertIsNone(payload["runs"][0]["report"])
        self.assertEqual(payload["runs"][0]["metrics_count"], 2)

    def test_compare_missing_or_empty_runs_rejected(self):
        """验证 对比 相关行为。"""
        for path in ("/api/v1/compare", "/api/v1/compare?runs=", "/api/v1/compare?runs=,,"):
            status, payload = self.get(path)
            self.assertEqual(status, 400, path)
            self.assertIn("error", payload)

    def test_compare_invalid_run_id_rejected(self):
        """验证 对比、非法输入 相关行为。"""
        status, payload = self.get("/api/v1/compare?runs=run-a,bad%20run")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        status, _ = self.get("/api/v1/compare?runs=..%2F..%2Fetc")
        self.assertEqual(status, 400)

    def test_compare_unknown_run_404(self):
        """验证 对比 相关行为。"""
        status, payload = self.get("/api/v1/compare?runs=run-a,missing-run")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_compare_too_many_runs_rejected(self):
        """验证 对比 相关行为。"""
        runs = ",".join(f"run-a" for _ in range(2))  # 去重后不超限
        status, _ = self.get(f"/api/v1/compare?runs={runs}")
        self.assertEqual(status, 200)
        many = ",".join(f"r{i}" for i in range(9))
        status, payload = self.get(f"/api/v1/compare?runs={many}")
        self.assertEqual(status, 400)

    def test_compare_method_not_allowed(self):
        """验证 对比 相关行为。"""
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            connection.request(
                "POST", "/api/v1/compare", body=b"{}", headers={"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 405)
            response.read()
        finally:
            connection.close()


class CompareCoreNodeTests(unittest.TestCase):
    """前端纯函数（web/js/compare-core.js）用 node 直接单测。"""

    def test_compare_core_pure_functions(self):
        """验证 对比、纯后端 相关行为。"""
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        script = ROOT / "tests" / "compare_core_check.mjs"
        proc = subprocess.run(
            [node, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(result["ok"])
        self.assertEqual(result["lossDatasetCount"], 4)  # 2 run × (train+val)
        self.assertEqual(result["pplDatasetCount"], 2)
        self.assertEqual(result["skippedNull"], 1)
        self.assertEqual(result["summaryHeaders"], ["指标", "run-a", "run-b"])


class CompareDomTests(unittest.TestCase):
    """覆盖 CompareDomTests 相关行为。"""
    def test_compare_tab_and_view_wired(self):
        """验证 对比 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        router = (ROOT / "web" / "js" / "router.js").read_text(encoding="utf-8")
        for marker in (
            'href="#/compare"',
            'data-route="compare"',
            'id="view-compare"',
            'data-view="compare"',
            'id="compare-run-list"',
            'id="compare-loss-chart"',
            'id="compare-ppl-chart"',
            'id="compare-summary"',
            'id="compare-refresh-btn"',
        ):
            self.assertIn(marker, index)
        self.assertIn('from "./js/compare.js"', app)
        self.assertIn("initCompareView()", app)
        self.assertLess(app.index("initCompareView()"), app.index("initRouter()"))
        self.assertIn("compare:", router)
        compare = (ROOT / "web" / "js" / "compare.js").read_text(encoding="utf-8")
        self.assertIn("/api/v1/compare", compare)
        self.assertIn("spa:view", compare)
        self.assertIn("setInterval", compare)
        self.assertIn("textContent", compare)


if __name__ == "__main__":
    unittest.main()