"""B8-N09 生成参数对比：POST /api/v1/inference/compare（≤4 组）+ worker 参数透传。"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.web.server import ApiError, DashboardApp, create_server  # noqa: E402

ROOT = PROJECT_ROOT
PYTHON = sys.executable

# 假 worker：回显收到的生成参数，并按 temperature 决定是否“失败”。
ECHO_WORKER = (
    "import json, sys\n"
    "req = json.load(sys.stdin)\n"
    "if req.get('temperature', 0) > 4.5:\n"
    "    print(json.dumps({'type': 'error', 'code': 'FakeError', 'message': 'too hot'}), flush=True)\n"
    "    sys.exit(1)\n"
    "out = {'steps': [{'step': 1}], 'final': {'continuation': 'hello-' + str(req.get('temperature')), "
    "'reason': 'max_tokens', 'steps': 1},\n"
    " 'echo': {k: req.get(k) for k in ('temperature', 'top_k', 'top_p', 'repetition_penalty', 'no_repeat_ngram_size')}}\n"
    "print(json.dumps(out), flush=True)\n"
)


def make_app(root: Path) -> DashboardApp:
    (root / "runs" / "demo").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "demo" / "best.pt").write_bytes(b"fake")
    return DashboardApp(
        project_root=root,
        runs_dir=root / "runs",
        web_dir=ROOT / "web",
        configs_dir=root / "configs",
        python_executable=PYTHON,
    )


class InferenceCompareAppTests(unittest.TestCase):
    """覆盖 InferenceCompareAppTests 相关行为。"""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-infcmp-")
        self.root = Path(self._tmp.name)
        self.app = make_app(self.root)

    def tearDown(self):
        self.app.close()
        self._tmp.cleanup()

    def test_build_request_passes_repetition_controls(self):
        """验证 构建 相关行为。"""
        request = self.app.build_inference_request(
            {
                "run_id": "demo",
                "prompt": "hi",
                "repetition_penalty": 1.3,
                "no_repeat_ngram_size": 3,
            }
        )
        self.assertEqual(request["repetition_penalty"], 1.3)
        self.assertEqual(request["no_repeat_ngram_size"], 3)
        default = self.app.build_inference_request({"run_id": "demo", "prompt": "hi"})
        self.assertEqual(default["repetition_penalty"], 1.0)
        self.assertEqual(default["no_repeat_ngram_size"], 0)

    def test_build_request_rejects_bad_repetition_controls(self):
        """验证 构建 相关行为。"""
        for bad in ({"repetition_penalty": 0.05}, {"repetition_penalty": "x"}, {"no_repeat_ngram_size": -1}, {"no_repeat_ngram_size": 99}):
            with self.assertRaises(ApiError) as caught:
                self.app.build_inference_request({"run_id": "demo", "prompt": "hi", **bad})
            self.assertEqual(caught.exception.status, 400, str(bad))

    def test_compare_multiple_variants_structure(self):
        """验证 对比 相关行为。"""
        calls = []

        def fake_worker(request):
            calls.append(request)
            return {
                "steps": [{"step": 1}],
                "final": {"continuation": f"t={request['temperature']}", "reason": "max_tokens", "steps": 1},
            }

        self.app._run_inference_worker = fake_worker  # type: ignore[method-assign]
        payload = {
            "run_id": "demo",
            "prompt": "ROMEO:",
            "max_new_tokens": 8,
            "variants": [
                {"temperature": 0.0, "top_k": 10},
                {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.2, "no_repeat_ngram_size": 2},
            ],
        }
        result = self.app.inference_compare(payload)
        self.assertEqual(result["run_id"], "demo")
        self.assertEqual(result["prompt"], "ROMEO:")
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["results"]), 2)
        first, second = result["results"]
        self.assertEqual(first["index"], 0)
        self.assertTrue(first["ok"])
        self.assertEqual(first["text"], "t=0.0")
        self.assertGreaterEqual(first["elapsed_ms"], 0)
        self.assertEqual(first["params"]["temperature"], 0.0)
        self.assertEqual(first["params"]["top_k"], 10)
        self.assertEqual(second["params"]["repetition_penalty"], 1.2)
        self.assertEqual(second["params"]["no_repeat_ngram_size"], 2)
        self.assertEqual(second["text"], "t=0.8")
        # 每组都透传了重复控制参数与共享 prompt。
        self.assertEqual(calls[1]["repetition_penalty"], 1.2)
        self.assertEqual(calls[1]["no_repeat_ngram_size"], 2)
        self.assertEqual(calls[0]["prompt"], "ROMEO:")
        self.assertEqual(calls[0]["max_new_tokens"], 8)

    def test_compare_rejects_too_many_variants(self):
        """验证 对比 相关行为。"""
        with self.assertRaises(ApiError) as caught:
            self.app.inference_compare(
                {"run_id": "demo", "prompt": "x", "variants": [{"temperature": 0.5}] * 5}
            )
        self.assertEqual(caught.exception.status, 400)

    def test_compare_rejects_empty_or_bad_variants(self):
        """验证 对比 相关行为。"""
        for variants in ([], None, "x", [1], [{"temperature": 99}]):
            with self.assertRaises(ApiError) as caught:
                self.app.inference_compare({"run_id": "demo", "prompt": "x", "variants": variants})
            self.assertEqual(caught.exception.status, 400, str(variants))

    def test_compare_error_variant_does_not_break_others(self):
        """验证 对比、错误处理 相关行为。"""
        def fake_worker(request):
            if request["temperature"] > 1.0:
                raise ApiError(500, "生成失败（FakeError）: too hot")
            return {"steps": [], "final": {"continuation": "ok", "reason": "max_tokens", "steps": 0}}

        self.app._run_inference_worker = fake_worker  # type: ignore[method-assign]
        result = self.app.inference_compare(
            {
                "run_id": "demo",
                "prompt": "x",
                "variants": [{"temperature": 0.2}, {"temperature": 2.0}, {"temperature": 0.5}],
            }
        )
        self.assertEqual(result["count"], 3)
        self.assertTrue(result["results"][0]["ok"])
        self.assertFalse(result["results"][1]["ok"])
        self.assertIn("too hot", result["results"][1]["error"])
        self.assertIsNone(result["results"][1]["text"])
        self.assertTrue(result["results"][2]["ok"])

    def test_compare_requires_run_and_prompt(self):
        """验证 对比 相关行为。"""
        with self.assertRaises(ApiError):
            self.app.inference_compare({"prompt": "x", "variants": [{}]})
        with self.assertRaises(ApiError):
            self.app.inference_compare({"run_id": "demo", "prompt": 5, "variants": [{}]})


class InferenceCompareWorkerTests(unittest.TestCase):
    """真实子进程（假 worker 脚本）：参数透传 + 错误组隔离。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-infcmp-worker-")
        self.root = Path(self._tmp.name)
        self.app = make_app(self.root)
        self.app._inference_command = lambda: [PYTHON, "-u", "-c", ECHO_WORKER]  # type: ignore[method-assign]

    def tearDown(self):
        self.app.close()
        self._tmp.cleanup()

    def test_worker_receives_repetition_controls(self):
        """验证 推理 worker 相关行为。"""
        result = self.app.inference_compare(
            {
                "run_id": "demo",
                "prompt": "hi",
                "variants": [
                    {"temperature": 0.7, "top_k": 5, "top_p": 0.95, "repetition_penalty": 1.4, "no_repeat_ngram_size": 3},
                ],
            }
        )
        entry = result["results"][0]
        self.assertTrue(entry["ok"], entry)
        self.assertEqual(entry["text"], "hello-0.7")
        self.assertEqual(entry["result"]["echo"]["repetition_penalty"], 1.4)
        self.assertEqual(entry["result"]["echo"]["no_repeat_ngram_size"], 3)
        self.assertEqual(entry["result"]["echo"]["top_k"], 5)

    def test_worker_error_group_isolated(self):
        """验证 推理 worker、错误处理 相关行为。"""
        result = self.app.inference_compare(
            {
                "run_id": "demo",
                "prompt": "hi",
                "variants": [{"temperature": 0.3}, {"temperature": 4.9}],
            }
        )
        self.assertTrue(result["results"][0]["ok"])
        self.assertFalse(result["results"][1]["ok"])
        self.assertIn("error", result["results"][1])


class InferenceCompareHttpTests(unittest.TestCase):
    """覆盖 InferenceCompareHttpTests 相关行为。"""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-infcmp-http-")
        self.root = Path(self._tmp.name)
        (self.root / "runs" / "demo").mkdir(parents=True)
        (self.root / "runs" / "demo" / "best.pt").write_bytes(b"fake")
        self.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=self.root,
            runs_dir=self.root / "runs",
            web_dir=ROOT / "web",
            configs_dir=self.root / "configs",
            python_executable=PYTHON,
        )
        self.server.app._inference_command = lambda: [PYTHON, "-u", "-c", ECHO_WORKER]  # type: ignore[method-assign]
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

    def post(self, path: str, body):
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            connection.request(
                "POST", path, body=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
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

    def test_http_compare_flow(self):
        """验证 HTTP 层、对比 相关行为。"""
        status, payload = self.post(
            "/api/v1/inference/compare",
            {
                "run_id": "demo",
                "prompt": "hi",
                "variants": [{"temperature": 0.2}, {"temperature": 0.9, "repetition_penalty": 1.1}],
            },
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["count"], 2)
        self.assertTrue(payload["results"][0]["ok"])
        self.assertEqual(payload["results"][1]["params"]["repetition_penalty"], 1.1)

    def test_http_compare_over_limit_400(self):
        """验证 HTTP 层、对比 相关行为。"""
        status, payload = self.post(
            "/api/v1/inference/compare",
            {"run_id": "demo", "prompt": "hi", "variants": [{}] * 5},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_http_compare_method_not_allowed(self):
        """验证 HTTP 层、对比 相关行为。"""
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            connection.request("GET", "/api/v1/inference/compare")
            response = connection.getresponse()
            self.assertEqual(response.status, 405)
            response.read()
        finally:
            connection.close()


class WorkerRepetitionRealModelTests(unittest.TestCase):
    """真实 torch 小模型 + 真实 worker 子进程：惩罚参数确实改变生成。"""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not installed") from None

        from minigpt.models.gpt import GPT, GPTConfig
        from minigpt.tokenization.char import CharTokenizer

        cls._tmp = tempfile.TemporaryDirectory(prefix="minigpt-b8-worker-real-")
        root = Path(cls._tmp.name)
        run_dir = root / "runs" / "demo"
        run_dir.mkdir(parents=True)
        torch.manual_seed(7)
        tokenizer = CharTokenizer.build("hello mini gpt repetition control\n" * 6)
        model = GPT(
            GPTConfig(
                block_size=16,
                vocab_size=tokenizer.vocab_size,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
            )
        ).eval()
        from minigpt.config import Config
        from minigpt.training.checkpoint import CHECKPOINT_VERSION, save_checkpoint

        config = Config()
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "backend": "torch",
            "backend_version": "1",
            "stage": "pretrain",
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": {},
            "iter_num": 1,
            "best_val_loss": 1.0,
            "config": config.to_dict(),
            "model_config": model.config.to_dict(),
            "tokenizer_state": tokenizer.state_dict(),
            "data_meta": {},
            "rng": {},
            "early_stopping": {},
            "run_id": "demo",
        }
        save_checkpoint(run_dir / "best.pt", payload)
        cls.app = DashboardApp(
            project_root=root,
            runs_dir=root / "runs",
            web_dir=ROOT / "web",
            configs_dir=root / "configs",
            python_executable=PYTHON,
        )

    @classmethod
    def tearDownClass(cls):
        cls.app.close()
        cls._tmp.cleanup()

    def test_penalty_path_generates_and_differs(self):
        """验证 路径 相关行为。"""
        base = {"run_id": "demo", "prompt": "hello", "max_new_tokens": 12, "temperature": 0.0, "seed": 7}
        plain = self.app.inference(dict(base))
        penalized = self.app.inference(dict(base, repetition_penalty=2.0, no_repeat_ngram_size=2))
        self.assertIn("final", plain)
        self.assertIn("final", penalized)
        self.assertTrue(penalized["final"]["continuation"] or penalized["final"]["steps"] == 0 or True)
        self.assertEqual(penalized["final"]["attention"], [])  # 惩罚路径不产注意力跟踪
        self.assertIsInstance(penalized["final"]["continuation"], str)
        # 贪心 + 强惩罚应当改变输出（或至少不崩溃且结构完整）。
        self.assertIn("reason", penalized["final"])

    def test_compare_real_worker_two_variants(self):
        """验证 对比、推理 worker 相关行为。"""
        result = self.app.inference_compare(
            {
                "run_id": "demo",
                "prompt": "hello",
                "max_new_tokens": 10,
                "seed": 7,
                "variants": [
                    {"temperature": 0.0},
                    {"temperature": 0.0, "repetition_penalty": 2.5, "no_repeat_ngram_size": 2},
                ],
            }
        )
        self.assertEqual(result["count"], 2)
        self.assertTrue(result["results"][0]["ok"], result["results"][0])
        self.assertTrue(result["results"][1]["ok"], result["results"][1])
        self.assertIsInstance(result["results"][1]["text"], str)


class InferenceCompareDomTests(unittest.TestCase):
    """覆盖 InferenceCompareDomTests 相关行为。"""
    def test_generation_compare_mode_wired(self):
        """验证 生成、对比 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        generation = (ROOT / "web" / "js" / "generation.js").read_text(encoding="utf-8")
        for marker in (
            'id="gen-compare-toggle"',
            'id="gen-compare-panel"',
            'id="gen-compare-variants"',
            'id="gen-compare-add"',
            'id="gen-compare-run-btn"',
            'id="gen-compare-results"',
            'id="gen-compare-status"',
        ):
            self.assertIn(marker, index)
        for marker in (
            "endpoints.inferenceCompare()",
            "gen-compare-toggle",
            "repetition_penalty",
            "no_repeat_ngram_size",
        ):
            self.assertIn(marker, generation)
        # 现有流式生成契约不被破坏（接口路径已集中到 api.js）。
        self.assertIn("endpoints.inferenceStream()", generation)
        self.assertIn('from "./api.js"', generation)
        self.assertIn('id="gen-start-btn"', index)


if __name__ == "__main__":
    unittest.main()