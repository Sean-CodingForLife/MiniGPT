"""Streaming inference tests: sampling trace, iter_trace events, NDJSON endpoint."""

from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt.models.gpt import GPT, GPTConfig  # noqa: E402
from minigpt.tokenization.char import CharTokenizer  # noqa: E402
from minigpt.web.server import ApiError, DashboardApp, create_server  # noqa: E402
from visualize import iter_trace  # noqa: E402

# 假推理 worker：输出 3 个 step + done，每行 flush；用于测试 chunked 转发。
STREAM_WORKER = (
    "import json, sys, time\n"
    "for index in range(1, 4):\n"
    "    print(json.dumps({'type': 'step', 'data': {'step': index}}), flush=True)\n"
    "    time.sleep(0.05)\n"
    "print(json.dumps({'type': 'done'}), flush=True)\n"
)


def make_bundle():
    torch.manual_seed(11)
    tokenizer = CharTokenizer.build("hello mini gpt visualization\n" * 4)
    model = GPT(
        GPTConfig(
            block_size=12,
            vocab_size=tokenizer.vocab_size,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
    ).eval()
    backend = SimpleNamespace(device_info=lambda: {"device": "cpu"})
    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        payload={"stage": "pretrain"},
    )


class SamplingTraceTests(unittest.TestCase):
    """覆盖 SamplingTraceTests 相关行为。"""
    def setUp(self) -> None:
        self.bundle = make_bundle()
        self.model = self.bundle.model

    def test_sampling_trace_covers_filters_and_draw(self) -> None:
        """验证 采样、追踪 相关行为。"""
        steps = list(
            self.model.generate_stream(
                [1, 2, 3],
                max_new_tokens=3,
                temperature=0.8,
                top_k=6,
                top_p=0.9,
                eos_id=None,
                trace_top_k=3,
                trace_sampling=True,
                trace_candidates=8,
            )
        )
        self.assertEqual(len(steps), 3)
        for step in steps:
            sampling = step["sampling"]
            self.assertEqual(sampling["mode"], "sample")
            self.assertEqual(sampling["chosen_id"], step["token_id"])
            self.assertGreaterEqual(sampling["u"], 0.0)
            self.assertLess(sampling["u"], 1.0)
            self.assertLessEqual(
                sampling["chosen_cum_low"], sampling["u"] + 1e-9
            )
            self.assertLessEqual(
                sampling["u"], sampling["chosen_cum_high"] + 1e-9
            )
            candidates = sampling["candidates"]
            self.assertTrue(candidates)
            self.assertTrue(any(candidate["chosen"] for candidate in candidates))
            raw = [candidate["prob_raw"] for candidate in candidates]
            self.assertEqual(raw, sorted(raw, reverse=True))
            for candidate in candidates:
                self.assertTrue(0.0 <= candidate["prob_raw"] <= 1.0)
                if candidate["in_top_k"]:
                    self.assertGreaterEqual(
                        candidate["prob_top_k"], candidate["prob_raw"] - 1e-6
                    )
                else:
                    self.assertEqual(candidate["prob_top_k"], 0.0)
                self.assertTrue(candidate["prob_final"] <= candidate["prob_top_k"] + 1e-6)
                if candidate["chosen"]:
                    self.assertAlmostEqual(
                        candidate["prob_final"], step["prob"], places=5
                    )

    def test_greedy_trace_has_no_draw(self) -> None:
        """验证 追踪 相关行为。"""
        steps = list(
            self.model.generate_stream(
                [1, 2],
                max_new_tokens=2,
                temperature=0,
                eos_id=None,
                trace_sampling=True,
                trace_candidates=5,
            )
        )
        sampling = steps[0]["sampling"]
        self.assertEqual(sampling["mode"], "greedy")
        self.assertIsNone(sampling["u"])
        chosen = [item for item in sampling["candidates"] if item["chosen"]]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(
            chosen[0]["prob_final"], max(item["prob_final"] for item in sampling["candidates"])
        )

    def test_iter_trace_yields_meta_steps_final_incrementally(self) -> None:
        """验证 追踪 相关行为。"""
        events = []
        forced = [self.bundle.tokenizer.encode("h")[0]]
        for event in iter_trace(
            self.bundle,
            prompt="he",
            max_new_tokens=2,
            temperature=0.8,
            top_k=5,
            top_p=1.0,
            trace_top_k=3,
            max_context=8,
            template="raw",
            forced_ids=forced,
        ):
            events.append(event)
        self.assertEqual(events[0]["type"], "meta")
        self.assertEqual(events[-1]["type"], "final")
        steps = [event["data"] for event in events if event["type"] == "step"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(events[0]["forced_tokens"], [self.bundle.tokenizer.token_repr(forced[0])])
        self.assertEqual(events[0]["prompt_tokens"][-1], self.bundle.tokenizer.token_repr(forced[0]))
        for step in steps:
            self.assertIn("sampling", step)
            self.assertEqual(step["sampling"]["chosen_id"], step["token_id"])
            self.assertGreaterEqual(step["elapsed_ms"], 0.0)
        final = events[-1]["data"]
        self.assertEqual(final["steps"], 2)
        self.assertIn(final["reason"], {"eos", "max_tokens"})
        self.assertTrue(final["context"])
        self.assertTrue(final["attention"])


class BuildRequestTests(unittest.TestCase):
    """覆盖 BuildRequestTests 相关行为。"""
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "runs" / "demo").mkdir(parents=True)
        (root / "runs" / "demo" / "best.pt").write_bytes(b"fake")
        self.app = DashboardApp(
            project_root=root,
            runs_dir=root / "runs",
            web_dir=PROJECT_ROOT / "web",
            configs_dir=root / "configs",
            python_executable=sys.executable,
        )

    def tearDown(self) -> None:
        self.app.close()
        self.tmp.cleanup()

    def test_request_defaults_and_prefix(self) -> None:
        """验证 `test_request_defaults_and_prefix` 对应的行为约定。"""
        request = self.app.build_inference_request(
            {"run_id": "demo", "prompt": "hi", "prefix_token_ids": [3, 4]}
        )
        self.assertEqual(request["checkpoint"], str(Path(self.app.runs_dir) / "demo" / "best.pt"))
        self.assertEqual(request["prefix_token_ids"], [3, 4])
        self.assertEqual(request["seed"], 1337)
        self.assertEqual(request["trace_candidates"], 20)

    def test_request_rejects_bad_prefix(self) -> None:
        """验证 `test_request_rejects_bad_prefix` 对应的行为约定。"""
        for bad in ([True], ["3"], [-1]):
            with self.assertRaises(ApiError) as caught:
                self.app.build_inference_request({"run_id": "demo", "prefix_token_ids": bad})
            self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(ApiError):
            self.app.build_inference_request({"run_id": "demo", "prefix_token_ids": [1] * 513})

    def test_request_rejects_bad_seed(self) -> None:
        """验证 随机种子 相关行为。"""
        with self.assertRaises(ApiError):
            self.app.build_inference_request({"run_id": "demo", "seed": -1})


class StreamEndpointTests(unittest.TestCase):
    """覆盖 StreamEndpointTests 相关行为。"""
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "runs" / "demo").mkdir(parents=True)
        (root / "runs" / "demo" / "best.pt").write_bytes(b"fake")
        self.server = create_server(
            host="127.0.0.1",
            port=0,
            project_root=root,
            runs_dir=root / "runs",
            web_dir=PROJECT_ROOT / "web",
            configs_dir=root / "configs",
            python_executable=sys.executable,
        )
        self.app = self.server.app
        self.app._inference_command = lambda: [  # type: ignore[method-assign]
            sys.executable,
            "-u",
            "-c",
            STREAM_WORKER,
        ]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()
        self.host, self.port = self.server.server_address[0], self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.tmp.cleanup()

    def _connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=15)

    def test_stream_forwards_each_ndjson_event(self) -> None:
        """验证 流式输出、事件 相关行为。"""
        connection = self._connect()
        try:
            body = json.dumps({"run_id": "demo", "prompt": "hi"})
            connection.request(
                "POST",
                "/api/v1/inference/stream",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.getheader("Content-Type"),
                "application/x-ndjson; charset=utf-8",
            )
            events = [
                json.loads(line.decode("utf-8"))
                for line in response
                if line.strip()
            ]
        finally:
            connection.close()
        self.assertEqual([event["type"] for event in events], ["step", "step", "step", "done"])
        self.assertEqual([event["data"]["step"] for event in events[:3]], [1, 2, 3])

    def test_client_abort_kills_worker_and_server_stays_healthy(self) -> None:
        """验证 推理 worker 相关行为。"""
        connection = self._connect()
        try:
            connection.request(
                "POST",
                "/api/v1/inference/stream",
                body=json.dumps({"run_id": "demo", "prompt": "hi"}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.readline().strip())
        finally:
            connection.close()
        deadline = time.time() + 10
        while time.time() < deadline and self.app._inference_processes:
            time.sleep(0.05)
        self.assertEqual(self.app._inference_processes, set())

        health = self._connect()
        try:
            health.request("GET", "/api/v1/health")
            self.assertEqual(health.getresponse().status, 200)
        finally:
            health.close()


if __name__ == "__main__":
    unittest.main()
