"""针对 `test_spa.py` 的单元测试与回归保护。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minigpt.web.server import ApiError, DashboardApp

ROOT = Path(__file__).resolve().parents[1]


class SpaTests(unittest.TestCase):
    """覆盖 SpaTests 相关行为。"""
    def test_index_and_modules_are_esm_spa(self) -> None:
        """验证 SPA 前端 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        generation = (ROOT / "web" / "js" / "generation.js").read_text(encoding="utf-8")
        router = (ROOT / "web" / "js" / "router.js").read_text(encoding="utf-8")
        dashboard = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        self.assertIn('type="module"', index)
        self.assertIn('href="#/generate"', index)
        self.assertIn('id="view-generate"', index)
        self.assertIn('id="gen-start-btn"', index)
        self.assertIn('id="gen-heatmap"', index)
        self.assertIn('import { initGenerationView }', app)
        self.assertIn('from "./js/router.js"', app)
        # 接口路径已集中到 api.js；视图模块通过 endpoints 调用。
        self.assertIn('from "./api.js"', generation)
        self.assertIn("endpoints.inferenceStream()", generation)
        self.assertIn("initRouter", router)
        self.assertIn("hashchange", router)
        self.assertIn("section[data-view]", router)
        self.assertNotIn('document.querySelectorAll("[data-view]")', router)

        # 实时视图：流式读取、固定视口（页面不随 token 数变长）、逐 token 面板。
        self.assertIn("getReader()", generation)
        self.assertIn("AbortController", generation)
        self.assertIn("data-reroll", generation)
        self.assertIn("data-branch-token", generation)
        self.assertIn("nucleus_segments", generation)
        self.assertIn("gen-axis", style)
        self.assertIn("overflow: hidden", style)
        self.assertIn("body[data-active-view=\"generate\"] .footer", style)
        self.assertIn(".gen-sum", style)

    def test_stage_tabs_and_read_only_views(self) -> None:
        """验证 阶段页 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        router = (ROOT / "web" / "js" / "router.js").read_text(encoding="utf-8")
        dashboard = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        # 六个阶段页签 + 共用训练工作区 + 只读视图
        for route in ("data", "pretrain", "sft", "eval", "generate", "export"):
            self.assertIn(f'href="#/{route}"', index)
            self.assertIn(f'data-route="{route}"', index)
        self.assertIn('id="view-train"', index)
        self.assertIn('id="view-data"', index)
        self.assertIn('id="view-eval"', index)
        self.assertIn('id="view-export"', index)
        self.assertIn('id="stage-tag"', index)
        self.assertIn('id="task-drawer"', index)
        self.assertIn('id="task-drawer-btn"', index)

        # 路由表 + 首屏事件顺序（先注册视图，再启动路由）
        self.assertIn("const ROUTES", router)
        self.assertIn("spa:view", router)
        self.assertNotIn("LEGACY_ROUTES", router)
        self.assertLess(app.index("initDataView()"), app.index("initRouter()"))
        self.assertLess(app.index("initEvalView()"), app.index("initRouter()"))
        self.assertLess(app.index("initExportView()"), app.index("initRouter()"))

        # 训练工作区按阶段过滤预设/运行，并可被路由切换
        self.assertIn("presetsForStage", dashboard)
        self.assertIn("runsForStage", dashboard)
        self.assertIn("applyStage", dashboard)
        self.assertIn('window.addEventListener("spa:view"', dashboard)

        # 任务生命周期是原子操作：新建任务 / 启动 / 停止 / 删除任务，互不合并。
        self.assertIn('id="create-run-btn"', index)
        self.assertIn('id="run-name-input"', index)
        self.assertIn('id="start-btn"', index)
        self.assertIn('id="delete-run-btn"', index)
        self.assertIn('id="delete-config-btn"', index)
        self.assertIn("createRun", dashboard)
        self.assertIn("startSelectedRun", dashboard)
        self.assertIn("deleteRun", dashboard)
        self.assertIn("isRunStale", dashboard)
        # 接口路径已集中到 api.js（含 ?force=1 / /start 等端点构造）。
        self.assertIn('from "./api.js"', dashboard)
        self.assertIn("endpoints.runs()", dashboard)
        self.assertIn("endpoints.runStart(", dashboard)
        self.assertIn("endpoints.runDelete(", dashboard)
        api_layer = (ROOT / "web" / "js" / "api.js").read_text(encoding="utf-8")
        self.assertIn("?force=1", api_layer)
        self.assertIn('"start"', api_layer)

        # 只读模块请求的接口
        data = (ROOT / "web" / "js" / "data.js").read_text(encoding="utf-8")
        evals = (ROOT / "web" / "js" / "eval.js").read_text(encoding="utf-8")
        exports = (ROOT / "web" / "js" / "export.js").read_text(encoding="utf-8")
        self.assertIn("endpoints.datasets()", data)
        self.assertIn("endpoints.evaluations()", evals)
        self.assertIn("endpoints.exports()", exports)

        self.assertIn(".stage-view", style)
        self.assertIn(".task-drawer", style)

    def test_failure_feedback_and_quick_entry(self) -> None:
        """验证 失败路径 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        dashboard = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        for marker in (
            'id="toast-region"',
            'id="failure-banner"',
            'id="failure-detail-btn"',
            'id="global-failure-btn"',
            'id="failure-dialog"',
            'id="failure-dialog-error"',
            'id="failure-dialog-console"',
            'id="failure-copy-btn"',
            'id="failure-console-tab-btn"',
        ):
            self.assertIn(marker, index)
        for marker in (
            "showToast",
            "renderFailureBanner",
            "renderGlobalFailureEntry",
            "openFailureDialog",
            "data-failure-run",
            "endpoints.runConsole(target, 200)",
        ):
            self.assertIn(marker, dashboard)
        for marker in (".toast-region", ".failure-banner", ".failure-dialog"):
            self.assertIn(marker, style)

    def test_state_driven_run_card(self) -> None:
        """验证 `test_state_driven_run_card` 对应的行为约定。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        dashboard = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        for marker in (
            'id="run-card-status"',
            'id="next-step-btn"',
            'id="start-btn"',
            'id="stop-btn"',
            'id="delete-run-btn"',
        ):
            self.assertIn(marker, index)
        for marker in (
            "updateRunCard",
            "goNextStep",
            "consumePendingSftFrom",
            "基于此运行做 SFT",
            "去生成可视化",
        ):
            self.assertIn(marker, dashboard)
        self.assertIn(".run-card", style)

    def test_classic_attention_demo(self) -> None:
        """验证 注意力 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        generation = (ROOT / "web" / "js" / "generation.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
        attention_path = ROOT / "web" / "js" / "attention.js"
        self.assertTrue(attention_path.is_file())
        attention = attention_path.read_text(encoding="utf-8")

        # 炫技剧场已移除。
        self.assertFalse((ROOT / "web" / "js" / "theater.js").exists())
        self.assertNotIn("theater", generation)
        self.assertNotIn("theater-panel", style)

        for marker in (
            'id="attn-canvas"',
            'id="attn-layer"',
            'id="attn-head"',
            'id="attn-mode-links"',
            'id="attn-mode-matrix"',
            'id="attn-topk"',
            'id="attn-threshold"',
            'id="attn-palette"',
            'id="attn-normalize"',
            'id="attn-values"',
            'id="attn-reset"',
            'id="attn-export"',
            'id="attn-tooltip"',
            'id="attn-source"',
            'id="attn-query-select"',
            'id="attn-metrics"',
            'id="attn-search"',
        ):
            self.assertIn(marker, index)
        self.assertIn('import { initAttentionDemo } from "./attention.js"', generation)
        self.assertIn("state.attentionDemo.setData", generation)
        for marker in ("ResizeObserver", "toDataURL", "bezierCurveTo", "query", "key", "topK", "windowSum", "outsideWindow"):
            self.assertIn(marker, attention)
        for marker in (".attn-demo-panel", ".attn-rail", ".attn-canvas"):
            self.assertIn(marker, style)

        # 固定尺寸：舞台高度不依赖 token 数；概览与矩阵弹窗也有界。
        self.assertIn("height: 380px", style)
        self.assertIn("height: 160px", style)
        self.assertIn("max-height: min(92vh, 820px)", style)
        self.assertNotIn("T * rowH", attention)
        self.assertNotIn("T * cell", attention)
        self.assertNotIn("new Array(cols)", generation)
        self.assertNotIn("rows = shown.length", generation)

        # 长序列：只传当前 Query 行；历史下拉最多 64 项；概览最多 32 个 Key。
        self.assertIn("rows: step.attn_last", generation)
        self.assertIn("steps.slice(-64)", generation)
        self.assertIn("Math.min(32, context.length)", generation)
        self.assertIn("queryToken: context.length ? context[context.length - 1]", generation)
        self.assertIn("predictedToken: step.token", generation)

    def test_attention_long_sequence_runtime(self) -> None:
        """真实加载 attention.js，喂入 1024 token，验证固定尺寸与 Top-K 行为。"""
        import json
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        script = ROOT / "tests" / "attention_runtime_check.mjs"
        proc = subprocess.run(
            [node, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])

        # 画布像素高度在 10 / 1024 / 矩阵 / 回到短序列 时完全一致。
        self.assertEqual(result["shortHeight"], result["longHeight"])
        self.assertEqual(result["longHeight"], result["matrixHeight"])
        self.assertEqual(result["matrixHeight"], result["returnHeight"])

        # 长序列仍展示 Top-K，而不是空白。
        self.assertIn("展示 Key8 / 1024", result["longMetrics"])
        self.assertNotIn("展示 Key0 / 1024", result["longMetrics"])
        self.assertIn("展示覆盖", result["longMetrics"])

        # 权重守恒：展示覆盖 + 其余窗口内 = 已传输的 60%，窗口外 = 40%，
        # 且不因阈值/Top-K 改变而把未展示项当成 0。
        import re

        def pct_of(label, text):
            m = re.search(re.escape(label) + r"([\d.]+)%", text)
            self.assertIsNotNone(m, f"missing {label} in {text}")
            return float(m.group(1))

        partial = result["partialMetrics"]
        shown = pct_of("展示覆盖", partial)
        rest = pct_of("其余窗口内", partial)
        outside = pct_of("窗口外未传输", partial)
        self.assertAlmostEqual(outside, 40.0, delta=0.2)
        self.assertAlmostEqual(shown + rest, 60.0, delta=0.2)
        self.assertGreater(shown, 0.0)  # Top-K 确有展示，未把长序列判成空白

        # 交互回归：芯片点击 / 键盘选择 / 行放大图例 / 导出合成。
        self.assertIn("K[3]", result["clickDetail"])
        self.assertIn("α=", result["clickDetail"])
        self.assertIn("K[4]", result["keyDetail"])
        self.assertTrue(result["legendMax"].startswith("行最大"))
        self.assertTrue(result["exportOk"])
        self.assertEqual(result["narrowHeight"], result["shortHeight"])
        self.assertIn("展示 Key", result["narrowMetrics"])

    def test_generation_playback_controls(self) -> None:
        """验证 生成 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        generation = (ROOT / "web" / "js" / "generation.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
        for marker in (
            'id="gen-play-pause"',
            'id="gen-play-step"',
            'id="gen-play-replay"',
            'id="gen-play-speed"',
            'id="gen-play-progress"',
        ):
            self.assertIn(marker, index)
        for marker in ("scheduleReveal", "togglePlayPause", "playStep", "replayGeneration", "setPlaySpeed", "visibleCount"):
            self.assertIn(marker, generation)
        self.assertIn(".gen-playbar", style)

    def test_sft_checkpoint_guard(self) -> None:
        """验证 SFT、checkpoint、防护 相关行为。"""
        dashboard = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        for marker in ("sftConfigProblem", "去配置 checkpoint", "无法创建 SFT 任务", "无法启动 SFT 任务"):
            self.assertIn(marker, dashboard)

    def test_run_lifecycle_bar(self) -> None:
        """验证 生命周期 相关行为。"""
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        dashboard = (ROOT / "web" / "js" / "dashboard.js").read_text(encoding="utf-8")
        style = (ROOT / "web" / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="run-lifecycle"', index)
        for marker in ("renderLifecycle", "lifecycle-node", "openFailureDialog"):
            self.assertIn(marker, dashboard)
        for marker in (".run-lifecycle", ".lifecycle-node.current", ".lifecycle-node.failed"):
            self.assertIn(marker, style)

    def test_inference_endpoint_validation_and_worker_dispatch(self) -> None:
        """验证 推理、校验、推理 worker 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "demo"
            run_dir.mkdir(parents=True)
            (run_dir / "best.pt").write_bytes(b"not-a-real-checkpoint")
            empty_run = root / "runs" / "empty"
            empty_run.mkdir(parents=True)
            app = DashboardApp(
                project_root=root,
                runs_dir=root / "runs",
                web_dir=ROOT / "web",
                configs_dir=root / "configs",
                python_executable="python",
            )
            captured = {}

            def fake_worker(request):
                captured.update(request)
                return {"steps": [], "final": {"context": [], "attention": []}}

            app._run_inference_worker = fake_worker  # type: ignore[method-assign]
            result = app.inference(
                {
                    "run_id": "demo",
                    "prompt": "Hello",
                    "max_new_tokens": 3,
                    "temperature": 0,
                    "top_k": 5,
                    "top_p": 0.9,
                    "trace_top_k": 2,
                    "max_context": 8,
                    "template": "raw",
                    "device": "cpu",
                }
            )
            self.assertEqual(result["run_id"], "demo")
            self.assertEqual(result["checkpoint"], "best.pt")
            self.assertEqual(captured["prompt"], "Hello")
            self.assertEqual(captured["max_new_tokens"], 3)
            self.assertEqual(captured["checkpoint"], str(run_dir / "best.pt"))
            with self.assertRaises(ApiError) as error:
                app.inference({"run_id": "empty", "prompt": "x"})
            self.assertEqual(error.exception.status, 400)
            app.close()


if __name__ == "__main__":
    unittest.main()
