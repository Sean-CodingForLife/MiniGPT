"""可移植性回归：项目在「换工作目录 / 无 torch / checkpoint 换位置」时仍能运行。

这三个场景对应最常见的部署问题：用相对路径写死 cwd、Web 控制台被 torch
导入链拖死、checkpoint 里记录绝对路径导致拷贝后加载失败。测试一律通过
独立子进程执行，避免污染当前解释器状态。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "shakespeare_en"
SMOKE = ROOT / "runs" / "demo-en-sft" / "best.pt"

try:
    import torch
except ImportError:
    torch = None


class PortabilityTests(unittest.TestCase):
    """验证训练入口、控制台导入与推理加载都不依赖特定路径或重依赖。"""

    @unittest.skipIf(torch is None, "torch not installed")
    def test_cli_runs_from_another_cwd(self) -> None:
        """从临时目录启动 pretrain.py 也能完成 1 步训练，并只把产物写进 --runs_dir。"""
        if not (DATA / "train.bin").is_file():
            self.skipTest("English data not prepared")
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(ROOT / "pretrain.py"),
                "--preset",
                "debug_en",
                "--data_dir",
                str(DATA),
                "--runs_dir",
                str(Path(tmp) / "runs"),
                "--run_name",
                "from-other-cwd",
                "--max_steps",
                "1",
                "--eval_interval",
                "1",
                "--eval_iters",
                "1",
                "--save_interval",
                "1",
                "--callbacks",
                "telemetry",
            ]
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            self.assertTrue((Path(tmp) / "runs" / "from-other-cwd" / "last.pt").is_file())

    def test_dashboard_imports_without_torch(self) -> None:
        """把 torch 置空后仍能 import Web 服务端，防止控制台被训练依赖拖垮。"""
        code = "import sys; sys.modules['torch'] = None; import minigpt.web.server; print('ok')"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_checkpoint_can_move(self) -> None:
        """checkpoint 拷贝到另一目录后仍可加载，说明内部只保存相对/自描述信息。"""
        if not SMOKE.is_file():
            self.skipTest("smoke checkpoint not available")
        with tempfile.TemporaryDirectory() as tmp:
            # 复制而非移动，避免破坏仓库内的 smoke 产物供其他测试复用。
            moved = Path(tmp) / "moved.pt"
            moved.write_bytes(SMOKE.read_bytes())
            code = (
                "import sys; sys.path.insert(0, %r); "
                "from minigpt.inference import load_bundle; "
                "b = load_bundle(%r, device='cpu'); print(b.tokenizer.vocab_size)"
            ) % (str(ROOT), str(moved))
            result = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("68", result.stdout)


if __name__ == "__main__":
    unittest.main()
