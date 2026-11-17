"""纯 Python 后端的端到端 CLI 走查：pretrain → .mgckpt → chat / evaluate。

torch-free 路径是这台机器上的降级方案，必须在真实命令行下完整跑通，
且产物后缀必须是 .mgckpt（不能悄悄写出 .pt，否则打包/推理会走错分支）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "shakespeare_en" / "train.bin"


def run(args, cwd):
    """在指定目录以子进程跑 CLI，强制 UTF-8 输出避免 Windows 控制台编码干扰断言。"""
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


@unittest.skipUnless(DATA.is_file(), "data/shakespeare_en is not prepared")
class PureCliTests(unittest.TestCase):
    """验证纯后端从训练到对话、评估的完整闭环。"""

    def test_pretrain_chat_evaluate_roundtrip(self) -> None:
        """1 步训练产出双 checkpoint，随后 chat 能应答、evaluate 能落 JSON 报告。"""
        with tempfile.TemporaryDirectory(prefix="minigpt-pure-cli-") as tmp:
            base = Path(tmp)
            runs_dir = base / "runs"
            pretrain = run(
                [
                    "pretrain.py",
                    "--preset",
                    "pure_debug_en",
                    "--runs_dir",
                    str(runs_dir),
                    "--run_name",
                    "smoke",
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
                ],
                ROOT,
            )
            self.assertEqual(pretrain.returncode, 0, pretrain.stdout + pretrain.stderr)
            run_dir = runs_dir / "smoke"
            self.assertTrue((run_dir / "best.mgckpt").is_file())
            self.assertTrue((run_dir / "last.mgckpt").is_file())
            # 纯后端不得产出 torch 格式 .pt，否则说明后端选择被覆盖。
            self.assertFalse((run_dir / "best.pt").exists())

            chat = run(
                [
                    "chat.py",
                    "--checkpoint",
                    str(run_dir / "best.mgckpt"),
                    "--prompt",
                    "ROMEO:",
                    "--max_new_tokens",
                    "3",
                ],
                ROOT,
            )
            self.assertEqual(chat.returncode, 0, chat.stdout + chat.stderr)
            self.assertIn("[chat]", chat.stdout)
            self.assertIn("A:", chat.stdout)

            evaluate = run(
                [
                    "evaluate.py",
                    "--checkpoint",
                    str(run_dir / "best.mgckpt"),
                    "--split",
                    "val",
                    "--num_batches",
                    "1",
                    "--max_samples",
                    "1",
                    "--max_new_tokens",
                    "2",
                    "--output_dir",
                    str(base / "eval_val"),
                ],
                ROOT,
            )
            self.assertEqual(evaluate.returncode, 0, evaluate.stdout + evaluate.stderr)
            self.assertTrue((base / "eval_val" / "evaluation.json").is_file())


if __name__ == "__main__":
    unittest.main()
