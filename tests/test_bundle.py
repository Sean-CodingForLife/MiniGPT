"""针对 `test_bundle.py` 的单元测试与回归保护。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt.bundle import export_bundle, verify_bundle
from minigpt.models.gpt import GPT, GPTConfig
from minigpt.tokenization.char import CharTokenizer
from minigpt.training.checkpoint import CHECKPOINT_VERSION, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def make_checkpoint(directory: Path) -> Path:
    torch.manual_seed(0)
    text = "hello mini gpt world\n" * 20
    tokenizer = CharTokenizer.build(text)
    model_config = GPTConfig(
        block_size=16,
        vocab_size=tokenizer.vocab_size,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(model_config)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "backend": "torch",
        "backend_version": "1",
        "stage": "sft",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": {},
        "iter_num": 2,
        "best_val_loss": 1.0,
        "config": {
            "max_new_tokens": 2,
            "temperature": 0.8,
            "top_k": 10,
            "top_p": 1.0,
            "stage": "sft",
        },
        "model_config": model_config.to_dict(),
        "tokenizer_state": tokenizer.state_dict(),
        "data_meta": {"dataset": "test"},
        "rng": {},
        "early_stopping": {},
        "run_id": "bundle-unit",
    }
    path = directory / "best.pt"
    save_checkpoint(path, payload)
    return path


def answer_line(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.startswith("A:")]
    return lines[-1] if lines else ""


class BundleTests(unittest.TestCase):
    """覆盖 BundleTests 相关行为。"""
    def _export(self, base: Path) -> Path:
        checkpoint = make_checkpoint(base)
        bundle_dir = base / "bundle"
        export_bundle(checkpoint, bundle_dir, create_zip=True, force=True)
        return bundle_dir

    def _run(self, bundle_dir: Path, cwd: Path, *args: str, env_extra: dict[str, str] | None = None):
        env = os.environ.copy()
        env.update(env_extra or {})
        return subprocess.run(
            [sys.executable, str(bundle_dir / "run.py"), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
            env=env,
        )

    def test_export_verify_and_run(self) -> None:
        """验证 导出、校验 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bundle_dir = self._export(base)
            result = {"bundle": str(bundle_dir)}
            self.assertTrue(Path(result["bundle"]).is_dir())
            self.assertTrue((base / "bundle.zip").is_file())
            for name in (
                "config.json",
                "pytorch_model.bin",
                "weights.f32",
                "tokenizer.json",
                "run.py",
                "serve.py",
                "MANIFEST.json",
                "MODEL_CARD.md",
            ):
                self.assertTrue((bundle_dir / name).is_file(), name)
            self.assertTrue((bundle_dir / "minigpt" / "pure" / "tensor.py").is_file())
            verified = verify_bundle(bundle_dir)
            self.assertEqual(verified["config"]["format"], "minigpt-bundle-v1")
            manifest = json.loads((bundle_dir / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertIn("run.py", manifest["files"])
            self.assertIn("weights.f32", manifest["files"])
            self.assertIn("minigpt/pure/gpt.py", manifest["files"])
            self.assertNotIn(str(base), json.dumps(manifest))
            self.assertNotIn(str(base), (bundle_dir / "config.json").read_text(encoding="utf-8"))

            run = self._run(bundle_dir, base, "--prompt", "he", "--max_new_tokens", "2", "--device", "cpu", "--temperature", "0")
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("engine=torch", run.stdout)
            self.assertIn("A:", run.stdout)

    def test_pure_engine_matches_torch_engine(self) -> None:
        """验证 纯后端 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bundle_dir = self._export(base)
            common = ("--prompt", "he", "--max_new_tokens", "3", "--temperature", "0", "--device", "cpu")
            torch_run = self._run(bundle_dir, base, "--engine", "torch", *common)
            pure_run = self._run(bundle_dir, base, "--engine", "pure", *common)
            self.assertEqual(torch_run.returncode, 0, torch_run.stdout + torch_run.stderr)
            self.assertEqual(pure_run.returncode, 0, pure_run.stdout + pure_run.stderr)
            self.assertIn("engine=torch", torch_run.stdout)
            self.assertIn("engine=pure", pure_run.stdout)
            self.assertEqual(answer_line(torch_run.stdout), answer_line(pure_run.stdout))

    def test_auto_engine_falls_back_to_pure_without_torch(self) -> None:
        """验证 纯后端 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bundle_dir = self._export(base)
            shim = base / "no-torch-shim"
            shim.mkdir()
            (shim / "torch.py").write_text("raise ImportError('torch disabled for this test')\n", encoding="utf-8")
            env_extra = {"PYTHONPATH": str(shim)}
            auto_run = self._run(
                bundle_dir,
                base,
                "--engine",
                "auto",
                "--prompt",
                "he",
                "--max_new_tokens",
                "2",
                "--temperature",
                "0",
                env_extra=env_extra,
            )
            self.assertEqual(auto_run.returncode, 0, auto_run.stdout + auto_run.stderr)
            self.assertIn("engine=pure", auto_run.stdout)
            self.assertIn("A:", auto_run.stdout)

            torch_run = self._run(
                bundle_dir,
                base,
                "--engine",
                "torch",
                "--prompt",
                "he",
                "--max_new_tokens",
                "1",
                env_extra=env_extra,
            )
            self.assertNotEqual(torch_run.returncode, 0)
            self.assertIn("torch", (torch_run.stdout + torch_run.stderr).lower())


if __name__ == "__main__":
    unittest.main()
