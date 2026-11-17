"""observe.py 检查点后端分发测试（F21）。

.pt -> torch 路径；.mgckpt -> pure 引擎路径；无法识别的扩展名必须明确报错。
夹具构造方式与 tests/test_pure_backend.py 保持一致（内存构造 + 临时目录）。
"""

from __future__ import annotations

import io
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None

import observe
from minigpt.config import Config
from minigpt.pure.backend import PureBackend
from minigpt.pure.checkpoint import save_pure_checkpoint
from minigpt.tokenization.char import CharTokenizer
from minigpt.training.checkpoint import CHECKPOINT_VERSION, save_checkpoint
from tests.test_pure_backend import make_config


def _pure_checkpoint(path: Path) -> None:
    config = make_config()
    backend = PureBackend(config)
    tokenizer = CharTokenizer.build("hello mini gpt observe\n" * 3)
    model = backend.build_model(config, vocab_size=tokenizer.vocab_size)
    flat = Config()
    flat.backend_type = "pure"
    payload = {
        "format_version": 1,
        "backend": "pure",
        "backend_version": "1",
        "stage": "pretrain",
        "model": backend.state_dict(model),
        "optimizer": {},
        "iter_num": 1,
        "best_val_loss": 1.0,
        "config": flat.to_dict(portable=True),
        "model_config": model.config.to_dict(),
        "tokenizer_state": tokenizer.state_dict(),
        "data_meta": {},
        "rng": {"python": random.getstate()},
        "early_stopping": {},
        "python_version": "3.14",
        "torch_version": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "run_id": "unit-observe-pure",
    }
    save_pure_checkpoint(path, payload)


def _torch_checkpoint(path: Path) -> None:
    from minigpt.models.gpt import GPT, GPTConfig

    tokenizer = CharTokenizer.build("hello mini gpt observe\n" * 3)
    model_config = GPTConfig(
        block_size=16, vocab_size=tokenizer.vocab_size, n_layer=1, n_head=2, n_embd=8, dropout=0.0
    )
    torch.manual_seed(3)
    model = GPT(model_config)
    flat = Config()
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "backend": "torch",
        "stage": "pretrain",
        "model": model.state_dict(),
        "optimizer": {},
        "iter_num": 1,
        "best_val_loss": 1.0,
        "config": flat.to_dict(portable=True),
        "model_config": model_config.to_dict(),
        "tokenizer_state": tokenizer.state_dict(),
    }
    save_checkpoint(path, payload)


class ObserveDispatchTests(unittest.TestCase):
    """覆盖 ObserveDispatchTests 相关行为。"""
    def test_pure_checkpoint_is_observable(self) -> None:
        """验证 纯后端、checkpoint 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.mgckpt"
            _pure_checkpoint(path)
            output = io.StringIO()
            with redirect_stdout(output):
                code = observe.main(["--checkpoint", str(path), "--prompt", "hi", "--top_k", "3"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("stage=pretrain", text)
            self.assertIn("next-token top-k:", text)
            self.assertIn("embedding_params=", text)

    def test_pure_checkpoint_attention_mode_gives_clear_error(self) -> None:
        # pure 引擎不记录注意力权重：--attention 必须得到清晰错误而不是崩溃。
        """验证 纯后端、checkpoint、注意力、错误处理 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.mgckpt"
            _pure_checkpoint(path)
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaises(RuntimeError) as ctx:
                    observe.main(["--checkpoint", str(path), "--prompt", "hi", "--attention"])
            self.assertIn("attention", str(ctx.exception).lower())
            self.assertIn("pure", str(ctx.exception).lower())
            # 核心观察输出在报错前已经打印
            self.assertIn("next-token top-k:", output.getvalue())

    @unittest.skipIf(torch is None, "torch not installed")
    def test_torch_checkpoint_still_observable(self) -> None:
        """验证 checkpoint 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            _torch_checkpoint(path)
            output = io.StringIO()
            with redirect_stdout(output):
                code = observe.main(["--checkpoint", str(path), "--prompt", "hi", "--top_k", "3"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("stage=pretrain", text)
            self.assertIn("next-token top-k:", text)

    def test_unknown_checkpoint_suffix_raises_clear_error(self) -> None:
        """验证 checkpoint、错误处理 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaises(ValueError) as ctx:
                observe.main(["--checkpoint", str(path)])
            self.assertIn(".pt", str(ctx.exception))
            self.assertIn(".mgckpt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
