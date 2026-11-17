"""F22: the SFT resume path must build the model exactly once, with the real
vocab size from the SFT tokenizer (the vocab=1 placeholder build is dead code).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sft
from minigpt.config import Config
from minigpt.tokenization.char import CharTokenizer
from minigpt.training.trainer import Trainer

from tests.test_trainer_failures import FakeSource, torch


def write_sft_data(data_dir: Path) -> None:
    record = {"q": "hello", "a": "world"}
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )


@unittest.skipIf(torch is None, "torch not installed")
class SftResumeBuildTests(unittest.TestCase):
    """覆盖 SftResumeBuildTests 相关行为。"""
    def test_resume_builds_model_once_with_real_vocab(self) -> None:
        """验证 断点续训、模型、词表 相关行为。"""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "sft_data"
            data_dir.mkdir()
            write_sft_data(data_dir)
            config = Config(
                project_root=tmp,
                data_dir=str(data_dir),
                runs_dir=str(Path(tmp) / "runs"),
                run_name="sft-resume-run",
                stage="sft",
                n_layer=1,
                n_head=2,
                n_embd=16,
                block_size=8,
                dropout=0.0,
                batch_size=2,
                max_steps=1,
                warmup_steps=0,
                eval_interval=1,
                eval_iters=1,
                log_interval=1,
                save_interval=1,
                patience=0,
                callbacks="telemetry",
                pretrain_mix_ratio=0.0,
                seed=7,
            )
            # Phase 0: a tiny pretrain run provides the init checkpoint.
            pre_config = Config(
                project_root=tmp,
                data_dir=str(data_dir),
                runs_dir=str(Path(tmp) / "runs"),
                run_name="pre-for-sft",
                stage="pretrain",
                n_layer=1,
                n_head=2,
                n_embd=16,
                block_size=8,
                dropout=0.0,
                batch_size=2,
                max_steps=1,
                warmup_steps=0,
                eval_interval=1,
                eval_iters=1,
                log_interval=1,
                save_interval=1,
                patience=0,
                callbacks="telemetry",
                seed=7,
            )
            pre_backend = sft.registry.BACKENDS.get(pre_config.backend_type)(pre_config)
            pre_tokenizer = CharTokenizer.build(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 Q:A\n"
            )
            pre_tokenizer.save(data_dir)
            pre_model = pre_backend.build_model(pre_config, pre_tokenizer.vocab_size)
            pre_optimizer = pre_backend.build_optimizer(pre_model, pre_config)
            pre_source = FakeSource()
            pre_trainer = Trainer(
                config=pre_config,
                backend=pre_backend,
                model=pre_model,
                optimizer=pre_optimizer,
                tokenizer=pre_tokenizer,
                data_source=pre_source,
            )
            pre_trainer.train()
            config.init_from = str(pre_trainer.run_dir / "last.pt")

            # Phase 1: a real 1-step SFT run creates last.pt in the run dir.
            first = sft._build_trainer(config, resume=False)
            first.train()
            self.assertTrue((first.run_dir / "last.pt").is_file())

            # Phase 2: resume must build the model exactly once (real vocab).
            config.max_steps = 2
            config.init_from = ""
            backend_cls = sft.registry.BACKENDS.get(config.backend_type)
            backend = backend_cls(config)
            real_build = backend.build_model
            calls: list[int] = []

            def spy_build(model_config, vocab_size):
                calls.append(int(vocab_size))
                return real_build(model_config, vocab_size)

            with mock.patch.object(backend_cls, "build_model", side_effect=spy_build):
                trainer = sft._build_trainer(config, resume=True)
            self.assertEqual(len(calls), 1, f"expected a single model build, got {calls}")
            self.assertEqual(calls[0], trainer.tokenizer.vocab_size)
            self.assertGreater(calls[0], 1)
            self.assertEqual(trainer.start_iter, 1)


if __name__ == "__main__":
    unittest.main()
