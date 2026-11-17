"""PureBackend + .mgckpt checkpoint tests (no subprocess, fast)."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from minigpt.config import Config
from minigpt.pure.backend import PureBackend
from minigpt.pure.checkpoint import (
    PureCheckpointError,
    is_pure_checkpoint,
    load_pure_checkpoint,
    save_pure_checkpoint,
)
from minigpt.pure.gpt import PureGPT
from minigpt.pure.inference import load_pure_bundle
from minigpt.tokenization.char import CharTokenizer


def make_config(**overrides):
    values = dict(
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
        tie_weights=True,
        seed=7,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class PureBackendTests(unittest.TestCase):
    """覆盖 PureBackendTests 相关行为。"""
    def setUp(self) -> None:
        self.config = make_config()
        self.backend = PureBackend(self.config)
        self.model = self.backend.build_model(self.config, vocab_size=13)
        self.optimizer = self.backend.build_optimizer(self.model, self.config)
        self.batch = {
            "x": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "y": [[-100, 2, 3, 4], [5, 6, -100, 8]],
        }

    def test_train_and_eval_steps(self) -> None:
        """验证 训练、评估 相关行为。"""
        before = list(self.model.params["transformer.wte.weight"].data)
        partial = self.backend.train_step(
            self.model, self.optimizer, self.batch, grad_clip=1.0, do_step=False
        )
        self.assertTrue(partial.loss == partial.loss)  # not NaN
        self.assertFalse(partial.stepped)
        self.assertEqual(partial.tokens, 6)

        result = self.backend.optimizer_step(
            self.model,
            self.optimizer,
            grad_clip=1.0,
            loss_value=partial.loss,
            tokens=partial.tokens,
        )
        self.assertTrue(result.stepped)
        self.assertTrue(result.grad_norm == result.grad_norm)
        self.assertNotEqual(before, list(self.model.params["transformer.wte.weight"].data))

        evaluation = self.backend.eval_step(self.model, self.batch)
        self.assertTrue(evaluation.loss == evaluation.loss)
        self.assertEqual(evaluation.tokens, 6)

    def test_grad_accum_loss_scale_normalizes(self) -> None:
        """验证 梯度、梯度累积、loss 相关行为。"""
        for tensor in self.model.parameters():
            tensor.zero_grad()
        self.backend.train_step(
            self.model, self.optimizer, self.batch, grad_clip=1.0, do_step=False, loss_scale=0.25
        )
        scaled = list(self.model.params["transformer.wte.weight"].grad.data)
        for tensor in self.model.parameters():
            tensor.zero_grad()
        self.backend.train_step(
            self.model, self.optimizer, self.batch, grad_clip=1.0, do_step=False, loss_scale=1.0
        )
        full = list(self.model.params["transformer.wte.weight"].grad.data)
        for a, b in zip(scaled, full):
            self.assertAlmostEqual(a, b * 0.25, places=10)

    def test_load_missing_parameter_raises(self) -> None:
        """验证 `test_load_missing_parameter_raises` 对应的行为约定。"""
        state = self.model.state_dict()
        del state["transformer.wte.weight"]
        fresh = self.backend.build_model(self.config, vocab_size=13)
        with self.assertRaises(KeyError):
            self.backend.load_state_dict(fresh, state)

    def test_pure_rejects_bias_and_untied(self) -> None:
        """验证 纯后端 相关行为。"""
        from minigpt.pure.gpt import PureGPTConfig

        with self.assertRaises(ValueError):
            PureGPT(PureGPTConfig(block_size=8, vocab_size=13, n_layer=1, n_head=2, n_embd=8, bias=True))
        with self.assertRaises(ValueError):
            PureGPT(PureGPTConfig(block_size=8, vocab_size=13, n_layer=1, n_head=2, n_embd=8, tie_weights=False))

    def test_generate_empty_prompt_raises(self) -> None:
        """验证 生成 相关行为。"""
        with self.assertRaises(ValueError):
            self.model.generate([], max_new_tokens=4)

    def test_seed_zero_is_not_replaced_by_default(self) -> None:
        """验证 随机种子 相关行为。"""
        backend_zero = PureBackend(make_config(seed=0))
        backend_default = PureBackend(make_config(seed=1337))
        self.assertEqual(backend_zero.seed, 0)
        self.assertNotEqual(
            backend_zero.rng.getstate(), backend_default.rng.getstate()
        )
        # seed=0 可复现：同种子两次构造得到相同随机序列
        again = PureBackend(make_config(seed=0))
        self.assertEqual(
            [backend_zero.rng.random() for _ in range(8)],
            [again.rng.random() for _ in range(8)],
        )

    def test_seed_zero_changes_generated_sequence(self) -> None:
        """验证 随机种子 相关行为。"""
        model = self.backend.build_model(make_config(seed=0), vocab_size=13)
        kwargs = dict(max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0, eos_id=None)
        zero_a = PureBackend(make_config(seed=0)).generate(model, [1, 2, 3], **kwargs)
        zero_b = PureBackend(make_config(seed=0)).generate(model, [1, 2, 3], **kwargs)
        default = PureBackend(make_config(seed=1337)).generate(model, [1, 2, 3], **kwargs)
        self.assertEqual(zero_a, zero_b)  # seed=0 可复现
        self.assertNotEqual(zero_a, default)  # seed=0 不再被静默换成 1337

    def test_missing_seed_falls_back_to_default(self) -> None:
        """验证 随机种子 相关行为。"""
        values = make_config().__dict__
        values.pop("seed")
        backend = PureBackend(SimpleNamespace(**values))
        self.assertEqual(backend.seed, 1337)

    def test_default_rng_uses_global_random_not_magic_seed(self) -> None:
        # 缺省 rng=None 时使用全局 random 模块，而不是固定的 Random(1337)。
        """验证 随机数状态、随机种子 相关行为。"""
        kwargs = dict(max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0, eos_id=None)
        random.seed(42)
        first = self.model.generate([1, 2, 3], **kwargs)
        random.seed(42)
        second = self.model.generate([1, 2, 3], **kwargs)
        random.seed(43)
        third = self.model.generate([1, 2, 3], **kwargs)
        self.assertEqual(first, second)  # 全局种子可复现
        self.assertNotEqual(first, third)  # 不同全局种子 -> 不同序列
        random.seed(1337)
        magic = self.model.generate([1, 2, 3], rng=random.Random(1337), **kwargs)
        random.seed(42)
        self.assertNotEqual(self.model.generate([1, 2, 3], **kwargs), magic)

    def test_default_stream_rng_uses_global_random(self) -> None:
        """验证 流式输出、随机数状态 相关行为。"""
        kwargs = dict(max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0, eos_id=None)
        random.seed(42)
        first = [item["token_id"] for item in self.model.generate_stream([1, 2, 3], **kwargs)]
        random.seed(43)
        second = [item["token_id"] for item in self.model.generate_stream([1, 2, 3], **kwargs)]
        self.assertNotEqual(first, second)

    def test_state_roundtrip_and_generate(self) -> None:
        """验证 生成 相关行为。"""
        state = self.backend.state_dict(self.model)
        clone = PureGPT(self.model.config)
        self.backend.load_state_dict(clone, state)
        for name, parameter in self.model.params.items():
            self.assertEqual(parameter.data, clone.params[name].data, name)

        generated = self.backend.generate(
            self.model, [1, 2, 3], max_new_tokens=4, temperature=0, top_k=40, top_p=1.0, eos_id=None
        )
        self.assertEqual(len(generated), 7)

        steps = list(
            self.backend.generate_stream(
                self.model, [1, 2, 3], 3, 0.8, 10, 1.0, None, trace_top_k=3
            )
        )
        self.assertEqual([item["step"] for item in steps], [1, 2, 3])
        self.assertTrue(all(len(item["topk"]) == 3 for item in steps))

    def test_checkpoint_roundtrip_and_corruption(self) -> None:
        """验证 checkpoint 相关行为。"""
        tokenizer = CharTokenizer.build("abcde")
        config = Config()
        config.backend_type = "pure"
        payload = {
            "format_version": 1,
            "backend": "pure",
            "backend_version": "1",
            "stage": "pretrain",
            "model": self.backend.state_dict(self.model),
            "optimizer": self.backend.optimizer_state_dict(self.optimizer),
            "iter_num": 3,
            "best_val_loss": 1.25,
            "config": config.to_dict(portable=True),
            "model_config": self.model.config.to_dict(),
            "tokenizer_state": tokenizer.state_dict(),
            "data_meta": {"dataset": "unit"},
            "rng": {"python": random.getstate()},
            "early_stopping": {},
            "python_version": "3.14",
            "torch_version": "",
            "created_at": "2026-01-01T00:00:00+00:00",
            "run_id": "unit-pure",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.mgckpt"
            save_pure_checkpoint(path, payload)
            self.assertTrue(is_pure_checkpoint(path))

            loaded = load_pure_checkpoint(path)
            self.assertEqual(loaded["iter_num"], 3)
            self.assertEqual(loaded["stage"], "pretrain")
            self.assertEqual(sorted(loaded["model"]), sorted(payload["model"]))
            for name, (shape, values) in payload["model"].items():
                loaded_shape, loaded_values = loaded["model"][name]
                self.assertEqual(tuple(loaded_shape), tuple(shape))
                for left, right in zip(loaded_values, values):
                    self.assertAlmostEqual(left, right, places=6)
            self.assertEqual(loaded["optimizer"]["step_count"], payload["optimizer"]["step_count"])
            self.assertEqual(len(loaded["optimizer"]["m"]), len(payload["optimizer"]["m"]))

            raw = bytearray(path.read_bytes())
            raw[-1] ^= 0xFF
            path.write_bytes(bytes(raw))
            with self.assertRaises(PureCheckpointError):
                load_pure_checkpoint(path)

    def test_bundle_loader_and_chat_dispatch(self) -> None:
        """验证 模型包、多轮对话 相关行为。"""
        tokenizer = CharTokenizer.build("hello mini gpt pure\n" * 3)
        model = self.backend.build_model(self.config, vocab_size=tokenizer.vocab_size)
        config = Config()
        config.backend_type = "pure"
        payload = {
            "format_version": 1,
            "backend": "pure",
            "backend_version": "1",
            "stage": "sft",
            "model": self.backend.state_dict(model),
            "optimizer": self.backend.optimizer_state_dict(self.optimizer),
            "iter_num": 1,
            "best_val_loss": 1.0,
            "config": config.to_dict(portable=True),
            "model_config": model.config.to_dict(),
            "tokenizer_state": tokenizer.state_dict(),
            "data_meta": {},
            "rng": {"python": random.getstate()},
            "early_stopping": {},
            "python_version": "3.14",
            "torch_version": "",
            "created_at": "2026-01-01T00:00:00+00:00",
            "run_id": "unit-sft",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.mgckpt"
            save_pure_checkpoint(path, payload)
            bundle = load_pure_bundle(path)
            self.assertEqual(bundle.backend.name, "pure")
            self.assertEqual(bundle.payload["run_id"], "unit-sft")
            self.assertEqual(bundle.tokenizer.vocab_size, tokenizer.vocab_size)

            import chat

            dispatched = chat._load_bundle(str(path), "cpu")
            self.assertEqual(dispatched.backend.name, "pure")
            answer = chat._answer(dispatched, "hi", max_new_tokens=2, temperature=0, top_k=40, top_p=1.0)
            self.assertIsInstance(answer, str)


if __name__ == "__main__":
    unittest.main()
