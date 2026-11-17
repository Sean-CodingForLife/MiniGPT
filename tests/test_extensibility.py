"""扩展点测试：后端协议、插件注册表与自定义 callback 是否真正生效。

项目承诺「不改核心代码即可扩展」，这里用最小假数据源和动态生成的插件
验证注册表查找稳定（同名返回同一对象）以及训练循环会调用回调钩子。
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

try:
    from minigpt.backends.torch_backend import TorchBackend
except ImportError:
    raise unittest.SkipTest("torch not installed") from None

from minigpt import registry
from minigpt.backends.base import ModelBackend
from minigpt.config import Config
from minigpt.registry import load_plugin
from minigpt.training.trainer import Trainer


class FakeTokenizer:
    """满足 Trainer 所需的最小 tokenizer 接口，省去构建真实词表。"""

    vocab_size = 20
    eos_id = 1
    pad_id = 2

    def state_dict(self):
        return {"type": "fake", "vocab_size": 20, "specials": {"<eos>": 1, "<pad>": 2}}


class FakeSource:
    """确定性数据源：用取模构造 x/y，使训练 2 步即可完成且无需磁盘数据。"""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.metadata = {"fake": True}

    def train_batch(self, batch_size, block_size, rng=None):
        x = [[(i + j) % 20 for j in range(block_size)] for i in range(batch_size)]
        return {"x": x, "y": [[(v + 1) % 20 for v in row] for row in x]}

    def eval_batches(self, split, batch_size, block_size, num_batches, seed=1234):
        return [self.train_batch(batch_size, block_size) for _ in range(max(1, num_batches))]


class ExtensibilityTests(unittest.TestCase):
    """验证注册表插件加载与 callbacks 配置项的实际执行链路。"""

    def test_protocol_registration_and_plugin_loading(self) -> None:
        """TorchBackend 需满足 ModelBackend 协议；外部插件注册后可按名取到同一实现。"""
        self.assertIsInstance(TorchBackend(Config()), ModelBackend)

        # 用 uuid 后缀避免与其它测试/多次运行的同名注册冲突。
        name = f"dummy_{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory() as tmp:
            plugin = Path(tmp) / "plugin.py"
            plugin.write_text(
                "from minigpt import registry\n"
                "class Dummy:\n"
                "    pass\n"
                f"registry.TOKENIZERS.register({name!r})(Dummy)\n",
                encoding="utf-8",
            )
            load_plugin(plugin)
        self.assertIn(name, registry.TOKENIZERS.names())
        self.assertIs(registry.TOKENIZERS.get(name), registry.TOKENIZERS.get(name))

    def test_custom_callback_is_invoked(self) -> None:
        """callbacks 列表里的自定义类必须被实例化并收到 train_end，留下可观测落盘证据。"""
        callback_name = f"marker_{uuid.uuid4().hex[:8]}"
        marker_name = f"{callback_name}.txt"

        @registry.CALLBACKS.register(callback_name)
        class MarkerCallback:
            # 把钩子执行结果写成文件，便于断言回调确实在训练流程中运行过。
            """覆盖 MarkerCallback 相关行为。"""
            def __init__(self, config, run_dir):
                self.path = Path(run_dir) / marker_name

            def on_train_start(self, state):
                self.path.write_text(f"start:{state.step}", encoding="utf-8")

            def on_eval_end(self, state):
                self.path.write_text(f"eval:{state.step}", encoding="utf-8")

            def on_train_end(self, state):
                self.path.write_text(f"end:{state.step}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            # 极小模型 + 单步评估，把训练成本压到秒级。
            config = Config(
                project_root=tmp,
                data_dir=tmp,
                runs_dir=str(Path(tmp) / "runs"),
                run_name="extension-run",
                n_layer=1,
                n_head=2,
                n_embd=32,
                block_size=8,
                dropout=0.0,
                batch_size=2,
                max_steps=2,
                eval_interval=1,
                eval_iters=1,
                save_interval=10,
                warmup_steps=1,
                patience=5,
                callbacks=f"telemetry,{callback_name}",
            )
            backend = TorchBackend(config)
            model = backend.build_model(config, 20)
            optimizer = backend.build_optimizer(model, config)
            source = FakeSource()
            trainer = Trainer(config, backend, model, optimizer, source.tokenizer, source)
            trainer.train()
            marker = trainer.run_dir / marker_name
            self.assertTrue(marker.is_file())
            self.assertIn("end:", marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
