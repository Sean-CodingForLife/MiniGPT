"""配置系统（preset / 配置文件 / 环境变量 / CLI 覆盖、校验与可移植序列化）测试。

优先级和校验口径直接决定训练能否启动与能否迁移：这里锁定 CLI > 文件
（以及 env 与 preset 的关系）、路径解析为绝对路径、非法超参在 validate
阶段就被拒绝、纯 Python 后端只接受无 bias/权重共享的架构。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from minigpt.config import Config, get_preset, parse_config


class ConfigTests(unittest.TestCase):
    """验证配置合并优先级、参数校验和跨机器保存时的相对路径处理。"""

    def test_presets_are_copies(self) -> None:
        """get_preset 必须每次返回新对象，修改调用方手里的配置不能污染全局预设。"""
        first = get_preset("debug_en")
        first.n_layer = 99
        self.assertEqual(get_preset("debug_en").n_layer, 4)

    def test_cli_overrides_config_and_env(self) -> None:
        """同一字段同时存在 CLI/文件/env 时以 CLI 为准，其余字段回落到文件值，且根路径为绝对。"""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"n_layer": 5, "n_head": 4}), encoding="utf-8")
            # 临时注入环境变量，结束后按原值精确还原，避免污染其它测试。
            old = os.environ.get("MINIGPT_N_LAYER")
            os.environ["MINIGPT_N_LAYER"] = "3"
            try:
                config = parse_config(
                    "pretrain",
                    ["--preset", "debug_en", "--config", str(config_path), "--n_layer", "2"],
                )
            finally:
                if old is None:
                    os.environ.pop("MINIGPT_N_LAYER", None)
                else:
                    os.environ["MINIGPT_N_LAYER"] = old
            self.assertEqual(config.n_layer, 2)
            self.assertEqual(config.n_head, 4)
            self.assertTrue(Path(config.project_root).is_absolute())

    def test_env_and_boolean(self) -> None:
        """布尔型环境变量（MINIGPT_BIAS=true）需被正确解析成 True 而非非空字符串。"""
        old = os.environ.get("MINIGPT_BIAS")
        os.environ["MINIGPT_BIAS"] = "true"
        try:
            config = parse_config("pretrain", ["--preset", "debug_en"])
        finally:
            if old is None:
                os.environ.pop("MINIGPT_BIAS", None)
            else:
                os.environ["MINIGPT_BIAS"] = old
        self.assertTrue(config.bias)

    def test_invalid_model_dimensions(self) -> None:
        """n_embd 不能被 n_head 整除时必须在 validate 阶段报错，避免训练中途崩在 reshape。"""
        config = Config(n_embd=10, n_head=3)
        with self.assertRaises(ValueError):
            config.validate()

    def test_portable_dict(self) -> None:
        """portable 序列化需把绝对路径降级为项目内相对路径，保证配置可迁移。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(project_root=tmp, data_dir="data/x", runs_dir="runs")
            config.resolve_paths()
            portable = config.to_dict(portable=True)
            self.assertEqual(portable["data_dir"], str(Path("data") / "x"))
            self.assertEqual(portable["runs_dir"], "runs")

    def test_list_presets_exits(self) -> None:
        """--list_presets 只做展示，应以 SystemExit 结束进程而不继续训练。"""
        with self.assertRaises(SystemExit):
            parse_config("pretrain", ["--list_presets"])

    def test_optimizer_validation(self) -> None:
        """梯度裁剪、学习率下界、Adam beta 与权重衰减的越界取值都要被拒绝。"""
        for overrides in (
            {"grad_clip": 0.0},
            {"grad_clip": -1.0},
            {"min_lr": 1.0, "lr": 1e-3},
            {"beta1": 1.0},
            {"beta2": -0.1},
            {"weight_decay": -1.0},
        ):
            with self.assertRaises(ValueError, msg=str(overrides)):
                Config(**overrides).validate()

    def test_pure_backend_rejects_bias_and_untied(self) -> None:
        """pure 后端不支持 bias 与解绑词嵌入，配置阶段就应报错而非运行期崩溃。"""
        with self.assertRaises(ValueError):
            Config(backend_type="pure", bias=True).validate()
        with self.assertRaises(ValueError):
            Config(backend_type="pure", tie_weights=False).validate()
        Config(backend_type="pure").validate()


if __name__ == "__main__":
    unittest.main()
