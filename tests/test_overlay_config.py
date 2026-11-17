"""阶段 3：overlay_config 统一配置叠加机制的行为测试。

覆盖：无配置层保留入口默认值、preset/config/env/显式 CLI 的优先级、
field_map 映射、布尔旗标、必填旗标在 --help 场景不误报。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from minigpt.config import overlay_config


def make_parser() -> argparse.ArgumentParser:
    """构造与真实推理入口同形的参数表，覆盖 int/float/str/store_true 与 required 各类字段。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--stream", action="store_true")
    return parser


class OverlayConfigTests(unittest.TestCase):
    """锁定配置层叠顺序：入口默认 < preset < config 文件 < env < 显式 CLI。"""

    def test_no_layer_keeps_entry_defaults(self) -> None:
        """不提供任何配置层时，参数必须保持 add_argument 的默认值不被改写。"""
        args = overlay_config(make_parser(), ["--checkpoint", "x.pt"], stage="sft")
        self.assertEqual(args.max_new_tokens, 40)
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.prompt, "ROMEO:")
        self.assertFalse(args.stream)

    def test_preset_overrides_defaults(self) -> None:
        """指定 preset 后，其字段应覆盖入口默认值。"""
        args = overlay_config(
            make_parser(), ["--checkpoint", "x.pt", "--preset", "pure_debug_en"], stage="pretrain"
        )
        # pure_debug_en 的 max_new_tokens 沿用 Config 默认 200。
        self.assertEqual(args.max_new_tokens, 200)

    def test_explicit_cli_wins_over_preset(self) -> None:
        """命令行显式给出的值优先级最高，preset 不能覆盖它。"""
        args = overlay_config(
            make_parser(),
            ["--checkpoint", "x.pt", "--preset", "pure_debug_en", "--max_new_tokens", "7"],
            stage="pretrain",
        )
        self.assertEqual(args.max_new_tokens, 7)

    def test_config_file_overrides_preset(self) -> None:
        """配置文件层需高于 preset，低于显式 CLI 与 env。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.json"
            cfg.write_text(json.dumps({"max_new_tokens": 13}), encoding="utf-8")
            args = overlay_config(
                make_parser(),
                ["--checkpoint", "x.pt", "--preset", "pure_debug_en", "--config", str(cfg)],
                stage="pretrain",
            )
            self.assertEqual(args.max_new_tokens, 13)

    def test_env_overrides_config_file(self) -> None:
        """环境变量层高于配置文件，便于容器部署时统一改参。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "c.json"
            cfg.write_text(json.dumps({"max_new_tokens": 13}), encoding="utf-8")
            old = os.environ.get("MINIGPT_MAX_NEW_TOKENS")
            os.environ["MINIGPT_MAX_NEW_TOKENS"] = "21"
            try:
                args = overlay_config(
                    make_parser(),
                    ["--checkpoint", "x.pt", "--config", str(cfg)],
                    stage="pretrain",
                )
            finally:
                # 还原环境，避免影响同进程内其它配置测试。
                if old is None:
                    os.environ.pop("MINIGPT_MAX_NEW_TOKENS", None)
                else:
                    os.environ["MINIGPT_MAX_NEW_TOKENS"] = old
            self.assertEqual(args.max_new_tokens, 21)

    def test_field_map_maps_prompt_to_sample_prompt(self) -> None:
        """field_map 需把配置里的不同字段名映射到入口参数名（sample_prompt → prompt）。"""
        args = overlay_config(
            make_parser(),
            ["--checkpoint", "x.pt", "--preset", "pure_debug_zh"],
            stage="pretrain",
            field_map={"prompt": "sample_prompt"},
        )
        self.assertEqual(args.prompt, "三国演义：")

    def test_boolean_flag_explicit_wins(self) -> None:
        """store_true 旗标在显式出现时必须为 True，不能被默认层重置。"""
        args = overlay_config(
            make_parser(), ["--checkpoint", "x.pt", "--stream"], stage="sft"
        )
        self.assertTrue(args.stream)

    def test_help_does_not_raise_required(self) -> None:
        """--help 应在必填参数校验之前以退出码 0 结束，而不是报缺 --checkpoint。"""
        with self.assertRaises(SystemExit) as ctx:
            overlay_config(make_parser(), ["--help"], stage="sft")
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()