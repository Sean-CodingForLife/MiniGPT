"""把 MiniGPT 训练 checkpoint 导出为可移植的推理包（MiniGPT Bundle v1）。

用法::

    python export.py --checkpoint runs/demo-en-sft/best.pt
    python export.py --checkpoint runs/demo-en-sft/best.pt --output dist/my-model
    python export.py --verify dist/my-model

输出目录包含：config、权重、tokenizer、模型卡（MODEL_CARD.md）、可独立运行的
``run.py``、仅依赖标准库的 ``serve.py``，以及记录各文件 sha256 的
``MANIFEST.json``。全过程不依赖 ``transformers``、``safetensors`` 或
``huggingface_hub``，便于离线分发。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  （把项目根目录加入 sys.path）
from minigpt.bundle import export_bundle, verify_bundle
from minigpt.config import default_project_root, overlay_config


def build_parser() -> argparse.ArgumentParser:
    """构造导出/校验命令行参数。

    这里只声明本脚本特有的参数；公共配置项由 ``overlay_config`` 统一补齐，
    因此 ``--config``、``--preset`` 等同样可用。
    """
    parser = argparse.ArgumentParser(description="Export / verify a MiniGPT Bundle v1")
    parser.add_argument("--checkpoint", default=None, help="training best.pt to export")
    parser.add_argument("--output", default=None, help="bundle directory (default dist/<run>)")
    parser.add_argument("--no-zip", action="store_true", help="do not also create a .zip")
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    parser.add_argument("--verify", default=None, help="verify an existing bundle directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """导出或校验 bundle，返回进程退出码：0 成功、2 参数错误。

    参数:
        argv: 命令行参数列表；为 ``None`` 时读取 ``sys.argv``。
    """
    args = overlay_config(build_parser(), argv, stage="pretrain")
    if args.verify:
        # 校验模式只读：重算 MANIFEST 中的哈希并输出格式与 bundle 名。
        result = verify_bundle(args.verify)
        print(json.dumps({"verified": True, "bundle": result["bundle"], "format": result["config"].get("format")}, ensure_ascii=False, indent=2))
        return 0
    if not args.checkpoint:
        print("--checkpoint is required unless --verify is used", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.output:
        # 显式输出目录：相对路径按项目根解析，而不是按当前工作目录。
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = default_project_root() / output
        output = output.resolve()
    else:
        # 默认约定：dist/<checkpoint 所在 run 目录名>，与现有 dist/ 布局一致。
        output = default_project_root() / "dist" / checkpoint.parent.name
    result = export_bundle(
        checkpoint=checkpoint,
        output_dir=output,
        create_zip=not args.no_zip,
        force=args.force,
    )
    # 给调用方（人或脚本）输出一段稳定结构的 JSON 摘要。
    print(json.dumps(
        {
            "bundle": result["bundle"],
            "zip": result["zip"],
            "format": result["config"]["format"],
            "run_id": result["config"]["source"]["run_id"],
            "stage": result["config"]["source"]["stage"],
            "files": len(result["manifest"]["files"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
