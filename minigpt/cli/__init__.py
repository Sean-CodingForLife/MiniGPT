"""命令行入口（规范包）。

本包用「模块别名」把每个入口暴露为 ``minigpt.cli.<name>``，同时与仓库根目录
的同名脚本指向**同一个模块对象**。这样既提供包内/console_scripts 的标准入口，
又不破坏历史上 ``import <name>`` / ``python <name>.py`` 的引用（文档、测试、
Web 面板与打包模板均依赖后者）。

设计要点
--------
- 绝对路径运行：每个脚本自身 ``import _bootstrap`` 保证任意 cwd 下可导入 minigpt。
- 单一实现：实现仍在根脚本，本包只做别名，避免出现两份会漂移的代码。
- 稳定性：``from minigpt.cli.chat import Conversation`` 与 ``from chat import
  Conversation`` 等价（同一对象），因此 mock.patch 与类型判断都成立。

新增命令请：1) 在根目录添加 ``<name>.py``（含 ``main(argv)``）；
2) 在本包添加同名别名模块；3) 在 ``pyproject.toml`` 的 ``[project.scripts]``
登记控制台命令。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅用于类型检查器，运行时不真正导入（避免循环/副作用）
    from chat import main as chat_main
    from dashboard import main as dashboard_main
    from evaluate import main as evaluate_main
    from export import main as export_main
    from observe import main as observe_main
    from prepare import main as prepare_main
    from prepare_sft import main as prepare_sft_main
    from pretrain import main as pretrain_main
    from sft import main as sft_main
    from visualize import main as visualize_main

__all__ = [
    "chat",
    "dashboard",
    "evaluate",
    "export",
    "observe",
    "prepare",
    "prepare_sft",
    "pretrain",
    "sft",
    "visualize",
]
