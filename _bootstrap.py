"""项目引导（bootstrap）。

用途
----
让所有命令行入口可以在**任意当前工作目录**下运行::

    python C:\\path\\to\\MiniGPT\\prepare.py ...

Python 默认把「脚本所在目录」加入 ``sys.path``，因此即使从其它目录启动，
也能通过导入本模块把仓库根目录放进 ``sys.path``。每个顶层 CLI 都应在导入
``minigpt`` 之前先 ``import _bootstrap``。

部署时可用环境变量 ``MINIGPT_ROOT`` 覆盖项目根目录；本模块不硬编码任何
机器相关的绝对路径。
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# PyTorch 在导入时会探测 NumPy，缺失时发出警告。NumPy 是本项目刻意不引入的
# 依赖，所以只精确屏蔽这一条警告，不影响其它警告的可见性。
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

# 项目根目录：优先取 MINIGPT_ROOT，否则取本文件所在目录（即仓库根）。
PROJECT_ROOT = Path(os.environ.get("MINIGPT_ROOT", Path(__file__).resolve().parent)).resolve()

# 保证后续 ``import minigpt`` 无论从哪个 cwd 启动都能生效。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 统一把标准输出/错误切成 UTF-8，避免 Windows 默认代码页把中文样本打成乱码；
# reconfigure 在部分被重定向的流上不可用，失败时静默降级。
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__all__ = ["PROJECT_ROOT"]
