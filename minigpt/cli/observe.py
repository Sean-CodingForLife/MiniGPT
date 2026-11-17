"""`observe` 命令的规范包入口（兼容别名）。

实现位于仓库根目录的 ``observe.py``；本模块把根脚本以模块别名方式加载到
``sys.modules``，使 ``minigpt.cli.observe`` 与顶层 ``observe`` 是同一个模块对象。
因此 ``main``、内部符号与 mock.patch 行为与直接用根脚本完全一致。

支持三种调用：``python observe.py``（根脚本）、``python -m minigpt.cli.observe``
（本模块）、以及 console_scripts ``minigpt-observe``（指向 main 对象）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# 仓库根目录：本文件位于 <root>/minigpt/cli/，向上两级即 root。
_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("observe", _ROOT / "observe.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - 防御性
    raise ImportError("无法定位根脚本 observe.py")

_impl = importlib.util.module_from_spec(_SPEC)
# 先注册再执行，保证脚本内部 `import _bootstrap` / 相互引用能解析到同一实例。
sys.modules["observe"] = _impl
_SPEC.loader.exec_module(_impl)
# 把当前包入口也指向同一对象：minigpt.cli.observe is observe
sys.modules[__name__] = _impl

# 支持 `python -m minigpt.cli.observe`：此时 __name__ == "__main__"。
if __name__ == "__main__":
    raise SystemExit(_impl.main())
