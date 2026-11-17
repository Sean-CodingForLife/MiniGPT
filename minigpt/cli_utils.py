"""命令行入口共享的小工具（纯标准库）。

收编原先在 prepare.py / prepare_sft.py / pretrain.py / sft.py 等入口里
成对复制的函数：别名翻译、旗标探测、UTC 时间戳、输入路径解析、插件加载。
入口脚本保留各自的薄包装（别名表/字段名不同），逻辑只此一份。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def translate_argv(argv: list[str], aliases: dict[str, str]) -> list[str]:
    """把友好别名（如 --input/--url）翻译成 Config 字段参数。

    支持 ``--alias value`` 与 ``--alias=value`` 两种写法；非别名原样保留。

    参数:
        argv: 原始命令行参数列表（通常不含脚本名）。
        aliases: 别名到真实参数名的映射，键需带前导 ``--``。
    返回:
        等长的新参数列表；``--alias value`` 会改写成 ``--real value``，
        未命中的参数逐字保留，顺序不变。
    """
    translated: list[str] = []
    for arg in argv:
        key, sep, value = arg.partition("=")
        new_key = aliases.get(key)
        if new_key is None:
            translated.append(arg)
        elif sep:
            translated.append(f"{new_key}={value}")
        else:
            translated.append(new_key)
    return translated


def has_flag(argv: list[str], flag: str) -> bool:
    """argv 中是否显式出现某旗标（含 ``--flag=value`` 写法）。"""
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def now_iso() -> str:
    """UTC 时间戳（秒精度），与 utils.utc_now 同一格式。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_input_path(value: str, project_root: str | Path) -> Path:
    """把输入路径解析为绝对路径（相对路径按项目根解释）。

    参数:
        value: 用户传入的路径字符串，可含 ``~``；Windows 下也接受反斜杠。
        project_root: 项目根目录，仅在 ``value`` 为相对路径时参与拼接。
    返回:
        展开并 ``resolve()`` 后的绝对路径（不检查文件是否存在）。
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def load_plugins(config: Any) -> None:
    """按 config.plugins（逗号分隔路径）逐个加载插件模块。

    参数:
        config: 任意带 ``plugins`` 属性的配置对象；空字符串或全空白项会被跳过。
    异常:
        FileNotFoundError / ImportError: 由 :func:`minigpt.registry.load_plugin`
        在路径不存在或模块无法执行时抛出，此处不做捕获。
    副作用:
        插件模块在导入时的注册（如 ``@TOKENIZERS.register``）会写入全局注册表。
    """
    from minigpt.registry import load_plugin

    for path in [p.strip() for p in str(config.plugins).split(",") if p.strip()]:
        load_plugin(path)


__all__ = ["translate_argv", "has_flag", "now_iso", "resolve_input_path", "load_plugins"]