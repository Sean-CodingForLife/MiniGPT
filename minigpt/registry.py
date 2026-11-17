"""极简的「名字 -> 实现」注册表集合。

这是模型、tokenizer、数据源、callback 与 backend 的扩展接缝。刻意做得很小
（没有插件市场、没有 entry points）：一个装饰器完成注册，配置里的字符串
选择实现，``--plugin path/to/module.py`` 可以导入本地模块执行额外注册。

注册名在同一注册表内唯一：重复注册会立刻报错，而不是静默覆盖。

示例
----
    @CALLBACKS.register("my_logger")
    class MyLogger:
        def on_eval_end(self, state):
            print("eval", state.step, state.val_loss)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """「名字 -> 对象」映射，错误信息带上可用名字列表以便排错。"""

    def __init__(self, name: str):
        """创建空注册表。

        参数:
            name: 注册表的人类可读名字（如 ``"tokenizer"``），仅用于报错信息。
        """
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """返回装饰器，把被装饰对象登记到 ``name`` 下。

        参数:
            name: 注册名；首尾空白会被裁掉，空名视为错误。
        异常:
            ValueError: 名字为空或在同一注册表内重复。
        """

        def decorator(obj: T) -> T:
            """完成实际登记，原样返回被装饰对象。"""
            key = name.strip()
            if not key:
                raise ValueError(f"{self.name}: empty registration name")
            if key in self._items:
                raise ValueError(f"{self.name}: duplicate registration {key!r}")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """按名字取对象；未注册时抛 KeyError 并列出全部可用名字。"""
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"{self.name}: unknown {name!r}; available: {available}") from exc

    def names(self) -> list[str]:
        """返回已注册名字的字典序列表（用于展示 CLI 选项）。"""
        return sorted(self._items)

    def create(self, name: str, **kwargs: Any) -> Any:
        """取出工厂对象并以关键字参数实例化。

        参数:
            name: 已注册的工厂名。
            **kwargs: 原样透传给工厂（通常是类的构造参数）。
        异常:
            TypeError: 该名字登记的不是可调用对象。
        """
        factory = self.get(name)
        if not callable(factory):
            raise TypeError(f"{self.name}: {name!r} is not callable")
        return factory(**kwargs)


TOKENIZERS: Registry[Any] = Registry("tokenizer")
MODELS: Registry[Any] = Registry("model")
DATA_SOURCES: Registry[Any] = Registry("data_source")
CALLBACKS: Registry[Any] = Registry("callback")
BACKENDS: Registry[Any] = Registry("backend")


def load_plugin(path: str | Path) -> None:
    """导入本地 Python 文件，使其导入期的注册副作用生效。

    只接受显式本地路径；这是面向实验/学习的特性，不是沙箱——插件代码拥有
    与主进程相同的权限。

    参数:
        path: 插件文件路径，支持 ``~`` 与相对路径（按当前工作目录解析）。
    异常:
        FileNotFoundError: 路径不存在或不是文件。
        ImportError: 无法为该路径构造 loader，或模块执行失败。
    副作用:
        模块会以 ``minigpt_plugin_<hash>`` 为名执行，但**不会**写入
        ``sys.modules``，因此重复加载同一文件会重复执行注册逻辑。
    """

    plugin_path = Path(path).expanduser().resolve()
    if not plugin_path.is_file():
        raise FileNotFoundError(f"plugin not found: {plugin_path}")
    module_name = f"minigpt_plugin_{abs(hash(str(plugin_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin: {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


__all__ = [
    "Registry",
    "TOKENIZERS",
    "MODELS",
    "DATA_SOURCES",
    "CALLBACKS",
    "BACKENDS",
    "load_plugin",
]
