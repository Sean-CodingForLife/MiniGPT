"""模型实现。

目前只有 :mod:`minigpt.models.gpt`（torch 版 GPT 与 ``GPTConfig``）；纯标准库
的同构实现在 :mod:`minigpt.pure.gpt`。本 ``__init__`` 不做再导出，调用方应
直接导入具体子模块，以便无 torch 环境不会被动加载 torch。
"""
