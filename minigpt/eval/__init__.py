"""评估指标。

当前只在 :mod:`minigpt.eval.metrics` 中实现语言模型的 loss/perplexity 与
字符级错误率统计；本 ``__init__`` 有意不做再导出，避免调用方依赖不稳定的
顶层符号。
"""
