"""MiniGPT：从零实现字符级语言模型的实验平台。

包结构概览（按依赖方向由内向外）：

* ``pure``：仅用 Python 标准库实现的张量、GPT、AdamW、检查点与推理引擎，
  是无 torch 环境下跑通全流程的参照实现；
* ``sampling``：torch 与 pure 双后端共用的采样规则内核（只操作 list/标量）；
* ``tokenization`` / ``models`` / ``backends`` / ``data`` / ``eval``：tokenizer、
  GPT 模型、计算后端、数据集与指标的注册式实现；
* ``config`` / ``registry`` / ``utils`` / ``cli_utils``：配置、注册表与通用工具；
* ``training`` / ``web``：训练循环与可视化后端（上层，依赖上述模块）。

依赖约束（有意为之）：

* 训练 / 数据处理：Python 标准库 + PyTorch（Stage 1）；
* web 后端：只用 Python 标准库；
* 前端：Chart.js 本地内置，不依赖 CDN。

本模块**不得**导入 torch：无 PyTorch 的机器上控制台同样会 ``import minigpt``，
一旦在此处引入 torch 就会让整个后端启动失败。
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
