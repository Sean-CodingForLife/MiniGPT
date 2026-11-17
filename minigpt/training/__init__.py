"""训练子系统：训练循环、checkpoint、callback、任务管理与配置/项目存储。

包内模块分工：

* ``trainer`` / ``callbacks`` / ``checkpoint`` / ``early_stop`` —— 真正跑在
  训练子进程里，属于「实验协议」层；
* ``job_manager`` / ``job_records`` / ``config_store`` / ``project_store`` ——
  ``job_manager`` 是看板后端与训练进程之间唯一的桥，其余三个是它拆出的基础层
  （异常与校验、``configs/`` 与 ``projects/`` 存储）。

本 ``__init__`` 刻意不导入任何子模块：看板后端在没装 PyTorch 的机器上也要能
按需单独导入 ``job_manager``。模块间的公共名称通过各自的 ``__all__`` 与
``job_manager`` 的 re-export 保持导入路径稳定。
"""
