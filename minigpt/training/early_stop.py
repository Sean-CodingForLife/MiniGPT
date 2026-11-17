"""早停（early stopping）状态机：显式状态，可随 checkpoint 保存与恢复。

训练器「何时停止」的唯一判定点：每轮验证后调用 :meth:`EarlyStopping.update`
得到「本次是否刷新最优」，再读 :attr:`EarlyStopping.should_stop` 决定是否
结束训练并写 ``training_report.json``。全部计数器都进入 checkpoint 的
``early_stopping`` 字段，续训时原样恢复，早停节奏不会因为重启而重置。

约定：``patience`` / ``min_delta`` 属于**当前配置**而非历史状态——续训时以
最新配置为准（详见 :meth:`EarlyStopping.load_state_dict`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EarlyStopping:
    """监控单个指标（验证 loss）的早停状态。

    字段语义：``best`` 是历史最优指标值（越小越好，初值 +inf），
    ``best_step`` 是最优值出现时的步号（``-1`` 表示还没有最优值），
    ``num_bad_evals`` 是连续「未刷新最优」的验证次数。整体是可 JSON /
    pickle 序列化的纯数据对象，便于写入 checkpoint。
    """

    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    best_step: int = -1
    num_bad_evals: int = 0

    def update(self, value: float, step: int) -> bool:
        """用一次新的验证值更新状态；刷新最优时返回 ``True``。

        判定条件是严格优于 ``best - min_delta``（``min_delta`` 用于过滤噪声
        级抖动）。未刷新最优时只递增 ``num_bad_evals``，不改 ``best``。
        """

        if value < self.best - self.min_delta:
            self.best = value
            self.best_step = step
            self.num_bad_evals = 0
            return True
        self.num_bad_evals += 1
        return False

    @property
    def should_stop(self) -> bool:
        """连续未刷新最优的次数达到 ``patience`` 时为 ``True``。

        ``patience <= 0`` 表示关闭早停（永不因该条件停止）。
        """

        return self.patience > 0 and self.num_bad_evals >= self.patience

    def state_dict(self) -> dict[str, Any]:
        """导出可写入 checkpoint 的状态（含 patience/min_delta 仅供参考）。"""

        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_step": self.best_step,
            "num_bad_evals": self.num_bad_evals,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """从 checkpoint 恢复计数与最优值（不清空当前 patience/min_delta）。

        参数 ``state`` 来自 :meth:`state_dict`；缺失字段回落到当前值，
        因此旧 checkpoint（字段更少）也能恢复。
        """

        # patience/min_delta 始终取自**当前配置**：续训时换了停止条件必须立刻
        # 生效。checkpoint 里的对应值只是记录，这里只恢复计数器与最优值。
        self.best = float(state.get("best", self.best))
        self.best_step = int(state.get("best_step", self.best_step))
        self.num_bad_evals = int(state.get("num_bad_evals", self.num_bad_evals))


__all__ = ["EarlyStopping"]
