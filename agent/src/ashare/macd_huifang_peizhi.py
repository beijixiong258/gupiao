"""MACD 历史研究的共享配置与校验，不加载行情或回放计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MacdHuifangPeizhi:
    """固定的研究回放口径。"""

    forward_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    decision_horizons: tuple[int, ...] = (5, 10, 20)
    minimum_signal_history_sessions: int = 80
    amount_window_sessions: int = 20
    target_notional_yuan: float = 20_000.0
    minimum_trailing_amount_yuan: float = 50_000_000.0
    maximum_participation_rate: float = 0.005
    minimum_baseline_peers_per_date: int = 5
    minimum_signal_samples: int = 30
    minimum_stocks: int = 30
    minimum_sample_scopes: int = 2
    stability_subperiods: int = 4
    minimum_valid_subperiods: int = 3
    minimum_favorable_sign_agreement: float = 0.75
    confidence_z: float = 1.96
    parameter_sensitivity_enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MacdHuifangPeizhi":
        if not value:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        if unknown:
            raise ValueError(f"MACD 回放配置包含未知字段：{', '.join(sorted(unknown))}")
        normalized = {key: value[key] for key in fields if key in value}
        for key in ("forward_horizons", "decision_horizons"):
            if key in normalized:
                normalized[key] = tuple(int(item) for item in normalized[key])
        settings = cls(**normalized)
        settings._validate()
        return settings

    def _validate(self) -> None:
        if not self.forward_horizons or any(item <= 0 for item in self.forward_horizons):
            raise ValueError("forward_horizons 必须是非空的正整数序列")
        if tuple(sorted(set(self.forward_horizons))) != self.forward_horizons:
            raise ValueError("forward_horizons 必须严格递增且不能重复")
        if not self.decision_horizons or not set(self.decision_horizons).issubset(self.forward_horizons):
            raise ValueError("decision_horizons 必须是 forward_horizons 的非空子集")
        integer_values = (
            self.minimum_signal_history_sessions,
            self.amount_window_sessions,
            self.minimum_baseline_peers_per_date,
            self.minimum_signal_samples,
            self.minimum_stocks,
            self.minimum_sample_scopes,
            self.stability_subperiods,
            self.minimum_valid_subperiods,
        )
        if any(int(item) != item or item <= 0 for item in integer_values):
            raise ValueError("MACD 回放窗口和样本门槛必须是正整数")
        if self.minimum_valid_subperiods > self.stability_subperiods:
            raise ValueError("minimum_valid_subperiods 不能大于 stability_subperiods")
        if self.target_notional_yuan <= 0 or self.minimum_trailing_amount_yuan < 0:
            raise ValueError("回放资金和成交额门槛无效")
        if not 0 < self.maximum_participation_rate <= 0.1:
            raise ValueError("maximum_participation_rate 必须在 0 到 0.1 之间")
        if not 0.5 <= self.minimum_favorable_sign_agreement <= 1:
            raise ValueError("minimum_favorable_sign_agreement 必须在 0.5 到 1 之间")
        if not 0 < self.confidence_z <= 5:
            raise ValueError("confidence_z 必须在 0 到 5 之间")
