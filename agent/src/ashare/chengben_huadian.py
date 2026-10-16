"""Fixed internal A-share cost assumptions for research estimates."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_NOTIONAL_YUAN = 20_000.0
DEFAULT_COST_SOURCE = "built_in_research_assumption"


@dataclass(frozen=True)
class CostScenario:
    name: str
    buy_commission_rate: float
    sell_commission_rate: float
    stamp_tax_sell_rate: float
    transfer_fee_buy_rate: float
    transfer_fee_sell_rate: float
    buy_slippage_bps: float
    sell_slippage_bps: float
    min_commission_yuan: float


DEFAULT_SCENARIO = CostScenario(
    "research_reference",
    0.00025,
    0.00025,
    0.0005,
    0.00001,
    0.00001,
    2.0,
    2.0,
    5.0,
)


def _commission_rate(rate: float, min_commission_yuan: float, notional_yuan: float) -> float:
    if notional_yuan > 0:
        return max(rate, min_commission_yuan / notional_yuan)
    return rate


def reference_roundtrip_cost_rate() -> float:
    """既定研究情景的往返比例成本；不是实际委托成交费用。"""
    scenario = DEFAULT_SCENARIO
    buy = (_commission_rate(scenario.buy_commission_rate, scenario.min_commission_yuan, DEFAULT_NOTIONAL_YUAN)
           + scenario.transfer_fee_buy_rate + scenario.buy_slippage_bps / 10000)
    sell = (_commission_rate(scenario.sell_commission_rate, scenario.min_commission_yuan, DEFAULT_NOTIONAL_YUAN)
            + scenario.transfer_fee_sell_rate + scenario.stamp_tax_sell_rate + scenario.sell_slippage_bps / 10000)
    return round(1 - (1 - buy) * (1 - sell), 6)


__all__ = [
    "CostScenario",
    "DEFAULT_COST_SOURCE",
    "DEFAULT_NOTIONAL_YUAN",
    "DEFAULT_SCENARIO",
    "reference_roundtrip_cost_rate",
]
