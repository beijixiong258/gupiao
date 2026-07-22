"""Shared A-share execution sizing, liquidity impact and transaction-cost helpers."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from src.ashare.chengben_huadian import CostScenario
from src.ashare.chengben_huadian import DEFAULT_CONFIG_PATH as COST_CONFIG_PATH
from src.ashare.chengben_huadian import _commission_rate, _load_cost_config


def _load_cost_assumption(scenario_name: str) -> tuple[float, CostScenario, str, list[str]]:
    budget, scenarios, path, errors = _load_cost_config(str(COST_CONFIG_PATH))
    scenario = next((item for item in scenarios if item.name == scenario_name), None)
    if scenario is None:
        raise ValueError(f"交易成本配置中不存在场景：{scenario_name}")
    return float(budget), scenario, path, errors


def _buy_order_rule(ts_code: str) -> tuple[int, int, str]:
    normalized = str(ts_code).upper()
    digits = normalized.split(".")[0]
    if normalized.endswith(".BJ"):
        return 100, 1, "beijing"
    if digits.startswith(("688", "689")):
        return 200, 1, "star"
    return 100, 100, "main_or_chinext"


def _valid_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _position_for_budget(
    ts_code: str,
    price: float,
    budget_yuan: float,
    *,
    daily_amount_yuan: float | None = None,
    max_participation_rate: float = 0.005,
) -> dict[str, Any]:
    minimum, increment, board = _buy_order_rule(ts_code)
    valid = math.isfinite(float(price)) and float(price) > 0 and float(budget_yuan) > 0
    affordable = int(math.floor(float(budget_yuan) / float(price) + 1e-9)) if valid else 0
    if affordable < minimum:
        shares = 0
    elif increment == 1:
        shares = affordable
    else:
        shares = affordable // increment * increment
    actual_notional = float(shares) * float(price) if shares else 0.0
    amount = _valid_positive(daily_amount_yuan)
    participation = actual_notional / amount if amount and actual_notional > 0 else None
    liquidity_feasible = participation is None or participation <= float(max_participation_rate)
    lot_feasible = shares >= minimum
    return {
        "board": board,
        "minimum_buy_shares": int(minimum),
        "buy_share_increment": int(increment),
        "target_budget_yuan": round(float(budget_yuan), 2),
        "sizing_price": round(float(price), 3) if valid else None,
        "estimated_buy_shares": int(shares),
        "estimated_buy_notional_yuan": round(actual_notional, 2),
        "budget_utilization": round(actual_notional / float(budget_yuan), 6) if budget_yuan > 0 else None,
        "daily_amount_yuan": round(amount, 2) if amount is not None else None,
        "estimated_participation_rate": round(participation, 8) if participation is not None else None,
        "maximum_participation_rate": round(float(max_participation_rate), 8),
        "lot_size_feasible": lot_feasible,
        "liquidity_capacity_feasible": liquidity_feasible,
        "execution_feasible": bool(lot_feasible and liquidity_feasible),
    }


def _dynamic_slippage_roundtrip_bps(
    *,
    notional_yuan: float,
    daily_amount_yuan: float | None,
    atr_pct: float | None,
    trading_settings: dict[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    settings = trading_settings or {}
    enabled = bool(settings.get("dynamic_slippage_enabled", True))
    maximum = max(0.0, float(settings.get("max_dynamic_slippage_bps_roundtrip", 40.0)))
    amount = _valid_positive(daily_amount_yuan)
    atr = _valid_positive(atr_pct)
    participation = float(notional_yuan) / amount if amount else None
    if not enabled:
        extra = 0.0
        liquidity_extra = 0.0
        volatility_extra = 0.0
    else:
        liquidity_extra = 6.0 * math.sqrt(max(participation or 0.0, 0.0) / 0.001) if participation else 0.0
        volatility_extra = max(0.0, (atr or 0.0) - 0.02) * 500.0
        extra = min(maximum, liquidity_extra + volatility_extra)
    return float(extra), {
        "enabled": enabled,
        "method": "成交额参与率平方根冲击 + ATR超过2%的波动附加项",
        "liquidity_extra_bps_roundtrip": round(float(liquidity_extra), 4),
        "volatility_extra_bps_roundtrip": round(float(volatility_extra), 4),
        "dynamic_extra_bps_roundtrip": round(float(extra), 4),
        "maximum_dynamic_bps_roundtrip": round(float(maximum), 4),
        "amount_or_atr_missing": bool(amount is None or atr is None),
    }


def _stock_roundtrip_cost(
    ts_code: str,
    price: float,
    budget_yuan: float,
    scenario: CostScenario,
    *,
    daily_amount_yuan: float | None = None,
    atr_pct: float | None = None,
    trading_settings: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    settings = trading_settings or {}
    position = _position_for_budget(
        ts_code,
        price,
        budget_yuan,
        daily_amount_yuan=daily_amount_yuan,
        max_participation_rate=float(settings.get("max_participation_rate", 0.005)),
    )
    notional = float(position["estimated_buy_notional_yuan"])
    if not position["execution_feasible"] or notional <= 0:
        reason = (
            "目标资金超过最新日成交额允许的参与率上限"
            if position.get("lot_size_feasible") and not position.get("liquidity_capacity_feasible")
            else "目标资金不足以按所属板块的最低买入数量建仓"
        )
        return None, {
            **position,
            "cost_scenario": scenario.name,
            "estimated_roundtrip_cost_rate": None,
            "reason": reason,
        }
    dynamic_bps, dynamic_meta = _dynamic_slippage_roundtrip_bps(
        notional_yuan=notional,
        daily_amount_yuan=daily_amount_yuan,
        atr_pct=atr_pct,
        trading_settings=settings,
    )
    buy_commission = _commission_rate(scenario.buy_commission_rate, scenario.min_commission_yuan, notional)
    sell_commission = _commission_rate(scenario.sell_commission_rate, scenario.min_commission_yuan, notional)
    dynamic_per_side = dynamic_bps / 2.0 / 10000.0
    buy_cost = (
        buy_commission
        + scenario.transfer_fee_buy_rate
        + scenario.buy_slippage_bps / 10000.0
        + dynamic_per_side
    )
    sell_cost = (
        sell_commission
        + scenario.transfer_fee_sell_rate
        + scenario.stamp_tax_sell_rate
        + scenario.sell_slippage_bps / 10000.0
        + dynamic_per_side
    )
    roundtrip = 1.0 - (1.0 - buy_cost) * (1.0 - sell_cost)
    return float(roundtrip), {
        **position,
        "cost_scenario": scenario.name,
        "estimated_roundtrip_cost_rate": round(float(roundtrip), 6),
        "configured_slippage_bps_roundtrip": round(
            float(scenario.buy_slippage_bps + scenario.sell_slippage_bps), 4
        ),
        "dynamic_slippage": dynamic_meta,
        "stamp_tax_sell_rate": scenario.stamp_tax_sell_rate,
    }


def _roundtrip_cost(scenario_name: str) -> tuple[float, dict[str, Any]]:
    notional, scenario, path, errors = _load_cost_assumption(scenario_name)
    buy_commission = _commission_rate(scenario.buy_commission_rate, scenario.min_commission_yuan, notional)
    sell_commission = _commission_rate(scenario.sell_commission_rate, scenario.min_commission_yuan, notional)
    buy_cost = buy_commission + scenario.transfer_fee_buy_rate + scenario.buy_slippage_bps / 10000.0
    sell_cost = (
        sell_commission
        + scenario.transfer_fee_sell_rate
        + scenario.stamp_tax_sell_rate
        + scenario.sell_slippage_bps / 10000.0
    )
    roundtrip = 1.0 - (1.0 - buy_cost) * (1.0 - sell_cost)
    return float(roundtrip), {
        "scenario": scenario.name,
        "config_path": path,
        "notional_yuan": float(notional),
        "reference_roundtrip_cost_rate": round(float(roundtrip), 6),
        "estimated_roundtrip_cost_rate": round(float(roundtrip), 6),
        "estimated_roundtrip_cost_rate_is_reference_only": True,
        "stamp_tax_sell_rate": scenario.stamp_tax_sell_rate,
        "capital_assumption": "个股成本会按整手、成交额参与率和ATR动态滑点重新计算，参考成本仅用于展示",
        "config_errors": errors,
    }


def _apply_cost(gross_return: float, cost_rate: float) -> float:
    return (1.0 + gross_return) * (1.0 - cost_rate) - 1.0


def _round_price_tick(value: float) -> float:
    return float(Decimal(str(max(float(value), 0.01))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _price_limit_bounds(reference_price: float, limit_rate: float | None, sessions: int) -> tuple[float, float]:
    if limit_rate is None or not math.isfinite(float(limit_rate)) or float(limit_rate) <= 0:
        return 0.01, math.inf
    lower = _round_price_tick(reference_price)
    upper = lower
    for _ in range(max(1, int(sessions))):
        lower = _round_price_tick(lower * (1.0 - float(limit_rate)))
        upper = _round_price_tick(upper * (1.0 + float(limit_rate)))
    return lower, upper


__all__ = [
    "_apply_cost",
    "_buy_order_rule",
    "_load_cost_assumption",
    "_position_for_budget",
    "_price_limit_bounds",
    "_round_price_tick",
    "_roundtrip_cost",
    "_stock_roundtrip_cost",
]
