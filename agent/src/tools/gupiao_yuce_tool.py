"""Second-stage, request-specific forecast and position-return tool."""

from __future__ import annotations

import json
import math
from typing import Any

from src.agent.tools import BaseTool
from src.ashare.bankuai_yuce import _load_cost_assumption, _position_for_budget
from src.ashare.chengben_huadian import CostScenario
from src.tools.gupiao_analysis_cache import get_analysis


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _cost_components(notional: float, *, side: str, scenario: CostScenario) -> dict[str, float]:
    commission_rate = scenario.buy_commission_rate if side == "buy" else scenario.sell_commission_rate
    transfer_rate = scenario.transfer_fee_buy_rate if side == "buy" else scenario.transfer_fee_sell_rate
    slippage_bps = scenario.buy_slippage_bps if side == "buy" else scenario.sell_slippage_bps
    commission = max(notional * commission_rate, scenario.min_commission_yuan)
    transfer = notional * transfer_rate
    stamp_tax = notional * scenario.stamp_tax_sell_rate if side == "sell" else 0.0
    slippage = notional * slippage_bps / 10000.0
    total = commission + transfer + stamp_tax + slippage
    return {
        "commission_yuan": round(commission, 2),
        "transfer_fee_yuan": round(transfer, 2),
        "stamp_tax_yuan": round(stamp_tax, 2),
        "slippage_assumption_yuan": round(slippage, 2),
        "total_yuan": round(total, 2),
    }


def _position_return(
    *,
    ts_code: str,
    buy_price: Any,
    shares: Any,
    position_value_yuan: Any,
    current_price: Any,
    projected_price: Any,
    scenario: CostScenario,
) -> dict[str, Any] | None:
    parsed_buy = _finite_positive(buy_price)
    parsed_current = _finite_positive(current_price)
    if parsed_buy is None:
        return None
    parsed_shares: int | None = None
    if shares is not None:
        try:
            candidate = int(shares)
        except (TypeError, ValueError):
            candidate = 0
        if candidate > 0:
            parsed_shares = candidate
    if parsed_shares is None:
        budget = _finite_positive(position_value_yuan)
        if budget is not None:
            sizing = _position_for_budget(ts_code, parsed_buy, budget)
            if sizing.get("execution_feasible"):
                parsed_shares = int(sizing["estimated_buy_shares"])
    if parsed_shares is None:
        return {
            "status": "need_position_size",
            "message": "已有买入价；还需要持仓股数或持仓金额，才能计入最低佣金并计算实际收益",
        }

    buy_notional = parsed_buy * parsed_shares
    buy_cost = _cost_components(buy_notional, side="buy", scenario=scenario)

    def calculate(exit_price: float) -> dict[str, Any]:
        sell_notional = exit_price * parsed_shares
        sell_cost = _cost_components(sell_notional, side="sell", scenario=scenario)
        invested = buy_notional + float(buy_cost["total_yuan"])
        proceeds = sell_notional - float(sell_cost["total_yuan"])
        pnl = proceeds - invested
        return {
            "exit_price": round(exit_price, 3),
            "sell_notional_yuan": round(sell_notional, 2),
            "sell_cost": sell_cost,
            "net_profit_yuan": round(pnl, 2),
            "net_return": round(pnl / invested, 6) if invested > 0 else None,
            "net_return_pct": round(pnl / invested * 100.0, 3) if invested > 0 else None,
        }

    result: dict[str, Any] = {
        "status": "ok",
        "buy_price": round(parsed_buy, 3),
        "shares": parsed_shares,
        "buy_notional_yuan": round(buy_notional, 2),
        "buy_cost": buy_cost,
        "cost_scenario": scenario.name,
        "cost_scope": "佣金（含最低佣金）、过户费、卖出印花税和双边滑点假设",
    }
    if parsed_current is not None:
        result["current_exit_estimate"] = calculate(parsed_current)
    parsed_projected = _finite_positive(projected_price)
    if parsed_projected is not None:
        result["projected_exit_estimate"] = calculate(parsed_projected)
    return result


def build_requested_forecast(
    full_result: dict[str, Any],
    *,
    analysis_id: str,
    horizon: int,
    mode: str,
    intent: str,
    buy_price: Any = None,
    shares: Any = None,
    position_value_yuan: Any = None,
) -> dict[str, Any]:
    label = f"T+{horizon}"
    stock = full_result.get("stock") or {}
    quantitative = full_result.get("quantitative_analysis") or {}
    diagnostics: dict[str, Any]
    raw_forecast: dict[str, Any]
    timing_valid = True
    unavailable_reason = ""

    if mode == "future_close":
        future = full_result.get("future_3_trading_days") or {}
        diagnostics = (future.get("validation") or {}).get("horizons", {}).get(label, {})
        raw_forecast = (future.get("forecast") or {}).get(label, {})
        if future.get("status") != "ok":
            unavailable_reason = str(future.get("error") or "未来交易日预测当前不可用")
    else:
        diagnostics = (quantitative.get("validation") or {}).get("horizons", {}).get(label, {})
        raw_forecast = (quantitative.get("forecast") or {}).get(label, {})
        scenario_timing = (quantitative.get("analysis_assessment") or {}).get("scenario_timing") or {}
        timing_valid = bool(scenario_timing.get("valid", True))
        if not timing_valid:
            unavailable_reason = str(scenario_timing.get("reason") or "持有期测算入口已经失效")

    validation_passed = bool(diagnostics.get("validation_passed"))
    if unavailable_reason:
        forecast_status = "obsolete_or_unavailable"
    elif not validation_passed:
        forecast_status = "not_validated"
        unavailable_reason = "指定周期没有通过样本外验证，原始点预测不向用户发布"
    elif not raw_forecast:
        forecast_status = "unavailable"
        unavailable_reason = "指定周期没有可用预测"
    else:
        forecast_status = "validated"

    published_forecast: dict[str, Any] | None = None
    projected_price = None
    if forecast_status == "validated":
        if mode == "future_close":
            gross = raw_forecast.get("cumulative_return_from_signal_close")
            cost_rate = (quantitative.get("cost_assumption") or {}).get("estimated_roundtrip_cost_rate")
            net = (
                (1.0 + float(gross)) * (1.0 - float(cost_rate)) - 1.0
                if gross is not None and cost_rate is not None
                else None
            )
            projected_price = raw_forecast.get("predicted_close_reference")
            published_forecast = {
                "target_trade_date": raw_forecast.get("target_trade_date"),
                "cumulative_return_from_signal_close": gross,
                "cumulative_return_from_signal_close_pct": raw_forecast.get(
                    "cumulative_return_from_signal_close_pct"
                ),
                "estimated_return_after_roundtrip_cost": round(net, 6) if net is not None else None,
                "estimated_return_after_roundtrip_cost_pct": round(net * 100.0, 3) if net is not None else None,
                "predicted_close_reference": projected_price,
                "predicted_close_interval_80": raw_forecast.get("predicted_close_interval_80"),
                "historical_similar_sample_positive_rate": raw_forecast.get("empirical_positive_probability"),
                "empirical_return_interval_80": raw_forecast.get("empirical_return_interval_80"),
            }
        else:
            published_forecast = {
                "assumed_entry_date": raw_forecast.get("assumed_entry_date"),
                "scenario_exit_date": raw_forecast.get("scenario_exit_date"),
                "entry_to_exit_gross_return": raw_forecast.get("entry_to_exit_gross_return"),
                "entry_to_exit_gross_return_pct": raw_forecast.get("entry_to_exit_gross_return_pct"),
                "estimated_net_return_after_cost": raw_forecast.get("estimated_net_return_after_cost"),
                "estimated_net_return_after_cost_pct": raw_forecast.get("estimated_net_return_after_cost_pct"),
                "historical_similar_sample_positive_rate": raw_forecast.get("empirical_positive_probability"),
                "empirical_net_return_interval_80": raw_forecast.get("empirical_net_return_interval_80"),
                "position_and_cost": raw_forecast.get("position_and_cost"),
            }

    scenario_name = str((quantitative.get("cost_assumption") or {}).get("scenario") or "normal_cost")
    _, scenario, cost_path, cost_errors = _load_cost_assumption(scenario_name)
    quote = full_result.get("current_quote") or {}
    technical = full_result.get("technical_analysis") or {}
    current_price = quote.get("last_price") if quote.get("status") == "ok" else technical.get("close")
    position = _position_return(
        ts_code=str(stock.get("ts_code") or ""),
        buy_price=buy_price,
        shares=shares,
        position_value_yuan=position_value_yuan,
        current_price=current_price,
        projected_price=projected_price,
        scenario=scenario,
    )

    return {
        "status": "ok" if forecast_status == "validated" else "unavailable",
        "tool_contract_version": 1,
        "analysis_id": analysis_id,
        "analysis_stage_required": "gupiao_fenxi completed",
        "stock": stock,
        "analysis_as_of": full_result.get("as_of"),
        "generated_at": full_result.get("generated_at"),
        "request": {"mode": mode, "intent": intent, "horizon": label},
        "forecast_status": forecast_status,
        "forecast": published_forecast,
        "unavailable_reason": unavailable_reason or None,
        "validation_diagnostics": {
            key: diagnostics.get(key)
            for key in [
                "validation_passed",
                "quality_score",
                "quality_label",
                "direction_accuracy",
                "mean_daily_rank_ic",
                "skill_vs_median_baseline",
                "walk_forward_folds_passed",
                "final_holdout_passed",
                "oos_samples",
            ]
        },
        "position_return_analysis": position,
        "cost_assumption": {
            "scenario": scenario.name,
            "config_path": cost_path,
            "config_errors": cost_errors,
        },
        "interpretation": (
            "买入问题解释为扣除广义交易成本后的上涨空间；卖出问题先分析后续空间，"
            "提供买入价和持仓股数或金额后再计算持仓净收益。"
        ),
    }


class GupiaoYuceTool(BaseTool):
    name = "gupiao_yuce"
    description = (
        "Second-stage request-specific T+1/T+2/T+3 forecast. It requires an analysis_id returned by gupiao_fenxi. "
        "Use future_close for '未来几天/几天后走势' and holding_return for '买入后持有几天'. Interpret buy questions "
        "as after-cost upside. Interpret sell questions as remaining upside; include buy_price and shares or position_value_yuan "
        "when known to calculate net position return. Unvalidated or obsolete point forecasts are deliberately withheld."
    )
    parameters = {
        "type": "object",
        "properties": {
            "analysis_id": {"type": "string", "description": "Identifier returned by the preceding gupiao_fenxi call."},
            "horizon": {"type": "integer", "enum": [1, 2, 3], "default": 2},
            "mode": {"type": "string", "enum": ["future_close", "holding_return"]},
            "intent": {"type": "string", "enum": ["forecast", "buy_upside", "sell_upside"], "default": "forecast"},
            "buy_price": {"type": "number", "exclusiveMinimum": 0},
            "shares": {"type": "integer", "minimum": 1},
            "position_value_yuan": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["analysis_id", "horizon", "mode"],
    }
    repeatable = True
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        analysis_id = str(kwargs.get("analysis_id") or "").strip()
        full_result = get_analysis(analysis_id)
        if full_result is None:
            return json.dumps(
                {
                    "status": "error",
                    "error_code": "analysis_not_found",
                    "error": "分析编号不存在或进程已重启；请先重新调用gupiao_fenxi",
                },
                ensure_ascii=False,
            )
        horizon = int(kwargs.get("horizon", 2))
        if horizon not in {1, 2, 3}:
            return json.dumps({"status": "error", "error": "horizon必须是1、2或3"}, ensure_ascii=False)
        mode = str(kwargs.get("mode") or "future_close")
        if mode not in {"future_close", "holding_return"}:
            return json.dumps({"status": "error", "error": "mode无效"}, ensure_ascii=False)
        result = build_requested_forecast(
            full_result,
            analysis_id=analysis_id,
            horizon=horizon,
            mode=mode,
            intent=str(kwargs.get("intent") or "forecast"),
            buy_price=kwargs.get("buy_price"),
            shares=kwargs.get("shares"),
            position_value_yuan=kwargs.get("position_value_yuan"),
        )
        return json.dumps(result, ensure_ascii=False)


__all__ = ["GupiaoYuceTool", "build_requested_forecast"]
