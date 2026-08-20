"""Second-stage forecast for the next three trading days."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_state import analysis_session_store


def _selected_stock(full_result: dict[str, Any]) -> dict[str, Any]:
    raw = full_result.get("stock") or full_result.get("selected_stock") or full_result.get("primary") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: raw.get(key)
        for key in ("ts_code", "name", "industry")
        if raw.get(key) is not None
    }


def _resolve_prediction_context(
    stored_context: dict[str, Any] | None,
    requested_stock: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not stored_context:
        return None, "当前分析没有达到门槛、可继续预测的候选"
    candidates = stored_context.get("candidates")
    if not isinstance(candidates, dict):
        return stored_context, None

    def with_shared(candidate: dict[str, Any]) -> dict[str, Any]:
        """把一次分析共享的数据策略与配置合并到选中的候选。"""
        shared = {
            key: stored_context.get(key)
            for key in ("source", "signal_date", "config")
            if stored_context.get(key) is not None
        }
        return {**shared, **candidate}

    query = str(requested_stock or "").strip().upper()
    if not query:
        primary_code = str(stored_context.get("primary_code") or "")
        context = candidates.get(primary_code)
        return (
            (with_shared(context), None)
            if isinstance(context, dict)
            else (None, "当前分析没有可用首选股票")
        )
    matched: list[dict[str, Any]] = []
    digits = "".join(character for character in query if character.isdigit())
    for code, context in candidates.items():
        if not isinstance(context, dict):
            continue
        code_text = str(code).upper()
        name = str(context.get("name") or "").strip().upper()
        if query in {code_text, code_text.split(".")[0], name} or (digits and digits == code_text.split(".")[0]):
            matched.append(context)
    if len(matched) == 1:
        return with_shared(matched[0]), None
    if not matched:
        return None, "指定股票不在本次达到门槛的首选或备选中"
    return None, "指定股票名称存在歧义"


def _bounded_probability(value: Any) -> float | None:
    """Normalize a model probability without manufacturing one."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return round(min(max(parsed, 0.0), 1.0), 6)


def _xiazai_moxing_lishi(prediction_context: dict[str, Any]) -> dict[str, Any]:
    """收到明确预测请求后，按模型窗口重新下载目标日线。"""
    context = dict(prediction_context)
    config = context.get("config") or {}
    history_days = int((config.get("dangu") or {}).get("history_calendar_days", 1440))
    signal_date = datetime.strptime(str(context.get("signal_date")), "%Y-%m-%d")
    start_date = signal_date - timedelta(days=history_days)
    from src.ashare.gupiao_yanjiu import huoqu_rili_xingqing

    market = huoqu_rili_xingqing(
        str(context.get("code")),
        start_date=start_date.strftime("%Y%m%d"),
        end_date=signal_date.strftime("%Y%m%d"),
        source=str(context.get("source") or "auto"),
    )
    if market.data is None or market.data.empty:
        detail = "；".join(str(value) for value in market.errors)
        raise RuntimeError(f"无法补齐预测所需的 {history_days} 天目标日线{f'：{detail}' if detail else ''}")
    trade_dates = pd.to_datetime(market.data.get("trade_date"), errors="coerce").dropna()
    if trade_dates.empty or pd.Timestamp(trade_dates.max()).normalize() != pd.Timestamp(signal_date).normalize():
        latest = pd.Timestamp(trade_dates.max()).strftime("%Y-%m-%d") if not trade_dates.empty else "未知"
        raise RuntimeError(f"目标股票扩展日线停留在 {latest}，与分析日 {signal_date:%Y-%m-%d} 不一致")
    context["target_history"] = market.data
    context["target_source"] = market.source
    context["target_adjustment"] = market.adjustment
    return context


def build_three_day_forecast(
    full_result: dict[str, Any],
    *,
    analysis_id: str,
) -> dict[str, Any]:
    """Publish the three forecasts calculated from one completed analysis.

    The prediction stage trains/calculates all three horizons together.  This
    wrapper deliberately accepts no horizon or portfolio arguments:
    one call returns the complete personal-use forecast for T+1/T+2/T+3.
    """
    future = full_result.get("future_3_trading_days") or {}
    raw_forecasts = future.get("forecast") or {}
    required_labels = {"T+1", "T+2", "T+3"}
    if future.get("status") != "ok" or not raw_forecasts or not required_labels.issubset(raw_forecasts):
        return {
            "status": "unavailable",
            "tool_contract_version": 6,
            "analysis_id": analysis_id,
            "stock": _selected_stock(full_result),
            "analysis_as_of": full_result.get("as_of"),
            "signal_close": future.get("signal_close"),
            "forecast": {},
            "error": future.get("error") or "未来三个交易日预测当前不可用",
        }

    forecasts: dict[str, dict[str, Any]] = {}
    for horizon in (1, 2, 3):
        label = f"T+{horizon}"
        raw = raw_forecasts.get(label) or {}
        probability = _bounded_probability(
            raw.get("direction_model_positive_probability")
            if raw.get("direction_model_positive_probability") is not None
            else raw.get("empirical_positive_probability")
        )
        forecasts[label] = {
            "target_trade_date": raw.get("target_trade_date"),
            "direction": raw.get("direction", "flat_or_unavailable"),
            "positive_probability": probability,
            "negative_probability": round(1.0 - probability, 6) if probability is not None else None,
            "predicted_return": raw.get("cumulative_return_from_signal_close"),
            "predicted_return_pct": raw.get("cumulative_return_from_signal_close_pct"),
            "predicted_close": raw.get("predicted_close_reference"),
            "interval_80": raw.get("predicted_close_interval_80"),
            "validation_passed": bool(raw.get("validation_passed")),
            "confidence": raw.get("model_quality", "low"),
        }

    return {
        "status": "ok",
        "tool_contract_version": 6,
        "analysis_id": analysis_id,
        "stock": _selected_stock(full_result),
        "analysis_as_of": full_result.get("as_of"),
        "signal_date": future.get("signal_date"),
        "signal_close": future.get("signal_close"),
        "forecast": forecasts,
        "data_policy": "预测确认后重新下载远端数据；不读取或写入本地市场数据缓存",
        "definition": "以最近完整收盘日为T，预测未来第1、2、3个交易日收盘相对T收盘的方向和累计收益",
        "note": "这是模型参考值，不是目标价或交易指令；confidence表示模型可信度，不是收益保证。",
    }


class GupiaoYuceTool(BaseTool):
    name = "gupiao_yuce"
    description = (
        "Expensive second-stage forecast that may run only after the user explicitly confirms prediction for a qualified "
        "candidate from a completed gupiao_fenxi result. It freshly downloads all required market data without a local "
        "market-data cache, trains the models, and returns T+1, T+2, and T+3 together. Never ask the user to provide or "
        "repeat the internal analysis identifier."
    )
    parameters = {
        "type": "object",
        "properties": {
            "analysis_id": {
                "type": "string",
                "description": "Identifier returned by the preceding gupiao_fenxi call.",
            },
            "gupiao": {
                "type": "string",
                "description": "Optional selected candidate code or name when the user explicitly refers to an alternative; omit for the primary stock.",
            },
        },
        "required": ["analysis_id"],
    }
    repeatable = True
    # This stage intentionally performs the deferred model fit against shared
    # in-process context, so it must be serialized with the first-stage tool.
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        analysis_id = str(kwargs.get("analysis_id") or "").strip()
        session = analysis_session_store.get(analysis_id)
        if session is None:
            return json.dumps(
                {
                    "status": "reanalysis_required",
                    "outcome": "reanalysis_required",
                    "error_code": "analysis_not_found",
                    "stage": "analysis_session_handoff",
                    "retryable": True,
                    "error": "分析会话只存在于原进程，当前必须先重新获取远端数据并完成量化分析",
                    "next_action": "重新分析原范围，展示新结果并再次询问是否预测",
                    "market_data_persistence": "none",
                },
                ensure_ascii=False,
            )
        full_result = session.result
        stored_context = session.prediction_context
        if str(full_result.get("analysis_type") or "") == "single_stock_analysis":
            return json.dumps(
                {
                    "status": "unavailable",
                    "outcome": "prediction_not_available",
                    "tool_contract_version": 6,
                    "analysis_id": analysis_id,
                    "stock": _selected_stock(full_result),
                    "forecast": {},
                    "error": "单股买入建议不产生统一选股的预测资格；预测仍只面向范围选股中的合格候选",
                    "next_action": "如需未来三交易日预测，请先完成选股分析并确认合格候选",
                },
                ensure_ascii=False,
            )
        prediction_context, context_error = _resolve_prediction_context(
            stored_context,
            str(kwargs.get("gupiao") or "").strip() or None,
        )
        if context_error:
            return json.dumps(
                {
                    "status": "unavailable",
                    "tool_contract_version": 6,
                    "analysis_id": analysis_id,
                    "forecast": {},
                    "error": context_error,
                },
                ensure_ascii=False,
            )
        if prediction_context is not None:
            # 第一阶段只保留候选身份和量化证据；用户确认后才下载市场数据并训练。
            try:
                from src.ashare.dangu_yuce import yanjiu_dangu_yuce

                hydrated_context = _xiazai_moxing_lishi(prediction_context)
                quantitative = yanjiu_dangu_yuce(**hydrated_context)
                full_result = dict(full_result)
                full_result["stock"] = {
                    "ts_code": hydrated_context.get("code"),
                    "name": hydrated_context.get("name"),
                    "industry": hydrated_context.get("industry"),
                }
                full_result["quantitative_analysis"] = quantitative
                full_result["future_3_trading_days"] = quantitative.get("future_3_trading_days", {})
                full_result["analysis_assessment"] = quantitative.get("analysis_assessment", {})
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "unavailable",
                        "tool_contract_version": 6,
                        "analysis_id": analysis_id,
                        "stock": _selected_stock(full_result),
                        "forecast": {},
                        "error": f"按需训练三交易日预测模型失败：{exc}",
                    },
                    ensure_ascii=False,
                )
        return json.dumps(
            build_three_day_forecast(full_result, analysis_id=analysis_id),
            ensure_ascii=False,
        )


__all__ = ["GupiaoYuceTool", "build_three_day_forecast"]
