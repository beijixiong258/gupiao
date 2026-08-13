"""Second-stage forecast for the next three trading days."""

from __future__ import annotations

import json
import math
from typing import Any

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_cache import get_analysis, get_prediction_context


def _bounded_probability(value: Any) -> float | None:
    """Normalize a model probability without manufacturing one."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return round(min(max(parsed, 0.0), 1.0), 6)


def build_three_day_forecast(
    full_result: dict[str, Any],
    *,
    analysis_id: str,
) -> dict[str, Any]:
    """Publish the three forecasts calculated from one completed diagnosis.

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
            "tool_contract_version": 4,
            "analysis_id": analysis_id,
            "stock": full_result.get("stock") or {},
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
        "tool_contract_version": 4,
        "analysis_id": analysis_id,
        "stock": full_result.get("stock") or {},
        "analysis_as_of": full_result.get("as_of"),
        "signal_date": future.get("signal_date"),
        "signal_close": future.get("signal_close"),
        "forecast": forecasts,
        "definition": "以最近完整收盘日为T，预测未来第1、2、3个交易日收盘相对T收盘的方向和累计收益",
        "note": "这是模型参考值，不是目标价或交易指令；confidence表示模型可信度，不是收益保证。",
    }


class GupiaoYuceTool(BaseTool):
    name = "gupiao_yuce"
    description = (
        "Second-stage forecast. It requires the analysis_id returned by gupiao_fenxi and returns one result containing "
        "the direction, positive probability, reference close, and confidence for T+1, T+2, and T+3."
    )
    parameters = {
        "type": "object",
        "properties": {
            "analysis_id": {
                "type": "string",
                "description": "Identifier returned by the preceding gupiao_fenxi call.",
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
        prediction_context = get_prediction_context(analysis_id)
        if prediction_context is not None:
            # The first stage deliberately stores only deterministic factor
            # evidence.  Fit the expensive return models here, after the user
            # has explicitly requested a forecast.
            try:
                from src.ashare.dangu_yuce import yanjiu_dangu_yuce

                quantitative = yanjiu_dangu_yuce(**prediction_context)
                full_result = dict(full_result)
                full_result["quantitative_analysis"] = quantitative
                full_result["future_3_trading_days"] = quantitative.get("future_3_trading_days", {})
                full_result["analysis_assessment"] = quantitative.get("analysis_assessment", {})
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "unavailable",
                        "tool_contract_version": 4,
                        "analysis_id": analysis_id,
                        "stock": full_result.get("stock") or {},
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
