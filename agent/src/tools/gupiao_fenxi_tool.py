"""Tool wrapper for single-stock A-share research."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_cache import store_analysis


def _direction_analysis(full_result: dict[str, Any]) -> dict[str, Any]:
    """Expose the compact directional conclusion from the completed analysis."""
    quantitative = full_result.get("quantitative_analysis") or {}
    assessment = quantitative.get("analysis_assessment") or {}
    # Direction analysis uses the same signal-close-to-future-close definition
    # as the published three-day forecast.  T+2 is the deliberately simple
    # middle horizon used when the user asks only for a direction.
    horizon = "T+2"
    future = full_result.get("future_3_trading_days") or {}
    forecast = (future.get("forecast") or {}).get(horizon) or {}
    validation = (future.get("validation") or {}).get("horizons", {}).get(horizon, {})
    calibrated = forecast.get("direction_model_positive_probability")
    empirical = forecast.get("empirical_positive_probability")
    probability = calibrated if calibrated is not None else empirical
    probability_method = (
        forecast.get("direction_probability_method")
        if calibrated is not None
        else "historical_similar_sample_positive_rate"
        if empirical is not None
        else None
    )
    try:
        probability = float(probability) if probability is not None else None
    except (TypeError, ValueError):
        probability = None
    if probability is not None and not 0 <= probability <= 1:
        probability = None
    if probability is not None:
        direction = "偏上涨" if probability >= 0.55 else "偏下跌" if probability <= 0.45 else "中性/不确定"
        probability = round(probability, 6)
        evidence_label = (
            "证据偏正面" if probability >= 0.55
            else "证据偏负面" if probability <= 0.45
            else "证据中性"
        )
    else:
        evidence_label = str(assessment.get("evidence_label") or "证据不足")
        direction = (
            "偏上涨" if evidence_label == "证据偏正面"
            else "偏下跌" if evidence_label == "证据偏负面"
            else "中性/不确定"
        )
    confidence = (
        validation.get("quality_label")
        or forecast.get("model_quality")
        or assessment.get("confidence", "insufficient")
    )
    reasons: list[str] = []
    if forecast:
        reasons.append(
            "T+2三交易日模型样本外验证通过"
            if validation.get("validation_passed")
            else "T+2三交易日模型未通过全部样本外门槛，可信度下调"
        )
    replacements = {
        "指定持有期": "三交易日",
        "指定期限": "三交易日",
        "入场后": "信号后",
        "成本后预测收益": "预测收益",
        "成本后收益": "预测收益",
        "成本后期望": "预测收益优势",
        "成本后优势": "预测收益优势",
        "成本后": "预测优势",
        "整手或流动性容量约束不允许按当前资金建仓": "可交易性或流动性约束限制信号",
        "按目标资金测算，无法满足最低买入数量或成交容量约束": "可交易性或流动性约束限制信号",
        "持有期量化模型": "三交易日量化模型",
    }
    for raw_reason in list(assessment.get("reasons") or []):
        reason = str(raw_reason)
        for old, new in replacements.items():
            reason = reason.replace(old, new)
        if reason and reason not in reasons:
            reasons.append(reason)
    for risk in list(full_result.get("risks") or [])[:3]:
        risk_text = str(risk)
        if risk_text not in reasons:
            reasons.append(f"风险提示：{risk_text}")
    return {
        "horizon": horizon,
        "horizon_definition": "默认以T+2作为未指定期限时的方向分析参考；完整T+1/T+2/T+3数值由gupiao_yuce返回",
        "direction": direction,
        "positive_probability": probability,
        "negative_probability": round(1.0 - probability, 6) if probability is not None else None,
        "probability_method": probability_method,
        "probability_definition": "上涨可能性：模型或历史相似样本中未来收益为正的比例",
        "evidence_label": evidence_label,
        "confidence": confidence,
        "summary": (
            f"{direction}：T+2方向参考概率为 {probability:.3f}"
            if probability is not None
            else assessment.get("summary", "当前没有形成明确方向结论")
        ),
        "reasons": reasons[:8],
        "validation_passed": bool(validation.get("validation_passed")),
        "note": "这是对当前证据的方向性判断，不是收益保证；具体三日数值由 gupiao_yuce 返回。",
    }


class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "Analyze one mainland China A-share using the latest complete daily data. Return the current technical, "
        "fundamental, volatility and risk evidence plus a directional conclusion (up/down/uncertain) and its "
        "model or historical positive-return likelihood. The returned analysis_id is required by gupiao_yuce "
        "for the separate three-trading-day forecast."
    )
    parameters = {
        "type": "object",
        "properties": {
            "gupiao": {"type": "string", "description": "A-share code or Chinese name, for example 600519.SH or 贵州茅台"},
            "source": {
                "type": "string",
                "enum": ["auto", "tushare", "akshare"],
                "description": (
                    "Stock-name resolution and daily-bar source. auto means Tushare first, then AKShare fallback. "
                    "Fundamentals still use their own Tushare-first fallback policy."
                ),
            },
            "history_calendar_days": {
                "type": "integer",
                "minimum": 540,
                "maximum": 1800,
                "default": 1440,
                "description": "Calendar days used for the daily-K peer-panel model; default 1440.",
            },
            "config_path": {"type": "string", "description": "Optional path to lianghua_peizhi.json."},
        },
        "required": ["gupiao"],
    }
    repeatable = True
    # Keep this serial in the agent loop: it writes the in-process snapshot
    # consumed by gupiao_yuce, so the two stages cannot race in one turn.
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.gupiao_yanjiu import fenxi_gupiao

        full_result = fenxi_gupiao(**kwargs)
        if full_result.get("status") != "ok":
            return json.dumps(full_result, ensure_ascii=False)

        analysis_id = store_analysis(full_result)
        hidden = {"quantitative_analysis", "future_3_trading_days", "analysis_assessment"}
        public_result = {key: value for key, value in full_result.items() if key not in hidden}
        quantitative = full_result.get("quantitative_analysis") or {}
        public_result.update(
            {
                "tool_contract_version": 4,
                "analysis_id": analysis_id,
                "peer_analysis": {
                    "status": quantitative.get("status"),
                    "peer_universe": quantitative.get("peer_universe"),
                    "daily_factor_data": quantitative.get("daily_factor_data"),
                    "methodology": quantitative.get("methodology"),
                    "limitations": quantitative.get("limitations"),
                    "error": quantitative.get("error"),
                },
                "analysis_stage": {
                    "status": "completed",
                    "scope": "行情时点、基本面、估值、技术面、波动、可交易约束、同行和风险已完成",
                    "next_tool_for_numbers": "gupiao_yuce",
                    "instruction": "如需要未来三交易日的具体预测，继续调用 gupiao_yuce",
                },
                "direction_analysis": _direction_analysis(full_result),
            }
        )
        return json.dumps(public_result, ensure_ascii=False)
