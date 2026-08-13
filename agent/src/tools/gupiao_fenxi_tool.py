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
    factor_analysis = quantitative.get("factor_analysis") or {}
    if factor_analysis.get("status") == "ok":
        direction = str(factor_analysis.get("direction") or "中性/不确定")
        evidence_label = str(factor_analysis.get("evidence_label") or "证据不足")
        reasons = [str(value) for value in (factor_analysis.get("evidence") or [])]
        for risk in list(full_result.get("risks") or [])[:3]:
            risk_text = f"风险提示：{risk}"
            if risk_text not in reasons:
                reasons.append(risk_text)
        return {
            "horizon": "当前因子状态",
            "horizon_definition": "只描述最近完整收盘日的量化因子和基本面证据；不代表未来收益预测期限",
            "direction": direction,
            "positive_probability": None,
            "negative_probability": None,
            "probability_method": None,
            "probability_definition": "分析阶段不训练方向概率模型，也不把因子分数换算成概率",
            "evidence_label": evidence_label,
            "confidence": assessment.get("confidence", "descriptive_factor_evidence"),
            "summary": assessment.get("summary") or f"{evidence_label}：基于当前量化因子证据",
            "reasons": reasons[:8],
            "validation_passed": None,
            "note": "这是当前因子证据的方向性解读，不是收益保证；如需 T+1/T+2/T+3 数值，请调用 gupiao_yuce。",
        }
    evidence_label = str(assessment.get("evidence_label") or "证据不足")
    direction = (
        "偏上涨" if evidence_label == "证据偏正面"
        else "偏下跌" if evidence_label == "证据偏负面"
        else "中性/不确定"
    )
    reasons = [str(value) for value in (factor_analysis.get("evidence") or assessment.get("reasons") or [])]
    for risk in list(full_result.get("risks") or [])[:3]:
        risk_text = str(risk)
        if risk_text not in reasons:
            reasons.append(f"风险提示：{risk_text}")
    return {
        "horizon": "当前因子状态",
        "horizon_definition": "只描述最近完整收盘日的证据；完整T+1/T+2/T+3数值由gupiao_yuce返回",
        "direction": direction,
        "positive_probability": None,
        "negative_probability": None,
        "probability_method": None,
        "probability_definition": "分析阶段不训练方向概率模型，也不把因子分数换算成概率",
        "evidence_label": evidence_label,
        "confidence": assessment.get("confidence", "descriptive_factor_evidence"),
        "summary": assessment.get("summary", "当前没有形成明确方向结论"),
        "reasons": reasons[:8],
        "validation_passed": None,
        "note": "这是当前证据的方向性判断，不是收益保证；具体三日数值由 gupiao_yuce 返回。",
    }


class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "Analyze one mainland China A-share using the latest complete daily data. Calculate deterministic technical, "
        "fundamental, price/volume, relative-strength and risk factors and return a plain directional evidence summary. "
        "This first stage does not fit a return-prediction model or probability. The returned analysis_id is required "
        "by gupiao_yuce when the user explicitly requests the separate three-trading-day forecast."
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
                "description": "Calendar days used for daily-K factor evidence and the deferred prediction context; default 1440.",
            },
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

        prediction_context = full_result.get("_prediction_context")
        stored_result = {
            key: value for key, value in full_result.items() if not str(key).startswith("_")
        }
        analysis_id = store_analysis(stored_result, prediction_context=prediction_context)
        hidden = {"quantitative_analysis", "future_3_trading_days", "analysis_assessment"}
        public_result = {key: value for key, value in stored_result.items() if key not in hidden}
        quantitative = stored_result.get("quantitative_analysis") or {}
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
                    "scope": "行情时点、基本面、估值、技术面、价量因子、相对强弱、可交易约束、同行和风险已完成；分析阶段未训练预测模型",
                    "next_tool_for_numbers": "gupiao_yuce",
                    "instruction": "如需要未来三交易日的具体预测，继续调用 gupiao_yuce；该调用才会训练预测模型",
                },
                "direction_analysis": _direction_analysis(full_result),
                "factor_analysis": quantitative.get("factor_analysis") or {},
            }
        )
        return json.dumps(public_result, ensure_ascii=False)
