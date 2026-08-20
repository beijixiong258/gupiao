"""把统一选股量化结果适配为单股买入结论。

本模块不计算任何新指标或分数。因子、基本面、形态、尾盘、风险扣分、综合分
和推荐门槛全部复用统一选股现有方法；这里只把同一份候选摘要翻译为明确的
“建议买入/暂不建议买入”结构，避免模型自行决定。
"""

from __future__ import annotations

from typing import Any

from src.ashare.xuangu_guize import goujian_houxuan_zhaiyao, zhuan_you_xian_shuzhi


def _zu_fen(groups: dict[str, Any], key: str) -> float | None:
    group = groups.get(key)
    return zhuan_you_xian_shuzhi(group.get("score_0_100")) if isinstance(group, dict) else None


def _goujian_shourenhua_shoushu(
    item: dict[str, Any],
    *,
    recommended: bool,
    score: float | None,
    minimum_score: float,
    minimum_confidence: float,
) -> tuple[str, list[str]]:
    """只解释既有计算结果，不新增指标、分数或买入门槛。"""

    technical = item.get("technical") if isinstance(item.get("technical"), dict) else {}
    returns = technical.get("returns") if isinstance(technical.get("returns"), dict) else {}
    averages = (
        technical.get("moving_averages")
        if isinstance(technical.get("moving_averages"), dict)
        else {}
    )
    factor = item.get("factor") if isinstance(item.get("factor"), dict) else {}
    groups = factor.get("groups") if isinstance(factor.get("groups"), dict) else {}
    structure = (
        technical.get("macd_structure")
        if isinstance(technical.get("macd_structure"), dict)
        else {}
    )
    classification = (
        structure.get("structure_classification")
        if isinstance(structure.get("structure_classification"), dict)
        else {}
    )
    classification_code = str(classification.get("code") or "")
    classification_label = str(classification.get("label") or "").strip()
    close = zhuan_you_xian_shuzhi(technical.get("close"))
    ret_5 = zhuan_you_xian_shuzhi(returns.get("5d"))
    ret_20 = zhuan_you_xian_shuzhi(returns.get("20d"))
    ma_5 = zhuan_you_xian_shuzhi(averages.get("ma5"))
    ma_10 = zhuan_you_xian_shuzhi(averages.get("ma10"))
    ma_20 = zhuan_you_xian_shuzhi(averages.get("ma20"))

    strengths: list[str] = []
    if ret_20 is not None and ret_20 > 0:
        strengths.append(f"近20日仍上涨约 {ret_20 * 100:.1f}%")
    if close is not None and ma_20 is not None and close >= ma_20:
        strengths.append("价格仍在MA20上方")
    for key in ("breakout_pullback_quality", "trend_structure", "price_volume_confirmation"):
        group = groups.get(key)
        group_score = _zu_fen(groups, key)
        if isinstance(group, dict) and group_score is not None and group_score >= 65:
            strengths.append(f"{str(group.get('label') or '相关因子')}达到 {group_score:.1f} 分")
    if classification_code == "trend_continuation" and classification_label:
        strengths.append(classification_label)
    strengths = list(dict.fromkeys(strengths))[:3]
    if not strengths:
        strengths.append("仍有部分正面量化证据")

    concerns: list[str] = []
    if ret_5 is not None and ret_5 < 0:
        concerns.append(f"近5日回落约 {abs(ret_5) * 100:.1f}%")
    below_short = [
        label
        for label, value in (("MA5", ma_5), ("MA10", ma_10))
        if close is not None and value is not None and close < value
    ]
    if below_short:
        concerns.append("价格尚未站稳" + "、".join(below_short))
    if classification_code in {"top_risk", "momentum_weakening", "weak_rebound"} and classification_label:
        concerns.append(classification_label)
    weak_groups: list[str] = []
    for key in (
        "momentum_reversal",
        "price_volume_confirmation",
        "relative_strength",
        "risk_liquidity",
        "market_context",
    ):
        group = groups.get(key)
        group_score = _zu_fen(groups, key)
        if isinstance(group, dict) and group_score is not None and group_score < 50:
            weak_groups.append(str(group.get("label") or key))
    if weak_groups:
        concerns.append("、".join(weak_groups[:3]) + "均不足50分")
    volatility = zhuan_you_xian_shuzhi(technical.get("annualized_volatility_20"))
    if volatility is not None and volatility > 0.55:
        concerns.append(f"20日年化波动率约 {volatility * 100:.1f}%")
    concerns = list(dict.fromkeys(concerns))[:5]

    tradability = item.get("tradability") if isinstance(item.get("tradability"), dict) else {}
    ranking = item.get("ranking") if isinstance(item.get("ranking"), dict) else {}
    ranking_confidence = float(ranking.get("confidence") or 0.0)
    reconsider: list[str] = []
    if ret_5 is not None and ret_5 < 0:
        reconsider.append("5日动量重新转正")
    if below_short:
        reconsider.append("重新站稳" + "、".join(below_short))
    price_volume_score = _zu_fen(groups, "price_volume_confirmation")
    if price_volume_score is not None and price_volume_score < 50:
        reconsider.append("价量确认明显改善")
    if classification_code == "top_risk":
        reconsider.append("MACD顶部风险消退")
    if tradability.get("basic_execution_feasible", True) is False:
        reconsider.append("基础可交易性恢复")
    if score is None or score < minimum_score:
        reconsider.append(f"综合分达到 {minimum_score:.1f} 分")
    if ranking_confidence < minimum_confidence:
        reconsider.append(f"证据完整度达到 {minimum_confidence * 100:.0f}%")
    reconsider = list(dict.fromkeys(reconsider))

    strength_text = "、".join(strengths)
    concern_text = "、".join(concerns) if concerns else "仍需留意已有风险提示"
    if recommended:
        text = (
            f"说人话，这只股票当前{strength_text}，综合分已经达到统一买入门槛，"
            f"所以按现有量化规则可以列入买入研究范围；不过{concern_text}，不能把建议买入理解成必涨。"
        )
    else:
        if score is None:
            decision_reason = "综合分当前不可用"
        elif score < minimum_score:
            decision_reason = (
                f"综合分只有 {score:.2f} 分，没有达到 {minimum_score:.1f} 分门槛"
            )
        else:
            other_failures: list[str] = []
            if ranking_confidence < minimum_confidence:
                other_failures.append("证据完整度不足")
            if tradability.get("basic_execution_feasible", True) is False:
                other_failures.append("基础可交易性未通过")
            decision_reason = (
                f"综合分虽有 {score:.2f} 分，但"
                + "、".join(other_failures or ["仍未取得统一推荐资格"])
            )
        text = (
            f"说人话，这只股票并不是没有亮点，{strength_text}；但当前{concern_text}，"
            f"{decision_reason}。因此更适合继续观察，暂不适合追着买。"
        )
        if reconsider:
            text += "等" + "、".join(reconsider[:5]) + "后再重新评估。"
    return text, reconsider


def goujian_dangu_mairu_juece(
    item: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """复用统一候选摘要和相同门槛，生成单股买入结论。"""

    settings = config.get("fenxi") if isinstance(config.get("fenxi"), dict) else {}
    minimum_score = float(settings.get("minimum_recommendation_score", 60.0))
    minimum_confidence = float(settings.get("minimum_confidence", 0.6))
    summary = goujian_houxuan_zhaiyao(
        item,
        rank=1,
        minimum_score=minimum_score,
        minimum_confidence=minimum_confidence,
    )
    recommended = bool(summary.get("meets_recommendation_threshold"))
    unmet = [
        str(value)
        for value in summary.get("all_unmet_conditions")
        or summary.get("unmet_conditions")
        or []
    ]
    tradability = summary.get("tradability") if isinstance(summary.get("tradability"), dict) else {}
    blocking = [str(value) for value in tradability.get("hard_blocks") or []]
    if tradability.get("basic_execution_feasible", True) is False and not blocking:
        blocking.extend(
            [str(value) for value in tradability.get("cautions") or []]
            or ["基础可交易性无法确认或未通过"]
        )
    score = summary.get("ranking_score_0_100")
    if score is None:
        blocking.append("统一综合分不可用")
    elif float(score) < minimum_score:
        blocking.append(
            f"统一综合分 {float(score):.2f}/100，低于推荐门槛 {minimum_score:.1f}/100"
        )
    confidence = float(summary.get("confidence") or 0.0)
    if confidence < minimum_confidence:
        blocking.append(
            f"证据完整度 {confidence * 100:.0f}%，低于推荐门槛 {minimum_confidence * 100:.0f}%"
        )
    if not recommended and not blocking:
        blocking.append("未通过统一选股的综合分、证据完整度或可交易性推荐资格")
    blocking = list(dict.fromkeys(blocking))
    numeric_score = float(score) if score is not None else None
    plain_summary, reassessment_conditions = _goujian_shourenhua_shoushu(
        item,
        recommended=recommended,
        score=numeric_score,
        minimum_score=minimum_score,
        minimum_confidence=minimum_confidence,
    )
    return {
        "status": "complete",
        "decision": "recommend_buy" if recommended else "not_recommended",
        "label": "建议买入" if recommended else "暂不建议买入",
        "meets_recommendation_threshold": recommended,
        "score_0_100": score,
        "score_definition": summary.get("ranking_score_definition"),
        "confidence": confidence,
        "thresholds": {
            "minimum_recommendation_score": minimum_score,
            "minimum_confidence": minimum_confidence,
        },
        "positive_evidence": summary.get("positive_evidence") or [],
        "blocking_conditions": blocking,
        "unmet_conditions": unmet,
        "plain_language_summary": plain_summary,
        "reassessment_conditions": reassessment_conditions,
        "risks": summary.get("risks") or [],
        "daily_factor_analysis": summary.get("daily_factor_analysis"),
        "fundamental_analysis": summary.get("fundamental_analysis"),
        "limit_up_pullback_pattern": summary.get("limit_up_pullback_pattern"),
        "late_session_analysis": summary.get("late_session_analysis"),
        "tradability": summary.get("tradability"),
        "ranking_details": summary.get("ranking_details"),
        "methodology": (
            "直接复用统一选股的八组日 K 因子、基本面、涨停回马枪、尾盘证据、风险扣分、"
            "综合评分与推荐门槛；单股适配层不新增或修改分数"
        ),
        "prediction_eligible": False,
    }


__all__ = ["goujian_dangu_mairu_juece"]
