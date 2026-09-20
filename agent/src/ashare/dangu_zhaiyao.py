"""把单股已计算的事实整理为证据总结，不产生买入标签或评分。"""

from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _texts(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value)) if isinstance(values, (list, tuple)) else []


def goujian_dangu_zhaiyao(
    *, technical: dict[str, Any], fundamentals: dict[str, Any],
    risks: list[str], evidence_gaps: list[dict[str, Any]], as_of: str,
    factor_analysis: dict[str, Any] | None = None, pattern: dict[str, Any] | None = None,
    late: dict[str, Any] | None = None, supplemental: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structure = technical.get("macd_structure") or {}
    usable_structure = structure.get("status") == "ok"
    supporting = _texts(structure.get("supporting_evidence")) if usable_structure else []
    counter = _texts(structure.get("counter_evidence")) if usable_structure else []
    conflicts: list[str] = []
    context: dict[str, list[str]] = {}
    reassessment = _texts([gap.get("reassessment_condition") for gap in evidence_gaps])
    raw = technical.get("raw_indicators") or {}
    returns = {**(technical.get("returns") or {}), **{f"{period}d": raw[f"ret_{period}"] for period in (5, 20) if f"ret_{period}" in raw}}
    price_facts: list[str] = []
    for period in ("5d", "20d"):
        value = _number(returns.get(period))
        if value is not None:
            fact = f"近{period[:-1]}个交易日收益为 {value:.2%}"
            price_facts.append(fact)
            if value > 0:
                supporting.append(fact)
            elif value < 0:
                counter.append(fact)
    close = _number(raw.get("close", technical.get("close")))
    averages = {**(technical.get("moving_averages") or {}), **{f"ma{period}": raw[f"ma_{period}"] for period in (20, 60) if f"ma_{period}" in raw}}
    for period in (20, 60):
        value = _number(averages.get(f"ma{period}"))
        if close is not None and value is not None:
            relation = "高于" if close > value else "低于" if close < value else "等于"
            fact = f"收盘价 {close:.3f} {relation}{period}日均线 {value:.3f}"
            price_facts.append(fact)
            if close != value:
                (supporting if close > value else counter).append(fact)
    ma20, ma60 = _number(averages.get("ma20")), _number(averages.get("ma60"))
    if close is not None and ma20 is not None and ma60 is not None and close > ma20 and close < ma60:
        conflicts.append("价格站上20日均线但仍低于60日均线；短期改善与中期位置偏弱并存，不能称为趋势全面转强")
    context["价格与趋势"] = price_facts
    context["动能结构"] = _texts([
        (structure.get("structure_classification") or {}).get("label") if usable_structure else structure.get("reason"),
        *(structure.get("warnings") or []),
    ])
    groups = (factor_analysis or {}).get("groups") or {}

    def factor(group: str, field: str) -> float | None:
        return _number(((groups.get(group) or {}).get("values") or {}).get(field))

    volume_ratio = _number(raw.get("volume_ratio_5_20", technical.get("volume_ratio_5_to_20")))
    if volume_ratio is None:
        volume_ratio = factor("price_volume_confirmation", "volume_ratio_5_20")
    context["量价配合"] = [f"5日与20日均量比为 {volume_ratio:.3f}；仅描述量能变化"] if volume_ratio is not None else ["均量比较证据缺失"]
    ret5 = _number(returns.get("5d"))
    if ret5 is not None and ret5 > 0 and volume_ratio is not None and volume_ratio < 1:
        conflicts.append("近5日上涨，但5日均量低于20日均量；没有同时获得均量放大的确认")
    relative_facts: list[str] = []
    for field, label in (("excess_vs_csi300_ret_20", "沪深300"), ("excess_vs_industry_ret_20", "本次真实同行样本")):
        value = factor("relative_strength", field)
        if value is not None:
            fact = f"近20日相对{label}收益差为 {value * 100:.2f} 个百分点"
            relative_facts.append(fact)
            if value != 0:
                (supporting if value > 0 else counter).append(fact)
    context["相对表现"] = relative_facts or ["指数或真实同行比较证据不足"]
    market_facts: list[str] = []
    for field, text in (
        ("market_regime_strong", "沪深300方向与本次比较池广度同时偏强"),
        ("market_regime_weak", "沪深300方向与本次比较池广度同时偏弱"),
        ("market_regime_sideways", "沪深300方向与本次比较池广度未形成同向强弱，不能据此认定本股横盘"),
    ):
        if factor("market_context", field) == 1:
            market_facts.append(text)
    context["市场背景"] = market_facts or ["市场方向与比较池广度证据不足"]
    if ret5 is not None and ret5 > 0 and factor("market_context", "market_regime_weak") == 1:
        conflicts.append("本股近5日上涨，但指数与比较池背景偏弱；个股改善不能替代市场环境核验")
    context["下行风险与流动性"] = _texts([
        block.get("summary") for block in (supplemental or {}).get("blocks", [])
        if block.get("key") in {"downside_risk", "atr_distance", "liquidity_stability"}
        and block.get("status") in {"ok", "partial"}
    ]) or ["下行风险、均线偏离或流动性稳定性证据不足"]
    cross = structure.get("latest_cross") or {}
    if usable_structure:
        reassessment.extend(_texts(structure.get("invalidation_conditions")))
        if cross.get("status") == "active" and cross.get("invalidation_condition"):
            reassessment.append(str(cross["invalidation_condition"]))
    pattern = pattern or {}
    state = str(pattern.get("state") or "")
    pattern_facts = _texts([pattern.get("state_label") or pattern.get("reason")])
    if pattern.get("eligible") and state in {"waiting_breakout", "extreme_shrink", "adjusting", "intraday_confirmed", "close_confirmed"}:
        risk_price = _number(pattern.get("risk_reference_price"))
        if risk_price is not None:
            reassessment.append(f"价格触及形态已计算的风险参考价 {risk_price:.3f} 时重新核验形态状态")
        if state == "close_confirmed":
            supporting.extend(pattern_facts)
        elif state == "intraday_confirmed":
            reassessment.append("盘中形态需等待当日完整收盘数据确认")
    context["形态适用状态"] = pattern_facts or ["没有可归纳的形态状态"]
    if (late or {}).get("stage") not in {None, "not_applicable"}:
        context["尾盘核验状态"] = _texts([(late or {}).get("stage_label"), (late or {}).get("reason")])
        if (late or {}).get("confirmation_level") in {"intraday_provisional", "pending", "not_confirmed"}:
            reassessment.append("取得当日完整收盘与所需分钟证据后复核尾盘状态")
    financials = fundamentals.get("financials") or {}
    for field, label in (("roe_pct", "净资产收益率"), ("net_profit_yoy_pct", "净利润同比")):
        value = _number(financials.get(field))
        if value is not None and value != 0:
            (supporting if value > 0 else counter).append(f"本次财报{label} {value:.6g}%")
    counter.extend([*conflicts, *risks])
    supporting, counter = _texts(supporting), _texts(counter)
    reassessment.append("新的完整日线、财报或当前成交状态出现变化后重新分析")
    summary = f"本次日线证据截至 {as_of}。"
    summary += "；".join(price_facts) + "。" if price_facts else "价格趋势证据不足。"
    if conflicts:
        summary += "证据分歧：" + "；".join(conflicts) + "。"
    summary += "量价与背景：" + "；".join([*context["量价配合"], *context["相对表现"], *context["市场背景"]]) + "。"
    if risks:
        summary += "已知风险与限制：" + "；".join(_texts(risks)) + "。"
    if not supporting and not counter:
        summary += "当前可确认的原始指标和来源已逐项列出，暂无足够证据归纳方向。"
    if evidence_gaps:
        components = _texts([gap.get("component") for gap in evidence_gaps])
        summary += "仍有待补齐或核验的部分：" + "、".join(components) + "；这些缺口不能当作利好或利空。"
    return {
        "summary": summary,
        "supporting_evidence": supporting,
        "counter_evidence": counter,
        "evidence_context": context,
        "evidence_conflicts": _texts(conflicts),
        "interpretation_basis": "关系判断优先采用未舍入原值，文字数值按展示精度舍入；同源价格指标不作独立投票，未计算综合分或上涨概率",
        "reassessment_conditions": _texts(reassessment),
    }


__all__ = ["goujian_dangu_zhaiyao"]
