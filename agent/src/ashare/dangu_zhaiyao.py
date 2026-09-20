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


def _intraday_effect(snapshot: dict[str, Any], tradability: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if not tradability.get("realtime_required"):
        return {"status": "not_applicable", "reason": "本次处于非盘中核验时段，完整日线仍为比较依据"}
    if tradability.get("current_quote_verified") is not True:
        return {"status": "unavailable", "reason": tradability.get("current_quote_reason") or "当前快照未通过时点与成交核验"}
    price = _number(snapshot.get("last_price", snapshot.get("latest_price")))
    previous = _number(snapshot.get("previous_close"))
    if price is None or previous is None or previous <= 0:
        return {"status": "unavailable", "reason": "缺少可比较的现价或前收盘价"}
    change = price / previous - 1
    result = {"status": "intraday_provisional", "source": snapshot.get("source"),
              "quote_time": snapshot.get("provider_quote_time"), "captured_at": snapshot.get("captured_at"),
              "price": price, "change_from_previous_close": change,
              "effect": "盘中较前收盘价走高" if change > 0 else "盘中较前收盘价走低" if change < 0 else "盘中价格与前收盘价相同",
              "ranking_effect": "当前变化补充解释日线优选理由，未完成的盘中收益不替换完整日线排序"}
    close, ma20 = _number(raw.get("close")), _number(raw.get("ma_20"))
    # 实时未复权价不能直接与历史前复权均线比较；先确认前收盘口径确实对齐。
    if close is not None and ma20 is not None and math.isclose(close, previous, rel_tol=0, abs_tol=0.005):
        result["ma20_reference"] = ma20
        if close > ma20 and price <= ma20:
            result["effect"] += "；价格已回落至本次日线MA20参考值附近或下方，原有均线上方依据减弱，需收盘复核"
        elif close <= ma20 < price:
            result["effect"] += "；价格越过本次日线MA20参考值，出现暂定改善，需收盘复核"
        else:
            result["effect"] += "；相对本次日线MA20的上下方关系暂未改变"
        result["reference_note"] = "沿用已完成日线MA20作观察参考，并非当日最终MA20"
    else:
        result["reference_note"] = "实时前收盘价与历史复权口径未对齐或MA20缺失，不作跨口径价格比较"
    return result


def goujian_dangu_zhaiyao(
    *, technical: dict[str, Any], fundamentals: dict[str, Any],
    risks: list[str], evidence_gaps: list[dict[str, Any]], as_of: str,
    factor_analysis: dict[str, Any] | None = None, pattern: dict[str, Any] | None = None,
    late: dict[str, Any] | None = None, supplemental: dict[str, Any] | None = None,
    research_assessment: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None, tradability: dict[str, Any] | None = None,
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
    for period, average in ((20, ma20), (60, ma60)):
        if close is not None and average is not None:
            reassessment.append(f"后续完整收盘价与{period}日均线的上下方关系改变时复评；本次均线为 {average:.3f}，新日线需重算，不把该数值固定为未来支撑或阻力")
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
    financial_facts = []
    for field, label in (("roe_pct", "净资产收益率"), ("net_profit_yoy_pct", "净利润同比")):
        value = _number(financials.get(field))
        if value is not None:
            financial_facts.append(f"本次财报{label} {value:.6g}%；仅据正负不能判断水平高低或确认价格趋势")
    context["财务背景"] = financial_facts or ["财务证据缺失，不能据此判断公司优劣"]
    intraday = _intraday_effect(snapshot or {}, tradability or {}, raw)
    context["盘中变化"] = _texts([intraday.get("effect") or intraday.get("reason"), intraday.get("reference_note")])
    if intraday.get("status") == "intraday_provisional":
        reassessment.append("取得本交易日完整收盘数据后重算收益、均线与候选比较；盘中变化仍属暂定")
    counter.extend([*conflicts, *risks])
    supporting, counter = _texts(supporting), _texts(counter)
    reassessment.append("新的完整日线、财报或当前成交状态出现变化后重新分析")
    ret20 = _number(returns.get("20d"))
    if close is None or ma20 is None or ret20 is None:
        judgment = "20日表现或均线位置证据不足，暂不能形成完整趋势判断"
    elif ret20 > 0 and close > ma20:
        judgment = "20日已实现表现与均线位置同向偏强"
        if ret5 is not None and ret5 < 0:
            judgment += "，但近5日正在回撤"
    elif ret20 < 0 and close < ma20:
        judgment = "20日已实现表现与均线位置同向偏弱"
        if ret5 is not None and ret5 > 0:
            judgment += "，近5日反弹尚未改变这一背景"
    else:
        judgment = "20日收益与均线位置未形成同向强弱，现有证据有分歧"
    if research_assessment:
        judgment = str(research_assessment["verdict_label"]) + "。总体背景：" + judgment
        reassessment.extend(research_assessment.get("reassessment_conditions") or [])
    summary = f"{judgment}。本次完整日线截至 {as_of}。"
    summary += "；".join(price_facts[:2]) + "。" if price_facts else "价格趋势证据不足。"
    if conflicts:
        summary += "证据分歧：" + "；".join(conflicts) + "。"
    summary += "量价：" + "；".join(context["量价配合"]) + "。"
    if intraday.get("effect"):
        summary += str(intraday["effect"]) + "。"
    if risks:
        summary += "已知风险与限制：" + "；".join(_texts(risks)) + "。"
    if not supporting and not counter:
        summary += "当前可确认的原始指标和来源已逐项列出，暂无足够证据归纳方向。"
    if evidence_gaps:
        components = _texts([gap.get("component") for gap in evidence_gaps])
        summary += "仍有待补齐或核验的部分：" + "、".join(components) + "；这些缺口不能当作利好或利空。"
    return {
        "summary": summary,
        "main_judgment": judgment,
        "research_assessment": research_assessment,
        "intraday_effect": intraday,
        "supporting_evidence": supporting,
        "counter_evidence": counter,
        "evidence_context": context,
        "evidence_conflicts": _texts(conflicts),
        "evidence_families": [
            {"family": "price", "contexts": ["价格与趋势", "动能结构", "相对表现", "形态适用状态"],
             "role": "相关价格观测及其比较；不同窗口、变换和形态不作为多个独立票数"},
            {"family": "price_volume", "contexts": ["量价配合"], "role": "检验价格变化的量能背景；放量本身不等于利好"},
            {"family": "financial", "contexts": ["财务背景"], "role": "公司经营背景；只有与待验证命题直接相关时才成为直接证据"},
        ],
        "interpretation_basis": "关系判断优先采用未舍入原值，文字数值按展示精度舍入；同源价格指标不作独立投票，未计算综合分或上涨概率",
        "reassessment_conditions": _texts(reassessment),
    }


__all__ = ["goujian_dangu_zhaiyao"]
