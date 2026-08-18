"""统一分析的硬过滤、风险计分和结果投影规则。

规则保持无状态，便于选股编排层复用，也避免数据源策略与业务评分互相依赖。
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.fenxi_weipan import WeipanJieduan
from src.ashare.shuju_yuan import huoqu_zhangdieting_guize


_HANGYE_LIQUIDITY_WEIGHT = 0.7


def zhuan_you_xian_shuzhi(value: Any, digits: int | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def zhuan_json_zhi(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _limit_price(previous_close: float, rate: float) -> float:
    return float(
        (Decimal(str(previous_close)) * (Decimal("1") + Decimal(str(rate)))).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def _mingcheng_ying_guolv(name: str) -> str | None:
    value = str(name).strip().upper()
    if "ST" in value:
        return "股票简称包含 ST 风险标记"
    if "退" in value:
        return "股票简称包含退市风险标记"
    if value.startswith(("N", "C")):
        return "新股缺少稳定历史或仍处于特殊涨跌幅阶段"
    return None


def shibie_yizijia_zhangting(
    quote: pd.Series | dict[str, Any],
    *,
    code: str,
    name: str,
    tolerance_yuan: float,
) -> str | None:
    """识别无法合理假设成交的一字或近似一字涨停。"""
    previous_close = zhuan_you_xian_shuzhi(quote.get("previous_close"))
    current_price = zhuan_you_xian_shuzhi(
        quote.get("last_price")
        if quote.get("last_price") is not None
        else quote.get("latest_price")
    )
    open_price = zhuan_you_xian_shuzhi(quote.get("open"))
    high = zhuan_you_xian_shuzhi(quote.get("high"))
    low = zhuan_you_xian_shuzhi(quote.get("low"))
    if previous_close is None or previous_close <= 0:
        return None
    if any(value is None for value in (current_price, open_price, high, low)):
        return None
    rule = huoqu_zhangdieting_guize(code, name)
    if rule.limit_rate is None:
        return None
    upper = _limit_price(previous_close, rule.limit_rate)
    if min(float(open_price), float(high), float(low), float(current_price)) >= upper - tolerance_yuan:
        return "一字或近似一字涨停，不能合理假设成交"
    return None


def jichu_ying_guolv(
    row: pd.Series,
    *,
    analysis_date: pd.Timestamp,
    config: dict[str, Any],
    quote_is_completed: bool,
) -> list[str]:
    reasons: list[str] = []
    name = str(row.get("name") or "")
    name_reason = _mingcheng_ying_guolv(name)
    if name_reason:
        reasons.append(name_reason)
    price = zhuan_you_xian_shuzhi(row.get("latest_price"))
    volume = zhuan_you_xian_shuzhi(row.get("volume"))
    amount = zhuan_you_xian_shuzhi(row.get("amount_yuan"))
    analysis = config.get("fenxi", {})
    if price is None or price <= 0:
        reasons.append("价格字段无效或疑似停牌")
    if quote_is_completed and (volume is None or volume <= 0):
        reasons.append("成交量字段无效或疑似停牌")
    if quote_is_completed and (
        amount is None or amount < float(analysis.get("min_amount_yuan", 50_000_000))
    ):
        reasons.append("成交额低于统一选股流动性底线")
    list_date = pd.to_datetime(row.get("list_date"), errors="coerce")
    # 有些实时板块接口不返回上市日期。此时不在第一层误杀，而由随后下载的
    # 完整日线实际跨度证明上市时间；无法证明仍会在历史完整性过滤中淘汰。
    if not pd.isna(list_date):
        listing_days = int((analysis_date.normalize() - pd.Timestamp(list_date).normalize()).days)
        if listing_days < int(analysis.get("minimum_listing_calendar_days", 180)):
            reasons.append(f"上市仅约 {listing_days} 个自然日，历史不稳定")
    if quote_is_completed:
        try:
            one_price_reason = shibie_yizijia_zhangting(
                row,
                code=str(row.get("ts_code")),
                name=name,
                tolerance_yuan=float(config.get("xingtai", {}).get("limit_up_tolerance_yuan", 0.005)),
            )
            if one_price_reason:
                reasons.append(f"最近完整交易日为{one_price_reason}")
        except ValueError:
            reasons.append("股票代码不满足程序已有的 A 股市场规则")
    return reasons


def choushu_liudongxing_houxuan(data: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(data) <= limit:
        return data.copy().reset_index(drop=True)
    work = data.copy()
    amount = pd.to_numeric(work["amount_yuan"], errors="coerce")
    work["_global_liquidity_rank"] = amount.rank(pct=True)
    if "industry" in work.columns and work["industry"].fillna("").astype(str).ne("").any():
        work["_industry_liquidity_rank"] = work.groupby(work["industry"].fillna("未知"))["amount_yuan"].rank(pct=True)
    else:
        work["_industry_liquidity_rank"] = work["_global_liquidity_rank"]
    work["_prefilter_score"] = (
        _HANGYE_LIQUIDITY_WEIGHT * work["_industry_liquidity_rank"]
        + (1.0 - _HANGYE_LIQUIDITY_WEIGHT) * work["_global_liquidity_rank"]
    )
    return (
        work.sort_values(["_prefilter_score", "amount_yuan"], ascending=False)
        .head(limit)
        .drop(columns=["_global_liquidity_rank", "_industry_liquidity_rank", "_prefilter_score"])
        .reset_index(drop=True)
    )


def guolv_lishi_wanzhengxing(
    histories: dict[str, pd.DataFrame],
    *,
    analysis_date: pd.Timestamp,
    minimum_rows: int,
    minimum_amount: float,
    minimum_listing_calendar_days: int = 180,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    accepted: dict[str, pd.DataFrame] = {}
    rejected: list[dict[str, Any]] = []
    for code, raw in histories.items():
        data = raw.copy() if raw is not None else pd.DataFrame()
        if data.empty or "trade_date" not in data.columns:
            rejected.append({"ts_code": code, "reason": "没有可用历史日线"})
            continue
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
        data = data.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        if len(data) < minimum_rows:
            rejected.append({"ts_code": code, "reason": f"历史只有 {len(data)} 行，少于 {minimum_rows} 行"})
            continue
        earliest_date = pd.Timestamp(data.iloc[0]["trade_date"]).normalize()
        observed_calendar_days = int((analysis_date.normalize() - earliest_date).days)
        if observed_calendar_days < minimum_listing_calendar_days:
            rejected.append(
                {
                    "ts_code": code,
                    "reason": (
                        f"远端日线只能证明约 {observed_calendar_days} 个自然日的上市历史，"
                        f"不足 {minimum_listing_calendar_days} 日新股风险门槛"
                    ),
                }
            )
            continue
        latest_date = pd.Timestamp(data.iloc[-1]["trade_date"]).normalize()
        if latest_date != analysis_date.normalize():
            rejected.append({"ts_code": code, "reason": f"最新日线停留在 {latest_date.strftime('%Y-%m-%d')}"})
            continue
        latest_amount = zhuan_you_xian_shuzhi(data.iloc[-1].get("amount_yuan"))
        if latest_amount is None or latest_amount < minimum_amount:
            rejected.append({"ts_code": code, "reason": "最新完整日线成交额低于流动性底线"})
            continue
        accepted[code] = data
    return accepted, rejected


def goujian_kuaizhao_jilu(row: pd.Series | dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    raw = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {
        "status": "ok",
        "source": metadata.get("source"),
        "captured_at": metadata.get("captured_at"),
        "name": zhuan_json_zhi(raw.get("name")),
        "last_price": zhuan_you_xian_shuzhi(raw.get("latest_price")),
        "latest_price": zhuan_you_xian_shuzhi(raw.get("latest_price")),
        "open": zhuan_you_xian_shuzhi(raw.get("open")),
        "high": zhuan_you_xian_shuzhi(raw.get("high")),
        "low": zhuan_you_xian_shuzhi(raw.get("low")),
        "previous_close": zhuan_you_xian_shuzhi(raw.get("previous_close")),
        "pct_change": zhuan_you_xian_shuzhi(raw.get("pct_chg")),
        "pct_chg": zhuan_you_xian_shuzhi(raw.get("pct_chg")),
        "volume": zhuan_you_xian_shuzhi(raw.get("volume")),
        "amount_yuan": zhuan_you_xian_shuzhi(raw.get("amount_yuan")),
        "turnover_rate_pct": zhuan_you_xian_shuzhi(raw.get("turnover_rate")),
        "turnover_rate": zhuan_you_xian_shuzhi(raw.get("turnover_rate")),
        "volume_ratio": zhuan_you_xian_shuzhi(raw.get("volume_ratio")),
        "circulating_market_value_yuan": zhuan_you_xian_shuzhi(raw.get("circulating_market_value_yuan")),
    }


def xuyao_shishi_kuaizhao(clock: dict[str, Any], late_stage: WeipanJieduan) -> bool:
    return str(clock.get("session_status")) in {"opening_auction", "trading", "midday_break", "close_pending"} or late_stage is not WeipanJieduan.BU_SHIYONG


def jisuan_fengxian_koufen(
    *,
    code: str,
    name: str,
    snapshot: dict[str, Any],
    factor: dict[str, Any],
    pattern: dict[str, Any],
    late: dict[str, Any],
    config: dict[str, Any],
    technical: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    settings = config.get("fenxi", {})
    penalties: list[dict[str, Any]] = []
    risks: list[str] = []
    previous_close = zhuan_you_xian_shuzhi(snapshot.get("previous_close"))
    current_price = zhuan_you_xian_shuzhi(snapshot.get("last_price") if snapshot.get("last_price") is not None else snapshot.get("latest_price"))
    if previous_close is not None and previous_close > 0 and current_price is not None:
        try:
            rule = huoqu_zhangdieting_guize(code, name)
            if rule.limit_rate is not None:
                upper = _limit_price(previous_close, rule.limit_rate)
                if current_price >= upper * 0.98:
                    points = float(settings.get("near_limit_up_penalty", 8))
                    penalties.append({"reason": "价格接近涨停，成交与追价风险较高", "points": points})
                    risks.append("价格接近涨停，不能假设按当前显示价顺利成交")
        except ValueError:
            risks.append("涨跌幅边界无法确认")
    volatility = None
    if technical:
        volatility = zhuan_you_xian_shuzhi(technical.get("annualized_volatility_20"))
    if volatility is None:
        volatility = zhuan_you_xian_shuzhi(((factor.get("groups") or {}).get("risk_liquidity") or {}).get("values", {}).get("volatility_20"))
    if volatility is not None and volatility > float(settings.get("high_volatility_threshold", 0.55)):
        points = float(settings.get("high_volatility_penalty", 6))
        penalties.append({"reason": "20 日年化波动较高", "points": points})
        risks.append(f"20 日年化波动约 {volatility:.1%}，短线不确定性较高")
    if float(factor.get("confidence") or 0.0) < 0.5:
        risks.append("八组日 K 因子的可用覆盖偏低")
    if pattern.get("state") in {"invalidated", "expired"}:
        risks.extend(str(value) for value in pattern.get("failure_reasons", [])[:2])
    if (late.get("actuals") or {}).get("latest_5min_volume_anomaly"):
        risks.append("最新 5 分钟成交额相对前序区间异常，尾盘证据稳定性下降")
    maximum = float(settings.get("risk_penalty_max", 30))
    used = 0.0
    bounded: list[dict[str, Any]] = []
    for item in penalties:
        remaining = max(0.0, maximum - used)
        points = min(float(item["points"]), remaining)
        if points <= 0:
            break
        bounded.append({**item, "points": round(points, 2)})
        used += points
    return bounded, list(dict.fromkeys(risks))


def hecheng_houxuan_fenshu(
    *,
    factor: dict[str, Any],
    fundamental: dict[str, Any],
    pattern: dict[str, Any],
    late: dict[str, Any],
    penalties: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    weights = {key: float(value) for key, value in config.get("fenxi", {}).get("component_weights", {}).items()}
    components = {
        "daily_factors": {
            "score": zhuan_you_xian_shuzhi(factor.get("score_0_100")),
            "confidence": zhuan_you_xian_shuzhi(factor.get("confidence")) or 0.0,
        },
        "fundamental": {
            "score": zhuan_you_xian_shuzhi(fundamental.get("score_0_100")),
            "confidence": zhuan_you_xian_shuzhi(fundamental.get("confidence")) or 0.0,
        },
        "pattern": {
            "score": zhuan_you_xian_shuzhi(pattern.get("score_0_100")),
            "confidence": 1.0 if pattern.get("eligible") else 0.0,
        },
        "late_session": {
            "score": zhuan_you_xian_shuzhi(late.get("score_0_100")),
            "confidence": zhuan_you_xian_shuzhi(late.get("confidence")) or 0.0,
        },
    }
    valid = [
        (name, values["score"], weights.get(name, 0.0))
        for name, values in components.items()
        if values["score"] is not None and weights.get(name, 0.0) > 0
    ]
    available_weight = sum(weight for _, _, weight in valid)
    base_score = (
        sum(float(score) * weight for _, score, weight in valid) / available_weight
        if available_weight > 0
        else None
    )
    penalty = sum(float(item.get("points") or 0.0) for item in penalties)
    final_score = max(0.0, min(100.0, float(base_score) - penalty)) if base_score is not None else None
    confidence = sum(
        weights.get(name, 0.0) * min(1.0, max(0.0, float(values["confidence"])))
        for name, values in components.items()
        if values["score"] is not None
    )
    return {
        "score_0_100": round(final_score, 2) if final_score is not None else None,
        "base_score_0_100": round(float(base_score), 2) if base_score is not None else None,
        "risk_penalty": round(penalty, 2),
        "confidence": round(min(1.0, max(0.0, confidence)), 4),
        "available_component_weight": round(available_weight, 4),
        "components": {
            name: {
                "score_0_100": values["score"],
                "confidence": round(float(values["confidence"]), 4),
                "configured_weight": weights.get(name, 0.0),
                "used": values["score"] is not None,
            }
            for name, values in components.items()
        },
        "penalties": penalties,
        "definition": "有效证据按配置权重重新归一后扣除独立风险分；只用于候选间比较，不是上涨概率",
    }


def jisuan_jibenmian_xinren(fundamentals: dict[str, Any]) -> float:
    financials = fundamentals.get("financials") or {}
    valuation = fundamentals.get("valuation") or {}
    observed = sum(
        zhuan_you_xian_shuzhi(value) is not None
        for value in (
            financials.get("roe_pct"),
            financials.get("net_profit_yoy_pct"),
            financials.get("debt_to_assets_pct"),
            valuation.get("pe_ttm") if valuation.get("pe_ttm") is not None else valuation.get("pe_dynamic"),
        )
    )
    return round(min(1.0, observed / 4.0), 4)


def goujian_kejiaoyixing_zhaiyao(
    *,
    code: str,
    name: str,
    snapshot: dict[str, Any],
    history: pd.DataFrame,
    minimum_amount: float,
) -> dict[str, Any]:
    latest = history.iloc[-1]
    amount = zhuan_you_xian_shuzhi(latest.get("amount_yuan"))
    hard_blocks: list[str] = []
    cautions: list[str] = []
    if amount is None or amount < minimum_amount:
        hard_blocks.append("最新完整日线成交额低于统一流动性底线")
    if snapshot.get("status") != "ok":
        cautions.append("实时快照不可用，可交易性仅按完整日线判断")
    current_price = zhuan_you_xian_shuzhi(
        snapshot.get("last_price")
        if snapshot.get("last_price") is not None
        else snapshot.get("latest_price")
    )
    analysis_price = current_price or zhuan_you_xian_shuzhi(latest.get("close"))
    try:
        price_rule = huoqu_zhangdieting_guize(code, name)
        price_limit_status = price_rule.status
        price_limit_pct = (
            round(float(price_rule.limit_rate) * 100.0, 2)
            if price_rule.limit_rate is not None
            else None
        )
    except ValueError as exc:
        price_limit_status = "unavailable"
        price_limit_pct = None
        cautions.append(f"涨跌幅规则不可用：{exc}")
    return {
        "status": "blocked" if hard_blocks else "caution" if cautions else "tradable",
        "basic_execution_feasible": not hard_blocks,
        "analysis_price": analysis_price,
        "analysis_price_basis": (
            "realtime_snapshot"
            if current_price is not None
            else "latest_completed_qfq_close"
        ),
        "amount_yuan": amount,
        "minimum_amount_yuan": minimum_amount,
        "price_limit_status": price_limit_status,
        "price_limit_pct": price_limit_pct,
        "hard_blocks": hard_blocks,
        "cautions": cautions,
    }


def goujian_houxuan_zhaiyao(
    item: dict[str, Any],
    *,
    rank: int,
    minimum_score: float,
    minimum_confidence: float,
) -> dict[str, Any]:
    ranking = item["ranking"]
    score = zhuan_you_xian_shuzhi(ranking.get("score_0_100"))
    confidence = float(ranking.get("confidence") or 0.0)
    eligible = bool(
        score is not None
        and score >= minimum_score
        and confidence >= minimum_confidence
        and item.get("tradability", {}).get("basic_execution_feasible", True)
    )
    factor = item.get("factor") or {}
    groups = factor.get("groups") or {}
    positive: list[str] = []
    unmet: list[str] = []
    for value in groups.values():
        if not isinstance(value, dict):
            continue
        group_score = zhuan_you_xian_shuzhi(value.get("score_0_100"))
        label = str(value.get("label") or "因子组")
        if group_score is not None and group_score >= 65:
            positive.append(f"{label} {group_score:.1f}/100")
        elif group_score is None:
            unmet.append(f"{label}信息有限")
        elif group_score < 50:
            unmet.append(f"{label}仅 {group_score:.1f}/100")
    positive.extend(str(value) for value in (item.get("fundamental") or {}).get("evidence", [])[:2])
    if item.get("pattern", {}).get("state") in {"intraday_confirmed", "close_confirmed", "waiting_breakout"}:
        positive.append(str(item["pattern"].get("state_label")))
    positive.extend(str(value) for value in item.get("late", {}).get("evidence", [])[:2])
    if item.get("pattern", {}).get("state") not in {"intraday_confirmed", "close_confirmed"}:
        unmet.append("涨停回马枪尚未确认")
    if item.get("late", {}).get("score_0_100") is None:
        unmet.append("当前时段没有可用尾盘证据")
    if score is not None and score < minimum_score:
        unmet.append(f"综合排名分低于推荐门槛 {minimum_score:.1f}")
    if confidence < minimum_confidence:
        unmet.append(f"证据可信度低于门槛 {minimum_confidence:.2f}")
    technical = item.get("technical") or {}
    return {
        "rank": rank,
        "ts_code": item["ts_code"],
        "name": item["name"],
        "industry": item.get("industry"),
        "ranking_score_0_100": score,
        "ranking_score_definition": ranking.get("definition"),
        "confidence": confidence,
        "meets_recommendation_threshold": eligible,
        "positive_evidence": list(dict.fromkeys(positive))[:8],
        "unmet_conditions": list(dict.fromkeys(unmet))[:8],
        "daily_factor_analysis": factor,
        "fundamental_analysis": item.get("fundamental"),
        "limit_up_pullback_pattern": item.get("pattern"),
        "late_session_analysis": item.get("late"),
        "technical_summary": {
            "trade_date": technical.get("trade_date"),
            "close": technical.get("close"),
            "score_0_100": technical.get("score_0_100"),
            "annualized_volatility_20": technical.get("annualized_volatility_20"),
            "evidence": technical.get("evidence"),
        },
        "tradability": item.get("tradability"),
        "data_quality": item.get("data_quality"),
        "risks": item.get("risks"),
        "risk_reference_price": item.get("pattern", {}).get("risk_reference_price"),
        "ranking_details": ranking,
        "suggest_prediction_stage": eligible,
    }


__all__ = [
    "choushu_liudongxing_houxuan",
    "goujian_houxuan_zhaiyao",
    "goujian_kejiaoyixing_zhaiyao",
    "goujian_kuaizhao_jilu",
    "guolv_lishi_wanzhengxing",
    "hecheng_houxuan_fenshu",
    "jichu_ying_guolv",
    "jisuan_fengxian_koufen",
    "jisuan_jibenmian_xinren",
    "shibie_yizijia_zhangting",
    "xuyao_shishi_kuaizhao",
    "zhuan_json_zhi",
    "zhuan_you_xian_shuzhi",
]
