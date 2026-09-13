"""统一分析的硬过滤、原始证据、上涨条件与 Pareto 比较规则。"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.fenxi_weipan import WeipanJieduan
from src.ashare.dangu_zhaiyao import goujian_dangu_zhaiyao
from src.ashare.shuju_yuan import heyan_kuaizhao_shidian, huoqu_zhangdieting_guize


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


def shishi_ying_guolv(
    snapshot: dict[str, Any],
    *,
    code: str,
    name: str,
    config: dict[str, Any],
) -> list[str]:
    """单股与范围选股共用当前行情的硬拦截及错误口径。"""
    try:
        reason = shibie_yizijia_zhangting(
            snapshot,
            code=code,
            name=name,
            tolerance_yuan=float(config.get("xingtai", {}).get("limit_up_tolerance_yuan", 0.005)),
        )
    except ValueError:
        reason = "股票代码不满足程序已有的 A 股市场规则"
    return [f"当前行情为{reason}"] if reason else []


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
    """按真实成交额排序，在行业内依次取样；轮次之间不合成流动性分数。"""
    if len(data) <= limit:
        return data.copy().reset_index(drop=True)
    work = data.copy()
    work["_amount"] = pd.to_numeric(work.get("amount_yuan"), errors="coerce")
    work["_industry"] = work.get("industry", pd.Series("", index=work.index)).fillna("").astype(str)
    work["_code"] = work.get("ts_code", pd.Series("", index=work.index)).astype(str)
    work = work.sort_values(["_amount", "_code"], ascending=[False, True], na_position="last", kind="stable")
    work["_industry_round"] = work.groupby("_industry", sort=False).cumcount()
    return (
        work.sort_values(["_industry_round", "_amount", "_code"], ascending=[True, False, True], na_position="last", kind="stable")
        .head(max(0, limit))
        .drop(columns=["_amount", "_industry", "_code", "_industry_round"])
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
        "provider_trade_date": zhuan_json_zhi(raw.get("provider_trade_date", raw.get("trade_date", metadata.get("provider_trade_date")))),
        "provider_quote_time": zhuan_json_zhi(raw.get("provider_quote_time")),
        "timeliness": metadata.get("timeliness"),
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


def hebing_jibenmian_zhengju(
    cross_section: dict[str, Any],
    fundamentals: dict[str, Any],
) -> dict[str, Any]:
    """保留深度财务原值与来源，再附上独立的同日估值比较证据。"""
    return {
        **fundamentals,
        "cross_section_analysis": cross_section,
        "relative_valuation": cross_section.get("relative_valuation") or {},
        "evidence": list(dict.fromkeys([
            *[str(value) for value in fundamentals.get("evidence", [])],
            *[str(value) for value in cross_section.get("evidence", [])],
        ])),
    }


def _factor_value(item: dict[str, Any], group: str, feature: str) -> float | None:
    values = (((item.get("factor") or {}).get("groups") or {}).get(group) or {}).get("values") or {}
    return zhuan_you_xian_shuzhi(values.get(feature))


def _technical_state(item: dict[str, Any]) -> tuple[str, str | None, str]:
    technical = item.get("technical") or {}
    structure = technical.get("macd_structure") or {}
    outcome = technical.get("outcome") or structure.get("outcome")
    status = str(technical.get("status") or structure.get("status") or "unavailable")
    if technical.get("status") == "error" or structure.get("status") == "error" or outcome == "program_error":
        status, outcome = "error", "program_error"
    reason = str(technical.get("reason") or technical.get("error") or structure.get("reason") or structure.get("error") or "")
    return status, outcome, reason


def pinggu_shangzhang_tiaojian(
    item: dict[str, Any],
    *,
    config: dict[str, Any],
    deep_reviewed: bool = False,
) -> dict[str, Any]:
    """逐项核验已有上涨信号；缺失不是通过，不折算命中分或概率。"""
    conditions: list[dict[str, Any]] = []

    def add(key: str, label: str, values: dict[str, Any], passed: bool, rule: str) -> None:
        missing = [field for field, value in values.items() if value is None]
        status = "unavailable" if missing else "met" if passed else "unmet"
        conditions.append({"key": key, "label": label, "status": status, "values": values,
                           "reason": "缺少必需字段：" + "、".join(missing) if missing else rule})

    ma_gap = _factor_value(item, "trend_structure", "ma_gap_20")
    ma_trend = _factor_value(item, "trend_structure", "ma_trend_5_20")
    add("trend_alignment", "收盘价与短期均线位于20日均线上方", {"ma_gap_20": ma_gap, "ma_trend_5_20": ma_trend},
        ma_gap is not None and ma_trend is not None and ma_gap > 0 and ma_trend > 0,
        "要求收盘价高于20日均线，且5日均线高于20日均线")
    ret5 = _factor_value(item, "momentum_reversal", "ret_5")
    ret20 = _factor_value(item, "momentum_reversal", "ret_20")
    macd_hist = _factor_value(item, "momentum_reversal", "macd_hist_pct")
    add("positive_momentum", "5日、20日收益及MACD柱同时为正", {"ret_5": ret5, "ret_20": ret20, "macd_hist_pct": macd_hist},
        all(value is not None and value > 0 for value in (ret5, ret20, macd_hist)),
        "要求5日收益、20日收益和MACD柱均大于0")
    excess = _factor_value(item, "relative_strength", "excess_vs_csi300_ret_20")
    add("outperform_csi300", "20日表现强于沪深300", {"excess_vs_csi300_ret_20": excess},
        excess is not None and excess > 0, "要求20日相对沪深300超额收益大于0")
    volume_ratio = _factor_value(item, "price_volume_confirmation", "volume_ratio_5_20")
    add("volume_confirmation", "5日均量不低于20日均量", {"volume_ratio_5_20": volume_ratio},
        volume_ratio is not None and volume_ratio >= 1, "要求5日/20日均量比不低于1倍")
    volatility = _factor_value(item, "risk_liquidity", "volatility_20")
    if volatility is not None and volatility < 0:
        volatility = None
    threshold = zhuan_you_xian_shuzhi((config.get("fenxi") or {}).get("high_volatility_threshold", 0.55))
    add("bounded_volatility", "20日年化波动未超过既有上限", {"volatility_20": volatility, "high_volatility_threshold": threshold},
        volatility is not None and threshold is not None and volatility <= threshold,
        "要求20日年化波动不高于配置上限")

    technical_status, technical_outcome, technical_reason = _technical_state(item)
    if deep_reviewed:
        reviewed_histogram = zhuan_you_xian_shuzhi(((item.get("technical") or {}).get("macd") or {}).get("histogram"))
        technical_available = technical_status in {"ok", "partial"} and technical_outcome != "program_error" and reviewed_histogram is not None
        conditions.append({
            "key": "technical_review", "label": "技术复核已返回有效证据",
            "status": "met" if technical_available else "unavailable",
            "values": {"status": technical_status, "outcome": technical_outcome, "macd_histogram": reviewed_histogram},
            "reason": technical_reason or ("技术复核已完成；必需指标仍逐项核验" if technical_available else "技术复核未完成、MACD柱缺失或来源不可用"),
        })
    tradability = item.get("tradability") or {}
    hard_blocks = list(tradability.get("hard_blocks") or [])
    if deep_reviewed or hard_blocks or tradability.get("basic_execution_feasible") is False:
        current_required = bool(tradability.get("realtime_required"))
        current_verified = tradability.get("current_quote_verified") is True
        feasible = tradability.get("basic_execution_feasible")
        if current_required and not current_verified:
            trade_status = "unavailable"
            reason = "当前行情尚未核验，历史日线不能替代盘中成交核验"
        elif hard_blocks or feasible is False:
            trade_status = "unmet"
            reason = "；".join(hard_blocks) or "当前可交易性检查未通过"
        elif feasible is True:
            trade_status = "met"
            reason = "当前适用的可交易性检查通过"
        else:
            trade_status = "unavailable"
            reason = "缺少可交易性检查结果"
        conditions.append({"key": "tradability", "label": "通过当前适用的可交易性检查", "status": trade_status,
                           "values": {"basic_execution_feasible": feasible, "realtime_required": current_required,
                                      "current_quote_verified": current_verified, "hard_blocks": hard_blocks}, "reason": reason})
    missing = [condition["key"] for condition in conditions if condition["status"] == "unavailable"]
    unmet = [condition["key"] for condition in conditions if condition["status"] == "unmet"]
    return {
        "status": "error" if deep_reviewed and technical_outcome == "program_error" else "partial" if missing else "ok",
        "outcome": "program_error" if deep_reviewed and technical_outcome == "program_error" else "evidence_unavailable" if missing else "conditions_evaluated",
        "conditions": conditions,
        "eligible": all(condition["status"] == "met" for condition in conditions),
        "missing_conditions": missing,
        "unmet_conditions": unmet,
    }


def _real_amount(item: dict[str, Any]) -> tuple[float | None, str | None]:
    history = item.get("history")
    if isinstance(history, pd.DataFrame) and not history.empty:
        amount = zhuan_you_xian_shuzhi(history.iloc[-1].get("amount_yuan"))
        if amount is not None and amount > 0:
            return amount, "latest_completed_daily_bar"
    tradability = item.get("tradability") or {}
    if tradability.get("amount_basis") == "latest_completed_daily_bar":
        amount = zhuan_you_xian_shuzhi(tradability.get("amount_yuan"))
        if amount is not None and amount > 0:
            return amount, "latest_completed_daily_bar"
    return None, None


def paixu_shangzhang_houxuan(
    items: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    deep_reviewed: bool = False,
) -> list[dict[str, Any]]:
    """满足条件者优先，再做完整五维非支配分层；不合成标量排序分。"""
    prepared: list[dict[str, Any]] = []
    for item in items:
        result = dict(item)
        selection = pinggu_shangzhang_tiaojian(item, config=config, deep_reviewed=deep_reviewed)
        amount, amount_basis = _real_amount(item)
        dimensions = {
            "excess_vs_csi300_ret_20": _factor_value(item, "relative_strength", "excess_vs_csi300_ret_20"),
            "ret_5": _factor_value(item, "momentum_reversal", "ret_5"),
            "ret_20": _factor_value(item, "momentum_reversal", "ret_20"),
            "volatility_20": _factor_value(item, "risk_liquidity", "volatility_20"),
            "amount_yuan": amount,
        }
        if dimensions["volatility_20"] is not None and dimensions["volatility_20"] < 0:
            dimensions["volatility_20"] = None
        missing = [field for field, value in dimensions.items() if value is None]
        selection.update({
            "comparison_values": dimensions,
            "comparison_missing_fields": missing,
            "comparison_status": "unavailable" if missing else "complete",
            "amount_basis": amount_basis,
            "pareto_front": None,
            "ranking_basis": "先列出满足全部上涨条件且比较数据完整者；完整五维按非支配层排列，波动越低、其余四维越高越优。同层仅按真实成交额和代码稳定展示，不代表上涨概率优劣。缺比较字段者列为观察对象并排后。",
        })
        result["selection"] = selection
        prepared.append(result)

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        lv, rv = left["selection"]["comparison_values"], right["selection"]["comparison_values"]
        left_values = [value if key != "volatility_20" else -value for key, value in lv.items()]
        right_values = [value if key != "volatility_20" else -value for key, value in rv.items()]
        return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))

    for eligible in (True, False):
        remaining = [i for i, item in enumerate(prepared) if item["selection"]["eligible"] is eligible and not item["selection"]["comparison_missing_fields"]]
        front = 1
        while remaining:
            current = [i for i in remaining if not any(dominates(prepared[j], prepared[i]) for j in remaining if j != i)]
            for index in current:
                prepared[index]["selection"]["pareto_front"] = front
            chosen = set(current)
            remaining = [index for index in remaining if index not in chosen]
            front += 1

    def display_order(item: dict[str, Any]) -> tuple[Any, ...]:
        selection = item["selection"]
        complete = not selection["comparison_missing_fields"]
        category = 0 if complete and selection["eligible"] else 1 if complete else 2
        amount = selection["comparison_values"]["amount_yuan"]
        return (category, selection["pareto_front"] or math.inf, amount is None,
                -amount if amount is not None else 0, str(item.get("ts_code") or ""))

    return sorted(prepared, key=display_order)


def goujian_kejiaoyixing_zhaiyao(
    *,
    code: str,
    name: str,
    snapshot: dict[str, Any],
    history: pd.DataFrame,
    minimum_amount: float,
    realtime_required: bool = False,
    reference_time: Any = None,
) -> dict[str, Any]:
    latest = history.iloc[-1]
    amount = zhuan_you_xian_shuzhi(latest.get("amount_yuan"))
    amount_trade_date_raw = pd.to_datetime(latest.get("trade_date"), errors="coerce")
    amount_trade_date = (
        pd.Timestamp(amount_trade_date_raw).strftime("%Y-%m-%d")
        if not pd.isna(amount_trade_date_raw)
        else None
    )
    hard_blocks: list[str] = []
    cautions: list[str] = []
    if amount is None or amount < minimum_amount:
        hard_blocks.append("最新完整日线成交额低于统一流动性底线")
    if snapshot.get("status") != "ok":
        cautions.append("实时快照不可用，完整日线只能提供历史成交参考")
    current_price = zhuan_you_xian_shuzhi(
        snapshot.get("last_price")
        if snapshot.get("last_price") is not None
        else snapshot.get("latest_price")
    )
    current_fields = [current_price] + [
        zhuan_you_xian_shuzhi(snapshot.get(field))
        for field in ("previous_close", "open", "high", "low")
    ]
    time_check = heyan_kuaizhao_shidian(
        snapshot, expected_trade_date=reference_time if realtime_required else amount_trade_date,
        reference_time=reference_time, require_timestamp=realtime_required,
    )
    current_verified = bool(
        snapshot.get("status") == "ok"
        and all(value is not None and value > 0 for value in current_fields)
        and time_check["verified"]
    )
    quote_reason = (
        str(snapshot.get("error") or "实时快照不可用") if snapshot.get("status") != "ok"
        else "实时价格或涨跌停核验字段不完整" if any(value is None or value <= 0 for value in current_fields)
        else str(time_check["reason"])
    )
    if snapshot.get("status") == "ok" and not time_check["verified"]:
        cautions.append(str(time_check["reason"]))
    if realtime_required and not current_verified:
        hard_blocks.append("盘中行情时点或实时价格、涨跌停字段未通过核验，无法确认当前可交易性")
    analysis_price = current_price if current_verified else zhuan_you_xian_shuzhi(latest.get("close"))
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
        "realtime_required": realtime_required,
        "current_quote_verified": current_verified,
        "current_quote_status": "verified" if current_verified else "unavailable",
        "current_quote_reason": quote_reason,
        "quote_time_verification": time_check,
        "analysis_price": analysis_price,
        "analysis_price_basis": (
            "realtime_snapshot"
            if current_verified
            else "latest_completed_qfq_close"
        ),
        "amount_yuan": amount,
        "amount_basis": "latest_completed_daily_bar",
        "amount_trade_date": amount_trade_date,
        "minimum_amount_yuan": minimum_amount,
        "price_limit_status": price_limit_status,
        "price_limit_pct": price_limit_pct,
        "hard_blocks": hard_blocks,
        "cautions": cautions,
    }


def goujian_zhengju_quekou(item: dict[str, Any]) -> list[dict[str, Any]]:
    """逐项列出缺失证据及来源错误；缺失不会被解释成风险为零。"""
    gaps: list[dict[str, Any]] = []

    def append(code: str, component: str, reason: str, reassess: str, *, blocks_selection: bool = False, **details: Any) -> None:
        gaps.append({"code": code, "component": component, "reason": reason,
                     "impact": "该项证据尚不支持完整判断；已有原始指标仍保留",
                     "blocks_selection": blocks_selection, "reassessment_condition": reassess, **details})

    tradability = item.get("tradability") or {}
    if tradability.get("realtime_required") and not tradability.get("current_quote_verified"):
        append("current_quote_unverified", "当前行情",
               str(tradability.get("current_quote_reason") or (item.get("snapshot") or {}).get("error") or "实时价格或涨跌停核验字段不完整"),
               "重新取得完整实时行情并通过当前可交易性核验", blocks_selection=True)
    status, outcome, reason = _technical_state(item)
    if status == "error" or outcome == "program_error":
        append("technical_program_error", "技术复核", reason or "技术复核程序错误",
               "修复技术复核错误后重新获取数据并完成诊断", blocks_selection=True,
               source_status=status, source_outcome=outcome)
    elif status in {"unavailable", "insufficient_data"}:
        append("technical_evidence_unavailable", "技术复核", reason or "技术复核证据不足",
               "补齐有效技术指标所需日线后重新诊断", blocks_selection=True,
               source_status=status, source_outcome=outcome)
    for group_key, group in ((item.get("factor") or {}).get("groups") or {}).items():
        missing = list(group.get("missing_fields") or [])
        if missing:
            append("daily_factor_" + group_key, str(group.get("label") or group_key),
                   "部分原始指标或比较样本缺失", "补齐缺失日线、真实行业或比较样本后重新计算",
                   missing_fields=missing)
    fundamental = item.get("fundamental") or {}
    deep = fundamental.get("deep_analysis") or fundamental
    financials = deep.get("financials") or {}
    valuation = deep.get("valuation") or {}
    missing = [label for field, label in (("roe_pct", "净资产收益率"), ("net_profit_yoy_pct", "净利润同比增长"),
                                        ("debt_to_assets_pct", "资产负债率"))
               if zhuan_you_xian_shuzhi(financials.get(field)) is None]
    if all(zhuan_you_xian_shuzhi(valuation.get(field)) is None for field in ("pe_ttm", "pe_dynamic", "pe")):
        missing.append("市盈率")
    if missing or deep.get("errors"):
        append("fundamental_fields_missing", "基本面", "缺少" + "、".join(missing) if missing else "基本面来源报告错误",
               "取得并核验缺失的财务或估值字段后重新诊断", missing_fields=missing,
               source_errors=list(deep.get("errors") or []))
    late = item.get("late") or {}
    if late.get("status") == "unavailable":
        append("late_session_evidence_missing", "尾盘证据", str(late.get("reason") or "尾盘数据不可用"),
               "在适用时段取得有效尾盘数据后复核")
    for block in (item.get("supplemental_diagnostics") or {}).get("blocks", []):
        if not isinstance(block, dict) or block.get("status") == "ok":
            continue
        label = str(block.get("label") or "补充诊断")
        append("supplemental_" + str(block.get("key") or "missing"), label,
               str(block.get("missing_reason") or block.get("summary") or "补充证据不足"),
               f"补齐并核验{label}所需数据后重新诊断")
    existing_codes = {gap["code"] for gap in gaps}
    for condition in (item.get("selection") or {}).get("conditions", []):
        if condition.get("status") != "unavailable":
            continue
        code = "selection_" + str(condition["key"])
        if code not in existing_codes:
            append(code, str(condition["label"]), str(condition["reason"]),
                   "补齐该条件的必需数据后重新核验", blocks_selection=True)
    return gaps


def goujian_zhen_duan_shixiaoxing(
    *,
    as_of: str,
    generated_at: str,
    clock: dict[str, Any],
    realtime_status: str,
    realtime_required: bool,
) -> dict[str, Any]:
    session = str(clock.get("session_status") or "unknown")
    confirmation = (
        "intraday_provisional" if session in {"opening_auction", "trading", "midday_break"}
        else "close_pending" if session == "close_pending"
        else "completed_daily_close"
    )
    explanation = f"日线证据截至 {as_of}；本次诊断生成于 {generated_at}。"
    if realtime_required:
        explanation += (
            "盘中价格和成交状态仍会变化，当前结论为暂定，需收盘后复核。"
            if realtime_status == "verified"
            else "当前实时成交核验未完成，已有证据仅支持截至最近完整日线的历史诊断。"
        )
    else:
        explanation += "结论基于最近完整日线；下次交易前仍需重新确认价格和成交条件。"
    return {
        "daily_data_as_of": as_of,
        "generated_at": generated_at,
        "session_status": session,
        "result_confirmation": confirmation,
        "realtime_required": realtime_required,
        "realtime_status": realtime_status,
        "explanation": explanation,
        "reassess_when": ["出现新的完整日线后", "价格、成交状态或已列风险发生变化时", "缺失证据补齐后"],
    }


def goujian_houxuan_zhaiyao(
    item: dict[str, Any],
    *,
    rank: int,
) -> dict[str, Any]:
    """投影完整诊断证据和显式筛选条件，不生成分数或买卖结论。"""
    selection = dict(item.get("selection") or {})
    conditions = selection.get("conditions") or []
    positive = [f"{condition['label']}：{condition['reason']}" for condition in conditions if condition.get("status") == "met"]
    unmet = [f"{condition['label']}：{condition['reason']}" for condition in conditions if condition.get("status") == "unmet"]
    technical = item.get("technical") or {}
    macd_structure = technical.get("macd_structure") or {}
    if macd_structure.get("status") == "ok":
        positive.extend(str(value) for value in macd_structure.get("supporting_evidence", []))
    pattern = item.get("pattern") or {}
    if pattern.get("eligible") and pattern.get("state") in {"intraday_confirmed", "close_confirmed"}:
        positive.append(str(pattern.get("state_label") or pattern["state"]))
    late = item.get("late") or {}
    positive.extend(str(condition.get("reason") or condition.get("label")) for condition in late.get("conditions", [])
                    if isinstance(condition, dict) and condition.get("status") == "met")
    evidence_gaps = goujian_zhengju_quekou(item)
    risks = list(dict.fromkeys([
        *[str(value) for value in macd_structure.get("risk_warnings", [])],
        *[str(value) for value in pattern.get("failure_reasons", []) if pattern.get("state") == "invalidated"],
        *[str(value) for value in (item.get("tradability") or {}).get("hard_blocks", [])],
        *[str(value) for value in (item.get("tradability") or {}).get("cautions", [])],
        *[str(value) for value in item.get("risks") or []],
    ]))
    diagnosis = goujian_dangu_zhaiyao(
        technical=technical, fundamentals=item.get("fundamental") or {}, risks=risks,
        evidence_gaps=evidence_gaps, as_of=str((item.get("data_quality") or {}).get("as_of") or technical.get("trade_date") or "未取得日期"),
        factor_analysis=item.get("factor"), pattern=pattern, late=late,
        supplemental=item.get("supplemental_diagnostics"),
    )
    return {
        "rank": rank,
        "diagnosis_summary": diagnosis,
        "ts_code": item["ts_code"],
        "name": item["name"],
        "industry": item.get("industry"),
        "selection_analysis": selection,
        "meets_selection_conditions": bool(selection.get("eligible")),
        "positive_evidence": list(dict.fromkeys(positive)),
        "unmet_conditions": list(dict.fromkeys(unmet)),
        "evidence_gaps": evidence_gaps,
        "reassessment_conditions": diagnosis["reassessment_conditions"],
        "daily_factor_analysis": item.get("factor") or {},
        "fundamental_analysis": item.get("fundamental") or {},
        "supplemental_diagnostics": item.get("supplemental_diagnostics"),
        "limit_up_pullback_pattern": pattern,
        "late_session_analysis": late,
        "technical_summary": technical,
        "tradability": item.get("tradability"),
        "snapshot": item.get("snapshot"),
        "data_quality": item.get("data_quality"),
        "risks": risks,
        "risk_reference_price": pattern.get("risk_reference_price"),
    }


__all__ = [
    "choushu_liudongxing_houxuan",
    "goujian_houxuan_zhaiyao",
    "goujian_kejiaoyixing_zhaiyao",
    "goujian_kuaizhao_jilu",
    "goujian_zhengju_quekou",
    "goujian_zhen_duan_shixiaoxing",
    "guolv_lishi_wanzhengxing",
    "hebing_jibenmian_zhengju",
    "jichu_ying_guolv",
    "pinggu_shangzhang_tiaojian",
    "paixu_shangzhang_houxuan",
    "shibie_yizijia_zhangting",
    "shishi_ying_guolv",
    "xuyao_shishi_kuaizhao",
    "zhuan_json_zhi",
    "zhuan_you_xian_shuzhi",
]
