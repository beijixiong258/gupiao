"""数据有效性、用户明确条件、原始证据与可解释的表现比较。"""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.fenxi_weipan import WeipanJieduan
from src.ashare.dangu_zhaiyao import goujian_dangu_zhaiyao
from src.ashare.fenxi_wenti import heyan_mingque_tiaojian
from src.ashare.shuju_yuan import heyan_kuaizhao_shidian, huoqu_zhangdieting_guize
from src.ashare.tuijian_gailv import goujian_tongzhuangtai_tongji, jisuan_dangqian_zhuangtai


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
    return [f"已核验行情为{reason}"] if reason else []


def jichu_ying_guolv(
    row: pd.Series,
    *,
    analysis_date: pd.Timestamp,
    config: dict[str, Any],
    quote_is_completed: bool,
) -> list[str]:
    reasons: list[str] = []
    name = str(row.get("name") or "")
    price = zhuan_you_xian_shuzhi(row.get("latest_price"))
    volume = zhuan_you_xian_shuzhi(row.get("volume"))
    amount = zhuan_you_xian_shuzhi(row.get("amount_yuan"))
    if price is None or price <= 0:
        reasons.append("价格字段无效或疑似停牌")
    if quote_is_completed and (volume is None or volume <= 0):
        reasons.append("成交量字段无效或疑似停牌")
    if quote_is_completed and (amount is None or amount <= 0):
        reasons.append("完整日线成交额缺失、无效或当日没有成交")
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


def guolv_lishi_wanzhengxing(
    histories: dict[str, pd.DataFrame],
    *,
    analysis_date: pd.Timestamp,
    minimum_rows: int,
    minimum_amount: float,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    accepted: dict[str, pd.DataFrame] = {}
    rejected: list[dict[str, Any]] = []
    for code, raw in histories.items():
        data = raw.copy() if raw is not None else pd.DataFrame()
        if data.empty or "trade_date" not in data.columns:
            rejected.append({"ts_code": code, "status": "unavailable", "reason": "没有可用历史日线"})
            continue
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
        if data["trade_date"].isna().any() or data["trade_date"].duplicated().any():
            rejected.append({"ts_code": code, "status": "unavailable", "reason": "日线日期无效或重复，不能跳过坏行后比较"})
            continue
        data = data.sort_values("trade_date").reset_index(drop=True)
        if len(data) < minimum_rows:
            rejected.append({"ts_code": code, "status": "unavailable", "reason": f"历史只有 {len(data)} 行，少于 {minimum_rows} 行"})
            continue
        latest_date = pd.Timestamp(data.iloc[-1]["trade_date"]).normalize()
        if latest_date != analysis_date.normalize():
            rejected.append({"ts_code": code, "status": "unavailable", "reason": f"最新日线停留在 {latest_date.strftime('%Y-%m-%d')}"})
            continue
        latest_amount = zhuan_you_xian_shuzhi(data.iloc[-1].get("amount_yuan"))
        if latest_amount is None or latest_amount < 0:
            rejected.append({"ts_code": code, "status": "unavailable", "reason": "最新完整日线成交额缺失或无效"})
            continue
        if latest_amount <= 0 or latest_amount < minimum_amount:
            rejected.append({"ts_code": code, "status": "unmet", "reason": "最新完整日线没有有效成交或未通过明确成交额条件"})
            continue
        latest_volume = zhuan_you_xian_shuzhi(data.iloc[-1].get("volume"))
        if latest_volume is None or latest_volume <= 0:
            rejected.append({"ts_code": code, "status": "unavailable", "reason": "最新完整日线成交量缺失或与正成交额矛盾"})
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
    ranking_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """默认只核验比较基础；方向、波动等限制须由用户明确提出。"""
    conditions: list[dict[str, Any]] = []
    technical = item.get("technical") or {}
    raw = technical.get("raw_indicators") or {}
    conflicts: list[str] = []

    state = _dangqian_zhuangtai(item, ranking_context)
    dimensions = dict(state.get("raw_values") or {})
    state_available = state.get("status") == "success" and state.get("calendar_verified") is True
    conditions.append({
        "key": "comparison_evidence", "label": "当前价量状态具备有效比较基础",
        "status": "met" if state_available else "unavailable",
        "values": {**dimensions, "state": state.get("state"), "calendar_verified": state.get("calendar_verified")},
        "reason": (
            "同一完整交易日的连续价量窗口已按权威日历核验；最近方向、ATR标准化价格位置及成交额变化可计算"
            if state_available else "；".join(state.get("missing_reasons") or ["当前状态或权威连续交易日窗口不可用"])
        ),
    })
    conditions.extend(heyan_mingque_tiaojian(item.get("user_conditions") or [], item))

    technical_status, technical_outcome, technical_reason = _technical_state(item)
    if deep_reviewed:
        # 所有双方均有的共享原始值都核对；可选长窗口缺失不阻断有效的短窗口比较。
        for group in ((item.get("factor") or {}).get("groups") or {}).values():
            for field, value in (group.get("values") or {}).items():
                if field not in raw:
                    continue
                first, reviewed = zhuan_you_xian_shuzhi(value), zhuan_you_xian_shuzhi(raw[field])
                if (first is None) != (reviewed is None) or (first is not None and not math.isclose(first, reviewed, rel_tol=1e-10, abs_tol=1e-12)):
                    conflicts.append(f"{field} 初筛与技术复核的原始值或缺失状态不一致")
        reviewed_histogram = zhuan_you_xian_shuzhi((technical.get("macd") or {}).get("histogram"))
        raw_histogram = zhuan_you_xian_shuzhi(raw.get("macd_hist"))
        close = zhuan_you_xian_shuzhi(raw.get("close"))
        macd_hist = _factor_value(item, "momentum_reversal", "macd_hist_pct")
        if any(value is not None for value in (raw_histogram, reviewed_histogram, macd_hist)):
            if any(value is None for value in (raw_histogram, reviewed_histogram, macd_hist, close)) or close <= 0:
                conflicts.append("已有MACD证据的原始柱值、收盘价或展示值不完整")
            elif not math.isclose(raw_histogram / close, macd_hist, rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(raw_histogram, reviewed_histogram, rel_tol=0, abs_tol=0.000050000001):
                conflicts.append("MACD原始柱、价格归一值与展示值不一致")
        factor_date = pd.to_datetime((item.get("factor") or {}).get("trade_date"), errors="coerce")
        technical_date = pd.to_datetime(technical.get("trade_date"), errors="coerce")
        expected_date = pd.to_datetime((item.get("data_quality") or {}).get("as_of"), errors="coerce")
        if any(pd.isna(value) or value is None for value in (factor_date, technical_date, expected_date)) or not (factor_date == technical_date == expected_date):
            conflicts.append("初筛、技术复核与目标日线日期缺失或不一致")
        if (item.get("factor") or {}).get("ts_code") != item.get("ts_code"):
            conflicts.append("初筛股票身份与复核对象不一致")
        technical_available = technical_status in {"ok", "partial"} and technical_outcome != "program_error" and not conflicts
        conditions.append({
            "key": "technical_review", "label": "技术复核已返回有效证据",
            "status": "met" if technical_available else "unavailable",
            "values": {"status": technical_status, "outcome": technical_outcome, "macd_histogram": raw_histogram,
                       "factor_date": (item.get("factor") or {}).get("trade_date"), "technical_date": technical.get("trade_date"), "conflicts": conflicts},
            "reason": technical_reason or ("技术复核日期、身份和原始数值一致；按未舍入值重新核验条件" if technical_available else "；".join(conflicts) or "技术复核未完成或来源不可用"),
        })
    tradability = item.get("tradability") or {}
    hard_blocks = list(tradability.get("hard_blocks") or [])
    if tradability or deep_reviewed:
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
            reason = (
                "当前行情基础条件检查通过，不保证即时成交" if current_required
                else "最近完整日线的基础成交条件检查通过，未确认当前或下次开市可成交"
            )
        else:
            trade_status = "unavailable"
            reason = "缺少可交易性检查结果"
        conditions.append({"key": "tradability", "label": "通过本次适用的基础成交条件检查", "status": trade_status,
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


_SHOUYI_GUJI_FIELDS = (
    "up_probability", "conditional_upside_mean", "conditional_downside_mean",
    "expected_gross_return", "expected_net_return", "composite_score_bps",
)


def _dangqian_zhuangtai(
    item: dict[str, Any], ranking_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """当前特征与逐股核估计共享本次请求的日历和截止日。"""
    if ranking_context is None and isinstance(item.get("probability_state"), dict):
        return item["probability_state"]
    context = ranking_context if ranking_context is not None else {}
    calendar = context.get("trading_calendar")
    if hasattr(calendar, "open_dates"):
        calendar = calendar.open_dates
    as_of = context.get("as_of") or (item.get("data_quality") or {}).get("as_of")
    history = item.get("history")
    key = (str(item.get("ts_code") or ""), str(as_of), id(history))
    cache = context.setdefault("current_state_cache", {}) if ranking_context is not None else {}
    if key not in cache:
        cache[key] = jisuan_dangqian_zhuangtai(history, trading_calendar=calendar, as_of=as_of)
    return cache[key]


def _tongzhuangtai_tongji(
    pool: list[dict[str, Any]], ranking_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """仅在全部资格核验后，聚合完整合格池；不按最终展示数量截取样本。"""
    if not pool:
        empty = {"candidate_estimates": {}, "state_statistics": {}}
        if ranking_context is not None:
            ranking_context["latest_statistics"] = empty
        return empty
    context = ranking_context if ranking_context is not None else {}
    calendar = context.get("trading_calendar")
    if hasattr(calendar, "open_dates"):
        calendar = calendar.open_dates
    as_of = context.get("as_of") or (pool[0].get("data_quality") or {}).get("as_of")
    histories = {str(item["ts_code"]): item["history"] for item in pool}
    key = (str(as_of), tuple(sorted((code, id(history)) for code, history in histories.items())))
    cache = context.setdefault("statistics_cache", {}) if ranking_context is not None else {}
    if key not in cache:
        cache[key] = goujian_tongzhuangtai_tongji(histories, calendar, as_of)
    if ranking_context is not None:
        ranking_context["latest_statistics"] = cache[key]
    return cache[key]


def paixu_shangzhang_houxuan(
    items: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    deep_reviewed: bool = False,
    ranking_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """近期相似价量样本核加权估计未来五交易日净收益，仅按未舍入期望排序。"""
    prepared: list[dict[str, Any]] = []
    for item in items:
        result = dict(item)
        result["probability_state"] = _dangqian_zhuangtai(item, ranking_context)
        selection = pinggu_shangzhang_tiaojian(result, config=config, deep_reviewed=deep_reviewed,
                                              ranking_context=ranking_context)
        dimensions = dict(result["probability_state"].get("raw_values") or {})
        missing = [field for field, value in dimensions.items() if value is None]
        selection.update({
            "comparison_values": dimensions,
            "comparison_missing_fields": missing,
            "comparison_status": "unavailable" if missing or selection["missing_conditions"] else "pending" if selection["eligible"] else "not_applicable",
            "recommendation_index": None,
            **{field: None for field in _SHOUYI_GUJI_FIELDS},
            "ranking_basis": "先完成用户明确条件、当前状态、日期身份和技术原值及适用成交核验；评分仅用最近120交易日内已到期的五交易日参考事件，20交易日半衰期与逐股连续特征相似度核权重相乘；不共享24状态组分数。有效日期不足10，或删除任意1至2个信号日期后均值变化超过原值绝对值50%时，不发布排序估计。未来5个交易日期望分=10000×预计净收益，100分对应1%；综合上涨机会、幅度、非上涨损失及成本，内部明细不在公开报告列出。仅按未舍入预计净收益降序，同值明确并列，代码只稳定展示。参数固定，不训练或寻优，估计未经未来收益有效性确认。",
            "preference_order": ["expected_net_return_descending", "code_for_exact_ties"],
        })
        result["selection"] = selection
        prepared.append(result)

    pool = [item for item in prepared if item["selection"]["eligible"]]
    statistics = _tongzhuangtai_tongji(pool, ranking_context)
    estimate_map = statistics.get("candidate_estimates") or {}
    for item in pool:
        selection = item["selection"]
        estimate = estimate_map.get(str(item["ts_code"])) or {
            "status": "unavailable", "missing_reasons": ["该对象没有逐股核估计结果"], "statistics": {},
        }
        stats = estimate.get("statistics") or {}
        net = zhuan_you_xian_shuzhi(stats.get("expected_net_return"))
        available = estimate.get("status") == "success" and net is not None
        selection["recommendation_index_analysis"] = {
            **estimate, "current_state": item["probability_state"],
            "statistics_pool_size": len(pool), "score_unit": "basis_points",
            "score_definition": "10000×近期相似价量样本核加权预计五交易日净收益；100分对应1%",
        }
        selection["comparison_pool_size"] = len(pool)
        selection["statistics_pool_size"] = len(pool)
        selection["comparison_status"] = "complete" if available else "unavailable"
        selection["comparison_state_values"] = dict(item["probability_state"].get("raw_values") or {})
        if available:
            selection.update({field: zhuan_you_xian_shuzhi(stats.get(field)) for field in _SHOUYI_GUJI_FIELDS})
            # 兼容既有字段名，但综合分语义由期望净收益明确决定。
            selection["composite_score_bps"] = 10000.0 * net
            selection["recommendation_index"] = selection["composite_score_bps"]
            selection["comparison_values"] = {field: selection[field] for field in _SHOUYI_GUJI_FIELDS}
            selection["comparison_missing_fields"] = []
        else:
            selection["conditions"].append({
                "key": "recommendation_statistics", "label": "逐股核加权五交易日期望分可估计",
                "status": "unavailable", "values": {"statistics_pool_size": len(pool),
                    "known_count": stats.get("known_count"), "known_signal_dates": stats.get("known_signal_dates"),
                    "effective_signal_date_count": stats.get("effective_signal_date_count"),
                    "sample_support": stats.get("sample_support"),
                    "date_stability_check": stats.get("date_stability_check")},
                "reason": "；".join(estimate.get("missing_reasons") or ["局部事件样本或期望净收益不可估计"]),
            })
            selection["missing_conditions"].append("recommendation_statistics")
            selection.update(eligible=False, status="partial", outcome="evidence_unavailable")

    def display_order(item: dict[str, Any]) -> tuple[Any, ...]:
        selection = item["selection"]
        net = selection["expected_net_return"]
        category = 0 if selection["eligible"] and net is not None else 1 if not selection["unmet_conditions"] else 2
        return (category, -net if net is not None else math.inf, str(item.get("ts_code") or ""))

    ordered = sorted(prepared, key=display_order)
    eligible_pool = [item for item in ordered if item["selection"]["eligible"]]
    equivalent_counts = Counter(item["selection"]["expected_net_return"] for item in eligible_pool)
    previous_value, comparison_rank = None, 0
    for index, item in enumerate(eligible_pool):
        selection = item["selection"]
        if selection["expected_net_return"] != previous_value:
            comparison_rank = index + 1
        selection["comparison_rank"] = comparison_rank
        previous_value = selection["expected_net_return"]
        selection["comparison_pool_size"] = len(eligible_pool)
        selection["equivalent_index_count"] = equivalent_counts[selection["expected_net_return"]]
        selection["absolute_performance"] = (
            "未来五交易日净收益点估计为正，尚非收益保证" if selection["expected_net_return"] > 0
            else "未来五交易日净收益点估计非正；排名靠前仍缺少正收益依据"
        )
        other = eligible_pool[index + 1] if index + 1 < len(eligible_pool) else None
        if other is not None:
            right = other["selection"]
            equal = selection["expected_net_return"] == right["expected_net_return"]
            reason = (
                "预计净收益完全相同，明确并列；代码仅用于稳定展示，没有额外优劣次序"
                if equal else "按近期相似价量事件核加权后的未舍入预计净收益取舍，已综合上涨机会、幅度、非上涨损失和成本"
            )
            differences = {
                field + "_difference": selection[field] - right[field]
                if selection[field] is not None and right[field] is not None else None
                for field in _SHOUYI_GUJI_FIELDS
            }
            selection["comparison_to_next"] = {"ts_code": other["ts_code"], "name": other.get("name"), "reason": reason,
                                                "role": "comparison_reference",
                                                "current_values": selection["comparison_values"], "next_values": right["comparison_values"],
                                                "current_index": selection["recommendation_index"], "next_index": right["recommendation_index"],
                                                "index_difference": selection["recommendation_index"] - right["recommendation_index"],
                                                "index_unit": "basis_points", **differences,
                                                "current_state": item["probability_state"].get("state"),
                                                "next_state": other["probability_state"].get("state"),
                                                "raw_advantages": [field for field in ("expected_net_return",)
                                                    if selection[field] is not None and right[field] is not None and selection[field] > right[field]],
                                                "raw_tradeoffs": [field for field in ("expected_net_return",)
                                                    if selection[field] is not None and right[field] is not None and selection[field] < right[field]]}
    return ordered


def goujian_kejiaoyixing_zhaiyao(
    *,
    code: str,
    name: str,
    snapshot: dict[str, Any],
    history: pd.DataFrame,
    minimum_amount: float,
    realtime_required: bool = False,
    reference_time: Any = None,
    market_clock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session = str((market_clock or {}).get("session_status") or "unknown")
    # 时段来自调用方的权威日历；不依据系统星期数猜测交易日。
    if (market_clock or {}).get("is_trading_day") is False or session in {"non_trading_day", "pre_open", "post_close"}:
        realtime_required = False
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
    if amount is None or amount <= 0:
        hard_blocks.append("最新完整日线成交额缺失、无效或没有成交")
    elif amount < minimum_amount:
        cautions.append("最新完整日线成交额低于流动性观察基准；该基准不是默认选股限制")
    completed_volume = zhuan_you_xian_shuzhi(latest.get("volume"))
    if completed_volume is None or completed_volume <= 0:
        hard_blocks.append("最新完整日线成交量缺失、无效或没有成交")
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
    current_volume = zhuan_you_xian_shuzhi(snapshot.get("volume"))
    current_amount = zhuan_you_xian_shuzhi(snapshot.get("amount_yuan"))
    current_turnover_valid = all(value is not None and value > 0 for value in (current_volume, current_amount))
    time_check = heyan_kuaizhao_shidian(
        snapshot, expected_trade_date=reference_time if realtime_required else amount_trade_date,
        reference_time=reference_time, require_timestamp=realtime_required,
    )
    if (market_clock or {}).get("is_trading_day") is False:
        time_check["session_phase"] = "non_trading_day"
    quote_time = pd.Timestamp(time_check["provider_quote_time"]) if time_check["provider_quote_time"] else None
    auction_pending = bool(realtime_required and (
        time_check["session_phase"] == "opening_auction"
        or (quote_time is not None and quote_time < quote_time.normalize() + pd.Timedelta(hours=9, minutes=25))
    ))
    snapshot_verified = bool(
        snapshot.get("status") == "ok"
        and all(value is not None and value > 0 for value in current_fields)
        and current_turnover_valid and not auction_pending
        and time_check["verified"]
    )
    current_verified = bool(realtime_required and snapshot_verified)
    execution_note = (
        "仅核验最近完整日线的基础成交条件；历史快照不代表当前行情或当前可交易，下次交易需重新核验"
        if not realtime_required
        else "仅核验当前行情时点、价格和成交字段；这些基础条件不保证即时成交"
    )
    quote_reason = (
        str(snapshot.get("error") or "实时快照不可用") if snapshot.get("status") != "ok"
        else str(time_check["reason"]) if not time_check["verified"]
        else "开盘集合竞价尚无已核验的撮合成交，快照仅供参考" if auction_pending
        else "实时价格或涨跌停核验字段不完整" if any(value is None or value <= 0 for value in current_fields)
        else "当前成交量或成交额缺失、无效或尚无成交，无法确认当前可交易性" if not current_turnover_valid
        else str(time_check["reason"])
    )
    if snapshot.get("status") == "ok" and not time_check["verified"]:
        cautions.append(str(time_check["reason"]))
    if realtime_required and not current_verified:
        hard_blocks.append("当前行情时点、价格、涨跌停或成交字段未通过核验，无法确认当前可交易性")
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
        "status": "blocked" if hard_blocks else "historical_reference" if not realtime_required else "caution" if cautions else "conditions_verified",
        "basic_execution_feasible": not hard_blocks,
        "execution_check_scope": "current_quote" if realtime_required else "historical_daily",
        "execution_note": execution_note,
        "market_session_status": session,
        "realtime_required": realtime_required,
        "current_quote_verified": current_verified,
        "current_quote_status": "not_required" if not realtime_required else "verified" if current_verified else "unavailable",
        "current_quote_reason": quote_reason if realtime_required else execution_note,
        "historical_quote_verified": bool(not realtime_required and snapshot_verified),
        "quote_time_verification": time_check,
        "current_volume": current_volume,
        "current_amount_yuan": current_amount,
        "current_turnover_requirement": "当前成交量和成交额须为有效正数；全天流动性底线仅用于最近完整日线",
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
    ownership = item.get("ownership_analysis") or {}
    ownership_gap_count = 0
    for index, gap in enumerate(ownership.get("evidence_gaps") or [], start=1):
        if isinstance(gap, dict):
            gaps.append({
                "component": "持仓分析（通达信控盘指标）",
                "impact": "该项证据尚不支持完整持仓判断；已取得控盘原值仍保留",
                "reassessment_condition": "补齐并核验该项控盘证据后复评",
                **gap, "code": str(gap.get("code") or f"ownership_evidence_{index}"),
                "blocks_selection": False,
            })
        elif isinstance(gap, str) and gap.strip():
            append(f"ownership_evidence_{index}", "持仓分析（通达信控盘指标）", gap,
                   "补齐并核验该项控盘证据后复评")
        else:
            continue
        ownership_gap_count += 1
    if ownership.get("status") in {"partial", "unavailable", "unsupported"} and not ownership_gap_count:
        append("ownership_evidence_missing", "持仓分析（通达信控盘指标）",
               str(ownership.get("missing_reason") or "控盘数据或解释依据尚未完整核验"),
               "重新取得并核验控盘序列、指标口径及实际更新时间后复评")
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
        "realtime_status": realtime_status if realtime_required else "not_required",
        "explanation": explanation,
        "reassess_when": ["出现新的完整日线后", "价格、成交状态或已列风险发生变化时", "缺失证据补齐后"],
    }


def goujian_houxuan_zhaiyao(
    item: dict[str, Any],
    *,
    rank: int,
) -> dict[str, Any]:
    """投影完整诊断、期望净收益和显式条件，内部概率与幅度保留供审计。"""
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
        *[str(value) for value in (item.get("tradability") or {}).get("hard_blocks", [])],
        *[str(value) for value in (item.get("tradability") or {}).get("cautions", [])],
        *[str(value) for value in item.get("risks") or []],
    ]))
    name = str(item.get("name") or "")
    if "ST" in name.upper():
        risks.append("本次股票简称含ST标记；保留事实，不作为未提出的选股排除条件")
    if "退" in name:
        risks.append("本次股票简称含退市标记，须结合有来源的退市安排解释")
    diagnosis = goujian_dangu_zhaiyao(
        technical=technical, fundamentals=item.get("fundamental") or {}, risks=risks,
        evidence_gaps=evidence_gaps, as_of=str((item.get("data_quality") or {}).get("as_of") or technical.get("trade_date") or "未取得日期"),
        factor_analysis=item.get("factor"), pattern=pattern, late=late,
        supplemental=item.get("supplemental_diagnostics"),
        snapshot=item.get("snapshot"), tradability=item.get("tradability"),
        ownership_analysis=item.get("ownership_analysis"),
        research_assessment=item.get("research_assessment"),
    )
    return {
        "rank": selection.get("comparison_rank", rank),
        "recommendation_index": selection.get("recommendation_index"),
        "recommendation_index_semantics": "composite_score_bps_expected_net_return",
        **{field: selection.get(field) for field in _SHOUYI_GUJI_FIELDS},
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
        "ownership_analysis": item.get("ownership_analysis"),
        "research_assessment": item.get("research_assessment"),
        "limit_up_pullback_pattern": pattern,
        "late_session_analysis": late,
        "technical_summary": technical,
        "tradability": item.get("tradability"),
        "snapshot": item.get("snapshot"),
        "data_quality": item.get("data_quality"),
        "risks": risks,
        "risk_reference_price": pattern.get("risk_reference_price") if pattern.get("eligible") and pattern.get("state") in {"waiting_breakout", "extreme_shrink", "adjusting", "intraday_confirmed", "close_confirmed"} else None,
    }


__all__ = [
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
