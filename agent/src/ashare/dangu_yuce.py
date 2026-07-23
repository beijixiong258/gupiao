"""Horizon-aware quantitative research for one mainland China A-share.

The single-stock workflow deliberately trains on a liquid peer panel instead
of fitting a tiny model to one stock's own history.  Signals are created after
the latest completed daily close; entry is the next market-session open and
T+1/T+2/T+3 are the first, second and third sellable closes after entry.
"""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from src.ashare.bankuai_yuce import (
    _blend_component_predictions,
    _calibrate_direction_probability,
    _daily_rank_ic,
    _experiment_fingerprint,
    _fit_direction_probabilities,
    _fit_model_components,
    _fit_quantile_model_components,
    _fold_stability,
    _quality_label,
    _quality_score,
    _regime_stability,
    _rolling_conformal_interval,
    _rolling_cqr_interval,
    _select_stable_features,
    _select_ensemble_weight,
    _top_n_validation_metrics,
    goujian_moxing_shuju,
)
from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    _akshare_name_table,
    _completed_market_history,
    _json_value,
    _round_optional,
    _stock_basic_cache,
    akshare_zhilian,
    biaozhunhua_daima,
    huoqu_rili_xingqing,
    shi_a_gu,
)
from src.ashare.jiaoyi_zhixing import (
    _apply_cost,
    _load_cost_assumption,
    _price_limit_bounds,
    _round_price_tick,
    _stock_roundtrip_cost,
)
from src.ashare.moxing_pinggu import regression_baseline_metrics, signal_evidence_gate
from src.ashare.shuju_yuan import _load_or_fetch_stock_basic, _price_limit_rule, _tushare_pro
from src.ashare.riping_yinzi import DAILY_FACTOR_FEATURE_COLUMNS, enrich_daily_factor_panel


SINGLE_STOCK_EXTRA_FEATURES = [
    "gap_open",
    "intraday_return",
    "close_location",
    "log_amount_yuan",
    "amount_ratio_5_20",
    "peer_mean_ret_1",
    "peer_mean_ret_5",
    "peer_mean_ret_20",
    "excess_ret_1",
    "excess_ret_5",
    "excess_ret_20",
    "peer_breadth_above_ma20",
    "peer_breadth_positive_5d",
    "peer_dispersion_ret_5",
    "rank_ret_5",
    "rank_ma_gap_20",
    "rank_volume_ratio_5_20",
    "rank_volatility_20",
    "rank_log_amount",
]
SINGLE_STOCK_FEATURE_COLUMNS = FEATURE_COLUMNS + SINGLE_STOCK_EXTRA_FEATURES + DAILY_FACTOR_FEATURE_COLUMNS


def shichang_shizhong(reference: datetime | None = None) -> dict[str, Any]:
    """Describe the local A-share session without pretending to know holidays."""
    current = reference or datetime.now()
    minute = current.hour * 60 + current.minute
    if current.weekday() >= 5:
        status = "non_trading_day"
        note = "周末；法定节假日仍以交易所日历为准"
    elif minute < 9 * 60 + 15:
        status = "pre_market"
        note = "开盘前，只使用最近完整收盘日生成信号"
    elif minute < 9 * 60 + 30:
        status = "opening_auction"
        note = "集合竞价阶段，实时快照仅作可交易性参考"
    elif minute < 11 * 60 + 30:
        status = "trading"
        note = "盘中，模型仍只使用最近完整收盘日，实时价格仅作执行检查"
    elif minute < 13 * 60:
        status = "midday_break"
        note = "午间休市，模型仍只使用最近完整收盘日"
    elif minute < 15 * 60:
        status = "trading"
        note = "盘中，模型仍只使用最近完整收盘日，实时价格仅作执行检查"
    elif minute < 15 * 60 + 5:
        status = "close_pending"
        note = "刚收盘，今日完整日线可能尚未落库，暂不把盘中快照当收盘信号"
    else:
        status = "post_close"
        note = "收盘后；模型使用数据源已确认完成的最新日线"
    return {
        "captured_at": current.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Shanghai",
        "session_status": status,
        "calendar_precision": "工作日和本地时钟判断；具体休市日由后续交易日历确认",
        "analysis_basis": note,
    }


def huoqu_dangqian_kuaizhao(code: str, reference: datetime | None = None) -> dict[str, Any]:
    """Fetch a best-effort current quote used only for execution checks."""
    captured = reference or datetime.now()
    digits = str(code).split(".")[0]
    try:
        import akshare as ak

        with akshare_zhilian():
            table = ak.stock_zh_a_spot_em()
        if table is None or table.empty or "代码" not in table.columns:
            raise RuntimeError("实时行情返回空表或缺少代码列")
        codes = table["代码"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        hit = table[codes == digits]
        if hit.empty:
            raise RuntimeError(f"实时行情未找到 {digits}")
        row = hit.iloc[0]
        return {
            "status": "ok",
            "source": "akshare_eastmoney_spot",
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
            "provider_trade_date": None,
            "timeliness": "接口不返回逐行交易日期；盘中只作尽力而为的执行快照，不参与模型训练",
            "name": _json_value(row.get("名称")),
            "last_price": _round_optional(row.get("最新价"), 3),
            "open": _round_optional(row.get("今开"), 3),
            "high": _round_optional(row.get("最高"), 3),
            "low": _round_optional(row.get("最低"), 3),
            "previous_close": _round_optional(row.get("昨收"), 3),
            "pct_change": _round_optional(row.get("涨跌幅"), 3),
            "volume": _round_optional(row.get("成交量"), 2),
            "amount_yuan": _round_optional(row.get("成交额"), 2),
            "turnover_rate_pct": _round_optional(row.get("换手率"), 3),
            "volume_ratio": _round_optional(row.get("量比"), 3),
            "pe_dynamic": _round_optional(row.get("市盈率-动态"), 3),
            "pb": _round_optional(row.get("市净率"), 3),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "akshare_eastmoney_spot",
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(exc),
            "note": "实时快照不可用时，分析退回最近完整日线，不伪装成实时价格",
        }


def pinggu_kejiaoyixing(
    *,
    code: str,
    name: str,
    profile: dict[str, Any],
    history: pd.DataFrame,
    freshness: dict[str, Any],
    current_quote: dict[str, Any],
    config: dict[str, Any],
    reference: datetime | None = None,
) -> dict[str, Any]:
    """Assess A-share execution constraints used as evidence in the analysis."""
    current = reference or datetime.now()
    clock = shichang_shizhong(current)
    settings = config.get("dangu", {})
    minimum_amount = float(settings.get("min_amount_yuan", 30_000_000))
    hard_blocks: list[str] = []
    cautions: list[str] = []
    upper_name = str(name).strip().upper()

    if "退" in upper_name:
        hard_blocks.append("股票简称包含退市风险标记")
    if "ST" in upper_name:
        hard_blocks.append("股票简称包含 ST 风险标记，短期模型证据可靠性受限")
    if upper_name.startswith(("N", "C")):
        hard_blocks.append("新股无涨跌幅阶段缺少稳定可比样本")

    list_date_raw = profile.get("list_date") or profile.get("上市时间")
    list_date = pd.to_datetime(str(list_date_raw), errors="coerce") if list_date_raw else pd.NaT
    listing_days = None
    if pd.notna(list_date):
        listing_days = int((pd.Timestamp(current.date()) - pd.Timestamp(list_date).normalize()).days)
        if listing_days < int(settings.get("min_listing_calendar_days", 180)):
            hard_blocks.append(f"上市仅约 {listing_days} 个自然日，历史样本不足")

    latest = history.sort_values("trade_date").iloc[-1]
    previous = history.sort_values("trade_date").iloc[-2] if len(history) >= 2 else latest
    quote_ok = current_quote.get("status") == "ok"
    last_price = _round_optional(current_quote.get("last_price"), 3) if quote_ok else None
    high = _round_optional(current_quote.get("high"), 3) if quote_ok else None
    low = _round_optional(current_quote.get("low"), 3) if quote_ok else None
    previous_close = _round_optional(current_quote.get("previous_close"), 3) if quote_ok else None
    snapshot_amount_yuan = _round_optional(current_quote.get("amount_yuan"), 2) if quote_ok else None
    completed_amount_yuan = _round_optional(latest.get("amount_yuan"), 2)
    if last_price is None:
        last_price = _round_optional(latest.get("close"), 3)
        high = _round_optional(latest.get("high"), 3)
        low = _round_optional(latest.get("low"), 3)
        previous_close = _round_optional(previous.get("close"), 3)
        price_basis = "latest_completed_qfq_close"
    else:
        price_basis = "best_effort_current_snapshot"
    amount_yuan = completed_amount_yuan if completed_amount_yuan is not None else snapshot_amount_yuan
    amount_basis = "latest_completed_daily_bar" if completed_amount_yuan is not None else "best_effort_current_snapshot"

    if freshness.get("status") == "too_stale":
        hard_blocks.append("完整日线已严重滞后，可能停牌或数据源异常")
    elif freshness.get("status") == "possibly_stale":
        cautions.append("完整日线可能滞后，需人工核对是否停牌或长假")

    active_session = clock["session_status"] in {"opening_auction", "trading", "midday_break"}
    if active_session and quote_ok and (last_price is None or last_price <= 0):
        hard_blocks.append("交易时段没有有效最新价，疑似停牌或行情异常")
    if active_session and quote_ok and snapshot_amount_yuan is not None and snapshot_amount_yuan <= 0:
        hard_blocks.append("交易时段成交额为零，无法确认可正常成交")
    if amount_yuan is None:
        cautions.append("缺少成交额，无法完整验证流动性")
    elif amount_yuan < minimum_amount:
        hard_blocks.append(
            f"最近可用成交额约 {amount_yuan:,.0f} 元，低于单股研究流动性门槛 {minimum_amount:,.0f} 元"
        )

    price_rule = _price_limit_rule(code, name, price_limit_exempt=upper_name.startswith(("N", "C")))
    limit_up = None
    limit_down = None
    one_price_limit_up = False
    near_limit_up = False
    if previous_close is not None and price_rule.limit_rate is not None:
        limit_down, limit_up = _price_limit_bounds(previous_close, price_rule.limit_rate, 1)
        if high is not None and low is not None and last_price is not None:
            one_price_limit_up = (
                low >= limit_up - 0.005 and high <= limit_up + 0.005 and last_price >= limit_up - 0.005
            )
            near_limit_up = last_price >= limit_up - 0.015
    if one_price_limit_up:
        hard_blocks.append("当前为一字或近似一字涨停，不能假设能够买入")
    elif near_limit_up:
        cautions.append("价格接近涨停，追价后的成交和次日收益假设不可靠")

    signal_date = pd.to_datetime(latest.get("trade_date"), errors="coerce")
    expected_date = pd.to_datetime(freshness.get("expected_latest_date"), errors="coerce")
    session_status = str(clock.get("session_status"))
    model_entry_timing_valid = session_status in {
        "pre_market",
        "opening_auction",
        "post_close",
        "non_trading_day",
    }
    entry_timing_reason = "下一交易日开盘尚未发生，模型情景的入口时点仍有效"
    if session_status in {"trading", "midday_break", "close_pending"}:
        model_entry_timing_valid = False
        entry_timing_reason = "最近完整收盘信号对应的下一交易日开盘已经过去，当前盘中价不属于模型入口口径"
    elif session_status == "post_close" and pd.notna(signal_date) and pd.notna(expected_date):
        if pd.Timestamp(signal_date).normalize() < pd.Timestamp(expected_date).normalize():
            model_entry_timing_valid = False
            entry_timing_reason = "收盘后行情源尚未提供今日完整日线，旧信号的计划开盘入口已经过去"
    if not model_entry_timing_valid:
        cautions.append(entry_timing_reason + "；需要在最新完整收盘日后重新分析")

    status = "blocked" if hard_blocks else ("caution" if cautions else "tradable")
    return {
        "status": status,
        "basic_execution_feasible": not hard_blocks,
        "analysis_price": last_price,
        "analysis_price_basis": price_basis,
        "amount_yuan": amount_yuan,
        "amount_basis": amount_basis,
        "current_snapshot_amount_yuan": snapshot_amount_yuan,
        "minimum_amount_yuan": minimum_amount,
        "listing_calendar_days": listing_days,
        "price_limit_status": price_rule.status,
        "price_limit_pct": round(price_rule.limit_rate * 100.0, 2) if price_rule.limit_rate is not None else None,
        "limit_up_price": limit_up,
        "limit_down_price": limit_down,
        "one_price_limit_up": one_price_limit_up,
        "near_limit_up": near_limit_up,
        "model_entry_timing_valid": model_entry_timing_valid,
        "model_entry_timing_reason": entry_timing_reason,
        "hard_blocks": hard_blocks,
        "cautions": cautions,
        "market_clock": clock,
    }


def _normalise_universe_codes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data = data[data["ts_code"].map(shi_a_gu)].copy()
    data["ts_code"] = data["ts_code"].map(biaozhunhua_daima)
    return data.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def _latest_tushare_cross_section(
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    pro = _tushare_pro()
    warnings: list[str] = []
    quality: dict[str, Any] = {}
    stock_master_meta: dict[str, Any] = {"status": "live_current_snapshot"}
    try:
        from src.ashare.riping_cangku import load_stock_snapshot_asof

        basic, stock_master_meta = load_stock_snapshot_asof(signal_date.strftime("%Y%m%d"))
        if (
            not basic.empty
            and int(stock_master_meta.get("snapshot_age_calendar_days", 0)) > 45
        ):
            warnings.append(
                f"本地股票资料PIT快照已距分析日 {stock_master_meta.get('snapshot_age_calendar_days')} 天，"
                "改用当前接口并明确保留当前行业标签偏差"
            )
            basic = pd.DataFrame()
    except Exception as exc:
        basic = pd.DataFrame()
        stock_master_meta = {"status": "warehouse_snapshot_failed", "error": str(exc)}
    if basic.empty:
        basic = _load_or_fetch_stock_basic(pro, quality)
        warnings.extend(str(value) for value in quality.get("warnings", []))
        stock_master_meta = {
            "status": "live_current_snapshot",
            "source": str(quality.get("stock_basic", {}).get("source") or "tushare_stock_basic"),
            "known_bias": "没有不晚于分析日的本地股票资料快照，行业标签来自当前接口",
        }
    else:
        warnings.append(
            f"同行股票资料使用 {stock_master_meta.get('snapshot_date')} 的本地PIT快照"
        )
    daily = pd.DataFrame()
    daily_date = ""
    for offset in range(12):
        candidate = signal_date - timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue
        candidate_text = candidate.strftime("%Y%m%d")
        raw = pro.daily(trade_date=candidate_text)
        if raw is not None and not raw.empty:
            daily = raw.copy()
            daily_date = candidate_text
            break
    if daily.empty:
        raise RuntimeError("Tushare 未返回信号日前的全市场日行情")
    daily["amount_yuan"] = pd.to_numeric(daily.get("amount"), errors="coerce") * 1000.0
    daily["latest_price"] = pd.to_numeric(daily.get("close"), errors="coerce")
    daily["pct_chg"] = pd.to_numeric(daily.get("pct_chg"), errors="coerce")
    daily = daily[[column for column in ["ts_code", "latest_price", "pct_chg", "amount_yuan"] if column in daily.columns]]

    try:
        daily_basic = pro.daily_basic(
            trade_date=daily_date,
            fields="ts_code,trade_date,turnover_rate,pe_ttm,pb,total_mv,circ_mv",
        )
        if daily_basic is not None and not daily_basic.empty:
            for column in ["turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv"]:
                daily_basic[column] = pd.to_numeric(daily_basic.get(column), errors="coerce")
            daily_basic["total_market_value_yuan"] = daily_basic["total_mv"] * 10000.0
            daily_basic["circulating_market_value_yuan"] = daily_basic["circ_mv"] * 10000.0
            keep = [
                "ts_code",
                "turnover_rate",
                "pe_ttm",
                "pb",
                "total_market_value_yuan",
                "circulating_market_value_yuan",
            ]
            daily = daily.merge(daily_basic[keep], on="ts_code", how="left")
    except Exception as exc:
        warnings.append(f"同行估值横截面不可用：{exc}")

    basic = _normalise_universe_codes(basic)
    daily = _normalise_universe_codes(daily)
    data = basic.merge(daily, on="ts_code", how="left")
    return data, pd.Timestamp(daily_date).strftime("%Y-%m-%d"), warnings, stock_master_meta


def _latest_akshare_cross_section(
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    import akshare as ak

    with akshare_zhilian():
        spot = ak.stock_zh_a_spot_em()
    if spot is None or spot.empty:
        raise RuntimeError("AKShare 全市场快照为空")
    rename = {
        "代码": "ts_code",
        "名称": "name",
        "最新价": "latest_price",
        "涨跌幅": "pct_chg",
        "成交额": "amount_yuan",
        "换手率": "turnover_rate",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "total_market_value_yuan",
        "流通市值": "circulating_market_value_yuan",
    }
    data = spot.rename(columns={key: value for key, value in rename.items() if key in spot.columns}).copy()
    data["ts_code"] = data["ts_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    data = _normalise_universe_codes(data)
    for column in [
        "latest_price",
        "pct_chg",
        "amount_yuan",
        "turnover_rate",
        "pe_ttm",
        "pb",
        "total_market_value_yuan",
        "circulating_market_value_yuan",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    basic = _stock_basic_cache()
    if basic.empty:
        try:
            basic = _akshare_name_table()
        except Exception:
            basic = pd.DataFrame()
    if not basic.empty:
        basic = _normalise_universe_codes(basic)
        supplement = [column for column in ["ts_code", "name", "industry", "market", "list_date"] if column in basic.columns]
        data = data.merge(basic[supplement], on="ts_code", how="left", suffixes=("", "_basic"))
        if "name_basic" in data.columns:
            data["name"] = data.get("name").fillna(data["name_basic"])
            data = data.drop(columns=["name_basic"])
    return (
        data,
        signal_date.strftime("%Y-%m-%d"),
        ["同行池降级为 AKShare 当前快照，横截面日期由分析信号日近似"],
        {
            "status": "live_current_snapshot",
            "source": "akshare_current_snapshot",
            "known_bias": "AKShare 快照没有历史行业成员时点，不能用于回填过去行业标签",
        },
    )


def xuanze_tonghang_yangben(
    *,
    code: str,
    name: str,
    industry: str,
    signal_date: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a liquid current peer universe, preferring the same industry."""
    settings = config.get("dangu", {})
    base_maximum = max(8, int(settings.get("max_peer_stocks", 20)))
    maximum = base_maximum
    base_same_industry = int(settings.get("same_industry_stocks", 16))
    min_amount = float(settings.get("min_amount_yuan", 30_000_000))
    signal = pd.Timestamp(signal_date).normalize()
    warehouse_range: dict[str, Any] = {
        "status": "not_checked",
        "ready": False,
        "coverage": 0.0,
    }
    try:
        from src.ashare.riping_cangku import warehouse_range_coverage

        history_days = int(settings.get("history_calendar_days", 1440))
        warehouse_range = warehouse_range_coverage(
            start_date=(signal - timedelta(days=history_days)).strftime("%Y%m%d"),
            end_date=signal.strftime("%Y%m%d"),
        )
        if warehouse_range.get("ready"):
            maximum = max(
                maximum,
                int(settings.get("warehouse_max_peer_stocks", 60)),
            )
            base_same_industry = int(
                settings.get("warehouse_same_industry_stocks", max(16, maximum * 3 // 4))
            )
    except Exception as exc:
        warehouse_range = {
            "status": "check_failed",
            "ready": False,
            "coverage": 0.0,
            "error": str(exc),
        }
    same_industry_limit = max(4, min(maximum - 1, base_same_industry))
    warnings: list[str] = []
    try:
        universe, as_of, source_warnings, stock_master_meta = _latest_tushare_cross_section(signal)
        source = "tushare"
        warnings.extend(source_warnings)
    except Exception as tushare_exc:
        warnings.append(f"Tushare 同行池失败：{tushare_exc}")
        universe, as_of, source_warnings, stock_master_meta = _latest_akshare_cross_section(signal)
        source = "akshare"
        warnings.extend(source_warnings)

    if universe.empty:
        raise RuntimeError("没有可用的 A 股同行横截面")
    if "name" not in universe.columns:
        universe["name"] = ""
    if "industry" not in universe.columns:
        universe["industry"] = ""
    for column in ["latest_price", "amount_yuan"]:
        if column not in universe.columns:
            universe[column] = np.nan
        universe[column] = pd.to_numeric(universe[column], errors="coerce")
    universe["name"] = universe["name"].fillna("").astype(str)
    universe["industry"] = universe["industry"].fillna("").astype(str)

    target_hit = universe[universe["ts_code"] == code]
    resolved_industry = str(industry or "").strip()
    if not resolved_industry and not target_hit.empty:
        resolved_industry = str(target_hit.iloc[0].get("industry") or "").strip()
    if not name and not target_hit.empty:
        name = str(target_hit.iloc[0].get("name") or "")

    eligible = universe[
        universe["latest_price"].between(2.0, 300.0, inclusive="both")
        & universe["amount_yuan"].ge(min_amount)
    ].copy()
    bad_name = eligible["name"].str.upper().str.contains(r"ST|退", regex=True, na=False)
    new_name = eligible["name"].str.upper().str.startswith(("N", "C"), na=False)
    eligible = eligible[~bad_name & ~new_name]
    if "list_date" in eligible.columns:
        listed = pd.to_datetime(eligible["list_date"].astype(str), errors="coerce")
        eligible = eligible[listed.isna() | ((signal - listed.dt.normalize()).dt.days >= 180)]
    eligible = eligible.sort_values("amount_yuan", ascending=False, na_position="last")

    selected_parts: list[pd.DataFrame] = []
    if resolved_industry:
        industry_rows = eligible[(eligible["industry"] == resolved_industry) & (eligible["ts_code"] != code)]
        selected_parts.append(industry_rows.head(same_industry_limit).assign(peer_role="same_industry"))
    selected_codes = {code}
    for part in selected_parts:
        selected_codes.update(part["ts_code"].astype(str))
    room = maximum - 1 - sum(len(part) for part in selected_parts)
    if room > 0:
        references = eligible[~eligible["ts_code"].isin(selected_codes)].head(room)
        selected_parts.append(references.assign(peer_role="market_reference"))

    if not target_hit.empty:
        target_row = target_hit.head(1).copy()
    else:
        target_row = pd.DataFrame([{"ts_code": code, "name": name, "industry": resolved_industry}])
    target_row["peer_role"] = "target"
    selected = pd.concat([target_row] + selected_parts, ignore_index=True, sort=False)
    selected = selected.drop_duplicates("ts_code", keep="first").head(maximum).reset_index(drop=True)
    selected["name"] = selected["name"].fillna("").astype(str)
    selected.loc[selected["ts_code"] == code, "name"] = name or selected.loc[selected["ts_code"] == code, "name"]
    role_counts = Counter(selected["peer_role"].astype(str))
    return selected, {
        "source": source,
        "as_of": as_of,
        "target_industry": resolved_industry or None,
        "selected_stocks": int(len(selected)),
        "role_counts": dict(role_counts),
        "minimum_amount_yuan": min_amount,
        "configured_peer_limit_without_warehouse": base_maximum,
        "applied_peer_limit": maximum,
        "warehouse_range": warehouse_range,
        "warehouse_expansion_applied": bool(warehouse_range.get("ready") and maximum > base_maximum),
        "stock_master_snapshot": stock_master_meta,
        "selection_method": "当前同行优先，再用全市场高流动性股票补足；目标股票始终保留",
        "known_bias": (
            "优先使用不晚于分析日的本地股票资料快照；若快照不可用则退回当前行业标签。"
            "即使有快照，仓库建立前的行业变化仍无法倒推"
        ),
        "warnings": warnings,
    }


def _peer_snapshot(peer_table: pd.DataFrame, code: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target = peer_table[peer_table["ts_code"] == code]
    if target.empty:
        return result
    target_row = target.iloc[0]
    mappings = {
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "turnover_rate": "turnover_rate_pct",
        "amount_yuan": "amount_yuan",
        "circulating_market_value_yuan": "circulating_market_value_yuan",
    }
    for source_column, output_column in mappings.items():
        if source_column not in peer_table.columns:
            continue
        values = pd.to_numeric(peer_table[source_column], errors="coerce")
        value = _round_optional(target_row.get(source_column), 4)
        valid = values.dropna()
        if value is None or len(valid) < 5:
            result[output_column] = {"value": value, "peer_percentile": None}
            continue
        if source_column == "pe_ttm":
            valid = valid[valid > 0]
            if value <= 0 or len(valid) < 5:
                result[output_column] = {"value": value, "peer_percentile": None}
                continue
        percentile = float((valid <= value).mean())
        result[output_column] = {
            "value": value,
            "peer_percentile": round(percentile, 4),
            "peer_count": int(len(valid)),
        }
    return result


def _fetch_peer_histories(
    *,
    peer_table: pd.DataFrame,
    target_code: str,
    target_history: pd.DataFrame,
    target_source: str,
    target_adjustment: str,
    signal_date: str,
    source: str,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    settings = config.get("dangu", {})
    minimum_rows = int(settings.get("minimum_history_rows", 180))
    pause_seconds = float(settings.get("request_pause_seconds", 0.08))
    signal = pd.Timestamp(signal_date).normalize()
    start = pd.to_datetime(target_history["trade_date"], errors="coerce").min()
    if pd.isna(start):
        raise RuntimeError("目标股票历史行情缺少有效日期")
    histories: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    sources: Counter[str] = Counter()
    warehouse_histories: dict[str, pd.DataFrame] = {}
    warehouse_batch: dict[str, Any] = {}
    try:
        from src.ashare.riping_cangku import load_qfq_histories_from_warehouse

        warehouse_codes = [
            str(value)
            for value in peer_table["ts_code"].astype(str)
            if str(value) != target_code
        ]
        warehouse_histories, warehouse_batch = load_qfq_histories_from_warehouse(
            warehouse_codes,
            start_date=pd.Timestamp(start).strftime("%Y%m%d"),
            end_date=signal.strftime("%Y%m%d"),
            minimum_rows=minimum_rows,
        )
        if warehouse_histories:
            warnings.append(
                f"本地仓库一次批量读取 {len(warehouse_histories)} 只同行，剩余同行才访问外部行情源"
            )
    except Exception as exc:
        warnings.append(f"本地仓库批量读取不可用，改为逐只读取：{exc}")

    for _, row in peer_table.iterrows():
        peer_code = str(row["ts_code"])
        peer_name = str(row.get("name") or "")
        if peer_code == target_code:
            data = target_history.copy()
            data_source = target_source
            adjustment = target_adjustment
            source_warnings: list[str] = []
            source_errors: list[str] = []
        elif peer_code in warehouse_histories:
            warehouse_data = warehouse_histories[peer_code]
            adjustment = str(warehouse_data.attrs.get("adjustment") or "qfq_by_warehouse")
            data = warehouse_data.copy()
            data_source = str(warehouse_batch.get("source") or "tushare_daily_warehouse_batch")
            source_warnings = []
            source_errors = []
        else:
            fetched = huoqu_rili_xingqing(
                peer_code,
                start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                end_date=signal.strftime("%Y%m%d"),
                source=source,
                use_cache=True,
            )
            if source == "auto" and fetched.adjustment == "raw_unadjusted":
                ak_fallback = huoqu_rili_xingqing(
                    peer_code,
                    start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                    end_date=signal.strftime("%Y%m%d"),
                    source="akshare",
                    use_cache=True,
                )
                if not ak_fallback.data.empty and ak_fallback.adjustment != "raw_unadjusted":
                    fetched = ak_fallback
                    warnings.append(f"{peer_code}: 已单独改用 AKShare 前复权行情")
            data, completion_warnings = _completed_market_history(fetched.data)
            data_source = fetched.source
            adjustment = fetched.adjustment
            source_warnings = list(fetched.warnings) + completion_warnings
            source_errors = list(fetched.errors)
        data = data[pd.to_datetime(data.get("trade_date"), errors="coerce").dt.normalize() <= signal].copy()
        if data.empty:
            errors.append(f"{peer_code} {peer_name}: 没有可用完整日线；{'；'.join(source_errors)}")
        elif adjustment == "raw_unadjusted":
            errors.append(f"{peer_code} {peer_name}: 未复权行情不进入单股预测模型")
        elif len(data) < minimum_rows:
            errors.append(f"{peer_code} {peer_name}: 有效日线 {len(data)} 行，少于 {minimum_rows}")
        else:
            data["data_source"] = data_source
            data["adjustment"] = adjustment
            data["peer_role"] = str(row.get("peer_role") or "market_reference")
            histories[peer_code] = data
            names[peer_code] = peer_name
            sources[str(data_source)] += 1
            warnings.extend(f"{peer_code}: {value}" for value in source_warnings)
        if (
            peer_code != target_code
            and pause_seconds > 0
            and "warehouse" not in str(data_source)
        ):
            time.sleep(pause_seconds)
    return histories, names, {
        "usable_stocks": int(len(histories)),
        "history_sources": dict(sources),
        "minimum_history_rows": minimum_rows,
        "errors": errors,
        "warnings": warnings,
    }


def _add_single_stock_features(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = data.groupby("ts_code", group_keys=False)
    close = pd.to_numeric(data["close"], errors="coerce")
    open_price = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    previous_close = grouped["close"].shift(1)
    data["gap_open"] = open_price / pd.to_numeric(previous_close, errors="coerce") - 1.0
    data["intraday_return"] = close / open_price.replace(0, np.nan) - 1.0
    daily_range = (high - low).replace(0, np.nan)
    data["close_location"] = ((close - low) / daily_range).clip(0.0, 1.0).fillna(0.5)
    if "amount_yuan" not in data.columns:
        data["amount_yuan"] = np.nan
    amount = pd.to_numeric(data["amount_yuan"], errors="coerce")
    data["log_amount_yuan"] = np.log1p(amount.clip(lower=0))
    data["amount_mean_5"] = data.groupby("ts_code")["amount_yuan"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(5, min_periods=5).mean()
    )
    data["amount_mean_20"] = data.groupby("ts_code")["amount_yuan"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=20).mean()
    )
    data["amount_ratio_5_20"] = data["amount_mean_5"] / data["amount_mean_20"].replace(0, np.nan)

    by_date = data.groupby("trade_date", group_keys=False)
    for period in [1, 5, 20]:
        data[f"peer_mean_ret_{period}"] = by_date[f"ret_{period}"].transform("mean")
        data[f"excess_ret_{period}"] = data[f"ret_{period}"] - data[f"peer_mean_ret_{period}"]
    data["peer_breadth_above_ma20"] = by_date["ma_gap_20"].transform(
        lambda values: values.gt(0).where(values.notna()).mean()
    )
    data["peer_breadth_positive_5d"] = by_date["ret_5"].transform(
        lambda values: values.gt(0).where(values.notna()).mean()
    )
    data["peer_dispersion_ret_5"] = by_date["ret_5"].transform("std")
    ranks = {
        "ret_5": "rank_ret_5",
        "ma_gap_20": "rank_ma_gap_20",
        "volume_ratio_5_20": "rank_volume_ratio_5_20",
        "volatility_20": "rank_volatility_20",
        "log_amount_yuan": "rank_log_amount",
    }
    for source_column, output_column in ranks.items():
        data[output_column] = by_date[source_column].transform(lambda values: values.rank(pct=True))
    return data.replace([np.inf, -np.inf], np.nan)


def _walk_forward_boundaries(dates: list[pd.Timestamp], config: dict[str, Any]) -> list[tuple[pd.Timestamp, pd.Timestamp | None]]:
    settings = config.get("dangu", {})
    folds = max(2, int(settings.get("walk_forward_folds", 6)))
    requested_window = max(20, int(settings.get("validation_window_days", 45)))
    minimum_training_dates = max(80, int(settings.get("minimum_training_dates", 120)))
    available = len(dates) - minimum_training_dates
    window = min(requested_window, available // folds if folds else 0)
    if window < 20:
        return []
    first_start = len(dates) - folds * window
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp | None]] = []
    for index in range(folds):
        start_index = first_start + index * window
        end_index = start_index + window
        validation_start = pd.Timestamp(dates[start_index])
        validation_end = pd.Timestamp(dates[end_index]) if end_index < len(dates) else None
        boundaries.append((validation_start, validation_end))
    return boundaries


def _fit_single_stock_models(
    *,
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    config: dict[str, Any],
    budget_yuan: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = panel.copy()
    latest = latest.copy()
    for column in SINGLE_STOCK_FEATURE_COLUMNS:
        if column not in panel.columns:
            panel[column] = np.nan
        if column not in latest.columns:
            latest[column] = np.nan
    model_config = config["moxing"]
    single_config = config.get("dangu", {})
    minimum_coverage = float(single_config.get("min_feature_coverage", 0.20))
    coverage = panel[SINGLE_STOCK_FEATURE_COLUMNS].notna().mean()
    feature_columns = [column for column in SINGLE_STOCK_FEATURE_COLUMNS if float(coverage.get(column, 0.0)) >= minimum_coverage]
    if not feature_columns:
        raise RuntimeError("单股模型没有达到覆盖率门槛的特征")
    predictions = latest[["ts_code", "name", "trade_date", "close"] + feature_columns].copy()
    validation: dict[str, Any] = {
        "split_method": "purged_expanding_walk_forward_with_final_holdout",
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "feature_coverage": {column: round(float(coverage[column]), 4) for column in feature_columns},
        "latest_missing_features": [column for column in feature_columns if latest[column].isna().any()],
        "feature_preprocessing": "每个滚动训练窗口独立做因子稳定筛选和逐特征分位数去极值；Ridge与方向Logistic另做稳健缩放",
        "horizons": {},
    }
    horizons = [int(value) for value in model_config["horizons"]]
    quantiles = model_config.get("prediction_clip_quantiles", [0.01, 0.99])
    lower_q, upper_q = float(quantiles[0]), float(quantiles[1])
    minimum_train = int(single_config.get("min_fold_training_samples", model_config.get("min_training_samples", 500)))
    minimum_validation = int(single_config.get("min_fold_validation_samples", 80))
    expected_folds = max(2, int(single_config.get("walk_forward_folds", 6)))
    min_passed_folds = max(1, int(single_config.get("min_passed_folds", 4)))
    validation_top_n = max(1, int(model_config.get("validation_top_n", 3)))
    cost_scenario_name = str(config.get("jiaoyi", {}).get("cost_scenario", "normal_cost"))
    _, cost_scenario, _, _ = _load_cost_assumption(cost_scenario_name)

    for horizon in horizons:
        target_column = f"target_t{horizon}"
        target_date_column = f"target_date_t{horizon}"
        entry_date_column = f"entry_date_t{horizon}"
        entry_open_column = f"entry_open_t{horizon}"
        usable = panel.dropna(subset=[target_column, target_date_column, entry_date_column, entry_open_column]).copy()
        usable["trade_date"] = pd.to_datetime(usable["trade_date"])
        usable[target_date_column] = pd.to_datetime(usable[target_date_column])
        usable[entry_date_column] = pd.to_datetime(usable[entry_date_column])
        dates = [pd.Timestamp(value) for value in sorted(usable["trade_date"].dropna().unique())]
        boundaries = _walk_forward_boundaries(dates, config)
        fold_records: list[dict[str, Any]] = []
        oof_frames: list[pd.DataFrame] = []
        prior_actual: list[np.ndarray] = []
        prior_tree_prediction: list[np.ndarray] = []
        prior_linear_prediction: list[np.ndarray] = []
        regime_column = (
            "market_csi300_ret_20"
            if "market_csi300_ret_20" in usable.columns and usable["market_csi300_ret_20"].notna().any()
            else "universe_mean_ret_20"
        )

        for fold_number, (validation_start, validation_end) in enumerate(boundaries, start=1):
            fold_role = "final_holdout" if fold_number == len(boundaries) else "walk_forward_validation"
            train = usable[(usable["trade_date"] < validation_start) & (usable[target_date_column] < validation_start)]
            validation_frame = usable[usable["trade_date"] >= validation_start]
            if validation_end is not None:
                validation_frame = validation_frame[validation_frame["trade_date"] < validation_end]
            if len(train) < minimum_train or len(validation_frame) < minimum_validation:
                fold_records.append({
                    "fold": fold_number,
                    "role": fold_role,
                    "status": "insufficient_samples",
                    "train_samples": int(len(train)),
                    "validation_samples": int(len(validation_frame)),
                    "validation_start": validation_start.strftime("%Y-%m-%d"),
                    "validation_end": validation_end.strftime("%Y-%m-%d") if validation_end is not None else None,
                })
                continue
            y_train_raw = train[target_column].astype(float).to_numpy()
            clip_low = float(np.nanquantile(y_train_raw, lower_q))
            clip_high = float(np.nanquantile(y_train_raw, upper_q))
            fold_features, factor_selection = _select_stable_features(
                train,
                feature_columns,
                target_column,
                model_config,
            )
            if not fold_features:
                fold_records.append({
                    "fold": fold_number,
                    "role": fold_role,
                    "status": "no_stable_features",
                    "train_samples": int(len(train)),
                    "validation_samples": int(len(validation_frame)),
                    "factor_selection": factor_selection,
                })
                continue
            tree_weight, ensemble_diagnostics = _select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            tree_prediction, linear_prediction = _fit_model_components(
                train_features=train[fold_features],
                train_target=np.clip(y_train_raw, clip_low, clip_high),
                predict_features=validation_frame[fold_features],
                model_config=model_config,
            )
            raw_direction_probability, direction_model = _fit_direction_probabilities(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
            )
            quantile_prediction, quantile_model = _fit_quantile_model_components(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
            )
            tree_prediction = np.clip(tree_prediction, clip_low, clip_high)
            linear_prediction = np.clip(linear_prediction, clip_low, clip_high)
            actual = validation_frame[target_column].astype(float).to_numpy()
            predicted = np.clip(
                _blend_component_predictions(tree_prediction, linear_prediction, tree_weight),
                clip_low,
                clip_high,
            )
            naive_comparison = regression_baseline_metrics(
                actual=actual,
                predicted=predicted,
                training_target=y_train_raw,
            )
            baseline_value = float(np.median(y_train_raw))
            baseline = np.full(len(actual), baseline_value)
            mae = float(mean_absolute_error(actual, predicted))
            baseline_mae = float(mean_absolute_error(actual, baseline))
            skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
            direction = float(np.mean((predicted > 0) == (actual > 0)))
            rank_ic, rank_days = _daily_rank_ic(validation_frame["trade_date"], actual, predicted)
            top_n = _top_n_validation_metrics(
                validation_frame,
                actual,
                predicted,
                horizon=horizon,
                budget_yuan=budget_yuan,
                scenario=cost_scenario,
                top_n=validation_top_n,
                trading_settings=config.get("jiaoyi", {}),
            )
            fold_passed = bool(
                direction >= 0.50
                and skill > 0
                and float(naive_comparison["skill_vs_best_naive_baseline"]) > 0
                and rank_ic >= 0
                and top_n["top_n_mean_net_return"] > 0
            )
            fold_records.append({
                "fold": fold_number,
                "role": fold_role,
                "status": "ok",
                "train_samples": int(len(train)),
                "validation_samples": int(len(validation_frame)),
                "validation_start": validation_start.strftime("%Y-%m-%d"),
                "validation_end": validation_frame["trade_date"].max().strftime("%Y-%m-%d"),
                "mae": round(mae, 6),
                "baseline_mae": round(baseline_mae, 6),
                "skill_vs_median_baseline": round(skill, 6),
                "naive_baseline_comparison": naive_comparison,
                "direction_accuracy": round(direction, 6),
                "mean_daily_rank_ic": round(rank_ic, 6),
                "rank_ic_days": int(rank_days),
                "factor_selection": factor_selection,
                "direction_model": {
                    **direction_model,
                    "brier_score": round(
                        float(brier_score_loss((actual > 0).astype(int), raw_direction_probability)),
                        6,
                    ),
                    "classification_accuracy": round(
                        float(np.mean((raw_direction_probability >= 0.5) == (actual > 0))),
                        6,
                    ),
                },
                "quantile_interval_model": quantile_model,
                "model_ensemble": {
                    **ensemble_diagnostics,
                    "weight_uses_only_prior_folds": True,
                    "mean_absolute_component_disagreement": round(
                        float(np.mean(np.abs(tree_prediction - linear_prediction))),
                        6,
                    ),
                },
                **top_n,
                "fold_passed": fold_passed,
            })
            oof_columns = ["trade_date", "ts_code", entry_open_column]
            oof_columns.extend(
                column for column in ["amount_yuan", "atr_14_pct"] if column in validation_frame.columns
            )
            if regime_column in validation_frame.columns:
                oof_columns.append(regime_column)
            oof = validation_frame[oof_columns].copy()
            oof["actual"] = actual
            oof["predicted"] = predicted
            oof["baseline"] = baseline
            oof["tree_prediction"] = tree_prediction
            oof["linear_prediction"] = linear_prediction
            oof["raw_direction_probability"] = raw_direction_probability
            if quantile_prediction:
                oof["quantile_lower"] = quantile_prediction["lower"]
                oof["quantile_median"] = quantile_prediction["median"]
                oof["quantile_upper"] = quantile_prediction["upper"]
            oof_frames.append(oof)
            prior_actual.append(actual)
            prior_tree_prediction.append(tree_prediction)
            prior_linear_prediction.append(linear_prediction)

        latest_prediction = None
        production_ensemble: dict[str, Any] | None = None
        latest_component_predictions: dict[str, float] | None = None
        final_clip_low = None
        final_clip_high = None
        latest_raw_direction_probability = None
        latest_quantile_prediction: dict[str, np.ndarray] = {}
        production_quantile_model: dict[str, Any] = {"status": "unavailable"}
        production_factor_selection: dict[str, Any] | None = None
        production_features: list[str] = []
        if len(usable) >= minimum_train:
            full_y = usable[target_column].astype(float).to_numpy()
            final_clip_low = float(np.nanquantile(full_y, lower_q))
            final_clip_high = float(np.nanquantile(full_y, upper_q))
            production_tree_weight, production_ensemble = _select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            production_features, production_factor_selection = _select_stable_features(
                usable,
                feature_columns,
                target_column,
                model_config,
            )
            latest_tree_prediction, latest_linear_prediction = _fit_model_components(
                train_features=usable[production_features],
                train_target=np.clip(full_y, final_clip_low, final_clip_high),
                predict_features=latest[production_features],
                model_config=model_config,
            )
            latest_direction_array, production_direction_model = _fit_direction_probabilities(
                train_features=usable[production_features],
                train_target=full_y,
                predict_features=latest[production_features],
                model_config=model_config,
            )
            latest_quantile_prediction, production_quantile_model = (
                _fit_quantile_model_components(
                    train_features=usable[production_features],
                    train_target=full_y,
                    predict_features=latest[production_features],
                    model_config=model_config,
                )
            )
            latest_raw_direction_probability = float(latest_direction_array[0])
            latest_tree_prediction = np.clip(
                latest_tree_prediction, final_clip_low, final_clip_high
            )
            latest_linear_prediction = np.clip(
                latest_linear_prediction, final_clip_low, final_clip_high
            )
            latest_prediction = float(np.clip(
                _blend_component_predictions(
                    latest_tree_prediction,
                    latest_linear_prediction,
                    production_tree_weight,
                )[0],
                final_clip_low,
                final_clip_high,
            ))
            latest_component_predictions = {
                "tree": round(float(latest_tree_prediction[0]), 6),
                "linear": round(float(latest_linear_prediction[0]), 6),
            }
            predictions[f"pred_t{horizon}"] = latest_prediction
        else:
            predictions[f"pred_t{horizon}"] = np.nan

        successful_folds = [record for record in fold_records if record.get("status") == "ok"]
        passed_folds = sum(bool(record.get("fold_passed")) for record in successful_folds)
        final_holdout = next(
            (record for record in successful_folds if record.get("role") == "final_holdout"),
            None,
        )
        final_holdout_passed = bool(final_holdout and final_holdout.get("fold_passed"))
        horizon_validation: dict[str, Any] = {
            "status": "ok" if latest_prediction is not None else "insufficient_training_samples",
            "walk_forward_folds_requested": expected_folds,
            "walk_forward_folds_completed": int(len(successful_folds)),
            "walk_forward_folds_passed": int(passed_folds),
            "minimum_passed_folds": min_passed_folds,
            "final_holdout_passed": final_holdout_passed,
            "folds": fold_records,
            "final_train_samples": int(len(usable)),
            "final_training_end": usable[target_date_column].max().strftime("%Y-%m-%d") if not usable.empty else None,
            "final_prediction_clip": [round(final_clip_low, 6), round(final_clip_high, 6)] if final_clip_low is not None else None,
            "retrained_on_all_labeled_data": latest_prediction is not None,
            "production_factor_selection": production_factor_selection,
            "experiment_fingerprint": _experiment_fingerprint(
                feature_columns=production_features,
                target_definition=f"next_session_open_to_{horizon}th_sellable_close_return",
                split_method="purged_expanding_walk_forward_with_final_holdout",
                model_config=model_config,
            ) if production_features else None,
            "production_model_ensemble": {
                "components": ["HistGradientBoostingRegressor", "Ridge"],
                "weight_selection": production_ensemble,
                "latest_component_predictions": latest_component_predictions,
            },
            "production_direction_model": {
                **(production_direction_model if latest_raw_direction_probability is not None else {}),
                "latest_raw_positive_probability": round(latest_raw_direction_probability, 6)
                if latest_raw_direction_probability is not None
                else None,
            },
            "production_quantile_interval_model": production_quantile_model,
        }
        if oof_frames and latest_prediction is not None:
            oof = pd.concat(oof_frames, ignore_index=True)
            actual = oof["actual"].to_numpy(dtype=float)
            predicted = oof["predicted"].to_numpy(dtype=float)
            baseline = oof["baseline"].to_numpy(dtype=float)
            mae = float(mean_absolute_error(actual, predicted))
            baseline_mae = float(mean_absolute_error(actual, baseline))
            skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
            first_validation_start = boundaries[0][0] if boundaries else usable["trade_date"].min()
            baseline_training = usable[
                (usable["trade_date"] < first_validation_start)
                & (usable[target_date_column] < first_validation_start)
            ][target_column].to_numpy(dtype=float)
            naive_comparison = regression_baseline_metrics(
                actual=actual,
                predicted=predicted,
                training_target=baseline_training,
            )
            direction = float(np.mean((predicted > 0) == (actual > 0)))
            rank_ic, rank_days = _daily_rank_ic(oof["trade_date"], actual, predicted)
            top_n = _top_n_validation_metrics(
                oof.rename(columns={entry_open_column: f"entry_open_t{horizon}"}),
                actual,
                predicted,
                horizon=horizon,
                budget_yuan=budget_yuan,
                scenario=cost_scenario,
                top_n=validation_top_n,
                trading_settings=config.get("jiaoyi", {}),
            )
            residual = actual - predicted
            residual_low, residual_high = np.nanquantile(residual, [0.10, 0.90])
            nearest_count = min(len(oof), max(40, len(oof) // 5))
            nearest_index = np.argsort(np.abs(predicted - latest_prediction))[:nearest_count]
            nearest_actual = actual[nearest_index]
            positive_probability = float(np.mean(nearest_actual > 0)) if nearest_count else None
            empirical_low, empirical_high = np.nanquantile(nearest_actual, [0.10, 0.90])
            calibrated_probability, probability_calibration = _calibrate_direction_probability(
                actual=actual,
                raw_probability=oof["raw_direction_probability"].to_numpy(dtype=float),
                dates=oof["trade_date"],
                latest_raw_probability=float(latest_raw_direction_probability),
                model_config=model_config,
            )
            conformal_interval, conformal_diagnostics = _rolling_conformal_interval(
                actual=actual,
                predicted=predicted,
                dates=oof["trade_date"],
                latest_prediction=float(latest_prediction),
                model_config=model_config,
            )
            cqr_interval = None
            cqr_diagnostics: dict[str, Any] = {
                "status": "unavailable",
                "reason": "分位数样本外预测不可用",
            }
            if (
                {"quantile_lower", "quantile_upper"}.issubset(oof.columns)
                and latest_quantile_prediction
            ):
                cqr_interval, cqr_diagnostics = _rolling_cqr_interval(
                    actual=actual,
                    lower_prediction=oof["quantile_lower"].to_numpy(dtype=float),
                    upper_prediction=oof["quantile_upper"].to_numpy(dtype=float),
                    dates=oof["trade_date"],
                    latest_lower=float(latest_quantile_prediction["lower"][0]),
                    latest_upper=float(latest_quantile_prediction["upper"][0]),
                    model_config=model_config,
                )
            preferred_interval = cqr_interval or conformal_interval
            preferred_interval_method = (
                "rolling_conformalized_quantile_regression"
                if cqr_interval
                else "rolling_symmetric_conformal"
                if conformal_interval
                else "nearest_oos_empirical_quantile"
            )
            quality = _quality_score(
                train_count=len(usable),
                direction_accuracy=direction,
                rank_ic=rank_ic,
                skill_vs_baseline=skill,
            )
            minimum_rank_ic = float(model_config.get("min_mean_daily_rank_ic", 0.01))
            minimum_skill = float(model_config.get("min_skill_vs_baseline", 0.01))
            minimum_best_naive_skill = float(model_config.get("min_skill_vs_best_naive_baseline", 0.0))
            minimum_direction = float(model_config.get("min_direction_accuracy", 0.52))
            minimum_rank_days = int(model_config.get("min_rank_ic_days", 10))
            minimum_top_days = int(model_config.get("min_top_n_days", 10))
            validation_passed = bool(
                len(successful_folds) == expected_folds
                and passed_folds >= min_passed_folds
                and final_holdout_passed
                and direction >= minimum_direction
                and rank_ic >= minimum_rank_ic
                and rank_days >= minimum_rank_days
                and skill >= minimum_skill
                and float(naive_comparison["skill_vs_best_naive_baseline"]) >= minimum_best_naive_skill
                and top_n["top_n_days"] >= minimum_top_days
                and top_n["top_n_mean_net_return"] > 0
                and top_n["top_n_mean_excess_vs_universe"] > 0
            )
            horizon_validation.update({
                "oos_samples": int(len(oof)),
                "mae": round(mae, 6),
                "baseline_mae": round(baseline_mae, 6),
                "skill_vs_median_baseline": round(skill, 6),
                "naive_baseline_comparison": naive_comparison,
                "direction_accuracy": round(direction, 6),
                "mean_daily_rank_ic": round(rank_ic, 6),
                "rank_ic_days": int(rank_days),
                **top_n,
                "residual_std": round(float(np.std(residual, ddof=1)), 6) if len(residual) > 1 else None,
                "residual_quantiles_10_90": [round(float(residual_low), 6), round(float(residual_high), 6)],
                "latest_empirical_positive_probability": round(positive_probability, 6) if positive_probability is not None else None,
                "probability_calibration_samples": int(nearest_count),
                "probability_method": "与当前预测最接近的滚动样本外预测，其实际毛收益为正的比例",
                "latest_direction_positive_probability": round(float(calibrated_probability), 6),
                "direction_probability_method": probability_calibration.get("method"),
                "direction_probability_calibration": probability_calibration,
                "prediction_interval_80": [
                    round(float(empirical_low), 6),
                    round(float(empirical_high), 6),
                ],
                "prediction_interval_method": "与正收益比例相同的近邻滚动样本外实际收益的10%至90%经验分位数",
                "conformal_prediction_interval_80": [round(float(value), 6) for value in conformal_interval]
                if conformal_interval
                else None,
                "conformal_diagnostics": conformal_diagnostics,
                "quantile_prediction_interval_80": [
                    round(float(latest_quantile_prediction["lower"][0]), 6),
                    round(float(latest_quantile_prediction["upper"][0]), 6),
                ]
                if latest_quantile_prediction
                else None,
                "quantile_median_prediction": round(
                    float(latest_quantile_prediction["median"][0]), 6
                )
                if latest_quantile_prediction
                else None,
                "conformalized_quantile_prediction_interval_80": [
                    round(float(value), 6) for value in cqr_interval
                ]
                if cqr_interval
                else None,
                "cqr_diagnostics": cqr_diagnostics,
                "preferred_prediction_interval_80": [
                    round(float(value), 6) for value in preferred_interval
                ]
                if preferred_interval
                else [
                    round(float(empirical_low), 6),
                    round(float(empirical_high), 6),
                ],
                "preferred_prediction_interval_method": preferred_interval_method,
                "fold_stability": _fold_stability(fold_records),
                "market_regime_stability": _regime_stability(oof, regime_column=regime_column),
                "quality_score": round(quality, 4),
                "quality_label": _quality_label(quality),
                "validation_passed": validation_passed,
                "validation_thresholds": {
                    "completed_folds": expected_folds,
                    "passed_folds": min_passed_folds,
                    "final_holdout_must_pass": True,
                    "direction_accuracy": minimum_direction,
                    "mean_daily_rank_ic": minimum_rank_ic,
                    "rank_ic_days": minimum_rank_days,
                    "skill_vs_median_baseline": minimum_skill,
                    "skill_vs_best_naive_baseline": minimum_best_naive_skill,
                    "top_n_days": minimum_top_days,
                    "top_n_mean_net_return": "> 0",
                    "top_n_mean_excess_vs_universe": "> 0",
                },
            })
        else:
            horizon_validation.update({
                "validation_passed": False,
                "quality_score": 0.0,
                "quality_label": "low",
                "unavailable_reason": "滚动样本外折数或训练样本不足",
            })
        validation["horizons"][f"T+{horizon}"] = horizon_validation

    validation["passed_horizons"] = sum(
        bool(value.get("validation_passed")) for value in validation["horizons"].values()
    )
    qualities = [float(value.get("quality_score", 0.0)) for value in validation["horizons"].values()]
    validation["overall_quality_score"] = round(float(np.mean(qualities)), 4) if qualities else 0.0
    validation["overall_quality_label"] = _quality_label(float(validation["overall_quality_score"]))
    return predictions, validation


def _fit_future_session_models(
    *,
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict the next one to three market-session closes from the signal close."""
    panel = panel.copy()
    latest = latest.copy()
    for column in SINGLE_STOCK_FEATURE_COLUMNS:
        if column not in panel.columns:
            panel[column] = np.nan
        if column not in latest.columns:
            latest[column] = np.nan
    model_config = config["moxing"]
    single_config = config.get("dangu", {})
    minimum_coverage = float(single_config.get("min_feature_coverage", 0.20))
    coverage = panel[SINGLE_STOCK_FEATURE_COLUMNS].notna().mean()
    feature_columns = [
        column
        for column in SINGLE_STOCK_FEATURE_COLUMNS
        if float(coverage.get(column, 0.0)) >= minimum_coverage
    ]
    if not feature_columns:
        raise RuntimeError("未来三交易日模型没有达到覆盖率门槛的特征")

    predictions = latest[["ts_code", "name", "trade_date", "close"] + feature_columns].copy()
    validation: dict[str, Any] = {
        "split_method": "purged_expanding_walk_forward_with_final_holdout",
        "forecast_basis": "最近完整收盘价到未来第1/2/3个市场交易日收盘的累计收益",
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "feature_preprocessing": "每个滚动训练窗口独立做因子稳定筛选和逐特征分位数去极值；Ridge与方向Logistic另做稳健缩放",
        "horizons": {},
    }
    horizons = [int(value) for value in model_config["horizons"]]
    lower_q, upper_q = [
        float(value) for value in model_config.get("prediction_clip_quantiles", [0.01, 0.99])
    ]
    minimum_train = int(
        single_config.get("min_fold_training_samples", model_config.get("min_training_samples", 500))
    )
    minimum_validation = int(single_config.get("min_fold_validation_samples", 80))
    expected_folds = max(2, int(single_config.get("walk_forward_folds", 6)))
    min_passed_folds = max(1, int(single_config.get("min_passed_folds", 4)))
    minimum_direction = float(model_config.get("min_direction_accuracy", 0.52))
    minimum_rank_ic = float(model_config.get("min_mean_daily_rank_ic", 0.01))
    minimum_rank_days = int(model_config.get("min_rank_ic_days", 10))
    minimum_skill = float(model_config.get("min_skill_vs_baseline", 0.01))

    for horizon in horizons:
        target_column = f"future_return_t{horizon}"
        target_date_column = f"future_date_t{horizon}"
        usable = panel.dropna(subset=[target_column, target_date_column]).copy()
        usable["trade_date"] = pd.to_datetime(usable["trade_date"])
        usable[target_date_column] = pd.to_datetime(usable[target_date_column])
        dates = [pd.Timestamp(value) for value in sorted(usable["trade_date"].dropna().unique())]
        boundaries = _walk_forward_boundaries(dates, config)
        fold_records: list[dict[str, Any]] = []
        oof_frames: list[pd.DataFrame] = []
        prior_actual: list[np.ndarray] = []
        prior_tree_prediction: list[np.ndarray] = []
        prior_linear_prediction: list[np.ndarray] = []
        regime_column = (
            "market_csi300_ret_20"
            if "market_csi300_ret_20" in usable.columns and usable["market_csi300_ret_20"].notna().any()
            else "universe_mean_ret_20"
        )

        for fold_number, (validation_start, validation_end) in enumerate(boundaries, start=1):
            role = "final_holdout" if fold_number == len(boundaries) else "walk_forward_validation"
            train = usable[
                (usable["trade_date"] < validation_start)
                & (usable[target_date_column] < validation_start)
            ]
            validation_frame = usable[usable["trade_date"] >= validation_start]
            if validation_end is not None:
                validation_frame = validation_frame[validation_frame["trade_date"] < validation_end]
            if len(train) < minimum_train or len(validation_frame) < minimum_validation:
                fold_records.append(
                    {
                        "fold": fold_number,
                        "role": role,
                        "status": "insufficient_samples",
                        "train_samples": int(len(train)),
                        "validation_samples": int(len(validation_frame)),
                        "validation_start": validation_start.strftime("%Y-%m-%d"),
                        "validation_end": validation_end.strftime("%Y-%m-%d") if validation_end is not None else None,
                    }
                )
                continue

            y_train_raw = train[target_column].astype(float).to_numpy()
            clip_low = float(np.nanquantile(y_train_raw, lower_q))
            clip_high = float(np.nanquantile(y_train_raw, upper_q))
            fold_features, factor_selection = _select_stable_features(
                train,
                feature_columns,
                target_column,
                model_config,
            )
            if not fold_features:
                fold_records.append(
                    {
                        "fold": fold_number,
                        "role": role,
                        "status": "no_stable_features",
                        "train_samples": int(len(train)),
                        "validation_samples": int(len(validation_frame)),
                        "factor_selection": factor_selection,
                    }
                )
                continue
            tree_weight, ensemble_diagnostics = _select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            tree_prediction, linear_prediction = _fit_model_components(
                train_features=train[fold_features],
                train_target=np.clip(y_train_raw, clip_low, clip_high),
                predict_features=validation_frame[fold_features],
                model_config=model_config,
            )
            raw_direction_probability, direction_model = _fit_direction_probabilities(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
            )
            quantile_prediction, quantile_model = _fit_quantile_model_components(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
            )
            tree_prediction = np.clip(tree_prediction, clip_low, clip_high)
            linear_prediction = np.clip(linear_prediction, clip_low, clip_high)
            actual = validation_frame[target_column].astype(float).to_numpy()
            predicted = np.clip(
                _blend_component_predictions(tree_prediction, linear_prediction, tree_weight),
                clip_low,
                clip_high,
            )
            baseline_value = float(np.median(y_train_raw))
            baseline = np.full(len(actual), baseline_value)
            mae = float(mean_absolute_error(actual, predicted))
            baseline_mae = float(mean_absolute_error(actual, baseline))
            skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
            direction = float(np.mean((predicted > 0) == (actual > 0)))
            rank_ic, rank_days = _daily_rank_ic(validation_frame["trade_date"], actual, predicted)
            fold_passed = bool(direction >= 0.50 and skill > 0 and rank_ic >= 0)
            fold_records.append(
                {
                    "fold": fold_number,
                    "role": role,
                    "status": "ok",
                    "train_samples": int(len(train)),
                    "validation_samples": int(len(validation_frame)),
                    "validation_start": validation_start.strftime("%Y-%m-%d"),
                    "validation_end": validation_frame["trade_date"].max().strftime("%Y-%m-%d"),
                    "mae": round(mae, 6),
                    "baseline_mae": round(baseline_mae, 6),
                    "skill_vs_median_baseline": round(skill, 6),
                    "direction_accuracy": round(direction, 6),
                    "mean_daily_rank_ic": round(rank_ic, 6),
                    "rank_ic_days": int(rank_days),
                    "factor_selection": factor_selection,
                    "direction_model": {
                        **direction_model,
                        "brier_score": round(
                            float(brier_score_loss((actual > 0).astype(int), raw_direction_probability)),
                            6,
                        ),
                        "classification_accuracy": round(
                            float(np.mean((raw_direction_probability >= 0.5) == (actual > 0))),
                            6,
                        ),
                    },
                    "quantile_interval_model": quantile_model,
                    "model_ensemble": {
                        **ensemble_diagnostics,
                        "weight_uses_only_prior_folds": True,
                        "mean_absolute_component_disagreement": round(
                            float(np.mean(np.abs(tree_prediction - linear_prediction))),
                            6,
                        ),
                    },
                    "fold_passed": fold_passed,
                }
            )
            oof_columns = ["trade_date", "ts_code"]
            if regime_column in validation_frame.columns:
                oof_columns.append(regime_column)
            oof = validation_frame[oof_columns].copy()
            oof["actual"] = actual
            oof["predicted"] = predicted
            oof["baseline"] = baseline
            oof["tree_prediction"] = tree_prediction
            oof["linear_prediction"] = linear_prediction
            oof["raw_direction_probability"] = raw_direction_probability
            if quantile_prediction:
                oof["quantile_lower"] = quantile_prediction["lower"]
                oof["quantile_median"] = quantile_prediction["median"]
                oof["quantile_upper"] = quantile_prediction["upper"]
            oof_frames.append(oof)
            prior_actual.append(actual)
            prior_tree_prediction.append(tree_prediction)
            prior_linear_prediction.append(linear_prediction)

        latest_prediction = None
        production_ensemble: dict[str, Any] | None = None
        latest_component_predictions: dict[str, float] | None = None
        final_clip_low = None
        final_clip_high = None
        latest_raw_direction_probability = None
        latest_quantile_prediction: dict[str, np.ndarray] = {}
        production_quantile_model: dict[str, Any] = {"status": "unavailable"}
        production_factor_selection: dict[str, Any] | None = None
        production_features: list[str] = []
        if len(usable) >= minimum_train:
            full_y = usable[target_column].astype(float).to_numpy()
            final_clip_low = float(np.nanquantile(full_y, lower_q))
            final_clip_high = float(np.nanquantile(full_y, upper_q))
            production_tree_weight, production_ensemble = _select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            production_features, production_factor_selection = _select_stable_features(
                usable,
                feature_columns,
                target_column,
                model_config,
            )
            latest_tree_prediction, latest_linear_prediction = _fit_model_components(
                train_features=usable[production_features],
                train_target=np.clip(full_y, final_clip_low, final_clip_high),
                predict_features=latest[production_features],
                model_config=model_config,
            )
            latest_direction_array, production_direction_model = _fit_direction_probabilities(
                train_features=usable[production_features],
                train_target=full_y,
                predict_features=latest[production_features],
                model_config=model_config,
            )
            latest_quantile_prediction, production_quantile_model = (
                _fit_quantile_model_components(
                    train_features=usable[production_features],
                    train_target=full_y,
                    predict_features=latest[production_features],
                    model_config=model_config,
                )
            )
            latest_raw_direction_probability = float(latest_direction_array[0])
            latest_tree_prediction = np.clip(
                latest_tree_prediction, final_clip_low, final_clip_high
            )
            latest_linear_prediction = np.clip(
                latest_linear_prediction, final_clip_low, final_clip_high
            )
            latest_prediction = float(
                np.clip(
                    _blend_component_predictions(
                        latest_tree_prediction,
                        latest_linear_prediction,
                        production_tree_weight,
                    )[0],
                    final_clip_low,
                    final_clip_high,
                )
            )
            latest_component_predictions = {
                "tree": round(float(latest_tree_prediction[0]), 6),
                "linear": round(float(latest_linear_prediction[0]), 6),
            }
            predictions[f"future_pred_t{horizon}"] = latest_prediction
        else:
            predictions[f"future_pred_t{horizon}"] = np.nan

        successful_folds = [record for record in fold_records if record.get("status") == "ok"]
        passed_folds = sum(bool(record.get("fold_passed")) for record in successful_folds)
        final_holdout = next(
            (record for record in successful_folds if record.get("role") == "final_holdout"),
            None,
        )
        final_holdout_passed = bool(final_holdout and final_holdout.get("fold_passed"))
        horizon_validation: dict[str, Any] = {
            "status": "ok" if latest_prediction is not None else "insufficient_training_samples",
            "walk_forward_folds_requested": expected_folds,
            "walk_forward_folds_completed": int(len(successful_folds)),
            "walk_forward_folds_passed": int(passed_folds),
            "minimum_passed_folds": min_passed_folds,
            "final_holdout_passed": final_holdout_passed,
            "folds": fold_records,
            "final_train_samples": int(len(usable)),
            "final_training_end": (
                usable[target_date_column].max().strftime("%Y-%m-%d") if not usable.empty else None
            ),
            "final_prediction_clip": (
                [round(final_clip_low, 6), round(final_clip_high, 6)]
                if final_clip_low is not None
                else None
            ),
            "production_factor_selection": production_factor_selection,
            "experiment_fingerprint": _experiment_fingerprint(
                feature_columns=production_features,
                target_definition=f"signal_close_to_future_market_session_{horizon}_close_return",
                split_method="purged_expanding_walk_forward_with_final_holdout",
                model_config=model_config,
            ) if production_features else None,
            "production_model_ensemble": {
                "components": ["HistGradientBoostingRegressor", "Ridge"],
                "weight_selection": production_ensemble,
                "latest_component_predictions": latest_component_predictions,
            },
            "production_direction_model": {
                **(production_direction_model if latest_raw_direction_probability is not None else {}),
                "latest_raw_positive_probability": round(latest_raw_direction_probability, 6)
                if latest_raw_direction_probability is not None
                else None,
            },
            "production_quantile_interval_model": production_quantile_model,
        }

        if oof_frames and latest_prediction is not None:
            oof = pd.concat(oof_frames, ignore_index=True)
            actual = oof["actual"].to_numpy(dtype=float)
            predicted = oof["predicted"].to_numpy(dtype=float)
            baseline = oof["baseline"].to_numpy(dtype=float)
            mae = float(mean_absolute_error(actual, predicted))
            baseline_mae = float(mean_absolute_error(actual, baseline))
            skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
            direction = float(np.mean((predicted > 0) == (actual > 0)))
            rank_ic, rank_days = _daily_rank_ic(oof["trade_date"], actual, predicted)
            residual = actual - predicted
            residual_low, residual_high = np.nanquantile(residual, [0.10, 0.90])
            nearest_count = min(len(oof), max(40, len(oof) // 5))
            nearest_index = np.argsort(np.abs(predicted - latest_prediction))[:nearest_count]
            nearest_actual = actual[nearest_index]
            positive_probability = (
                float(np.mean(nearest_actual > 0)) if nearest_count else None
            )
            empirical_low, empirical_high = np.nanquantile(nearest_actual, [0.10, 0.90])
            calibrated_probability, probability_calibration = _calibrate_direction_probability(
                actual=actual,
                raw_probability=oof["raw_direction_probability"].to_numpy(dtype=float),
                dates=oof["trade_date"],
                latest_raw_probability=float(latest_raw_direction_probability),
                model_config=model_config,
            )
            conformal_interval, conformal_diagnostics = _rolling_conformal_interval(
                actual=actual,
                predicted=predicted,
                dates=oof["trade_date"],
                latest_prediction=float(latest_prediction),
                model_config=model_config,
            )
            cqr_interval = None
            cqr_diagnostics: dict[str, Any] = {
                "status": "unavailable",
                "reason": "分位数样本外预测不可用",
            }
            if (
                {"quantile_lower", "quantile_upper"}.issubset(oof.columns)
                and latest_quantile_prediction
            ):
                cqr_interval, cqr_diagnostics = _rolling_cqr_interval(
                    actual=actual,
                    lower_prediction=oof["quantile_lower"].to_numpy(dtype=float),
                    upper_prediction=oof["quantile_upper"].to_numpy(dtype=float),
                    dates=oof["trade_date"],
                    latest_lower=float(latest_quantile_prediction["lower"][0]),
                    latest_upper=float(latest_quantile_prediction["upper"][0]),
                    model_config=model_config,
                )
            preferred_interval = cqr_interval or conformal_interval
            preferred_interval_method = (
                "rolling_conformalized_quantile_regression"
                if cqr_interval
                else "rolling_symmetric_conformal"
                if conformal_interval
                else "nearest_oos_empirical_quantile"
            )
            quality = _quality_score(
                train_count=len(usable),
                direction_accuracy=direction,
                rank_ic=rank_ic,
                skill_vs_baseline=skill,
            )
            validation_passed = bool(
                len(successful_folds) == expected_folds
                and passed_folds >= min_passed_folds
                and final_holdout_passed
                and direction >= minimum_direction
                and rank_ic >= minimum_rank_ic
                and rank_days >= minimum_rank_days
                and skill >= minimum_skill
            )
            horizon_validation.update(
                {
                    "oos_samples": int(len(oof)),
                    "mae": round(mae, 6),
                    "baseline_mae": round(baseline_mae, 6),
                    "skill_vs_median_baseline": round(skill, 6),
                    "direction_accuracy": round(direction, 6),
                    "mean_daily_rank_ic": round(rank_ic, 6),
                    "rank_ic_days": int(rank_days),
                    "residual_std": (
                        round(float(np.std(residual, ddof=1)), 6) if len(residual) > 1 else None
                    ),
                    "latest_empirical_positive_probability": (
                        round(positive_probability, 6) if positive_probability is not None else None
                    ),
                    "probability_calibration_samples": int(nearest_count),
                    "probability_method": "与当前预测最接近的滚动样本外预测，其实际累计收益为正的比例",
                    "latest_direction_positive_probability": round(float(calibrated_probability), 6),
                    "direction_probability_method": probability_calibration.get("method"),
                    "direction_probability_calibration": probability_calibration,
                    "prediction_interval_80": [
                        round(float(empirical_low), 6),
                        round(float(empirical_high), 6),
                    ],
                    "prediction_interval_method": "与正收益比例相同的近邻滚动样本外实际收益的10%至90%经验分位数",
                    "conformal_prediction_interval_80": [round(float(value), 6) for value in conformal_interval]
                    if conformal_interval
                    else None,
                    "conformal_diagnostics": conformal_diagnostics,
                    "quantile_prediction_interval_80": [
                        round(float(latest_quantile_prediction["lower"][0]), 6),
                        round(float(latest_quantile_prediction["upper"][0]), 6),
                    ]
                    if latest_quantile_prediction
                    else None,
                    "quantile_median_prediction": round(
                        float(latest_quantile_prediction["median"][0]), 6
                    )
                    if latest_quantile_prediction
                    else None,
                    "conformalized_quantile_prediction_interval_80": [
                        round(float(value), 6) for value in cqr_interval
                    ]
                    if cqr_interval
                    else None,
                    "cqr_diagnostics": cqr_diagnostics,
                    "preferred_prediction_interval_80": [
                        round(float(value), 6) for value in preferred_interval
                    ]
                    if preferred_interval
                    else [
                        round(float(empirical_low), 6),
                        round(float(empirical_high), 6),
                    ],
                    "preferred_prediction_interval_method": preferred_interval_method,
                    "fold_stability": _fold_stability(fold_records),
                    "market_regime_stability": _regime_stability(oof, regime_column=regime_column),
                    "quality_score": round(quality, 4),
                    "quality_label": _quality_label(quality),
                    "validation_passed": validation_passed,
                    "validation_thresholds": {
                        "completed_folds": expected_folds,
                        "passed_folds": min_passed_folds,
                        "final_holdout_must_pass": True,
                        "direction_accuracy": minimum_direction,
                        "mean_daily_rank_ic": minimum_rank_ic,
                        "rank_ic_days": minimum_rank_days,
                        "skill_vs_median_baseline": minimum_skill,
                    },
                }
            )
        else:
            horizon_validation.update(
                {
                    "validation_passed": False,
                    "quality_score": 0.0,
                    "quality_label": "low",
                    "unavailable_reason": "滚动样本外折数或训练样本不足",
                }
            )
        validation["horizons"][f"T+{horizon}"] = horizon_validation

    validation["passed_horizons"] = sum(
        bool(value.get("validation_passed")) for value in validation["horizons"].values()
    )
    qualities = [float(value.get("quality_score", 0.0)) for value in validation["horizons"].values()]
    validation["overall_quality_score"] = round(float(np.mean(qualities)), 4) if qualities else 0.0
    validation["overall_quality_label"] = _quality_label(float(validation["overall_quality_score"]))
    return predictions, validation


def _next_market_sessions(signal_date: str, count: int = 4) -> dict[str, Any]:
    signal = pd.Timestamp(signal_date).normalize()
    warnings: list[str] = []
    sessions: list[pd.Timestamp] = []
    try:
        pro = _tushare_pro()
        calendar = pro.trade_cal(
            exchange="SSE",
            start_date=(signal + timedelta(days=1)).strftime("%Y%m%d"),
            end_date=(signal + timedelta(days=25)).strftime("%Y%m%d"),
            is_open="1",
            fields="cal_date,is_open",
        )
        if calendar is not None and not calendar.empty:
            sessions = sorted(pd.to_datetime(calendar["cal_date"], errors="coerce").dropna())[:count]
        source = "tushare_exchange_calendar"
    except Exception as exc:
        warnings.append(f"交易所日历不可用：{exc}")
        source = "weekday_fallback"
    if len(sessions) < count:
        sessions = []
        candidate = signal + timedelta(days=1)
        while len(sessions) < count:
            if candidate.weekday() < 5:
                sessions.append(candidate)
            candidate += timedelta(days=1)
        source = "weekday_fallback"
        warnings.append("计划日期按工作日推算，法定节假日需以交易所实际开市日校正")
    return {
        "source": source,
        "signal_date": signal.strftime("%Y-%m-%d"),
        "future_session_dates": {
            f"T+{horizon}": sessions[horizon - 1].strftime("%Y-%m-%d")
            for horizon in [1, 2, 3]
        },
        "assumed_entry_date": sessions[0].strftime("%Y-%m-%d"),
        "scenario_exit_dates": {f"T+{horizon}": sessions[horizon].strftime("%Y-%m-%d") for horizon in [1, 2, 3]},
        "warnings": warnings,
    }


def _future_schedule_unavailable_reason(
    schedule: dict[str, Any],
    tradability: dict[str, Any],
) -> str:
    """Reject a so-called future horizon whose first target session has already ended."""
    clock = tradability.get("market_clock") or {}
    captured_at = pd.to_datetime(clock.get("captured_at"), errors="coerce")
    first_target = pd.to_datetime(
        (schedule.get("future_session_dates") or {}).get("T+1"),
        errors="coerce",
    )
    if pd.isna(first_target):
        return "无法确定未来第一个交易日"
    if pd.isna(captured_at):
        return "缺少分析时间，无法确认预测日期仍属于未来"
    captured_day = pd.Timestamp(captured_at).normalize()
    target_day = pd.Timestamp(first_target).normalize()
    session_status = str(clock.get("session_status") or "")
    if target_day < captured_day or (
        target_day == captured_day
        and session_status in {"close_pending", "post_close", "non_trading_day"}
    ):
        return (
            f"行情源最新完整日线对应的首个预测日 {target_day.strftime('%Y-%m-%d')} 已经结束；"
            "需要等待数据源更新到最新完整收盘后重新预测"
        )
    return ""


def _fundamental_risk_flags(fundamentals: dict[str, Any]) -> list[str]:
    financials = fundamentals.get("financials") or {}
    valuation = fundamentals.get("valuation") or {}
    profile = fundamentals.get("profile") or {}
    industry = str(profile.get("industry") or profile.get("所属行业") or "")
    financial_industry = any(value in industry for value in ("银行", "保险", "证券", "多元金融"))
    flags: list[str] = []
    roe = _round_optional(financials.get("roe_pct"))
    growth = _round_optional(financials.get("net_profit_yoy_pct"))
    debt = _round_optional(financials.get("debt_to_assets_pct"))
    pe = _round_optional(valuation.get("pe_ttm"))
    if pe is None:
        pe = _round_optional(valuation.get("pe_dynamic"))
    if roe is not None and roe < 0:
        flags.append("最新已公告口径 ROE 为负")
    if growth is not None and growth < -30:
        flags.append("最新已公告口径净利润同比下降超过30%")
    if debt is not None and debt > 80 and not financial_industry:
        flags.append("非金融企业资产负债率超过80%")
    if pe is not None and pe <= 0:
        flags.append("当前市盈率口径为负")
    return flags


def _build_analysis_assessment(
    *,
    holding_days: int,
    forecast: dict[str, Any],
    validation: dict[str, Any],
    tradability: dict[str, Any],
    technical: dict[str, Any],
    fundamentals: dict[str, Any],
    schedule: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config.get("dangu", {})
    minimum_net = float(settings.get("assessment_min_net_return", 0.003))
    minimum_probability = float(settings.get("assessment_min_positive_probability", 0.55))
    evidence_label = "证据不足"
    reasons: list[str] = []
    requested = forecast.get(f"T+{holding_days}") or {}
    metrics = validation.get("horizons", {}).get(f"T+{holding_days}") or {}
    fundamental_flags = _fundamental_risk_flags(fundamentals)
    entry_timing_valid = bool(tradability.get("model_entry_timing_valid", True))
    entry_timing_reason = str(tradability.get("model_entry_timing_reason") or "模型假设的开盘测算基准可用")
    assumed_entry = pd.to_datetime(schedule.get("assumed_entry_date"), errors="coerce")
    captured_at = pd.to_datetime(
        (tradability.get("market_clock") or {}).get("captured_at"),
        errors="coerce",
    )
    if pd.notna(assumed_entry) and pd.notna(captured_at):
        assumed_entry = pd.Timestamp(assumed_entry).normalize()
        captured_day = pd.Timestamp(captured_at).normalize()
        captured_minute = pd.Timestamp(captured_at).hour * 60 + pd.Timestamp(captured_at).minute
        if assumed_entry < captured_day or (assumed_entry == captured_day and captured_minute >= 9 * 60 + 30):
            entry_timing_valid = False
            entry_timing_reason = "该收盘信号对应的假设开盘入口已经过去，当前价格不属于模型训练的入口口径"
        elif assumed_entry > captured_day or captured_minute < 9 * 60 + 30:
            entry_timing_valid = True
            entry_timing_reason = "模型假设的开盘入口尚未发生，情景时点仍有效"

    if not tradability.get("basic_execution_feasible"):
        evidence_label = "证据偏负面"
        reasons.extend(str(value) for value in tradability.get("hard_blocks", []))
    elif not entry_timing_valid:
        evidence_label = "证据不足"
        reasons.append(entry_timing_reason)
    elif not requested or requested.get("entry_to_exit_gross_return") is None:
        reasons.append("指定持有期没有可用模型预测")
    elif not metrics.get("validation_passed"):
        reasons.append("指定持有期模型没有通过滚动样本外门槛")
    elif not requested.get("position_and_cost", {}).get("execution_feasible"):
        evidence_label = "证据偏负面"
        reasons.append(
            str(
                requested.get("position_and_cost", {}).get("reason")
                or "按目标资金测算，无法满足最低买入数量或成交容量约束"
            )
        )
    else:
        net = float(requested.get("estimated_net_return_after_cost") or 0.0)
        probability = requested.get("direction_model_positive_probability")
        if probability is None:
            probability = requested.get("empirical_positive_probability")
        probability_value = float(probability) if probability is not None else 0.5
        if net <= -minimum_net or probability_value < 0.45:
            evidence_label = "证据偏负面"
            reasons.append("通过验证的成本后期望或样本外方向概率明显不利")
        elif net >= minimum_net and probability_value >= minimum_probability:
            evidence_label = "证据偏正面"
            reasons.append("指定期限模型通过滚动样本外验证，成本后期望和校准方向概率同时达到分析门槛")
        else:
            evidence_label = "证据中性"
            reasons.append("模型有效，但成本后优势或样本外方向概率没有形成明显方向")

    rsi = _round_optional(technical.get("rsi_14"))
    ret_5 = _round_optional((technical.get("returns") or {}).get("5d"))
    if evidence_label == "证据偏正面" and (
        (rsi is not None and rsi >= 80) or (ret_5 is not None and ret_5 >= 0.18)
    ):
        evidence_label = "证据中性"
        reasons.append("短线指标处于过热区，正面模型证据受到追高风险削弱")
    if evidence_label == "证据偏正面" and fundamental_flags:
        evidence_label = "证据中性"
        reasons.append("基本面存在明显风险项，短线量化证据不足以覆盖这些不确定性")
    reasons.extend(fundamental_flags)
    requested_probability = requested.get("direction_model_positive_probability") if requested else None
    if requested_probability is None and requested:
        requested_probability = requested.get("empirical_positive_probability")
    signal_gate = signal_evidence_gate(
        validation_passed=bool(metrics.get("validation_passed")),
        execution_feasible=bool((requested.get("position_and_cost") or {}).get("execution_feasible"))
        if requested
        else False,
        net_return=_round_optional(requested.get("estimated_net_return_after_cost")) if requested else None,
        positive_probability=_round_optional(requested_probability),
        quality_score=_round_optional(metrics.get("quality_score")),
        minimum_net_return=minimum_net,
        minimum_positive_probability=minimum_probability,
        minimum_quality_score=float(config.get("moxing", {}).get("abstain_min_quality_score", 0.40)),
    )
    if evidence_label == "证据偏正面" and not signal_gate["actionable_signal"]:
        evidence_label = "证据中性"
        reasons.extend(value for value in signal_gate["reasons"] if value not in reasons)
    if evidence_label != "证据偏正面":
        signal_gate["actionable_signal"] = False
        signal_gate["decision"] = "abstain"
        if not signal_gate["reasons"]:
            signal_gate["reasons"] = ["综合技术、基本面或时点约束后没有形成明确正面证据"]
    summary_detail = reasons[0] if reasons else "指定期限证据不足"
    return {
        "evidence_label": evidence_label,
        "requested_horizon": f"T+{holding_days}",
        "summary": f"{evidence_label}：{summary_detail}",
        "confidence": metrics.get("quality_label", "low") if metrics.get("validation_passed") else "insufficient",
        "reasons": reasons,
        "signal_gate": signal_gate,
        "assessment_thresholds": {
            "minimum_net_return_after_cost": minimum_net,
            "minimum_oos_direction_positive_probability": minimum_probability,
            "model_validation_must_pass": True,
            "execution_constraints_must_be_clear": True,
        },
        "scenario_timing": {
            "valid": entry_timing_valid,
            "reason": entry_timing_reason,
            "assumed_entry_date": schedule.get("assumed_entry_date"),
            "scenario_exit_date": schedule.get("scenario_exit_dates", {}).get(f"T+{holding_days}"),
        },
        "fundamental_risk_flags": fundamental_flags,
        "responsibility_note": "这是分析证据汇总，不是买入、卖出或持有指令；最终决定由用户自行作出。",
    }


def yanjiu_dangu_yuce(
    *,
    code: str,
    name: str,
    industry: str,
    target_history: pd.DataFrame,
    target_source: str,
    target_adjustment: str,
    source: str,
    signal_date: str,
    holding_days: int,
    budget_yuan: float | None,
    config: dict[str, Any],
    technical: dict[str, Any],
    fundamentals: dict[str, Any],
    tradability: dict[str, Any],
) -> dict[str, Any]:
    """Run peer selection, walk-forward modeling, cost adjustment and evidence assessment."""
    configured_budget, cost_scenario, cost_path, cost_errors = _load_cost_assumption(
        str(config.get("jiaoyi", {}).get("cost_scenario", "normal_cost"))
    )
    actual_budget = float(budget_yuan) if budget_yuan is not None else float(configured_budget)
    schedule = _next_market_sessions(signal_date)
    peer_table, peer_meta = xuanze_tonghang_yangben(
        code=code,
        name=name,
        industry=industry,
        signal_date=signal_date,
        config=config,
    )
    peer_snapshot = _peer_snapshot(peer_table, code)
    histories, names, history_meta = _fetch_peer_histories(
        peer_table=peer_table,
        target_code=code,
        target_history=target_history,
        target_source=target_source,
        target_adjustment=target_adjustment,
        signal_date=signal_date,
        source=source,
        config=config,
    )
    minimum_peers = int(config.get("dangu", {}).get("minimum_peer_stocks", 8))
    if code not in histories or len(histories) < minimum_peers:
        reason = (
            f"可用同行历史只有 {len(histories)} 只，至少需要 {minimum_peers} 只"
            if code in histories
            else "目标股票没有可用于模型的前复权历史"
        )
        history_errors = [str(value) for value in history_meta.get("errors", [])]
        unadjusted_count = sum("未复权行情" in value for value in history_errors)
        if unadjusted_count:
            reason += f"；另有 {unadjusted_count} 只股票仅取得未复权行情，不能进入模型"
            if source == "tushare":
                reason += "；严格 Tushare 模式不会自动降级到 AKShare"
        return {
            "status": "unavailable",
            "requested_horizon": f"T+{holding_days}",
            "schedule": schedule,
            "peer_universe": {**peer_meta, "history_fetch": history_meta, "relative_snapshot": peer_snapshot},
            "forecast": {},
            "future_3_trading_days": {
                "status": "unavailable",
                "signal_date": signal_date,
                "forecast": {},
                "error": reason,
            },
            "validation": {"horizons": {}, "passed_horizons": 0},
            "analysis_assessment": {
                "evidence_label": "证据不足" if tradability.get("basic_execution_feasible") else "证据偏负面",
                "requested_horizon": f"T+{holding_days}",
                "summary": (
                    f"证据不足：{reason}"
                    if tradability.get("basic_execution_feasible")
                    else "证据偏负面：存在明显可交易性约束"
                ),
                "reasons": [reason] + list(tradability.get("hard_blocks", [])),
                "signal_gate": {
                    "actionable_signal": False,
                    "decision": "abstain",
                    "reasons": [reason] + list(tradability.get("hard_blocks", [])),
                },
                "responsibility_note": "这是分析证据汇总，不是交易指令；最终决定由用户自行作出。",
            },
            "limitations": ["同行历史不足时不退化为单股票自拟合模型，也不把启发式技术分冒充预测"],
        }

    panel = goujian_moxing_shuju(histories, names, [1, 2, 3])
    panel = _add_single_stock_features(panel)
    panel, daily_factor_meta = enrich_daily_factor_panel(
        panel,
        source=source,
        include_historical_valuation=True,
    )
    target_rows = panel[
        (panel["ts_code"] == code)
        & (pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize() == pd.Timestamp(signal_date).normalize())
    ]
    if target_rows.empty:
        target_rows = panel[panel["ts_code"] == code].sort_values("trade_date").tail(1)
    if target_rows.empty:
        raise RuntimeError("模型面板中没有目标股票的最新特征")
    predictions, validation = _fit_single_stock_models(
        panel=panel,
        latest=target_rows.tail(1),
        config=config,
        budget_yuan=actual_budget,
    )
    future_schedule_error = _future_schedule_unavailable_reason(schedule, tradability)
    try:
        if future_schedule_error:
            raise RuntimeError(future_schedule_error)
        future_predictions, future_validation = _fit_future_session_models(
            panel=panel,
            latest=target_rows.tail(1),
            config=config,
        )
        future_error = ""
    except Exception as exc:
        future_predictions = pd.DataFrame()
        future_validation = {
            "horizons": {},
            "passed_horizons": 0,
            "overall_quality_score": 0.0,
            "overall_quality_label": "low",
        }
        future_error = str(exc)
    prediction_row = predictions.iloc[0]
    analysis_price = _round_optional(tradability.get("analysis_price"), 3) or _round_optional(prediction_row.get("close"), 3)
    latest_model_row = target_rows.tail(1).iloc[0]
    stock_cost_rate, position_cost = _stock_roundtrip_cost(
        code,
        float(analysis_price),
        actual_budget,
        cost_scenario,
        daily_amount_yuan=_round_optional(latest_model_row.get("amount_yuan")),
        atr_pct=_round_optional(latest_model_row.get("atr_14_pct")),
        trading_settings=config.get("jiaoyi", {}),
    )
    forecast: dict[str, Any] = {}
    for horizon in [1, 2, 3]:
        gross = _round_optional(prediction_row.get(f"pred_t{horizon}"))
        metrics = validation["horizons"].get(f"T+{horizon}", {})
        net = _apply_cost(float(gross), stock_cost_rate) if gross is not None and stock_cost_rate is not None else None
        interval = metrics.get("prediction_interval_80")
        conformal_interval = metrics.get("conformal_prediction_interval_80")
        quantile_interval = metrics.get("quantile_prediction_interval_80")
        cqr_interval = metrics.get("conformalized_quantile_prediction_interval_80")
        preferred_interval = (
            metrics.get("preferred_prediction_interval_80")
            or conformal_interval
            or interval
        )
        net_interval = [
            round(_apply_cost(float(value), stock_cost_rate), 6) for value in interval
        ] if interval and stock_cost_rate is not None else None
        conformal_net_interval = [
            round(_apply_cost(float(value), stock_cost_rate), 6) for value in conformal_interval
        ] if conformal_interval and stock_cost_rate is not None else None
        preferred_net_interval = [
            round(_apply_cost(float(value), stock_cost_rate), 6) for value in preferred_interval
        ] if preferred_interval and stock_cost_rate is not None else None
        forecast[f"T+{horizon}"] = {
            "entry_to_exit_gross_return": round(float(gross), 6) if gross is not None else None,
            "entry_to_exit_gross_return_pct": round(float(gross) * 100.0, 3) if gross is not None else None,
            "estimated_net_return_after_cost": round(float(net), 6) if net is not None else None,
            "estimated_net_return_after_cost_pct": round(float(net) * 100.0, 3) if net is not None else None,
            "empirical_positive_probability": metrics.get("latest_empirical_positive_probability"),
            "direction_model_positive_probability": metrics.get("latest_direction_positive_probability"),
            "direction_probability_method": metrics.get("direction_probability_method"),
            "empirical_return_interval_80": interval,
            "empirical_net_return_interval_80": net_interval,
            "conformal_return_interval_80": conformal_interval,
            "conformal_net_return_interval_80": conformal_net_interval,
            "quantile_return_interval_80": quantile_interval,
            "quantile_median_return": metrics.get("quantile_median_prediction"),
            "conformalized_quantile_return_interval_80": cqr_interval,
            "preferred_return_interval_80": preferred_interval,
            "preferred_net_return_interval_80": preferred_net_interval,
            "preferred_return_interval_method": metrics.get(
                "preferred_prediction_interval_method"
            ),
            "validation_passed": bool(metrics.get("validation_passed")),
            "model_quality": metrics.get("quality_label", "low"),
            "position_and_cost": position_cost,
            "timing": f"{signal_date} 收盘后信号，假设下一交易日开盘作为测算基准；比较入场后第{horizon}个可卖出交易日收盘",
            "assumed_entry_date": schedule.get("assumed_entry_date"),
            "scenario_exit_date": schedule.get("scenario_exit_dates", {}).get(f"T+{horizon}"),
            "predicted_close": None,
            "predicted_close_unavailable_reason": "入场开盘价尚未知，模型预测的是入场到退出收益，不能伪造精确目标价",
        }

    future_forecast: dict[str, Any] = {}
    signal_close = _round_optional(prediction_row.get("close"), 3)
    future_status = "ok" if not future_predictions.empty and signal_close is not None else "unavailable"
    if future_status == "ok":
        future_row = future_predictions.iloc[0]
        limit_pct = _round_optional(tradability.get("price_limit_pct"))
        limit_rate = limit_pct / 100.0 if limit_pct is not None else None
        for horizon in [1, 2, 3]:
            predicted_return = _round_optional(future_row.get(f"future_pred_t{horizon}"))
            metrics = future_validation.get("horizons", {}).get(f"T+{horizon}", {})
            interval = metrics.get("prediction_interval_80")
            conformal_interval = metrics.get("conformal_prediction_interval_80")
            quantile_interval = metrics.get("quantile_prediction_interval_80")
            cqr_interval = metrics.get("conformalized_quantile_prediction_interval_80")
            preferred_interval = (
                metrics.get("preferred_prediction_interval_80")
                or conformal_interval
                or interval
            )
            lower_price, upper_price = _price_limit_bounds(signal_close, limit_rate, horizon)
            predicted_close = None
            predicted_close_interval = None
            if predicted_return is not None:
                raw_close = signal_close * (1.0 + predicted_return)
                predicted_close = _round_price_tick(min(max(raw_close, lower_price), upper_price))
            if preferred_interval and len(preferred_interval) == 2:
                interval_prices = [signal_close * (1.0 + float(value)) for value in preferred_interval]
                predicted_close_interval = [
                    _round_price_tick(min(max(value, lower_price), upper_price))
                    for value in interval_prices
                ]
            future_forecast[f"T+{horizon}"] = {
                "target_trade_date": schedule.get("future_session_dates", {}).get(f"T+{horizon}"),
                "cumulative_return_from_signal_close": (
                    round(float(predicted_return), 6) if predicted_return is not None else None
                ),
                "cumulative_return_from_signal_close_pct": (
                    round(float(predicted_return) * 100.0, 3) if predicted_return is not None else None
                ),
                "predicted_close_reference": predicted_close,
                "predicted_close_interval_80": predicted_close_interval,
                "predicted_close_interval_method": metrics.get(
                    "preferred_prediction_interval_method",
                    "nearest_oos_empirical_quantile",
                ),
                "empirical_return_interval_80": interval,
                "conformal_return_interval_80": conformal_interval,
                "quantile_return_interval_80": quantile_interval,
                "quantile_median_return": metrics.get("quantile_median_prediction"),
                "conformalized_quantile_return_interval_80": cqr_interval,
                "preferred_return_interval_80": preferred_interval,
                "empirical_positive_probability": metrics.get("latest_empirical_positive_probability"),
                "direction_model_positive_probability": metrics.get("latest_direction_positive_probability"),
                "direction_probability_method": metrics.get("direction_probability_method"),
                "validation_passed": bool(metrics.get("validation_passed")),
                "model_quality": metrics.get("quality_label", "low"),
                "direction": (
                    "up" if predicted_return is not None and predicted_return > 0
                    else "down" if predicted_return is not None and predicted_return < 0
                    else "flat_or_unavailable"
                ),
            }

    future_three_days = {
        "status": future_status,
        "signal_date": signal_date,
        "signal_close": signal_close,
        "definition": "以最近完整收盘日为T，预测未来第1、2、3个市场交易日收盘相对T收盘的累计收益",
        "forecast": future_forecast,
        "validation": future_validation,
        "error": future_error or None,
        "interpretation": (
            "预测收盘价是模型参考值，不是目标价或成交承诺；未通过样本外验证的周期只作观察。"
        ),
    }
    analysis_assessment = _build_analysis_assessment(
        holding_days=holding_days,
        forecast=forecast,
        validation=validation,
        tradability=tradability,
        technical=technical,
        fundamentals=fundamentals,
        schedule=schedule,
        config=config,
    )
    return {
        "status": "ok",
        "requested_horizon": f"T+{holding_days}",
        "requested_holding_trading_days": holding_days,
        "schedule": schedule,
        "peer_universe": {
            **peer_meta,
            "history_fetch": history_meta,
            "relative_snapshot": peer_snapshot,
        },
        "daily_factor_data": daily_factor_meta,
        "forecast": forecast,
        "future_3_trading_days": future_three_days,
        "validation": validation,
        "analysis_assessment": analysis_assessment,
        "cost_assumption": {
            "scenario": cost_scenario.name,
            "config_path": cost_path,
            "budget_yuan": round(actual_budget, 2),
            "estimated_roundtrip_cost_rate": round(float(stock_cost_rate), 6) if stock_cost_rate is not None else None,
            "config_errors": cost_errors,
        },
        "methodology": {
            "model": "HistGradientBoostingRegressor + 稳健缩放Ridge的小型集成",
            "ensemble_weighting": "每个滚动折只使用更早折的样本外预测选权重；最终生产权重使用全部滚动样本外预测",
            "training_universe": "目标股票的当前同行优先，加少量全市场高流动性参考股票",
            "validation": "六折扩展窗口、标签跨界清除的滚动样本外验证，最后一折作为最终保留测试窗口",
            "signal": "只使用已确认收盘的日线、同日精确匹配的历史估值、市场指数日K和当日横截面特征",
            "execution_scenario": "假设下一交易日开盘作为收益测算基准，并按A股T+1比较指定T+1/T+2/T+3收盘",
            "future_forecast": "另行预测从最近完整收盘到未来第1/2/3个交易日收盘，两类结果都只用于分析",
            "llm_boundary": "数值、验证与证据标签均由程序生成；LLM只能解释，不能改写或作交易决定",
        },
        "limitations": [
            "当前同行池用于历史训练，仍存在当前成分与幸存者偏差",
            "产品永久只做日K；不能模拟集合竞价排队、盘口深度和突发公告冲击",
            "经验区间和上涨比例来自历史样本外相似预测，不是收益保证",
        ],
    }


__all__ = [
    "SINGLE_STOCK_FEATURE_COLUMNS",
    "huoqu_dangqian_kuaizhao",
    "pinggu_kejiaoyixing",
    "shichang_shizhong",
    "yanjiu_dangu_yuce",
]
