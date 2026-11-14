"""历史日线的价格口径、交易状态和成交约束。"""

from __future__ import annotations

import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PRICE_FIELDS = ("open", "high", "low", "close")
LIQUIDITY_FIELDS = ("turnover_rate", "circulating_market_value_yuan")
_LIQUIDITY_REPORT_FIELDS = (
    "trade_date", "trade_status", "raw_close", "volume", "amount_yuan", *LIQUIDITY_FIELDS,
    "reported_turnover_rate_pct", "turnover_status", "turnover_source", "turnover_volume_shares",
    "reported_circulating_market_value_yuan", "reported_valuation_close", "valuation_status", "valuation_source",
)
RAW_FIELDS = tuple(f"raw_{field}" for field in PRICE_FIELDS)
LIMIT_FIELDS = ("limit_up_price", "limit_down_price")
RULES_2026 = pd.Timestamp("2026-07-06")
EXECUTED_STATUSES = ("traded", "intraday_halt")
KNOWN_LIMIT_STATUSES = ("reported", "calculated", "no_limit")
EXCHANGE_RULES = {
    "SH": {"2026": "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml",
           "previous": "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml"},
    "SZ": {"2026": "https://docs.static.szse.cn/www/lawrules/rule/trade/W020260424690713155663.pdf",
           "previous": "https://www.szse.cn/lawrules/rule/repeal/rules/P020231230545143336292.pdf"},
}


def jiaoyi_riqi(value: Any) -> pd.Timestamp:
    """兼容 YYYYMMDD、普通日期与来源的空日期占位。"""
    if value is None or pd.isna(value) or str(value) in {"0", "0000-00-00", "00000000", ""}:
        return pd.NaT
    parsed = pd.to_datetime(str(value), errors="coerce")
    return parsed.normalize() if pd.notna(parsed) else pd.NaT


def lishi_shuzhi(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def zhengli_lishi_rili(frame: pd.DataFrame) -> pd.DataFrame:
    """按日期建立唯一索引，来源日期错误交给适配层降级。"""
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"].astype(str), errors="coerce").dt.normalize()
    if result["trade_date"].isna().any() or result["trade_date"].duplicated().any():
        raise ValueError("历史交易证据含无效或重复的交易日期")
    return result.set_index("trade_date").sort_index()


def zhankai_shengxiao_yinzi(factors: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """将来源明确给出的生效事件展开至下一事件前，逐日因子另行逐日核验。"""
    if factors.empty:
        return pd.Series(np.nan, index=dates)
    series = zhengli_lishi_rili(factors)["adj_factor"]
    values = pd.to_numeric(series, errors="coerce")
    if not (np.isfinite(values) & values.gt(0)).all():
        raise ValueError("生效事件中的复权因子无效")
    return values.reindex(dates, method="ffill")


def shi_fengxian_jingshi(name: str) -> bool:
    return bool(re.match(r"^(?:S)?(?:\*)?ST", name.upper()))


def lishi_xingming_zhuangtai(frame: pd.DataFrame, names: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按公告日与生效区间映射历史简称，保留相异来源的 ST 标记。"""
    conflicts = []
    frame["historical_name"] = None
    if not names:
        return conflicts
    for day, row in frame.iterrows():
        matching = [item for item in names if item["start_date"] <= day
                    and (pd.isna(item["end_date"]) or day <= item["end_date"])
                    and pd.notna(item["announcement_date"]) and item["announcement_date"] <= day]
        if len({item["name"] for item in matching}) > 1:
            conflicts.append({"trade_date": day.strftime("%Y-%m-%d"), "field": "historical_name",
                              "values": sorted({item["name"] for item in matching}), "source": matching[0]["source"]})
        if len(matching) != 1:
            continue
        item = matching[0]
        name = item["name"]
        is_st = shi_fengxian_jingshi(name)
        old_st = row.get("is_st")
        if pd.notna(old_st) and bool(old_st) != is_st:
            conflicts.append({"trade_date": day.strftime("%Y-%m-%d"), "field": "is_st",
                              "daily_value": bool(old_st), "daily_source": row.get("st_source"),
                              "effective_value": is_st, "effective_source": item["source"]})
            # 两份供应商资料相互冲突时，等待发行人或交易所公告确认。
            frame.loc[day, ["is_st", "st_source"]] = [None, None]
            continue
        frame.loc[day, ["historical_name", "is_st", "st_source"]] = [name, is_st, item["source"]]
        if name.startswith("退市") or name.endswith("退"):
            frame.loc[day, "is_delisting"] = True
    return conflicts


def _dingshi_jiaoyi(
    row: pd.Series, day: pd.Timestamp, sessions: pd.DatetimeIndex, profile: Mapping[str, Any],
) -> tuple[str, float | None]:
    """仅覆盖已有规则依据、能确定历史板块与例外状态的区间。"""
    board = profile.get("board")
    earliest = {
        "main_board": pd.Timestamp("2023-04-10"),
        "chinext": pd.Timestamp("2020-08-24"),
        "star": pd.Timestamp("2019-07-22"),
        "bse": pd.Timestamp("2021-11-15"),
    }.get(board)
    listed = jiaoyi_riqi(profile.get("list_date"))
    last = jiaoyi_riqi(profile.get("delist_date"))
    if earliest is None or day < earliest or pd.isna(listed) or day < listed or (pd.notna(last) and day >= last):
        return "unknown", None
    delisting_start = jiaoyi_riqi(row.get("delisting_start"))
    delisting = row.get("is_delisting")
    if pd.notna(delisting_start) and delisting is True:
        if day == delisting_start and row.get("delisting_first_session_no_limit") is True:
            return "no_limit", None
        rate = lishi_shuzhi(row.get("delisting_limit_rate"))
        return ("calculated", rate) if day > delisting_start and rate is not None else ("unknown", None)
    if board != "bse" and listed >= earliest and listed in sessions:
        first_sessions = sessions[sessions >= listed][:5]
        if day in first_sessions:
            return "no_limit", None
    if board == "bse":
        return "unknown", None
    if not row.get("ordinary_session_verified"):
        return "unknown", None
    if board != "main_board":
        return "calculated", 0.20
    if day >= RULES_2026:
        return "calculated", 0.10
    if pd.isna(row.get("is_st")):
        return "unknown", None
    return "calculated", 0.05 if row["is_st"] else 0.10


def _round_limits(reference: float, rate: float) -> tuple[float, float]:
    base = Decimal(str(reference))
    change = Decimal(str(rate))
    cent = Decimal("0.01")
    upper, lower = (value.quantize(cent, rounding=ROUND_HALF_UP) for value in (base * (1 + change), base * (1 - change)))
    return float(max(upper, base + cent)), float(max(cent, min(lower, base - cent)))


def zhengli_jiaoyi_zhengju(
    daily: pd.DataFrame,
    calendar: Iterable[pd.Timestamp],
    profile: Mapping[str, Any],
    *,
    notices: Iterable[dict[str, Any]] = (),
    names: list[dict[str, Any]] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """补齐上市区间内的交易日并保留原始行情，计算价只使用真实成交行。"""
    frame = zhengli_lishi_rili(daily)
    notices = tuple(notices)
    dates = pd.DatetimeIndex(sorted(set(calendar)))
    listed, delisted = (jiaoyi_riqi(profile.get(key)) for key in ("list_date", "delist_date"))
    expected = dates if end is None else dates[dates <= end]
    if start is not None:
        prior = expected[expected < start]
        earlier_rows = frame.index[(frame.index < start) & pd.to_numeric(frame.get("volume", pd.Series(index=frame.index, dtype=float)), errors="coerce").gt(0)]
        first = earlier_rows[-1] if len(earlier_rows) else prior[-1] if len(prior) else start
        expected = expected[expected >= first]
    if pd.notna(listed):
        expected = expected[expected >= listed]
    if pd.notna(delisted):
        expected = expected[expected < delisted]
    frame = frame.reindex(expected).rename_axis("trade_date")
    for field in (*RAW_FIELDS, "limit_reference_price", "volume", "amount_yuan", "adj_factor", *LIMIT_FIELDS):
        frame[field] = pd.to_numeric(frame.get(field, pd.Series(index=frame.index, dtype=float)), errors="coerce")
        frame[field] = frame[field].replace([np.inf, -np.inf], np.nan)
    frame["adj_factor"] = frame["adj_factor"].where(frame["adj_factor"].gt(0))
    for field in ("is_suspended", "is_st", "is_delisting", "st_source", "suspension_source", "limit_source",
                  "raw_price_source", "factor_source", "limit_reference_source"):
        frame[field] = frame.get(field, pd.Series(None, index=frame.index, dtype=object)).astype(object)
    frame["reported_is_st"] = frame["is_st"]
    frame["reported_is_suspended"] = frame["is_suspended"]
    frame["halt_intervals"] = pd.Series([()] * len(frame), index=frame.index, dtype=object)
    frame["delisting_start"] = pd.NaT
    frame["delisting_first_session_no_limit"] = False
    frame["delisting_limit_rate"] = np.nan
    frame["delisting_period_source"] = None
    conflicts = lishi_xingming_zhuangtai(frame, names or [])
    for event in sorted(notices, key=lambda value: value["start_date"]):
        event_start, event_end = event["start_date"], event["end_date"]
        covered = frame.index[(frame.index >= event_start) & (frame.index <= event_end)]
        if event["kind"] == "full_day":
            for day in covered:
                reported = frame.at[day, "reported_is_suspended"]
                if (pd.notna(reported) and not bool(reported)) or (lishi_shuzhi(frame.at[day, "volume"]) or 0) > 0:
                    conflicts.append({"trade_date": day.strftime("%Y-%m-%d"), "field": "is_suspended",
                                      "daily_value": reported, "daily_source": frame.at[day, "suspension_source"],
                                      "effective_value": True, "effective_source": event["source"]})
            frame.loc[covered, "is_suspended"] = True
            frame.loc[covered, "suspension_source"] = event["source"]
        elif event["kind"] == "intraday":
            for day in covered:
                frame.at[day, "halt_intervals"] += (event,)
        elif event["kind"] == "risk_warning":
            if event.get("previous_name") and pd.notna(event.get("announcement_date")):
                prior = frame.index[(frame.index >= event["announcement_date"]) & (frame.index < event_start)]
                frame.loc[prior, "historical_name"] = event["previous_name"]
                for day in prior:
                    _set_notice_st(frame, day, shi_fengxian_jingshi(event["previous_name"]), event["source"], conflicts)
            for day in covered:
                _set_notice_st(frame, day, event["is_st"], event["source"], conflicts)
            if event.get("new_name"):
                frame.loc[covered, "historical_name"] = event["new_name"]
        elif event["kind"] == "delisting_period":
            frame.loc[covered, "is_delisting"] = True
            frame.loc[covered, "delisting_start"] = event["start_date"]
            frame.loc[covered, "delisting_first_session_no_limit"] = event["first_session_no_limit"]
            frame.loc[covered, "delisting_limit_rate"] = event["limit_rate"]
            frame.loc[covered, "delisting_period_source"] = event["source"]
    suspended = frame["is_suspended"].eq(True)
    raw = frame.loc[:, list(RAW_FIELDS)]
    valid_price = raw.gt(0).all(axis=1)
    valid_price &= raw["raw_low"].le(raw[["raw_open", "raw_close"]].min(axis=1))
    valid_price &= raw["raw_high"].ge(raw[["raw_open", "raw_close"]].max(axis=1))
    official_halts = {event["source"] for event in notices if event["kind"] == "full_day"}
    incompatible = suspended & valid_price & frame["volume"].gt(0) & ~frame["suspension_source"].isin(official_halts)
    for day, row in frame.loc[incompatible].iterrows():
        conflicts.append({"trade_date": day.strftime("%Y-%m-%d"), "field": "is_suspended", "daily_value": True,
                          "daily_source": row["suspension_source"], "raw_price_source": row["raw_price_source"],
                          "raw_volume": row["volume"]})
    frame.loc[incompatible, ["is_suspended", "suspension_source"]] = [None, None]
    suspended &= ~incompatible
    traded = valid_price & frame["volume"].gt(0) & ~suspended & ~incompatible
    frame["trade_status"] = "unknown"
    frame["raw_bar_valid"] = traded
    frame.loc[traded, "trade_status"] = "traded"
    frame.loc[~suspended & frame["halt_intervals"].map(bool), "trade_status"] = "intraday_halt"
    frame.loc[suspended, "trade_status"] = "suspended"
    # 确认停牌可越过以找上一笔真实收盘，中间未解释的缺行会中断查找。
    groups = (~traded & ~suspended).cumsum()
    previous = frame["raw_close"].where(traded).groupby(groups).ffill().groupby(groups).shift()
    frame["raw_previous_close"] = previous.where(traded | suspended)
    frame["reported_limit_up_price"] = frame["limit_up_price"]
    frame["reported_limit_down_price"] = frame["limit_down_price"]
    frame["price_limit_status"] = "unknown"
    # 有前一市场日正常成交及已核验上市状态时，才尝试常规交易日的板块规则。
    # 复牌首日、历史身份缺失和整理期等例外留给来源原值或明确公告。
    active = profile.get("listing_status") == "listed" and pd.notna(listed) and pd.isna(delisted)
    if active:
        frame.loc[traded & frame["is_delisting"].isna(), "is_delisting"] = False
        frame.loc[traded & frame["is_delisting"].eq(False), "delisting_status_source"] = profile.get("identity_source")
    consecutive = traded.shift(1, fill_value=False)
    frame["ordinary_session_verified"] = consecutive & active & frame["is_delisting"].ne(True)
    exchange = profile["ts_code"].rsplit(".", 1)[-1]
    for day, row in frame.loc[traded | suspended].iterrows():
        upper, lower = (lishi_shuzhi(row[field]) for field in LIMIT_FIELDS)
        if traded.at[day] and upper == lower == 0 and row["limit_source"] == "rqalpha_stocks":
            frame.at[day, "price_limit_status"] = "no_limit"
        elif upper is not None and lower is not None and upper >= lower > 0:
            if not traded.at[day] or (row["raw_high"] <= upper + 1e-8 and row["raw_low"] >= lower - 1e-8):
                frame.at[day, "price_limit_status"] = "reported"
            else:
                frame.at[day, "price_limit_status"] = "conflict"
                conflicts.append({"trade_date": day.strftime("%Y-%m-%d"), "field": "price_limits",
                                  "daily_source": row["limit_source"], "daily_value": [upper, lower],
                                  "raw_range": [row["raw_low"], row["raw_high"]]})
        elif traded.at[day]:
            status, rate = _dingshi_jiaoyi(row, day, dates, profile)
            ref = lishi_shuzhi(row["limit_reference_price"])
            rule = EXCHANGE_RULES.get(exchange, {}).get("2026" if day >= RULES_2026 else "previous")
            rule_source = row.get("delisting_period_source") or rule
            if status == "no_limit":
                frame.at[day, "price_limit_status"] = status
                frame.at[day, "limit_source"] = rule_source
            elif rate is not None and ref is not None and ref > 0 and pd.notna(row.get("limit_reference_source")):
                limits = _round_limits(ref, rate)
                if row["raw_high"] <= limits[0] + 1e-8 and row["raw_low"] >= limits[1] - 1e-8:
                    frame.loc[day, list(LIMIT_FIELDS)] = limits
                    frame.at[day, "price_limit_status"] = status
                    frame.at[day, "limit_source"] = rule_source
    price_limits_known = frame["price_limit_status"].isin(("reported", "calculated"))
    frame.loc[~price_limits_known, list(LIMIT_FIELDS)] = np.nan
    frame["price_limit_exempt"] = frame["price_limit_status"].eq("no_limit").where(
        frame["price_limit_status"].isin(KNOWN_LIMIT_STATUSES), pd.NA,
    ).astype("boolean")
    factor = frame["adj_factor"].where(frame["adj_factor"].gt(0))
    anchor = factor.where(traded).dropna()
    for price, original in zip(PRICE_FIELDS, RAW_FIELDS):
        frame[price] = frame[original].where(traded) * (factor / anchor.iloc[-1] if not anchor.empty else factor)
    frame["pct_chg"] = (frame["raw_close"].where(traded) / frame["limit_reference_price"].where(lambda value: value.gt(0)) - 1) * 100
    frame["pre_close"] = frame["close"].shift()
    frame["adjusted_valid"] = traded & factor.notna()
    frame["ts_code"] = profile["ts_code"]
    frame["delist_date"] = delisted
    if start is not None:
        frame = frame.loc[frame.index >= start]
        conflicts = [item for item in conflicts if item["trade_date"] >= start.strftime("%Y-%m-%d")]
    return frame.reset_index(), conflicts


def _set_notice_st(frame: pd.DataFrame, day: pd.Timestamp, value: bool, source: str, conflicts: list[dict[str, Any]]) -> None:
    observed = frame.at[day, "is_st"]
    if pd.notna(observed) and bool(observed) != value:
        conflicts.append({"trade_date": day.strftime("%Y-%m-%d"), "field": "is_st", "daily_value": bool(observed),
                          "daily_source": frame.at[day, "st_source"], "effective_value": value, "effective_source": source})
    frame.loc[day, ["is_st", "st_source"]] = [value, source]


def shiji_jiaoyi_xianzhi(row: pd.Series, *, side: str) -> tuple[str, str]:
    """按参考买卖时点判断阻断、可作参考或未知。"""
    if row.get("trade_status") == "suspended":
        return "blocked", "confirmed_full_day_suspension"
    intervals = row.get("halt_intervals")
    timestamp = "09:30:00" if side == "buy" else "15:00:00"
    if isinstance(intervals, (list, tuple)) and any(
        event["start_time"] <= timestamp < event["resume_time"]
        or (side == "buy" and event["start_time"] <= "09:25:00" < event["resume_time"])
        or (side == "sell" and event.get("through_close"))
        for event in intervals
    ):
        return "blocked", "confirmed_intraday_suspension"
    if row.get("trade_status") not in EXECUTED_STATUSES or not row.get("raw_bar_valid", False):
        return "unknown", "trade_status_or_raw_bar_unavailable"
    limit_status = row.get("price_limit_status")
    if limit_status == "no_limit":
        return "reference", "price_limit_exemption"
    if limit_status not in {"reported", "calculated"}:
        return "unknown", "historical_limit_unavailable"
    boundary = lishi_shuzhi(row.get("limit_up_price" if side == "buy" else "limit_down_price"))
    price = lishi_shuzhi(row.get("raw_open" if side == "buy" else "raw_close"))
    other = lishi_shuzhi(row.get("raw_low" if side == "buy" else "raw_high"))
    if boundary is None or price is None or other is None:
        return "unknown", "historical_limit_or_raw_bar_unavailable"
    if abs(price - boundary) <= 1e-8:
        if abs(other - boundary) <= 1e-8:
            return "blocked", "one_price_limit_" + limit_status
        return "unknown", "auction_price_at_limit"
    return "reference", "price_limit_" + limit_status


def zuihou_shiji_chengjiao(frame: pd.DataFrame, profile: Mapping[str, Any], *, end: pd.Timestamp) -> pd.Timestamp:
    """要求摘牌已到期，并且最后有量日到摘牌前每个开市日都有停牌证据。"""
    delisted = jiaoyi_riqi(profile.get("delist_date"))
    last = frame.loc[frame["raw_bar_valid"], "trade_date"].max()
    if pd.notna(delisted) and end >= delisted and pd.notna(last):
        later = frame.loc[frame["trade_date"].gt(last)]
        if later["trade_status"].eq("suspended").all():
            return last
    return pd.NaT


def huizong_lishi_jiaoyi(frame: pd.DataFrame, source: Mapping[str, Any], *, end: pd.Timestamp) -> dict[str, Any]:
    """保留逐日面板的覆盖、状态区间、目标日原值与已核验生命周期。"""
    profile = source["profile"]
    valid = frame["raw_bar_valid"]
    traded = frame.loc[valid]
    delisted = jiaoyi_riqi(profile.get("delist_date"))
    st = frame["is_st"]
    suspensions = frame["trade_status"].eq("suspended")
    last_observed = traded["trade_date"].max()
    last_trade = jiaoyi_riqi(profile.get("last_actual_trade_date"))
    if pd.isna(last_trade):
        last_trade = zuihou_shiji_chengjiao(frame, profile, end=end)
    target = frame.loc[frame["trade_date"].eq(end)]
    current = target.iloc[0].to_dict() if not target.empty else {"trade_date": end, "trade_status": "unknown"}
    if pd.notna(delisted) and end >= delisted:
        current["trade_status"] = "delisted"
    elif pd.notna(profile.get("list_date")) and end < profile["list_date"]:
        current["trade_status"] = "not_listed"
    keys = ("trade_date", *RAW_FIELDS, "raw_previous_close", "limit_reference_price", "adj_factor", "volume", "amount_yuan",
            "trade_status", "is_st", "reported_is_st", "historical_name", "is_delisting", "price_limit_status",
            *LIMIT_FIELDS, "raw_price_source", "factor_source", "limit_reference_source", "st_source", "limit_source",
            "suspension_source", "limit_quote_time", "source_ts_code", "delisting_status_source")
    range_rows = []
    for values, kind in ((suspensions, "full_day"), (st.eq(True), "risk_warning")):
        series = frame.loc[values, "trade_date"].groupby((~values).cumsum()[values])
        for _, days in series:
            record = {"kind": kind, "start_date": days.iloc[0], "end_date": days.iloc[-1], "sessions": len(days)}
            if kind == "full_day":
                following = frame.loc[frame["trade_date"].gt(days.iloc[-1])].head(1)
                record["observed_resume_date"] = following.iloc[0]["trade_date"] if following["raw_bar_valid"].any() else pd.NaT
            range_rows.append(record)
    range_rows.sort(key=lambda item: item["start_date"])
    counts = {"expected_sessions": len(frame), "raw_price_sessions": int(valid.sum()),
              "factor_sessions": int(traded["adj_factor"].gt(0).sum()), "st_known_sessions": int(st.notna().sum()),
              "amount_sessions": int(traded["amount_yuan"].ge(0).sum()),
              "full_day_suspensions": int(suspensions.sum()),
              "unknown_sessions": int((~valid & ~suspensions).sum()),
              "reported_limit_sessions": int(traded["price_limit_status"].eq("reported").sum()),
              "calculated_limit_sessions": int(traded["price_limit_status"].eq("calculated").sum()),
              "no_limit_sessions": int(traded["price_limit_status"].eq("no_limit").sum()),
              "unknown_limit_sessions": int((~traded["price_limit_status"].isin(KNOWN_LIMIT_STATUSES)).sum())}
    gaps = []
    if counts["unknown_sessions"] or current["trade_status"] == "unknown":
        gaps.append("部分交易日缺少原始价量或全天停牌证据")
    if counts["factor_sessions"] < counts["raw_price_sessions"]:
        gaps.append("部分真实成交日缺少同源复权因子")
    if counts["amount_sessions"] < counts["raw_price_sessions"]:
        gaps.append("部分真实成交日缺少真实成交额")
    if counts["st_known_sessions"] < counts["expected_sessions"]:
        gaps.append("部分历史交易日的 ST 状态未核验")
    if counts["unknown_limit_sessions"]:
        gaps.append("部分真实成交日的涨跌停价或限价状态未核验")
    if pd.isna(profile.get("list_date")) or (profile.get("listing_status") != "listed" and pd.isna(delisted)):
        gaps.append("上市或摘牌日期证据不足")
    if any(
        item.get("field") == "historical_name"
        or (item.get("field") == "is_st" and not str(item.get("effective_source")).startswith("https://"))
        for item in source.get("conflicts", ())
    ):
        gaps.append("部分来源的历史简称或风险警示记录存在冲突，详见原值")
    if any(item["status"] == "unavailable" for item in source.get("attempts", ()) if str(item["source"]).startswith("https://")):
        gaps.append("部分指定公告未成功核验")
    liquidity = source.get("liquidity")
    if liquidity:
        recent = frame.loc[frame["trade_date"].between(liquidity["start_date"], liquidity["end_date"])]
        turnover_rows, valuation_rows = (int(recent[field].notna().sum()) for field in LIQUIDITY_FIELDS)
        traded_rows = int(recent["raw_bar_valid"].sum())
        suspended_rows = int(recent["trade_status"].eq("suspended").sum())
        unverified_rows = len(recent) - traded_rows - suspended_rows
        liquidity = {**liquidity, "traded_sessions": traded_rows, "suspended_sessions": suspended_rows,
                     "unverified_trade_sessions": unverified_rows, "turnover_verified_sessions": turnover_rows,
                     "circulating_value_verified_sessions": valuation_rows,
                     "status": "unavailable" if not (turnover_rows or valuation_rows)
                     else "partial" if unverified_rows or min(turnover_rows, valuation_rows) < traded_rows else "ok",
                     "series": recent[list(_LIQUIDITY_REPORT_FIELDS)].to_dict("records")}
        if turnover_rows < traded_rows:
            gaps.append("历史流动性窗口内部分成交日缺少已核验换手率")
        if valuation_rows < traded_rows:
            gaps.append("历史流动性窗口内部分成交日缺少已核验流通市值")
    return {
        "status": "unavailable" if not valid.any() and not suspensions.any() and current["trade_status"] == "unknown"
        else "partial" if gaps else "ok",
        "profile": {**profile, "last_observed_trade_date": last_observed, "last_actual_trade_date": last_trade},
        "coverage": counts, "target_session": {key: current.get(key) for key in keys},
        "last_observed_session": {key: traded.iloc[-1].get(key) for key in keys} if not traded.empty and last_observed != end else None,
        "status_intervals": range_rows, "missing_reasons": gaps,
        "name_intervals": source.get("name_intervals", []), "corporate_actions": source.get("corporate_actions", []),
        "notice_events": source.get("notice_events", []), "conflicts": source.get("conflicts", []),
        "unconfirmed_suspension_records": source.get("unconfirmed_suspension_records", []),
        "intraday_coverage": "exchange_notices_only",
        **({"historical_liquidity": liquidity} if liquidity else {}),
    }
