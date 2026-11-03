"""固定24价量状态及可选96环境状态的次日涨幅估计；不训练、不取数、不落盘。

历史事件为同源前复权信号日收盘至下一权威交易日收盘的价格涨幅，
不扣交易费用、不代表可实现的交易收益。历史频率不是未来保证。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ashare.yinzi_gongcheng import daily_atr, interval_return


MIN_KNOWN_SIGNAL_DATES = 10
SIGNAL_STRIDE = 1
STATE_WINDOW = 20
SCORE_LOOKBACK_SESSIONS = 120
RECENCY_HALF_LIFE_SESSIONS = 20
VALIDATION_LOOKBACK_SESSIONS = 60
_BAR_COLUMNS = ("open", "high", "low", "close", "volume", "amount_yuan")
_RAW_COLUMNS = ("ret_1", "ret_5", "ma20", "atr14", "atr_distance", "amount_ratio_5_15")
_DIRECTIONS = ("up_up", "up_nonup", "nonup_up", "nonup_nonup")
_POSITIONS = ("at_or_below_ma20", "above_ma20_within_2_atr", "above_ma20_over_2_atr")
_AMOUNT_STATES = ("increased", "not_increased")
_MARKET_STATES = ("market_up", "market_nonup")
_INDUSTRY_STATES = ("industry_outperforming", "industry_not_outperforming")
_ENVIRONMENT_RETURN_COLUMNS = ("market_ret_5", "market_ret_20", "industry_ret_5", "industry_ret_20")
_ENVIRONMENT_RAW_COLUMNS = (*_ENVIRONMENT_RETURN_COLUMNS, "industry_excess_ret_5", "industry_excess_ret_20",
    "stock_ret_5", "stock_ret_20", "stock_excess_ret_5", "stock_excess_ret_20")


def _dates(values: Iterable) -> pd.DatetimeIndex:
    sequence = pd.Series(list(values), dtype=object)
    numeric_dates = sequence.map(lambda value: isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)))
    sequence.loc[numeric_dates] = sequence.loc[numeric_dates].astype(str)
    result = pd.to_datetime(sequence, errors="coerce")
    if isinstance(result.dtype, pd.DatetimeTZDtype):
        result = result.dt.tz_localize(None)
    return pd.DatetimeIndex(result).normalize()


def _calendar(trading_calendar: Iterable, as_of: Any) -> pd.DatetimeIndex:
    if isinstance(trading_calendar, pd.DataFrame):
        frame = trading_calendar
        if "is_open" in frame:
            frame = frame[pd.to_numeric(frame["is_open"], errors="coerce").eq(1)]
        column = "cal_date" if "cal_date" in frame else "trade_date"
        trading_calendar = frame[column] if column in frame else []
    cutoff = _dates([as_of])[0]
    if pd.isna(cutoff):
        raise ValueError("as_of必须是有效的已完成分析日期")
    dates = _dates(trading_calendar).dropna().unique().sort_values()
    return dates[dates <= cutoff]


def _history(history: pd.DataFrame, dates: pd.DatetimeIndex | None, as_of: Any = None) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or history.empty or "trade_date" not in history:
        return pd.DataFrame(columns=_BAR_COLUMNS, index=dates if dates is not None else pd.DatetimeIndex([]))
    frame = history.copy()
    if "ts_code" in frame and frame["ts_code"].dropna().nunique() > 1:
        raise ValueError("价量状态必须逐股计算，不能跨证券混合历史")
    frame["trade_date"] = _dates(frame["trade_date"])
    frame = frame.dropna(subset=["trade_date"])
    for column in _BAR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    duplicated = frame["trade_date"].duplicated(keep=False)
    frame.loc[duplicated, list(_BAR_COLUMNS)] = np.nan
    frame = frame.drop_duplicates("trade_date").set_index("trade_date").sort_index()
    frame = frame.loc[:, list(_BAR_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    if as_of is not None:
        cutoff = _dates([as_of])[0]
        if pd.isna(cutoff):
            raise ValueError("as_of必须是有效的已完成分析日期")
        frame = frame.loc[frame.index <= cutoff]
    return frame.reindex(dates) if dates is not None else frame


def _state_series(frame: pd.DataFrame) -> pd.DataFrame:
    """仅使用每行及此前输入；日历缺行与非法OHLC使连续窗口重新预热。"""
    close, high, low, opening = (frame[column] for column in ("close", "high", "low", "open"))
    price_valid = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    price_valid &= low.le(opening) & opening.le(high) & low.le(close) & close.le(high)
    close = close.where(price_valid)
    amount = frame["amount_yuan"].where(frame["amount_yuan"].ge(0))
    complete = price_valid.rolling(STATE_WINDOW, min_periods=STATE_WINDOW).sum().eq(STATE_WINDOW)
    complete &= amount.rolling(STATE_WINDOW, min_periods=STATE_WINDOW).count().eq(STATE_WINDOW)
    atr = daily_atr(high.where(price_valid), low.where(price_valid), close)
    ma20 = close.rolling(20, min_periods=20).mean()
    recent = amount.rolling(5, min_periods=5).mean()
    earlier = amount.shift(5).rolling(15, min_periods=15).mean()
    values = pd.DataFrame({"ret_1": close / close.shift(1) - 1,
        "ret_5": close / close.shift(5) - 1, "ma20": ma20, "atr14": atr,
        "atr_distance": (close - ma20) / atr.where(atr.gt(0)),
        "amount_ratio_5_15": recent.where(recent.gt(0)) / earlier.where(earlier.gt(0))}, index=frame.index)
    values = values.replace([np.inf, -np.inf], np.nan)
    values.loc[~complete, :] = np.nan
    return values


def _state(row: pd.Series | None, date: Any, calendar_verified: bool) -> dict[str, Any]:
    raw = {column: float(row[column]) if row is not None and pd.notna(row[column]) else None for column in _RAW_COLUMNS}
    missing = [f"{column}:missing_or_invalid_continuous_input" for column, value in raw.items() if value is None]
    result: dict[str, Any] = {"status": "unavailable" if missing else "success", "state": None,
        "state_key": None, "raw_values": raw, "missing_reasons": missing,
        "analysis_date": pd.Timestamp(date).strftime("%Y-%m-%d") if date is not None else None,
        "calendar_verified": calendar_verified, "window": STATE_WINDOW}
    if not missing:
        direction = ("up" if raw["ret_5"] > 0 else "nonup") + "_" + ("up" if raw["ret_1"] > 0 else "nonup")
        position = _POSITIONS[0 if raw["atr_distance"] <= 0 else 1 if raw["atr_distance"] <= 2 else 2]
        amount_change = "increased" if raw["amount_ratio_5_15"] > 1 else "not_increased"
        key = f"{direction}|{position}|{amount_change}"
        result.update(state={"id": key, "direction": direction, "position": position, "amount_change": amount_change}, state_key=key)
    return result


def jisuan_dangqian_zhuangtai(history: pd.DataFrame, trading_calendar: Iterable | None = None,
    as_of: Any = None) -> dict[str, Any]:
    """计算当前固定24状态；正式调用提供权威日历与完整分析日以核验缺行。"""
    if trading_calendar is not None and as_of is None:
        raise ValueError("提供交易日历时必须明确as_of")
    dates = _calendar(trading_calendar, as_of) if trading_calendar is not None else None
    if dates is not None and len(dates) and dates[-1] != _dates([as_of])[0]:
        raise ValueError("as_of必须属于提供的权威交易日历，不能用最近日期替换")
    frame = _history(history, dates, as_of)
    if frame.empty:
        return _state(None, None, trading_calendar is not None)
    values = _state_series(frame)
    return _state(values.iloc[-1], frame.index[-1], trading_calendar is not None)


def _environment_series(environment: pd.DataFrame | None, frame: pd.DataFrame,
    dates: pd.DatetimeIndex) -> pd.DataFrame:
    """指数收益两端由数据层核验；这里拒绝重复日期并按权威日历逐日对齐。"""
    columns = ("industry_code", "industry_name", "membership_verified", "membership_usable",
        "membership_approximate", "membership_basis", "membership_reference_date", *_ENVIRONMENT_RETURN_COLUMNS)
    if isinstance(environment, pd.DataFrame) and "trade_date" in environment:
        source = environment.copy()
        source["trade_date"] = _dates(source["trade_date"])
        source = source.dropna(subset=["trade_date"])
        duplicate_dates = source["trade_date"].duplicated(keep=False)
        source = source.reindex(columns=["trade_date", *columns]).astype({column: object for column in columns})
        # 日期重复不能任意取一条，也不以相邻日期填补。
        source.loc[duplicate_dates, list(columns)] = None
        source = source.drop_duplicates("trade_date").set_index("trade_date")
        result = source.reindex(dates).copy()
    else:
        result = pd.DataFrame(index=dates, columns=columns)
    for column in _ENVIRONMENT_RETURN_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        result[column] = values.where(values.gt(-1))
    for window in (5, 20):
        result[f"stock_ret_{window}"] = interval_return(frame["close"], window)
        result[f"industry_excess_ret_{window}"] = result[f"industry_ret_{window}"] - result[f"market_ret_{window}"]
        result[f"stock_excess_ret_{window}"] = result[f"stock_ret_{window}"] - result[f"industry_ret_{window}"]
    return result.replace([np.inf, -np.inf], np.nan)


def _environment_state(row: pd.Series | None, date: Any) -> dict[str, Any]:
    raw = {column: float(row[column]) if row is not None and pd.notna(row[column]) else None
           for column in _ENVIRONMENT_RAW_COLUMNS}
    industry_code = row.get("industry_code") if row is not None else None
    industry_name = row.get("industry_name") if row is not None else None
    industry_code = industry_code.strip() if isinstance(industry_code, str) and industry_code.strip() else None
    industry_name = industry_name.strip() if isinstance(industry_name, str) and industry_name.strip() else None
    membership = row.get("membership_verified") if row is not None else None
    verified = isinstance(membership, (bool, np.bool_)) and bool(membership)
    usable_value = row.get("membership_usable") if row is not None else None
    usable = bool(usable_value) if isinstance(usable_value, (bool, np.bool_)) else verified
    approximate_value = row.get("membership_approximate") if row is not None else None
    approximate = bool(approximate_value) if isinstance(approximate_value, (bool, np.bool_)) else False
    reference_date = row.get("membership_reference_date") if row is not None else None
    basis = row.get("membership_basis") if row is not None else None
    required = ("market_ret_20", "industry_ret_20", "industry_excess_ret_20", "stock_ret_20", "stock_excess_ret_20")
    missing = [f"environment:{column}:missing_or_invalid_return_input" for column in required if raw[column] is None]
    if industry_code is None:
        missing.append("environment:industry_code_missing")
    if not usable:
        missing.append("environment:membership_reference_missing")
    result = {"status": "unavailable" if missing else "success", "state_key": None, "state": None,
        "analysis_date": pd.Timestamp(date).strftime("%Y-%m-%d") if date is not None else None,
        "industry_code": industry_code, "industry_name": industry_name, "membership_verified": verified,
        "membership_usable": usable, "membership_approximate": approximate,
        "membership_basis": basis if isinstance(basis, str) else "verified_signal_date_reference" if verified else "unavailable",
        "membership_reference_date": reference_date if isinstance(reference_date, str) else None,
        "point_in_time_verified": False,
        "raw_values": raw, "missing_reasons": missing,
        "supplemental_missing_fields": [column for column in _ENVIRONMENT_RAW_COLUMNS if column.endswith("_5") and raw[column] is None],
        "relative_return_unit": "return_fraction_difference", "stock_window_sessions": 21,
        "stock_return_required_closes": 2,
        "index_window_validation": "provided_by_environment_data_adapter",
        "return_missing_policy": "valid_endpoints_on_authoritative_calendar; no_fill_or_calendar_compression"}
    if not missing:
        market = _MARKET_STATES[0 if raw["market_ret_20"] > 0 else 1]
        industry = _INDUSTRY_STATES[0 if raw["industry_excess_ret_20"] > 0 else 1]
        result.update(state_key=f"{market}|{industry}", state={"market_direction": market,
            "industry_relative_strength": industry})
    return result


def _joint_state(individual: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    valid = individual["status"] == "success" and environment["status"] == "success"
    key = f"{individual['state_key']}|{environment['state_key']}" if valid else None
    return {**individual, "individual_state_key": individual["state_key"], "individual_state": individual["state"],
        "current_environment": environment, "status": "success" if valid else "unavailable", "state_key": key,
        "state": {**individual["state"], **environment["state"], "id": key} if valid else None,
        "missing_reasons": [*individual["missing_reasons"], *environment["missing_reasons"]]}


def _reference_limitations(row: np.ndarray) -> list[str]:
    reasons: list[str] = []
    if not np.isfinite(row[4:]).all() or (row[4:] < 0).any():
        reasons.append("target:turnover_missing_or_invalid")
    elif row[4] == 0 and row[5] == 0:
        reasons.append("target:observed_zero_volume_and_amount")
    elif row[4] == 0 or row[5] == 0:
        reasons.append("target:volume_amount_inconsistent_zero")
    return reasons


def _event(code: str, key: str, bars: np.ndarray, position: int, dates: pd.DatetimeIndex) -> dict[str, Any]:
    exit_pos = position + 1
    baseline, target = bars[position, 3], bars[exit_pos, 3]
    flags = _reference_limitations(bars[exit_pos])
    known = bool(np.isfinite([baseline, target]).all() and baseline > 0 and target > 0)
    change = float(target / baseline - 1) if known else None
    missing = [] if known else ["baseline_or_next_session_close_missing_or_invalid"]
    if known and not math.isfinite(change):
        known, change = False, None
        missing = ["reference_return_nonfinite_after_price_ratio"]
    return {"ts_code": code, "state_key": key,
        "signal_date": dates[position].strftime("%Y-%m-%d"),
        "signal_session_index": position, "exit_session_index": exit_pos,
        "baseline_date": dates[position].strftime("%Y-%m-%d"), "target_date": dates[exit_pos].strftime("%Y-%m-%d"),
        "status": "reference_known" if known else "reference_unknown", "price_up": bool(change > 0) if known else None,
        "next_day_return": change,
        "baseline_close": float(baseline) if np.isfinite(baseline) and baseline > 0 else None,
        "target_close": float(target) if np.isfinite(target) and target > 0 else None,
        "price_basis": "same_source_qfq_close_ratio", "costs_applied": False,
        "execution_status": "limitations_observed" if flags else "daily_reference_only",
        "execution_limitations": flags, "execution_verified": False,
        "missing_reasons": missing}


def _stable_mean(values: list[float], weights: list[float] | None = None) -> float | None:
    """先按样本绝对最大值缩放，避免有限输入累计时溢出。"""
    if weights is not None and (len(weights) != len(values) or any(not math.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError("样本权重必须与观测一一对应且为有限正数")
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if not scale:
        return 0.0
    if weights is None:
        normalized = math.fsum(value / scale for value in values) / len(values)
    else:
        normalized = math.fsum(value / scale * w for value, w in zip(values, weights)) / math.fsum(weights)
    return scale * min(1.0, max(-1.0, normalized))


def _recent_events(events: list[dict[str, Any]], as_of_position: int) -> list[dict[str, Any]]:
    """最近120个交易日的信号且已到期；年龄由权威交易日序号计算。"""
    return [{**event, "signal_age_sessions": as_of_position - event["signal_session_index"],
             "recency_weight": 2.0 ** (-(as_of_position - event["signal_session_index"]) / RECENCY_HALF_LIFE_SESSIONS)}
            for event in events
            if 0 <= as_of_position - event["signal_session_index"] < SCORE_LOOKBACK_SESSIONS
            and event["exit_session_index"] <= as_of_position]


def _observed(events: list[dict[str, Any]]) -> dict[str, Any]:
    known = [event for event in events if event["status"] == "reference_known"]
    gains = [event for event in known if event["price_up"]]
    losses = [event for event in known if not event["price_up"]]
    n, m, k = len(known), len(events) - len(known), len(gains)
    weights = [event["recency_weight"] for event in known]
    known_weight = math.fsum(weights)
    up_weight = math.fsum(event["recency_weight"] for event in gains)
    unknown_weight = math.fsum(event["recency_weight"] for event in events if event["status"] != "reference_known")
    total_weight = known_weight + unknown_weight
    date_weights: dict[str, float] = {}
    for event in known:
        date_weights[event["signal_date"]] = date_weights.get(event["signal_date"], 0.0) + event["recency_weight"]
    gross = _stable_mean([event["next_day_return"] for event in known], weights)
    p = up_weight / known_weight if known_weight else None
    gain = _stable_mean([event["next_day_return"] for event in gains], [event["recency_weight"] for event in gains])
    loss = _stable_mean([-event["next_day_return"] for event in losses], [event["recency_weight"] for event in losses])
    decomposed = p * (gain if gain is not None else 0) - (1 - p) * (loss if loss is not None else 0) if n else None
    score = 10000 * gross if gross is not None else None
    score_finite = score is None or math.isfinite(score)
    return {"sample_count": len(events), "known_count": n, "unknown_count": m, "up_count": k,
        "non_up_count": n - k, "up_probability": p, "conditional_upside_mean": gain,
        "conditional_downside_mean": loss, "expected_gross_return": gross,
        "decomposed_expected_gross_return": decomposed, "expected_next_day_return": gross,
        "composite_score_bps": score if score_finite else None,
        "numeric_range_reasons": [] if score_finite else ["composite_score_bps_exceeds_finite_float_range"],
        "known_weight_sum": known_weight, "unknown_weight_sum": unknown_weight,
        "effective_event_count": known_weight ** 2 / math.fsum(w * w for w in weights) if weights else None,
        "effective_signal_date_count": known_weight ** 2 / math.fsum(w * w for w in date_weights.values()) if date_weights else None,
        "probability_missing_bounds": [up_weight / total_weight, (up_weight + unknown_weight) / total_weight] if total_weight else None,
        "return_coverage": n / (n + m) if n + m else None}


def _uncertainty(events: list[dict[str, Any]], signal_dates: list[str], supported: bool) -> dict[str, Any]:
    """按完整信号日期槽聚集重抽样；区间只描述已观测样本的不确定性。"""
    result: dict[str, Any] = {"method": "recency_weighted_circular_signal_date_block_bootstrap", "block_signal_date_slots": 4,
        "repetitions": 2000, "seed": 20261017, "valid_repetitions": 0,
        "up_probability_95_interval": None, "expected_next_day_return_95_interval": None,
        "status": "unavailable", "missing_reasons": [], "future_probability_calibrated": False,
        "interpretation": "exploratory_observed_sample_distribution_not_future_guarantee"}
    if not supported:
        result["missing_reasons"] = ["fewer_than_ten_distinct_known_signal_dates"]
        return result
    known_returns = [event["next_day_return"] for event in events if event["status"] == "reference_known"]
    net_scale = max(abs(value) for value in known_returns) or 1.0
    aggregates = {date: [0., 0., 0.] for date in signal_dates}
    for event in events:
        if event["status"] == "reference_known":
            row = aggregates[event["signal_date"]]
            weight = event["recency_weight"]
            row[0] += weight
            row[1] += weight * int(event["price_up"])
            row[2] += weight * event["next_day_return"] / net_scale
    arrays = np.array(list(aggregates.values()), dtype=float)
    rng, count = np.random.default_rng(20261017), len(arrays)
    starts = rng.integers(0, count, size=(2000, math.ceil(count / 4)))
    locations = ((starts[:, :, None] + np.arange(4)) % count).reshape(2000, -1)[:, :count]
    sums = arrays[locations].sum(axis=1)
    sums = sums[sums[:, 0] > 0]
    frequencies = sums[:, 1] / sums[:, 0]
    returns = np.clip(sums[:, 2] / sums[:, 0], -1, 1) * net_scale
    result.update(status="exploratory", valid_repetitions=len(sums))
    known_labels = {event["price_up"] for event in events if event["status"] == "reference_known"}
    if len(known_labels) == 1:
        result["missing_reasons"].append("probability_interval_degenerate_boundary_all_known_labels_equal")
    elif np.ptp(frequencies) == 0:
        result["missing_reasons"].append("probability_interval_degenerate_empirical_resamples")
    else:
        result["up_probability_95_interval"] = np.quantile(frequencies, [.025, .975]).tolist()
    if min(known_returns) == max(known_returns) or np.ptp(returns) == 0:
        result["missing_reasons"].append("next_day_return_interval_degenerate_empirical_resamples")
    else:
        result["expected_next_day_return_95_interval"] = np.quantile(returns, [.025, .975]).tolist()
    return result


def _walk_forward(events: list[dict[str, Any]], positions: list[int], dates: pd.DatetimeIndex,
    codes: list[str], *, reference_events: list[dict[str, Any]] | None = None,
    environment_missing_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """逐股逐日配对；估计只用当时已到期的同状态样本。"""
    checks = {code: [] for code in codes}
    targets = {(event["ts_code"], event["signal_session_index"]): event
               for event in events if event["ts_code"] in checks}
    references = {(event["ts_code"], event["signal_session_index"]): event
                  for event in (reference_events if reference_events is not None else events) if event["ts_code"] in checks}
    environment_missing = {(event["ts_code"], event["signal_session_index"]): event
                           for event in (environment_missing_events or []) if event["ts_code"] in checks}
    for position in positions:
        current_targets = {code: targets.get((code, position)) for code in codes}
        states = {target["state_key"] for target in current_targets.values() if target}
        groups = {key: [] for key in states}
        for event in _recent_events(events, position):
            if event["state_key"] in groups:
                groups[event["state_key"]].append(event)
        estimates = {key: _statistics(items, []) for key, items in groups.items()}
        for code, target in current_targets.items():
            reference = references.get((code, position))
            missing_environment = environment_missing.get((code, position))
            stats = estimates.get(target["state_key"], {}) if target else {}
            predicted = stats.get("expected_next_day_return") if stats.get("status") == "success" else None
            realized = reference["next_day_return"] if reference and reference["status"] == "reference_known" else None
            sample = groups.get(target["state_key"], []) if target else []
            checks[code].append({
                "signal_date": dates[position].strftime("%Y-%m-%d"),
                "target_date": dates[position + 1].strftime("%Y-%m-%d"),
                "state_key": target["state_key"] if target else None,
                "individual_state_key": reference["state_key"] if reference else None,
                "environment_status": ("unavailable" if not target or missing_environment else "success")
                    if reference_events is not None else "not_requested",
                "status": "available" if predicted is not None and realized is not None else "unavailable",
                "training_known_signal_dates": stats.get("known_signal_dates"),
                "training_effective_signal_date_count": stats.get("effective_signal_date_count"),
                "training_latest_exit_date": max((event["target_date"] for event in sample), default=None),
                "estimated_next_day_return": predicted, "realized_next_day_return": realized,
                "prediction_error": predicted - realized if predicted is not None and realized is not None else None,
                "evaluation_known_events": int(realized is not None),
                "evaluation_unknown_events": int(reference is not None and realized is None),
                "environment_missing_events": int(missing_environment is not None),
                "missing_reasons": ([] if predicted is not None else ["training_sample_unsupported", *stats.get("missing_reasons", [])])
                    + ([] if realized is not None else ["evaluation_reference_unavailable"])
                    + (missing_environment["environment_missing_reasons"] if missing_environment else []),
            })
    results = {}
    for code, rows in checks.items():
        valid = [row for row in rows if row["status"] == "available"]
        results[code] = {
            "status": "exploratory" if valid else "unavailable", "validation_lookback_sessions": VALIDATION_LOOKBACK_SESSIONS,
            "checked_signal_dates": len(rows), "available_signal_dates": len(valid),
            "environment_missing_signal_dates": sum(row["environment_missing_events"] for row in rows),
            "mean_estimated_next_day_return": _stable_mean([row["estimated_next_day_return"] for row in valid]),
            "mean_realized_next_day_return": _stable_mean([row["realized_next_day_return"] for row in valid]),
            "mean_prediction_error": _stable_mean([row["prediction_error"] for row in valid]),
            "mean_absolute_error": _stable_mean([abs(row["prediction_error"]) for row in valid]),
            "date_checks": rows, "parameters_tuned": False, "future_return_validated": False,
            "interpretation": "按当前入选股票追溯，只用当时已成熟同状态样本估计下一交易日收盘相对信号日收盘的涨幅，不扣费用；各日期等权汇总。包含事后选股、当前存续及资格池偏差，不代替事前预测与次日实际涨幅核验，不证明未来有效。",
        }
    return results


def _statistics(events: list[dict[str, Any]], signal_dates: list[str], *, detailed: bool = False) -> dict[str, Any]:
    observed = _observed(events)
    known_dates = sorted({event["signal_date"] for event in events if event["status"] == "reference_known"})
    event_dates = sorted({event["signal_date"] for event in events})
    date_events: dict[str, list[dict[str, Any]]] = {date: [] for date in event_dates}
    for event in events:
        date_events[event["signal_date"]].append(event)
    supported = len(known_dates) >= MIN_KNOWN_SIGNAL_DATES
    estimates = ("up_probability", "conditional_upside_mean", "conditional_downside_mean", "expected_gross_return", "expected_next_day_return", "composite_score_bps")
    ready = supported and not observed["numeric_range_reasons"]
    result = {**observed, **{key: observed[key] if ready else None for key in estimates},
        "status": "success" if ready else "unavailable", "observed_summary": observed,
        "known_signal_dates": len(known_dates), "date_count": len(known_dates), "signal_dates": len(event_dates),
        "sample_support": {"supported": supported, "minimum_distinct_known_signal_dates": MIN_KNOWN_SIGNAL_DATES,
            "meaning": "engineering_support_gate_not_return_effectiveness_or_calibration"},
        "missing_reasons": ([] if supported else ["fewer_than_ten_distinct_known_signal_dates"]) + observed["numeric_range_reasons"],
        "uncertainty": _uncertainty(events, signal_dates, ready) if detailed else None,
        "date_summaries": [{"signal_date": date, **_observed(items),
            "unknown_reason_counts": dict(Counter(reason for event in items for reason in event["missing_reasons"])),
            "execution_limitation_counts": dict(Counter(reason for event in items for reason in event["execution_limitations"]))}
            for date, items in date_events.items()] if detailed else [],
        "event_counts": dict(Counter(event["status"] for event in events)),
        "execution_limitation_counts": dict(Counter(reason for event in events for reason in event["execution_limitations"])),
        "unknown_reason_counts": dict(Counter(reason for event in events for reason in event["missing_reasons"])),
        "execution_verified": False, "reference_only": True, "future_probability_calibrated": False,
        "signal_start": event_dates[0] if event_dates else None, "signal_end": event_dates[-1] if event_dates else None,
        "known_signal_start": known_dates[0] if known_dates else None, "known_signal_end": known_dates[-1] if known_dates else None,
        "chronological_subperiods": []}
    for part in np.array_split(np.array(event_dates if detailed else [], dtype=object), 4):
        if len(part):
            subset = [event for date in part for event in date_events[date]]
            result["chronological_subperiods"].append({"start": part[0], "end": part[-1],
                "known_signal_dates": len({event["signal_date"] for event in subset if event["status"] == "reference_known"}), **_observed(subset)})
    return result


def _candidate_estimate(code: str, state: dict[str, Any], statistics: dict[str, dict[str, Any]],
    cutoff: pd.Timestamp, baseline_close: float | None) -> dict[str, Any]:
    # 共享状态点估计，但逐股检查属于各自报告，不能写回共享字典。
    stats = dict(statistics[state["state_key"]]) if state["state_key"] in statistics else None
    reasons = list(state["missing_reasons"]) + (stats["missing_reasons"] if stats else [])
    success = state["status"] == "success" and stats is not None and stats["status"] == "success"
    return {
        **state, "ts_code": code, "status": "success" if success else "unavailable", "statistics": stats,
        "recommendation_index": stats["composite_score_bps"] if success else None,
        **{key: stats[key] if success else None for key in
           ("up_probability", "conditional_upside_mean", "conditional_downside_mean", "expected_next_day_return")},
        "prediction_reference": {"baseline_date": cutoff.strftime("%Y-%m-%d"), "baseline_close": baseline_close,
            "target_session_offset": 1, "price_basis": "same_source_qfq_close_ratio", "costs_applied": False,
            "formula": "next_session_close / signal_session_close - 1"},
        "missing_reasons": reasons,
    }


def goujian_tongzhuangtai_tongji(histories: dict[str, pd.DataFrame], trading_calendar: Iterable,
    as_of: Any, *, environment_by_code: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """None保留24状态；传环境映射启用96状态，缺失不自动回退旧估计。"""
    if not isinstance(histories, dict):
        raise TypeError("histories必须是证券代码到单股DataFrame的映射")
    if environment_by_code is not None and not isinstance(environment_by_code, dict):
        raise TypeError("environment_by_code必须是证券代码到逐日环境DataFrame的映射或None")
    use_environment = environment_by_code is not None
    dates = _calendar(trading_calendar, as_of)
    cutoff = _dates([as_of])[0]
    if len(dates) and dates[-1] != cutoff:
        raise ValueError("as_of必须属于提供的权威交易日历，不能用最近日期替换")
    scheduled = list(range(STATE_WINDOW - 1, len(dates), SIGNAL_STRIDE))
    mature = [position for position in scheduled if position + 1 < len(dates)]
    recent_mature = [position for position in mature if len(dates) - 1 - position < SCORE_LOOKBACK_SESSIONS]
    validation_positions = [position for position in mature if len(dates) - 1 - position < VALIDATION_LOOKBACK_SESSIONS]
    current, individual_current, events, baselines = {}, {}, [], {}
    baseline_events, environment_missing_events = [], []
    invalid_signal_count = older_invalid_signal_count = 0
    if any(not isinstance(code, str) or not code.strip() for code in histories):
        raise ValueError("histories证券代码不能为空")
    for code in sorted(histories):
        frame = _history(histories[code], dates, cutoff)
        values = _state_series(frame)
        individual_current[code] = _state(values.iloc[-1] if len(values) else None, dates[-1] if len(dates) else None, True)
        environment_values = _environment_series(environment_by_code.get(code), frame, dates) if use_environment else None
        if use_environment:
            current_environment = _environment_state(environment_values.iloc[-1] if len(environment_values) else None,
                dates[-1] if len(dates) else None)
            current[code] = _joint_state(individual_current[code], current_environment)
        else:
            current[code] = individual_current[code]
        bars = frame.to_numpy(dtype=float)
        close = bars[-1, 3] if len(bars) else np.nan
        baselines[code] = float(close) if np.isfinite(close) and close > 0 else None
        for position in mature:
            state = _state(values.iloc[position], dates[position], True)
            if state["status"] != "success":
                if len(dates) - 1 - position < SCORE_LOOKBACK_SESSIONS:
                    invalid_signal_count += 1
                else:
                    older_invalid_signal_count += 1
                continue
            event = _event(code, state["state_key"], bars, position, dates)
            baseline_events.append(event)
            if use_environment:
                environment = _environment_state(environment_values.iloc[position], dates[position])
                if environment["status"] != "success":
                    environment_missing_events.append({**event,
                        "environment_missing_reasons": environment["missing_reasons"], "current_environment": environment})
                    continue
                joint = _joint_state(state, environment)
                events.append({**event, "state_key": joint["state_key"], "individual_state_key": state["state_key"],
                    "current_environment": environment})
            else:
                events.append(event)
    recent = _recent_events(events, len(dates) - 1)
    individual_groups = {f"{direction}|{position}|{amount}": []
                         for direction in _DIRECTIONS for position in _POSITIONS for amount in _AMOUNT_STATES}
    groups = {f"{key}|{market}|{industry}": [] for key in individual_groups
              for market in _MARKET_STATES for industry in _INDUSTRY_STATES} if use_environment else individual_groups
    for event in recent:
        groups[event["state_key"]].append(event)
    statistics = {key: _statistics(items, []) for key, items in groups.items()}
    valid_codes = [code for code, state in current.items() if state["status"] == "success"]
    baseline_statistics = None
    recent_baseline = _recent_events(baseline_events, len(dates) - 1) if use_environment else recent
    if use_environment:
        for event in recent_baseline:
            individual_groups[event["state_key"]].append(event)
        baseline_statistics = {key: _statistics(items, []) for key, items in individual_groups.items()}
    estimates = {}
    for code, state in current.items():
        estimates[code] = _candidate_estimate(code, state, statistics, cutoff, baselines[code])
        if use_environment:
            estimates[code]["baseline_estimate"] = {
                **_candidate_estimate(code, individual_current[code], baseline_statistics, cutoff, baselines[code]),
                "method": "fixed_state_recent_weighted_next_day_return", "comparison_only": True, "used_for_ranking": False}
    available = sum(item["status"] == "success" for item in estimates.values())
    score_counts = Counter(item["expected_next_day_return"] for item in estimates.values() if item["status"] == "success")
    known = sum(event["status"] == "reference_known" for event in recent)
    recent_environment_missing = _recent_events(environment_missing_events, len(dates) - 1)
    environment_coverage = {
        "individual_state_valid": sum(state["status"] == "success" for state in individual_current.values()),
        "current_environment_valid": sum(state["current_environment"]["status"] == "success" for state in current.values()),
        "current_environment_missing": sum(state["current_environment"]["status"] != "success" for state in current.values()),
        "current_environment_missing_by_code": {code: state["current_environment"]["missing_reasons"]
            for code, state in current.items() if state["current_environment"]["status"] != "success"},
        "historical_environment_missing_events": len(recent_environment_missing),
        "historical_environment_missing_known_events": sum(event["status"] == "reference_known" for event in recent_environment_missing),
        "historical_environment_missing_unknown_events": sum(event["status"] != "reference_known" for event in recent_environment_missing),
        "historical_approximate_membership_events": sum(event["current_environment"].get("membership_approximate", False) for event in recent),
        "current_approximate_membership_stocks": sum(state["current_environment"].get("membership_approximate", False) for state in current.values()),
        "older_environment_missing_events": len(environment_missing_events) - len(recent_environment_missing),
        "environment_missing_reason_counts": dict(Counter(reason for event in recent_environment_missing
            for reason in event["environment_missing_reasons"])),
        "baseline_historical_events": len(recent_baseline),
        "baseline_estimate_available": sum(item["baseline_estimate"]["status"] == "success" for item in estimates.values()),
        "environment_missing_is_failed_condition": False,
        "historical_environment_coverage_basis": "通过原24价量状态的成熟事件中，行情按信号日期核验；行业允许沿用有公开来源的当前参考归类，近似归属事件单列；原价量状态缺失另行计数",
    } if use_environment else {}
    return {
        "status": "ok" if available == len(estimates) and available else "partial" if available else "unavailable",
        "as_of": cutoff.strftime("%Y-%m-%d"), "candidate_estimates": estimates,
        "state_statistics": statistics,
        "coverage": {
            "input_stocks": len(histories), "current_state_valid": len(valid_codes),
            "estimate_available": available, "estimate_unavailable": len(estimates) - available,
            "scheduled_signal_dates": len(scheduled), "mature_signal_dates": len(recent_mature),
            "older_mature_signal_dates_excluded": len(mature) - len(recent_mature),
            "older_state_missing_events": older_invalid_signal_count, "immature_signal_dates": len(scheduled) - len(mature),
            "historical_state_missing_events": invalid_signal_count, "historical_events": len(recent),
            "reference_known_events": known, "reference_unknown_events": len(recent) - known,
            "reference_coverage": known / len(recent) if recent else None,
            "calendar_start": dates[0].strftime("%Y-%m-%d") if len(dates) else None,
            "calendar_end": dates[-1].strftime("%Y-%m-%d") if len(dates) else None,
            "distinct_expected_returns": len(score_counts), "largest_exact_tie_group": max(score_counts.values(), default=0),
            **environment_coverage,
        },
        "cost_assumption": {"costs_applied": False, "reason": "核验次日收盘涨幅，不模拟买卖，不扣交易费用"},
        "methodology": {
            "method": "environment_conditioned_recent_weighted_next_day_return" if use_environment else "fixed_state_recent_weighted_next_day_return",
            "state_count": 96 if use_environment else 24,
            "state_role": "仅同状态成熟事件参与估计；同状态共享点估计，允许精确并列", "state_window": STATE_WINDOW, "signal_stride": SIGNAL_STRIDE,
            "score_lookback_sessions": SCORE_LOOKBACK_SESSIONS, "recency_half_life_sessions": RECENCY_HALF_LIFE_SESSIONS,
            "recency_formula": "w=2**(-signal_age_sessions/20); age<120; target<=as_of",
            "age_basis": "权威交易日历中分析日到信号日的交易间隔，分析日年龄为0",
            "score_window_start": dates[max(0, len(dates) - SCORE_LOOKBACK_SESSIONS)].strftime("%Y-%m-%d") if len(dates) else None,
            "score_window_end": cutoff.strftime("%Y-%m-%d"),
            "parameter_status": f"固定{'96环境联合' if use_environment else '24'}状态与120/20口径，不训练、不拟合、不寻优，非最优参数声明",
            "effective_sample_meaning": "有效事件量与日期量=(sum(w))^2/sum(w^2)，日期量先按日期汇总权重；只描述集中程度，不认证独立性，不作为额外资格门槛",
            "minimum_distinct_known_signal_dates": MIN_KNOWN_SIGNAL_DATES,
            "public_display": "next_day_expectation_score_and_return_only",
            "horizon": "signal_session_close_to_next_session_close", "up_label": "next_session_close>signal_session_close",
            "price_basis": "same_source_qfq_close_ratio", "costs_applied": False,
            "gain_loss_basis": "next_day_return", "ranking": "expected_next_day_return_descending_exact_ties",
            "composite_score": "10000*expected_next_day_return", "score_unit": "basis_points",
            "probability_missing_bounds_meaning": "observed_mature_event_unknown_label_bounds_not_future_confidence_interval",
            "historical_pool": "current_request_currently_eligible_stocks_not_reconstructed_historical_full_market",
            "historical_state_sample": "valid_price_amount_references_not_verified_executable_trades",
            "historical_identity_or_limit_rules_verified": False, "future_probability_calibrated": False,
            "training": False, "parameter_search": False, "market_data_persisted": False,
            "industry_reference_reuse_allowed": use_environment,
            **({"individual_state_count": 24, "market_environment_count": 2, "industry_environment_count": 2,
                "environment_definition": "market_ret_20>0/<=0；industry_ret_20-market_ret_20>0/<=0；逐信号日期匹配，行业代码不作为分组键",
                "environment_history_basis": "按用户要求沿用有公开来源的当前行业归类作为历史近似分组；不宣称历史当时归属。个股与指数行情仍只使用信号日及此前价格，内部缺口保留",
                "stock_relative_strength_basis": "个股同源前复权收盘与20个权威交易日前收盘计算收益，再减同日行业20日收益；只要求两端真实有效，不填价格、不压缩日历；差值单位为收益率差",
                "environment_missing_policy": "有来源的行业近似归类允许参与并标记；完全没有行业参考或必要行情缺失才单列证据不足，不用旧24估计补位或混排",
                "baseline_role": "同一批原始事件的旧24状态对照，不参与96状态资格或排序",
                "environment_limitations": ["二元方向不完整刻画市场、行业及波动环境，不预设强状态更优",
                    "当前行业归类可以用于历史分组，包含事后归类偏差；近似归属不冒充历史当时已知信息",
                    "行业身份和指数口径以本次来源核验范围为准，指数收益与个股复权收益口径差异须保留",
                    "环境细分减少样本，同日及连续阶段事件相关，十个已知日期门槛不证明未来有效",
                    "保留本次候选、事后入榜及当前存续资格池偏差，不是历史全市场重建"],
                } if use_environment else {}),
        },
        **({"baseline_state_statistics": baseline_statistics} if use_environment else {}),
        "_state_context": {"events": events, "dates": dates, "recent_groups": groups, "validation_positions": validation_positions,
            **({"baseline_events": baseline_events, "baseline_recent_groups": individual_groups,
                "environment_missing_events": environment_missing_events} if use_environment else {})},
    }


def buquan_zhuangtai_guji_jianyan(result: dict[str, Any], codes: list[str]) -> None:
    """只补充最终候选的检验，点估计、资格和排序保持原值。"""
    context = result.get("_state_context")
    if not context:
        return
    estimates = result.get("candidate_estimates") or {}
    codes = sorted(set(code for code in codes if code in estimates and estimates[code]["status"] == "success"))
    if not codes:
        return
    checks = _walk_forward(context["events"], context["validation_positions"], context["dates"], codes,
        reference_events=context.get("baseline_events"), environment_missing_events=context.get("environment_missing_events"))
    detailed_states = {}
    for code in codes:
        state_key = estimates[code]["state_key"]
        selected = context["recent_groups"][state_key]
        if state_key not in detailed_states:
            signal_dates = sorted({event["signal_date"] for event in selected})
            detailed_states[state_key] = _statistics(selected, signal_dates, detailed=True)
        rich = detailed_states[state_key]
        stats = estimates[code]["statistics"]
        if stats["expected_next_day_return"] != rich["expected_next_day_return"]:
            raise ValueError("同状态明细与排序原值不一致")
        for key in ("uncertainty", "date_summaries", "event_counts", "execution_limitation_counts",
                    "unknown_reason_counts", "chronological_subperiods"):
            stats[key] = rich[key]
        stats["walk_forward_check"] = checks[code]


__all__ = ["MIN_KNOWN_SIGNAL_DATES", "STATE_WINDOW", "jisuan_dangqian_zhuangtai",
           "goujian_tongzhuangtai_tongji", "buquan_zhuangtai_guji_jianyan"]
