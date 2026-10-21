"""逐股核加权的五交易日条件期望净收益；不训练、不取数、不落盘。

历史事件仅描述次日开盘至第五个交易日收盘的价格参考结果，不能证明
实际可按这些价格成交。经验上涨频率未经样本外校准，不是未来保证。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.chengben_huadian import (
    DEFAULT_COST_SOURCE,
    DEFAULT_NOTIONAL_YUAN,
    reference_roundtrip_cost_rate,
)
from src.ashare.yinzi_gongcheng import daily_atr
from src.ashare.he_jiaquan_guji import (
    prepare_kernel_pool, kernel_weights, estimate_kernel,
    MAX_REMOVED_SIGNAL_DATES, MAX_DATE_REMOVAL_RELATIVE_CHANGE,
)


MIN_KNOWN_SIGNAL_DATES = 10
SIGNAL_STRIDE = 5
STATE_WINDOW = 20
SCORE_LOOKBACK_SESSIONS = 120
RECENCY_HALF_LIFE_SESSIONS = 20
VALIDATION_LOOKBACK_SESSIONS = 60
_KERNEL_COLUMNS = ("ret_1", "ret_5", "atr_distance", "log_amount_ratio_5_15")
_BAR_COLUMNS = ("open", "high", "low", "close", "volume", "amount_yuan")
_RAW_COLUMNS = ("ret_1", "ret_5", "ma20", "atr14", "atr_distance", "amount_ratio_5_15")
_DIRECTIONS = ("up_up", "up_nonup", "nonup_up", "nonup_nonup")
_POSITIONS = ("at_or_below_ma20", "above_ma20_within_2_atr", "above_ma20_over_2_atr")
_AMOUNT_STATES = ("increased", "not_increased")


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


def _valid_ohlc(row: np.ndarray) -> bool:
    opening, high, low, close = row[:4]
    return bool(np.isfinite(row[:4]).all() and min(opening, high, low, close) > 0 and low <= min(opening, close) <= max(opening, close) <= high)


def _execution_flags(row: np.ndarray, previous_close: float, stage: str) -> list[str]:
    reasons: list[str] = []
    if not np.isfinite(row[4:]).all() or (row[4:] < 0).any():
        reasons.append(f"{stage}:turnover_missing_or_invalid")
    elif row[4] == 0 and row[5] == 0:
        reasons.append(f"{stage}:observed_zero_volume_and_amount")
    elif row[4] == 0 or row[5] == 0:
        reasons.append(f"{stage}:volume_amount_inconsistent_zero")
    direction = 1 if stage == "entry" else -1
    if _valid_ohlc(row) and np.isfinite(previous_close) and previous_close > 0:
        if row[1] - row[2] <= max(1e-8, row[3] * 1e-8) and direction * (row[0] / previous_close - 1) >= .045:
            reasons.append(f"{stage}:one_price_directional_shape_not_verified_limit")
    return reasons


def _event(code: str, key: str, bars: np.ndarray, position: int, dates: pd.DatetimeIndex, cost: float,
    raw_values: dict[str, Any] | None = None) -> dict[str, Any]:
    entry, exit_pos = position + 1, position + 5
    buy, sell = bars[entry], bars[exit_pos]
    flags = _execution_flags(buy, bars[position, 3], "entry") + _execution_flags(sell, bars[exit_pos - 1, 3], "exit")
    known = _valid_ohlc(buy) and _valid_ohlc(sell)
    gross = float(sell[3] / buy[0] - 1) if known else None
    net = float((1 + gross) * (1 - cost) - 1) if known else None
    missing = [] if known else ["entry_or_exit_ohlc_reference_missing_or_invalid"]
    if known and not (math.isfinite(gross) and math.isfinite(net)):
        known, gross, net = False, None, None
        missing = ["reference_return_nonfinite_after_price_ratio_or_cost"]
    return {"ts_code": code, "state_key": key, "kernel_features": _kernel_vector(raw_values) if raw_values else None,
        "signal_date": dates[position].strftime("%Y-%m-%d"),
        "signal_session_index": position, "exit_session_index": exit_pos,
        "entry_date": dates[entry].strftime("%Y-%m-%d"), "exit_date": dates[exit_pos].strftime("%Y-%m-%d"),
        "status": "reference_known" if known else "reference_unknown", "price_up": bool(sell[3] > buy[0]) if known else None,
        "gross_reference_return": gross, "net_reference_return": net,
        "entry_open": float(buy[0]) if np.isfinite(buy[0]) and buy[0] > 0 else None,
        "exit_close": float(sell[3]) if np.isfinite(sell[3]) and sell[3] > 0 else None,
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
    gross = _stable_mean([event["gross_reference_return"] for event in known], weights)
    net = _stable_mean([event["net_reference_return"] for event in known], weights)
    p = up_weight / known_weight if known_weight else None
    gain = _stable_mean([event["gross_reference_return"] for event in gains], [event["recency_weight"] for event in gains])
    loss = _stable_mean([-event["gross_reference_return"] for event in losses], [event["recency_weight"] for event in losses])
    decomposed = p * (gain if gain is not None else 0) - (1 - p) * (loss if loss is not None else 0) if n else None
    score = 10000 * net if net is not None else None
    score_finite = score is None or math.isfinite(score)
    return {"sample_count": len(events), "known_count": n, "unknown_count": m, "up_count": k,
        "non_up_count": n - k, "up_probability": p, "conditional_upside_mean": gain,
        "conditional_downside_mean": loss, "expected_gross_return": gross,
        "decomposed_expected_gross_return": decomposed, "expected_net_return": net,
        "composite_score_bps": score if score_finite else None,
        "numeric_range_reasons": [] if score_finite else ["composite_score_bps_exceeds_finite_float_range"],
        "known_weight_sum": known_weight, "unknown_weight_sum": unknown_weight,
        "effective_event_count": known_weight ** 2 / math.fsum(w * w for w in weights) if weights else None,
        "effective_signal_date_count": known_weight ** 2 / math.fsum(w * w for w in date_weights.values()) if date_weights else None,
        "probability_missing_bounds": [up_weight / total_weight, (up_weight + unknown_weight) / total_weight] if total_weight else None,
        "return_coverage": n / (n + m) if n + m else None}


def _uncertainty(events: list[dict[str, Any]], signal_dates: list[str], supported: bool) -> dict[str, Any]:
    """按完整信号日期槽聚集重抽样；区间只描述已观测样本的不确定性。"""
    result: dict[str, Any] = {"method": "kernel_time_weighted_circular_signal_date_block_bootstrap", "block_signal_date_slots": 4,
        "repetitions": 2000, "seed": 20261017, "valid_repetitions": 0,
        "up_probability_95_interval": None, "expected_net_return_95_interval": None,
        "status": "unavailable", "missing_reasons": [], "future_probability_calibrated": False,
        "interpretation": "exploratory_observed_sample_distribution_not_future_guarantee"}
    if not supported:
        result["missing_reasons"] = ["fewer_than_ten_distinct_known_signal_dates"]
        return result
    known_returns = [event["net_reference_return"] for event in events if event["status"] == "reference_known"]
    net_scale = max(abs(value) for value in known_returns) or 1.0
    aggregates = {date: [0., 0., 0.] for date in signal_dates}
    for event in events:
        if event["status"] == "reference_known":
            row = aggregates[event["signal_date"]]
            weight = event["recency_weight"]
            row[0] += weight
            row[1] += weight * int(event["price_up"])
            row[2] += weight * event["net_reference_return"] / net_scale
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
        result["missing_reasons"].append("net_return_interval_degenerate_empirical_resamples")
    else:
        result["expected_net_return_95_interval"] = np.quantile(returns, [.025, .975]).tolist()
    return result


def _kernel_vector(raw: dict[str, Any]) -> list[float]:
    ratio = float(raw["amount_ratio_5_15"])
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("核估计成交额比必须有限且为正")
    vector = [float(raw["ret_1"]), float(raw["ret_5"]), float(raw["atr_distance"]), math.log(ratio)]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("核估计连续特征必须有限")
    return vector


def _walk_forward(events: list[dict[str, Any]], positions: list[int], dates: pd.DatetimeIndex,
    codes: list[str]) -> dict[str, Any]:
    """最终候选逐股检查；每个起点重新确定当时可知样本、尺度和权重。"""
    checks = {code: [] for code in codes}
    targets = {(event["ts_code"], event["signal_session_index"]): event
               for event in events if event["ts_code"] in checks}
    for position in positions:
        pool = prepare_kernel_pool(events, position, SCORE_LOOKBACK_SESSIONS, RECENCY_HALF_LIFE_SESSIONS)
        available_codes = [code for code in codes if (code, position) in targets]
        estimates = estimate_kernel(pool, [targets[(code, position)]["kernel_features"] for code in available_codes],
                                    MIN_KNOWN_SIGNAL_DATES)
        estimate_by_code = dict(zip(available_codes, estimates))
        for code in codes:
            target = targets.get((code, position))
            stats = estimate_by_code.get(code) or {}
            predicted = stats.get("expected_net_return") if stats.get("status") == "success" else None
            realized = target["net_reference_return"] if target and target["status"] == "reference_known" else None
            checks[code].append({
                "signal_date": dates[position].strftime("%Y-%m-%d"),
                "status": "available" if predicted is not None and realized is not None else "unavailable",
                "training_known_signal_dates": stats.get("known_signal_dates"),
                "training_effective_signal_date_count": stats.get("effective_signal_date_count"),
                "training_latest_exit_date": pool["latest_exit_date"],
                "estimated_net_return": predicted, "realized_net_return": realized,
                "prediction_error": predicted - realized if predicted is not None and realized is not None else None,
                "evaluation_known_events": int(realized is not None),
                "evaluation_unknown_events": int(target is not None and realized is None),
                "missing_reasons": ([] if predicted is not None else ["training_sample_unsupported", *stats.get("missing_reasons", [])])
                    + ([] if realized is not None else ["evaluation_reference_unavailable"]),
            })
    results = {}
    for code, rows in checks.items():
        valid = [row for row in rows if row["status"] == "available"]
        results[code] = {
            "status": "exploratory" if valid else "unavailable", "validation_lookback_sessions": VALIDATION_LOOKBACK_SESSIONS,
            "checked_signal_dates": len(rows), "available_signal_dates": len(valid),
            "mean_estimated_net_return": _stable_mean([row["estimated_net_return"] for row in valid]),
            "mean_realized_net_return": _stable_mean([row["realized_net_return"] for row in valid]),
            "mean_prediction_error": _stable_mean([row["prediction_error"] for row in valid]),
            "mean_absolute_error": _stable_mean([abs(row["prediction_error"]) for row in valid]),
            "date_checks": rows, "parameters_tuned": False, "future_return_validated": False,
            "interpretation": "按当前入选股票追溯，用当时成熟样本估计该股随后五日净收益，各日期等权汇总。包含事后选股、当前存续及资格池偏差；不是历史选股策略或组合回测，不证明未来有效。",
        }
    return results


def _statistics(events: list[dict[str, Any]], signal_dates: list[str]) -> dict[str, Any]:
    observed = _observed(events)
    known_dates = sorted({event["signal_date"] for event in events if event["status"] == "reference_known"})
    event_dates = sorted({event["signal_date"] for event in events})
    date_events: dict[str, list[dict[str, Any]]] = {date: [] for date in event_dates}
    for event in events:
        date_events[event["signal_date"]].append(event)
    supported = len(known_dates) >= MIN_KNOWN_SIGNAL_DATES
    estimates = ("up_probability", "conditional_upside_mean", "conditional_downside_mean", "expected_gross_return", "expected_net_return", "composite_score_bps")
    ready = supported and not observed["numeric_range_reasons"]
    result = {**observed, **{key: observed[key] if supported else None for key in estimates},
        "status": "success" if ready else "unavailable", "observed_summary": observed,
        "known_signal_dates": len(known_dates), "date_count": len(known_dates), "signal_dates": len(event_dates),
        "sample_support": {"supported": supported, "minimum_distinct_known_signal_dates": MIN_KNOWN_SIGNAL_DATES,
            "meaning": "engineering_support_gate_not_return_effectiveness_or_calibration"},
        "missing_reasons": ([] if supported else ["fewer_than_ten_distinct_known_signal_dates"]) + observed["numeric_range_reasons"],
        "uncertainty": _uncertainty(events, signal_dates, supported),
        "date_summaries": [{"signal_date": date, **_observed(items),
            "unknown_reason_counts": dict(Counter(reason for event in items for reason in event["missing_reasons"])),
            "execution_limitation_counts": dict(Counter(reason for event in items for reason in event["execution_limitations"]))}
            for date, items in date_events.items()],
        "event_counts": dict(Counter(event["status"] for event in events)),
        "execution_limitation_counts": dict(Counter(reason for event in events for reason in event["execution_limitations"])),
        "unknown_reason_counts": dict(Counter(reason for event in events for reason in event["missing_reasons"])),
        "execution_verified": False, "reference_only": True, "future_probability_calibrated": False,
        "signal_start": event_dates[0] if event_dates else None, "signal_end": event_dates[-1] if event_dates else None,
        "known_signal_start": known_dates[0] if known_dates else None, "known_signal_end": known_dates[-1] if known_dates else None,
        "chronological_subperiods": []}
    for part in np.array_split(np.array(event_dates, dtype=object), 4):
        if len(part):
            subset = [event for date in part for event in date_events[date]]
            result["chronological_subperiods"].append({"start": part[0], "end": part[-1],
                "known_signal_dates": len({event["signal_date"] for event in subset if event["status"] == "reference_known"}), **_observed(subset)})
    return result


def goujian_tongzhuangtai_tongji(histories: dict[str, pd.DataFrame], trading_calendar: Iterable,
    as_of: Any) -> dict[str, Any]:
    """兼容原入口名；24状态仅作说明，排序使用逐股连续特征核估计。"""
    if not isinstance(histories, dict):
        raise TypeError("histories必须是证券代码到单股DataFrame的映射")
    dates = _calendar(trading_calendar, as_of)
    cutoff = _dates([as_of])[0]
    if len(dates) and dates[-1] != cutoff:
        raise ValueError("as_of必须属于提供的权威交易日历，不能用最近日期替换")
    cost = reference_roundtrip_cost_rate()
    scheduled = list(range(STATE_WINDOW - 1, len(dates), SIGNAL_STRIDE))
    mature = [position for position in scheduled if position + 5 < len(dates)]
    recent_mature = [position for position in mature if len(dates) - 1 - position < SCORE_LOOKBACK_SESSIONS]
    validation_positions = [position for position in mature if len(dates) - 1 - position < VALIDATION_LOOKBACK_SESSIONS]
    current, events = {}, []
    invalid_signal_count = older_invalid_signal_count = 0
    if any(not isinstance(code, str) or not code.strip() for code in histories):
        raise ValueError("histories证券代码不能为空")
    for code in sorted(histories):
        frame = _history(histories[code], dates, cutoff)
        values = _state_series(frame)
        current[code] = _state(values.iloc[-1] if len(values) else None, dates[-1] if len(dates) else None, True)
        bars = frame.to_numpy(dtype=float)
        for position in mature:
            state = _state(values.iloc[position], dates[position], True)
            if state["status"] != "success":
                if len(dates) - 1 - position < SCORE_LOOKBACK_SESSIONS:
                    invalid_signal_count += 1
                else:
                    older_invalid_signal_count += 1
                continue
            events.append(_event(code, state["state_key"], bars, position, dates, cost, state["raw_values"]))
    pool = prepare_kernel_pool(events, len(dates) - 1, SCORE_LOOKBACK_SESSIONS, RECENCY_HALF_LIFE_SESSIONS)
    valid_codes = [code for code, state in current.items() if state["status"] == "success"]
    statistics = estimate_kernel(pool, [_kernel_vector(current[code]["raw_values"]) for code in valid_codes], MIN_KNOWN_SIGNAL_DATES)
    by_code = dict(zip(valid_codes, statistics))
    estimates = {}
    for code, state in current.items():
        stats = by_code.get(code)
        reasons = list(state["missing_reasons"]) + (stats["missing_reasons"] if stats else [])
        success = state["status"] == "success" and stats is not None and stats["status"] == "success"
        estimates[code] = {
            **state, "ts_code": code, "status": "success" if success else "unavailable", "statistics": stats,
            "recommendation_index": stats["composite_score_bps"] if success else None,
            **{key: stats[key] if success else None for key in
               ("up_probability", "conditional_upside_mean", "conditional_downside_mean", "expected_net_return")},
            "missing_reasons": reasons,
        }
    available = sum(item["status"] == "success" for item in estimates.values())
    score_counts = Counter(item["expected_net_return"] for item in estimates.values() if item["status"] == "success")
    known = int(pool["known"].sum())
    return {
        "status": "ok" if available == len(estimates) and available else "partial" if available else "unavailable",
        "as_of": cutoff.strftime("%Y-%m-%d"), "candidate_estimates": estimates,
        # 旧字段保留空映射，避免调用者把组均值错用为逐股估计。
        "state_statistics": {},
        "coverage": {
            "input_stocks": len(histories), "current_state_valid": len(valid_codes),
            "estimate_available": available, "estimate_unavailable": len(estimates) - available,
            "scheduled_signal_dates": len(scheduled), "mature_signal_dates": len(recent_mature),
            "older_mature_signal_dates_excluded": len(mature) - len(recent_mature),
            "older_state_missing_events": older_invalid_signal_count, "immature_signal_dates": len(scheduled) - len(mature),
            "historical_state_missing_events": invalid_signal_count, "historical_events": len(pool["events"]),
            "reference_known_events": known, "reference_unknown_events": len(pool["events"]) - known,
            "reference_coverage": known / len(pool["events"]) if pool["events"] else None,
            "calendar_start": dates[0].strftime("%Y-%m-%d") if len(dates) else None,
            "calendar_end": dates[-1].strftime("%Y-%m-%d") if len(dates) else None,
            "distinct_expected_returns": len(score_counts), "largest_exact_tie_group": max(score_counts.values(), default=0),
        },
        "cost_assumption": {"source": DEFAULT_COST_SOURCE, "roundtrip_reference_cost_rate": cost,
            "notional_yuan": DEFAULT_NOTIONAL_YUAN, "net_formula": "(1+gross_reference_return)*(1-cost_rate)-1",
            "applied_once_per_known_reference_event": True, "actual_fill_or_broker_cost_verified": False},
        "methodology": {
            "method": "stock_specific_recent_kernel_expected_net_return", "state_count": 24,
            "state_role": "仅保留分组说明，不共享分组分数", "state_window": STATE_WINDOW, "signal_stride": SIGNAL_STRIDE,
            "score_lookback_sessions": SCORE_LOOKBACK_SESSIONS, "recency_half_life_sessions": RECENCY_HALF_LIFE_SESSIONS,
            "recency_formula": "w=2**(-signal_age_sessions/20)*max(1-d_squared/9,0)**2; age<120; exit<=as_of",
            "kernel_features": list(_KERNEL_COLUMNS), "kernel_parameters": pool["geometry"],
            "age_basis": "权威交易日历中分析日到信号日的交易间隔，分析日年龄为0",
            "score_window_start": dates[max(0, len(dates) - SCORE_LOOKBACK_SESSIONS)].strftime("%Y-%m-%d") if len(dates) else None,
            "score_window_end": cutoff.strftime("%Y-%m-%d"),
            "parameter_status": "固定核半径3与120/20口径；尺度只由当时可知特征计算，不拟合收益、不寻优，非最优参数声明",
            "effective_sample_meaning": "有效事件量与日期量=(sum(w))^2/sum(w^2)，日期量先按日期汇总权重；只描述集中程度，不认证独立性。有效日期量不足10不发布估计",
            "sample_stability_rule": {"minimum_effective_signal_dates": MIN_KNOWN_SIGNAL_DATES,
                "maximum_removed_signal_dates": MAX_REMOVED_SIGNAL_DATES,
                "maximum_date_removal_relative_change": MAX_DATE_REMOVAL_RELATIVE_CHANGE,
                "interpretation": "固定其余核与时间权重，删除任意1至2个完整信号日期后，均值变化不得超过原均值绝对值的50%；未通过不公布排序估计。固定工程敏感性门槛，不寻优、不作为未来有效性证明。"},
            "public_display": "future_five_session_expectation_score_and_net_return_only",
            "horizon": "next_session_open_to_fifth_session_close", "up_label": "exit_close>entry_open_before_cost",
            "gain_loss_basis": "gross_reference_return", "ranking": "expected_net_return_descending_exact_ties",
            "composite_score": "10000*expected_net_return", "score_unit": "basis_points",
            "probability_missing_bounds_meaning": "observed_mature_event_unknown_label_bounds_not_future_confidence_interval",
            "historical_pool": "current_request_currently_eligible_stocks_not_reconstructed_historical_full_market",
            "historical_state_sample": "valid_price_amount_references_not_verified_executable_trades",
            "historical_identity_or_limit_rules_verified": False, "future_probability_calibrated": False,
            "training": False, "parameter_search": False, "market_data_persisted": False,
        },
        "_kernel_context": {"events": events, "dates": dates, "pool": pool, "validation_positions": validation_positions},
    }


def buquan_he_guji_jianyan(result: dict[str, Any], codes: list[str]) -> None:
    """只补充最终候选的检验，点估计、资格和排序保持原值。"""
    context = result.get("_kernel_context")
    if not context:
        return
    estimates = result.get("candidate_estimates") or {}
    codes = sorted(set(code for code in codes if code in estimates and estimates[code]["status"] == "success"))
    if not codes:
        return
    pool = context["pool"]
    weights = kernel_weights(pool, [_kernel_vector(estimates[code]["raw_values"]) for code in codes])
    checks = _walk_forward(context["events"], context["validation_positions"], context["dates"], codes)
    for code, row in zip(codes, weights):
        selected = [{**event, "recency_weight": float(weight)} for event, weight in zip(pool["events"], row) if weight > 0]
        rich = _statistics(selected, pool["date_labels"])
        stats = estimates[code]["statistics"]
        if not math.isclose(stats["expected_net_return"], rich["expected_net_return"], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("核估计逐股明细与排序原值不一致")
        for key in ("uncertainty", "date_summaries", "event_counts", "execution_limitation_counts",
                    "unknown_reason_counts", "chronological_subperiods"):
            stats[key] = rich[key]
        stats["walk_forward_check"] = checks[code]


__all__ = ["MIN_KNOWN_SIGNAL_DATES", "STATE_WINDOW", "jisuan_dangqian_zhuangtai",
           "goujian_tongzhuangtai_tongji", "buquan_he_guji_jianyan"]
