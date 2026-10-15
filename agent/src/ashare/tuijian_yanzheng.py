"""固定推荐指数的独立事件组历史研究；不训练、不取数、不调权、不落盘。"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.ashare.chengben_huadian import DEFAULT_NOTIONAL_YUAN, DEFAULT_SCENARIO
from src.ashare.tuijian_zhishu import jisuan_tuijian_yuanzhi, jisuan_tuijian_zhishu


def _cost_rate() -> float:
    s, notional = DEFAULT_SCENARIO, DEFAULT_NOTIONAL_YUAN
    buy = max(s.buy_commission_rate, s.min_commission_yuan / notional) + s.transfer_fee_buy_rate + s.buy_slippage_bps / 10000
    sell = max(s.sell_commission_rate, s.min_commission_yuan / notional) + s.transfer_fee_sell_rate + s.stamp_tax_sell_rate + s.sell_slippage_bps / 10000
    return round(1 - (1 - buy) * (1 - sell), 6)


def _old_order(records: list[dict[str, Any]], top_n: int) -> list[str]:
    """独立重建旧20/5日非支配分层；完全相同原值在同一层。"""
    ordered = sorted(records, key=lambda r: (-r["ret_20"], -r["ret_5"], r["ts_code"]))
    positions = {v: i + 1 for i, v in enumerate(sorted({r["ret_5"] for r in ordered}, reverse=True))}
    tree = [0] * (len(positions) + 1)
    layers: dict[str, int] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and (ordered[end]["ret_20"], ordered[end]["ret_5"]) == (ordered[start]["ret_20"], ordered[start]["ret_5"]):
            end += 1
        position = positions[ordered[start]["ret_5"]]
        cursor, previous = position, 0
        while cursor:
            previous = max(previous, tree[cursor])
            cursor -= cursor & -cursor
        layer = previous + 1
        for row in ordered[start:end]:
            layers[row["ts_code"]] = layer
        cursor = position
        while cursor < len(tree):
            tree[cursor] = max(tree[cursor], layer)
            cursor += cursor & -cursor
        start = end
    return [r["ts_code"] for r in sorted(ordered, key=lambda r: (layers[r["ts_code"]], -r["ret_20"], -r["ret_5"], r["ts_code"]))[:top_n]]


def _valid_bar(row: np.ndarray) -> bool:
    o, h, low, c, volume, amount = row
    return bool(np.isfinite(row).all() and min(o, h, low, c, volume, amount) > 0 and low <= min(o, c) <= max(o, c) <= h)


def _one_price(row: np.ndarray, previous: float, direction: int) -> bool:
    if not _valid_bar(row) or not np.isfinite(previous) or previous <= 0:
        return False
    o, h, low, c = row[:4]
    return bool(h - low <= max(1e-8, c * 1e-8) and direction * (o / previous - 1) >= .045)


def _event(bars: np.ndarray, signal: int, dates: pd.DatetimeIndex, cost: float) -> dict[str, Any]:
    entry, exit_pos = signal + 1, signal + 5
    buy, sell = bars[entry], bars[exit_pos]
    result: dict[str, Any] = {"status": "unknown", "reference_return": None, "valuation_reference_return": None,
        "entry_date": dates[entry].strftime("%Y-%m-%d"), "target_exit_date": dates[exit_pos].strftime("%Y-%m-%d"),
        "deferred_exit_return": None, "deferred_exit_trading_days": None}
    if buy[4] == 0 and buy[5] == 0:
        result.update(status="entry_blocked", reference_return=0.0, reason="observed_zero_volume_and_amount")
        return result
    if _one_price(buy, bars[signal, 3], 1):
        result.update(status="entry_blocked", reference_return=0.0, reason="one_price_up_shape_assumption")
        return result
    if not _valid_bar(buy):
        result["reason"] = "entry_reference_or_execution_fields_missing_invalid"
        return result
    if np.isfinite(sell[3]) and sell[3] > 0:
        result["valuation_reference_return"] = float(sell[3] / buy[0] * (1 - cost) - 1)
    zero_turnover = sell[4] == 0 and sell[5] == 0
    down_shape = _one_price(sell, bars[exit_pos - 1, 3], -1)
    blocked = zero_turnover or down_shape
    if blocked:
        result.update(status="exit_blocked", reference_return=result["valuation_reference_return"],
            reason="observed_zero_volume_and_amount" if zero_turnover else "one_price_down_shape_assumption")
        for later in range(exit_pos + 1, len(bars)):
            row, previous = bars[later], bars[later - 1, 3]
            if _valid_bar(row) and np.isfinite(previous) and previous > 0 and not _one_price(row, previous, -1):
                result.update(deferred_exit_return=float(row[3] / buy[0] * (1 - cost) - 1),
                    deferred_exit_trading_days=later - exit_pos, deferred_exit_date=dates[later].strftime("%Y-%m-%d"))
                break
    elif _valid_bar(sell):
        result.update(status="reference_completed", reference_return=result["valuation_reference_return"])
    else:
        result["reason"] = "exit_reference_or_execution_fields_missing_invalid"
    return result


def _finite_return(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    sequence = list(values)
    valid = np.array([v for v in sequence if _finite_return(v)], dtype=float)
    return {"signal_dates": len(sequence), "known_dates": len(valid), "unknown_dates": len(sequence) - len(valid),
        "mean_reference_return": float(valid.mean()) if len(valid) else None,
        "median_reference_return": float(np.median(valid)) if len(valid) else None,
        "positive_reference_date_fraction": float((valid > 0).mean()) if len(valid) else None}


def _paired(rows: list[dict[str, Any]], reference: str, reference_coverage: str | None = None) -> dict[str, Any]:
    if rows and all("signal_sequence" in row for row in rows):
        first = min(row["signal_sequence"] for row in rows)
        differences = np.full(max(row["signal_sequence"] for row in rows) - first + 1, np.nan)
        for row in rows:
            if _finite_return(row["index_return"]) and _finite_return(row[reference]):
                differences[row["signal_sequence"] - first] = row["index_return"] - row[reference]
    else:
        differences = np.array([row["index_return"] - row[reference] if _finite_return(row["index_return"]) and _finite_return(row[reference]) else np.nan for row in rows])
    valid = differences[np.isfinite(differences)]
    paired_rows = [row for row in rows if _finite_return(row["index_return"]) and _finite_return(row[reference])]
    coverages = [row[reference_coverage] for row in paired_rows if reference_coverage and row.get(reference_coverage) is not None]
    index_coverages = [row["index_return_coverage"] for row in paired_rows if "index_return_coverage" in row]
    result = {"signal_dates": len(rows), "paired_dates": len(valid), "unpaired_dates": len(rows) - len(valid),
        "mean_paired_difference": float(valid.mean()) if len(valid) else None,
        "positive_difference_date_fraction": float((valid > 0).mean()) if len(valid) else None,
        "paired_reference_coverage": {"field": reference_coverage, "dates": len(coverages), "minimum": min(coverages) if coverages else None,
            "mean": float(np.mean(coverages)) if coverages else None},
        "paired_index_coverage": {"dates": len(index_coverages), "minimum": min(index_coverages) if index_coverages else None,
            "mean": float(np.mean(index_coverages)) if index_coverages else None},
        "bootstrap_95_interval": None, "bootstrap_block_signal_dates": 4, "bootstrap_repetitions": 2000,
        "seed": 20261016, "status": "inconclusive", "interpretation": "exploratory_same_date_reference_return_difference"}
    if len(valid) < 10:
        result["reason"] = "at_least_ten_paired_signal_dates_required"
        return result
    rng, n = np.random.default_rng(20261016), len(differences)
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 4)))
    locations = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    samples = differences[locations]
    counts = np.isfinite(samples).sum(axis=1)
    means = np.nansum(samples, axis=1)[counts > 0] / counts[counts > 0]
    low, high = np.quantile(means, [.025, .975])
    result.update(bootstrap_95_interval=[float(low), float(high)], valid_bootstrap_repetitions=len(means),
        status="exploratory_supported" if low > 0 else "not_supported" if high <= 0 else "inconclusive")
    return result


def _pool_reference(codes: list[str], events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """保留信号日成员；未知不填零，单列可计算均值及纯多头收益下界。"""
    known = [events[code]["reference_return"] for code in codes if _finite_return(events[code]["reference_return"])]
    count, missing = len(codes), len(codes) - len(known)
    known_sum = math.fsum(known)
    return {"pool_size": count, "known_count": len(known), "unknown_count": missing,
        "return_coverage": len(known) / count if count else None,
        "full_pool_return": known_sum / count if count and not missing else None,
        "available_return_mean": known_sum / len(known) if known else None,
        "known_return_sum": known_sum, "known_return_mass": known_sum / count if count else None,
        "missing_weight": missing / count if count else None,
        "full_pool_lower_bound": (known_sum - missing) / count if count else None,
        "missing_reasons": (["future_reference_returns_unknown"] if missing else []) + (["empty_signal_group"] if not count else [])}


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    references = {"index_vs_old_rule": "old_rule_return", "index_vs_full_pool": "reference_full_pool_return",
        "index_vs_available_subset": "reference_available_return_mean", "index_vs_ret20_only": "ret20_only_return",
        "index_vs_ret5_only": "ret5_only_return"}
    return {name: _paired(rows, field, "reference_return_coverage" if name == "index_vs_available_subset" else None)
        for name, field in references.items()}


def _missing_impact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """在TopN收益已知的同日样本计算参照下界和未知收益翻转阈值。"""
    paired = [row for row in rows if _finite_return(row["index_return"])]
    result: dict[str, Any] = {"status": "unavailable", "signal_dates": len(rows), "paired_dates": len(paired),
        "excluded_index_unknown_dates": len(rows) - len(paired), "unknown_return_lower_bound": -1.0,
        "mean_index_return": None, "mean_known_return_mass": None, "mean_missing_weight": None,
        "mean_full_pool_lower_bound": None, "upper_bound_mean_index_minus_full_pool": None,
        "missing_weighted_mean_return_to_tie": None, "unknown_stock_events": 0,
        "weighting": "equal_signal_date_weight_then_equal_signal_pool_member_weight_1_over_N",
        "interpretation": "bounds_and_threshold_only_not_imputed_returns_or_future_prediction",
        "missing_reasons": []}
    if not paired:
        result["missing_reasons"] = ["no_signal_date_with_complete_index_portfolio_return"]
        result["summary"] = "没有指数TopN收益完整的日期，不能计算与完整池的同日缺失敏感性。"
        return result
    index_mean = float(np.mean([row["index_return"] for row in paired]))
    known_mass = float(np.mean([row["reference_known_return_mass"] for row in paired]))
    missing_weight = float(np.mean([row["reference_missing_weight"] for row in paired]))
    lower_bound = known_mass - missing_weight
    excess_upper = index_mean - lower_bound
    result.update(mean_index_return=index_mean, mean_known_return_mass=known_mass,
        mean_missing_weight=missing_weight, mean_full_pool_lower_bound=lower_bound,
        upper_bound_mean_index_minus_full_pool=excess_upper,
        unknown_stock_events=sum(row["pool_size"] - row["reference_known_count"] for row in paired))
    if missing_weight == 0:
        result.update(status="fully_observed", observed_mean_index_minus_full_pool=excess_upper,
            threshold_relation="not_applicable_no_unknown_returns", summary="这些同日比较日期的完整池收益均已知，无需缺失收益临界值。")
        return result
    threshold = (index_mean - known_mass) / missing_weight
    at_boundary = math.isclose(excess_upper, 0.0, rel_tol=0, abs_tol=1e-12)
    relation = "at_long_only_lower_bound" if at_boundary else "below_long_only_lower_bound" if threshold < -1 else "above_long_only_lower_bound"
    result.update(missing_weighted_mean_return_to_tie=threshold, threshold_relation=relation,
        numerical_zero_tolerance=1e-12,
        missing_reasons=["reference_pool_contains_unknown_returns"],
        status="index_can_at_most_tie_at_missing_lower_bound" if at_boundary else
            "index_cannot_outperform_even_at_missing_lower_bound" if excess_upper < 0 else "unresolved_without_missing_upper_bound")
    result["summary"] = ("即使未知股票参考收益全部取纯多头下界−100%，指数TopN的同日平均参考收益仍低于完整池。"
        if result["status"] == "index_cannot_outperform_even_at_missing_lower_bound" else
        "只有未知股票参考收益全部取纯多头下界−100%时才能打平完整池，不能严格领先。"
        if at_boundary else "完整池缺失仍可能改变相对结论；已给出下界和使同日平均收益打平的未知收益临界值。")
    return result


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """固定五个信号日名次分位组，后续缺失只影响覆盖，不改变成员。"""
    common = [row for row in rows if all(_finite_return(group["available_return_mean"]) for group in row["index_groups"])]
    summaries = []
    for i in range(5):
        groups = [row["index_groups"][i] for row in rows]
        common_values = [row["index_groups"][i]["available_return_mean"] for row in common]
        coverage = [group["return_coverage"] for group in groups if group["return_coverage"] is not None]
        summaries.append({"group": i + 1, "rank_percentile_lower_inclusive": i / 5,
            "rank_percentile_upper_exclusive": (i + 1) / 5,
            "available_subset": _summary(group["available_return_mean"] for group in groups),
            "complete_group": _summary(group["full_pool_return"] for group in groups),
            "common_date_available_subset": _summary(common_values),
            "signal_stock_events": sum(group["pool_size"] for group in groups),
            "known_stock_events": sum(group["known_count"] for group in groups),
            "unknown_stock_events": sum(group["unknown_count"] for group in groups),
            "mean_date_coverage": float(np.mean(coverage)) if coverage else None,
            "minimum_date_coverage": min(coverage) if coverage else None,
            "empty_group_dates": sum(not group["pool_size"] for group in groups)})
    means = [group["common_date_available_subset"]["mean_reference_return"] for group in summaries]
    monotone = all(means[i] <= means[i + 1] for i in range(4)) if common else None
    increasing_dates = sum(all(row["index_groups"][i]["available_return_mean"] <=
        row["index_groups"][i + 1]["available_return_mean"] for i in range(4)) for row in common)
    contrast = [{"signal_sequence": row["signal_sequence"],
        "index_return": row["index_groups"][4]["available_return_mean"],
        "low_group_return": row["index_groups"][0]["available_return_mean"],
        "index_return_coverage": row["index_groups"][4]["return_coverage"],
        "reference_return_coverage": row["index_groups"][0]["return_coverage"]} for row in rows]
    return {"status": "available" if common else "unavailable", "group_count": 5,
        "grouping": "signal_day_index_ascending_average_rank_percentile_fixed_quintile_bands",
        "method": "按信号日指数升序平均名次分位划分固定五组；同分同组，允许空组，未来缺失不改变组成员。",
        "same_score_same_group": True, "future_missing_changes_members": False,
        "common_available_group_dates": len(common), "excluded_from_common_group_dates": len(rows) - len(common),
        "aggregate_common_date_means_nondecreasing": monotone,
        "aggregate_common_date_means_strictly_increasing": all(means[i] < means[i + 1] for i in range(4)) if common else None,
        "nondecreasing_date_fraction": increasing_dates / len(common) if common else None,
        "highest_vs_lowest_available_subset": _paired(contrast, "low_group_return", "reference_return_coverage"), "groups": summaries,
        "missing_reasons": (["no_date_with_available_return_in_all_five_fixed_groups"] if not common else []) +
            (["some_group_future_reference_returns_unknown"] if any(group["unknown_stock_events"] for group in summaries) else []),
        "summary": "共同可计算日期的五组平均参考收益随指数组别非递减；是否严格递增另列。" if monotone else
            "共同可计算日期的五组平均参考收益未随指数组别保持非递减。" if common else
            "没有五组均可计算参考收益的共同日期，暂不能比较整体递增关系。",
        "interpretation": "descriptive_gradient_and_exploratory_extreme_group_pairing_on_disclosed_available_subsets"}


def yanzheng_tuijian_zhishu(histories: dict[str, pd.DataFrame], trading_calendar: Iterable,
    top_n: int = 4, progress: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """固定五日步长、次日开盘至第五日收盘的独立事件组参考研究。

    所有同日选择仅消费截至信号日的窗口。完整池参考与可计算子集参考
    分开；TopN缺任何必要参考收益时整日期未知。每日期独立，不合并
    成本金可重复投入的组合净值，不计算年化或复利。
    """
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n必须是正整数")
    if isinstance(trading_calendar, pd.DataFrame):
        calendar_frame = trading_calendar
        if "is_open" in calendar_frame:
            calendar_frame = calendar_frame[pd.to_numeric(calendar_frame["is_open"], errors="coerce").eq(1)]
        trading_calendar = calendar_frame["cal_date" if "cal_date" in calendar_frame else "trade_date"]
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_calendar), errors="coerce")).dropna().normalize().unique().sort_values()
    cost, prepared, skipped = _cost_rate(), {}, {}
    columns = ["open", "high", "low", "close", "volume", "amount_yuan"]
    for code, history in histories.items():
        if history is None or history.empty or "trade_date" not in history:
            skipped[code] = "history_or_date_missing"
            continue
        frame = history.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.dropna(subset=["trade_date"])
        duplicates = frame["trade_date"].duplicated(keep=False)
        frame.loc[duplicates, [c for c in columns if c in frame]] = np.nan
        frame = frame.drop_duplicates("trade_date").set_index("trade_date").reindex(dates)
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame.index.name = "trade_date"
        aligned = frame.reset_index()
        raw = jisuan_tuijian_yuanzhi(aligned).to_numpy(dtype=float)
        close = aligned["close"].where(aligned["close"].gt(0))
        ret5 = (close / close.shift(5) - 1).where(close.rolling(6, min_periods=6).count().eq(6)).to_numpy()
        bars = aligned[columns].to_numpy(dtype=float)
        o, h, low, c, volume, amount = bars.T
        valid = np.isfinite(raw).all(axis=1) & np.isfinite(ret5) & np.isfinite(bars).all(axis=1) & (bars > 0).all(axis=1)
        valid &= (low <= o) & (o <= h) & (low <= c) & (c <= h)
        previous = close.shift(1).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            up_shape = valid & np.isfinite(previous) & (previous > 0) & ((h - low) <= np.maximum(1e-8, c * 1e-8)) & ((o / previous - 1) >= .045)
        prepared[code] = {"raw": raw, "ret5": ret5, "bars": bars, "valid": valid & ~up_shape, "signal_up_shape": up_shape}
    scheduled = list(range(20, len(dates), 5))
    mature = [p for p in scheduled if p + 5 < len(dates)]
    rows, skipped_dates, participating = [], [], set()
    event_counts = {name: Counter() for name in ("index", "old_rule", "ret20_only", "ret5_only", "reference_pool")}
    for sequence, position in enumerate(mature):
        records = [{"ts_code": code, "ret_20": float(item["raw"][position, 0]),
            "directional_efficiency_20": float(item["raw"][position, 1]), "up_amount_dominance_20": float(item["raw"][position, 2]),
            "ret_5": float(item["ret5"][position])} for code, item in prepared.items() if item["valid"][position]]
        if len(records) < max(2, top_n):
            skipped_dates.append(dates[position].strftime("%Y-%m-%d"))
            continue
        participating.update(r["ts_code"] for r in records)
        scores = jisuan_tuijian_zhishu(records)
        index_codes = sorted(scores, key=lambda code: (-scores[code]["recommendation_index"], code))[:top_n]
        old_codes = _old_order(records, top_n)
        ret20_codes = [row["ts_code"] for row in sorted(records, key=lambda row: (-row["ret_20"], row["ts_code"]))[:top_n]]
        ret5_codes = [row["ts_code"] for row in sorted(records, key=lambda row: (-row["ret_5"], row["ts_code"]))[:top_n]]
        score_values = pd.Series({code: value["recommendation_index"] for code, value in scores.items()})
        score_percentiles = (score_values.rank(method="average", ascending=True) - .5) / len(scores)
        group_codes = [[code for code in scores if min(4, int(score_percentiles[code] * 5)) == group] for group in range(5)]
        events = {r["ts_code"]: _event(prepared[r["ts_code"]]["bars"], position, dates, cost) for r in records}
        for name, codes in (("index", index_codes), ("old_rule", old_codes), ("ret20_only", ret20_codes),
            ("ret5_only", ret5_codes), ("reference_pool", list(events))):
            for code in codes:
                event = events[code]
                event_counts[name][event["status"]] += 1
                if not _finite_return(event["reference_return"]):
                    event_counts[name]["reference_return_unknown"] += 1
                if event["status"] == "exit_blocked":
                    event_counts[name]["deferred_exit_resolved" if event["deferred_exit_return"] is not None else "deferred_exit_unresolved"] += 1
        references = {name: _pool_reference(codes, events) for name, codes in
            (("index", index_codes), ("old_rule", old_codes), ("ret20_only", ret20_codes), ("ret5_only", ret5_codes), ("pool", list(events)))}
        pool = references["pool"]
        rows.append({"signal_date": dates[position].strftime("%Y-%m-%d"), "signal_sequence": sequence, "pool_size": len(records),
            "signal_one_price_up_shape_excluded": sum(bool(item["signal_up_shape"][position]) for item in prepared.values()),
            "index_codes": index_codes, "old_rule_codes": old_codes, "ret20_only_codes": ret20_codes, "ret5_only_codes": ret5_codes,
            "index_return": references["index"]["full_pool_return"], "old_rule_return": references["old_rule"]["full_pool_return"],
            "ret20_only_return": references["ret20_only"]["full_pool_return"], "ret5_only_return": references["ret5_only"]["full_pool_return"],
            "reference_full_pool_return": pool["full_pool_return"], "reference_available_return_mean": pool["available_return_mean"],
            "reference_known_count": pool["known_count"], "reference_unknown_count": pool["unknown_count"],
            "reference_return_coverage": pool["return_coverage"], "index_return_coverage": references["index"]["return_coverage"],
            "reference_known_return_sum": pool["known_return_sum"], "reference_known_return_mass": pool["known_return_mass"],
            "reference_missing_weight": pool["missing_weight"], "reference_full_pool_lower_bound": pool["full_pool_lower_bound"],
            "index_groups": [{"group": group + 1, "signal_codes": codes, **_pool_reference(codes, events)} for group, codes in enumerate(group_codes)],
            "index_details": [{"ts_code": code, "recommendation_index": scores[code]["recommendation_index"], **events[code]} for code in index_codes],
            "old_rule_details": [{"ts_code": code, **events[code]} for code in old_codes],
            "ret20_only_details": [{"ts_code": code, **events[code]} for code in ret20_codes],
            "ret5_only_details": [{"ts_code": code, **events[code]} for code in ret5_codes]})
        if progress is not None and (sequence % 10 == 0 or sequence + 1 == len(mature)):
            progress({"stage": "recommendation_historical_validation", "completed_signal_dates": sequence + 1,
                "total_signal_dates": len(mature), "signal_date": rows[-1]["signal_date"], "pool_size": len(records)})
    strategies = {"index": "index_return", "old_rule": "old_rule_return", "ret20_only": "ret20_only_return", "ret5_only": "ret5_only_return",
        "reference_full_pool": "reference_full_pool_return", "reference_available_subset": "reference_available_return_mean"}
    comparisons = _comparisons(rows)
    strategy_summaries = {name: _summary(row[field] for row in rows) for name, field in strategies.items()}
    missing_impact, group_analysis = _missing_impact(rows), _group_summary(rows)
    absolute = strategy_summaries["index"]
    absolute_comparison = _paired([{**row, "zero_return": 0.0} for row in rows], "zero_return")
    absolute_status = "unavailable" if absolute["mean_reference_return"] is None else "positive_sample_mean" if absolute["mean_reference_return"] > 0 else "non_positive_sample_mean"
    input_status = "unavailable" if not rows else "partial" if skipped or skipped_dates else "ok"
    future_status = "unavailable" if not rows else "partial" if event_counts["reference_pool"]["reference_return_unknown"] else "ok"
    status = "unavailable" if not rows else "partial" if input_status == "partial" or future_status == "partial" or not group_analysis["common_available_group_dates"] else "ok"
    input_missing_reasons = (["no_mature_signal_date_with_required_common_pool"] if not rows else []) + (
        ["some_input_histories_unavailable"] if skipped else []) + (["some_signal_pools_insufficient"] if skipped_dates else [])
    missing_reasons = input_missing_reasons + (
        ["some_future_reference_returns_unknown"] if any(counts["reference_return_unknown"] for counts in event_counts.values()) else []) + (
        ["no_date_with_available_return_in_all_five_fixed_groups"] if not group_analysis["common_available_group_dates"] else [])
    assessment = {
        "mathematical_sorting": {"status": "not_reassessed_here", "summary": "本模块检验历史参考收益，不代替推荐指数数学排序的独立检查。"},
        "input_coverage": {"status": input_status, "history_stocks": len(histories), "prepared_stocks": len(prepared),
            "participating_stocks": len(participating), "signal_dates": len(rows), "skipped_histories": skipped,
            "insufficient_pool_dates": skipped_dates,
            "signal_valid_stock_events": sum(row["pool_size"] for row in rows),
            "minimum_signal_pool_size": min((row["pool_size"] for row in rows), default=None),
            "missing_reasons": input_missing_reasons,
            "summary": "这里只描述推荐指标及信号日共同有效池；后续收益缺失不改变本项输入状态。"},
        "future_reference_coverage": {"status": future_status, "signal_dates": len(rows),
            "full_pool_known_dates": sum(_finite_return(row["reference_full_pool_return"]) for row in rows),
            "reference_known_stock_events": sum(row["reference_known_count"] for row in rows),
            "reference_unknown_stock_events": sum(row["reference_unknown_count"] for row in rows),
            "mean_date_coverage": float(np.mean([row["reference_return_coverage"] for row in rows])) if rows else None,
            "minimum_date_coverage": min((row["reference_return_coverage"] for row in rows), default=None),
            "missing_reasons": ["some_future_reference_returns_unknown"] if future_status == "partial" else
                ["no_valid_signal_pool"] if future_status == "unavailable" else [],
            "summary": "后续参考收益覆盖单独报告；可计算子集、完整池和缺失影响使用各自明确口径。"},
        "absolute_reference_return": {"status": absolute_status, **absolute, "zero_return_comparison": absolute_comparison,
            "summary": "本次已知日期的指数TopN平均参考收益为正；这是样本描述，不能直接认定总体正期望。" if absolute_status == "positive_sample_mean" else
                "本次已知日期的指数TopN未显示正的平均参考收益；负点估计不等于已证明总体负期望。" if absolute_status == "non_positive_sample_mean" else
                "没有指数TopN完整参考收益日期，不能评价绝对参考收益。"},
        "relative_reference_returns": comparisons, "full_pool_missing_impact": missing_impact,
        "index_group_gradient": group_analysis}
    subperiods = []
    for part in np.array_split(np.arange(len(rows)), 4):
        group = [rows[int(i)] for i in part]
        subperiods.append({"start": group[0]["signal_date"] if group else None, "end": group[-1]["signal_date"] if group else None,
            "strategies": {name: _summary(r[field] for r in group) for name, field in strategies.items()},
            "paired_comparisons": _comparisons(group)})
    return {"analysis_type": "recommendation_index_validation", "status": status,
        "missing_reasons": missing_reasons,
        "status_semantics": "research_output_coverage_only_not_return_efficacy", "purpose": "fixed_formula_exploratory_historical_event_reference",
        "sample_basis": "current_survivors_and_latest_history_qualified_sample", "top_n": top_n,
        "history_stocks": len(histories), "prepared_stocks": len(prepared), "skipped_histories": skipped,
        "participating_stocks": len(participating),
        "calendar_start": dates[0].strftime("%Y-%m-%d") if len(dates) else None,
        "calendar_end": dates[-1].strftime("%Y-%m-%d") if len(dates) else None,
        "signal_start": rows[0]["signal_date"] if rows else None, "signal_end": rows[-1]["signal_date"] if rows else None,
        "non_overlapping_target_signal_dates": len(rows), "scheduled_mature_signal_dates": len(mature),
        "signal_one_price_up_shape_excluded_observations": sum(int(item["signal_up_shape"][mature].sum()) for item in prepared.values()),
        "tail_unmatured_signal_dates": len(scheduled) - len(mature), "insufficient_pool_dates": skipped_dates,
        "signal_step_trading_days": 5, "holding_reference": "T+1 open to T+5 close",
        "cost": {"scenario": DEFAULT_SCENARIO.name, "notional_reference_yuan": DEFAULT_NOTIONAL_YUAN,
            "roundtrip_proportional_rate": cost, "whole_lot_or_raw_quote_verified": False},
        "strategies": strategy_summaries, "assessment": assessment,
        "full_pool_missing_impact": missing_impact, "index_group_analysis": group_analysis,
        "paired_comparisons": comparisons, "four_time_subperiods": subperiods,
        "execution_counts": {name: dict(counts) for name, counts in event_counts.items()}, "date_results": rows,
        "limitations": ["当前目录、当前基础通过和当前末端历史核验共同造成幸存与末端条件偏差，不能代表历史完整主板",
            "前复权开盘与收盘只作价格参考，不是对应时点原始成交报价；历史证券身份、真实涨跌停及除权信息不完整",
            "仅有日线，正成交量不保证开盘成交；4.5%以上一字方向形态仅作保守阻买阻卖假设，不声称核实涨跌停",
            "信号日一字向上形态按同一保守假设排除；仅为共同价量池近似研究，不是已核实历史身份和真实限价的完整生产流程重放",
            "阻买的未投入资金参考收益为零；阻卖日净值参考已预留往返成本，未称已实现收益；缺失不冒充零收益",
            "可计算子集参考公开覆盖，完整池参考与配对检验不静默删掉未来数据缺失者",
            "可计算子集、单项TopN对照及固定五分位组均在同日配对；组内未来缺失只影响覆盖，不能重排信号日成员",
            "完整池纯多头参考收益下界使用未知收益−100%，TopN减去该下界是超额收益上界；边界分析不是缺失值填补",
            "跨日期未知平均收益临界值采用每日期等权、日期内每股票1/N权重；未来收益无有限上界，不能凭下界断言必定领先",
            "分组递增诊断使用共同日期的可计算子集，仍受组内未知收益影响；同分同组可能产生空组，不强拆同分证券",
            "独立事件组参考不构成可重复使用同一本金的组合账本，不生成年化、复利或上涨概率",
            "五日步长仍可能有时间相关；95%区间采用固定种子、四信号日期连续循环块bootstrap，少于十对日期不推断",
            "研究不训练模型、不寻优权重；即使得到正向探索证据，也不认证未来潜力或保证收益"]}


__all__ = ["yanzheng_tuijian_zhishu"]
