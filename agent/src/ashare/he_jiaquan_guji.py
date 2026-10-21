"""连续价量特征的局部条件分布；只做纯计算，不拟合收益参数。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


KERNEL_RADIUS = 3.0
MIN_EFFECTIVE_EVENTS = 10.0
MAX_REMOVED_SIGNAL_DATES = 2
MAX_DATE_REMOVAL_RELATIVE_CHANGE = 0.5
_ESTIMATES = ("up_probability", "conditional_upside_mean", "conditional_downside_mean",
              "expected_gross_return", "expected_net_return", "composite_score_bps")


def prepare_kernel_pool(events: list[dict[str, Any]], as_of_position: int,
    lookback: int, half_life: int) -> dict[str, Any]:
    """尺度仅由当时已成熟窗口的特征确定，不读取待预测结果。"""
    selected = [event for event in events
                if 0 <= as_of_position - event["signal_session_index"] < lookback
                and event["exit_session_index"] <= as_of_position]
    features = np.asarray([event["kernel_features"] for event in selected], dtype=float).reshape(-1, 4)
    if not np.isfinite(features).all():
        raise ValueError("核估计特征必须全部有限")
    center = np.median(features, axis=0) if len(features) else np.zeros(4)
    if len(features):
        q25, q75 = np.quantile(features, [.25, .75], axis=0)
        scale = q75 - q25
        # 四分位距为零时只借用特征标准差；完全常量维度禁止外推。
        deviation = np.std(features, axis=0)
        tolerance = np.maximum(1.0, np.abs(center)) * 1e-12
        scale = np.where(scale > tolerance, scale, deviation)
        constant = scale <= tolerance
        scale = np.where(constant, 1.0, scale)
    else:
        scale, constant = np.ones(4), np.ones(4, dtype=bool)
    known = np.asarray([event["status"] == "reference_known" for event in selected], dtype=bool)
    signal_positions = np.asarray([event["signal_session_index"] for event in selected], dtype=int)
    unique_positions, date_index = np.unique(signal_positions, return_inverse=True)
    date_labels = {event["signal_session_index"]: event["signal_date"] for event in selected}
    return {
        "events": selected, "features": features, "center": center, "scale": scale, "constant": constant,
        "known": known, "up": np.asarray([event["price_up"] is True for event in selected], dtype=bool),
        # 未知结果在计算前由known掩码剔除，不把占位数当作已知零收益。
        "gross": np.asarray([event["gross_reference_return"] if ok else np.nan
                             for event, ok in zip(selected, known)], dtype=float),
        "net": np.asarray([event["net_reference_return"] if ok else np.nan
                           for event, ok in zip(selected, known)], dtype=float),
        "time_weights": np.exp2(-(as_of_position - signal_positions) / half_life),
        "date_positions": unique_positions, "date_index": date_index,
        "date_labels": [date_labels[position] for position in unique_positions],
        "latest_exit_date": max((event["exit_date"] for event in selected), default=None),
        "geometry": {"kernel": "compact_biweight", "radius": KERNEL_RADIUS,
                     "feature_center": center.tolist(), "feature_scale": scale.tolist(),
                     "constant_dimensions": constant.tolist(),
                     "scale_rule": "当时成熟窗口特征的四分位距；为零时用标准差；完全常量维度不外推",
                     "distance_rule": "四项标准化连续特征的联合欧氏距离；相关特征不视为独立确认",
                     "weight_formula": "2**(-age/20) * max(1-d_squared/9,0)**2",
                     "bandwidth_fitted_to_returns": False},
    }


def kernel_weights(pool: dict[str, Any], queries: np.ndarray) -> np.ndarray:
    queries = np.asarray(queries, dtype=float).reshape(-1, 4)
    if not np.isfinite(queries).all():
        raise ValueError("待估计特征必须全部有限")
    distance = np.zeros((len(queries), len(pool["events"])), dtype=float)
    # 逐维广播，限制内存，不生成查询数×事件数×维数的三维副本。
    with np.errstate(over="ignore", invalid="ignore"):
        for dimension in range(4):
            delta = (queries[:, dimension, None] - pool["features"][None, :, dimension]) / pool["scale"][dimension]
            distance += delta * delta
    weights = np.maximum(1.0 - distance / KERNEL_RADIUS ** 2, 0.0) ** 2
    weights *= pool["time_weights"][None, :]
    constant = pool["constant"]
    if constant.any():
        mismatch = np.any(np.abs(queries[:, constant] - pool["center"][constant]) >
                          np.maximum(1.0, np.abs(pool["center"][constant])) * 1e-12, axis=1)
        weights[mismatch] = 0.0
    if not np.isfinite(weights).all():
        raise ValueError("核权重计算产生非有限数")
    return weights


def _mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if not len(values) or weights.sum() <= 0:
        return None
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return float(np.clip(np.sum(weights * (values / scale)) / weights.sum(), -1.0, 1.0) * scale)


def _date_removal_check(pool: dict[str, Any], known: np.ndarray, known_weights: np.ndarray,
    date_weights: np.ndarray, net: float | None) -> dict[str, Any]:
    """穷举删除一个或两个完整信号日；固定尺度及其余权重，检查均值敏感性。"""
    result: dict[str, Any] = {
        "method": "fixed_weight_leave_one_or_two_signal_dates_out", "status": "unavailable",
        "maximum_removed_signal_dates": MAX_REMOVED_SIGNAL_DATES,
        "maximum_relative_change_allowed": MAX_DATE_REMOVAL_RELATIVE_CHANGE,
        "evaluated_removals": 0, "full_expected_net_return": net,
        "minimum_expected_net_return": None, "maximum_expected_net_return": None,
        "maximum_absolute_change": None, "maximum_relative_change": None,
        "most_influential_signal_dates": [],
        "interpretation": "仅检验既有样本对少数日期的敏感性；不重拟尺度、不证明未来有效。正负估计使用同一门槛，零均值不以除零获得通过。",
    }
    active = np.flatnonzero(date_weights > 0)
    if net is None or len(active) <= MAX_REMOVED_SIGNAL_DATES:
        return result
    total = float(known_weights.sum())
    scale = float(np.max(np.abs(pool["net"][known]))) or 1.0
    shares = date_weights[active] / total
    contributions = np.bincount(pool["date_index"][known],
        weights=(known_weights / total) * (pool["net"][known] / scale),
        minlength=len(date_weights))[active]
    # 每个单日及所有日期对均检查，不能只删除权重最大的日期。
    left, right = np.triu_indices(len(active), k=1)
    removed_shares = np.concatenate((shares, shares[left] + shares[right]))
    removed_returns = np.concatenate((contributions, contributions[left] + contributions[right]))
    remaining = 1.0 - removed_shares
    if np.any(remaining <= 0):
        return result
    mean = net / scale
    after = np.clip((mean - removed_returns) / remaining, -1.0, 1.0)
    changes = np.abs(after - mean)
    worst = int(np.argmax(changes))
    absolute_change = float(changes[worst]) * scale
    relative_change = float(changes[worst] / abs(mean)) if mean else (0.0 if not changes.any() else None)
    removed = [worst] if worst < len(active) else [int(left[worst - len(active)]), int(right[worst - len(active)])]
    # 1e-12为收益比例的浮点容差，不是收益门槛或置信度分。
    stable = absolute_change <= MAX_DATE_REMOVAL_RELATIVE_CHANGE * abs(net) + 1e-12
    result.update(
        status="stable" if stable else "unstable", evaluated_removals=len(after),
        minimum_expected_net_return=float(np.min(after)) * scale,
        maximum_expected_net_return=float(np.max(after)) * scale,
        maximum_absolute_change=absolute_change if math.isfinite(absolute_change) else None,
        maximum_relative_change=relative_change if relative_change is None or math.isfinite(relative_change) else None,
        most_influential_signal_dates=[pool["date_labels"][active[index]] for index in removed],
    )
    return result


def kernel_summary(pool: dict[str, Any], weights: np.ndarray, minimum_dates: int) -> dict[str, Any]:
    positive = weights > 0
    known = positive & pool["known"]
    up = known & pool["up"]
    nonup = known & ~pool["up"]
    unknown = positive & ~pool["known"]
    known_weights = weights[known]
    total = float(known_weights.sum())
    unknown_total = float(weights[unknown].sum())
    gain_total = float(weights[up].sum())
    count, unknown_count = int(known.sum()), int(unknown.sum())
    date_weights = np.bincount(pool["date_index"][known], weights=known_weights,
                              minlength=len(pool["date_positions"]))
    date_count = int((date_weights > 0).sum())
    effective = total ** 2 / float(np.sum(known_weights ** 2)) if total else None
    effective_dates = total ** 2 / float(np.sum(date_weights ** 2)) if total else None
    probability = gain_total / total if total else None
    gain, loss = _mean(pool["gross"][up], weights[up]), _mean(-pool["gross"][nonup], weights[nonup])
    gross, net = _mean(pool["gross"][known], known_weights), _mean(pool["net"][known], known_weights)
    score = 10000.0 * net if net is not None else None
    numeric_reasons = [] if score is None or math.isfinite(score) else ["composite_score_bps_exceeds_finite_float_range"]
    reasons = ([] if date_count >= minimum_dates else ["fewer_than_ten_distinct_known_signal_dates"])
    if effective is None or effective < MIN_EFFECTIVE_EVENTS - 1e-10:
        reasons.append("kernel_effective_events_below_ten")
    if effective_dates is None or effective_dates < minimum_dates - 1e-10:
        reasons.append("kernel_effective_signal_dates_below_minimum")
    date_stability = _date_removal_check(pool, known, known_weights, date_weights, net)
    if date_stability["status"] != "stable":
        reasons.append("kernel_signal_date_removal_unstable" if date_stability["status"] == "unstable"
                       else "kernel_signal_date_removal_unavailable")
    reasons += numeric_reasons
    observed = {
        "sample_count": count + unknown_count, "known_count": count, "unknown_count": unknown_count,
        "up_count": int(up.sum()), "non_up_count": int(nonup.sum()), "up_probability": probability,
        "conditional_upside_mean": gain, "conditional_downside_mean": loss,
        "expected_gross_return": gross,
        "decomposed_expected_gross_return": probability * (gain or 0) - (1 - probability) * (loss or 0) if probability is not None else None,
        "expected_net_return": net, "composite_score_bps": score if not numeric_reasons else None,
        "numeric_range_reasons": numeric_reasons, "known_weight_sum": total, "unknown_weight_sum": unknown_total,
        "effective_event_count": effective, "effective_signal_date_count": effective_dates,
        "probability_missing_bounds": [gain_total / (total + unknown_total), (gain_total + unknown_total) / (total + unknown_total)] if total + unknown_total else None,
        "return_coverage": count / (count + unknown_count) if count + unknown_count else None,
    }
    supported = not reasons
    event_date_counts = np.bincount(pool["date_index"][positive], minlength=len(pool["date_positions"]))
    event_dates = [date for date, count in zip(pool["date_labels"], event_date_counts) if count]
    known_dates = [date for date, weight in zip(pool["date_labels"], date_weights) if weight > 0]
    return {
        **observed, **{key: observed[key] if supported else None for key in _ESTIMATES},
        "status": "success" if supported else "unavailable", "observed_summary": observed,
        "known_signal_dates": date_count, "date_count": date_count, "signal_dates": len(set(event_dates)),
        "sample_support": {"supported": supported, "minimum_distinct_known_signal_dates": minimum_dates,
                           "minimum_effective_events": MIN_EFFECTIVE_EVENTS,
                           "minimum_effective_signal_dates": minimum_dates,
                           "maximum_removed_signal_dates": MAX_REMOVED_SIGNAL_DATES,
                           "maximum_date_removal_relative_change": MAX_DATE_REMOVAL_RELATIVE_CHANGE,
                           "meaning": "engineering_support_gate_not_return_effectiveness_or_calibration"},
        "date_stability_check": date_stability,
        "missing_reasons": reasons, "kernel_parameters": pool["geometry"],
        "signal_start": min(event_dates, default=None), "signal_end": max(event_dates, default=None),
        "known_signal_start": min(known_dates, default=None), "known_signal_end": max(known_dates, default=None),
        "uncertainty": {"status": "not_requested", "scope": "最终候选补充日期分块重抽样；不用于排名"},
        "walk_forward_check": {"status": "not_requested", "scope": "最终候选补充逐股逐期检查；不用于排名"},
        "execution_verified": False, "reference_only": True, "future_probability_calibrated": False,
    }


def estimate_kernel(pool: dict[str, Any], queries: np.ndarray, minimum_dates: int) -> list[dict[str, Any]]:
    queries = np.asarray(queries, dtype=float).reshape(-1, 4)
    summaries = []
    for start in range(0, len(queries), 16):
        weights = kernel_weights(pool, queries[start:start + 16])
        summaries.extend(kernel_summary(pool, row, minimum_dates) for row in weights)
    return summaries
