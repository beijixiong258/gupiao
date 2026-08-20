"""Shared, leak-aware daily factor engineering for stock and board models.

The module deliberately separates three concerns:

* raw price/volume measurements are calculated per security from data available
  on or before the signal close;
* named economic groups expose a small, traceable production representation;
* the model layer can select and combine factors inside each training window.

No target, future label, intraday bar, or current-day information from another
date is used here.  Cross-sectional ranks are only used on the supplied
``trade_date`` and therefore remain reproducible at a historical signal time.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FACTOR_ENGINEERING_VERSION = "daily-factor-engineering-v2"


# The registry is intentionally data rather than executable logic.  It is
# returned in diagnostics so a prediction can be traced back to its economic
# interpretation and availability requirements.
FACTOR_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "trend_structure",
            (
                "ma_gap_5",
                "ma_gap_10",
                "ma_gap_20",
                "ma_gap_60",
                "ma_trend_5_20",
                "trend_slope_20",
                "trend_fit_quality_20",
                "ma_5_20_gap_change",
            ),
        ),
        (
            "momentum_reversal",
            ("ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "rsi_14", "macd_dif_pct", "macd_hist_pct"),
        ),
        (
            "candle_pressure",
            (
                "gap_open",
                "intraday_return",
                "close_location",
                "body_pct",
                "upper_shadow_pct",
                "lower_shadow_pct",
                "signed_close_pressure",
                "shadow_imbalance",
            ),
        ),
        (
            "price_volume_confirmation",
            (
                "volume_ratio_5_20",
                "amount_ratio_5_20",
                "amount_anomaly_20",
                "signed_amount_shock",
                "return_amount_corr_20",
                "price_turnover_corr_20",
                "wvma_20",
                "volume_price_residual_20",
                "overnight_intraday_corr_20",
            ),
        ),
        (
            "breakout_pullback_quality",
            (
                "breakout_distance_20",
                "breakout_volume_confirmation",
                "pullback_quality_20",
                "stalling_pressure_20",
                "low_volume_long_lower_shadow",
            ),
        ),
        (
            "relative_strength",
            (
                "peer_mean_ret_1",
                "peer_mean_ret_5",
                "peer_mean_ret_20",
                "excess_ret_1",
                "excess_ret_5",
                "excess_ret_20",
                "excess_vs_universe_ret_1",
                "excess_vs_universe_ret_5",
                "excess_vs_universe_ret_20",
                "excess_vs_industry_ret_1",
                "excess_vs_industry_ret_5",
                "excess_vs_industry_ret_20",
                "excess_vs_csi300_ret_1",
                "excess_vs_csi300_ret_5",
                "excess_vs_csi300_ret_20",
                "rank_ret_5",
                "rank_ma_gap_20",
            ),
        ),
        (
            "risk_liquidity",
            (
                "atr_14_pct",
                "volatility_20",
                "amplitude_1",
                "drawdown_20",
                "peer_dispersion_ret_5",
                "log_amount_yuan",
                "rank_volume_ratio_5_20",
                "rank_volatility_20",
                "rank_log_amount",
                "turnover_rate_daily",
                "rank_turnover_rate_daily",
                "log_circ_mv",
                "rank_log_circ_mv",
            ),
        ),
        (
            "market_context",
            (
                "universe_mean_ret_1",
                "universe_mean_ret_5",
                "universe_mean_ret_20",
                "universe_breadth_above_ma20",
                "universe_breadth_positive_5d",
                "universe_dispersion_ret_5",
                "industry_mean_ret_1",
                "industry_mean_ret_5",
                "industry_mean_ret_20",
                "industry_breadth_above_ma20",
                "industry_breadth_positive_5d",
                "industry_dispersion_ret_5",
                "market_reference_mean_ret_1",
                "market_reference_mean_ret_5",
                "market_reference_mean_ret_20",
                "market_regime_score",
                "market_regime_weak",
                "market_regime_sideways",
                "market_regime_strong",
                "market_regime_trend_breadth_interaction",
                "market_regime_volatility_stress",
            ),
        ),
    ]
)


RAW_PRICE_VOLUME_FEATURE_COLUMNS: tuple[str, ...] = (
    "gap_open",
    "intraday_return",
    "close_location",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "signed_close_pressure",
    "shadow_imbalance",
    "amount_ratio_5_20",
    "amount_anomaly_20",
    "signed_amount_shock",
    "return_amount_corr_20",
    "price_turnover_corr_20",
    "wvma_20",
    "volume_price_residual_20",
    "overnight_intraday_corr_20",
    "trend_slope_20",
    "trend_fit_quality_20",
    "breakout_distance_20",
    "breakout_volume_confirmation",
    "pullback_quality_20",
    "stalling_pressure_20",
    "ma_5_20_gap_change",
    "low_volume_long_lower_shadow",
)


COMPOSITE_FACTOR_COLUMNS: tuple[str, ...] = tuple(
    f"factor_{group}_composite" for group in FACTOR_GROUPS
)


def _registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, members in FACTOR_GROUPS.items():
        role = "risk" if group == "risk_liquidity" else "context" if group == "market_context" else "alpha"
        for feature in members:
            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "role": role,
                    "direction": "diagnostic_only" if role != "alpha" else "learned_in_fold",
                    "frequency": "daily_k",
                    "availability": "signal_close_or_earlier",
                }
            )
        rows.append(
            {
                "feature": f"factor_{group}_composite",
                "group": group,
                "role": role,
                "direction": "group_rank_equal_weight",
                "frequency": "daily_k",
                "availability": "signal_close_or_earlier",
            }
        )
    return rows


FACTOR_REGISTRY: tuple[dict[str, Any], ...] = tuple(_registry_rows())
_FEATURE_TO_GROUP = {
    feature: group for group, members in FACTOR_GROUPS.items() for feature in members
}
_FEATURE_TO_GROUP.update({f"factor_{group}_composite": group for group in FACTOR_GROUPS})
# 旧字段实际描述 MA5 与 MA20 相对间距的日变化，并非 MACD 金叉。
# 仅保留查询兼容，不再把旧名放入生产因子列表，避免同一证据重复计权。
_FEATURE_TO_GROUP["golden_cross_speed"] = "trend_structure"


def factor_group(feature: str) -> str:
    """Return the registered economic group, or a stable fallback group."""
    value = str(feature)
    if value in _FEATURE_TO_GROUP:
        return _FEATURE_TO_GROUP[value]
    if value.startswith(("market_", "universe_")):
        return "market_context"
    if value.startswith(("peer_", "industry_", "excess_", "rank_")):
        return "relative_strength"
    return "unregistered"


def factor_role(feature: str) -> str:
    """Return the registry role used by the selector and output explainers."""
    group = factor_group(feature)
    if group == "risk_liquidity":
        return "risk"
    if group == "market_context":
        return "context"
    return "alpha"


def factor_registry_rows() -> list[dict[str, Any]]:
    """Return JSON-safe registry rows for run diagnostics and artifacts."""
    return [dict(row) for row in FACTOR_REGISTRY]


def _numeric(data: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _group_indices(data: pd.DataFrame) -> Iterable[pd.Index]:
    if "ts_code" in data.columns and data["ts_code"].nunique(dropna=True) > 1:
        yield from (group.index for _, group in data.groupby("ts_code", sort=False))
    else:
        yield data.index


def add_price_volume_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add continuous price/volume and objective candle-shape factors.

    Every rolling operation is performed independently per security.  The
    implementation uses small fixed windows so it cannot become a parameter
    sweep disguised as feature generation.
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    data = frame.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data = data.sort_values([column for column in ["ts_code", "trade_date"] if column in data.columns]).copy()
    close = _numeric(data, "close")
    open_price = _numeric(data, "open")
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    volume = _numeric(data, "volume")
    amount = _numeric(data, "amount_yuan")
    # Some providers omit amount.  Volume*close is a transparent proxy, not a
    # fabricated valuation field, and is marked by the same daily availability.
    amount = amount.where(amount.notna(), (volume * close).where(volume.notna() & close.notna()))

    # These fields are useful outside the model too (for technical diagnosis),
    # so calculate them even when an older caller did not request the new set.
    groups = list(_group_indices(data))
    for indices in groups:
        local = data.loc[indices].sort_values("trade_date")
        c = _numeric(local, "close")
        o = _numeric(local, "open")
        h = _numeric(local, "high")
        l = _numeric(local, "low")
        v = _numeric(local, "volume")
        a = _numeric(local, "amount_yuan")
        a = a.where(a.notna(), (v * c).where(v.notna() & c.notna()))
        prev = c.shift(1)
        ret1 = c.pct_change(fill_method=None)
        gap = _safe_div(o, prev) - 1.0
        intraday = _safe_div(c, o) - 1.0
        daily_range = (h - l).replace(0, np.nan)
        location = _safe_div(c - l, daily_range).clip(0.0, 1.0)
        body = _safe_div(c - o, prev)
        upper = _safe_div(h - pd.concat([o, c], axis=1).max(axis=1), prev)
        lower = _safe_div(pd.concat([o, c], axis=1).min(axis=1) - l, prev)
        amount_log = np.log1p(a.clip(lower=0))
        amount_change = amount_log.diff()
        mean5 = a.rolling(5, min_periods=5).mean()
        mean20 = a.rolling(20, min_periods=20).mean()
        ratio = _safe_div(mean5, mean20)
        rolling_mean = amount_log.rolling(20, min_periods=10).mean()
        rolling_std = amount_log.rolling(20, min_periods=10).std().replace(0, np.nan)
        anomaly = _safe_div(amount_log - rolling_mean, rolling_std)
        ret_amount_corr = ret1.rolling(20, min_periods=10).corr(amount_change)
        # Turnover is unavailable in price-only mode; amount change is the
        # explicitly disclosed proxy and keeps the feature missing-safe.
        price_turnover_corr = ret_amount_corr.copy()
        weight = _safe_div(a, a.rolling(20, min_periods=10).median()).clip(lower=0.25, upper=4.0)
        wvma = np.sqrt(_safe_div((ret1.pow(2) * weight).rolling(20, min_periods=10).sum(), weight.rolling(20, min_periods=10).sum()))
        cov = ret1.rolling(20, min_periods=10).cov(amount_change)
        var = amount_change.rolling(20, min_periods=10).var().replace(0, np.nan)
        beta = _safe_div(cov, var)
        residual = ret1 - beta * amount_change
        overnight_corr = gap.rolling(20, min_periods=10).corr(intraday)
        log_close = np.log(c.where(c > 0))
        slope = log_close.diff(20) / 20.0
        volatility = ret1.rolling(20, min_periods=10).std()
        fit_quality = _safe_div(log_close.diff(20).abs(), volatility * math.sqrt(20.0))
        prior_high = h.rolling(20, min_periods=10).max().shift(1)
        breakout = _safe_div(c, prior_high) - 1.0
        breakout_confirmation = breakout.clip(lower=0.0) * (ratio - 1.0)
        ma_gap20 = _numeric(local, "ma_gap_20")
        position20 = _numeric(local, "position_20", 0.5).fillna(0.5)
        ma_trend = _numeric(local, "ma_trend_5_20")
        pullback = (-c.pct_change(5, fill_method=None)).clip(lower=0.0) * (1.0 - ratio).clip(lower=0.0) * ma_trend.clip(lower=0.0)
        stalling = position20.clip(lower=0.0, upper=1.0) * anomaly.clip(lower=0.0) * (1.0 - location.fillna(0.5))
        ma_gap_change = (_numeric(local, "ma_gap_5") - ma_gap20).diff()
        low_volume_shadow = (1.0 - position20).clip(lower=0.0, upper=1.0) * lower.clip(lower=0.0) / (1.0 + ratio.clip(lower=0.0))
        values = {
            "gap_open": gap,
            "intraday_return": intraday,
            "close_location": location,
            "body_pct": body,
            "upper_shadow_pct": upper,
            "lower_shadow_pct": lower,
            "signed_close_pressure": location.fillna(0.5).sub(0.5) * body.abs().fillna(0.0) * np.sign(body.fillna(0.0)),
            "shadow_imbalance": lower - upper,
            "amount_ratio_5_20": ratio,
            "amount_anomaly_20": anomaly,
            "signed_amount_shock": np.sign(ret1.fillna(0.0)) * anomaly,
            "return_amount_corr_20": ret_amount_corr,
            "price_turnover_corr_20": price_turnover_corr,
            "wvma_20": wvma,
            "volume_price_residual_20": residual,
            "overnight_intraday_corr_20": overnight_corr,
            "trend_slope_20": slope,
            "trend_fit_quality_20": fit_quality,
            "breakout_distance_20": breakout,
            "breakout_volume_confirmation": breakout_confirmation,
            "pullback_quality_20": pullback,
            "stalling_pressure_20": stalling,
            "ma_5_20_gap_change": ma_gap_change,
            # 兼容仍读取旧列名的调用方；因子注册表只登记上面的准确名称。
            "golden_cross_speed": ma_gap_change,
            "low_volume_long_lower_shadow": low_volume_shadow,
        }
        for column, series in values.items():
            data.loc[local.index, column] = pd.to_numeric(series, errors="coerce").to_numpy()
    return data.replace([np.inf, -np.inf], np.nan)


def add_factor_composites(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create deterministic equal-weight group scores from same-date ranks."""
    if frame is None or frame.empty:
        return (pd.DataFrame() if frame is None else frame.copy(), {"status": "unavailable"})
    data = frame.copy()
    if "trade_date" not in data.columns:
        return data, {"status": "unavailable", "reason": "缺少 trade_date"}
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    group_meta: dict[str, Any] = {}
    for group, members in FACTOR_GROUPS.items():
        available = [column for column in members if column in data.columns]
        composite = f"factor_{group}_composite"
        if not available:
            data[composite] = np.nan
            group_meta[group] = {"available_components": [], "status": "unavailable"}
            continue
        numeric = data[available].apply(pd.to_numeric, errors="coerce")
        if group == "market_context":
            # Context variables are often identical for every stock on a
            # date.  Cross-sectional ranking would turn them into a useless
            # constant (and can accidentally imply an alpha direction), so
            # they remain raw controls and are selected separately by the
            # temporal context branch below.
            data[composite] = np.nan
            group_meta[group] = {
                "available_components": available,
                "component_count": int(len(available)),
                "method": "raw_context_controls_no_cross_sectional_rank",
                "status": "context_raw_only",
            }
            continue
        ranks = numeric.groupby(data["trade_date"]).rank(pct=True)
        data[composite] = ranks.mean(axis=1, skipna=True, numeric_only=True)
        data[composite] = data[composite].where(ranks.notna().any(axis=1))
        group_meta[group] = {
            "available_components": available,
            "component_count": int(len(available)),
            "method": "same_trade_date_percentile_rank_equal_weight",
            "status": "ok",
        }
    return data.replace([np.inf, -np.inf], np.nan), {
        "status": "ok",
        "version": FACTOR_ENGINEERING_VERSION,
        "groups": group_meta,
        "composite_columns": list(COMPOSITE_FACTOR_COLUMNS),
        "daily_k_only": True,
    }


def engineered_factor_columns(*, include_composites: bool = True) -> list[str]:
    """Return de-duplicated raw/composite columns in registry order."""
    columns: list[str] = []
    for members in FACTOR_GROUPS.values():
        for column in members:
            if column not in columns:
                columns.append(column)
    if include_composites:
        columns.extend(column for column in COMPOSITE_FACTOR_COLUMNS if column not in columns)
    return columns


def _daily_rank_ic_values(frame: pd.DataFrame, feature: str, target_column: str) -> list[float]:
    """Return one cross-sectional Rank IC per usable signal date."""
    if feature not in frame.columns or target_column not in frame.columns or "trade_date" not in frame.columns:
        return []
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    values: list[float] = []
    for _, group in frame.loc[dates.notna()].groupby(dates[dates.notna()]):
        x = pd.to_numeric(group[feature], errors="coerce")
        y = pd.to_numeric(group[target_column], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 5 or int(x[valid].nunique()) < 2 or int(y[valid].nunique()) < 2:
            continue
        value = spearmanr(x[valid], y[valid]).statistic
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _context_rank_ic_values(frame: pd.DataFrame, feature: str, target_column: str) -> list[float]:
    """Measure a context control against date-level mean target, leak-free."""
    if feature not in frame.columns or target_column not in frame.columns or "trade_date" not in frame.columns:
        return []
    work = frame[["trade_date", feature, target_column]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    date_level = (
        work.dropna(subset=["trade_date"])
        .groupby("trade_date", as_index=False)
        .agg({feature: "median", target_column: "mean"})
        .dropna()
        .sort_values("trade_date")
    )
    if len(date_level) < 20 or date_level[feature].nunique() < 2 or date_level[target_column].nunique() < 2:
        return []
    dates = date_level["trade_date"].tolist()
    slices = [list(values) for values in np.array_split(np.asarray(dates, dtype=object), 6) if len(values)]
    values: list[float] = []
    for date_slice in slices:
        part = date_level[date_level["trade_date"].isin(date_slice)]
        if len(part) < 5 or part[feature].nunique() < 2 or part[target_column].nunique() < 2:
            continue
        value = spearmanr(part[feature], part[target_column]).statistic
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def select_fold_factor_features(
    frame: pd.DataFrame,
    candidate_features: list[str],
    target_column: str,
    model_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Select compact, group-aware factors using only one training window.

    The selector intentionally does not back-fill unstable columns to reach a
    requested count.  It first removes exact/near-duplicate candidates inside
    each economic group, then keeps only factors with adequate daily Rank IC
    coverage and sign stability.  Composite columns are preferred over their
    components, while the latter remain available for diagnostics.
    """
    if frame is None or frame.empty:
        return [], {"status": "empty_training_window", "selection_scope": "training_window_only"}
    enabled = bool(model_config.get("factor_stability_enabled", True))
    coverage_threshold = float(model_config.get("min_feature_coverage", 0.20))
    slices = max(2, int(model_config.get("factor_stability_slices", 6)))
    minimum_valid_slices = max(1, int(model_config.get("factor_min_valid_slices", 4)))
    minimum_sign_agreement = float(model_config.get("factor_min_sign_agreement", 0.67))
    minimum_abs_ic = float(model_config.get("factor_min_abs_mean_rank_ic", 0.005))
    minimum_features = max(1, int(model_config.get("factor_min_features", 15)))
    maximum_features = max(minimum_features, int(model_config.get("factor_max_features", 20)))
    maximum_per_group = max(1, int(model_config.get("factor_max_per_group", 3)))
    correlation_threshold = float(model_config.get("factor_dedup_abs_spearman", 0.80))
    ordered_candidates: list[str] = []
    for feature in candidate_features:
        name = str(feature)
        if name in frame.columns and name not in ordered_candidates:
            ordered_candidates.append(name)
    if not ordered_candidates:
        return [], {"status": "no_candidate_feature", "selection_scope": "training_window_only"}
    # Deterministic exact duplicate removal is performed before stability
    # scoring, including across groups (e.g. peer/universe aliases).
    coverage = frame[ordered_candidates].notna().mean()
    deduplicated: list[str] = []
    exact_duplicate_dropped: list[dict[str, Any]] = []
    for feature in ordered_candidates:
        duplicate_of: str | None = None
        for prior in deduplicated:
            pair = frame[[feature, prior]].apply(pd.to_numeric, errors="coerce")
            valid = pair.notna().all(axis=1)
            if int(valid.sum()) < 20:
                continue
            if not pair[feature].isna().equals(pair[prior].isna()):
                continue
            if np.allclose(
                pair.loc[valid, feature].to_numpy(),
                pair.loc[valid, prior].to_numpy(),
                rtol=1e-10,
                atol=1e-12,
            ):
                duplicate_of = prior
        if duplicate_of:
            exact_duplicate_dropped.append(
                {"feature": feature, "duplicate_of": duplicate_of, "reason": "exact_same_panel_values"}
            )
        else:
            deduplicated.append(feature)
    eligible = [feature for feature in deduplicated if float(coverage.get(feature, 0.0)) >= coverage_threshold]
    if not enabled:
        return eligible[:maximum_features], {
            "status": "disabled",
            "selection_scope": "training_window_only",
            "selected_features": eligible[:maximum_features],
            "candidate_count": len(ordered_candidates),
            "coverage_eligible_count": len(eligible),
            "minimum_features": minimum_features,
            "maximum_features": maximum_features,
            "fallback_to_unstable_features": False,
        }

    all_dates = sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().dt.normalize().unique())
    date_slices = [list(values) for values in np.array_split(np.asarray(all_dates, dtype=object), slices) if len(values)]
    diagnostics: dict[str, Any] = {}
    ranking: dict[str, tuple[float, float, float, float, int]] = {}
    context_candidates = [feature for feature in eligible if factor_role(feature) == "context"]
    context_limit = max(1, int(model_config.get("factor_max_context_features", 4)))
    context_ranking: list[tuple[tuple[float, float, float], str]] = []
    selected_context: list[str] = []
    for feature in context_candidates:
        values = _context_rank_ic_values(frame, feature, target_column)
        mean_ic = float(np.mean(values)) if values else 0.0
        context_ranking.append(((abs(mean_ic), float(coverage.get(feature, 0.0)), float(len(values))), feature))
        diagnostics[feature] = {
            "group": factor_group(feature),
            "role": "context",
            "coverage": round(float(coverage.get(feature, 0.0)), 4),
            "temporal_rank_ic": [round(value, 6) for value in values],
            "temporal_rank_ic_mean": round(mean_ic, 6),
            "selected_as_context_control": False,
            "stability_method": "date_level_context_rank_ic",
        }
    for _, feature in sorted(context_ranking, reverse=True)[:context_limit]:
        selected_context.append(feature)
        diagnostics[feature]["selected_as_context_control"] = True
    for feature in eligible:
        if factor_role(feature) == "context":
            continue
        daily_values = _daily_rank_ic_values(frame, feature, target_column)
        slice_values: list[float] = []
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        for date_slice in date_slices:
            mask = dates.isin(date_slice)
            part = frame.loc[mask, [feature, target_column]]
            x = pd.to_numeric(part[feature], errors="coerce")
            y = pd.to_numeric(part[target_column], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 10 or int(x[valid].nunique()) < 2 or int(y[valid].nunique()) < 2:
                continue
            value = spearmanr(x[valid], y[valid]).statistic
            if value is not None and math.isfinite(float(value)):
                slice_values.append(float(value))
        mean_ic = float(np.mean(daily_values)) if daily_values else 0.0
        sign_agreement = (
            max(sum(value > 0 for value in daily_values), sum(value < 0 for value in daily_values)) / len(daily_values)
            if daily_values
            else 0.0
        )
        slice_sign_agreement = (
            max(sum(value > 0 for value in slice_values), sum(value < 0 for value in slice_values)) / len(slice_values)
            if slice_values
            else 0.0
        )
        stable = bool(
            len(slice_values) >= minimum_valid_slices
            and abs(mean_ic) >= minimum_abs_ic
            and sign_agreement >= minimum_sign_agreement
        )
        is_composite = feature.startswith("factor_") and feature.endswith("_composite")
        diagnostics[feature] = {
            "group": factor_group(feature),
            "role": factor_role(feature),
            "coverage": round(float(coverage.get(feature, 0.0)), 4),
            "daily_rank_ic_count": int(len(daily_values)),
            "daily_rank_ic_mean": round(mean_ic, 6),
            "daily_rank_ic_median": round(float(np.median(daily_values)), 6) if daily_values else 0.0,
            "slice_rank_ic": [round(value, 6) for value in slice_values],
            "slice_sign_agreement": round(float(slice_sign_agreement), 4),
            "sign_agreement": round(float(sign_agreement), 4),
            "stable": stable,
            "composite_preferred": is_composite,
        }
        # Composite factors win ties; the remaining terms keep the selection
        # deterministic while preferring stronger and better-covered signals.
        ranking[feature] = (
            float(stable),
            float(is_composite),
            abs(mean_ic),
            float(coverage.get(feature, 0.0)),
            len(daily_values),
        )

    stable_candidates = [feature for feature in eligible if feature not in context_candidates and diagnostics[feature]["stable"]]
    stable_candidates.sort(key=lambda feature: ranking[feature], reverse=True)
    selected: list[str] = []
    selected_by_group: dict[str, list[str]] = {}
    duplicate_dropped: list[dict[str, Any]] = []
    # Reserve a small, explicit slice of the production input for context
    # controls.  They are not ranked as same-day alpha signals.
    selected.extend(selected_context[:maximum_features])
    for feature in stable_candidates:
        if len(selected) >= maximum_features:
            break
        group = factor_group(feature)
        group_selected = selected_by_group.setdefault(group, [])
        if len(group_selected) >= maximum_per_group:
            continue
        # Only compare candidates that belong to the same economic group; this
        # avoids suppressing a useful market context factor merely because it
        # correlates with a price trend factor.
        duplicate_of: str | None = None
        for prior in group_selected:
            pair = frame[[feature, prior]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 20 or pair[feature].nunique() < 2 or pair[prior].nunique() < 2:
                continue
            corr = pair[feature].corr(pair[prior], method="spearman")
            if corr is not None and math.isfinite(float(corr)) and abs(float(corr)) >= correlation_threshold:
                duplicate_of = prior
                break
        if duplicate_of:
            duplicate_dropped.append(
                {"feature": feature, "duplicate_of": duplicate_of, "group": group, "abs_spearman": correlation_threshold}
            )
            continue
        selected.append(feature)
        group_selected.append(feature)

    return selected, {
        "status": "ok" if selected else "no_stable_feature",
        "selection_scope": "training_window_only",
        "engineering_version": FACTOR_ENGINEERING_VERSION,
        "candidate_count": int(len(ordered_candidates)),
        "coverage_eligible_count": int(len(eligible)),
        "stable_candidate_count": int(len(stable_candidates)),
        "selected_count": int(len(selected)),
        "selected_features": selected,
        "selected_by_group": selected_by_group,
        "duplicate_dropped": duplicate_dropped,
        "exact_duplicate_dropped": exact_duplicate_dropped,
        "selected_context_controls": selected_context[:maximum_features],
        "fallback_to_strongest_factors": False,
        "stability": "daily_cross_sectional_rank_ic_with_six_date_slices",
        "thresholds": {
            "minimum_coverage": coverage_threshold,
            "minimum_valid_slices": minimum_valid_slices,
            "minimum_sign_agreement": minimum_sign_agreement,
            "minimum_abs_mean_rank_ic": minimum_abs_ic,
            "minimum_features_target": minimum_features,
            "maximum_features": maximum_features,
            "maximum_per_group": maximum_per_group,
            "dedup_abs_spearman": correlation_threshold,
        },
        "factor_diagnostics": diagnostics,
        "registry": factor_registry_rows(),
        "warning": (
            f"稳定因子只有 {len(selected)} 个，低于目标 {minimum_features} 个；未强行补入不稳定因子"
            if len(selected) < minimum_features
            else None
        ),
    }


__all__ = [
    "COMPOSITE_FACTOR_COLUMNS",
    "FACTOR_ENGINEERING_VERSION",
    "FACTOR_GROUPS",
    "FACTOR_REGISTRY",
    "RAW_PRICE_VOLUME_FEATURE_COLUMNS",
    "add_factor_composites",
    "add_price_volume_factors",
    "engineered_factor_columns",
    "factor_group",
    "factor_role",
    "factor_registry_rows",
    "select_fold_factor_features",
]
