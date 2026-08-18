"""分析与预测共用的纯模型辅助函数。"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.ashare.chengben_huadian import CostScenario
from src.ashare.gupiao_yanjiu import (
    biaozhunhua_daima,
    guifan_you_xian_shuzhi,
    jisuan_tezheng_biao,
)
from src.ashare.jiaoyi_zhixing import jisuan_gupiao_wangfan_chengben, koujian_jiaoyi_chengben
from src.ashare.riping_yinzi import DAILY_FACTOR_FEATURE_COLUMNS
from src.ashare.yinzi_gongcheng import select_fold_factor_features

MARKET_BASELINE_FEATURE_COLUMNS = [
    column
    for column in DAILY_FACTOR_FEATURE_COLUMNS
    if column.startswith(("market_", "universe_", "industry_"))
]


def _historical_limit_rates(ts_code: str) -> tuple[float, ...]:
    """Return plausible historical price-limit rates without using today's stock name."""
    normalized = biaozhunhua_daima(ts_code)
    digits = normalized.split(".")[0]
    if normalized.endswith(".BJ"):
        return (0.30,)
    if digits.startswith(("688", "689")):
        return (0.20,)
    if digits.startswith(("300", "301")):
        return (0.10, 0.20)
    return (0.05, 0.10)


def _one_price_limit_session(
    prices: pd.DataFrame,
    session_dates: pd.Series,
    *,
    ts_code: str,
    direction: str,
) -> pd.Series:
    """Identify historical one-price limit sessions from their observed bars."""
    high = session_dates.map(pd.to_numeric(prices["high"], errors="coerce"))
    low = session_dates.map(pd.to_numeric(prices["low"], errors="coerce"))
    close = session_dates.map(pd.to_numeric(prices["close"], errors="coerce"))
    if "pct_chg" in prices.columns:
        returns_by_date = pd.to_numeric(prices["pct_chg"], errors="coerce") / 100.0
    else:
        if "pre_close" in prices.columns:
            pre_close = pd.to_numeric(prices["pre_close"], errors="coerce")
        else:
            pre_close = pd.Series(np.nan, index=prices.index, dtype=float)
        pre_close = pre_close.where(pre_close.gt(0), pd.to_numeric(prices["close"], errors="coerce").shift(1))
        returns_by_date = pd.to_numeric(prices["close"], errors="coerce") / pre_close - 1.0
    session_return = session_dates.map(returns_by_date)
    price_tolerance = close.abs().mul(0.0002).clip(lower=0.005)
    one_price = (
        high.notna()
        & low.notna()
        & close.notna()
        & (high.sub(low).abs() <= price_tolerance)
        & (high.sub(close).abs() <= price_tolerance)
        & (low.sub(close).abs() <= price_tolerance)
    )
    limited_session_by_date = pd.Series(np.arange(len(prices)) >= 5, index=prices.index)
    limited_session = session_dates.map(limited_session_by_date).eq(True)
    sign = 1.0 if direction == "up" else -1.0
    at_supported_limit = pd.Series(False, index=session_dates.index, dtype=bool)
    for rate in _historical_limit_rates(ts_code):
        at_supported_limit |= session_return.sub(sign * rate).abs() <= 0.0015
    return one_price & limited_session & at_supported_limit


def goujian_moxing_shuju(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str],
    horizons: list[int],
) -> pd.DataFrame:
    """Build executable labels on a shared market-date calendar.

    A signal is produced after the close of date T.  The assumed entry is the
    next market session's open.  ``T+1`` therefore means the first *sellable*
    close after that entry (the second market session after the signal), with
    ``T+2`` and ``T+3`` following on subsequent market sessions.  A suspended
    stock has no label when it lacks a quote on the required common-market
    entry or exit date; its next observed bar is never silently treated as the
    next market session.
    """
    market_dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for history in histories.values()
            if "trade_date" in history.columns
            for value in pd.to_datetime(history["trade_date"], errors="coerce").dropna()
        }
    )
    market_position = {value: index for index, value in enumerate(market_dates)}
    frames: list[pd.DataFrame] = []
    for code, history in histories.items():
        features = jisuan_tezheng_biao(history)
        features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce").dt.normalize()
        price_columns = ["trade_date", "open", "high", "low", "close"]
        price_columns.extend(column for column in ["pre_close", "pct_chg"] if column in features.columns)
        prices = (
            features[price_columns]
            .dropna(subset=["trade_date"])
            .drop_duplicates("trade_date", keep="last")
            .set_index("trade_date")
        )
        features["ts_code"] = code
        features["name"] = names.get(code, "")
        for horizon in horizons:
            future_dates: list[Any] = []
            entry_dates: list[Any] = []
            exit_dates: list[Any] = []
            for signal_date in features["trade_date"]:
                position = market_position.get(pd.Timestamp(signal_date)) if pd.notna(signal_date) else None
                future_index = position + int(horizon) if position is not None else len(market_dates)
                entry_index = position + 1 if position is not None else len(market_dates)
                exit_index = entry_index + int(horizon)
                future_dates.append(market_dates[future_index] if future_index < len(market_dates) else pd.NaT)
                entry_dates.append(market_dates[entry_index] if entry_index < len(market_dates) else pd.NaT)
                exit_dates.append(market_dates[exit_index] if exit_index < len(market_dates) else pd.NaT)

            future_series = pd.Series(future_dates, index=features.index, dtype="datetime64[ns]")
            entry_series = pd.Series(entry_dates, index=features.index, dtype="datetime64[ns]")
            exit_series = pd.Series(exit_dates, index=features.index, dtype="datetime64[ns]")
            future_close = future_series.map(pd.to_numeric(prices["close"], errors="coerce"))
            features[f"future_date_t{horizon}"] = future_series
            features[f"future_close_t{horizon}"] = future_close
            features[f"future_return_t{horizon}"] = (
                future_close / pd.to_numeric(features["close"], errors="coerce") - 1.0
            )
            entry_open = entry_series.map(pd.to_numeric(prices["open"], errors="coerce"))
            blocked_limit_up = _one_price_limit_session(
                prices,
                entry_series,
                ts_code=code,
                direction="up",
            )
            entry_open = entry_open.mask(blocked_limit_up)
            exit_close = exit_series.map(pd.to_numeric(prices["close"], errors="coerce"))
            blocked_limit_down = _one_price_limit_session(
                prices,
                exit_series,
                ts_code=code,
                direction="down",
            )
            exit_close = exit_close.mask(blocked_limit_down)
            features[f"entry_date_t{horizon}"] = entry_series
            features[f"entry_open_t{horizon}"] = entry_open
            features[f"entry_blocked_limit_up_t{horizon}"] = blocked_limit_up
            features[f"target_date_t{horizon}"] = exit_series
            features[f"target_close_t{horizon}"] = exit_close
            features[f"exit_blocked_limit_down_t{horizon}"] = blocked_limit_down
            features[f"target_t{horizon}"] = exit_close / entry_open - 1.0
        frames.append(features)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _daily_rank_ic(dates: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "actual": actual, "predicted": predicted})
    values: list[float] = []
    for _, group in frame.groupby("date"):
        if len(group) < 5 or group["actual"].nunique() < 2 or group["predicted"].nunique() < 2:
            continue
        value = spearmanr(group["actual"], group["predicted"]).statistic
        if math.isfinite(float(value)):
            values.append(float(value))
    if values:
        return float(np.mean(values)), len(values)
    if len(frame) >= 5 and frame["actual"].nunique() >= 2 and frame["predicted"].nunique() >= 2:
        value = float(spearmanr(frame["actual"], frame["predicted"]).statistic)
        return (value if math.isfinite(value) else 0.0), 1
    return 0.0, 0


def _quality_score(
    *,
    train_count: int,
    direction_accuracy: float,
    rank_ic: float,
    skill_vs_baseline: float,
) -> float:
    sample = min(1.0, math.sqrt(max(train_count, 0) / 2000.0))
    direction = float(np.clip((direction_accuracy - 0.5) / 0.12, 0.0, 1.0))
    rank = float(np.clip(rank_ic / 0.10, 0.0, 1.0))
    skill = float(np.clip(skill_vs_baseline / 0.10, 0.0, 1.0))
    return float(np.clip(sample * (0.45 * direction + 0.35 * rank + 0.20 * skill), 0.0, 1.0))


def _quality_label(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.40:
        return "medium"
    return "low"


class _TrainingQuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to bounds learned only from the active training window."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, values: Any, _target: Any = None) -> "_TrainingQuantileClipper":
        array = np.asarray(values, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper_quantile, axis=0)
        return self

    def transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.minimum(np.maximum(array, self.lower_bounds_), self.upper_bounds_)


def _feature_clipper(model_config: dict[str, Any]) -> _TrainingQuantileClipper:
    quantiles = model_config.get("feature_winsor_quantiles", [0.01, 0.99])
    return _TrainingQuantileClipper(float(quantiles[0]), float(quantiles[1]))


def _build_model_pipeline(model_config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=float(model_config.get("learning_rate", 0.05)),
                    max_iter=int(model_config.get("max_iter", 180)),
                    max_leaf_nodes=int(model_config.get("max_leaf_nodes", 15)),
                    max_depth=int(model_config.get("max_depth", 4)),
                    min_samples_leaf=int(model_config.get("min_samples_leaf", 30)),
                    l2_regularization=float(model_config.get("l2_regularization", 1.0)),
                    random_state=int(model_config.get("random_state", 42)),
                ),
            ),
        ]
    )


def _build_linear_model_pipeline(model_config: dict[str, Any]) -> Pipeline:
    """Build the regularized linear component used to offset tree-model bias."""
    return Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            ("model", Ridge(alpha=float(model_config.get("ridge_alpha", 10.0)))),
        ]
    )


def _time_decay_weights(
    dates: pd.Series | np.ndarray,
    half_life_dates: float,
    minimum_weight: float = 0.10,
) -> np.ndarray:
    """Return mean-one exponential weights based on trading-date age."""
    values = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce").dt.normalize()
    if values.empty or not math.isfinite(float(half_life_dates)) or float(half_life_dates) <= 0:
        return np.ones(len(values), dtype=float)
    ordered_dates = list(pd.Index(values.dropna().unique()).sort_values())
    if not ordered_dates:
        return np.ones(len(values), dtype=float)
    age_by_date = {
        pd.Timestamp(value): len(ordered_dates) - index - 1
        for index, value in enumerate(ordered_dates)
    }
    ages = values.map(age_by_date).fillna(len(ordered_dates) - 1).to_numpy(dtype=float)
    weights = np.power(0.5, ages / float(half_life_dates))
    weights = np.maximum(weights, float(minimum_weight))
    mean_weight = float(np.mean(weights))
    return weights / mean_weight if mean_weight > 0 else np.ones(len(values), dtype=float)


def _select_time_decay_half_life(
    *,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    train_dates: pd.Series | np.ndarray,
    model_config: dict[str, Any],
    train_target_dates: pd.Series | np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Choose recency weighting on a purged tail inside the active training window."""
    if not model_config.get("time_decay_enabled", True):
        return 0.0, {"status": "disabled", "selected_half_life_dates": 0.0}

    candidates = sorted(
        {
            float(value)
            for value in model_config.get("time_decay_half_life_candidates", [0, 252, 504])
        }
    )
    signal_dates = pd.to_datetime(pd.Series(train_dates).reset_index(drop=True), errors="coerce").dt.normalize()
    target = np.asarray(train_target, dtype=float)
    if len(train_features) != len(signal_dates) or len(target) != len(signal_dates):
        raise ValueError("时间衰减选择的特征、目标和日期长度不一致")
    target_dates = (
        pd.to_datetime(pd.Series(train_target_dates).reset_index(drop=True), errors="coerce").dt.normalize()
        if train_target_dates is not None
        else signal_dates
    )
    unique_dates = [pd.Timestamp(value) for value in sorted(signal_dates.dropna().unique())]
    ratio = float(model_config.get("time_decay_calibration_ratio", 0.15))
    minimum_dates = int(model_config.get("time_decay_min_calibration_dates", 20))
    minimum_samples = int(model_config.get("time_decay_min_calibration_samples", 80))
    fallback = {
        "status": "insufficient_nested_data",
        "selection_scope": "purged_tail_of_active_training_only",
        "selected_half_life_dates": 0.0,
        "candidate_half_life_dates": candidates,
    }
    if len(unique_dates) < minimum_dates * 2:
        fallback["training_dates"] = int(len(unique_dates))
        return 0.0, fallback

    calibration_date_count = max(minimum_dates, int(len(unique_dates) * ratio))
    calibration_date_count = min(calibration_date_count, len(unique_dates) - minimum_dates)
    cutoff = unique_dates[-calibration_date_count]
    inner_mask = signal_dates.lt(cutoff) & target_dates.lt(cutoff)
    calibration_mask = signal_dates.ge(cutoff)
    if int(inner_mask.sum()) < minimum_samples or int(calibration_mask.sum()) < minimum_samples:
        fallback.update(
            {
                "calibration_start": cutoff.strftime("%Y-%m-%d"),
                "nested_train_samples": int(inner_mask.sum()),
                "nested_calibration_samples": int(calibration_mask.sum()),
            }
        )
        return 0.0, fallback

    inner_features = train_features.reset_index(drop=True).loc[inner_mask.to_numpy()]
    calibration_features = train_features.reset_index(drop=True).loc[calibration_mask.to_numpy()]
    inner_target = target[inner_mask.to_numpy()]
    calibration_target = target[calibration_mask.to_numpy()]
    minimum_weight = float(model_config.get("time_decay_min_weight", 0.10))
    candidate_mae: dict[str, float] = {}
    for candidate in candidates:
        try:
            model = _build_linear_model_pipeline(model_config)
            sample_weight = _time_decay_weights(
                signal_dates.loc[inner_mask],
                candidate,
                minimum_weight,
            )
            model.fit(inner_features, inner_target, model__sample_weight=sample_weight)
            prediction = np.asarray(model.predict(calibration_features), dtype=float)
            candidate_mae[f"{candidate:g}"] = float(mean_absolute_error(calibration_target, prediction))
        except Exception:
            continue
    baseline_mae = candidate_mae.get("0")
    if baseline_mae is None or not candidate_mae:
        fallback.update(
            {
                "status": "selection_failed",
                "calibration_start": cutoff.strftime("%Y-%m-%d"),
                "candidate_mae": {key: round(value, 8) for key, value in candidate_mae.items()},
            }
        )
        return 0.0, fallback

    best_key, best_mae = min(candidate_mae.items(), key=lambda item: item[1])
    best_half_life = float(best_key)
    relative_improvement = (
        (baseline_mae - best_mae) / abs(baseline_mae)
        if abs(baseline_mae) > 1e-12
        else 0.0
    )
    minimum_improvement = float(model_config.get("time_decay_min_relative_improvement", 0.002))
    selected = best_half_life if best_half_life > 0 and relative_improvement >= minimum_improvement else 0.0
    return selected, {
        "status": "selected" if selected > 0 else "kept_uniform_weights",
        "selection_scope": "purged_tail_of_active_training_only",
        "calibration_start": cutoff.strftime("%Y-%m-%d"),
        "nested_train_samples": int(inner_mask.sum()),
        "nested_calibration_samples": int(calibration_mask.sum()),
        "candidate_mae": {key: round(value, 8) for key, value in candidate_mae.items()},
        "best_candidate_half_life_dates": best_half_life,
        "relative_mae_improvement": round(float(relative_improvement), 8),
        "minimum_relative_improvement": minimum_improvement,
        "selected_half_life_dates": selected,
    }


def _build_quantile_model_pipeline(
    model_config: dict[str, Any],
    quantile: float,
) -> Pipeline:
    return Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(quantile),
                    learning_rate=float(model_config.get("learning_rate", 0.05)),
                    max_iter=int(model_config.get("max_iter", 180)),
                    max_leaf_nodes=int(model_config.get("max_leaf_nodes", 15)),
                    max_depth=int(model_config.get("max_depth", 4)),
                    min_samples_leaf=int(model_config.get("min_samples_leaf", 30)),
                    l2_regularization=float(model_config.get("l2_regularization", 1.0)),
                    random_state=int(model_config.get("random_state", 42)),
                ),
            ),
        ]
    )


def _fit_quantile_model_components(
    *,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    predict_features: pd.DataFrame,
    model_config: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    levels = [
        float(value)
        for value in model_config.get("quantile_levels", [0.10, 0.50, 0.90])
    ]
    if not model_config.get("quantile_interval_enabled", True):
        return {}, {"status": "disabled"}
    if levels != sorted(levels) or len(levels) != 3:
        raise ValueError("moxing.quantile_levels 必须是三个递增分位数")
    raw_predictions: list[np.ndarray] = []
    fit_kwargs = {"model__sample_weight": np.asarray(sample_weight, dtype=float)} if sample_weight is not None else {}
    for level in levels:
        model = _build_quantile_model_pipeline(model_config, level)
        model.fit(train_features, np.asarray(train_target, dtype=float), **fit_kwargs)
        raw_predictions.append(
            np.asarray(model.predict(predict_features), dtype=float)
        )
    stacked = np.vstack(raw_predictions)
    crossings = int(
        np.sum(
            (stacked[0] > stacked[1])
            | (stacked[1] > stacked[2])
        )
    )
    ordered = np.sort(stacked, axis=0)
    return {
        "lower": ordered[0],
        "median": ordered[1],
        "upper": ordered[2],
    }, {
        "status": "ok",
        "model": "HistGradientBoostingRegressor(loss=quantile)",
        "quantile_levels": levels,
        "predictions_reordered_for_crossing": crossings,
    }


def _fit_model_components(
    *,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    predict_features: pd.DataFrame,
    model_config: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tree = _build_model_pipeline(model_config)
    linear = _build_linear_model_pipeline(model_config)
    fit_kwargs = {"model__sample_weight": np.asarray(sample_weight, dtype=float)} if sample_weight is not None else {}
    tree.fit(train_features, train_target, **fit_kwargs)
    linear.fit(train_features, train_target, **fit_kwargs)
    return (
        np.asarray(tree.predict(predict_features), dtype=float),
        np.asarray(linear.predict(predict_features), dtype=float),
    )


def _fit_market_baseline(
    *,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    target_column: str,
    model_config: dict[str, Any],
    time_decay_half_life: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict the board-wide return separately from stock-specific excess return."""
    candidate_features = [
        column
        for column in MARKET_BASELINE_FEATURE_COLUMNS
        if column in train.columns and column in predict.columns
    ]
    coverage_threshold = float(model_config.get("min_feature_coverage", 0.20))
    feature_columns = [
        column
        for column in candidate_features
        if float(train[column].notna().mean()) >= coverage_threshold
    ]
    fallback = float(pd.to_numeric(train[target_column], errors="coerce").median())
    if not math.isfinite(fallback):
        fallback = 0.0
    if not feature_columns:
        return np.full(len(predict), fallback, dtype=float), {
            "status": "constant_fallback",
            "reason": "没有达到覆盖率门槛的市场/板块状态特征",
            "constant_return": round(fallback, 6),
        }

    date_level = (
        train[["trade_date", target_column] + feature_columns]
        .groupby("trade_date", as_index=False)
        .agg({target_column: "mean", **{column: "median" for column in feature_columns}})
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[target_column])
        .sort_values("trade_date")
    )
    if len(date_level) < 60:
        return np.full(len(predict), fallback, dtype=float), {
            "status": "constant_fallback",
            "reason": f"市场基准训练日期只有{len(date_level)}个，少于60个",
            "constant_return": round(fallback, 6),
        }
    model = _build_model_pipeline(model_config)
    sample_weight = _time_decay_weights(
        date_level["trade_date"],
        time_decay_half_life,
        float(model_config.get("time_decay_min_weight", 0.10)),
    )
    model.fit(
        date_level[feature_columns],
        date_level[target_column].to_numpy(dtype=float),
        model__sample_weight=sample_weight,
    )
    prediction = np.asarray(model.predict(predict[feature_columns]), dtype=float)
    return prediction, {
        "status": "modelled",
        "model": "HistGradientBoostingRegressor(loss=absolute_error)",
        "training_dates": int(len(date_level)),
        "features": feature_columns,
        "target": "同一交易日板块股票平均持有期收益",
        "time_decay_half_life_dates": float(time_decay_half_life),
    }


def _historical_net_returns(
    frame: pd.DataFrame,
    actual: np.ndarray,
    *,
    horizon: int,
    budget_yuan: float,
    scenario: CostScenario,
    trading_settings: dict[str, Any] | None = None,
) -> np.ndarray:
    values: list[float] = []
    for (_, row), gross_return in zip(frame.iterrows(), np.asarray(actual, dtype=float)):
        cost_rate, _ = jisuan_gupiao_wangfan_chengben(
            str(row["ts_code"]),
            float(row[f"entry_open_t{horizon}"]),
            budget_yuan,
            scenario,
            daily_amount_yuan=guifan_you_xian_shuzhi(row.get("amount_yuan")),
            atr_pct=guifan_you_xian_shuzhi(row.get("atr_14_pct")),
            trading_settings=trading_settings,
        )
        values.append(koujian_jiaoyi_chengben(float(gross_return), cost_rate) if cost_rate is not None else np.nan)
    return np.asarray(values, dtype=float)


def _daily_relevance_labels(
    dates: pd.Series,
    net_returns: np.ndarray,
    *,
    grades: int,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "net_return": np.asarray(net_returns, dtype=float),
            "row_order": np.arange(len(net_returns)),
        }
    )
    labels = np.zeros(len(frame), dtype=np.int32)
    for _, group in frame.dropna(subset=["date", "net_return"]).groupby("date", sort=False):
        if len(group) < 2 or group["net_return"].nunique() < 2:
            continue
        percentile = group["net_return"].rank(method="average", pct=True).to_numpy(dtype=float)
        relevance = np.minimum(grades - 1, np.floor(percentile * grades - 1e-12)).astype(np.int32)
        labels[group["row_order"].to_numpy(dtype=int)] = relevance
    return labels


def _fit_ranking_model(
    *,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_columns: list[str],
    net_target: np.ndarray,
    model_config: dict[str, Any],
    time_decay_half_life: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one LambdaMART ranker with each signal date used as a qid group."""
    try:
        from xgboost import XGBRanker
    except ImportError as exc:
        raise RuntimeError("缺少xgboost依赖，请重新执行 python -m pip install -e .") from exc

    work = train[["trade_date", "ts_code"] + feature_columns].copy()
    work["_net_target"] = np.asarray(net_target, dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["trade_date", "_net_target"])
    group_status = work.groupby("trade_date")["_net_target"].agg(["size", "nunique"])
    usable_dates = group_status[(group_status["size"] >= 2) & (group_status["nunique"] >= 2)].index
    work = work[work["trade_date"].isin(usable_dates)].sort_values(["trade_date", "ts_code"])
    if len(work) < 100 or len(usable_dates) < 20:
        raise RuntimeError(
            f"学习排序样本不足：{len(work)}条、{len(usable_dates)}个有效交易日；至少需要100条和20日"
        )

    grades = int(model_config.get("ranking_relevance_grades", 5))
    relevance = _daily_relevance_labels(work["trade_date"], work["_net_target"].to_numpy(), grades=grades)
    qid = pd.factorize(work["trade_date"], sort=True)[0].astype(np.int32)
    clipper = _feature_clipper(model_config)
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(clipper.fit_transform(work[feature_columns]))
    predict_values = imputer.transform(clipper.transform(predict[feature_columns]))
    pair_top_k = int(model_config.get("ranking_pair_top_k", 8))
    ranker = XGBRanker(
        objective="rank:ndcg",
        tree_method="hist",
        n_estimators=int(model_config.get("ranking_n_estimators", model_config.get("max_iter", 180))),
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        max_depth=int(model_config.get("max_depth", 4)),
        min_child_weight=max(1.0, float(model_config.get("min_samples_leaf", 30)) / 5.0),
        reg_lambda=float(model_config.get("l2_regularization", 1.0)),
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=pair_top_k,
        eval_metric=["ndcg@3", "ndcg@8"],
        random_state=int(model_config.get("random_state", 42)),
        n_jobs=-1,
        verbosity=0,
    )
    group_dates = work["trade_date"].drop_duplicates().reset_index(drop=True)
    group_weight = _time_decay_weights(
        group_dates,
        time_decay_half_life,
        float(model_config.get("time_decay_min_weight", 0.10)),
    )
    ranker.fit(
        train_values,
        relevance,
        qid=qid,
        sample_weight=group_weight,
        verbose=False,
    )
    prediction = np.asarray(ranker.predict(predict_values), dtype=float)
    return prediction, {
        "status": "ok",
        "model": "XGBRanker(LambdaMART, rank:ndcg)",
        "training_samples": int(len(work)),
        "qid_groups": int(len(usable_dates)),
        "relevance_grades": int(grades),
        "pair_method": "topk",
        "pairs_per_sample": int(pair_top_k),
        "metrics": ["NDCG@3", "NDCG@8"],
        "target": "同一交易日股票未来成本后收益的分档排名",
        "time_decay_half_life_dates": float(time_decay_half_life),
    }


def _daily_ndcg(
    dates: pd.Series,
    relevance: np.ndarray,
    predicted: np.ndarray,
    *,
    k: int,
) -> tuple[float, int]:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "relevance": np.asarray(relevance, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    values: list[float] = []
    for _, group in frame.groupby("date"):
        if len(group) < 2 or group["relevance"].nunique() < 2:
            continue
        value = ndcg_score(
            group[["relevance"]].to_numpy(dtype=float).T,
            group[["predicted"]].to_numpy(dtype=float).T,
            k=min(int(k), len(group)),
        )
        if math.isfinite(float(value)):
            values.append(float(value))
    return (float(np.mean(values)), len(values)) if values else (0.0, 0)


def _confidence_label(score: float) -> str:
    return _quality_label(float(score))


def _select_stable_features(
    frame: pd.DataFrame,
    candidate_features: list[str],
    target_column: str,
    model_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Select compact factors using the shared daily factor engine."""
    return select_fold_factor_features(frame, candidate_features, target_column, model_config)


def _select_stable_features_legacy(
    frame: pd.DataFrame,
    candidate_features: list[str],
    target_column: str,
    model_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Legacy implementation retained for artifact compatibility only."""
    enabled = bool(model_config.get("factor_stability_enabled", True))
    slices = max(2, int(model_config.get("factor_stability_slices", 3)))
    minimum_valid_slices = max(1, int(model_config.get("factor_min_valid_slices", 2)))
    minimum_sign_agreement = float(model_config.get("factor_min_sign_agreement", 0.67))
    minimum_abs_ic = float(model_config.get("factor_min_abs_mean_rank_ic", 0.005))
    minimum_features = max(1, int(model_config.get("factor_min_features", 12)))
    coverage_threshold = float(model_config.get("min_feature_coverage", 0.20))
    coverage = frame[candidate_features].notna().mean()
    eligible = [
        feature
        for feature in candidate_features
        if float(coverage.get(feature, 0.0)) >= coverage_threshold
    ]
    if not enabled:
        return eligible, {
            "status": "disabled",
            "selected_features": eligible,
            "selection_scope": "training_window_only",
        }

    dates = [pd.Timestamp(value) for value in sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique())]
    date_slices = [list(values) for values in np.array_split(np.asarray(dates, dtype=object), slices) if len(values)]
    diagnostics: dict[str, Any] = {}
    ranked: list[tuple[tuple[float, float, float], str]] = []
    selected: list[str] = []
    for feature in eligible:
        values: list[float] = []
        for date_slice in date_slices:
            part = frame[pd.to_datetime(frame["trade_date"]).isin(date_slice)]
            x = pd.to_numeric(part[feature], errors="coerce")
            y = pd.to_numeric(part[target_column], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 30 or int(x[valid].nunique()) < 2 or int(y[valid].nunique()) < 2:
                continue
            correlation = float(spearmanr(x[valid], y[valid]).statistic)
            if math.isfinite(correlation):
                values.append(correlation)
        mean_ic = float(np.mean(values)) if values else 0.0
        sign_agreement = (
            max(sum(value > 0 for value in values), sum(value < 0 for value in values)) / len(values)
            if values
            else 0.0
        )
        is_stable = bool(
            len(values) >= minimum_valid_slices
            and abs(mean_ic) >= minimum_abs_ic
            and sign_agreement >= minimum_sign_agreement
        )
        diagnostics[feature] = {
            "coverage": round(float(coverage[feature]), 4),
            "slice_rank_ic": [round(value, 6) for value in values],
            "mean_rank_ic": round(mean_ic, 6),
            "sign_agreement": round(float(sign_agreement), 4),
            "stable": is_stable,
        }
        ranked.append(((float(is_stable), abs(mean_ic), float(coverage[feature])), feature))
        if is_stable:
            selected.append(feature)
    fallback_used = False
    if len(selected) < min(minimum_features, len(eligible)):
        fallback_used = True
        for _, feature in sorted(ranked, reverse=True):
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= min(minimum_features, len(eligible)):
                break
    selected = [feature for feature in candidate_features if feature in set(selected)]
    return selected, {
        "status": "ok" if selected else "no_eligible_feature",
        "selection_scope": "training_window_only",
        "slices": int(len(date_slices)),
        "candidate_count": int(len(candidate_features)),
        "coverage_eligible_count": int(len(eligible)),
        "selected_count": int(len(selected)),
        "selected_features": selected,
        "fallback_to_strongest_factors": fallback_used,
        "thresholds": {
            "minimum_coverage": coverage_threshold,
            "minimum_valid_slices": minimum_valid_slices,
            "minimum_sign_agreement": minimum_sign_agreement,
            "minimum_abs_mean_rank_ic": minimum_abs_ic,
            "minimum_features": minimum_features,
        },
        "factor_diagnostics": diagnostics,
    }


def _fit_direction_probabilities(
    *,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    predict_features: pd.DataFrame,
    model_config: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = (np.asarray(train_target, dtype=float) > 0).astype(int)
    if len(np.unique(labels)) < 2:
        probability = float(labels[0]) if len(labels) else 0.5
        return np.full(len(predict_features), probability, dtype=float), {
            "status": "single_training_class",
            "training_positive_rate": probability,
        }
    model = Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=float(model_config.get("direction_logistic_c", 0.5)),
                    max_iter=int(model_config.get("direction_logistic_max_iter", 500)),
                    random_state=int(model_config.get("random_state", 42)),
                ),
            ),
        ]
    )
    try:
        fit_kwargs = {"model__sample_weight": np.asarray(sample_weight, dtype=float)} if sample_weight is not None else {}
        model.fit(train_features, labels, **fit_kwargs)
        probability = np.asarray(model.predict_proba(predict_features)[:, 1], dtype=float)
        return probability, {
            "status": "ok",
            "training_samples": int(len(labels)),
            "training_positive_rate": round(float(np.mean(labels)), 6),
            "model": "winsorized_robust_scaled_logistic_regression",
        }
    except Exception as exc:
        probability = float(np.mean(labels))
        return np.full(len(predict_features), probability, dtype=float), {
            "status": "fallback_to_training_base_rate",
            "training_samples": int(len(labels)),
            "training_positive_rate": round(probability, 6),
            "error": str(exc),
        }


def _probability_reliability_bins(actual: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = (np.asarray(actual, dtype=float) > 0).astype(int)
    probabilities = np.asarray(probability, dtype=float)
    for lower, upper in zip(np.linspace(0.0, 0.8, 5), np.linspace(0.2, 1.0, 5)):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if not mask.any():
            continue
        rows.append(
            {
                "probability_bin": [round(float(lower), 2), round(float(upper), 2)],
                "samples": int(mask.sum()),
                "mean_predicted_probability": round(float(np.mean(probabilities[mask])), 6),
                "actual_positive_rate": round(float(np.mean(labels[mask])), 6),
            }
        )
    return rows


def _calibrate_direction_probability(
    *,
    actual: np.ndarray,
    raw_probability: np.ndarray,
    dates: pd.Series,
    latest_raw_probability: float,
    model_config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "actual": np.asarray(actual, dtype=float),
            "raw": np.asarray(raw_probability, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    minimum_samples = int(model_config.get("probability_calibration_min_samples", 120))
    evaluation_ratio = float(model_config.get("probability_calibration_evaluation_ratio", 0.30))
    minimum_improvement = float(model_config.get("probability_calibration_min_brier_improvement", 0.0005))
    raw_latest = float(np.clip(latest_raw_probability, 0.0, 1.0))
    labels = (frame["actual"].to_numpy(dtype=float) > 0).astype(int)
    raw = frame["raw"].to_numpy(dtype=float).clip(0.0, 1.0)
    base = {
        "status": "raw_probability_retained",
        "method": "uncalibrated_logistic_probability",
        "oos_samples": int(len(frame)),
        "raw_oos_brier_score": round(float(brier_score_loss(labels, raw)), 6) if len(frame) else None,
        "reliability_bins": _probability_reliability_bins(frame["actual"].to_numpy(dtype=float), raw),
    }
    if len(frame) < minimum_samples:
        base["reason"] = f"样本外方向概率只有{len(frame)}个，少于校准门槛{minimum_samples}"
        return raw_latest, base
    split = max(minimum_samples // 2, int(len(frame) * (1.0 - evaluation_ratio)))
    split = min(split, len(frame) - max(30, minimum_samples // 4))
    evaluation_start = pd.Timestamp(frame["date"].iloc[split])
    evaluation_dates = [
        pd.Timestamp(value)
        for value in sorted(frame.loc[frame["date"] >= evaluation_start, "date"].unique())
    ]
    evaluation_indices: list[int] = []
    evaluation_calibrated_values: list[float] = []
    evaluation_baseline_values: list[float] = []
    for evaluation_date in evaluation_dates:
        train_mask = frame["date"] < evaluation_date
        evaluation_mask = frame["date"] == evaluation_date
        train_index = np.flatnonzero(train_mask.to_numpy())
        current_index = np.flatnonzero(evaluation_mask.to_numpy())
        if (
            len(train_index) < minimum_samples // 2
            or not len(current_index)
            or len(np.unique(labels[train_index])) < 2
            or len(np.unique(raw[train_index])) < 2
        ):
            continue
        calibration = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibration.fit(raw[train_index], labels[train_index])
        current_calibrated = np.asarray(
            calibration.predict(raw[current_index]),
            dtype=float,
        ).clip(0.0, 1.0)
        evaluation_indices.extend(int(value) for value in current_index)
        evaluation_calibrated_values.extend(float(value) for value in current_calibrated)
        historical_rate = float(np.mean(labels[train_index]))
        evaluation_baseline_values.extend([historical_rate] * len(current_index))
    if not evaluation_indices:
        base["reason"] = "没有形成只使用更早日期样本的滚动概率校准评估"
        return raw_latest, base
    evaluation_index = np.asarray(evaluation_indices, dtype=int)
    evaluation_labels = labels[evaluation_index]
    evaluation_raw = raw[evaluation_index]
    evaluation_calibrated = np.asarray(evaluation_calibrated_values, dtype=float)
    evaluation_baseline = np.asarray(evaluation_baseline_values, dtype=float)
    if len(np.unique(evaluation_labels)) < 2:
        base["reason"] = "滚动校准评估窗只有单一方向类别"
        return raw_latest, base
    raw_brier = float(brier_score_loss(evaluation_labels, evaluation_raw))
    calibrated_brier = float(brier_score_loss(evaluation_labels, evaluation_calibrated))
    historical_rate_brier = float(
        brier_score_loss(evaluation_labels, evaluation_baseline)
    )
    base["time_ordered_evaluation"] = {
        "method": "rolling_cross_fitted_isotonic_by_date",
        "initial_calibration_samples": int(split),
        "evaluation_samples": int(len(evaluation_index)),
        "evaluation_dates": int(len(set(frame["date"].iloc[evaluation_index]))),
        "raw_brier_score": round(raw_brier, 6),
        "isotonic_brier_score": round(calibrated_brier, 6),
        "historical_positive_rate_brier_score": round(historical_rate_brier, 6),
        "improvement": round(raw_brier - calibrated_brier, 6),
        "improvement_vs_historical_positive_rate": round(
            historical_rate_brier - calibrated_brier,
            6,
        ),
    }
    if (
        raw_brier - calibrated_brier < minimum_improvement
        or historical_rate_brier - calibrated_brier < minimum_improvement
    ):
        base["reason"] = "滚动保序校准未同时优于原始概率和历史上涨比例基准"
        return raw_latest, base
    production_calibration = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    production_calibration.fit(raw, labels)
    calibrated_latest = float(production_calibration.predict([raw_latest])[0])
    return calibrated_latest, {
        **base,
        "status": "calibrated",
        "method": "rolling_cross_fitted_isotonic_then_refit_on_all_oos",
        "latest_raw_probability": round(raw_latest, 6),
        "latest_calibrated_probability": round(calibrated_latest, 6),
        "reliability_bins": _probability_reliability_bins(
            frame["actual"].to_numpy(dtype=float)[evaluation_index],
            evaluation_calibrated,
        ),
    }


def _rolling_conformal_interval(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: pd.Series,
    latest_prediction: float,
    model_config: dict[str, Any],
) -> tuple[list[float] | None, dict[str, Any]]:
    coverage = float(model_config.get("conformal_coverage", 0.80))
    minimum_samples = int(model_config.get("conformal_min_samples", 80))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    residuals = np.abs(frame["actual"].to_numpy(dtype=float) - frame["predicted"].to_numpy(dtype=float))
    if len(residuals) < minimum_samples:
        return None, {
            "status": "insufficient_oos_samples",
            "samples": int(len(residuals)),
            "minimum_samples": minimum_samples,
            "target_coverage": coverage,
        }

    def radius(values: np.ndarray) -> float:
        adjusted = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
        return float(np.quantile(values, adjusted, method="higher"))

    hits: list[bool] = []
    for index in range(minimum_samples, len(frame)):
        current_radius = radius(residuals[:index])
        hits.append(bool(residuals[index] <= current_radius))
    latest_radius = radius(residuals)
    interval = [float(latest_prediction - latest_radius), float(latest_prediction + latest_radius)]
    return interval, {
        "status": "ok",
        "method": "rolling_split_conformal_absolute_residual",
        "target_coverage": coverage,
        "calibration_samples": int(len(residuals)),
        "rolling_evaluation_samples": int(len(hits)),
        "rolling_empirical_coverage": round(float(np.mean(hits)), 6) if hits else None,
        "latest_radius": round(latest_radius, 6),
    }


def _rolling_cqr_interval(
    *,
    actual: np.ndarray,
    lower_prediction: np.ndarray,
    upper_prediction: np.ndarray,
    dates: pd.Series,
    latest_lower: float,
    latest_upper: float,
    model_config: dict[str, Any],
) -> tuple[list[float] | None, dict[str, Any]]:
    coverage = float(model_config.get("conformal_coverage", 0.80))
    minimum_samples = int(model_config.get("conformal_min_samples", 80))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "actual": np.asarray(actual, dtype=float),
            "lower": np.asarray(lower_prediction, dtype=float),
            "upper": np.asarray(upper_prediction, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    if len(frame) < minimum_samples:
        return None, {
            "status": "insufficient_oos_samples",
            "samples": int(len(frame)),
            "minimum_samples": minimum_samples,
            "target_coverage": coverage,
        }
    lower = np.minimum(
        frame["lower"].to_numpy(dtype=float),
        frame["upper"].to_numpy(dtype=float),
    )
    upper = np.maximum(
        frame["lower"].to_numpy(dtype=float),
        frame["upper"].to_numpy(dtype=float),
    )
    actual_values = frame["actual"].to_numpy(dtype=float)
    scores = np.maximum(lower - actual_values, actual_values - upper)

    def correction(values: np.ndarray) -> float:
        adjusted = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
        return max(0.0, float(np.quantile(values, adjusted, method="higher")))

    hits: list[bool] = []
    evaluation_dates = 0
    for current_date in sorted(frame["date"].unique()):
        prior_mask = frame["date"] < current_date
        current_mask = frame["date"] == current_date
        prior_count = int(prior_mask.sum())
        if prior_count < minimum_samples:
            continue
        current_correction = correction(scores[prior_mask.to_numpy()])
        current_indices = np.flatnonzero(current_mask.to_numpy())
        hits.extend(
            bool(
                lower[index] - current_correction
                <= actual_values[index]
                <= upper[index] + current_correction
            )
            for index in current_indices
        )
        evaluation_dates += 1
    latest_correction = correction(scores)
    raw_lower = min(float(latest_lower), float(latest_upper))
    raw_upper = max(float(latest_lower), float(latest_upper))
    interval = [
        raw_lower - latest_correction,
        raw_upper + latest_correction,
    ]
    raw_coverage = float(np.mean((actual_values >= lower) & (actual_values <= upper)))
    return interval, {
        "status": "ok",
        "method": "rolling_conformalized_quantile_regression",
        "target_coverage": coverage,
        "calibration_samples": int(len(frame)),
        "rolling_evaluation_samples": int(len(hits)),
        "rolling_evaluation_dates": int(evaluation_dates),
        "rolling_empirical_coverage": round(float(np.mean(hits)), 6) if hits else None,
        "raw_quantile_interval_coverage": round(raw_coverage, 6),
        "latest_correction": round(latest_correction, 6),
        "latest_raw_interval": [round(raw_lower, 6), round(raw_upper, 6)],
    }


def _fold_stability(folds: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [fold for fold in folds if fold.get("status") == "ok"]
    if not successful:
        return {"status": "unavailable", "folds": 0}
    result: dict[str, Any] = {
        "status": "ok",
        "folds": int(len(successful)),
        "passed_rate": round(float(np.mean([bool(fold.get("fold_passed")) for fold in successful])), 6),
    }
    for field in ["direction_accuracy", "skill_vs_median_baseline", "mean_daily_rank_ic"]:
        values = np.asarray([float(fold[field]) for fold in successful if fold.get(field) is not None], dtype=float)
        result[field] = {
            "mean": round(float(np.mean(values)), 6) if len(values) else None,
            "std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0 if len(values) else None,
            "minimum": round(float(np.min(values)), 6) if len(values) else None,
            "maximum": round(float(np.max(values)), 6) if len(values) else None,
        }
    return result


def _regime_stability(
    oof: pd.DataFrame,
    *,
    regime_column: str,
) -> dict[str, Any]:
    if regime_column not in oof.columns:
        return {"status": "unavailable", "reason": f"缺少{regime_column}"}
    frame = oof.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual", "predicted", regime_column]
    )
    if len(frame) < 90 or frame[regime_column].nunique() < 3:
        return {"status": "unavailable", "samples": int(len(frame))}
    lower, upper = frame[regime_column].quantile([1 / 3, 2 / 3]).tolist()
    groups = {
        "weak_market": frame[frame[regime_column] <= lower],
        "sideways_market": frame[(frame[regime_column] > lower) & (frame[regime_column] < upper)],
        "strong_market": frame[frame[regime_column] >= upper],
    }
    metrics: dict[str, Any] = {}
    for name, group in groups.items():
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        metrics[name] = {
            "samples": int(len(group)),
            "mae": round(float(mean_absolute_error(actual, predicted)), 6) if len(group) else None,
            "direction_accuracy": round(float(np.mean((actual > 0) == (predicted > 0))), 6) if len(group) else None,
            "mean_actual_return": round(float(np.mean(actual)), 6) if len(group) else None,
            "mean_predicted_return": round(float(np.mean(predicted)), 6) if len(group) else None,
        }
    return {
        "status": "ok",
        "regime_factor": regime_column,
        "tercile_cutoffs": [round(float(lower), 6), round(float(upper), 6)],
        "regimes": metrics,
    }


def _experiment_fingerprint(
    *,
    feature_columns: list[str],
    target_definition: str,
    split_method: str,
    model_config: dict[str, Any],
) -> str:
    payload = {
        "contract": "daily_k_quant_research_v3",
        "features": feature_columns,
        "target": target_definition,
        "split": split_method,
        "model_config": model_config,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _select_ensemble_weight(
    actual: np.ndarray,
    tree_prediction: np.ndarray,
    linear_prediction: np.ndarray,
    model_config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Select a convex tree weight using only already out-of-sample observations."""
    enabled = bool(model_config.get("ensemble_enabled", True))
    default_weight = float(model_config.get("ensemble_default_tree_weight", 0.75))
    minimum_samples = int(model_config.get("ensemble_min_calibration_samples", 80))
    arrays = [np.asarray(value, dtype=float) for value in (actual, tree_prediction, linear_prediction)]
    finite = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])
    clean_actual, clean_tree, clean_linear = (value[finite] for value in arrays)
    if not enabled:
        return 1.0, {
            "status": "disabled",
            "selection_method": "tree_only_by_configuration",
            "calibration_samples": int(len(clean_actual)),
            "tree_weight": 1.0,
            "linear_weight": 0.0,
        }
    if len(clean_actual) < minimum_samples:
        return default_weight, {
            "status": "insufficient_oos_calibration_samples",
            "selection_method": "configured_default_until_enough_oos_samples",
            "calibration_samples": int(len(clean_actual)),
            "minimum_calibration_samples": minimum_samples,
            "tree_weight": round(default_weight, 4),
            "linear_weight": round(1.0 - default_weight, 4),
        }

    raw_grid = model_config.get("ensemble_tree_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    grid = sorted({float(value) for value in raw_grid if 0.0 <= float(value) <= 1.0})
    if not grid:
        grid = [default_weight]
    candidates: list[tuple[float, float, np.ndarray]] = []
    for tree_weight in grid:
        blended = tree_weight * clean_tree + (1.0 - tree_weight) * clean_linear
        candidates.append((float(mean_absolute_error(clean_actual, blended)), tree_weight, blended))
    selected_mae, selected_weight, selected_prediction = min(
        candidates,
        key=lambda item: (item[0], abs(item[1] - default_weight)),
    )
    return selected_weight, {
        "status": "selected_from_oos_predictions",
        "selection_method": "minimum_mae_on_prior_oos_grid",
        "calibration_samples": int(len(clean_actual)),
        "tree_weight": round(float(selected_weight), 4),
        "linear_weight": round(float(1.0 - selected_weight), 4),
        "tree_mae": round(float(mean_absolute_error(clean_actual, clean_tree)), 6),
        "linear_mae": round(float(mean_absolute_error(clean_actual, clean_linear)), 6),
        "ensemble_mae": round(float(selected_mae), 6),
        "ensemble_direction_accuracy": round(
            float(np.mean((selected_prediction > 0) == (clean_actual > 0))),
            6,
        ),
        "candidate_tree_weights": grid,
    }


def _blend_component_predictions(
    tree_prediction: np.ndarray,
    linear_prediction: np.ndarray,
    tree_weight: float,
) -> np.ndarray:
    return tree_weight * np.asarray(tree_prediction, dtype=float) + (
        1.0 - tree_weight
    ) * np.asarray(linear_prediction, dtype=float)


def _nested_training_ensemble_weight(
    *,
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_date_column: str,
    clip_low: float,
    clip_high: float,
    model_config: dict[str, Any],
    time_decay_half_life: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Calibrate a holdout-safe ensemble weight on the tail of the training period."""
    dates = [pd.Timestamp(value) for value in sorted(train["trade_date"].dropna().unique())]
    ratio = float(model_config.get("ensemble_calibration_ratio", 0.15))
    minimum_dates = int(model_config.get("ensemble_min_calibration_dates", 20))
    if len(dates) < minimum_dates * 2:
        return _select_ensemble_weight(np.array([]), np.array([]), np.array([]), model_config)
    calibration_dates = max(minimum_dates, int(len(dates) * ratio))
    calibration_dates = min(calibration_dates, len(dates) - minimum_dates)
    cutoff = dates[-calibration_dates]
    inner_train = train[
        (train["trade_date"] < cutoff)
        & (pd.to_datetime(train[target_date_column]) < cutoff)
    ]
    calibration = train[train["trade_date"] >= cutoff]
    minimum_samples = int(model_config.get("ensemble_min_calibration_samples", 80))
    if len(inner_train) < minimum_samples or len(calibration) < minimum_samples:
        weight, diagnostics = _select_ensemble_weight(
            np.array([]), np.array([]), np.array([]), model_config
        )
        diagnostics.update(
            {
                "nested_train_samples": int(len(inner_train)),
                "nested_calibration_samples": int(len(calibration)),
            }
        )
        return weight, diagnostics
    inner_target = np.clip(
        inner_train[target_column].astype(float).to_numpy(),
        clip_low,
        clip_high,
    )
    tree_prediction, linear_prediction = _fit_model_components(
        train_features=inner_train[feature_columns],
        train_target=inner_target,
        predict_features=calibration[feature_columns],
        model_config=model_config,
        sample_weight=_time_decay_weights(
            inner_train["trade_date"],
            time_decay_half_life,
            float(model_config.get("time_decay_min_weight", 0.10)),
        ),
    )
    tree_prediction = np.clip(tree_prediction, clip_low, clip_high)
    linear_prediction = np.clip(linear_prediction, clip_low, clip_high)
    weight, diagnostics = _select_ensemble_weight(
        calibration[target_column].astype(float).to_numpy(),
        tree_prediction,
        linear_prediction,
        model_config,
    )
    diagnostics.update(
        {
            "selection_scope": "nested_tail_of_outer_training_only",
            "calibration_start": cutoff.strftime("%Y-%m-%d"),
            "nested_train_samples": int(len(inner_train)),
            "nested_calibration_samples": int(len(calibration)),
        }
    )
    return weight, diagnostics


def _top_n_validation_metrics(
    validation_frame: pd.DataFrame,
    actual: np.ndarray,
    ranking_score: np.ndarray,
    *,
    horizon: int,
    budget_yuan: float,
    scenario: CostScenario,
    top_n: int,
    trading_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_columns = ["trade_date", "ts_code", f"entry_open_t{horizon}"]
    evaluation_columns.extend(
        column for column in ["amount_yuan", "atr_14_pct"] if column in validation_frame.columns
    )
    evaluation = validation_frame[evaluation_columns].copy()
    evaluation["net_actual"] = _historical_net_returns(
        validation_frame,
        actual,
        horizon=horizon,
        budget_yuan=budget_yuan,
        scenario=scenario,
        trading_settings=trading_settings,
    )
    evaluation["ranking_score"] = np.asarray(ranking_score, dtype=float)
    evaluation = evaluation.dropna(subset=["net_actual", "ranking_score"])

    selected_returns: list[float] = []
    excess_returns: list[float] = []
    for _, group in evaluation.groupby("trade_date"):
        if len(group) < 2:
            continue
        selected = group.nlargest(min(int(top_n), len(group)), "ranking_score")
        selected_return = float(selected["net_actual"].mean())
        selected_returns.append(selected_return)
        excess_returns.append(selected_return - float(group["net_actual"].mean()))
    return {
        "top_n": int(top_n),
        "top_n_days": int(len(selected_returns)),
        "top_n_mean_net_return": round(float(np.mean(selected_returns)), 6) if selected_returns else 0.0,
        "top_n_positive_day_rate": round(float(np.mean(np.asarray(selected_returns) > 0)), 6)
        if selected_returns
        else 0.0,
        "top_n_mean_excess_vs_universe": round(float(np.mean(excess_returns)), 6) if excess_returns else 0.0,
    }




__all__ = [
    "_historical_limit_rates",
    "_one_price_limit_session",
    "goujian_moxing_shuju",
    "_daily_rank_ic",
    "_quality_score",
    "_quality_label",
    "_TrainingQuantileClipper",
    "_feature_clipper",
    "_build_model_pipeline",
    "_build_linear_model_pipeline",
    "_time_decay_weights",
    "_select_time_decay_half_life",
    "_build_quantile_model_pipeline",
    "_fit_quantile_model_components",
    "_fit_model_components",
    "_fit_market_baseline",
    "_historical_net_returns",
    "_daily_relevance_labels",
    "_fit_ranking_model",
    "_daily_ndcg",
    "_confidence_label",
    "_select_stable_features",
    "_select_stable_features_legacy",
    "_fit_direction_probabilities",
    "_probability_reliability_bins",
    "_calibrate_direction_probability",
    "_rolling_conformal_interval",
    "_rolling_cqr_interval",
    "_fold_stability",
    "_regime_stability",
    "_experiment_fingerprint",
    "_select_ensemble_weight",
    "_blend_component_predictions",
    "_nested_training_ensemble_weight",
    "_top_n_validation_metrics",
]
