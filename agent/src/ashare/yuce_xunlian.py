"""单股预测模型的特征筛选、滚动验证与训练实现。

本模块属于预测基础设施层，只接收已经准备好的面板数据，不负责市场数据获取、
交易日编排或最终业务结果组装。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from src.ashare.gupiao_yanjiu import FEATURE_COLUMNS
from src.ashare.jiaoyi_zhixing import jiazai_chengben_jiashe
from src.ashare.moxing_xunlian_api import (
    blend_component_predictions,
    calibrate_direction_probability,
    daily_rank_ic,
    experiment_fingerprint,
    fit_direction_probabilities,
    fit_model_components,
    fit_quantile_model_components,
    fold_stability,
    quality_label,
    quality_score,
    regime_stability,
    rolling_conformal_interval,
    rolling_cqr_interval,
    select_stable_features,
    select_time_decay_half_life,
    select_ensemble_weight,
    time_decay_weights,
    top_n_validation_metrics,
)
from src.ashare.moxing_pinggu import regression_baseline_metrics
from src.ashare.riping_yinzi import DAILY_FACTOR_FEATURE_COLUMNS
from src.ashare.yinzi_gongcheng import (
    FACTOR_ENGINEERING_VERSION,
    engineered_factor_columns,
)


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
SINGLE_STOCK_FEATURE_COLUMNS = list(
    dict.fromkeys(
        FEATURE_COLUMNS
        + SINGLE_STOCK_EXTRA_FEATURES
        + DAILY_FACTOR_FEATURE_COLUMNS
        + engineered_factor_columns()
    )
)


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


def xunlian_chiyouqi_yuce_moxing(
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
        "feature_preprocessing": "共享因子工程；每个滚动训练窗口独立做逐日Rank IC筛选、同组去重和分位数去极值；Ridge与方向Logistic另做稳健缩放",
        "factor_engineering_version": FACTOR_ENGINEERING_VERSION,
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
    _, cost_scenario, _, _ = jiazai_chengben_jiashe("research_reference")

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
            fold_features, factor_selection = select_stable_features(
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
            decay_half_life, time_decay = select_time_decay_half_life(
                train_features=train[fold_features],
                train_target=np.clip(y_train_raw, clip_low, clip_high),
                train_dates=train["trade_date"],
                train_target_dates=train[target_date_column],
                model_config=model_config,
            )
            sample_weight = time_decay_weights(
                train["trade_date"],
                decay_half_life,
                float(model_config.get("time_decay_min_weight", 0.10)),
            )
            tree_weight, ensemble_diagnostics = select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            tree_prediction, linear_prediction = fit_model_components(
                train_features=train[fold_features],
                train_target=np.clip(y_train_raw, clip_low, clip_high),
                predict_features=validation_frame[fold_features],
                model_config=model_config,
                sample_weight=sample_weight,
            )
            raw_direction_probability, direction_model = fit_direction_probabilities(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
                sample_weight=sample_weight,
            )
            quantile_prediction, quantile_model = fit_quantile_model_components(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
                sample_weight=sample_weight,
            )
            tree_prediction = np.clip(tree_prediction, clip_low, clip_high)
            linear_prediction = np.clip(linear_prediction, clip_low, clip_high)
            actual = validation_frame[target_column].astype(float).to_numpy()
            predicted = np.clip(
                blend_component_predictions(tree_prediction, linear_prediction, tree_weight),
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
            rank_ic, rank_days = daily_rank_ic(validation_frame["trade_date"], actual, predicted)
            top_n = top_n_validation_metrics(
                validation_frame,
                actual,
                predicted,
                horizon=horizon,
                budget_yuan=budget_yuan,
                scenario=cost_scenario,
                top_n=validation_top_n,
                trading_settings=None,
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
                "meandaily_rank_ic": round(rank_ic, 6),
                "rank_ic_days": int(rank_days),
                "factor_selection": factor_selection,
                "time_decay": time_decay,
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
        production_time_decay: dict[str, Any] = {"status": "not_trained"}
        production_features: list[str] = []
        if len(usable) >= minimum_train:
            full_y = usable[target_column].astype(float).to_numpy()
            final_clip_low = float(np.nanquantile(full_y, lower_q))
            final_clip_high = float(np.nanquantile(full_y, upper_q))
            production_tree_weight, production_ensemble = select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            production_features, production_factor_selection = select_stable_features(
                usable,
                feature_columns,
                target_column,
                model_config,
            )
            production_decay_half_life, production_time_decay = select_time_decay_half_life(
                train_features=usable[production_features],
                train_target=np.clip(full_y, final_clip_low, final_clip_high),
                train_dates=usable["trade_date"],
                train_target_dates=usable[target_date_column],
                model_config=model_config,
            )
            production_sample_weight = time_decay_weights(
                usable["trade_date"],
                production_decay_half_life,
                float(model_config.get("time_decay_min_weight", 0.10)),
            )
            latest_tree_prediction, latest_linear_prediction = fit_model_components(
                train_features=usable[production_features],
                train_target=np.clip(full_y, final_clip_low, final_clip_high),
                predict_features=latest[production_features],
                model_config=model_config,
                sample_weight=production_sample_weight,
            )
            latest_direction_array, production_direction_model = fit_direction_probabilities(
                train_features=usable[production_features],
                train_target=full_y,
                predict_features=latest[production_features],
                model_config=model_config,
                sample_weight=production_sample_weight,
            )
            latest_quantile_prediction, production_quantile_model = (
                fit_quantile_model_components(
                    train_features=usable[production_features],
                    train_target=full_y,
                    predict_features=latest[production_features],
                    model_config=model_config,
                    sample_weight=production_sample_weight,
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
                blend_component_predictions(
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
            "production_time_decay": production_time_decay,
            "experiment_fingerprint": experiment_fingerprint(
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
            rank_ic, rank_days = daily_rank_ic(oof["trade_date"], actual, predicted)
            top_n = top_n_validation_metrics(
                oof.rename(columns={entry_open_column: f"entry_open_t{horizon}"}),
                actual,
                predicted,
                horizon=horizon,
                budget_yuan=budget_yuan,
                scenario=cost_scenario,
                top_n=validation_top_n,
                trading_settings=None,
            )
            residual = actual - predicted
            residual_low, residual_high = np.nanquantile(residual, [0.10, 0.90])
            nearest_count = min(len(oof), max(40, len(oof) // 5))
            nearest_index = np.argsort(np.abs(predicted - latest_prediction))[:nearest_count]
            nearest_actual = actual[nearest_index]
            positive_probability = float(np.mean(nearest_actual > 0)) if nearest_count else None
            empirical_low, empirical_high = np.nanquantile(nearest_actual, [0.10, 0.90])
            calibrated_probability, probability_calibration = calibrate_direction_probability(
                actual=actual,
                raw_probability=oof["raw_direction_probability"].to_numpy(dtype=float),
                dates=oof["trade_date"],
                latest_raw_probability=float(latest_raw_direction_probability),
                model_config=model_config,
            )
            conformal_interval, conformal_diagnostics = rolling_conformal_interval(
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
                cqr_interval, cqr_diagnostics = rolling_cqr_interval(
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
            quality = quality_score(
                train_count=len(usable),
                direction_accuracy=direction,
                rank_ic=rank_ic,
                skill_vs_baseline=skill,
            )
            minimum_rank_ic = float(model_config.get("min_meandaily_rank_ic", 0.01))
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
                "meandaily_rank_ic": round(rank_ic, 6),
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
                "fold_stability": fold_stability(fold_records),
                "marketregime_stability": regime_stability(oof, regime_column=regime_column),
                "quality_score": round(quality, 4),
                "quality_label": quality_label(quality),
                "validation_passed": validation_passed,
                "validation_thresholds": {
                    "completed_folds": expected_folds,
                    "passed_folds": min_passed_folds,
                    "final_holdout_must_pass": True,
                    "direction_accuracy": minimum_direction,
                    "meandaily_rank_ic": minimum_rank_ic,
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
    validation["overallquality_score"] = round(float(np.mean(qualities)), 4) if qualities else 0.0
    validation["overallquality_label"] = quality_label(float(validation["overallquality_score"]))
    return predictions, validation


def xunlian_weilai_shoupan_yuce_moxing(
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
        "feature_preprocessing": "共享因子工程；每个滚动训练窗口独立做逐日Rank IC筛选、同组去重和分位数去极值；Ridge与方向Logistic另做稳健缩放",
        "factor_engineering_version": FACTOR_ENGINEERING_VERSION,
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
    minimum_rank_ic = float(model_config.get("min_meandaily_rank_ic", 0.01))
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
            fold_features, factor_selection = select_stable_features(
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
            decay_half_life, time_decay = select_time_decay_half_life(
                train_features=train[fold_features],
                train_target=np.clip(y_train_raw, clip_low, clip_high),
                train_dates=train["trade_date"],
                train_target_dates=train[target_date_column],
                model_config=model_config,
            )
            sample_weight = time_decay_weights(
                train["trade_date"],
                decay_half_life,
                float(model_config.get("time_decay_min_weight", 0.10)),
            )
            tree_weight, ensemble_diagnostics = select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            tree_prediction, linear_prediction = fit_model_components(
                train_features=train[fold_features],
                train_target=np.clip(y_train_raw, clip_low, clip_high),
                predict_features=validation_frame[fold_features],
                model_config=model_config,
                sample_weight=sample_weight,
            )
            raw_direction_probability, direction_model = fit_direction_probabilities(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
                sample_weight=sample_weight,
            )
            quantile_prediction, quantile_model = fit_quantile_model_components(
                train_features=train[fold_features],
                train_target=y_train_raw,
                predict_features=validation_frame[fold_features],
                model_config=model_config,
                sample_weight=sample_weight,
            )
            tree_prediction = np.clip(tree_prediction, clip_low, clip_high)
            linear_prediction = np.clip(linear_prediction, clip_low, clip_high)
            actual = validation_frame[target_column].astype(float).to_numpy()
            predicted = np.clip(
                blend_component_predictions(tree_prediction, linear_prediction, tree_weight),
                clip_low,
                clip_high,
            )
            baseline_value = float(np.median(y_train_raw))
            baseline = np.full(len(actual), baseline_value)
            mae = float(mean_absolute_error(actual, predicted))
            baseline_mae = float(mean_absolute_error(actual, baseline))
            skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
            direction = float(np.mean((predicted > 0) == (actual > 0)))
            rank_ic, rank_days = daily_rank_ic(validation_frame["trade_date"], actual, predicted)
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
                    "meandaily_rank_ic": round(rank_ic, 6),
                    "rank_ic_days": int(rank_days),
                    "factor_selection": factor_selection,
                    "time_decay": time_decay,
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
        production_time_decay: dict[str, Any] = {"status": "not_trained"}
        production_features: list[str] = []
        if len(usable) >= minimum_train:
            full_y = usable[target_column].astype(float).to_numpy()
            final_clip_low = float(np.nanquantile(full_y, lower_q))
            final_clip_high = float(np.nanquantile(full_y, upper_q))
            production_tree_weight, production_ensemble = select_ensemble_weight(
                np.concatenate(prior_actual) if prior_actual else np.array([]),
                np.concatenate(prior_tree_prediction) if prior_tree_prediction else np.array([]),
                np.concatenate(prior_linear_prediction) if prior_linear_prediction else np.array([]),
                model_config,
            )
            production_features, production_factor_selection = select_stable_features(
                usable,
                feature_columns,
                target_column,
                model_config,
            )
            production_decay_half_life, production_time_decay = select_time_decay_half_life(
                train_features=usable[production_features],
                train_target=np.clip(full_y, final_clip_low, final_clip_high),
                train_dates=usable["trade_date"],
                train_target_dates=usable[target_date_column],
                model_config=model_config,
            )
            production_sample_weight = time_decay_weights(
                usable["trade_date"],
                production_decay_half_life,
                float(model_config.get("time_decay_min_weight", 0.10)),
            )
            latest_tree_prediction, latest_linear_prediction = fit_model_components(
                train_features=usable[production_features],
                train_target=np.clip(full_y, final_clip_low, final_clip_high),
                predict_features=latest[production_features],
                model_config=model_config,
                sample_weight=production_sample_weight,
            )
            latest_direction_array, production_direction_model = fit_direction_probabilities(
                train_features=usable[production_features],
                train_target=full_y,
                predict_features=latest[production_features],
                model_config=model_config,
                sample_weight=production_sample_weight,
            )
            latest_quantile_prediction, production_quantile_model = (
                fit_quantile_model_components(
                    train_features=usable[production_features],
                    train_target=full_y,
                    predict_features=latest[production_features],
                    model_config=model_config,
                    sample_weight=production_sample_weight,
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
                    blend_component_predictions(
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
            "production_time_decay": production_time_decay,
            "experiment_fingerprint": experiment_fingerprint(
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
            rank_ic, rank_days = daily_rank_ic(oof["trade_date"], actual, predicted)
            residual = actual - predicted
            residual_low, residual_high = np.nanquantile(residual, [0.10, 0.90])
            nearest_count = min(len(oof), max(40, len(oof) // 5))
            nearest_index = np.argsort(np.abs(predicted - latest_prediction))[:nearest_count]
            nearest_actual = actual[nearest_index]
            positive_probability = (
                float(np.mean(nearest_actual > 0)) if nearest_count else None
            )
            empirical_low, empirical_high = np.nanquantile(nearest_actual, [0.10, 0.90])
            calibrated_probability, probability_calibration = calibrate_direction_probability(
                actual=actual,
                raw_probability=oof["raw_direction_probability"].to_numpy(dtype=float),
                dates=oof["trade_date"],
                latest_raw_probability=float(latest_raw_direction_probability),
                model_config=model_config,
            )
            conformal_interval, conformal_diagnostics = rolling_conformal_interval(
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
                cqr_interval, cqr_diagnostics = rolling_cqr_interval(
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
            quality = quality_score(
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
                    "meandaily_rank_ic": round(rank_ic, 6),
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
                    "fold_stability": fold_stability(fold_records),
                    "marketregime_stability": regime_stability(oof, regime_column=regime_column),
                    "quality_score": round(quality, 4),
                    "quality_label": quality_label(quality),
                    "validation_passed": validation_passed,
                    "validation_thresholds": {
                        "completed_folds": expected_folds,
                        "passed_folds": min_passed_folds,
                        "final_holdout_must_pass": True,
                        "direction_accuracy": minimum_direction,
                        "meandaily_rank_ic": minimum_rank_ic,
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
    validation["overallquality_score"] = round(float(np.mean(qualities)), 4) if qualities else 0.0
    validation["overallquality_label"] = quality_label(float(validation["overallquality_score"]))
    return predictions, validation


__all__ = [
    "SINGLE_STOCK_FEATURE_COLUMNS",
    "xunlian_chiyouqi_yuce_moxing",
    "xunlian_weilai_shoupan_yuce_moxing",
]
