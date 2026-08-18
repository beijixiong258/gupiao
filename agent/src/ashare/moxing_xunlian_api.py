"""预测训练所需模型算法的稳定门面。

只有本模块了解 moxing_gongju 的内部函数命名。训练服务依赖这里的公开接口，
从而允许底层算法继续演进而不影响业务编排。
"""

from src.ashare.moxing_gongju import (
    _blend_component_predictions as blend_component_predictions,
    _calibrate_direction_probability as calibrate_direction_probability,
    _daily_rank_ic as daily_rank_ic,
    _experiment_fingerprint as experiment_fingerprint,
    _fit_direction_probabilities as fit_direction_probabilities,
    _fit_model_components as fit_model_components,
    _fit_quantile_model_components as fit_quantile_model_components,
    _fold_stability as fold_stability,
    _quality_label as quality_label,
    _quality_score as quality_score,
    _regime_stability as regime_stability,
    _rolling_conformal_interval as rolling_conformal_interval,
    _rolling_cqr_interval as rolling_cqr_interval,
    _select_ensemble_weight as select_ensemble_weight,
    _select_stable_features as select_stable_features,
    _select_time_decay_half_life as select_time_decay_half_life,
    _time_decay_weights as time_decay_weights,
    _top_n_validation_metrics as top_n_validation_metrics,
)


__all__ = [
    "blend_component_predictions",
    "calibrate_direction_probability",
    "daily_rank_ic",
    "experiment_fingerprint",
    "fit_direction_probabilities",
    "fit_model_components",
    "fit_quantile_model_components",
    "fold_stability",
    "quality_label",
    "quality_score",
    "regime_stability",
    "rolling_conformal_interval",
    "rolling_cqr_interval",
    "select_ensemble_weight",
    "select_stable_features",
    "select_time_decay_half_life",
    "time_decay_weights",
    "top_n_validation_metrics",
]
