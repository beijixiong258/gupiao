"""统一加载并校验程序内部量化配置。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "lianghua_peizhi.json"


def _peizhi_duixiang(value: dict[str, Any], key: str) -> dict[str, Any]:
    section = value.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"{key} 必须是 JSON 对象")
    return section


def _youxian_shuzhi(section: dict[str, Any], key: str, label: str) -> float:
    try:
        number = float(section[key])
    except KeyError as exc:
        raise ValueError(f"{label}.{key} 不能为空") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} 必须是数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}.{key} 必须是有限数值")
    return number


def _jiaoyishijian(value: Any, label: str) -> int:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ValueError(f"{label} 必须使用 HH:MM 格式")
    hour, minute = (int(part) for part in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"{label} 不是有效时间")
    return hour * 60 + minute


def _xiaoyan_quanzhong(
    section: dict[str, Any],
    *,
    label: str,
    required_keys: set[str],
) -> None:
    if set(section) != required_keys:
        raise ValueError(f"{label} 必须且只能包含：{', '.join(sorted(required_keys))}")
    weights = [_youxian_shuzhi(section, key, label) for key in sorted(required_keys)]
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError(f"{label} 必须为非负数且总和严格等于 1")


def _xiaoyan_fenxi_peizhi(value: dict[str, Any]) -> None:
    analysis = _peizhi_duixiang(value, "fenxi")
    positive_integer_bounds = {
        "history_calendar_days": (180, 1800),
        "prefilter_limit": (20, 1000),
        "factor_candidate_limit": (10, 500),
        "deep_analysis_limit": (1, 10),
        "backup_limit": (0, 10),
        "minimum_history_rows": (60, 500),
        "minimum_listing_calendar_days": (0, 3650),
    }
    integers: dict[str, int] = {}
    for key, (minimum, maximum) in positive_integer_bounds.items():
        number = _youxian_shuzhi(analysis, key, "fenxi")
        integer = int(number)
        if number != integer or not minimum <= integer <= maximum:
            raise ValueError(f"fenxi.{key} 必须是 {minimum} 到 {maximum} 之间的整数")
        integers[key] = integer
    if integers["factor_candidate_limit"] > integers["prefilter_limit"]:
        raise ValueError("fenxi.factor_candidate_limit 不能大于 prefilter_limit")
    if integers["deep_analysis_limit"] > integers["factor_candidate_limit"]:
        raise ValueError("fenxi.deep_analysis_limit 不能大于 factor_candidate_limit")
    for key in ("min_amount_yuan", "risk_penalty_max", "near_limit_up_penalty", "high_volatility_penalty"):
        if _youxian_shuzhi(analysis, key, "fenxi") < 0:
            raise ValueError(f"fenxi.{key} 不能小于 0")
    if not 0 <= _youxian_shuzhi(analysis, "minimum_recommendation_score", "fenxi") <= 100:
        raise ValueError("fenxi.minimum_recommendation_score 必须在 0 到 100 之间")
    if not 0 <= _youxian_shuzhi(analysis, "minimum_confidence", "fenxi") <= 1:
        raise ValueError("fenxi.minimum_confidence 必须在 0 到 1 之间")
    if not 0.5 <= _youxian_shuzhi(analysis, "minimum_history_session_coverage", "fenxi") <= 1:
        raise ValueError("fenxi.minimum_history_session_coverage 必须在 0.5 到 1 之间")
    if _youxian_shuzhi(analysis, "high_volatility_threshold", "fenxi") <= 0:
        raise ValueError("fenxi.high_volatility_threshold 必须大于 0")
    if not 0 <= _youxian_shuzhi(analysis, "fundamental_deep_weight", "fenxi") <= 1:
        raise ValueError("fenxi.fundamental_deep_weight 必须在 0 到 1 之间")
    macd_structure = analysis.get("macd_structure")
    if not isinstance(macd_structure, dict):
        raise ValueError("fenxi.macd_structure 必须是 JSON 对象")
    required_macd_keys = {
        "zero_near_threshold_pct",
        "pivot_left_sessions",
        "pivot_right_sessions",
        "pivot_match_sessions",
        "minimum_pivot_separation_sessions",
        "maximum_pivot_separation_sessions",
        "minimum_price_change_pct",
        "minimum_indicator_change_pct",
        "cross_fresh_sessions",
        "cross_recent_sessions",
        "cross_max_age_sessions",
        "divergence_max_age_sessions",
        "invalidation_price_tolerance_pct",
    }
    if set(macd_structure) != required_macd_keys:
        raise ValueError("fenxi.macd_structure 的结构检测字段不完整或包含未知字段")
    integer_macd_keys = {
        "pivot_left_sessions",
        "pivot_right_sessions",
        "pivot_match_sessions",
        "minimum_pivot_separation_sessions",
        "maximum_pivot_separation_sessions",
        "cross_fresh_sessions",
        "cross_recent_sessions",
        "cross_max_age_sessions",
        "divergence_max_age_sessions",
    }
    macd_numbers = {
        key: _youxian_shuzhi(macd_structure, key, "fenxi.macd_structure")
        for key in required_macd_keys
    }
    if any(macd_numbers[key] != int(macd_numbers[key]) for key in integer_macd_keys):
        raise ValueError("fenxi.macd_structure 的交易日窗口必须是整数")
    if not 1 <= int(macd_numbers["pivot_left_sessions"]) <= 10:
        raise ValueError("fenxi.macd_structure.pivot_left_sessions 必须在 1 到 10 之间")
    if not 1 <= int(macd_numbers["pivot_right_sessions"]) <= 10:
        raise ValueError("fenxi.macd_structure.pivot_right_sessions 必须在 1 到 10 之间")
    if not 0 <= int(macd_numbers["pivot_match_sessions"]) <= 10:
        raise ValueError("fenxi.macd_structure.pivot_match_sessions 必须在 0 到 10 之间")
    if not (
        2 <= int(macd_numbers["minimum_pivot_separation_sessions"])
        < int(macd_numbers["maximum_pivot_separation_sessions"])
        <= 250
    ):
        raise ValueError("fenxi.macd_structure 的拐点间隔范围无效")
    if not (
        0 <= int(macd_numbers["cross_fresh_sessions"])
        <= int(macd_numbers["cross_recent_sessions"])
        <= int(macd_numbers["cross_max_age_sessions"])
        <= 250
    ):
        raise ValueError("fenxi.macd_structure 的交叉新鲜度窗口必须递增")
    if not 1 <= int(macd_numbers["divergence_max_age_sessions"]) <= 250:
        raise ValueError("fenxi.macd_structure.divergence_max_age_sessions 必须在 1 到 250 之间")
    for key in (
        "zero_near_threshold_pct",
        "minimum_price_change_pct",
        "minimum_indicator_change_pct",
        "invalidation_price_tolerance_pct",
    ):
        if not 0 < macd_numbers[key] <= 0.2:
            raise ValueError(f"fenxi.macd_structure.{key} 必须在 0 到 0.2 之间")
    macd_validation = analysis.get("macd_structure_validation")
    if not isinstance(macd_validation, dict):
        raise ValueError("fenxi.macd_structure_validation 必须是 JSON 对象")
    # 回放模块自身是这组研究口径的唯一校验入口，避免配置层复制一套逐渐分叉的规则。
    from src.ashare.macd_huifang import MacdHuifangPeizhi

    required_validation_keys = set(MacdHuifangPeizhi.__dataclass_fields__)
    if set(macd_validation) != required_validation_keys:
        raise ValueError("fenxi.macd_structure_validation 的历史回放字段不完整或包含未知字段")
    MacdHuifangPeizhi.from_mapping(macd_validation)
    component_weights = analysis.get("component_weights")
    factor_group_weights = analysis.get("factor_group_weights")
    if not isinstance(component_weights, dict) or not isinstance(factor_group_weights, dict):
        raise ValueError("fenxi 的 component_weights 和 factor_group_weights 必须是 JSON 对象")
    _xiaoyan_quanzhong(
        component_weights,
        label="fenxi.component_weights",
        required_keys={"daily_factors", "fundamental", "pattern", "late_session"},
    )
    _xiaoyan_quanzhong(
        factor_group_weights,
        label="fenxi.factor_group_weights",
        required_keys={
            "trend_structure",
            "momentum_reversal",
            "candle_pressure",
            "price_volume_confirmation",
            "breakout_pullback_quality",
            "relative_strength",
            "risk_liquidity",
            "market_context",
        },
    )

    pattern = _peizhi_duixiang(value, "xingtai")
    shrink_window_value = _youxian_shuzhi(pattern, "shrink_window_sessions", "xingtai")
    breakout_deadline_value = _youxian_shuzhi(pattern, "breakout_deadline_sessions", "xingtai")
    shrink_window = int(shrink_window_value)
    breakout_deadline = int(breakout_deadline_value)
    if shrink_window != shrink_window_value or breakout_deadline != breakout_deadline_value:
        raise ValueError("xingtai 的缩量窗口和突破截止交易日必须是整数")
    if not 1 <= shrink_window < breakout_deadline <= 30:
        raise ValueError("xingtai 的缩量窗口必须小于突破截止交易日，且截止日不能超过 30")
    if not 0 < _youxian_shuzhi(pattern, "shrink_volume_ratio_max", "xingtai") <= 1:
        raise ValueError("xingtai.shrink_volume_ratio_max 必须在 0 到 1 之间")
    if _youxian_shuzhi(pattern, "breakout_volume_median_ratio_min", "xingtai") < 1:
        raise ValueError("xingtai.breakout_volume_median_ratio_min 不能小于 1")
    if not 0 <= _youxian_shuzhi(pattern, "limit_up_tolerance_yuan", "xingtai") <= 0.02:
        raise ValueError("xingtai.limit_up_tolerance_yuan 必须在 0 到 0.02 之间")
    state_scores = pattern.get("state_scores")
    required_states = {
        "adjusting",
        "extreme_shrink",
        "waiting_breakout",
        "intraday_confirmed",
        "close_confirmed",
        "invalidated",
        "expired",
    }
    if not isinstance(state_scores, dict) or set(state_scores) != required_states:
        raise ValueError("xingtai.state_scores 缺少必需的形态状态")
    if any(not 0 <= _youxian_shuzhi(state_scores, key, "xingtai.state_scores") <= 100 for key in required_states):
        raise ValueError("xingtai.state_scores 的分数必须在 0 到 100 之间")

    late_session = _peizhi_duixiang(value, "weipan")
    if not isinstance(late_session.get("enabled"), bool):
        raise ValueError("weipan.enabled 必须是 true 或 false")
    time_keys = (
        "initial_screen_start",
        "minute_validation_start",
        "market_close",
        "close_confirmation_time",
    )
    time_points = [_jiaoyishijian(late_session.get(key), f"weipan.{key}") for key in time_keys]
    if time_points != sorted(time_points) or len(set(time_points)) != len(time_points):
        raise ValueError("weipan 的阶段时间必须严格递增")
    _jiaoyishijian(late_session.get("high_time_after"), "weipan.high_time_after")
    for key in ("pct_change_high_score_range", "turnover_high_score_range", "circulating_market_value_high_score_range_yuan"):
        bounds = late_session.get(key)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"weipan.{key} 必须是两个数值组成的数组")
        low, high = (_youxian_shuzhi({"low": bounds[0]}, "low", f"weipan.{key}"), _youxian_shuzhi({"high": bounds[1]}, "high", f"weipan.{key}"))
        if low < 0 or high <= low:
            raise ValueError(f"weipan.{key} 必须满足 0 <= 下界 < 上界")
    if _youxian_shuzhi(late_session, "volume_ratio_target_min", "weipan") <= 0:
        raise ValueError("weipan.volume_ratio_target_min 必须大于 0")
    if _youxian_shuzhi(late_session, "max_pullback_from_high_pct", "weipan") < 0:
        raise ValueError("weipan.max_pullback_from_high_pct 不能小于 0")
    minute_limit = _youxian_shuzhi(late_session, "minute_candidate_limit", "weipan")
    if minute_limit != int(minute_limit) or not 1 <= int(minute_limit) <= integers["factor_candidate_limit"]:
        raise ValueError("weipan.minute_candidate_limit 必须是有效的候选数量上限")


def jiazai_lianghua_peizhi() -> tuple[dict[str, Any], str]:
    """Load and validate the fixed internal daily-model configuration."""
    path = DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"量化配置文件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("量化配置必须是 JSON 对象")

    def finite_number(section: dict[str, Any], key: str, default: float, label: str) -> float:
        try:
            number = float(section.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{key} 必须是数值") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}.{key} 必须是有限数值")
        return number

    data_settings = value.get("shuju", {})
    if not isinstance(data_settings, dict):
        raise ValueError("shuju 必须是 JSON 对象")
    data_pause = finite_number(data_settings, "request_pause_seconds", 0.15, "shuju")
    if not 0 <= data_pause <= 10:
        raise ValueError("shuju.request_pause_seconds 必须在 0 到 10 之间")
    if data_settings.get("frequency", "daily_only") != "daily_only":
        raise ValueError("shuju.frequency 必须为 daily_only；预测模型固定使用完整日K，尾盘分钟证据单独处理")
    if data_settings.get("minute_bars_enabled", True) is not True:
        raise ValueError("shuju.minute_bars_enabled 必须为 true；14:45 后尾盘复核需要少量 5 分钟行情")

    network = value.get("wangluo", {})
    if not isinstance(network, dict):
        raise ValueError("wangluo 必须是 JSON 对象")
    if network.get("domestic_connection_mode", "direct") not in {"direct", "system_proxy"}:
        raise ValueError("wangluo.domestic_connection_mode 必须是 direct 或 system_proxy")
    for key in ("connect_timeout_seconds", "read_timeout_seconds"):
        timeout = finite_number(network, key, 6.0 if key.startswith("connect") else 20.0, "wangluo")
        if not 1 <= timeout <= 120:
            raise ValueError(f"wangluo.{key} 必须在 1 到 120 秒之间")
    for key in ("max_attempts_per_endpoint", "tushare_max_attempts"):
        raw_attempts = finite_number(network, key, 2, "wangluo")
        if raw_attempts != int(raw_attempts) or not 1 <= int(raw_attempts) <= 5:
            raise ValueError(f"wangluo.{key} 必须是 1 到 5 之间的整数")
    network_integer_bounds = {
        "history_max_workers": (1, 32, 8),
        "history_fallback_max_stocks": (0, 100, 16),
    }
    for key, (minimum, maximum, default) in network_integer_bounds.items():
        raw_value = finite_number(network, key, default, "wangluo")
        if raw_value != int(raw_value) or not minimum <= int(raw_value) <= maximum:
            raise ValueError(f"wangluo.{key} 必须是 {minimum} 到 {maximum} 之间的整数")
    retry_backoff = finite_number(network, "retry_backoff_seconds", 0.35, "wangluo")
    if not 0 <= retry_backoff <= 10:
        raise ValueError("wangluo.retry_backoff_seconds 必须在 0 到 10 秒之间")

    run_logs = value.get("run_logs", {})
    if not isinstance(run_logs, dict):
        raise ValueError("run_logs 必须是 JSON 对象")
    if not isinstance(run_logs.get("enabled", True), bool):
        raise ValueError("run_logs.enabled 必须是 true 或 false")
    if run_logs.get("content_mode", "metadata_only") not in {"metadata_only", "full_redacted"}:
        raise ValueError("run_logs.content_mode 必须是 metadata_only 或 full_redacted")
    log_integer_bounds = {
        "retention_days": (1, 3650, 14),
        "maximum_runs": (1, 10000, 100),
        "maximum_total_mb": (1, 102400, 100),
    }
    for key, (minimum, maximum, default) in log_integer_bounds.items():
        raw_value = finite_number(run_logs, key, default, "run_logs")
        if raw_value != int(raw_value) or not minimum <= int(raw_value) <= maximum:
            raise ValueError(f"run_logs.{key} 必须是 {minimum} 到 {maximum} 之间的整数")

    model = value.get("moxing", {})
    if not isinstance(model, dict):
        raise ValueError("moxing 必须是 JSON 对象")
    horizons = model.get("horizons")
    if horizons != [1, 2, 3]:
        raise ValueError("moxing.horizons 必须严格为 [1, 2, 3]")
    try:
        validation_ratio = float(model.get("validation_ratio"))
        clip_quantiles = [float(item) for item in model.get("prediction_clip_quantiles", [])]
        weights = {int(key): float(item) for key, item in model.get("horizon_weights", {}).items()}
        integer_defaults = {
            "min_training_samples": 500,
            "min_validation_samples": 100,
            "min_rank_ic_days": 10,
            "validation_top_n": 3,
            "min_top_n_days": 10,
            "validation_subwindows": 6,
            "min_passed_subwindows": 4,
            "ensemble_min_calibration_samples": 80,
            "ensemble_min_calibration_dates": 20,
            "time_decay_min_calibration_samples": 80,
            "time_decay_min_calibration_dates": 20,
            "factor_stability_slices": 6,
            "factor_min_valid_slices": 4,
            "factor_min_features": 15,
            "factor_max_features": 20,
            "factor_max_per_group": 3,
            "direction_logistic_max_iter": 500,
            "probability_calibration_min_samples": 120,
            "conformal_min_samples": 80,
            "ranking_relevance_grades": 5,
            "ranking_pair_top_k": 8,
            "ranking_n_estimators": 180,
        }
        positive_integer_fields = {
            key: int(model.get(key, default)) for key, default in integer_defaults.items()
        }
        min_direction = float(model.get("min_direction_accuracy", 0.52))
        min_rank_ic = float(model.get("min_mean_daily_rank_ic", 0.01))
        min_skill = float(model.get("min_skill_vs_baseline", 0.01))
        min_best_naive_skill = float(model.get("min_skill_vs_best_naive_baseline", 0.0))
        abstain_min_net_return = float(model.get("abstain_min_net_return", 0.003))
        abstain_min_probability = float(model.get("abstain_min_positive_probability", 0.55))
        abstain_min_quality = float(model.get("abstain_min_quality_score", 0.40))
        ridge_alpha = float(model.get("ridge_alpha", 10.0))
        ensemble_default_tree_weight = float(model.get("ensemble_default_tree_weight", 0.75))
        ensemble_calibration_ratio = float(model.get("ensemble_calibration_ratio", 0.15))
        ensemble_weight_grid = [
            float(item)
            for item in model.get("ensemble_tree_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
        ]
        time_decay_candidates = [
            float(item)
            for item in model.get("time_decay_half_life_candidates", [0, 252, 504])
        ]
        time_decay_calibration_ratio = float(model.get("time_decay_calibration_ratio", 0.15))
        time_decay_min_improvement = float(model.get("time_decay_min_relative_improvement", 0.002))
        time_decay_min_weight = float(model.get("time_decay_min_weight", 0.10))
        feature_winsor_quantiles = [
            float(item) for item in model.get("feature_winsor_quantiles", [0.01, 0.99])
        ]
        model_feature_coverage = float(model.get("min_feature_coverage", 0.20))
        factor_min_sign_agreement = float(model.get("factor_min_sign_agreement", 0.67))
        factor_min_abs_ic = float(model.get("factor_min_abs_mean_rank_ic", 0.005))
        factor_dedup_abs_spearman = float(model.get("factor_dedup_abs_spearman", 0.80))
        direction_logistic_c = float(model.get("direction_logistic_c", 0.5))
        calibration_evaluation_ratio = float(model.get("probability_calibration_evaluation_ratio", 0.30))
        calibration_min_improvement = float(model.get("probability_calibration_min_brier_improvement", 0.0005))
        conformal_coverage = float(model.get("conformal_coverage", 0.80))
        ranking_min_ndcg_improvement = float(model.get("ranking_min_ndcg_improvement", 0.0))
        return_interval_coverage_range = [
            float(item) for item in model.get("return_interval_coverage_range", [0.75, 0.85])
        ]
        quantile_levels = [
            float(item) for item in model.get("quantile_levels", [0.10, 0.50, 0.90])
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"moxing 数值配置无效：{exc}") from exc
    if not 0.05 <= validation_ratio <= 0.4:
        raise ValueError("moxing.validation_ratio 必须在 0.05 到 0.4 之间")
    if len(clip_quantiles) != 2 or not 0 <= clip_quantiles[0] < clip_quantiles[1] <= 1:
        raise ValueError("moxing.prediction_clip_quantiles 必须是两个递增的 0~1 数值")
    if (
        set(weights) != {1, 2, 3}
        or any(not math.isfinite(item) or item < 0 for item in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("moxing.horizon_weights 必须为 T+1/T+2/T+3 提供非负权重且总和大于0")
    if any(item <= 0 for item in positive_integer_fields.values()):
        raise ValueError("moxing 的样本数、验证天数和 Top-N 配置必须为正整数")
    if not 2 <= positive_integer_fields["ranking_relevance_grades"] <= 31:
        raise ValueError("moxing.ranking_relevance_grades 必须在 2 到 31 之间")
    if positive_integer_fields["ranking_pair_top_k"] > 100:
        raise ValueError("moxing.ranking_pair_top_k 不能大于 100")
    if positive_integer_fields["min_passed_subwindows"] > positive_integer_fields["validation_subwindows"]:
        raise ValueError("moxing.min_passed_subwindows 不能大于 validation_subwindows")
    if positive_integer_fields["factor_min_valid_slices"] > positive_integer_fields["factor_stability_slices"]:
        raise ValueError("moxing.factor_min_valid_slices 不能大于 factor_stability_slices")
    if positive_integer_fields["factor_max_features"] < positive_integer_fields["factor_min_features"]:
        raise ValueError("moxing.factor_max_features 不能小于 factor_min_features")
    if positive_integer_fields["factor_max_per_group"] > positive_integer_fields["factor_max_features"]:
        raise ValueError("moxing.factor_max_per_group 不能大于 factor_max_features")
    if not 0.5 <= min_direction <= 1:
        raise ValueError("moxing.min_direction_accuracy 必须在 0.5 到 1 之间")
    if not -1 <= min_rank_ic <= 1 or not -1 <= min_skill <= 1 or not -1 <= min_best_naive_skill <= 1:
        raise ValueError("moxing 的 Rank IC 和基线提升门槛必须在 -1 到 1 之间")
    if not 0 <= abstain_min_net_return <= 0.2:
        raise ValueError("moxing.abstain_min_net_return 必须在 0 到 0.2 之间")
    if not 0.5 <= abstain_min_probability <= 1:
        raise ValueError("moxing.abstain_min_positive_probability 必须在 0.5 到 1 之间")
    if not 0 <= abstain_min_quality <= 1:
        raise ValueError("moxing.abstain_min_quality_score 必须在 0 到 1 之间")
    if not isinstance(model.get("ensemble_enabled", True), bool):
        raise ValueError("moxing.ensemble_enabled 必须是 true 或 false")
    if not isinstance(model.get("time_decay_enabled", True), bool):
        raise ValueError("moxing.time_decay_enabled 必须是 true 或 false")
    if not isinstance(model.get("factor_stability_enabled", True), bool):
        raise ValueError("moxing.factor_stability_enabled 必须是 true 或 false")
    if not isinstance(model.get("ranking_enabled", True), bool):
        raise ValueError("moxing.ranking_enabled 必须是 true 或 false")
    if not isinstance(model.get("quantile_interval_enabled", True), bool):
        raise ValueError("moxing.quantile_interval_enabled 必须是 true 或 false")
    if not 0 < model_feature_coverage <= 1:
        raise ValueError("moxing.min_feature_coverage 必须在 0 到 1 之间")
    if not 0.5 <= factor_min_sign_agreement <= 1:
        raise ValueError("moxing.factor_min_sign_agreement 必须在 0.5 到 1 之间")
    if not 0 <= factor_min_abs_ic <= 1:
        raise ValueError("moxing.factor_min_abs_mean_rank_ic 必须在 0 到 1 之间")
    if not 0 < factor_dedup_abs_spearman <= 1:
        raise ValueError("moxing.factor_dedup_abs_spearman 必须在 0 到 1 之间")
    if not math.isfinite(direction_logistic_c) or direction_logistic_c <= 0:
        raise ValueError("moxing.direction_logistic_c 必须是正有限数")
    if not 0.1 <= calibration_evaluation_ratio <= 0.5:
        raise ValueError("moxing.probability_calibration_evaluation_ratio 必须在 0.1 到 0.5 之间")
    if not 0 <= calibration_min_improvement <= 0.2:
        raise ValueError("moxing.probability_calibration_min_brier_improvement 必须在 0 到 0.2 之间")
    if not 0.5 < conformal_coverage < 1:
        raise ValueError("moxing.conformal_coverage 必须在 0.5 到 1 之间")
    if not -1 <= ranking_min_ndcg_improvement <= 1:
        raise ValueError("moxing.ranking_min_ndcg_improvement 必须在 -1 到 1 之间")
    if (
        len(return_interval_coverage_range) != 2
        or not 0 < return_interval_coverage_range[0] < return_interval_coverage_range[1] < 1
    ):
        raise ValueError("moxing.return_interval_coverage_range 必须是两个递增的 0~1 数值")
    if (
        len(quantile_levels) != 3
        or quantile_levels != sorted(quantile_levels)
        or not 0 < quantile_levels[0] < quantile_levels[1] < quantile_levels[2] < 1
    ):
        raise ValueError("moxing.quantile_levels 必须是三个递增的 0~1 数值")
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("moxing.ridge_alpha 必须是正有限数")
    if not 0 <= ensemble_default_tree_weight <= 1:
        raise ValueError("moxing.ensemble_default_tree_weight 必须在 0 到 1 之间")
    if not 0.05 <= ensemble_calibration_ratio <= 0.4:
        raise ValueError("moxing.ensemble_calibration_ratio 必须在 0.05 到 0.4 之间")
    if (
        not time_decay_candidates
        or 0.0 not in time_decay_candidates
        or len(time_decay_candidates) != len(set(time_decay_candidates))
        or time_decay_candidates != sorted(time_decay_candidates)
        or any(not math.isfinite(item) or item < 0 for item in time_decay_candidates)
    ):
        raise ValueError("moxing.time_decay_half_life_candidates 必须是包含0的非负递增无重复数值数组")
    if not 0.05 <= time_decay_calibration_ratio <= 0.4:
        raise ValueError("moxing.time_decay_calibration_ratio 必须在 0.05 到 0.4 之间")
    if not 0 <= time_decay_min_improvement <= 0.2:
        raise ValueError("moxing.time_decay_min_relative_improvement 必须在 0 到 0.2 之间")
    if not 0 < time_decay_min_weight <= 1:
        raise ValueError("moxing.time_decay_min_weight 必须在 0 到 1 之间")
    if (
        not ensemble_weight_grid
        or any(not math.isfinite(item) or not 0 <= item <= 1 for item in ensemble_weight_grid)
    ):
        raise ValueError("moxing.ensemble_tree_weight_grid 必须是非空的 0 到 1 数值数组")
    if (
        len(feature_winsor_quantiles) != 2
        or not 0 <= feature_winsor_quantiles[0] < feature_winsor_quantiles[1] <= 1
    ):
        raise ValueError("moxing.feature_winsor_quantiles 必须是两个递增的 0 到 1 数值")
    learning_rate = finite_number(model, "learning_rate", 0.05, "moxing")
    l2_regularization = finite_number(model, "l2_regularization", 1.0, "moxing")
    model_integer_fields = {
        "max_iter": int(finite_number(model, "max_iter", 180, "moxing")),
        "max_leaf_nodes": int(finite_number(model, "max_leaf_nodes", 15, "moxing")),
        "max_depth": int(finite_number(model, "max_depth", 4, "moxing")),
        "min_samples_leaf": int(finite_number(model, "min_samples_leaf", 30, "moxing")),
    }
    try:
        int(model.get("random_state", 42))
    except (TypeError, ValueError) as exc:
        raise ValueError("moxing.random_state 必须是整数") from exc
    if not 0 < learning_rate <= 1:
        raise ValueError("moxing.learning_rate 必须在 0 到 1 之间")
    if l2_regularization < 0:
        raise ValueError("moxing.l2_regularization 不能小于 0")
    if any(item <= 0 for item in model_integer_fields.values()):
        raise ValueError("moxing 的迭代次数、树规模、深度和叶节点样本数必须为正整数")
    single = value.get("dangu", {})
    if not isinstance(single, dict):
        raise ValueError("dangu 必须是 JSON 对象")
    try:
        single_history_days = int(single.get("history_calendar_days", 1440))
        maximum_peers = int(single.get("max_peer_stocks", 20))
        same_industry_peers = int(single.get("same_industry_stocks", 16))
        minimum_peers = int(single.get("minimum_peer_stocks", 8))
        walk_forward_folds = int(single.get("walk_forward_folds", 6))
        minimum_passed_folds = int(single.get("min_passed_folds", 4))
        minimum_feature_coverage = float(single.get("min_feature_coverage", 0.2))
        minimum_net_return = float(single.get("assessment_min_net_return", 0.003))
        minimum_probability = float(single.get("assessment_min_positive_probability", 0.55))
        single_minimum_history_rows = int(single.get("minimum_history_rows", 180))
        minimum_listing_days = int(single.get("min_listing_calendar_days", 180))
        single_minimum_amount = float(single.get("min_amount_yuan", 30_000_000))
        single_pause = float(single.get("request_pause_seconds", 0.08))
        validation_window_days = int(single.get("validation_window_days", 45))
        minimum_training_dates = int(single.get("minimum_training_dates", 120))
        minimum_fold_training = int(single.get("min_fold_training_samples", 500))
        minimum_fold_validation = int(single.get("min_fold_validation_samples", 80))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dangu 数值配置无效：{exc}") from exc
    if not 540 <= single_history_days <= 1800:
        raise ValueError("dangu.history_calendar_days 必须在 540 到 1800 之间")
    if not 8 <= maximum_peers <= 40:
        raise ValueError("dangu.max_peer_stocks 必须在 8 到 40 之间")
    if not 4 <= same_industry_peers < maximum_peers:
        raise ValueError("dangu.same_industry_stocks 必须至少为4且小于 max_peer_stocks")
    if not 5 <= minimum_peers <= maximum_peers:
        raise ValueError("dangu.minimum_peer_stocks 必须在5到 max_peer_stocks 之间")
    if walk_forward_folds < 2 or not 1 <= minimum_passed_folds <= walk_forward_folds:
        raise ValueError("dangu 的滚动验证折数或最少通过折数无效")
    if not 0 < minimum_feature_coverage <= 1:
        raise ValueError("dangu.min_feature_coverage 必须在0到1之间")
    if not 0 <= minimum_net_return <= 0.2 or not 0.5 <= minimum_probability <= 1:
        raise ValueError("dangu 的证据评估收益或上涨比例门槛无效")
    if single_minimum_history_rows < 60:
        raise ValueError("dangu.minimum_history_rows 必须至少为 60")
    if minimum_listing_days < 0 or single_minimum_amount < 0:
        raise ValueError("dangu 的最少上市天数和最低成交额不能小于 0")
    if not 0 <= single_pause <= 10:
        raise ValueError("dangu.request_pause_seconds 必须在 0 到 10 之间")
    if validation_window_days < 20 or minimum_training_dates < 80:
        raise ValueError("dangu 的验证窗口至少为 20 日，训练日期至少为 80 日")
    if minimum_fold_training <= 0 or minimum_fold_validation <= 0:
        raise ValueError("dangu 的每折训练和验证样本数必须为正整数")
    _xiaoyan_fenxi_peizhi(value)
    return value, str(path)




__all__ = ["DEFAULT_CONFIG_PATH", "jiazai_lianghua_peizhi"]
