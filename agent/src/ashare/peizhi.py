"""统一加载并校验程序内部量化配置。"""

from __future__ import annotations

import json
import math
from typing import Any

from src.core.config import DEFAULT_CONFIG_PATH


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


def _xiaoyan_fenxi_peizhi(value: dict[str, Any]) -> None:
    analysis = _peizhi_duixiang(value, "fenxi")
    # 旧显式配置的抽样上限仅兼容为取数批大小，不再截断候选或追加旧风险过滤。
    analysis.setdefault("history_batch_size", analysis.get("prefilter_limit", 240))
    positive_integer_bounds = {
        "history_calendar_days": (180, 1800),
        "history_batch_size": (1, 1000),
        "deep_analysis_limit": (1, 10),
        "backup_limit": (0, 10),
    }
    integers: dict[str, int] = {}
    for key, (minimum, maximum) in positive_integer_bounds.items():
        number = _youxian_shuzhi(analysis, key, "fenxi")
        integer = int(number)
        if number != integer or not minimum <= integer <= maximum:
            raise ValueError(f"fenxi.{key} 必须是 {minimum} 到 {maximum} 之间的整数")
        integers[key] = integer
    for key in ("min_amount_yuan",):
        if _youxian_shuzhi(analysis, key, "fenxi") < 0:
            raise ValueError(f"fenxi.{key} 不能小于 0")
    if not 0.5 <= _youxian_shuzhi(analysis, "minimum_history_session_coverage", "fenxi") <= 1:
        raise ValueError("fenxi.minimum_history_session_coverage 必须在 0.5 到 1 之间")
    if _youxian_shuzhi(analysis, "high_volatility_threshold", "fenxi") <= 0:
        raise ValueError("fenxi.high_volatility_threshold 必须大于 0")
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
    # 共享轻量配置定义，不为校验参数加载整条历史研究链路。
    from src.ashare.macd_huifang_peizhi import MacdHuifangPeizhi

    required_validation_keys = set(MacdHuifangPeizhi.__dataclass_fields__)
    if set(macd_validation) != required_validation_keys:
        raise ValueError("fenxi.macd_structure_validation 的历史回放字段不完整或包含未知字段")
    MacdHuifangPeizhi.from_mapping(macd_validation)
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
    if _youxian_shuzhi(late_session, "volume_ratio_target_min", "weipan") <= 0:
        raise ValueError("weipan.volume_ratio_target_min 必须大于 0")
    if _youxian_shuzhi(late_session, "max_pullback_from_high_pct", "weipan") < 0:
        raise ValueError("weipan.max_pullback_from_high_pct 不能小于 0")
    minute_limit = _youxian_shuzhi(late_session, "minute_candidate_limit", "weipan")
    if minute_limit != int(minute_limit) or not 1 <= int(minute_limit) <= 500:
        raise ValueError("weipan.minute_candidate_limit 必须是有效的候选数量上限")


def jiazai_lianghua_peizhi() -> tuple[dict[str, Any], str]:
    """加载并校验股票分析诊断所需的内部配置。"""
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
        raise ValueError("shuju.frequency 必须为 daily_only；分析因子固定使用完整日K，尾盘分钟证据单独处理")
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

    single = _peizhi_duixiang(value, "dangu")
    single_integer_bounds = {
        "history_calendar_days": (540, 1800),
        "max_peer_stocks": (8, 40),
        "same_industry_stocks": (4, 39),
    }
    single_values: dict[str, int] = {}
    for key, (minimum, maximum) in single_integer_bounds.items():
        number = _youxian_shuzhi(single, key, "dangu")
        if number != int(number) or not minimum <= int(number) <= maximum:
            raise ValueError(f"dangu.{key} 必须是 {minimum} 到 {maximum} 之间的整数")
        single_values[key] = int(number)
    if single_values["same_industry_stocks"] >= single_values["max_peer_stocks"]:
        raise ValueError("dangu.same_industry_stocks 必须小于 max_peer_stocks")
    _xiaoyan_fenxi_peizhi(value)
    return value, str(path)




__all__ = ["DEFAULT_CONFIG_PATH", "jiazai_lianghua_peizhi"]
