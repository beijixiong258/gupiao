"""14:30 后尾盘实时价量证据与收盘复核。"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class WeipanJieduan(str, Enum):
    BU_SHIYONG = "not_applicable"
    CHUSHAI = "late_session_screen"
    PANZHONG_ZANDING = "intraday_provisional"
    SHOUPAN_DAIDING = "close_pending"
    SHOUPAN_FUHE = "close_review"


_STAGE_LABELS = {
    WeipanJieduan.BU_SHIYONG: "不使用尾盘证据",
    WeipanJieduan.CHUSHAI: "尾盘初筛",
    WeipanJieduan.PANZHONG_ZANDING: "盘中暂定",
    WeipanJieduan.SHOUPAN_DAIDING: "等待完整收盘确认",
    WeipanJieduan.SHOUPAN_FUHE: "收盘复核",
}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _minute(value: Any) -> int:
    hour, minute = (int(part) for part in str(value).split(":"))
    return hour * 60 + minute


def _interval_score(value: float | None, low: float, high: float) -> float | None:
    """目标区间内满分，区间外按一个区间宽度连续衰减。"""
    if value is None:
        return None
    if low <= value <= high:
        return 100.0
    width = max(high - low, abs(high) * 0.25, 1e-9)
    distance = low - value if value < low else value - high
    return round(max(0.0, 100.0 * (1.0 - distance / width)), 4)


def _threshold_score(value: float | None, target: float) -> float | None:
    if value is None:
        return None
    if value >= target:
        return round(min(100.0, 50.0 + 50.0 * (value - target) / max(target * 0.5, 1e-9)), 4)
    return round(max(0.0, 50.0 * value / max(target, 1e-9)), 4)


def panduan_weipan_jieduan(clock: dict[str, Any], config: dict[str, Any]) -> WeipanJieduan:
    if not bool(config.get("enabled", True)) or not bool(clock.get("is_trading_day")):
        return WeipanJieduan.BU_SHIYONG
    captured = pd.to_datetime(clock.get("captured_at"), errors="coerce")
    if pd.isna(captured):
        return WeipanJieduan.BU_SHIYONG
    current_minute = pd.Timestamp(captured).hour * 60 + pd.Timestamp(captured).minute
    initial = _minute(config.get("initial_screen_start", "14:30"))
    validation = _minute(config.get("minute_validation_start", "14:45"))
    close = _minute(config.get("market_close", "15:00"))
    confirmation = _minute(config.get("close_confirmation_time", "15:05"))
    if current_minute < initial:
        return WeipanJieduan.BU_SHIYONG
    if current_minute < validation:
        return WeipanJieduan.CHUSHAI
    if current_minute < close:
        return WeipanJieduan.PANZHONG_ZANDING
    if current_minute < confirmation:
        return WeipanJieduan.SHOUPAN_DAIDING
    return WeipanJieduan.SHOUPAN_FUHE


def _history_before_current_session(history: pd.DataFrame, captured: pd.Timestamp) -> pd.DataFrame:
    if history is None or history.empty or not {"trade_date", "close"}.issubset(history.columns):
        return pd.DataFrame()
    data = history.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    return data[
        data["trade_date"].notna()
        & data["close"].notna()
        & data["trade_date"].lt(captured.normalize())
    ].sort_values("trade_date")


def _dynamic_ma(
    history: pd.DataFrame,
    current_price: float | None,
    period: int,
) -> float | None:
    if current_price is None or history.empty or len(history) < period - 1:
        return None
    closes = pd.to_numeric(history["close"], errors="coerce").dropna().tail(period - 1)
    if len(closes) < period - 1:
        return None
    return round(float((closes.sum() + current_price) / period), 4)


def _minute_evidence(
    minute_data: pd.DataFrame,
    *,
    current_price: float | None,
    high_time_after: str,
    max_pullback_pct: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    actuals: dict[str, Any] = {}
    evidence: list[str] = []
    unavailable: list[str] = []
    required = {"trade_time", "close", "high"}
    if minute_data is None or minute_data.empty or not required.issubset(minute_data.columns):
        return actuals, evidence, ["5 分钟走势、最高价时间和 VWAP"]
    data = minute_data.copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"], errors="coerce")
    for column in ("open", "close", "high", "low", "volume", "amount_yuan", "average_price"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["trade_time", "close", "high"]).sort_values("trade_time")
    if data.empty:
        return actuals, evidence, ["5 分钟走势、最高价时间和 VWAP"]
    high_index = data["high"].idxmax()
    high_price = _number(data.loc[high_index, "high"])
    high_time = pd.Timestamp(data.loc[high_index, "trade_time"])
    threshold_minute = _minute(high_time_after)
    high_is_late = high_time.hour * 60 + high_time.minute >= threshold_minute
    actuals.update(
        {
            "intraday_high": high_price,
            "intraday_high_time": high_time.strftime("%H:%M"),
            "high_after_threshold": high_is_late,
        }
    )
    evidence.append(
        f"当日高点出现在 {high_time.strftime('%H:%M')}，{'达到' if high_is_late else '早于'} {high_time_after} 尾盘高点条件"
    )
    if current_price is not None and high_price is not None and high_price > 0:
        pullback = max(0.0, (high_price - current_price) / high_price * 100.0)
        actuals["pullback_from_high_pct"] = round(pullback, 4)
        actuals["pullback_within_limit"] = pullback <= max_pullback_pct
        evidence.append(f"当前价较日内高点回撤 {pullback:.2f}%")
    else:
        unavailable.append("当前价相对日内高点回撤")

    vwap = None
    if "amount_yuan" in data.columns and "volume" in data.columns:
        amount = _number(pd.to_numeric(data["amount_yuan"], errors="coerce").sum())
        volume = _number(pd.to_numeric(data["volume"], errors="coerce").sum())
        if amount is not None and volume is not None and volume > 0:
            candidate = amount / volume
            reference_price = _number(data.iloc[-1]["close"])
            if reference_price is not None and candidate < reference_price * 0.1:
                candidate *= 100.0
            if reference_price is None or 0.1 * reference_price <= candidate <= 10 * reference_price:
                vwap = candidate
    if vwap is None and "average_price" in data.columns:
        vwap = _number(data.iloc[-1]["average_price"])
    if vwap is not None:
        above_vwap = current_price is not None and current_price >= vwap
        actuals["vwap"] = round(vwap, 4)
        actuals["above_vwap"] = above_vwap
        evidence.append(f"当前价{'位于' if above_vwap else '低于'}累计成交均价 {vwap:.3f}")
    else:
        unavailable.append("累计成交均价 VWAP")

    if len(data) >= 2:
        previous_close = _number(data.iloc[-2]["close"])
        latest_close = _number(data.iloc[-1]["close"])
        if previous_close is not None and previous_close > 0 and latest_close is not None:
            actuals["latest_5min_return_pct"] = round((latest_close / previous_close - 1.0) * 100.0, 4)
    if "amount_yuan" in data.columns and len(data) >= 4:
        previous_amounts = pd.to_numeric(data["amount_yuan"], errors="coerce").iloc[-7:-1].dropna()
        latest_amount = _number(data.iloc[-1]["amount_yuan"])
        median_amount = _number(previous_amounts.median()) if not previous_amounts.empty else None
        if latest_amount is not None and median_amount is not None and median_amount > 0:
            amount_ratio = latest_amount / median_amount
            actuals["latest_5min_amount_yuan"] = round(latest_amount, 2)
            actuals["latest_5min_amount_ratio"] = round(amount_ratio, 4)
            actuals["latest_5min_volume_anomaly"] = amount_ratio >= 3.0 or amount_ratio <= 0.2
    return actuals, evidence, unavailable


def fenxi_weipan(
    history: pd.DataFrame,
    *,
    snapshot: dict[str, Any],
    clock: dict[str, Any],
    config: dict[str, Any],
    minute_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """计算尾盘连续分和分钟形态；缺失项不按失败处理。"""
    stage = panduan_weipan_jieduan(clock, config)
    base = {
        "status": "not_applicable" if stage is WeipanJieduan.BU_SHIYONG else "ok",
        "stage": stage.value,
        "stage_label": _STAGE_LABELS[stage],
        "confirmation_level": (
            "completed_daily_close"
            if stage is WeipanJieduan.SHOUPAN_FUHE
            else "intraday_provisional"
            if stage in {WeipanJieduan.CHUSHAI, WeipanJieduan.PANZHONG_ZANDING}
            else "pending"
            if stage is WeipanJieduan.SHOUPAN_DAIDING
            else "not_used"
        ),
        "score_definition": "尾盘价量适配分，只用于综合排序，不是上涨概率",
    }
    if stage is WeipanJieduan.BU_SHIYONG:
        return {**base, "score_0_100": None, "confidence": 0.0, "evidence": [], "unavailable_items": []}
    if stage is WeipanJieduan.SHOUPAN_DAIDING:
        return {
            **base,
            "score_0_100": None,
            "confidence": 0.0,
            "evidence": ["15:00 至 15:05 等待数据源确认完整收盘，不提升为正式信号"],
            "unavailable_items": ["完整收盘日线确认"],
        }
    captured = pd.to_datetime(clock.get("captured_at"), errors="coerce")
    if pd.isna(captured):
        return {**base, "status": "unavailable", "score_0_100": None, "confidence": 0.0, "error": "市场时钟无效"}
    captured = pd.Timestamp(captured)
    prior_history = _history_before_current_session(history, captured)
    last_price = _number(snapshot.get("last_price") if snapshot.get("last_price") is not None else snapshot.get("latest_price"))
    pct_change = _number(snapshot.get("pct_change") if snapshot.get("pct_change") is not None else snapshot.get("pct_chg"))
    volume_ratio = _number(snapshot.get("volume_ratio"))
    turnover = _number(snapshot.get("turnover_rate_pct") if snapshot.get("turnover_rate_pct") is not None else snapshot.get("turnover_rate"))
    circulating_mv = _number(snapshot.get("circulating_market_value_yuan"))
    dynamic_ma5 = _dynamic_ma(prior_history, last_price, 5)
    dynamic_ma10 = _dynamic_ma(prior_history, last_price, 10)
    completed_ma5 = None
    if len(prior_history) >= 5:
        completed_ma5 = _number(pd.to_numeric(prior_history["close"], errors="coerce").tail(5).mean())
    ma5_rising = dynamic_ma5 is not None and completed_ma5 is not None and dynamic_ma5 > completed_ma5
    above_ma5 = last_price is not None and dynamic_ma5 is not None and last_price >= dynamic_ma5
    above_ma10 = last_price is not None and dynamic_ma10 is not None and last_price >= dynamic_ma10

    pct_range = [float(value) for value in config.get("pct_change_high_score_range", [3.0, 5.0])]
    turnover_range = [float(value) for value in config.get("turnover_high_score_range", [5.0, 10.0])]
    mv_range = [float(value) for value in config.get("circulating_market_value_high_score_range_yuan", [5e9, 2e10])]
    scores: list[tuple[str, float | None, float]] = [
        ("pct_change", _interval_score(pct_change, *pct_range), 0.16),
        ("volume_ratio", _threshold_score(volume_ratio, float(config.get("volume_ratio_target_min", 1.0))), 0.14),
        ("turnover", _interval_score(turnover, *turnover_range), 0.14),
        ("circulating_market_value", _interval_score(circulating_mv, *mv_range), 0.10),
        ("above_dynamic_ma", 100.0 if above_ma5 and above_ma10 else 50.0 if above_ma5 or above_ma10 else 0.0 if dynamic_ma5 is not None and dynamic_ma10 is not None else None, 0.13),
        ("dynamic_ma5_rising", 100.0 if ma5_rising else 0.0 if dynamic_ma5 is not None and completed_ma5 is not None else None, 0.08),
    ]
    evidence = [
        f"当日涨幅 {pct_change:.2f}%" if pct_change is not None else "当日涨幅不可用",
        f"实时量比 {volume_ratio:.2f}" if volume_ratio is not None else "实时量比不可用",
        f"换手率 {turnover:.2f}%" if turnover is not None else "换手率不可用",
        f"流通市值约 {circulating_mv / 1e8:.2f} 亿元" if circulating_mv is not None else "流通市值不可用",
    ]
    unavailable = [name for name, score, _ in scores if score is None]
    minute_actuals: dict[str, Any] = {}
    if stage is WeipanJieduan.PANZHONG_ZANDING:
        minute_actuals, minute_evidence, minute_unavailable = _minute_evidence(
            minute_data if minute_data is not None else pd.DataFrame(),
            current_price=last_price,
            high_time_after=str(config.get("high_time_after", "14:40")),
            max_pullback_pct=float(config.get("max_pullback_from_high_pct", 1.5)),
        )
        evidence.extend(minute_evidence)
        unavailable.extend(minute_unavailable)
        shape_parts = [
            100.0 if minute_actuals.get("high_after_threshold") else 0.0 if "high_after_threshold" in minute_actuals else None,
            100.0 if minute_actuals.get("above_vwap") else 0.0 if "above_vwap" in minute_actuals else None,
            _interval_score(
                _number(minute_actuals.get("pullback_from_high_pct")),
                0.0,
                float(config.get("max_pullback_from_high_pct", 1.5)),
            ),
        ]
        available_shape = [value for value in shape_parts if value is not None]
        scores.append(("late_price_shape", float(np.mean(available_shape)) if available_shape else None, 0.25))
    else:
        scores.append(("late_price_shape", None, 0.25))
    valid_scores = [(score, weight) for _, score, weight in scores if score is not None]
    total_weight = sum(weight for _, weight in valid_scores)
    score = (
        round(sum(float(value) * weight for value, weight in valid_scores) / total_weight, 2)
        if total_weight > 0
        else None
    )
    configured_weight = sum(weight for _, _, weight in scores)
    confidence = round(total_weight / configured_weight, 4) if configured_weight > 0 else 0.0
    return {
        **base,
        "score_0_100": score,
        "confidence": confidence,
        "captured_at": clock.get("captured_at"),
        "actuals": {
            "pct_change": pct_change,
            "volume_ratio": volume_ratio,
            "turnover_rate_pct": turnover,
            "circulating_market_value_yuan": circulating_mv,
            **minute_actuals,
        },
        "dynamic_moving_averages": {
            "price": last_price,
            "ma5": dynamic_ma5,
            "ma10": dynamic_ma10,
            "ma5_rising": ma5_rising if dynamic_ma5 is not None and completed_ma5 is not None else None,
            "above_ma5": above_ma5 if dynamic_ma5 is not None and last_price is not None else None,
            "above_ma10": above_ma10 if dynamic_ma10 is not None and last_price is not None else None,
            "definition": "使用前 N-1 个完整收盘价与当前实时价计算，不覆盖历史日 K 特征",
        },
        "component_scores": {name: value for name, value, _ in scores},
        "evidence": evidence,
        "unavailable_items": list(dict.fromkeys(unavailable)),
        "data_quality_note": "缺失项不按失败处理；有效证据重新归一，同时降低 confidence",
    }


__all__ = ["WeipanJieduan", "fenxi_weipan", "panduan_weipan_jieduan"]
