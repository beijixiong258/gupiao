"""MACD 结构证据的无未来数据历史回放与基线比较。

本模块只处理调用方放入内存的历史面板，不访问行情源、不落盘，也不修改生产评分。
信号在收盘后确认，统一尝试下一市场交易日开盘建仓，并对停牌、封板、流动性、
整手与交易成本逐笔留痕。行业时点、现有排名基线或交易状态缺失时会明确降级，
不能把不完整回放表述为第五阶段已经通过。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.ashare.gupiao_yanjiu import jisuan_tezheng_biao
from src.ashare.jiaoyi_zhixing import (
    jisuan_gupiao_wangfan_chengben,
    jiazai_chengben_jiashe,
    koujian_jiaoyi_chengben,
)
from src.ashare.macd_jiegou import (
    MACD_JIEGOU_METHOD_VERSION,
    MacdJiegouPeizhi,
    lieju_macd_jiegou_shijian,
)


MACD_HUIFANG_METHOD_VERSION = "macd-structure-validation-v1"


@dataclass(frozen=True)
class MacdHuifangPeizhi:
    """固定的研究回放口径，不是生产评分权重。"""

    forward_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    decision_horizons: tuple[int, ...] = (5, 10, 20)
    minimum_signal_history_sessions: int = 80
    amount_window_sessions: int = 20
    target_notional_yuan: float = 20_000.0
    minimum_trailing_amount_yuan: float = 50_000_000.0
    maximum_participation_rate: float = 0.005
    minimum_baseline_peers_per_date: int = 5
    minimum_signal_samples: int = 30
    minimum_stocks: int = 30
    minimum_sample_scopes: int = 2
    stability_subperiods: int = 4
    minimum_valid_subperiods: int = 3
    minimum_favorable_sign_agreement: float = 0.75
    confidence_z: float = 1.96
    parameter_sensitivity_enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MacdHuifangPeizhi":
        if not value:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        if unknown:
            raise ValueError(f"MACD 回放配置包含未知字段：{', '.join(sorted(unknown))}")
        normalized = {key: value[key] for key in fields if key in value}
        for key in ("forward_horizons", "decision_horizons"):
            if key in normalized:
                normalized[key] = tuple(int(item) for item in normalized[key])
        settings = cls(**normalized)
        settings._validate()
        return settings

    def _validate(self) -> None:
        if not self.forward_horizons or any(item <= 0 for item in self.forward_horizons):
            raise ValueError("forward_horizons 必须是非空的正整数序列")
        if tuple(sorted(set(self.forward_horizons))) != self.forward_horizons:
            raise ValueError("forward_horizons 必须严格递增且不能重复")
        if not self.decision_horizons or not set(self.decision_horizons).issubset(self.forward_horizons):
            raise ValueError("decision_horizons 必须是 forward_horizons 的非空子集")
        integer_values = (
            self.minimum_signal_history_sessions,
            self.amount_window_sessions,
            self.minimum_baseline_peers_per_date,
            self.minimum_signal_samples,
            self.minimum_stocks,
            self.minimum_sample_scopes,
            self.stability_subperiods,
            self.minimum_valid_subperiods,
        )
        if any(int(item) != item or item <= 0 for item in integer_values):
            raise ValueError("MACD 回放窗口和样本门槛必须是正整数")
        if self.minimum_valid_subperiods > self.stability_subperiods:
            raise ValueError("minimum_valid_subperiods 不能大于 stability_subperiods")
        if self.target_notional_yuan <= 0 or self.minimum_trailing_amount_yuan < 0:
            raise ValueError("回放资金和成交额门槛无效")
        if not 0 < self.maximum_participation_rate <= 0.1:
            raise ValueError("maximum_participation_rate 必须在 0 到 0.1 之间")
        if not 0.5 <= self.minimum_favorable_sign_agreement <= 1:
            raise ValueError("minimum_favorable_sign_agreement 必须在 0.5 到 1 之间")
        if not 0 < self.confidence_z <= 5:
            raise ValueError("confidence_z 必须在 0 到 5 之间")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict, tuple, set)) and bool(pd.isna(value))):
        return ""
    return str(value).strip()


def _row_bool(row: pd.Series, key: str) -> bool | None:
    if key not in row.index or pd.isna(row.get(key)):
        return None
    value = row.get(key)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "是"}:
            return True
        if text in {"false", "0", "no", "n", "否"}:
            return False
        return None
    return bool(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _date(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    return value


def _prepare_panel(panel: pd.DataFrame, settings: MacdHuifangPeizhi) -> tuple[pd.DataFrame, list[str], bool]:
    if panel is None or panel.empty:
        raise ValueError("历史回放面板为空")
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"历史回放面板缺少字段：{', '.join(missing)}")
    data = panel.copy()
    data["ts_code"] = data["ts_code"].astype(str).str.strip().str.upper()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    invalid_dates = int(data["trade_date"].isna().sum())
    data = data.dropna(subset=["trade_date"])
    duplicates = int(data.duplicated(["ts_code", "trade_date"], keep="last").sum())
    data = (
        data.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    for column in ("open", "high", "low", "close", "volume", "amount_yuan", "pct_chg", "limit_rate"):
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    has_production_baseline = "baseline_selected" in data.columns
    if has_production_baseline:
        data["_baseline_selected"] = data.apply(
            lambda row: bool(_row_bool(row, "baseline_selected")),
            axis=1,
        )
    else:
        data["_baseline_selected"] = True
    feature_frames: list[pd.DataFrame] = []
    for _, history in data.groupby("ts_code", sort=True):
        features = jisuan_tezheng_biao(history)
        amount = pd.to_numeric(features["amount_yuan"], errors="coerce")
        features["_trailing_amount_yuan"] = amount.rolling(
            settings.amount_window_sessions,
            min_periods=settings.amount_window_sessions,
        ).mean()
        features["_history_sessions"] = np.arange(1, len(features) + 1, dtype=int)
        feature_frames.append(features)
    enriched = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    warnings: list[str] = []
    if invalid_dates:
        warnings.append(f"已忽略 {invalid_dates} 条日期无效的历史行")
    if duplicates:
        warnings.append(f"发现 {duplicates} 条重复股票交易日，已保留最后一条")
    if not has_production_baseline:
        warnings.append("面板未提供 baseline_selected；比较基线降级为同日全部可执行样本，不代表现有生产排名基线")
    if int(enriched["amount_yuan"].notna().sum()) == 0:
        warnings.append("面板缺少真实成交额；流动性和动态成本约束无法完成")
    return enriched.replace([np.inf, -np.inf], np.nan), warnings, has_production_baseline


def _market_regime(row: pd.Series) -> str:
    score = _number(row.get("market_regime_score"))
    if score is not None:
        return "weak" if score <= -0.20 else "strong" if score >= 0.20 else "sideways"
    for column, label in (
        ("market_regime_weak", "weak"),
        ("market_regime_strong", "strong"),
        ("market_regime_sideways", "sideways"),
    ):
        value = _number(row.get(column))
        if value is not None and value >= 0.5:
            return label
    return "unavailable"


def _industry_strength(row: pd.Series) -> str:
    industry_return = _number(row.get("industry_mean_ret_20"))
    universe_return = _number(row.get("universe_mean_ret_20"))
    if industry_return is None or universe_return is None:
        return "unavailable"
    if industry_return > max(universe_return, 0.0):
        return "strong"
    if industry_return < min(universe_return, 0.0):
        return "weak"
    return "neutral"


def _relative_strength(row: pd.Series) -> str:
    value = _number(row.get("excess_vs_industry_ret_20"))
    if value is None:
        value = _number(row.get("excess_vs_universe_ret_20"))
    if value is None:
        return "unavailable"
    return "strong" if value > 0 else "weak_or_equal"


def _price_volume_state(row: pd.Series) -> str:
    price_return = _number(row.get("ret_5"))
    volume_ratio = _number(row.get("volume_ratio_5_20"))
    if price_return is None or volume_ratio is None:
        return "unavailable"
    if price_return > 0 and volume_ratio >= 1:
        return "rising_with_volume"
    if price_return > 0:
        return "rising_without_volume"
    if price_return < 0 and volume_ratio >= 1:
        return "falling_with_volume"
    if price_return < 0:
        return "falling_on_contracting_volume"
    return "flat"


def _sample_scope(row: pd.Series) -> str:
    for column in ("sample_scope", "stock_scope"):
        value = _text(row.get(column))
        if value:
            return value
    code = _text(row.get("ts_code"))
    if code.endswith(".BJ"):
        return "beijing"
    digits = code.split(".")[0]
    if digits.startswith(("300", "301")):
        return "chinext"
    if digits.startswith(("688", "689")):
        return "star"
    return "main_board"


def _structure_variants(
    config: Mapping[str, Any] | MacdJiegouPeizhi | None,
    *,
    enabled: bool,
) -> dict[str, MacdJiegouPeizhi]:
    base = config if isinstance(config, MacdJiegouPeizhi) else MacdJiegouPeizhi.from_mapping(config)
    variants = {"configured": base}
    if not enabled:
        return variants
    base_values = asdict(base)
    for label, ratio in (("threshold_looser_20pct", 0.8), ("threshold_stricter_20pct", 1.2)):
        values = dict(base_values)
        values["minimum_price_change_pct"] *= ratio
        values["minimum_indicator_change_pct"] *= ratio
        variants[label] = MacdJiegouPeizhi(**values)
    shorter = dict(base_values)
    shorter["pivot_left_sessions"] = max(1, int(base.pivot_left_sessions) - 1)
    shorter["pivot_right_sessions"] = max(1, int(base.pivot_right_sessions) - 1)
    if shorter != base_values:
        variants["confirmation_window_shorter"] = MacdJiegouPeizhi(**shorter)
    longer = dict(base_values)
    longer["pivot_left_sessions"] = min(10, int(base.pivot_left_sessions) + 1)
    longer["pivot_right_sessions"] = min(10, int(base.pivot_right_sessions) + 1)
    if longer != base_values:
        variants["confirmation_window_longer"] = MacdJiegouPeizhi(**longer)
    return variants


def _event_ledger(
    panel: pd.DataFrame,
    variants: Mapping[str, MacdJiegouPeizhi],
    settings: MacdHuifangPeizhi,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows_by_key = {
        (str(row["ts_code"]), pd.Timestamp(row["trade_date"])): row
        for _, row in panel.iterrows()
    }
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    for variant_name, structure_settings in variants.items():
        for code, history in panel.groupby("ts_code", sort=True):
            result = lieju_macd_jiegou_shijian(history, structure_settings)
            if result.get("status") == "error":
                raise RuntimeError(f"{code} 历史结构事件识别失败：{result.get('reason')}")
            if result.get("status") != "ok":
                warnings.append(f"{code} 历史结构事件不可用：{result.get('reason')}")
                continue
            for event in result.get("events", []):
                if int(event.get("signal_history_sessions") or 0) < settings.minimum_signal_history_sessions:
                    continue
                signal_date = pd.Timestamp(event["confirmation_date"])
                row = rows_by_key.get((str(code), signal_date))
                if row is None or not bool(row.get("_baseline_selected", False)):
                    continue
                if _number(row.get("volume")) is None or float(row.get("volume")) <= 0:
                    continue
                industry = _text(row.get("industry")) or None
                industry_verified = _row_bool(row, "industry_membership_as_of_verified")
                events.append(
                    {
                        **event,
                        "configuration_variant": variant_name,
                        "ts_code": str(code),
                        "name": _text(row.get("name")) or None,
                        "signal_date": _date(signal_date),
                        "market_regime": _market_regime(row),
                        "industry": industry,
                        "industry_membership_as_of_verified": bool(industry_verified),
                        "industry_strength": _industry_strength(row),
                        "relative_strength": _relative_strength(row),
                        "price_volume_state": _price_volume_state(row),
                        "sample_scope": _sample_scope(row),
                    }
                )
    return events, list(dict.fromkeys(warnings))


def _family_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            str(event["configuration_variant"]),
            str(event["ts_code"]),
            str(event["signal_date"]),
            str(event["signal_family"]),
        )
        grouped.setdefault(key, []).append(event)
    combined: list[dict[str, Any]] = []
    for values in grouped.values():
        strongest = max(values, key=lambda item: float(item.get("normalized_strength") or 0.0))
        combined.append(
            {
                **strongest,
                "signal_code": str(strongest["signal_family"]),
                "signal_label": str(strongest["signal_label"]),
                "source_signal_codes": sorted({str(item["signal_code"]) for item in values}),
                "source_indicators": sorted(
                    {str(item["indicator"]) for item in values if item.get("indicator")}
                ),
            }
        )
    return sorted(
        combined,
        key=lambda item: (
            str(item["configuration_variant"]),
            str(item["signal_date"]),
            str(item["ts_code"]),
            str(item["signal_family"]),
        ),
    )


def _calendar(panel: pd.DataFrame, trading_calendar: Iterable[Any] | None) -> tuple[list[pd.Timestamp], str]:
    if trading_calendar is None:
        values = pd.to_datetime(panel["trade_date"], errors="coerce").dropna().dt.normalize().unique()
        return sorted(pd.Timestamp(item) for item in values), "union_of_input_stock_dates"
    values = pd.to_datetime(list(trading_calendar), errors="coerce")
    dates = sorted({pd.Timestamp(item).normalize() for item in values if not pd.isna(item)})
    if not dates:
        raise ValueError("显式市场交易日历为空或全部无效")
    return dates, "explicit_market_trading_calendar"


def _locked_limit(row: pd.Series, previous_close: float, *, side: str) -> tuple[bool, str]:
    if _row_bool(row, "price_limit_exempt") is True:
        return False, "point_in_time_limit_exemption"
    explicit_permission = _row_bool(row, "can_buy_at_open" if side == "buy" else "can_sell_at_close")
    if explicit_permission is False:
        return True, "point_in_time_execution_flag"
    explicit_lock = _row_bool(row, "is_limit_up_locked" if side == "buy" else "is_limit_down_locked")
    if explicit_lock is True:
        return True, "point_in_time_limit_lock_flag"
    boundary_column = "limit_up_price" if side == "buy" else "limit_down_price"
    boundary = _number(row.get(boundary_column))
    high = _number(row.get("high"))
    low = _number(row.get("low"))
    if boundary is not None and high is not None and low is not None:
        if side == "buy" and low >= boundary - 0.011:
            return True, "point_in_time_limit_price"
        if side == "sell" and high <= boundary + 0.011:
            return True, "point_in_time_limit_price"
    limit_rate = _number(row.get("limit_rate"))
    if limit_rate is not None and limit_rate > 0 and high is not None and low is not None:
        theoretical = previous_close * (1.0 + limit_rate if side == "buy" else 1.0 - limit_rate)
        tolerance = max(0.011, previous_close * 0.001)
        if side == "buy" and low >= theoretical - tolerance:
            return True, "point_in_time_limit_rate"
        if side == "sell" and high <= theoretical + tolerance:
            return True, "point_in_time_limit_rate"
    open_price = _number(row.get("open"))
    close = _number(row.get("close"))
    if high is None or low is None or open_price is None or close is None or previous_close <= 0:
        return False, "insufficient_limit_fields"
    one_price = abs(high - low) <= max(0.011, previous_close * 0.0005)
    move = open_price / previous_close - 1.0
    if one_price and ((side == "buy" and move >= 0.045) or (side == "sell" and move <= -0.045)):
        return True, "inferred_one_price_limit_lock"
    return False, "not_locked"


def _execution_outcome(
    *,
    code: str,
    signal_date: pd.Timestamp,
    horizon: int,
    panel_rows: Mapping[tuple[str, pd.Timestamp], pd.Series],
    calendar: list[pd.Timestamp],
    calendar_positions: Mapping[pd.Timestamp, int],
    settings: MacdHuifangPeizhi,
    cost_notional: float,
    cost_scenario: Any,
) -> dict[str, Any]:
    base = {
        "ts_code": code,
        "signal_date": _date(signal_date),
        "horizon_sessions": int(horizon),
        "entry_date": None,
        "exit_date": None,
        "execution_status": "unavailable",
        "execution_reason": "",
        "gross_return": None,
        "net_return": None,
        "roundtrip_cost_rate": None,
        "maximum_adverse_excursion": None,
        "maximum_favorable_excursion": None,
        "suspended_sessions_during_hold": None,
    }
    position = calendar_positions.get(signal_date)
    if position is None or position + horizon >= len(calendar):
        return {**base, "execution_status": "future_incomplete", "execution_reason": "目标退出交易日尚无完整行情"}
    entry_date = calendar[position + 1]
    exit_date = calendar[position + horizon]
    base.update({"entry_date": _date(entry_date), "exit_date": _date(exit_date)})
    signal_row = panel_rows.get((code, signal_date))
    entry_row = panel_rows.get((code, entry_date))
    exit_row = panel_rows.get((code, exit_date))
    if signal_row is None:
        return {**base, "execution_status": "data_unavailable", "execution_reason": "信号日行情缺失"}
    if entry_row is None or _row_bool(entry_row, "is_suspended") is True:
        return {**base, "execution_status": "entry_blocked", "execution_reason": "次一市场交易日停牌或行情缺失"}
    entry_volume = _number(entry_row.get("volume"))
    entry_price = _number(entry_row.get("open"))
    previous_close = _number(signal_row.get("close"))
    if entry_volume is None or entry_volume <= 0 or entry_price is None or entry_price <= 0 or previous_close is None:
        return {**base, "execution_status": "entry_blocked", "execution_reason": "次一交易日价量无效或疑似停牌"}
    locked, lock_basis = _locked_limit(entry_row, previous_close, side="buy")
    if locked:
        return {
            **base,
            "execution_status": "entry_blocked",
            "execution_reason": "次一交易日封涨停，按保守口径不可买入",
            "limit_constraint_basis": lock_basis,
        }
    trailing_amount = _number(signal_row.get("_trailing_amount_yuan"))
    if trailing_amount is None:
        return {**base, "execution_status": "entry_blocked", "execution_reason": "信号日前滚动成交额不足，无法验证流动性"}
    if trailing_amount < settings.minimum_trailing_amount_yuan:
        return {**base, "execution_status": "entry_blocked", "execution_reason": "信号日前滚动成交额低于回放流动性门槛"}
    cost_rate, cost_meta = jisuan_gupiao_wangfan_chengben(
        code,
        entry_price,
        cost_notional,
        cost_scenario,
        daily_amount_yuan=trailing_amount,
        atr_pct=_number(signal_row.get("atr_14_pct")),
        trading_settings={
            "max_participation_rate": settings.maximum_participation_rate,
            "dynamic_slippage_enabled": True,
        },
    )
    if cost_rate is None:
        return {
            **base,
            "execution_status": "entry_blocked",
            "execution_reason": str(cost_meta.get("reason") or "整手或成交容量约束不允许建仓"),
        }
    if exit_row is None or _row_bool(exit_row, "is_suspended") is True:
        return {**base, "execution_status": "exit_blocked", "execution_reason": "目标退出日停牌或行情缺失"}
    exit_volume = _number(exit_row.get("volume"))
    exit_price = _number(exit_row.get("close"))
    exit_previous_close = None
    for previous_position in range(position + horizon - 1, -1, -1):
        previous_row = panel_rows.get((code, calendar[previous_position]))
        if previous_row is None:
            continue
        exit_previous_close = _number(previous_row.get("close"))
        if exit_previous_close is not None and exit_previous_close > 0:
            break
    if exit_volume is None or exit_volume <= 0 or exit_price is None or exit_price <= 0 or exit_previous_close is None:
        return {**base, "execution_status": "exit_blocked", "execution_reason": "目标退出日价量无效或前收盘不可用"}
    locked, lock_basis = _locked_limit(exit_row, exit_previous_close, side="sell")
    if locked:
        return {
            **base,
            "execution_status": "exit_blocked",
            "execution_reason": "目标退出日封跌停，按保守口径不可卖出",
            "limit_constraint_basis": lock_basis,
        }
    interval_dates = calendar[position + 1 : position + horizon + 1]
    interval_rows = [panel_rows.get((code, item)) for item in interval_dates]
    missing_sessions = sum(item is None or _number(item.get("volume")) in {None, 0.0} for item in interval_rows)
    lows = [_number(item.get("low")) for item in interval_rows if item is not None]
    highs = [_number(item.get("high")) for item in interval_rows if item is not None]
    valid_lows = [item for item in lows if item is not None and item > 0]
    valid_highs = [item for item in highs if item is not None and item > 0]
    gross_return = exit_price / entry_price - 1.0
    net_return = koujian_jiaoyi_chengben(gross_return, cost_rate)
    dynamic = cost_meta.get("dynamic_slippage") or {}
    return {
        **base,
        "execution_status": "completed",
        "execution_reason": "",
        "gross_return": round(gross_return, 8),
        "net_return": round(float(net_return), 8),
        "roundtrip_cost_rate": round(float(cost_rate), 8),
        "estimated_buy_shares": cost_meta.get("estimated_buy_shares"),
        "estimated_participation_rate": cost_meta.get("estimated_participation_rate"),
        "dynamic_slippage_bps_roundtrip": dynamic.get("dynamic_extra_bps_roundtrip"),
        "maximum_adverse_excursion": round(min(valid_lows) / entry_price - 1.0, 8) if valid_lows else None,
        "maximum_favorable_excursion": round(max(valid_highs) / entry_price - 1.0, 8) if valid_highs else None,
        "suspended_sessions_during_hold": int(missing_sessions),
        "limit_constraint_basis": lock_basis,
    }


def _observations(
    *,
    events: list[dict[str, Any]],
    panel: pd.DataFrame,
    calendar: list[pd.Timestamp],
    settings: MacdHuifangPeizhi,
    outcome_cache: dict[tuple[str, pd.Timestamp, int], dict[str, Any]] | None = None,
    baseline_cache: dict[tuple[pd.Timestamp, int, str, str | None], tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    panel_rows = {
        (str(row["ts_code"]), pd.Timestamp(row["trade_date"])): row
        for _, row in panel.iterrows()
    }
    calendar_positions = {value: index for index, value in enumerate(calendar)}
    baseline_codes_by_date = {
        pd.Timestamp(date): group.loc[group["_baseline_selected"], "ts_code"].astype(str).drop_duplicates().tolist()
        for date, group in panel.groupby("trade_date")
    }
    cost_notional, cost_scenario, _, _ = jiazai_chengben_jiashe("research_reference")
    cost_notional = float(settings.target_notional_yuan or cost_notional)
    cache = outcome_cache if outcome_cache is not None else {}
    matched_baseline_cache = baseline_cache if baseline_cache is not None else {}

    def outcome(code: str, signal_date: pd.Timestamp, horizon: int) -> dict[str, Any]:
        key = (code, signal_date, horizon)
        if key not in cache:
            cache[key] = _execution_outcome(
                code=code,
                signal_date=signal_date,
                horizon=horizon,
                panel_rows=panel_rows,
                calendar=calendar,
                calendar_positions=calendar_positions,
                settings=settings,
                cost_notional=cost_notional,
                cost_scenario=cost_scenario,
            )
        return cache[key]

    observations: list[dict[str, Any]] = []
    for event in events:
        code = str(event["ts_code"])
        signal_date = pd.Timestamp(event["signal_date"])
        for horizon in settings.forward_horizons:
            actual = outcome(code, signal_date, horizon)
            industry_key = (
                str(event["industry"])
                if event.get("industry_membership_as_of_verified") and event.get("industry")
                else None
            )
            baseline_key = (signal_date, int(horizon), code, industry_key)
            cached_baselines = matched_baseline_cache.get(baseline_key)
            if cached_baselines is None:
                baseline_values = []
                industry_values = []
                for peer_code in baseline_codes_by_date.get(signal_date, []):
                    if peer_code == code:
                        continue
                    peer = outcome(peer_code, signal_date, horizon)
                    if peer.get("execution_status") != "completed":
                        continue
                    baseline_values.append(peer)
                    if industry_key:
                        peer_row = panel_rows.get((peer_code, signal_date))
                        if (
                            peer_row is not None
                            and _row_bool(peer_row, "industry_membership_as_of_verified") is True
                            and _text(peer_row.get("industry")) == industry_key
                        ):
                            industry_values.append(peer)
                cached_baselines = (baseline_values, industry_values)
                matched_baseline_cache[baseline_key] = cached_baselines
            baseline_values, industry_values = cached_baselines
            baseline_returns = [float(item["net_return"]) for item in baseline_values]
            baseline_drawdowns = [
                float(item["maximum_adverse_excursion"])
                for item in baseline_values
                if _finite(item.get("maximum_adverse_excursion"))
            ]
            industry_returns = [float(item["net_return"]) for item in industry_values]
            baseline_mean = (
                float(np.mean(baseline_returns))
                if len(baseline_returns) >= settings.minimum_baseline_peers_per_date
                else None
            )
            industry_mean = (
                float(np.mean(industry_returns))
                if len(industry_returns) >= settings.minimum_baseline_peers_per_date
                else None
            )
            net_return = _number(actual.get("net_return"))
            raw_excess = net_return - baseline_mean if net_return is not None and baseline_mean is not None else None
            favorable_effect = (
                raw_excess
                if raw_excess is not None and event.get("evidence_side") == "support"
                else -raw_excess
                if raw_excess is not None
                else None
            )
            observations.append(
                {
                    **{key: value for key, value in event.items() if key != "details"},
                    **actual,
                    "same_date_baseline_samples": int(len(baseline_returns)),
                    "same_date_baseline_net_return": round(baseline_mean, 8) if baseline_mean is not None else None,
                    "same_date_baseline_drawdown": (
                        round(float(np.mean(baseline_drawdowns)), 8) if baseline_drawdowns else None
                    ),
                    "same_date_same_industry_samples": int(len(industry_returns)),
                    "same_date_same_industry_net_return": round(industry_mean, 8) if industry_mean is not None else None,
                    "raw_excess_vs_baseline": round(raw_excess, 8) if raw_excess is not None else None,
                    "favorable_evidence_effect": round(favorable_effect, 8) if favorable_effect is not None else None,
                }
            )
    return observations


def _time_stability(frame: pd.DataFrame, settings: MacdHuifangPeizhi) -> dict[str, Any]:
    valid = frame.dropna(subset=["favorable_evidence_effect"]).copy()
    dates = sorted(pd.to_datetime(valid["signal_date"], errors="coerce").dropna().dt.normalize().unique())
    slices: list[dict[str, Any]] = []
    for number, date_slice in enumerate(np.array_split(np.asarray(dates, dtype="datetime64[ns]"), settings.stability_subperiods), start=1):
        if not len(date_slice):
            continue
        allowed = {pd.Timestamp(item) for item in date_slice}
        local = valid[pd.to_datetime(valid["signal_date"]).dt.normalize().isin(allowed)]
        effect = _number(local["favorable_evidence_effect"].mean())
        slices.append(
            {
                "subperiod": number,
                "start_date": _date(min(allowed)),
                "end_date": _date(max(allowed)),
                "samples": int(len(local)),
                "signal_dates": int(len(allowed)),
                "mean_favorable_effect": round(effect, 8) if effect is not None else None,
                "favorable": bool(effect is not None and effect > 0),
            }
        )
    usable = [item for item in slices if item["samples"] > 0 and item["mean_favorable_effect"] is not None]
    agreement = float(np.mean([bool(item["favorable"]) for item in usable])) if usable else None
    return {
        "subperiods": slices,
        "valid_subperiods": int(len(usable)),
        "favorable_sign_agreement": round(agreement, 6) if agreement is not None else None,
        "passed": bool(
            len(usable) >= settings.minimum_valid_subperiods
            and agreement is not None
            and agreement >= settings.minimum_favorable_sign_agreement
        ),
    }


def _metrics(frame: pd.DataFrame, settings: MacdHuifangPeizhi) -> dict[str, Any]:
    attempts = int(len(frame))
    completed = frame[frame["execution_status"].eq("completed")].copy()
    matched = completed.dropna(subset=["net_return", "same_date_baseline_net_return", "favorable_evidence_effect"])
    daily_effect = matched.groupby("signal_date")["favorable_evidence_effect"].mean() if not matched.empty else pd.Series(dtype=float)
    mean_effect = _number(daily_effect.mean())
    if len(daily_effect) >= 2:
        standard_error = float(daily_effect.std(ddof=1) / math.sqrt(len(daily_effect)))
        ci_low = float(mean_effect - settings.confidence_z * standard_error) if mean_effect is not None else None
        ci_high = float(mean_effect + settings.confidence_z * standard_error) if mean_effect is not None else None
    else:
        standard_error = ci_low = ci_high = None
    time_stability = _time_stability(matched, settings)
    return {
        "attempted_signals": attempts,
        "completed_trades": int(len(completed)),
        "matched_baseline_samples": int(len(matched)),
        "signal_dates": int(matched["signal_date"].nunique()) if not matched.empty else 0,
        "stocks": int(matched["ts_code"].nunique()) if not matched.empty else 0,
        "execution_completion_rate": round(len(completed) / attempts, 6) if attempts else None,
        "blocked_by_reason": {
            str(key): int(value)
            for key, value in frame.loc[~frame["execution_status"].eq("completed"), "execution_reason"]
            .fillna("未知原因")
            .value_counts()
            .items()
        },
        "mean_gross_return": _json_value(completed["gross_return"].mean()) if not completed.empty else None,
        "mean_net_return": _json_value(completed["net_return"].mean()) if not completed.empty else None,
        "median_net_return": _json_value(completed["net_return"].median()) if not completed.empty else None,
        "positive_net_return_rate": _json_value(completed["net_return"].gt(0).mean()) if not completed.empty else None,
        "mean_maximum_adverse_excursion": (
            _json_value(completed["maximum_adverse_excursion"].mean()) if not completed.empty else None
        ),
        "mean_same_date_baseline_return": (
            _json_value(matched["same_date_baseline_net_return"].mean()) if not matched.empty else None
        ),
        "mean_raw_excess_vs_baseline": (
            _json_value(matched["raw_excess_vs_baseline"].mean()) if not matched.empty else None
        ),
        "mean_favorable_evidence_effect": _json_value(mean_effect),
        "daily_effect_standard_error": _json_value(standard_error),
        "daily_effect_confidence_interval": [_json_value(ci_low), _json_value(ci_high)],
        "time_stability": time_stability,
        "stable_favorable_evidence": bool(
            len(matched) >= settings.minimum_signal_samples
            and ci_low is not None
            and ci_low > 0
            and time_stability["passed"]
        ),
    }


def _group_metrics(
    observations: pd.DataFrame,
    *,
    group_columns: list[str],
    settings: MacdHuifangPeizhi,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if observations.empty:
        return records
    for keys, frame in observations.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = {column: _json_value(value) for column, value in zip(group_columns, key_values)}
        records.append({**identity, **_metrics(frame, settings)})
    return records


def _stratified_metrics(observations: pd.DataFrame, settings: MacdHuifangPeizhi) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dimensions = (
        "market_regime",
        "industry_strength",
        "relative_strength",
        "price_volume_state",
        "sample_scope",
        "cross_region",
    )
    configured = observations[observations["configuration_variant"].eq("configured")]
    for dimension in dimensions:
        if dimension not in configured.columns:
            continue
        for keys, frame in configured.groupby(["signal_family", "horizon_sessions", dimension], dropna=False, sort=True):
            family, horizon, value = keys
            records.append(
                {
                    "signal_family": _json_value(family),
                    "horizon_sessions": int(horizon),
                    "dimension": dimension,
                    "value": _json_value(value),
                    **_metrics(frame, settings),
                }
            )
    return records


def _score_research_assessment(
    family_performance: list[dict[str, Any]],
    sensitivity_performance: list[dict[str, Any]],
    settings: MacdHuifangPeizhi,
) -> list[dict[str, Any]]:
    families = sorted({str(item["signal_family"]) for item in family_performance})
    assessments: list[dict[str, Any]] = []
    for family in families:
        primary = [
            item
            for item in family_performance
            if item["signal_family"] == family and int(item["horizon_sessions"]) in settings.decision_horizons
        ]
        passed_horizons = [int(item["horizon_sessions"]) for item in primary if item["stable_favorable_evidence"]]
        sensitivity = [
            item
            for item in sensitivity_performance
            if item["signal_family"] == family
            and int(item["horizon_sessions"]) in settings.decision_horizons
            and int(item["matched_baseline_samples"]) >= settings.minimum_signal_samples
        ]
        favorable_variants = [
            item for item in sensitivity if _number(item.get("mean_favorable_evidence_effect")) is not None
            and float(item["mean_favorable_evidence_effect"]) > 0
        ]
        sensitivity_agreement = len(favorable_variants) / len(sensitivity) if sensitivity else None
        eligible = bool(
            len(passed_horizons) >= 2
            and sensitivity_agreement is not None
            and sensitivity_agreement >= settings.minimum_favorable_sign_agreement
        )
        assessments.append(
            {
                "signal_family": family,
                "passed_decision_horizons": passed_horizons,
                "parameter_horizon_records": int(len(sensitivity)),
                "parameter_favorable_sign_agreement": _json_value(sensitivity_agreement),
                "research_status": "eligible_for_manual_weight_design" if eligible else "insufficient_stable_gain",
                "production_effect": "none",
                "reason": (
                    "多个期限、时间子区间和附近参数方向一致，可进入受限权重的人工设计与独立复核"
                    if eligible
                    else "尚未同时满足样本量、置信区间、时间稳定性和附近参数稳定性"
                ),
            }
        )
    return assessments


def _coverage(panel: pd.DataFrame, events: list[dict[str, Any]], has_baseline: bool) -> dict[str, Any]:
    def ratio(values: Iterable[bool]) -> float:
        items = list(values)
        return round(float(np.mean(items)), 6) if items else 0.0

    limit_columns = [
        column
        for column in (
            "limit_rate",
            "limit_up_price",
            "limit_down_price",
            "can_buy_at_open",
            "can_sell_at_close",
            "is_limit_up_locked",
            "is_limit_down_locked",
            "price_limit_exempt",
        )
        if column in panel.columns
    ]
    limit_coverage = (
        float(panel[limit_columns].notna().any(axis=1).mean())
        if limit_columns
        else 0.0
    )
    return {
        "production_baseline_available": has_baseline,
        "market_regime_coverage": ratio(event.get("market_regime") != "unavailable" for event in events),
        "industry_strength_coverage": ratio(event.get("industry_strength") != "unavailable" for event in events),
        "point_in_time_industry_verified_coverage": ratio(
            bool(event.get("industry_membership_as_of_verified")) for event in events
        ),
        "relative_strength_coverage": ratio(event.get("relative_strength") != "unavailable" for event in events),
        "price_volume_coverage": ratio(event.get("price_volume_state") != "unavailable" for event in events),
        "actual_amount_coverage": round(float(panel["amount_yuan"].notna().mean()), 6),
        "point_in_time_limit_constraint_coverage": round(limit_coverage, 6),
        "stocks": int(panel["ts_code"].nunique()),
        "start_date": _date(panel["trade_date"].min()),
        "end_date": _date(panel["trade_date"].max()),
        "calendar_days": int((panel["trade_date"].max() - panel["trade_date"].min()).days),
        "sample_scopes": sorted({_sample_scope(row) for _, row in panel.iterrows()}),
    }


def _huifang_macd_jiegou_impl(
    panel: pd.DataFrame,
    *,
    structure_config: Mapping[str, Any] | MacdJiegouPeizhi | None = None,
    validation_config: Mapping[str, Any] | MacdHuifangPeizhi | None = None,
    trading_calendar: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """执行内存历史回放，并返回逐笔证据、分层结果和生产评分保持决定。"""

    try:
        settings = (
            validation_config
            if isinstance(validation_config, MacdHuifangPeizhi)
            else MacdHuifangPeizhi.from_mapping(validation_config)
        )
        settings._validate()
        features, warnings, has_baseline = _prepare_panel(panel, settings)
        calendar, calendar_source = _calendar(features, trading_calendar)
        variants = _structure_variants(
            structure_config,
            enabled=settings.parameter_sensitivity_enabled,
        )
        component_events, event_warnings = _event_ledger(features, variants, settings)
        warnings.extend(event_warnings)
    except ValueError as exc:
        return {
            "status": "unavailable",
            "outcome": "data_unavailable",
            "reason": str(exc),
            "production_score_decision": "unchanged",
        }
    except Exception as exc:
        return {
            "status": "error",
            "outcome": "program_error",
            "reason": str(exc),
            "production_score_decision": "unchanged",
        }
    if not component_events:
        return {
            "status": "insufficient_data",
            "outcome": "information_insufficient",
            "reason": "满足预热期和现有基线范围的历史结构事件为空",
            "production_score_decision": "unchanged",
            "warnings": list(dict.fromkeys(warnings)),
            "data_coverage": _coverage(features, [], has_baseline),
        }

    family_events = _family_events(component_events)
    shared_outcome_cache: dict[tuple[str, pd.Timestamp, int], dict[str, Any]] = {}
    shared_baseline_cache: dict[
        tuple[pd.Timestamp, int, str, str | None],
        tuple[list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    component_observations = _observations(
        events=component_events,
        panel=features,
        calendar=calendar,
        settings=settings,
        outcome_cache=shared_outcome_cache,
        baseline_cache=shared_baseline_cache,
    )
    family_observations = _observations(
        events=family_events,
        panel=features,
        calendar=calendar,
        settings=settings,
        outcome_cache=shared_outcome_cache,
        baseline_cache=shared_baseline_cache,
    )
    component_frame = pd.DataFrame(component_observations)
    family_frame = pd.DataFrame(family_observations)
    configured_components = component_frame[component_frame["configuration_variant"].eq("configured")]
    configured_families = family_frame[family_frame["configuration_variant"].eq("configured")]
    component_performance = _group_metrics(
        configured_components,
        group_columns=["signal_code", "horizon_sessions"],
        settings=settings,
    )
    family_performance = _group_metrics(
        configured_families,
        group_columns=["signal_family", "horizon_sessions"],
        settings=settings,
    )
    sensitivity_performance = _group_metrics(
        family_frame,
        group_columns=["configuration_variant", "signal_family", "horizon_sessions"],
        settings=settings,
    )
    assessments = _score_research_assessment(
        family_performance,
        sensitivity_performance,
        settings,
    )
    coverage = _coverage(features, component_events, has_baseline)
    readiness_reasons: list[str] = []
    if not has_baseline:
        readiness_reasons.append("缺少现有生产排序生成的 baseline_selected 时点标记")
    if coverage["market_regime_coverage"] < 0.8:
        readiness_reasons.append("大盘状态覆盖不足 80%")
    if coverage["point_in_time_industry_verified_coverage"] < 0.8:
        readiness_reasons.append("经时点核验的行业成员覆盖不足 80%")
    if coverage["relative_strength_coverage"] < 0.8 or coverage["price_volume_coverage"] < 0.8:
        readiness_reasons.append("相对强弱或价量分层覆盖不足 80%")
    if coverage["actual_amount_coverage"] < 0.8:
        readiness_reasons.append("真实成交额覆盖不足 80%")
    if coverage["point_in_time_limit_constraint_coverage"] < 0.8:
        readiness_reasons.append("历史时点涨跌停规则或可交易标记覆盖不足 80%")
    if coverage["calendar_days"] < 365:
        readiness_reasons.append("历史跨度不足一个自然年")
    if coverage["stocks"] < settings.minimum_stocks:
        readiness_reasons.append(f"股票数量少于 {settings.minimum_stocks} 只")
    if len(coverage["sample_scopes"]) < settings.minimum_sample_scopes:
        readiness_reasons.append(f"股票范围少于 {settings.minimum_sample_scopes} 类")
    if calendar_source != "explicit_market_trading_calendar":
        readiness_reasons.append("未提供经核验的显式市场交易日历")
    stable_candidates = [item for item in assessments if item["research_status"] == "eligible_for_manual_weight_design"]
    return {
        "status": "ok",
        "outcome": "analysis_success",
        "method": {
            "version": MACD_HUIFANG_METHOD_VERSION,
            "structure_version": MACD_JIEGOU_METHOD_VERSION,
            "future_data_used": False,
            "signal_timing": "收盘确认，次一市场交易日开盘尝试执行",
            "return_definition": "次一市场交易日开盘至指定未来市场交易日收盘的成本后收益",
            "baseline_definition": (
                "同一信号日、未使用结构字段的 baseline_selected 既有候选"
                if has_baseline
                else "同一信号日全部输入股票；仅为降级参照，不等同生产排名基线"
            ),
            "calendar_source": calendar_source,
            "cost_scenario": "research_reference_with_dynamic_liquidity_and_atr_slippage",
            "persistence": "none",
        },
        "configuration": {
            "validation": asdict(settings),
            "structure_variants": {key: asdict(value) for key, value in variants.items()},
        },
        "data_coverage": coverage,
        "validation_readiness": {
            "ready_for_score_decision": not readiness_reasons,
            "unmet_requirements": readiness_reasons,
        },
        "event_counts": {
            "component_events_all_variants": int(len(component_events)),
            "family_events_all_variants": int(len(family_events)),
            "configured_component_events": int(
                sum(event["configuration_variant"] == "configured" for event in component_events)
            ),
            "configured_family_events": int(
                sum(event["configuration_variant"] == "configured" for event in family_events)
            ),
        },
        "component_performance": component_performance,
        "family_performance": family_performance,
        "stratified_performance": _stratified_metrics(family_frame, settings),
        "parameter_sensitivity": sensitivity_performance,
        "research_assessment": assessments,
        "production_score_decision": "unchanged",
        "production_score_reason": (
            "回放结果只形成后续受限权重设计资格，不能自动改生产评分；"
            + ("仍有数据覆盖门槛未满足" if readiness_reasons else "需完成人工复核与独立样本确认")
        ),
        "stable_research_candidates": stable_candidates,
        "component_event_ledger": component_events,
        "family_trade_observations": family_observations,
        "warnings": list(dict.fromkeys(warnings)),
    }


def huifang_macd_jiegou(
    panel: pd.DataFrame,
    *,
    structure_config: Mapping[str, Any] | MacdJiegouPeizhi | None = None,
    validation_config: Mapping[str, Any] | MacdHuifangPeizhi | None = None,
    trading_calendar: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """执行内存历史回放，并保证程序错误也遵守结构化状态契约。"""

    try:
        return _huifang_macd_jiegou_impl(
            panel,
            structure_config=structure_config,
            validation_config=validation_config,
            trading_calendar=trading_calendar,
        )
    except Exception as exc:
        return {
            "status": "error",
            "outcome": "program_error",
            "reason": str(exc),
            "production_score_decision": "unchanged",
        }


__all__ = [
    "MACD_HUIFANG_METHOD_VERSION",
    "MacdHuifangPeizhi",
    "huifang_macd_jiegou",
]
