"""MACD 结构研判的纯计算模块。

本模块只接收统一日线特征，不访问数据源、不修改评分，也不产生买卖结论。
所有拐点都在右侧确认窗口结束后才可见，避免把未来数据回填到拐点日期。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


MACD_JIEGOU_METHOD_VERSION = "macd-structure-v1"


@dataclass(frozen=True)
class MacdJiegouPeizhi:
    """集中维护结构检测口径；所有比例均使用小数。"""

    zero_near_threshold_pct: float = 0.005
    pivot_left_sessions: int = 2
    pivot_right_sessions: int = 2
    pivot_match_sessions: int = 2
    minimum_pivot_separation_sessions: int = 5
    maximum_pivot_separation_sessions: int = 60
    minimum_price_change_pct: float = 0.02
    minimum_indicator_change_pct: float = 0.0002
    cross_fresh_sessions: int = 3
    cross_recent_sessions: int = 8
    cross_max_age_sessions: int = 20
    divergence_max_age_sessions: int = 20
    invalidation_price_tolerance_pct: float = 0.005

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MacdJiegouPeizhi":
        if not value:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        if unknown:
            raise ValueError(f"MACD 结构配置包含未知字段：{', '.join(sorted(unknown))}")
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass(frozen=True)
class MacdZhouZhuangtai:
    code: str
    label: str
    near_zero: bool
    normalized_zero_distance_pct: float


@dataclass(frozen=True)
class MacdJiaochaJieguo:
    status: str
    signal_type: str | None
    label: str
    event_date: str | None
    region: str | None
    region_label: str | None
    age_trading_sessions: int | None
    freshness: str | None
    freshness_label: str | None
    normalized_gap_pct: float | None
    normalized_strength: float | None
    invalidation_reason: str | None
    invalidation_condition: str | None


@dataclass(frozen=True)
class MacdDongnengJieguo:
    status: str
    code: str | None
    label: str
    observation_sessions: int
    latest_histogram_pct: float | None


@dataclass(frozen=True)
class MacdGuaiDian:
    date: str
    confirmation_date: str
    value: float
    index: int
    confirmation_index: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "confirmation_date": self.confirmation_date,
            "value": round(float(self.value), 8),
        }


@dataclass(frozen=True)
class MacdBeiliJieguo:
    status: str
    kind: str
    kind_label: str
    indicator: str
    indicator_label: str
    confirmation_date: str | None
    age_trading_sessions: int | None
    first_price_pivot: MacdGuaiDian | None
    second_price_pivot: MacdGuaiDian | None
    first_indicator_pivot: MacdGuaiDian | None
    second_indicator_pivot: MacdGuaiDian | None
    price_change_pct: float | None
    indicator_change_pct_of_price: float | None
    normalized_strength: float | None
    strength_definition: str
    evidence_reliability: str | None
    invalidation_date: str | None
    invalidation_reason: str | None
    invalidation_conditions: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "first_price_pivot",
            "second_price_pivot",
            "first_indicator_pivot",
            "second_indicator_pivot",
        ):
            pivot = getattr(self, key)
            value[key] = pivot.public_dict() if pivot is not None else None
        value["invalidation_conditions"] = list(self.invalidation_conditions)
        value["reason"] = (
            "有效历史不足以确认两个可比较拐点"
            if self.status == "insufficient_data"
            else "价格或指标序列不完整，无法可靠检测背离"
            if self.status == "unavailable"
            else "有效窗口内未发现满足间隔和变化门槛的背离"
            if self.status == "no_signal"
            else ""
        )
        return value


@dataclass(frozen=True)
class _MacdBeiliHouXuan:
    """已经满足门槛、但尚未套用回看期后失效状态的背离候选。"""

    confirmation_index: int
    first_price: MacdGuaiDian
    second_price: MacdGuaiDian
    first_indicator: MacdGuaiDian
    second_indicator: MacdGuaiDian
    price_change: float
    indicator_change: float


class _MacdDataUnavailable(ValueError):
    """输入特征不足以进行结构研判，而非程序逻辑错误。"""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _round(value: Any, digits: int = 8) -> float | None:
    return round(float(value), digits) if _finite(value) else None


def _base_result(
    *,
    status: str,
    outcome: str,
    reason: str,
    as_of: str | None,
    history_length: int,
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "status": status,
        "outcome": outcome,
        "reason": reason,
        "as_of": as_of,
        "effective_history_length": int(history_length),
        "zero_axis": None,
        "latest_cross": asdict(_empty_cross("insufficient_data")),
        "momentum": asdict(
            MacdDongnengJieguo("insufficient_data", None, "有效柱体历史不足", 0, None)
        ),
        "divergences": {
            "bottom": {
                "dif": _empty_divergence("bottom", "dif", "insufficient_data").public_dict(),
                "histogram": _empty_divergence("bottom", "histogram", "insufficient_data").public_dict(),
            },
            "top": {
                "dif": _empty_divergence("top", "dif", "insufficient_data").public_dict(),
                "histogram": _empty_divergence("top", "histogram", "insufficient_data").public_dict(),
            },
        },
        "structure_classification": {"code": "unavailable", "label": "结构信息不可用"},
        "evidence_reliability": {"code": "unavailable", "label": "结构证据不可用"},
        "supporting_evidence": [],
        "counter_evidence": [],
        "risk_warnings": [],
        "invalidation_conditions": [],
        "warnings": list(warnings),
        "score_effect": "仅作为解释性证据，不进入候选排名或预测模型",
        "purpose_statement": "该结果描述指标结构，不是上涨概率、买卖指令或收益承诺",
        "method": {"version": MACD_JIEGOU_METHOD_VERSION, "future_data_used": False},
    }


def _prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if features is None or features.empty:
        return pd.DataFrame(), []
    required = {"trade_date", "close", "macd_dif", "macd_dea", "macd_hist"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise _MacdDataUnavailable(f"统一日线特征缺少字段：{', '.join(missing)}")
    data = features.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    invalid_dates = int(data["trade_date"].isna().sum())
    data = data.dropna(subset=["trade_date"]).sort_values("trade_date")
    duplicate_dates = int(data.duplicated("trade_date", keep="last").sum())
    data = data.drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    for column in ("close", "high", "low", "macd_dif", "macd_dea", "macd_hist"):
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid_close = data["close"].where(data["close"] > 0)
    # 归一化字段统一从原始 DIF/DEA/柱体重算，避免不同调用入口各自持有不一致口径。
    data["macd_dif_pct"] = data["macd_dif"] / valid_close
    data["macd_dea_pct"] = data["macd_dea"] / valid_close
    data["macd_gap_pct"] = (data["macd_dif"] - data["macd_dea"]) / valid_close
    data["macd_hist_pct"] = data["macd_hist"] / valid_close
    data["macd_zero_distance_pct"] = pd.concat(
        [data["macd_dif_pct"].abs(), data["macd_dea_pct"].abs()], axis=1
    ).max(axis=1, skipna=False)
    warnings: list[str] = []
    if data["high"].isna().all() or data["low"].isna().all():
        warnings.append("缺少完整高低价，背离拐点检测暂不可用")
    data = data.replace([np.inf, -np.inf], np.nan)
    if invalid_dates:
        warnings.append(f"已忽略 {invalid_dates} 条日期无效的日线")
    if duplicate_dates:
        warnings.append(f"发现 {duplicate_dates} 个重复交易日，已按日期保留最后一条")
    return data, warnings


def _detect_zero_axis(row: pd.Series, settings: MacdJiegouPeizhi) -> MacdZhouZhuangtai:
    dif = float(row["macd_dif"])
    dea = float(row["macd_dea"])
    distance = float(row["macd_zero_distance_pct"])
    near_zero = distance <= settings.zero_near_threshold_pct
    if dif > 0 and dea > 0:
        code, label = "above", "快线和慢线均在零轴上方"
    elif dif < 0 and dea < 0:
        code, label = "below", "快线和慢线均在零轴下方"
    else:
        code, label = "mixed", "快线和慢线分处零轴两侧或有一条贴近零轴"
    return MacdZhouZhuangtai(code, label, near_zero, round(distance, 8))


def _cross_region(row: pd.Series, settings: MacdJiegouPeizhi) -> tuple[str, str]:
    if float(row["macd_zero_distance_pct"]) <= settings.zero_near_threshold_pct:
        return "near_zero", "零轴附近"
    dif, dea = float(row["macd_dif"]), float(row["macd_dea"])
    if dif > 0 and dea > 0:
        return "above", "零轴上方"
    if dif < 0 and dea < 0:
        return "below", "零轴下方"
    return "mixed", "零轴两侧"


def _empty_cross(status: str = "no_signal") -> MacdJiaochaJieguo:
    label = "有效历史内没有检测到快慢线交叉" if status == "no_signal" else "有效历史不足以判断交叉"
    return MacdJiaochaJieguo(
        status, None, label, None, None, None, None, None, None, None, None, None, None
    )


def _detect_cross(data: pd.DataFrame, settings: MacdJiegouPeizhi) -> MacdJiaochaJieguo:
    events: list[tuple[int, str]] = []
    for index in range(1, len(data)):
        previous = data.iloc[index - 1]
        current = data.iloc[index]
        if not (_finite(previous["macd_gap_pct"]) and _finite(current["macd_gap_pct"])):
            continue
        old_gap = float(previous["macd_gap_pct"])
        new_gap = float(current["macd_gap_pct"])
        if old_gap <= 0 < new_gap:
            events.append((index, "golden"))
        elif old_gap >= 0 > new_gap:
            events.append((index, "death"))
    if not events:
        return _empty_cross()
    event_index, signal_type = events[-1]
    row = data.iloc[event_index]
    previous = data.iloc[event_index - 1]
    age = len(data) - 1 - event_index
    if age <= settings.cross_fresh_sessions:
        freshness, freshness_label = "fresh", "较新"
    elif age <= settings.cross_recent_sessions:
        freshness, freshness_label = "recent", "近期"
    else:
        freshness, freshness_label = "stale", "陈旧"
    region, region_label = _cross_region(row, settings)
    active = float(data.iloc[-1]["macd_gap_pct"]) > 0 if signal_type == "golden" else float(data.iloc[-1]["macd_gap_pct"]) < 0
    invalidation_condition = "快线重新不高于慢线" if signal_type == "golden" else "快线重新不低于慢线"
    invalidation_reason = None
    status = "active"
    if not active:
        status = "invalidated"
        invalidation_reason = invalidation_condition
    elif age > settings.cross_max_age_sessions:
        status = "expired"
        invalidation_reason = f"交叉已超过 {settings.cross_max_age_sessions} 个交易日的展示有效期"
    label = "金叉" if signal_type == "golden" else "死叉"
    return MacdJiaochaJieguo(
        status=status,
        signal_type=signal_type,
        label=label,
        event_date=_date(row["trade_date"]),
        region=region,
        region_label=region_label,
        age_trading_sessions=age,
        freshness=freshness,
        freshness_label=freshness_label,
        normalized_gap_pct=_round(row["macd_gap_pct"]),
        normalized_strength=_round(abs(float(row["macd_gap_pct"]) - float(previous["macd_gap_pct"]))),
        invalidation_reason=invalidation_reason,
        invalidation_condition=invalidation_condition,
    )


def _detect_momentum(data: pd.DataFrame) -> MacdDongnengJieguo:
    values = pd.to_numeric(data["macd_hist_pct"], errors="coerce")
    if len(values) < 3 or values.iloc[-3:].isna().any():
        return MacdDongnengJieguo("insufficient_data", None, "连续柱体历史不足", 0, None)
    recent = values.iloc[-3:].to_numpy(dtype=float)
    changes = np.diff(recent)
    code = "mixed"
    label = "近三日柱体变化方向不连续"
    if np.all(recent > 0) and np.all(changes > 0):
        code, label = "positive_strengthening", "正柱连续扩大，正向动能增强"
    elif np.all(recent > 0) and np.all(changes < 0):
        code, label = "positive_weakening", "正柱连续收窄，正向动能衰减"
    elif np.all(recent < 0) and np.all(changes > 0):
        code, label = "negative_weakening", "负柱连续收窄，下跌动能减弱"
    elif np.all(recent < 0) and np.all(changes < 0):
        code, label = "negative_strengthening", "负柱连续扩大，下跌动能增强"
    return MacdDongnengJieguo("ok", code, label, 3, round(float(recent[-1]), 8))


def _find_pivots(
    data: pd.DataFrame,
    column: str,
    *,
    mode: str,
    settings: MacdJiegouPeizhi,
) -> list[MacdGuaiDian]:
    values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    pivots: list[MacdGuaiDian] = []
    left_count = settings.pivot_left_sessions
    right_count = settings.pivot_right_sessions
    for index in range(left_count, len(data) - right_count):
        window = values[index - left_count : index + right_count + 1]
        if not np.isfinite(window).all():
            continue
        center = float(values[index])
        left = window[:left_count]
        right = window[left_count + 1 :]
        tolerance = max(abs(center) * 1e-10, 1e-12)
        is_pivot = (
            center < float(left.min()) - tolerance and center < float(right.min()) - tolerance
            if mode == "low"
            else center > float(left.max()) + tolerance and center > float(right.max()) + tolerance
        )
        if not is_pivot:
            continue
        confirmation_index = index + right_count
        pivots.append(
            MacdGuaiDian(
                date=_date(data.iloc[index]["trade_date"]),
                confirmation_date=_date(data.iloc[confirmation_index]["trade_date"]),
                value=center,
                index=index,
                confirmation_index=confirmation_index,
            )
        )
    return pivots


def _matching_pivot(
    pivots: Iterable[MacdGuaiDian],
    price_pivot: MacdGuaiDian,
    settings: MacdJiegouPeizhi,
) -> MacdGuaiDian | None:
    matches = [
        pivot
        for pivot in pivots
        if abs(pivot.index - price_pivot.index) <= settings.pivot_match_sessions
    ]
    if not matches:
        return None
    return min(matches, key=lambda pivot: (abs(pivot.index - price_pivot.index), pivot.index))


def _empty_divergence(kind: str, indicator: str, status: str = "no_signal") -> MacdBeiliJieguo:
    kind_label = "底部背离" if kind == "bottom" else "顶部背离"
    indicator_label = "快线" if indicator == "dif" else "柱体"
    conditions = (
        "价格跌破第二个低点或指标形成新的更低低点"
        if kind == "bottom"
        else "价格突破第二个高点或指标形成新的更高高点",
        "信号超过最大有效期或被更新结构覆盖",
    )
    return MacdBeiliJieguo(
        status=status,
        kind=kind,
        kind_label=kind_label,
        indicator=indicator,
        indicator_label=indicator_label,
        confirmation_date=None,
        age_trading_sessions=None,
        first_price_pivot=None,
        second_price_pivot=None,
        first_indicator_pivot=None,
        second_indicator_pivot=None,
        price_change_pct=None,
        indicator_change_pct_of_price=None,
        normalized_strength=None,
        strength_definition="价格与指标变化分别相对最低门槛的较小倍数；不是概率",
        evidence_reliability=None,
        invalidation_date=None,
        invalidation_reason=None,
        invalidation_conditions=conditions,
    )


def _invalidation(
    data: pd.DataFrame,
    *,
    kind: str,
    indicator_column: str,
    second_price: MacdGuaiDian,
    second_indicator: MacdGuaiDian,
    confirmation_index: int,
    settings: MacdJiegouPeizhi,
) -> tuple[str | None, str | None]:
    later = data.iloc[confirmation_index + 1 :]
    for _, row in later.iterrows():
        if kind == "bottom":
            price_broken = _finite(row["low"]) and float(row["low"]) < second_price.value * (
                1.0 - settings.invalidation_price_tolerance_pct
            )
            indicator_broken = _finite(row[indicator_column]) and float(row[indicator_column]) < (
                second_indicator.value - settings.minimum_indicator_change_pct
            )
            reason = "价格跌破背离低点" if price_broken else "指标形成新的更低低点"
        else:
            price_broken = _finite(row["high"]) and float(row["high"]) > second_price.value * (
                1.0 + settings.invalidation_price_tolerance_pct
            )
            indicator_broken = _finite(row[indicator_column]) and float(row[indicator_column]) > (
                second_indicator.value + settings.minimum_indicator_change_pct
            )
            reason = "价格突破背离高点" if price_broken else "指标形成新的更高高点"
        if price_broken or indicator_broken:
            return _date(row["trade_date"]), reason
    return None, None


def _divergence_candidates(
    data: pd.DataFrame,
    *,
    kind: str,
    indicator: str,
    settings: MacdJiegouPeizhi,
) -> list[_MacdBeiliHouXuan]:
    """列出每个第二价格拐点在当时能够确认的最近一组背离。"""

    price_column = "low" if kind == "bottom" else "high"
    mode = "low" if kind == "bottom" else "high"
    indicator_column = "macd_dif_pct" if indicator == "dif" else "macd_hist_pct"
    price_pivots = _find_pivots(data, price_column, mode=mode, settings=settings)
    indicator_pivots = _find_pivots(data, indicator_column, mode=mode, settings=settings)
    candidates: list[_MacdBeiliHouXuan] = []
    for second_position in range(1, len(price_pivots)):
        second_price = price_pivots[second_position]
        for first_position in range(second_position - 1, -1, -1):
            first_price = price_pivots[first_position]
            separation = second_price.index - first_price.index
            if separation < settings.minimum_pivot_separation_sessions:
                continue
            if separation > settings.maximum_pivot_separation_sessions:
                break
            first_indicator = _matching_pivot(indicator_pivots, first_price, settings)
            second_indicator = _matching_pivot(indicator_pivots, second_price, settings)
            if (
                first_indicator is None
                or second_indicator is None
                or first_indicator.index >= second_indicator.index
            ):
                continue
            price_change = second_price.value / first_price.value - 1.0
            indicator_change = second_indicator.value - first_indicator.value
            matches = (
                price_change <= -settings.minimum_price_change_pct
                and indicator_change >= settings.minimum_indicator_change_pct
                if kind == "bottom"
                else price_change >= settings.minimum_price_change_pct
                and indicator_change <= -settings.minimum_indicator_change_pct
            )
            if not matches:
                continue
            candidates.append(
                _MacdBeiliHouXuan(
                    confirmation_index=max(
                        second_price.confirmation_index,
                        second_indicator.confirmation_index,
                    ),
                    first_price=first_price,
                    second_price=second_price,
                    first_indicator=first_indicator,
                    second_indicator=second_indicator,
                    price_change=price_change,
                    indicator_change=indicator_change,
                )
            )
            # 同一个第二价格拐点只保留当时最近的合格前拐点，避免重复计数。
            break
    return candidates


def _divergence_result(
    data: pd.DataFrame,
    *,
    kind: str,
    indicator: str,
    candidate: _MacdBeiliHouXuan,
    settings: MacdJiegouPeizhi,
    evaluate_later_invalidation: bool,
) -> MacdBeiliJieguo:
    confirmation_index = candidate.confirmation_index
    age = len(data) - 1 - confirmation_index
    invalidation_date: str | None = None
    invalidation_reason: str | None = None
    if evaluate_later_invalidation:
        invalidation_date, invalidation_reason = _invalidation(
            data,
            kind=kind,
            indicator_column="macd_dif_pct" if indicator == "dif" else "macd_hist_pct",
            second_price=candidate.second_price,
            second_indicator=candidate.second_indicator,
            confirmation_index=confirmation_index,
            settings=settings,
        )
    status = "confirmed"
    if invalidation_date:
        status = "invalidated"
    elif age > settings.divergence_max_age_sessions:
        status = "expired"
        invalidation_reason = f"背离已超过 {settings.divergence_max_age_sessions} 个交易日的有效期"
    price_multiple = abs(candidate.price_change) / settings.minimum_price_change_pct
    indicator_multiple = abs(candidate.indicator_change) / settings.minimum_indicator_change_pct
    offsets = (
        abs(candidate.first_price.index - candidate.first_indicator.index),
        abs(candidate.second_price.index - candidate.second_indicator.index),
    )
    reliability = "较高" if max(offsets) == 0 else "中等" if max(offsets) == 1 else "有限"
    base = _empty_divergence(kind, indicator)
    return MacdBeiliJieguo(
        status=status,
        kind=base.kind,
        kind_label=base.kind_label,
        indicator=base.indicator,
        indicator_label=base.indicator_label,
        confirmation_date=_date(data.iloc[confirmation_index]["trade_date"]),
        age_trading_sessions=age,
        first_price_pivot=candidate.first_price,
        second_price_pivot=candidate.second_price,
        first_indicator_pivot=candidate.first_indicator,
        second_indicator_pivot=candidate.second_indicator,
        price_change_pct=round(candidate.price_change, 8),
        indicator_change_pct_of_price=round(candidate.indicator_change, 8),
        normalized_strength=round(min(price_multiple, indicator_multiple), 4),
        strength_definition=base.strength_definition,
        evidence_reliability=reliability,
        invalidation_date=invalidation_date,
        invalidation_reason=invalidation_reason,
        invalidation_conditions=base.invalidation_conditions,
    )


def _detect_divergence(
    data: pd.DataFrame,
    *,
    kind: str,
    indicator: str,
    settings: MacdJiegouPeizhi,
) -> MacdBeiliJieguo:
    price_column = "low" if kind == "bottom" else "high"
    indicator_column = "macd_dif_pct" if indicator == "dif" else "macd_hist_pct"
    minimum_length = (
        settings.pivot_left_sessions
        + settings.pivot_right_sessions
        + settings.minimum_pivot_separation_sessions
        + 2
    )
    if len(data) < minimum_length:
        return _empty_divergence(kind, indicator, "insufficient_data")
    if (
        int(data[price_column].notna().sum()) < minimum_length
        or int(data[indicator_column].notna().sum()) < minimum_length
    ):
        return _empty_divergence(kind, indicator, "unavailable")
    candidates = _divergence_candidates(
        data,
        kind=kind,
        indicator=indicator,
        settings=settings,
    )
    if not candidates:
        return _empty_divergence(kind, indicator)
    return _divergence_result(
        data,
        kind=kind,
        indicator=indicator,
        candidate=max(candidates, key=lambda item: item.confirmation_index),
        settings=settings,
        evaluate_later_invalidation=True,
    )


def _active_divergences(divergences: dict[str, dict[str, MacdBeiliJieguo]], kind: str) -> list[MacdBeiliJieguo]:
    return [value for value in divergences[kind].values() if value.status == "confirmed"]


def _classification(
    zero_axis: MacdZhouZhuangtai,
    cross: MacdJiaochaJieguo,
    momentum: MacdDongnengJieguo,
    divergences: dict[str, dict[str, MacdBeiliJieguo]],
) -> tuple[str, str]:
    if _active_divergences(divergences, "top"):
        return "top_risk", "出现经确认的顶部风险结构"
    if cross.status == "active" and cross.signal_type == "death":
        return "momentum_weakening", "快慢线死叉，动能转弱"
    if zero_axis.code == "above" and momentum.code in {"positive_strengthening", "negative_weakening"}:
        return "trend_continuation", "零轴上方的趋势延续证据较多"
    if zero_axis.code == "below" and (
        (cross.status == "active" and cross.signal_type == "golden")
        or _active_divergences(divergences, "bottom")
    ):
        return "weak_rebound", "零轴下方出现修复证据，仍属于弱势反弹观察"
    if momentum.code in {"positive_weakening", "negative_strengthening"}:
        return "momentum_weakening", "柱体显示动能衰减或下跌动能增强"
    return "mixed", "结构证据方向不一致"


def lieju_macd_jiegou_shijian(
    features: pd.DataFrame,
    config: Mapping[str, Any] | MacdJiegouPeizhi | None = None,
) -> dict[str, Any]:
    """列出历史上当日已经可知的交叉和背离确认事件。

    该入口供历史回放使用。它一次扫描完整内存序列，但事件日期严格取交叉日或
    右侧确认完成日；事件内容只读取该日期及以前的行，不把后续失效信息写回事件。
    """

    try:
        settings = config if isinstance(config, MacdJiegouPeizhi) else MacdJiegouPeizhi.from_mapping(config)
        data, warnings = _prepare_features(features)
    except _MacdDataUnavailable as exc:
        return {
            "status": "unavailable",
            "outcome": "data_unavailable",
            "reason": str(exc),
            "events": [],
            "warnings": [],
        }
    except Exception as exc:
        return {
            "status": "error",
            "outcome": "program_error",
            "reason": str(exc),
            "events": [],
            "warnings": [],
        }
    if data.empty:
        return {
            "status": "unavailable",
            "outcome": "data_unavailable",
            "reason": "没有可用于历史事件识别的日线特征",
            "events": [],
            "warnings": warnings,
        }

    events: list[dict[str, Any]] = []
    for index in range(1, len(data)):
        previous = data.iloc[index - 1]
        current = data.iloc[index]
        if not (
            _finite(previous["macd_gap_pct"])
            and _finite(current["macd_gap_pct"])
            and _finite(current["close"])
            and float(current["close"]) > 0
        ):
            continue
        old_gap = float(previous["macd_gap_pct"])
        new_gap = float(current["macd_gap_pct"])
        signal_type = "golden" if old_gap <= 0 < new_gap else "death" if old_gap >= 0 > new_gap else None
        if signal_type is None:
            continue
        prefix = data.iloc[: index + 1]
        region, region_label = _cross_region(current, settings)
        zero_axis = _detect_zero_axis(current, settings)
        momentum = _detect_momentum(prefix)
        label = "金叉" if signal_type == "golden" else "死叉"
        events.append(
            {
                "signal_code": f"{signal_type}_cross",
                "signal_family": f"{signal_type}_cross",
                "signal_label": label,
                "evidence_side": "support" if signal_type == "golden" else "risk",
                "confirmation_date": _date(current["trade_date"]),
                "source_pivot_date": None,
                "confirmation_delay_sessions": 0,
                "indicator": None,
                "indicator_label": None,
                "cross_region": region,
                "cross_region_label": region_label,
                "zero_axis": asdict(zero_axis),
                "momentum": asdict(momentum),
                "normalized_strength": _round(abs(new_gap - old_gap)),
                "evidence_reliability": "较高",
                "signal_history_sessions": index + 1,
                "details": {
                    "previous_normalized_gap_pct": _round(old_gap),
                    "current_normalized_gap_pct": _round(new_gap),
                },
            }
        )

    minimum_length = (
        settings.pivot_left_sessions
        + settings.pivot_right_sessions
        + settings.minimum_pivot_separation_sessions
        + 2
    )
    divergence_availability: dict[str, str] = {}
    for kind in ("bottom", "top"):
        price_column = "low" if kind == "bottom" else "high"
        for indicator in ("dif", "histogram"):
            indicator_column = "macd_dif_pct" if indicator == "dif" else "macd_hist_pct"
            component = f"{kind}_{indicator}"
            if len(data) < minimum_length:
                divergence_availability[component] = "insufficient_data"
                continue
            if (
                int(data[price_column].notna().sum()) < minimum_length
                or int(data[indicator_column].notna().sum()) < minimum_length
            ):
                divergence_availability[component] = "unavailable"
                continue
            divergence_availability[component] = "ok"
            for candidate in _divergence_candidates(
                data,
                kind=kind,
                indicator=indicator,
                settings=settings,
            ):
                confirmation_index = candidate.confirmation_index
                prefix = data.iloc[: confirmation_index + 1]
                signal = _divergence_result(
                    prefix,
                    kind=kind,
                    indicator=indicator,
                    candidate=candidate,
                    settings=settings,
                    evaluate_later_invalidation=False,
                )
                row = data.iloc[confirmation_index]
                zero_axis = _detect_zero_axis(row, settings)
                events.append(
                    {
                        "signal_code": f"{kind}_divergence_{indicator}",
                        "signal_family": f"{kind}_divergence",
                        "signal_label": signal.kind_label,
                        "evidence_side": "support" if kind == "bottom" else "risk",
                        "confirmation_date": signal.confirmation_date,
                        "source_pivot_date": signal.second_price_pivot.date if signal.second_price_pivot else None,
                        "confirmation_delay_sessions": (
                            confirmation_index - candidate.second_price.index
                        ),
                        "indicator": indicator,
                        "indicator_label": signal.indicator_label,
                        "cross_region": None,
                        "cross_region_label": None,
                        "zero_axis": asdict(zero_axis),
                        "momentum": asdict(_detect_momentum(prefix)),
                        "normalized_strength": signal.normalized_strength,
                        "evidence_reliability": signal.evidence_reliability,
                        "signal_history_sessions": confirmation_index + 1,
                        "details": signal.public_dict(),
                    }
                )

    events.sort(
        key=lambda item: (
            str(item.get("confirmation_date") or ""),
            str(item.get("signal_code") or ""),
        )
    )
    return {
        "status": "ok",
        "outcome": "analysis_success",
        "reason": "",
        "as_of": _date(data.iloc[-1]["trade_date"]),
        "effective_history_length": int(len(data)),
        "events": events,
        "event_count": int(len(events)),
        "component_availability": {
            "cross": "ok",
            "divergence": divergence_availability,
        },
        "warnings": warnings,
        "method": {
            "version": MACD_JIEGOU_METHOD_VERSION,
            "future_data_used": False,
            "event_timing": "交叉使用实际发生日；背离使用右侧确认完成日",
            "configuration": asdict(settings),
        },
    }


def yanpan_macd_jiegou(
    features: pd.DataFrame,
    config: Mapping[str, Any] | MacdJiegouPeizhi | None = None,
) -> dict[str, Any]:
    """汇总零轴、交叉、动能、背离及失效状态，返回 JSON 安全结果。"""

    try:
        settings = config if isinstance(config, MacdJiegouPeizhi) else MacdJiegouPeizhi.from_mapping(config)
        data, warnings = _prepare_features(features)
    except _MacdDataUnavailable as exc:
        return _base_result(
            status="unavailable",
            outcome="data_unavailable",
            reason=str(exc),
            as_of=None,
            history_length=0,
        )
    except Exception as exc:
        return _base_result(
            status="error",
            outcome="program_error",
            reason=str(exc),
            as_of=None,
            history_length=0,
        )
    if data.empty:
        return _base_result(
            status="unavailable",
            outcome="data_unavailable",
            reason="没有可用于结构研判的日线特征",
            as_of=None,
            history_length=0,
            warnings=warnings,
        )
    as_of = _date(data.iloc[-1]["trade_date"])
    valid = data[["close", "macd_dif", "macd_dea", "macd_gap_pct", "macd_hist_pct"]].notna().all(axis=1)
    valid &= data["close"].gt(0)
    valid_count = int(valid.sum())
    if valid_count < 2:
        return _base_result(
            status="insufficient_data",
            outcome="information_insufficient",
            reason="历史不足以形成可比较的快线、慢线和柱体",
            as_of=as_of,
            history_length=valid_count,
            warnings=warnings,
        )
    if not bool(valid.iloc[-1]):
        return _base_result(
            status="unavailable",
            outcome="data_unavailable",
            reason="最新交易日缺少有效收盘价或完整指标，未沿用旧状态冒充当前结果",
            as_of=as_of,
            history_length=valid_count,
            warnings=warnings,
        )

    zero_axis = _detect_zero_axis(data.iloc[-1], settings)
    cross = _detect_cross(data, settings)
    momentum = _detect_momentum(data)
    divergences: dict[str, dict[str, MacdBeiliJieguo]] = {
        kind: {
            indicator: _detect_divergence(data, kind=kind, indicator=indicator, settings=settings)
            for indicator in ("dif", "histogram")
        }
        for kind in ("bottom", "top")
    }
    divergence_unavailable = [
        signal
        for values in divergences.values()
        for signal in values.values()
        if signal.status == "unavailable"
    ]
    supporting: list[str] = []
    counter: list[str] = []
    risks: list[str] = []
    if zero_axis.code == "above":
        supporting.append("快线和慢线均在零轴上方，属于偏强趋势背景，但不能单独支持推荐")
    elif zero_axis.code == "below":
        counter.append("快线和慢线均在零轴下方，当前仍处于弱势趋势背景")
    else:
        counter.append("快线和慢线未在零轴同侧，趋势背景尚未一致")
    if cross.status == "active":
        statement = (
            f"{cross.event_date} 在{cross.region_label}形成{cross.label}，距今 {cross.age_trading_sessions} 个交易日"
        )
        if cross.signal_type == "golden" and cross.region != "below":
            supporting.append(statement)
        elif cross.signal_type == "golden":
            counter.append(statement + "；零轴下方金叉只表示弱势修复")
        else:
            counter.append(statement)
    elif cross.status in {"expired", "invalidated"}:
        warnings.append(f"最近一次{cross.label}已{('过期' if cross.status == 'expired' else '失效')}，不作为新信号")
    if momentum.code in {"positive_strengthening", "negative_weakening"}:
        supporting.append(momentum.label)
    elif momentum.code in {"positive_weakening", "negative_strengthening"}:
        counter.append(momentum.label)
    for signal in _active_divergences(divergences, "bottom"):
        supporting.append(
            f"{signal.confirmation_date} 确认{signal.indicator_label}底部背离，只说明下跌动能减弱，不等于已经反转"
        )
    for signal in _active_divergences(divergences, "top"):
        text = f"{signal.confirmation_date} 确认{signal.indicator_label}顶部背离，存在动能衰减风险"
        counter.append(text)
        risks.append(text + "；该证据当前只提示风险、不扣排名分")
    code, label = _classification(zero_axis, cross, momentum, divergences)
    reliability = (
        {"code": "high", "label": "有效历史较完整，结构证据可靠性较高"}
        if valid_count >= 80 and not warnings
        else {"code": "medium", "label": "结构证据可用，但历史长度或数据完整性一般"}
        if valid_count >= 40
        else {"code": "limited", "label": "结构证据可用范围有限，解释时应降低权重"}
    )
    invalidation_conditions = list(
        dict.fromkeys(
            condition
            for kind_values in divergences.values()
            for signal in kind_values.values()
            for condition in signal.invalidation_conditions
        )
    )
    result = _base_result(
        status="unavailable" if divergence_unavailable else "ok",
        outcome="data_unavailable" if divergence_unavailable else "analysis_success",
        reason="价格或指标序列不完整，背离结构无法可靠判断" if divergence_unavailable else "",
        as_of=as_of,
        history_length=valid_count,
        warnings=warnings,
    )
    result.update(
        {
            "zero_axis": asdict(zero_axis),
            "latest_cross": asdict(cross),
            "momentum": asdict(momentum),
            "divergences": {
                kind: {indicator: signal.public_dict() for indicator, signal in values.items()}
                for kind, values in divergences.items()
            },
            "structure_classification": {"code": code, "label": label},
            "evidence_reliability": reliability,
            "supporting_evidence": list(dict.fromkeys(supporting)),
            "counter_evidence": list(dict.fromkeys(counter)),
            "risk_warnings": list(dict.fromkeys(risks)),
            "invalidation_conditions": invalidation_conditions,
            "method": {
                "version": MACD_JIEGOU_METHOD_VERSION,
                "future_data_used": False,
                "pivot_confirmation": (
                    f"左右各 {settings.pivot_left_sessions}/{settings.pivot_right_sessions} 个交易日，"
                    "信号日期取右侧确认完成日"
                ),
                "price_scale_invariant": True,
                "configuration": asdict(settings),
            },
        }
    )
    return result


__all__ = [
    "MACD_JIEGOU_METHOD_VERSION",
    "MacdBeiliJieguo",
    "MacdDongnengJieguo",
    "MacdJiaochaJieguo",
    "MacdJiegouPeizhi",
    "MacdZhouZhuangtai",
    "lieju_macd_jiegou_shijian",
    "yanpan_macd_jiegou",
]
