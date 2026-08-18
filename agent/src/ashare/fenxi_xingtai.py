"""涨停回马枪日 K 形态状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class HuimaqiangZhuangtai(str, Enum):
    WU_XINGTAI = "none"
    TIAOZHENGZHONG = "adjusting"
    JIZHI_SUOLIANG = "extreme_shrink"
    DENGDAI_TUPO = "waiting_breakout"
    PANZHONG_QUEREN = "intraday_confirmed"
    SHOUPAN_QUEREN = "close_confirmed"
    SHIXIAO = "invalidated"
    GUOQI = "expired"


_STATE_LABELS = {
    HuimaqiangZhuangtai.WU_XINGTAI: "无形态",
    HuimaqiangZhuangtai.TIAOZHENGZHONG: "调整观察中",
    HuimaqiangZhuangtai.JIZHI_SUOLIANG: "已出现极致缩量",
    HuimaqiangZhuangtai.DENGDAI_TUPO: "等待放量突破",
    HuimaqiangZhuangtai.PANZHONG_QUEREN: "盘中暂定确认",
    HuimaqiangZhuangtai.SHOUPAN_QUEREN: "收盘正式确认",
    HuimaqiangZhuangtai.SHIXIAO: "形态失效",
    HuimaqiangZhuangtai.GUOQI: "超过观察期",
}


@dataclass(frozen=True)
class HuimaqiangJieguo:
    state: HuimaqiangZhuangtai
    eligible: bool
    score: float | None
    baseline_date: str | None
    shrink_date: str | None
    breakout_date: str | None
    risk_reference_price: float | None
    actuals: dict[str, Any]
    conditions: dict[str, bool | None]
    evidence: tuple[str, ...]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.eligible else "not_applicable",
            "state": self.state.value,
            "state_label": _STATE_LABELS[self.state],
            "eligible": self.eligible,
            "score_0_100": self.score,
            "baseline_date": self.baseline_date,
            "shrink_date": self.shrink_date,
            "breakout_date": self.breakout_date,
            "confirmation_level": (
                "intraday_provisional"
                if self.state is HuimaqiangZhuangtai.PANZHONG_QUEREN
                else "completed_daily_close"
                if self.state is HuimaqiangZhuangtai.SHOUPAN_QUEREN
                else "not_confirmed"
            ),
            "risk_reference_price": self.risk_reference_price,
            "actuals": self.actuals,
            "conditions": self.conditions,
            "evidence": list(self.evidence),
            "failure_reasons": list(self.failure_reasons),
            "score_definition": "形态阶段强度分，只用于综合排序增强，不是上涨概率",
        }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _zhangtingjia(previous_close: float) -> float:
    return float((Decimal(str(previous_close)) * Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _shi_shizhupan(code: str) -> bool:
    digits = str(code).split(".")[0].zfill(6)
    return digits.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _hefa_biaodi(code: str, name: str) -> tuple[bool, str | None]:
    upper_name = str(name).strip().upper()
    if not _shi_shizhupan(code):
        return False, "初版只识别正常 10% 涨跌幅的沪深主板股票"
    if "ST" in upper_name or "退" in upper_name:
        return False, "ST 或退市风险股票不适用初版正常 10% 规则"
    if upper_name.startswith(("N", "C")):
        return False, "新股无涨跌幅限制阶段不适用该形态"
    return True, None


def _zhengli_lishi(history: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "open", "high", "low", "close", "volume"}
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    data = history.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "pre_close", "volume"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["trade_date", "open", "high", "low", "close", "volume"])
    data = data.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    if "pre_close" not in data.columns:
        data["pre_close"] = data["close"].shift(1)
    else:
        data["pre_close"] = data["pre_close"].fillna(data["close"].shift(1))
    return data


def _zhuangtai_fenshu(state: HuimaqiangZhuangtai, config: dict[str, Any]) -> float | None:
    if state is HuimaqiangZhuangtai.WU_XINGTAI:
        return None
    value = (config.get("state_scores") or {}).get(state.value)
    number = _number(value)
    return round(number, 2) if number is not None else None


def _fenxi_jizhunri(
    data: pd.DataFrame,
    *,
    baseline_index: int,
    realtime_quote: dict[str, Any] | None,
    config: dict[str, Any],
) -> HuimaqiangJieguo:
    baseline = data.iloc[baseline_index]
    baseline_date = pd.Timestamp(baseline["trade_date"]).strftime("%Y-%m-%d")
    baseline_open = float(baseline["open"])
    baseline_high = float(baseline["high"])
    baseline_volume = float(baseline["volume"])
    shrink_window = int(config.get("shrink_window_sessions", 7))
    breakout_deadline = int(config.get("breakout_deadline_sessions", 14))
    shrink_ratio_max = float(config.get("shrink_volume_ratio_max", 0.5))
    breakout_volume_ratio_min = float(config.get("breakout_volume_median_ratio_min", 1.2))
    observations = data.iloc[baseline_index + 1 : baseline_index + 1 + breakout_deadline].copy()
    observed_sessions = int(len(observations))
    failure_reasons: list[str] = []
    evidence = [
        f"基准涨停阳线为 {baseline_date}，开盘 {baseline_open:.2f}、最高 {baseline_high:.2f}、成交量 {baseline_volume:.0f} 股"
    ]
    conditions: dict[str, bool | None] = {
        "normal_10pct_main_board": True,
        "baseline_closed_at_limit_up": True,
        "baseline_bullish_candle": True,
        "all_adjustment_closes_above_baseline_open": None,
        "extreme_shrink_within_d7": None,
        "breakout_after_shrink_by_d14": None,
        "breakout_bullish_candle": None,
        "breakout_close_above_baseline_high": None,
        "breakout_volume_sufficient": None,
    }
    actuals: dict[str, Any] = {
        "baseline_open": round(baseline_open, 3),
        "baseline_high": round(baseline_high, 3),
        "baseline_close": round(float(baseline["close"]), 3),
        "baseline_volume": round(baseline_volume, 2),
        "observed_sessions_after_baseline": observed_sessions,
        "shrink_volume_ratio_threshold": shrink_ratio_max,
        "breakout_volume_median_ratio_threshold": breakout_volume_ratio_min,
        "breakout_deadline_sessions": breakout_deadline,
    }

    below_open = observations[pd.to_numeric(observations["close"], errors="coerce") < baseline_open]
    if not below_open.empty:
        failed = below_open.iloc[0]
        failed_date = pd.Timestamp(failed["trade_date"]).strftime("%Y-%m-%d")
        conditions["all_adjustment_closes_above_baseline_open"] = False
        failure_reasons.append(
            f"{failed_date} 收盘 {float(failed['close']):.2f} 跌破基准阳线开盘 {baseline_open:.2f}"
        )
        state = HuimaqiangZhuangtai.SHIXIAO
        return HuimaqiangJieguo(
            state=state,
            eligible=True,
            score=_zhuangtai_fenshu(state, config),
            baseline_date=baseline_date,
            shrink_date=None,
            breakout_date=None,
            risk_reference_price=round(baseline_open, 3),
            actuals=actuals,
            conditions=conditions,
            evidence=tuple(evidence),
            failure_reasons=tuple(failure_reasons),
        )
    conditions["all_adjustment_closes_above_baseline_open"] = True

    d1_d7 = observations.head(shrink_window)
    shrink_rows = d1_d7[
        pd.to_numeric(d1_d7["volume"], errors="coerce") <= baseline_volume * shrink_ratio_max
    ]
    shrink_row = shrink_rows.iloc[0] if not shrink_rows.empty else None
    shrink_date = (
        pd.Timestamp(shrink_row["trade_date"]).strftime("%Y-%m-%d")
        if shrink_row is not None
        else None
    )
    conditions["extreme_shrink_within_d7"] = shrink_row is not None
    if shrink_row is not None:
        shrink_ratio = float(shrink_row["volume"]) / baseline_volume if baseline_volume > 0 else np.nan
        actuals["shrink_volume_ratio"] = round(float(shrink_ratio), 4) if np.isfinite(shrink_ratio) else None
        evidence.append(
            f"{shrink_date} 成交量为涨停日的 {float(shrink_ratio) * 100:.1f}%，满足不高于 {shrink_ratio_max * 100:.1f}% 的极致缩量"
        )

    breakout_row: pd.Series | None = None
    breakout_volume_ratio: float | None = None
    if shrink_row is not None:
        shrink_position = int(observations.index.get_loc(shrink_row.name))
        after_shrink = observations.iloc[shrink_position + 1 :]
        for position, (_, candidate) in enumerate(after_shrink.iterrows(), start=shrink_position + 1):
            previous_adjustment = observations.iloc[:position]
            median_volume = _number(pd.to_numeric(previous_adjustment["volume"], errors="coerce").median())
            candidate_volume = _number(candidate.get("volume"))
            price_ok = float(candidate["close"]) > float(candidate["open"]) and float(candidate["close"]) > baseline_high
            volume_ok = bool(
                median_volume is not None
                and median_volume > 0
                and candidate_volume is not None
                and candidate_volume >= median_volume * breakout_volume_ratio_min
            )
            if price_ok and volume_ok:
                breakout_row = candidate
                breakout_volume_ratio = candidate_volume / median_volume
                break
    if breakout_row is not None:
        breakout_date = pd.Timestamp(breakout_row["trade_date"]).strftime("%Y-%m-%d")
        conditions.update(
            {
                "breakout_after_shrink_by_d14": True,
                "breakout_bullish_candle": True,
                "breakout_close_above_baseline_high": True,
                "breakout_volume_sufficient": True,
            }
        )
        actuals.update(
            {
                "breakout_open": round(float(breakout_row["open"]), 3),
                "breakout_close": round(float(breakout_row["close"]), 3),
                "breakout_volume_median_ratio": round(float(breakout_volume_ratio), 4),
            }
        )
        evidence.append(
            f"{breakout_date} 收阳并收于基准最高价上方，成交量为调整阶段中位数的 {float(breakout_volume_ratio):.2f} 倍"
        )
        latest_complete_date = pd.Timestamp(data.iloc[-1]["trade_date"]).strftime("%Y-%m-%d")
        if breakout_date != latest_complete_date:
            state = HuimaqiangZhuangtai.GUOQI
            failure_reasons.append(f"突破确认发生在 {breakout_date}，已不是最新完整日线信号")
            return HuimaqiangJieguo(
                state=state,
                eligible=True,
                score=_zhuangtai_fenshu(state, config),
                baseline_date=baseline_date,
                shrink_date=shrink_date,
                breakout_date=breakout_date,
                risk_reference_price=round(max(baseline_open, float(breakout_row["open"])), 3),
                actuals=actuals,
                conditions=conditions,
                evidence=tuple(evidence),
                failure_reasons=tuple(failure_reasons),
            )
        state = HuimaqiangZhuangtai.SHOUPAN_QUEREN
        return HuimaqiangJieguo(
            state=state,
            eligible=True,
            score=_zhuangtai_fenshu(state, config),
            baseline_date=baseline_date,
            shrink_date=shrink_date,
            breakout_date=breakout_date,
            risk_reference_price=round(max(baseline_open, float(breakout_row["open"])), 3),
            actuals=actuals,
            conditions=conditions,
            evidence=tuple(evidence),
            failure_reasons=(),
        )

    quote = realtime_quote or {}
    if shrink_row is not None and quote.get("status") == "ok" and observed_sessions < breakout_deadline:
        current_open = _number(quote.get("open"))
        current_price = _number(quote.get("last_price") if quote.get("last_price") is not None else quote.get("latest_price"))
        current_volume = _number(quote.get("volume"))
        median_volume = _number(pd.to_numeric(observations["volume"], errors="coerce").median())
        price_ok = current_open is not None and current_price is not None and current_price > current_open and current_price > baseline_high
        volume_ok = bool(
            median_volume is not None
            and median_volume > 0
            and current_volume is not None
            and current_volume >= median_volume * breakout_volume_ratio_min
        )
        conditions["breakout_bullish_candle"] = price_ok
        conditions["breakout_close_above_baseline_high"] = current_price is not None and current_price > baseline_high
        conditions["breakout_volume_sufficient"] = volume_ok
        if price_ok and volume_ok:
            conditions["breakout_after_shrink_by_d14"] = True
            actuals.update(
                {
                    "intraday_breakout_open": round(float(current_open), 3),
                    "intraday_breakout_price": round(float(current_price), 3),
                    "intraday_breakout_volume_median_ratio": round(float(current_volume / median_volume), 4),
                }
            )
            evidence.append("实时价量满足突破条件，但尚未形成完整收盘日线")
            state = HuimaqiangZhuangtai.PANZHONG_QUEREN
            return HuimaqiangJieguo(
                state=state,
                eligible=True,
                score=_zhuangtai_fenshu(state, config),
                baseline_date=baseline_date,
                shrink_date=shrink_date,
                breakout_date=None,
                risk_reference_price=round(max(baseline_open, float(current_open)), 3),
                actuals=actuals,
                conditions=conditions,
                evidence=tuple(evidence),
                failure_reasons=(),
            )

    if observed_sessions >= breakout_deadline:
        state = HuimaqiangZhuangtai.GUOQI
        conditions["breakout_after_shrink_by_d14"] = False
        failure_reasons.append(f"D{breakout_deadline} 后仍未形成放量突破")
    elif shrink_row is None and observed_sessions >= shrink_window:
        state = HuimaqiangZhuangtai.SHIXIAO
        failure_reasons.append(f"D1 至 D{shrink_window} 未出现成交量不高于涨停日 {shrink_ratio_max * 100:.1f}% 的极致缩量")
    elif shrink_row is None:
        state = HuimaqiangZhuangtai.TIAOZHENGZHONG
    else:
        latest_date = pd.Timestamp(observations.iloc[-1]["trade_date"]).strftime("%Y-%m-%d") if not observations.empty else None
        state = HuimaqiangZhuangtai.JIZHI_SUOLIANG if latest_date == shrink_date else HuimaqiangZhuangtai.DENGDAI_TUPO
    return HuimaqiangJieguo(
        state=state,
        eligible=True,
        score=_zhuangtai_fenshu(state, config),
        baseline_date=baseline_date,
        shrink_date=shrink_date,
        breakout_date=None,
        risk_reference_price=round(baseline_open, 3),
        actuals=actuals,
        conditions=conditions,
        evidence=tuple(evidence),
        failure_reasons=tuple(failure_reasons),
    )


def fenxi_zhangting_huimaqiang(
    history: pd.DataFrame,
    *,
    code: str,
    name: str,
    config: dict[str, Any],
    realtime_quote: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """识别最近一次回马枪尝试，并公开每个规则的实际命中值。"""
    eligible, reason = _hefa_biaodi(code, name)
    if not eligible:
        return HuimaqiangJieguo(
            state=HuimaqiangZhuangtai.WU_XINGTAI,
            eligible=False,
            score=None,
            baseline_date=None,
            shrink_date=None,
            breakout_date=None,
            risk_reference_price=None,
            actuals={},
            conditions={"normal_10pct_main_board": False},
            evidence=(),
            failure_reasons=(str(reason),),
        ).to_dict()
    data = _zhengli_lishi(history)
    if len(data) < 2:
        return HuimaqiangJieguo(
            state=HuimaqiangZhuangtai.WU_XINGTAI,
            eligible=True,
            score=None,
            baseline_date=None,
            shrink_date=None,
            breakout_date=None,
            risk_reference_price=None,
            actuals={"history_rows": int(len(data))},
            conditions={"normal_10pct_main_board": True},
            evidence=(),
            failure_reasons=("完整日线不足，无法识别形态",),
        ).to_dict()
    tolerance = float(config.get("limit_up_tolerance_yuan", 0.005))
    baselines: list[int] = []
    for index, row in data.iterrows():
        previous_close = _number(row.get("pre_close"))
        if previous_close is None or previous_close <= 0:
            continue
        legal_limit = _zhangtingjia(previous_close)
        if float(row["close"]) >= legal_limit - tolerance and float(row["close"]) > float(row["open"]):
            baselines.append(int(index))
    if not baselines:
        return HuimaqiangJieguo(
            state=HuimaqiangZhuangtai.WU_XINGTAI,
            eligible=True,
            score=None,
            baseline_date=None,
            shrink_date=None,
            breakout_date=None,
            risk_reference_price=None,
            actuals={"history_rows": int(len(data)), "limit_up_tolerance_yuan": tolerance},
            conditions={
                "normal_10pct_main_board": True,
                "baseline_closed_at_limit_up": False,
                "baseline_bullish_candle": None,
            },
            evidence=(),
            failure_reasons=("观察区间内没有合法涨停阳线基准日",),
        ).to_dict()
    results = [
        _fenxi_jizhunri(
            data,
            baseline_index=index,
            realtime_quote=realtime_quote,
            config=config,
        )
        for index in reversed(baselines)
    ]
    priority = {
        HuimaqiangZhuangtai.SHOUPAN_QUEREN: 7,
        HuimaqiangZhuangtai.PANZHONG_QUEREN: 6,
        HuimaqiangZhuangtai.DENGDAI_TUPO: 5,
        HuimaqiangZhuangtai.JIZHI_SUOLIANG: 4,
        HuimaqiangZhuangtai.TIAOZHENGZHONG: 3,
        HuimaqiangZhuangtai.SHIXIAO: 2,
        HuimaqiangZhuangtai.GUOQI: 1,
        HuimaqiangZhuangtai.WU_XINGTAI: 0,
    }
    active = next(
        (
            result
            for result in results
            if result.state
            in {
                HuimaqiangZhuangtai.SHOUPAN_QUEREN,
                HuimaqiangZhuangtai.PANZHONG_QUEREN,
                HuimaqiangZhuangtai.DENGDAI_TUPO,
                HuimaqiangZhuangtai.JIZHI_SUOLIANG,
                HuimaqiangZhuangtai.TIAOZHENGZHONG,
            }
        ),
        None,
    )
    selected = active or max(results, key=lambda result: priority[result.state])
    return selected.to_dict()


__all__ = ["HuimaqiangJieguo", "HuimaqiangZhuangtai", "fenxi_zhangting_huimaqiang"]
