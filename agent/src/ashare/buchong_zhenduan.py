"""共用的补充诊断：只解释已取得的证据，不另造评分或推荐门槛。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .caiwu_zhenduan import goujian_caiwu_zhenduan
from .yinzi_gongcheng import daily_atr


def _block(key: str, label: str, *, metrics: dict[str, Any], summaries: list[str], missing: list[str]) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": "partial" if summaries and missing else "ok" if summaries else "unavailable",
        "summary": "；".join(summaries) if summaries else f"{label}暂无法计算",
        "metrics": metrics,
        "missing_reason": "；".join(missing) or None,
    }


def _number_column(data: pd.DataFrame, name: str) -> pd.Series:
    values = data[name] if name in data else pd.Series(np.nan, index=data.index)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _complete(values: pd.Series, count: int, *, positive: bool = False) -> bool:
    tail = values.tail(count)
    return bool(len(tail) == count and tail.notna().all() and (not positive or tail.gt(0).all()))


def _technical_blocks(data: pd.DataFrame, minimum_amount: float) -> list[dict[str, Any]]:
    close = _number_column(data, "close").where(lambda values: values > 0)
    high = _number_column(data, "high").where(lambda values: values > 0)
    low = _number_column(data, "low").where(lambda values: values > 0)
    volume = _number_column(data, "volume").where(lambda values: values >= 0)
    amount = _number_column(data, "amount_yuan").where(lambda values: values >= 0)
    returns = close.pct_change(fill_method=None)
    blocks: list[dict[str, Any]] = []

    metrics: dict[str, Any] = {"downside_window": 20, "drawdown_window": 60, "annualization_days": 252,
                               "downside_deviation_20": None, "max_drawdown_60": None}
    summaries: list[str] = []
    missing: list[str] = []
    if _complete(returns, 20):
        # 以零收益为下行目标，分母包括窗口内所有日收益，不能只统计下跌日。
        downside = float(np.sqrt(np.mean(np.minimum(returns.tail(20), 0.0) ** 2) * 252))
        metrics["downside_deviation_20"] = round(downside, 8)
        summaries.append(f"近20个交易日下行波动率（年化）{downside:.2%}")
    else:
        missing.append("下行波动率需要连续21条有效收盘价，不跨缺失日计算收益")
    if _complete(close, 60, positive=True):
        window = close.tail(60)
        drawdown = max(0.0, float(-(window / window.cummax() - 1.0).min()))
        metrics["max_drawdown_60"] = round(drawdown, 8)
        summaries.append(f"近60个交易日收盘价最大回撤{drawdown:.2%}（窗口内先高后低的最大跌幅）")
    else:
        missing.append("60日最大回撤需要连续60条有效收盘价")
    blocks.append(_block("downside_risk", "下行风险", metrics=metrics, summaries=summaries, missing=missing))

    metrics = {"ma_window": 20, "atr_window": 14, "distance_atr": None, "ma20": None, "atr14": None,
               "atr_method": "ewm_alpha_1_over_14_adjust_false_same_as_technical_summary"}
    summaries, missing = [], []
    valid_bar = high.ge(low) & close.between(low, high) & low.gt(0)
    if _complete(close, 20, positive=True) and bool(valid_bar.tail(20).all()) and len(valid_bar) >= 20:
        # 与既有技术摘要的 ATR 完全相同，缺失日不被填平；只沿用最后一段有效日线。
        bad_positions = np.flatnonzero(~valid_bar.to_numpy())
        start = int(bad_positions[-1] + 1) if len(bad_positions) else 0
        atr = float(daily_atr(high, low, close).iloc[-1])
        metrics["atr_valid_history_rows"] = len(close) - start
        ma20 = float(close.tail(20).mean())
        metrics.update(ma20=round(ma20, 8), atr14=round(atr, 8))
        if np.isfinite(atr) and atr > 0:
            distance = float((close.iloc[-1] - ma20) / atr)
            metrics["distance_atr"] = round(distance, 6)
            direction = "高于" if distance > 0 else "低于" if distance < 0 else "位于"
            summaries.append(f"收盘价{direction}20日均线，偏离{abs(distance):.2f}个14日平均真实波幅；不单独据此判断反转")
        else:
            missing.append("14日平均真实波幅为零或不可用，不能把均线偏离除以零")
    else:
        missing.append("均线偏离需要最近20条完整且价格范围一致的日线")
    blocks.append(_block("atr_distance", "均线偏离尺度", metrics=metrics, summaries=summaries, missing=missing))

    metrics = {"window": 20, "cmf_20": None, "flat_range_days": None,
               "definition": "sum(((2*close-high-low)/(high-low))*volume)/sum(volume)",
               "interpretation": "收盘位置加权成交量，不是真实资金净流入"}
    summaries, missing = [], []
    if len(data) >= 20 and bool(valid_bar.tail(20).all()) and _complete(volume, 20):
        spread = (high - low).tail(20)
        flat_count = int(spread.eq(0).sum())
        metrics["flat_range_days"] = flat_count
        total_volume = float(volume.tail(20).sum())
        if total_volume > 0 and flat_count < 20:
            pressure = ((2 * close.tail(20) - high.tail(20) - low.tail(20)) / spread.where(spread > 0)).fillna(0.0)
            cmf = float((pressure * volume.tail(20)).sum() / total_volume)
            metrics["cmf_20"] = round(cmf, 6)
            direction = "偏向日内高位" if cmf > 0 else "偏向日内低位" if cmf < 0 else "高低位置压力相抵"
            summaries.append(f"20日收盘位置加权量能（CMF）{cmf:.3f}，{direction}；不代表真实主力资金净流入")
            if flat_count:
                missing.append(f"窗口内{flat_count}日最高价等于最低价，无法辨认日内压力，按零贡献处理；不能据此认定这些日子没有买卖压力")
        else:
            missing.append("窗口内总成交量为零或全部日线没有高低价差，不能辨认价量压力")
    else:
        missing.append("价量压力需要最近20条完整高低收盘价和成交量，不用成交额代理成交量")
    blocks.append(_block("volume_pressure", "持续价量压力", metrics=metrics, summaries=summaries, missing=missing))

    metrics = {"window": 20, "amount_median_20_yuan": None, "below_minimum_days": None,
               "minimum_amount_yuan": minimum_amount, "amount_basis": "observed_daily_amount_yuan"}
    summaries, missing = [], []
    if _complete(amount, 20) and np.isfinite(minimum_amount) and minimum_amount > 0:
        median = float(amount.tail(20).median())
        below = int(amount.tail(20).lt(minimum_amount).sum())
        metrics.update(amount_median_20_yuan=round(median, 2), below_minimum_days=below)
        summaries.append(f"近20个交易日成交额中位数{median / 10_000:.1f}万元，其中{below}日低于现有流动性底线{minimum_amount / 10_000:.1f}万元；仅说明历史成交稳定性")
    else:
        missing.append("流动性稳定性需要20日真实成交额和有效流动性底线，不用收盘价乘成交量填补")
    blocks.append(_block("liquidity_stability", "流动性稳定性", metrics=metrics, summaries=summaries, missing=missing))
    return blocks


def goujian_buchong_zhenduan(
    history: pd.DataFrame,
    fundamental: dict[str, Any],
    *,
    as_of_date: str,
    minimum_amount_yuan: float,
) -> dict[str, Any]:
    """同一输入用于单股及范围候选；只采用分析日及之前的原始日线。"""
    cutoff = pd.to_datetime(str(as_of_date), errors="coerce")
    data = history.copy()
    reason = None
    if pd.isna(cutoff):
        reason = "分析截止日期无效，不能确认数据时点"
    elif "trade_date" not in data:
        reason = "日线缺少交易日期，不能确认数据时点"
    else:
        dates = pd.to_datetime(data["trade_date"].astype("string"), errors="coerce", format="mixed")
        if dates.isna().any():
            reason = "日线存在无效交易日期，不能用跳过坏行的方式拼接观察窗口"
        else:
            data["trade_date"] = dates.dt.normalize()
            data = data.loc[data["trade_date"] <= cutoff.normalize()].sort_values("trade_date").reset_index(drop=True)
            if data["trade_date"].duplicated().any():
                reason = "同一交易日存在重复日线，不能任意选择记录计算"
    technical = _technical_blocks(data.iloc[:0] if reason else data, minimum_amount_yuan)
    if reason:
        for block in technical:
            block["missing_reason"] = reason
    blocks = [*technical, *goujian_caiwu_zhenduan(fundamental, as_of_date=as_of_date)]
    statuses = [block["status"] for block in blocks]
    return {
        "status": "ok" if all(value == "ok" for value in statuses) else "unavailable" if all(value == "unavailable" for value in statuses) else "partial",
        "data_as_of": None if pd.isna(cutoff) else cutoff.strftime("%Y-%m-%d"),
        "daily_data_as_of": None if reason or data.empty else data.iloc[-1]["trade_date"].strftime("%Y-%m-%d"),
        "blocks": blocks,
    }
