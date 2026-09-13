"""Deterministic daily factor engineering for stock analysis.

Price and volume measurements use only data available on or before the signal
close. Economic groups expose a traceable representation. Cross-sectional ranks
use the supplied trade_date and do not create future labels or train models.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import pandas as pd


FACTOR_ENGINEERING_VERSION = "daily-factor-engineering-v2"


# The registry is intentionally data rather than executable logic.  It is
# returned in diagnostics so an analysis can be traced back to its economic
# interpretation and availability requirements.
FACTOR_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "trend_structure",
            (
                "ma_gap_5",
                "ma_gap_10",
                "ma_gap_20",
                "ma_gap_60",
                "ma_trend_5_20",
                "trend_slope_20",
                "trend_fit_quality_20",
                "ma_5_20_gap_change",
            ),
        ),
        (
            "momentum_reversal",
            ("ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "rsi_14", "macd_dif_pct", "macd_hist_pct"),
        ),
        (
            "candle_pressure",
            (
                "gap_open",
                "intraday_return",
                "close_location",
                "body_pct",
                "upper_shadow_pct",
                "lower_shadow_pct",
                "signed_close_pressure",
                "shadow_imbalance",
            ),
        ),
        (
            "price_volume_confirmation",
            (
                "volume_ratio_5_20",
                "amount_ratio_5_20",
                "amount_anomaly_20",
                "signed_amount_shock",
                "return_amount_corr_20",
                "price_turnover_corr_20",
                "wvma_20",
                "volume_price_residual_20",
                "overnight_intraday_corr_20",
            ),
        ),
        (
            "breakout_pullback_quality",
            (
                "breakout_distance_20",
                "breakout_volume_confirmation",
                "pullback_quality_20",
                "stalling_pressure_20",
                "low_volume_long_lower_shadow",
            ),
        ),
        (
            "relative_strength",
            (
                "peer_mean_ret_1",
                "peer_mean_ret_5",
                "peer_mean_ret_20",
                "excess_ret_1",
                "excess_ret_5",
                "excess_ret_20",
                "excess_vs_universe_ret_1",
                "excess_vs_universe_ret_5",
                "excess_vs_universe_ret_20",
                "excess_vs_industry_ret_1",
                "excess_vs_industry_ret_5",
                "excess_vs_industry_ret_20",
                "excess_vs_csi300_ret_1",
                "excess_vs_csi300_ret_5",
                "excess_vs_csi300_ret_20",
                "rank_ret_5",
                "rank_ma_gap_20",
            ),
        ),
        (
            "risk_liquidity",
            (
                "atr_14_pct",
                "volatility_20",
                "amplitude_1",
                "drawdown_20",
                "peer_dispersion_ret_5",
                "log_amount_yuan",
                "rank_volume_ratio_5_20",
                "rank_volatility_20",
                "rank_log_amount",
                "turnover_rate_daily",
                "rank_turnover_rate_daily",
                "log_circ_mv",
                "rank_log_circ_mv",
            ),
        ),
        (
            "market_context",
            (
                "universe_mean_ret_1",
                "universe_mean_ret_5",
                "universe_mean_ret_20",
                "universe_breadth_above_ma20",
                "universe_breadth_positive_5d",
                "universe_dispersion_ret_5",
                "industry_mean_ret_1",
                "industry_mean_ret_5",
                "industry_mean_ret_20",
                "industry_breadth_above_ma20",
                "industry_breadth_positive_5d",
                "industry_dispersion_ret_5",
                "market_reference_mean_ret_1",
                "market_reference_mean_ret_5",
                "market_reference_mean_ret_20",
                "market_regime_weak",
                "market_regime_sideways",
                "market_regime_strong",
                "market_regime_trend_breadth_interaction",
                "market_regime_volatility_stress",
            ),
        ),
    ]
)


RAW_PRICE_VOLUME_FEATURE_COLUMNS: tuple[str, ...] = (
    "gap_open",
    "intraday_return",
    "close_location",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "signed_close_pressure",
    "shadow_imbalance",
    "amount_ratio_5_20",
    "amount_anomaly_20",
    "signed_amount_shock",
    "return_amount_corr_20",
    "price_turnover_corr_20",
    "wvma_20",
    "volume_price_residual_20",
    "overnight_intraday_corr_20",
    "trend_slope_20",
    "trend_fit_quality_20",
    "breakout_distance_20",
    "breakout_volume_confirmation",
    "pullback_quality_20",
    "stalling_pressure_20",
    "ma_5_20_gap_change",
    "low_volume_long_lower_shadow",
)


# 保留同义字段的来源映射，便于解释原值与同日排名。
# 不按相关性删除不同经济含义的因子。
FACTOR_ALIASES: dict[str, str] = {
    "rank_ret_5": "ret_5",
    "rank_ma_gap_20": "ma_gap_20",
    "rank_volume_ratio_5_20": "volume_ratio_5_20",
    "rank_volatility_20": "volatility_20",
    "rank_log_amount": "log_amount_yuan",
    "rank_turnover_rate_daily": "turnover_rate_daily",
    "rank_log_circ_mv": "log_circ_mv",
    "peer_dispersion_ret_5": "universe_dispersion_ret_5",
    **{f"peer_mean_ret_{period}": f"universe_mean_ret_{period}" for period in (1, 5, 20)},
    **{f"excess_ret_{period}": f"excess_vs_universe_ret_{period}" for period in (1, 5, 20)},
}
FACTOR_DIAGNOSTIC_ONLY = frozenset({"wvma_20", "overnight_intraday_corr_20", "market_regime_sideways"})
FACTOR_INPUT_REQUIREMENTS: dict[str, str] = {
    "price_turnover_corr_20": (
        "真实逐日 turnover_rate 百分数（5 表示 5%），先除以 100；"
        "计算日收益与换手率日变化的 20 日相关，至少 10 对有效观测；缺失不使用成交额代理"
    ),
    "wvma_20": "成交额加权日收益均方根，衡量无方向波动，仅供诊断",
}


def independent_factor_features(members: Iterable[str]) -> tuple[str, ...]:
    """返回非别名指标；所有字段仅用于证据展示，不产生分数。"""
    return tuple(
        feature for feature in members
        if feature not in FACTOR_ALIASES and feature not in FACTOR_DIAGNOSTIC_ONLY
    )

def factor_definition(feature: str) -> dict[str, Any]:
    """指标名称和单位的唯一说明，数值仍完全由量化程序计算。"""
    fixed = {
        "ma_trend_5_20": ("MA5相对MA20偏离", "%", 100, "短均线相对中期均线的距离"),
        "ma_5_20_gap_change": ("价格对MA5与MA20偏离差的变化", "百分点", 100, "价格相对MA5偏离减相对MA20偏离，再取日变化"),
        "trend_slope_20": ("20日对数价格日均变化", "%", 100, "20日对数收盘价变化除以20，不是回归预测斜率"),
        "trend_fit_quality_20": ("20日趋势变化与波动比", "倍", 1, "绝对价格变化相对波动的比值，不是回归拟合优度，也不表示上涨方向"),
        "rsi_14": ("14日相对强弱指标RSI", "", 1, "上涨与下跌幅度的相对强弱，范围0至100，不是股票评分"),
        "macd_dif_pct": ("MACD快线相对价格", "%", 100, "DIF除以当前收盘价"),
        "macd_hist_pct": ("MACD柱相对价格", "%", 100, "MACD柱除以当前收盘价"),
        "gap_open": ("开盘跳空", "%", 100, "开盘价相对前收盘价的变化"),
        "intraday_return": ("日内涨跌幅", "%", 100, "收盘价相对开盘价的变化"),
        "close_location": ("收盘在日内区间的位置", "%", 100, "最低价为0%，最高价为100%"),
        "body_pct": ("K线有方向实体幅度", "%", 100, "收盘减开盘再除以前收盘价，保留涨跌方向"),
        "upper_shadow_pct": ("上影线幅度", "%", 100, "最高价超出开收盘较高者的幅度，相对前收盘价"),
        "lower_shadow_pct": ("下影线幅度", "%", 100, "开收盘较低者超出最低价的幅度，相对前收盘价"),
        "signed_close_pressure": ("收盘位置与日内涨跌压力", "", 1, "收盘位置与日内涨跌方向共同描述的价量代理"),
        "shadow_imbalance": ("下影与上影差", "%", 100, "下影线幅度减上影线幅度"),
        "volume_ratio_5_20": ("5日与20日均量比", "倍", 1, "近5日平均成交量除以近20日平均成交量"),
        "amount_ratio_5_20": ("5日与20日均额比", "倍", 1, "近5日平均成交额除以近20日平均成交额"),
        "amount_anomaly_20": ("20日成交额异常程度", "标准差", 1, "成交额对数相对20日均值偏离多少个标准差"),
        "signed_amount_shock": ("带涨跌方向的成交额变化", "", 1, "收益方向与成交额异常的组合观测，不代表账户资金流向"),
        "return_amount_corr_20": ("20日收益与成交额变化相关", "", 1, "相关系数，描述统计关系，不代表因果"),
        "price_turnover_corr_20": ("20日收益与换手变化相关", "", 1, "使用真实历史换手率变化，缺失时不由成交额代替"),
        "wvma_20": ("成交额加权收益波幅", "%", 100, "成交额加权收益均方根，只描述无方向波动"),
        "volume_price_residual_20": ("20日量价残差", "", 1, "收益与成交额变化关系的偏离观测"),
        "overnight_intraday_corr_20": ("20日隔夜与日内收益相关", "", 1, "隔夜跳空与日内收益之间的相关系数"),
        "breakout_distance_20": ("相对前20日高点距离", "%", 100, "当前收盘价相对此前窗口高点的位置"),
        "breakout_volume_confirmation": ("突破距离与量能确认", "", 1, "历史高点距离与量能变化的组合观测"),
        "pullback_quality_20": ("20日回撤量价特征", "", 1, "回撤位置与量能组合，仅作为形态证据"),
        "stalling_pressure_20": ("放量滞涨压力", "", 1, "成交额变化与有限价格推进的组合观测"),
        "low_volume_long_lower_shadow": ("低量长下影特征", "", 1, "低成交量与下影线形态共同出现的程度"),
        "atr_14_pct": ("14日平均真实波幅占价格", "%", 100, "ATR除以收盘价，包含跳空影响"),
        "volatility_20": ("20日年化波动率", "%", 100, "日收益标准差乘以根号252"),
        "amplitude_1": ("当日振幅", "%", 100, "日内最高最低价差相对前收盘价"),
        "drawdown_20": ("距20日高点的当前回撤", "%", 100, "当前价相对20日最高价的变化，不是窗口最大回撤"),
        "peer_dispersion_ret_5": ("比较池5日收益离散度", "%", 100, "本次比较池股票5日收益的标准差"),
        "log_amount_yuan": ("成交额对数", "", 1, "ln(1+成交额元)，用于跨股票比较尺度"),
        "turnover_rate_daily": ("当日换手率", "%", 100, "已归一为小数的真实换手率"),
        "log_circ_mv": ("流通市值对数", "", 1, "ln(流通市值元)，仅使用正数市值"),
        "market_regime_weak": ("市场方向偏弱条件", "", 1, "沪深300近20日收益为负且比较池不到半数位于MA20上方；1成立、0不成立"),
        "market_regime_strong": ("市场方向偏强条件", "", 1, "沪深300近20日收益为正且比较池过半位于MA20上方；1成立、0不成立"),
        "market_regime_sideways": ("市场方向分歧条件", "", 1, "市场收益与比较池广度未同时偏强或偏弱；1成立、0不成立"),
        "market_regime_trend_breadth_interaction": ("市场收益与比较池广度乘积", "", 1, "沪深30020日收益乘以比较池MA20上方比例，无权重合成"),
        "market_regime_volatility_stress": ("沪深30020日年化波动", "%", 100, "直接展示指数实际波动率，不合成风险分"),
    }
    if feature in fixed:
        label, unit, scale, meaning = fixed[feature]
    elif feature.startswith("ma_gap_"):
        period = feature.removeprefix("ma_gap_")
        label, unit, scale, meaning = f"价格相对MA{period}偏离", "%", 100, f"收盘价相对{period}日均线的距离"
    elif feature.startswith("ret_"):
        period = feature.removeprefix("ret_")
        label, unit, scale, meaning = f"近{period}日收益", "%", 100, f"最新收盘价相对{period}个交易日前的变化"
    elif feature.startswith("rank_"):
        originals = {"rank_log_amount": "log_amount_yuan", "rank_log_circ_mv": "log_circ_mv"}
        original = factor_definition(originals.get(feature, feature.removeprefix("rank_")))
        label, unit, scale, meaning = original["label"] + "同日分位", "分位%", 100, "在本次有效比较池中的相对位置，不是评分或上涨概率"
    elif feature.startswith("excess_"):
        parts = feature.rsplit("_ret_", 1)
        prefix = parts[0]
        period = parts[-1] if len(parts) > 1 else feature.rsplit("_", 1)[-1]
        basis = {"excess": "比较池", "excess_vs_universe": "比较池", "excess_vs_industry": "真实同行分组", "excess_vs_csi300": "沪深300"}.get(prefix, "比较池")
        label, unit, scale, meaning = f"近{period}日相对{basis}超额", "百分点", 100, "本股区间收益减去参照收益，不是未来超额预测"
    else:
        prefix = next((key for key in ("market_reference", "universe", "industry", "peer") if feature.startswith(key + "_")), None)
        basis = {"market_reference": "市场参照子池", "universe": "本次比较池", "industry": "真实同行分组", "peer": "本次比较池"}.get(prefix, "")
        suffix = feature[len(prefix) + 1:] if prefix else feature
        if suffix.startswith("mean_ret_"):
            label, unit, scale, meaning = basis + suffix.removeprefix("mean_ret_") + "日平均收益", "%", 100, "同日参照样本的等权平均收益，不是综合打分"
        elif suffix == "dispersion_ret_5":
            label, unit, scale, meaning = basis + "5日收益离散度", "%", 100, "样本收益标准差"
        elif suffix in {"breadth_above_ma20", "breadth_positive_5d"}:
            condition = "位于MA20上方" if suffix == "breadth_above_ma20" else "5日收益为正"
            label, unit, scale, meaning = basis + condition + "比例", "%", 100, "满足条件的有效股票比例，仅代表当前样本"
        else:
            label, unit, scale, meaning = feature, "", 1, "程序已计算的原始观测"
    return {"label": label, "unit": unit, "display_scale": scale, "meaning": meaning}


def _registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, members in FACTOR_GROUPS.items():
        role = "risk" if group == "risk_liquidity" else "context" if group == "market_context" else "alpha"
        for feature in members:
            rows.append(
                {
                    "feature": feature,
                    **factor_definition(feature),
                    "group": group,
                    "role": role,
                    "use": "observed_evidence",
                    "canonical_source": FACTOR_ALIASES.get(feature, feature),
                    "input_requirement": FACTOR_INPUT_REQUIREMENTS.get(feature),
                    "frequency": "daily_k",
                    "availability": "signal_close_or_earlier",
                }
            )
    return rows


FACTOR_REGISTRY: tuple[dict[str, Any], ...] = tuple(_registry_rows())
_FEATURE_TO_GROUP = {
    feature: group for group, members in FACTOR_GROUPS.items() for feature in members
}
# 旧字段实际描述 MA5 与 MA20 相对间距的日变化，并非 MACD 金叉。
# 仅保留查询兼容，不再把旧名放入生产因子列表，避免同一证据重复计权。
_FEATURE_TO_GROUP["golden_cross_speed"] = "trend_structure"


def factor_group(feature: str) -> str:
    """Return the registered economic group, or a stable fallback group."""
    value = str(feature)
    if value in _FEATURE_TO_GROUP:
        return _FEATURE_TO_GROUP[value]
    if value.startswith(("market_", "universe_")):
        return "market_context"
    if value.startswith(("peer_", "industry_", "excess_", "rank_")):
        return "relative_strength"
    return "unregistered"


def factor_role(feature: str) -> str:
    """Return the registry role used by analysis output explainers."""
    group = factor_group(feature)
    if group == "risk_liquidity":
        return "risk"
    if group == "market_context":
        return "context"
    return "alpha"


def factor_registry_rows() -> list[dict[str, Any]]:
    """Return JSON-safe registry rows for run diagnostics and artifacts."""
    return [dict(row) for row in FACTOR_REGISTRY]


def _numeric(data: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _group_indices(data: pd.DataFrame) -> Iterable[pd.Index]:
    if "ts_code" in data.columns and data["ts_code"].nunique(dropna=True) > 1:
        yield from (group.index for _, group in data.groupby("ts_code", sort=False))
    else:
        yield data.index


def add_price_volume_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add continuous price/volume and objective candle-shape factors.

    Every rolling operation is performed independently per security.  The
    implementation uses small fixed windows so it cannot become a parameter
    sweep disguised as feature generation.
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    data = frame.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data = data.sort_values([column for column in ["ts_code", "trade_date"] if column in data.columns]).copy()
    close = _numeric(data, "close")
    open_price = _numeric(data, "open")
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    volume = _numeric(data, "volume")
    amount = _numeric(data, "amount_yuan")
    # Some providers omit amount.  Volume*close is a transparent proxy, not a
    # fabricated valuation field, and is marked by the same daily availability.
    amount = amount.where(amount.notna(), (volume * close).where(volume.notna() & close.notna()))

    # These fields expose the measured trend structure for technical diagnosis,
    # so calculate them even when an older caller did not request the new set.
    groups = list(_group_indices(data))
    for indices in groups:
        local = data.loc[indices].sort_values("trade_date")
        c = _numeric(local, "close")
        o = _numeric(local, "open")
        h = _numeric(local, "high")
        l = _numeric(local, "low")
        v = _numeric(local, "volume")
        a = _numeric(local, "amount_yuan")
        a = a.where(a.notna(), (v * c).where(v.notna() & c.notna()))
        prev = c.shift(1)
        ret1 = c.pct_change(fill_method=None)
        gap = _safe_div(o, prev) - 1.0
        intraday = _safe_div(c, o) - 1.0
        daily_range = (h - l).replace(0, np.nan)
        location = _safe_div(c - l, daily_range).clip(0.0, 1.0)
        body = _safe_div(c - o, prev)
        upper = _safe_div(h - pd.concat([o, c], axis=1).max(axis=1), prev)
        lower = _safe_div(pd.concat([o, c], axis=1).min(axis=1) - l, prev)
        amount_log = np.log1p(a.clip(lower=0))
        amount_change = amount_log.diff()
        mean5 = a.rolling(5, min_periods=5).mean()
        mean20 = a.rolling(20, min_periods=20).mean()
        ratio = _safe_div(mean5, mean20)
        rolling_mean = amount_log.rolling(20, min_periods=10).mean()
        rolling_std = amount_log.rolling(20, min_periods=10).std().replace(0, np.nan)
        anomaly = _safe_div(amount_log - rolling_mean, rolling_std)
        ret_amount_corr = ret1.rolling(20, min_periods=10).corr(amount_change)
        # 历史 turnover_rate 使用百分数（5 表示 5%），不读取当前快照回填。
        # 成交额变化不等于换手率；缺失、常量或有效配对不足时相关性留空。
        turnover = _numeric(local, "turnover_rate") / 100.0
        turnover = turnover.where(turnover.ge(0.0))
        price_turnover_corr = ret1.rolling(20, min_periods=10).corr(turnover.diff())
        weight = _safe_div(a, a.rolling(20, min_periods=10).median()).clip(lower=0.25, upper=4.0)
        wvma = np.sqrt(_safe_div((ret1.pow(2) * weight).rolling(20, min_periods=10).sum(), weight.rolling(20, min_periods=10).sum()))
        cov = ret1.rolling(20, min_periods=10).cov(amount_change)
        var = amount_change.rolling(20, min_periods=10).var().replace(0, np.nan)
        beta = _safe_div(cov, var)
        residual = ret1 - beta * amount_change
        overnight_corr = gap.rolling(20, min_periods=10).corr(intraday)
        log_close = np.log(c.where(c > 0))
        slope = log_close.diff(20) / 20.0
        volatility = ret1.rolling(20, min_periods=10).std()
        fit_quality = _safe_div(log_close.diff(20).abs(), volatility * math.sqrt(20.0))
        prior_high = h.rolling(20, min_periods=10).max().shift(1)
        breakout = _safe_div(c, prior_high) - 1.0
        breakout_confirmation = breakout.clip(lower=0.0) * (ratio - 1.0)
        ma_gap20 = _numeric(local, "ma_gap_20")
        position20 = _numeric(local, "position_20", 0.5).fillna(0.5)
        ma_trend = _numeric(local, "ma_trend_5_20")
        pullback = (-c.pct_change(5, fill_method=None)).clip(lower=0.0) * (1.0 - ratio).clip(lower=0.0) * ma_trend.clip(lower=0.0)
        stalling = position20.clip(lower=0.0, upper=1.0) * anomaly.clip(lower=0.0) * (1.0 - location.fillna(0.5))
        ma_gap_change = (_numeric(local, "ma_gap_5") - ma_gap20).diff()
        low_volume_shadow = (1.0 - position20).clip(lower=0.0, upper=1.0) * lower.clip(lower=0.0) / (1.0 + ratio.clip(lower=0.0))
        values = {
            "gap_open": gap,
            "intraday_return": intraday,
            "close_location": location,
            "body_pct": body,
            "upper_shadow_pct": upper,
            "lower_shadow_pct": lower,
            "signed_close_pressure": location.fillna(0.5).sub(0.5) * body.abs().fillna(0.0) * np.sign(body.fillna(0.0)),
            "shadow_imbalance": lower - upper,
            "amount_ratio_5_20": ratio,
            "amount_anomaly_20": anomaly,
            "signed_amount_shock": np.sign(ret1.fillna(0.0)) * anomaly,
            "return_amount_corr_20": ret_amount_corr,
            "price_turnover_corr_20": price_turnover_corr,
            "wvma_20": wvma,
            "volume_price_residual_20": residual,
            "overnight_intraday_corr_20": overnight_corr,
            "trend_slope_20": slope,
            "trend_fit_quality_20": fit_quality,
            "breakout_distance_20": breakout,
            "breakout_volume_confirmation": breakout_confirmation,
            "pullback_quality_20": pullback,
            "stalling_pressure_20": stalling,
            "ma_5_20_gap_change": ma_gap_change,
            # 兼容仍读取旧列名的调用方；因子注册表只登记上面的准确名称。
            "golden_cross_speed": ma_gap_change,
            "low_volume_long_lower_shadow": low_volume_shadow,
        }
        for column, series in values.items():
            data.loc[local.index, column] = pd.to_numeric(series, errors="coerce").to_numpy()
    return data.replace([np.inf, -np.inf], np.nan)


def describe_factor_coverage(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """保留原始因子并记录覆盖，不生成组均分或加权合成列。"""
    if frame is None or frame.empty:
        return (pd.DataFrame() if frame is None else frame.copy(), {"status": "unavailable"})
    data = frame.replace([np.inf, -np.inf], np.nan).copy()
    groups = {
        group: {"available_components": [column for column in members if column in data and data[column].notna().any()], "expected_components": list(members)}
        for group, members in FACTOR_GROUPS.items()
    }
    return data, {"status": "ok", "version": FACTOR_ENGINEERING_VERSION, "groups": groups, "daily_k_only": True, "method": "raw_evidence_only"}


def engineered_factor_columns() -> list[str]:
    """按注册顺序返回原始因子字段。"""
    return list(dict.fromkeys(column for members in FACTOR_GROUPS.values() for column in members))


__all__ = [
    "FACTOR_ENGINEERING_VERSION",
    "FACTOR_GROUPS",
    "FACTOR_ALIASES",
    "FACTOR_DIAGNOSTIC_ONLY",
    "FACTOR_INPUT_REQUIREMENTS",
    "independent_factor_features",
    "FACTOR_REGISTRY",
    "RAW_PRICE_VOLUME_FEATURE_COLUMNS",
    "describe_factor_coverage",
    "add_price_volume_factors",
    "engineered_factor_columns",
    "factor_group",
    "factor_role",
    "factor_registry_rows",
]
