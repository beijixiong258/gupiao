"""分析侧复用的八组日 K 因子面板与确定性证据汇总。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ashare.gupiao_yanjiu import jisuan_tezheng_biao
from src.ashare.riping_yinzi import enrich_daily_factor_panel
from src.ashare.yinzi_gongcheng import FACTOR_GROUPS


YINZI_ZU_MINGCHENG: dict[str, tuple[str, str]] = {
    "trend_structure": ("趋势结构", "均线结构、趋势斜率和趋势拟合质量"),
    "momentum_reversal": ("动量与反转", "多周期收益、RSI 与 MACD"),
    "candle_pressure": ("K线压力", "跳空、实体、收盘位置与影线压力"),
    "price_volume_confirmation": ("价量确认", "成交量、成交额异常和价量协同"),
    "breakout_pullback_quality": ("突破与回撤质量", "突破距离、放量确认和回撤质量"),
    "relative_strength": ("相对强弱", "相对同行、行业与市场的超额表现"),
    "risk_liquidity": ("风险与流动性", "波动、回撤、成交额、换手和市值流动性"),
    "market_context": ("市场背景", "全市场与行业的收益、广度、离散度和状态"),
}


# 1 表示同日横截面位置越高越有利，-1 表示越低越有利，0 表示只展示不定向。
# 映射集中维护，避免把不同经济含义的原始分位数直接平均。
YINZI_FANGXIANG: dict[str, int] = {
    "ma_gap_5": 1,
    "ma_gap_10": 1,
    "ma_gap_20": 1,
    "ma_gap_60": 1,
    "ma_trend_5_20": 1,
    "trend_slope_20": 1,
    "trend_fit_quality_20": 1,
    "golden_cross_speed": 1,
    "ret_1": 1,
    "ret_3": 1,
    "ret_5": 1,
    "ret_10": 1,
    "ret_20": 1,
    "rsi_14": 1,
    "macd_dif_pct": 1,
    "macd_hist_pct": 1,
    "gap_open": 1,
    "intraday_return": 1,
    "close_location": 1,
    "body_pct": 1,
    "upper_shadow_pct": -1,
    "lower_shadow_pct": 1,
    "signed_close_pressure": 1,
    "shadow_imbalance": 1,
    "volume_ratio_5_20": 1,
    "amount_ratio_5_20": 1,
    "amount_anomaly_20": 1,
    "signed_amount_shock": 1,
    "return_amount_corr_20": 1,
    "price_turnover_corr_20": 1,
    "wvma_20": 1,
    "volume_price_residual_20": 1,
    "overnight_intraday_corr_20": 0,
    "breakout_distance_20": 1,
    "breakout_volume_confirmation": 1,
    "pullback_quality_20": 1,
    "stalling_pressure_20": -1,
    "low_volume_long_lower_shadow": 1,
    "peer_mean_ret_1": 1,
    "peer_mean_ret_5": 1,
    "peer_mean_ret_20": 1,
    "excess_ret_1": 1,
    "excess_ret_5": 1,
    "excess_ret_20": 1,
    "excess_vs_universe_ret_1": 1,
    "excess_vs_universe_ret_5": 1,
    "excess_vs_universe_ret_20": 1,
    "excess_vs_industry_ret_1": 1,
    "excess_vs_industry_ret_5": 1,
    "excess_vs_industry_ret_20": 1,
    "excess_vs_csi300_ret_1": 1,
    "excess_vs_csi300_ret_5": 1,
    "excess_vs_csi300_ret_20": 1,
    "rank_ret_5": 1,
    "rank_ma_gap_20": 1,
    "atr_14_pct": -1,
    "volatility_20": -1,
    "amplitude_1": -1,
    "drawdown_20": 1,
    "peer_dispersion_ret_5": -1,
    "log_amount_yuan": 1,
    "rank_volume_ratio_5_20": 1,
    "rank_volatility_20": -1,
    "rank_log_amount": 1,
    "turnover_rate_daily": 1,
    "rank_turnover_rate_daily": 1,
    "log_circ_mv": 1,
    "rank_log_circ_mv": 1,
    "universe_mean_ret_1": 1,
    "universe_mean_ret_5": 1,
    "universe_mean_ret_20": 1,
    "universe_breadth_above_ma20": 1,
    "universe_breadth_positive_5d": 1,
    "universe_dispersion_ret_5": -1,
    "industry_mean_ret_1": 1,
    "industry_mean_ret_5": 1,
    "industry_mean_ret_20": 1,
    "industry_breadth_above_ma20": 1,
    "industry_breadth_positive_5d": 1,
    "industry_dispersion_ret_5": -1,
    "market_reference_mean_ret_1": 1,
    "market_reference_mean_ret_5": 1,
    "market_reference_mean_ret_20": 1,
    "market_regime_score": 1,
    "market_regime_weak": -1,
    "market_regime_sideways": 0,
    "market_regime_strong": 1,
    "market_regime_trend_breadth_interaction": 1,
    "market_regime_volatility_stress": -1,
}


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if np.isfinite(number) else None


def zengjia_hengjiemian_yinzi(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = data.groupby("ts_code", group_keys=False)
    close = pd.to_numeric(data["close"], errors="coerce")
    open_price = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    previous_close = grouped["close"].shift(1)
    data["gap_open"] = open_price / pd.to_numeric(previous_close, errors="coerce") - 1.0
    data["intraday_return"] = close / open_price.replace(0, np.nan) - 1.0
    daily_range = (high - low).replace(0, np.nan)
    data["close_location"] = ((close - low) / daily_range).clip(0.0, 1.0).fillna(0.5)
    if "amount_yuan" not in data.columns:
        data["amount_yuan"] = np.nan
    data["log_amount_yuan"] = np.log1p(pd.to_numeric(data["amount_yuan"], errors="coerce").clip(lower=0))
    data["amount_mean_5"] = data.groupby("ts_code")["amount_yuan"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(5, min_periods=5).mean()
    )
    data["amount_mean_20"] = data.groupby("ts_code")["amount_yuan"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=20).mean()
    )
    data["amount_ratio_5_20"] = data["amount_mean_5"] / data["amount_mean_20"].replace(0, np.nan)
    by_date = data.groupby("trade_date", group_keys=False)
    for period in (1, 5, 20):
        data[f"peer_mean_ret_{period}"] = by_date[f"ret_{period}"].transform("mean")
        data[f"excess_ret_{period}"] = data[f"ret_{period}"] - data[f"peer_mean_ret_{period}"]
    data["peer_breadth_above_ma20"] = by_date["ma_gap_20"].transform(
        lambda values: values.gt(0).where(values.notna()).mean()
    )
    data["peer_breadth_positive_5d"] = by_date["ret_5"].transform(
        lambda values: values.gt(0).where(values.notna()).mean()
    )
    data["peer_dispersion_ret_5"] = by_date["ret_5"].transform("std")
    for source, target in {
        "ret_5": "rank_ret_5",
        "ma_gap_20": "rank_ma_gap_20",
        "volume_ratio_5_20": "rank_volume_ratio_5_20",
        "volatility_20": "rank_volatility_20",
        "log_amount_yuan": "rank_log_amount",
    }.items():
        data[target] = by_date[source].transform(lambda values: values.rank(pct=True))
    return data.replace([np.inf, -np.inf], np.nan)


def goujian_fenxi_yinzi_mianban(
    histories: dict[str, pd.DataFrame],
    profiles: pd.DataFrame,
    *,
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从批量前复权日线构造不含未来标签的分析因子面板。"""
    profile_by_code = (
        profiles.drop_duplicates("ts_code").set_index("ts_code").to_dict("index")
        if profiles is not None and not profiles.empty and "ts_code" in profiles.columns
        else {}
    )
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for code, history in histories.items():
        if history is None or history.empty:
            continue
        try:
            features = jisuan_tezheng_biao(history)
        except Exception as exc:
            warnings.append(f"{code} 技术因子构造失败：{exc}")
            continue
        features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce").dt.normalize()
        features["ts_code"] = str(code)
        profile = profile_by_code.get(str(code), {})
        features["name"] = str(profile.get("name") or "")
        features["industry"] = str(profile.get("industry") or "")
        frames.append(features)
    if not frames:
        return pd.DataFrame(), {"status": "unavailable", "warnings": warnings or ["没有可构造因子面板的历史行情"]}
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = zengjia_hengjiemian_yinzi(panel)
    try:
        panel, factor_meta = enrich_daily_factor_panel(
            panel,
            source=source,
            include_historical_valuation=False,
        )
    except Exception as exc:
        factor_meta = {"status": "degraded", "warnings": [f"历史估值和市场背景增强失败：{exc}"]}
    factor_meta = dict(factor_meta)
    factor_meta["warnings"] = warnings + [str(value) for value in factor_meta.get("warnings", [])]
    factor_meta["future_labels_created"] = False
    return panel, factor_meta


def zengjia_dangri_guzhi_yinzi(
    latest_panel: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """用已取得的同日横截面补充最新风险因子，不逐股读取历史估值。"""
    if latest_panel is None or latest_panel.empty:
        return pd.DataFrame()
    data = latest_panel.copy()
    if profiles is None or profiles.empty or "ts_code" not in profiles.columns:
        return data
    available = [
        column
        for column in ("ts_code", "turnover_rate", "circulating_market_value_yuan")
        if column in profiles.columns
    ]
    profile_values = profiles[available].drop_duplicates("ts_code", keep="last").copy()
    profile_values = profile_values.rename(
        columns={
            "turnover_rate": "_snapshot_turnover_rate",
            "circulating_market_value_yuan": "_snapshot_circ_mv_yuan",
        }
    )
    data = data.drop(
        columns=["_snapshot_turnover_rate", "_snapshot_circ_mv_yuan"],
        errors="ignore",
    ).merge(profile_values, on="ts_code", how="left")
    turnover = pd.to_numeric(
        data.get("_snapshot_turnover_rate", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    circ_mv = pd.to_numeric(
        data.get("_snapshot_circ_mv_yuan", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    data["turnover_rate_daily"] = turnover / 100.0
    data["log_circ_mv"] = np.log(circ_mv.where(circ_mv > 0))
    by_date = data.groupby("trade_date", group_keys=False)
    data["rank_turnover_rate_daily"] = by_date["turnover_rate_daily"].transform(
        lambda values: values.rank(pct=True)
    )
    data["rank_log_circ_mv"] = by_date["log_circ_mv"].transform(
        lambda values: values.rank(pct=True)
    )
    return data.drop(
        columns=["_snapshot_turnover_rate", "_snapshot_circ_mv_yuan"],
        errors="ignore",
    ).replace([np.inf, -np.inf], np.nan)


def _context_score(feature: str, value: float, direction: int) -> float | None:
    if "breadth" in feature:
        raw = min(max(value, 0.0), 1.0) * 100.0
    elif feature.endswith(("_weak", "_sideways", "_strong")):
        raw = min(max(value, 0.0), 1.0) * 100.0
    elif "regime_score" in feature or "interaction" in feature:
        raw = 50.0 + 50.0 * np.tanh(value)
    elif "dispersion" in feature or "volatility_stress" in feature:
        raw = 50.0 + 50.0 * np.tanh(value / 0.03)
    elif "ret_" in feature:
        raw = 50.0 + 50.0 * np.tanh(value / 0.03)
    else:
        return None
    return float(raw if direction >= 0 else 100.0 - raw)


def _oriented_percentile(
    feature: str,
    value: float,
    comparable: pd.Series,
) -> float | None:
    direction = YINZI_FANGXIANG.get(feature, 0)
    if direction == 0:
        return None
    values = pd.to_numeric(comparable, errors="coerce").dropna()
    if len(values) >= 5 and int(values.nunique()) >= 2:
        percentile = float((values <= value).mean()) * 100.0
        return percentile if direction > 0 else 100.0 - percentile
    return _context_score(feature, value, direction)


def huizong_houxuan_yinzi(
    latest_panel: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """把八组因子转换为可追溯的候选排序证据。"""
    if latest_panel is None or latest_panel.empty or "ts_code" not in latest_panel.columns:
        return {}
    data = latest_panel.copy()
    weights = {key: float(value) for key, value in (config.get("factor_group_weights") or {}).items()}
    results: dict[str, dict[str, Any]] = {}
    for _, row in data.iterrows():
        code = str(row["ts_code"])
        group_results: dict[str, dict[str, Any]] = {}
        weighted_scores: list[tuple[float, float]] = []
        available_weight = 0.0
        total_oriented_factors = 0
        available_oriented_factors = 0
        evidence: list[str] = []
        for group, members in FACTOR_GROUPS.items():
            label, meaning = YINZI_ZU_MINGCHENG.get(group, (group, group))
            values: dict[str, float] = {}
            oriented: dict[str, float | None] = {}
            expected_oriented = sum(1 for feature in members if YINZI_FANGXIANG.get(feature, 0) != 0)
            total_oriented_factors += expected_oriented
            for feature in members:
                if feature not in row.index:
                    continue
                number = _number(row.get(feature))
                if number is None:
                    continue
                values[feature] = number
                score = _oriented_percentile(
                    feature,
                    number,
                    data[feature] if feature in data.columns else pd.Series(dtype=float),
                )
                oriented[feature] = round(score, 2) if score is not None else None
            available_scores = [value for value in oriented.values() if value is not None]
            available_oriented_factors += len(available_scores)
            group_score = round(float(np.mean(available_scores)), 2) if available_scores else None
            group_weight = float(weights.get(group, 0.0))
            if group_score is not None and group_weight > 0:
                weighted_scores.append((group_score, group_weight))
                available_weight += group_weight
            interpretation = (
                "正面证据较强"
                if group_score is not None and group_score >= 65
                else "负面或风险证据较强"
                if group_score is not None and group_score <= 35
                else "证据中性"
                if group_score is not None
                else "信息有限"
            )
            group_results[group] = {
                "label": label,
                "economic_meaning": meaning,
                "status": "ok" if group_score is not None else "unavailable",
                "factor_count": len(members),
                "oriented_factor_count": expected_oriented,
                "available_factor_count": len(values),
                "scored_factor_count": len(available_scores),
                "values": values,
                "oriented_percentiles_0_100": oriented,
                "score_0_100": group_score,
                "interpretation": interpretation,
            }
            evidence.append(
                f"{label}：{interpretation}；使用 {len(values)}/{len(members)} 个可得因子，定向得分 {group_score if group_score is not None else '不可用'}"
            )
        factor_score = (
            round(sum(score * weight for score, weight in weighted_scores) / available_weight, 2)
            if available_weight > 0
            else None
        )
        factor_coverage = (
            available_oriented_factors / total_oriented_factors
            if total_oriented_factors > 0
            else 0.0
        )
        # 组别覆盖决定结论骨架，组内字段覆盖用于进一步降级；避免一个数据源
        # 缺少少数扩展字段时把已有的八组证据整体误判为不可用。
        confidence = round(min(1.0, 0.7 * available_weight + 0.3 * factor_coverage), 4)
        results[code] = {
            "status": "ok" if factor_score is not None else "unavailable",
            "score_0_100": factor_score,
            "confidence": confidence,
            "available_group_weight": round(available_weight, 4),
            "available_oriented_factor_count": available_oriented_factors,
            "oriented_factor_count": total_oriented_factors,
            "groups": group_results,
            "evidence": evidence,
            "score_definition": "八组日 K 因子按配置权重汇总的横截面研究分，只用于候选排序，不是上涨概率",
            "orientation_definition": "每个因子的有利方向集中定义；混合极性的原始分位数不会直接相加",
        }
    return results


def jisuan_hengjiemian_jibenmian(snapshot: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """用同日估值横截面做批量基本面初筛；深度财务数据稍后只取少量候选。"""
    if snapshot is None or snapshot.empty or "ts_code" not in snapshot.columns:
        return {}
    data = snapshot.copy()
    industry = data.get("industry", pd.Series("", index=data.index)).fillna("").astype(str)
    results: dict[str, dict[str, Any]] = {}
    for index, row in data.iterrows():
        parts: list[float] = []
        evidence: list[str] = []
        for feature, label in (("pe_ttm", "市盈率"), ("pb", "市净率")):
            value = _number(row.get(feature))
            if value is None or value <= 0 or feature not in data.columns:
                continue
            same_industry = pd.to_numeric(data.loc[industry.eq(industry.loc[index]), feature], errors="coerce")
            comparable = same_industry[(same_industry > 0) & same_industry.notna()]
            if len(comparable) < 5:
                comparable = pd.to_numeric(data[feature], errors="coerce")
                comparable = comparable[(comparable > 0) & comparable.notna()]
            if len(comparable) >= 5:
                lower_is_better = 100.0 * float((comparable >= value).mean())
                parts.append(lower_is_better)
                evidence.append(f"{label} {value:.2f}，同类有效样本中的估值适配分 {lower_is_better:.1f}/100")
        score = round(float(np.mean(parts)), 2) if parts else None
        results[str(row["ts_code"])] = {
            "status": "ok" if score is not None else "unavailable",
            "score_0_100": score,
            "confidence": round(min(1.0, len(parts) / 2.0), 4),
            "evidence": evidence,
            "scope": "同日估值横截面初筛",
            "limitations": ["ROE、利润增长和负债等财务指标只对排名靠前的少量候选做深度读取"],
            "score_definition": "估值相对位置分，不是上涨概率",
        }
    return results


def jisuan_shendu_jibenmian(fundamentals: dict[str, Any]) -> tuple[int | None, list[str]]:
    """对少量候选的已取得财务与估值字段做确定性评分。"""
    financials = fundamentals.get("financials") or {}
    valuation = fundamentals.get("valuation") or {}
    profile = fundamentals.get("profile") or {}
    industry = str(profile.get("industry") or profile.get("所属行业") or "")
    financial_industry = any(
        keyword in industry for keyword in ("银行", "保险", "证券", "多元金融", "金融服务")
    )
    evidence: list[str] = []
    score = 50.0
    observed = 0
    roe = _number(financials.get("roe_pct"))
    if roe is not None:
        observed += 1
        if roe >= 15:
            score += 15
            evidence.append("ROE 较强")
        elif roe >= 8:
            score += 7
            evidence.append("ROE 为正且处于中等水平")
        elif roe < 0:
            score -= 18
            evidence.append("ROE 为负")
    growth = _number(financials.get("net_profit_yoy_pct"))
    if growth is not None:
        observed += 1
        if growth >= 15:
            score += 12
            evidence.append("净利润同比增长较快")
        elif growth < -15:
            score -= 15
            evidence.append("净利润同比明显下降")
    debt = _number(financials.get("debt_to_assets_pct"))
    if debt is not None:
        observed += 1
        if financial_industry:
            evidence.append("金融行业资产负债率口径特殊，本项只展示、不加减分")
        elif debt > 75:
            score -= 12
            evidence.append("资产负债率偏高，需结合行业解释")
        elif debt < 45:
            score += 5
            evidence.append("资产负债率相对温和")
    pe = _number(valuation.get("pe_ttm"))
    if pe is None:
        pe = _number(valuation.get("pe_dynamic"))
    if pe is not None:
        observed += 1
        if pe <= 0:
            score -= 12
            evidence.append("市盈率为负，通常意味着当前口径下亏损")
        elif pe > 80:
            score -= 8
            evidence.append("市盈率较高，估值对增长兑现要求较高")
        else:
            evidence.append("估值数据可用，需与同行比较后再下结论")
    return (int(round(max(0, min(100, score)))) if observed >= 2 else None), evidence


__all__ = [
    "YINZI_FANGXIANG",
    "YINZI_ZU_MINGCHENG",
    "goujian_fenxi_yinzi_mianban",
    "huizong_houxuan_yinzi",
    "jisuan_hengjiemian_jibenmian",
    "jisuan_shendu_jibenmian",
    "zengjia_dangri_guzhi_yinzi",
    "zengjia_hengjiemian_yinzi",
]
