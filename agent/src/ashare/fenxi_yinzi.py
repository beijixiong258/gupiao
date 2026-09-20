"""分析侧复用的八组日 K 因子面板与确定性证据汇总。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ashare.gupiao_yanjiu import jisuan_tezheng_biao
from src.ashare.riping_yinzi import enrich_daily_factor_panel
from src.ashare.yinzi_gongcheng import FACTOR_GROUPS, FACTOR_REGISTRY, RAW_PRICE_VOLUME_FEATURE_COLUMNS, add_price_volume_factors



YINZI_ZU_MINGCHENG: dict[str, tuple[str, str]] = {
    "trend_structure": ("趋势结构", "均线结构、趋势斜率和趋势拟合质量"),
    "momentum_reversal": ("动量与反转", "多周期收益、RSI 与 MACD"),
    "candle_pressure": ("K线压力", "跳空、实体、收盘位置与影线压力"),
    "price_volume_confirmation": ("价量确认", "成交量、成交额异常和价量协同"),
    "breakout_pullback_quality": ("突破与回撤质量", "突破距离、放量确认和回撤质量"),
    "relative_strength": ("相对强弱", "相对同行、行业与市场的超额表现"),
    "risk_liquidity": ("风险与流动性", "波动、回撤、成交额、换手和市值流动性"),
    "market_context": ("市场背景", "当前比较池与真实行业分组的收益、广度、离散度和状态"),
}


def _number(value: Any, digits: int | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return (round(number, digits) if digits is not None else number) if np.isfinite(number) else None


def zengjia_hengjiemian_yinzi(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if not set(RAW_PRICE_VOLUME_FEATURE_COLUMNS).issubset(data.columns):
        data = add_price_volume_factors(data)
    if "amount_yuan" not in data.columns:
        data["amount_yuan"] = np.nan
    amount = pd.to_numeric(data["amount_yuan"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data["amount_yuan"] = amount.where(amount >= 0)
    data["log_amount_yuan"] = np.log1p(data["amount_yuan"])
    data["amount_mean_5"] = data.groupby("ts_code")["amount_yuan"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(5, min_periods=5).mean()
    )
    data["amount_mean_20"] = data.groupby("ts_code")["amount_yuan"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(20, min_periods=20).mean()
    )
    data["amount_ratio_5_20"] = data["amount_mean_5"] / data["amount_mean_20"].replace(0, np.nan)
    by_date = data.groupby("trade_date", group_keys=False)
    for period in (1, 5, 20):
        data[f"peer_mean_ret_{period}"] = by_date[f"ret_{period}"].transform("mean").where(
            by_date[f"ret_{period}"].transform("count").ge(2)
        )
        data[f"excess_ret_{period}"] = data[f"ret_{period}"] - data[f"peer_mean_ret_{period}"]
    data["peer_breadth_above_ma20"] = by_date["ma_gap_20"].transform(
        lambda values: values.gt(0).where(values.notna()).mean() if values.count() >= 2 else np.nan
    )
    data["peer_breadth_positive_5d"] = by_date["ret_5"].transform(
        lambda values: values.gt(0).where(values.notna()).mean() if values.count() >= 2 else np.nan
    )
    data["peer_dispersion_ret_5"] = by_date["ret_5"].transform("std")
    for source, target in {
        "ret_5": "rank_ret_5",
        "ma_gap_20": "rank_ma_gap_20",
        "volume_ratio_5_20": "rank_volume_ratio_5_20",
        "volatility_20": "rank_volatility_20",
        "log_amount_yuan": "rank_log_amount",
    }.items():
        data[target] = by_date[source].transform(lambda values: values.rank(pct=True) if values.count() >= 2 else np.nan)
    return data.replace([np.inf, -np.inf], np.nan)


def goujian_fenxi_yinzi_mianban(
    histories: dict[str, pd.DataFrame],
    profiles: pd.DataFrame,
    *,
    source: str,
    calendar: Any | None = None,
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
        if profile.get("peer_role") is not None:
            features["peer_role"] = str(profile.get("peer_role") or "market_reference")
        frames.append(features)
    if not frames:
        return pd.DataFrame(), {"status": "unavailable", "warnings": warnings or ["没有可构造因子面板的历史行情"]}
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = zengjia_hengjiemian_yinzi(panel)
    try:
        panel, factor_meta = enrich_daily_factor_panel(
            panel,
            source=source,
            calendar=calendar,
        )
    except Exception as exc:
        factor_meta = {"status": "degraded", "warnings": [f"市场背景增强失败：{exc}"]}
    factor_meta = dict(factor_meta)
    factor_meta["warnings"] = warnings + [str(value) for value in factor_meta.get("warnings", [])]
    factor_meta["future_labels_created"] = False
    return panel, factor_meta


def _daily_valuation_dates(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    missing_dates = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    trade_dates = pd.to_datetime(data.get("trade_date", missing_dates), errors="coerce").dt.normalize()
    valuation_dates = pd.to_datetime(data.get("valuation_trade_date", missing_dates), errors="coerce").dt.normalize()
    verified = (
        data.get("valuation_is_complete_daily", pd.Series(False, index=data.index)).eq(True)
        & data.get("valuation_source", pd.Series("", index=data.index)).eq("tushare_daily_basic")
        & trade_dates.notna() & valuation_dates.eq(trade_dates)
    ).fillna(False)
    return trade_dates, valuation_dates, verified


def zengjia_dangri_guzhi_yinzi(
    latest_panel: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """仅用来源和逐行日期已核验的日终估值补缺，不覆盖已有日线值。"""
    if latest_panel is None or latest_panel.empty:
        return pd.DataFrame()
    data = latest_panel.copy()
    if profiles is None or profiles.empty or "ts_code" not in profiles.columns:
        return data
    required = (
        "trade_date", "valuation_trade_date", "valuation_source", "valuation_is_complete_daily",
    )
    if "trade_date" not in data.columns or not set(required).issubset(profiles.columns):
        return data
    available = [
        column for column in ("ts_code", *required, "turnover_rate", "turnover_rate_pct", "circulating_market_value_yuan")
        if column in profiles.columns
    ]
    profile_values = profiles[available].copy()
    if "turnover_rate_pct" in profile_values:
        turnover = pd.to_numeric(profile_values.get("turnover_rate", pd.Series(np.nan, index=profile_values.index)), errors="coerce")
        profile_values["turnover_rate"] = turnover.fillna(pd.to_numeric(profile_values["turnover_rate_pct"], errors="coerce"))
        profile_values = profile_values.drop(columns=["turnover_rate_pct"])
    profile_dates, _, verified = _daily_valuation_dates(profile_values)
    profile_values = profile_values.loc[verified].copy()
    profile_values["trade_date"] = profile_dates.loc[verified]
    profile_values = profile_values.drop_duplicates(["ts_code", "trade_date"], keep="last")
    if profile_values.empty:
        return data
    profile_values = profile_values.drop(columns=list(required[1:]))
    profile_values = profile_values.rename(
        columns={
            "turnover_rate": "_snapshot_turnover_rate",
            "circulating_market_value_yuan": "_snapshot_circ_mv_yuan",
        }
    )
    data["_snapshot_trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    profile_values = profile_values.rename(columns={"trade_date": "_snapshot_trade_date"})
    data = data.drop(
        columns=["_snapshot_turnover_rate", "_snapshot_circ_mv_yuan"],
        errors="ignore",
    ).merge(profile_values, on=["ts_code", "_snapshot_trade_date"], how="left")
    turnover = pd.to_numeric(
        data.get("_snapshot_turnover_rate", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    circ_mv = pd.to_numeric(
        data.get("_snapshot_circ_mv_yuan", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    for column, supplement in {
        "turnover_rate_daily": turnover.where(turnover.ge(0)) / 100.0,
        "log_circ_mv": np.log(circ_mv.where(circ_mv > 0)),
    }.items():
        current = pd.to_numeric(data.get(column, pd.Series(np.nan, index=data.index)), errors="coerce")
        current = current.where(np.isfinite(current))
        if column == "turnover_rate_daily":
            current = current.where(current.ge(0))
        data[column] = current.fillna(supplement.where(np.isfinite(supplement)))
    by_date = data.groupby("trade_date", group_keys=False)
    data["rank_turnover_rate_daily"] = by_date["turnover_rate_daily"].transform(
        lambda values: values.rank(pct=True) if values.count() >= 2 else np.nan
    )
    data["rank_log_circ_mv"] = by_date["log_circ_mv"].transform(
        lambda values: values.rank(pct=True) if values.count() >= 2 else np.nan
    )
    return data.drop(
        columns=["_snapshot_trade_date", "_snapshot_turnover_rate", "_snapshot_circ_mv_yuan"],
        errors="ignore",
    ).replace([np.inf, -np.inf], np.nan)


def huizong_houxuan_yinzi(
    latest_panel: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """逐项保留八组原始因子及缺失；config 仅保留调用兼容，不用于合成分数。"""
    if latest_panel is None or latest_panel.empty or "ts_code" not in latest_panel.columns:
        return {}
    definitions = {
        entry["feature"]: {
            key: entry[key]
            for key in ("label", "unit", "meaning", "display_scale", "canonical_source", "input_requirement", "frequency", "availability", "group", "role",
                        "formula", "window", "minimum_observations", "missing_rule", "source_basis")
            if key in entry
        }
        for entry in FACTOR_REGISTRY
    }
    results: dict[str, dict[str, Any]] = {}
    for _, row in latest_panel.iterrows():
        groups: dict[str, dict[str, Any]] = {}
        evidence: list[str] = []
        available_count = 0
        total_count = 0
        for group, members in FACTOR_GROUPS.items():
            label, meaning = YINZI_ZU_MINGCHENG.get(group, (group, group))
            values = {feature: _number(row.get(feature)) for feature in members}
            missing = [feature for feature, value in values.items() if value is None]
            available = len(members) - len(missing)
            groups[group] = {
                "label": label,
                "economic_meaning": meaning,
                "values": values,
                "missing_fields": missing,
                "metric_definitions": {
                    feature: {**definitions.get(feature, {}), **(
                        {"valid_observations": int(row[f"{feature}_sample_count"])}
                        if _number(row.get(f"{feature}_sample_count")) is not None else {}
                    )} for feature in members
                },
                "available_factor_count": available,
                "factor_count": len(members),
            }
            available_count += available
            total_count += len(members)
            evidence.append(
                f"{label}：已取得 {available}/{len(members)} 项原始指标"
                + (f"；缺少 {len(missing)} 项" if missing else "")
            )
        results[str(row["ts_code"])] = {
            "ts_code": str(row["ts_code"]),
            "trade_date": pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d") if pd.notna(row.get("trade_date")) else None,
            "status": "ok" if available_count == total_count else "partial" if available_count else "unavailable",
            "groups": groups,
            "available_factor_count": available_count,
            "total_factor_count": total_count,
            "evidence": evidence,
        }
    return results


def jisuan_hengjiemian_jibenmian(snapshot: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """原始估值完整保留，仅经逐行核验的同日日终估值参与相对分位。"""
    if snapshot is None or snapshot.empty or "ts_code" not in snapshot.columns:
        return {}
    data = snapshot.copy().drop_duplicates("ts_code", keep="last").reset_index(drop=True)
    industry = data.get("industry", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    known_industry = ~industry.str.lower().isin({"", "unknown", "nan", "none", "未知", "未分类"})
    trade_dates, valuation_dates, verified = _daily_valuation_dates(data)
    results: dict[str, dict[str, Any]] = {}
    fields = ("pe_ttm", "pe_dynamic", "pe", "pe_unspecified", "pb", "total_market_value_yuan", "circulating_market_value_yuan", "turnover_rate")
    for index, row in data.iterrows():
        values = {feature: _number(row.get(feature)) for feature in fields}
        relative: dict[str, dict[str, Any]] = {}
        evidence: list[str] = []
        date_reason = (
            "" if verified.loc[index]
            else "行情日期缺失，无法核验同日估值" if pd.isna(trade_dates.loc[index])
            else "估值日期缺失，无法核验同日估值" if pd.isna(valuation_dates.loc[index])
            else f"估值日期 {valuation_dates.loc[index]:%Y-%m-%d} 与行情日期 {trade_dates.loc[index]:%Y-%m-%d} 不一致"
            if valuation_dates.loc[index] != trade_dates.loc[index]
            else "估值来源未核验为日终来源" if str(row.get("valuation_source")) != "tushare_daily_basic"
            else "估值尚未确认为完整日终数据"
        )
        same_day_verified = verified & valuation_dates.eq(trade_dates.loc[index])
        for feature, label in (("pe_ttm", "市盈率"), ("pb", "市净率")):
            value = values[feature]
            series = pd.to_numeric(data.get(feature, pd.Series(np.nan, index=data.index)), errors="coerce")
            series = series.where(np.isfinite(series) & series.gt(0) & same_day_verified)
            peers = series[industry.eq(industry.loc[index])].dropna() if known_industry.loc[index] else pd.Series(dtype=float)
            source = "same_industry" if len(peers) >= 5 else "current_comparison_pool"
            comparable = peers if source == "same_industry" else series.dropna()
            percentile = (
                round(float(comparable.le(value).mean()) * 100.0, 4)
                if verified.loc[index] and value is not None and value > 0 and len(comparable) >= 5
                else None
            )
            reason = (
                "" if percentile is not None
                else date_reason if date_reason
                else "缺少有效正估值" if value is None or value <= 0
                else "有效正估值比较样本少于 5 只"
            )
            relative[feature] = {
                "value": value,
                "percentile": percentile,
                "percentile_unit": "percent",
                "definition": "不高于当前值的有效正估值样本占比；仅描述相对位置",
                "sample_count": len(comparable),
                "excluded_unverified_or_other_date_count": int((~same_day_verified).sum()),
                "comparison_source": source,
                "industry": industry.loc[index] if source == "same_industry" else None,
                "status": "ok" if percentile is not None else "unavailable",
                "reason": reason,
            }
            if value is None:
                evidence.append(f"{label}缺失")
            elif percentile is None:
                evidence.append(f"{label} {value:g}；{reason}")
            else:
                comparison_label = "同一行业" if source == "same_industry" else "当前比较池"
                evidence.append(
                    f"{label} {value:g}；在{comparison_label} {len(comparable)} 个有效样本中"
                    f"的分位为 {percentile:g}%"
                )
        source_fields = ("source", "valuation_source", "captured_at", "trade_date", "valuation_trade_date", "as_of")
        sources = {field: str(row.get(field)) for field in source_fields if pd.notna(row.get(field)) and row.get(field) is not None}
        sources.update({key: value for key, value in snapshot.attrs.items() if key in source_fields and key not in sources})
        available = any(value is not None for value in values.values())
        results[str(row["ts_code"])] = {
            "status": "ok" if available and all(part["status"] == "ok" for part in relative.values()) else "partial" if available else "unavailable",
            "values": values,
            "missing_fields": [field for field, value in values.items() if value is None],
            "relative_valuation": relative,
            "valuation_time_verification": {
                "status": "verified" if verified.loc[index] else "unavailable",
                "reason": date_reason,
                "trade_date": trade_dates.loc[index].strftime("%Y-%m-%d") if pd.notna(trade_dates.loc[index]) else None,
                "valuation_trade_date": valuation_dates.loc[index].strftime("%Y-%m-%d") if pd.notna(valuation_dates.loc[index]) else None,
            },
            "sources": sources,
            "evidence": evidence,
            "scope": "同日估值证据" if verified.loc[index] else "原始估值证据（时点未核验）",
            "limitations": ["估值分位仅反映当前有效比较样本，不代表上涨概率或价格低估程度"],
        }
    return results


__all__ = [
    "YINZI_ZU_MINGCHENG",
    "goujian_fenxi_yinzi_mianban",
    "huizong_houxuan_yinzi",
    "jisuan_hengjiemian_jibenmian",
    "zengjia_dangri_guzhi_yinzi",
    "zengjia_hengjiemian_yinzi",
]
