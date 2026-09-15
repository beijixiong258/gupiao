"""Deterministic daily-frequency factors and market context for stock analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.shichang_shuju import JiaoyiRili, akshare_zhilian
from src.ashare.shuju_yuan import _tushare_pro
from src.ashare.yinzi_gongcheng import describe_factor_coverage


BENCHMARKS = {
    "shanghai": {"ts_code": "000001.SH", "ak_symbol": "sh000001", "name": "上证指数"},
    "csi300": {"ts_code": "000300.SH", "ak_symbol": "sh000300", "name": "沪深300"},
    "csi1000": {"ts_code": "000852.SH", "ak_symbol": "sh000852", "name": "中证1000"},
}

BENCHMARK_FEATURE_COLUMNS = [
    f"market_{benchmark}_{feature}"
    for benchmark in BENCHMARKS
    for feature in ["ret_1", "ret_5", "ret_20", "volatility_20"]
]

DAILY_FACTOR_FEATURE_COLUMNS = [
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
    "excess_vs_universe_ret_1",
    "excess_vs_universe_ret_5",
    "excess_vs_universe_ret_20",
    "excess_vs_industry_ret_1",
    "excess_vs_industry_ret_5",
    "excess_vs_industry_ret_20",
    "excess_vs_csi300_ret_1",
    "excess_vs_csi300_ret_5",
    "excess_vs_csi300_ret_20",
    "turnover_rate_daily",
    "log_circ_mv",
    "earnings_yield_ttm",
    "book_to_price",
    "rank_turnover_rate_daily",
    "rank_log_circ_mv",
    "rank_earnings_yield_ttm",
    "rank_book_to_price",
    "size_neutral_ret_5",
    "size_neutral_ma_gap_20",
    "size_neutral_volume_ratio_5_20",
    "size_neutral_volatility_20",
    "size_neutral_log_amount_yuan",
    "market_regime_weak",
    "market_regime_sideways",
    "market_regime_strong",
    "market_regime_trend_breadth_interaction",
    "market_regime_volatility_stress",
] + BENCHMARK_FEATURE_COLUMNS

def _normalize_index_history(frame: pd.DataFrame, *, tushare: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if not tushare:
        data = data.rename(
            columns={
                "date": "trade_date",
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
    if not {"trade_date", "close"}.issubset(data.columns):
        return pd.DataFrame()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return (
        data.dropna(subset=["trade_date"])
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def _benchmark_calendar_dates(
    *, start: pd.Timestamp, end: pd.Timestamp, calendar: JiaoyiRili | None,
) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    """只消费本次权威日历；短个股历史也可请求指数所需的前 21 个交易日。"""
    quality: dict[str, Any] = {
        "source": calendar.source if calendar is not None else None,
        "fetched_at": calendar.fetched_at if calendar is not None else None,
        "required_latest_date": end.strftime("%Y-%m-%d"),
        "required_observations": 21,
    }
    if calendar is None:
        quality.update(status="unavailable", reason="calendar_unavailable")
        return pd.DatetimeIndex([]), quality
    if pd.Timestamp(calendar.start_date) > start or pd.Timestamp(calendar.end_date) < end:
        quality.update(status="unavailable", reason="calendar_coverage_insufficient")
        return pd.DatetimeIndex([]), quality
    dates = pd.DatetimeIndex(sorted(day for day in calendar.open_dates if day <= end))
    if end not in dates:
        quality.update(status="unavailable", reason="analysis_date_not_in_calendar")
        return pd.DatetimeIndex([]), quality
    if len(dates) < 21:
        quality.update(status="unavailable", reason="calendar_window_insufficient")
        return pd.DatetimeIndex([]), quality
    dates = dates[dates >= min(start, dates[-21])]
    quality.update(
        status="ok",
        requested_start_date=dates[0].strftime("%Y-%m-%d"),
        required_window_start_date=dates[-21].strftime("%Y-%m-%d"),
        expected_sessions=int(len(dates)),
    )
    return dates, quality


def _check_benchmark_history(
    history: pd.DataFrame, *, expected_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """当前窗口必须完整；更早的无效观测留空，不让行数冒充交易日。"""
    data = history.copy()
    if not {"trade_date", "close"}.issubset(data.columns):
        data = pd.DataFrame(columns=["trade_date", "close"])
    data = data[data["trade_date"].between(expected_dates[0], expected_dates[-1])].copy()
    dates = data["trade_date"]
    close = pd.to_numeric(data["close"], errors="coerce")
    invalid_close = ~np.isfinite(close) | close.le(0)
    duplicate = dates.duplicated(keep=False)
    outside_calendar = ~dates.isin(expected_dates)
    valid = data.loc[~invalid_close & ~duplicate & ~outside_calendar].copy()
    missing = expected_dates.difference(pd.DatetimeIndex(valid["trade_date"]))
    required_missing = expected_dates[-21:].intersection(missing)

    def labels(values: Any) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in sorted(set(values))]

    reasons: list[str] = []
    if data.empty:
        reasons.append("empty_history")
    if expected_dates[-1] in missing:
        reasons.append("latest_session_missing")
    if len(required_missing):
        reasons.append("required_window_incomplete")
    if bool((invalid_close & dates.isin(expected_dates[-21:])).any()):
        reasons.append("invalid_close_in_required_window")
    if bool((duplicate & dates.isin(expected_dates[-21:])).any()):
        reasons.append("duplicate_date_in_required_window")
    quality = {
        "accepted": not reasons,
        "rows": int(len(valid)),
        "required_latest_date": expected_dates[-1].strftime("%Y-%m-%d"),
        "latest_date": valid["trade_date"].max().strftime("%Y-%m-%d") if not valid.empty else None,
        "missing_trade_dates": labels(missing),
        "required_window_missing_dates": labels(required_missing),
        "invalid_close_dates": labels(dates[invalid_close]),
        "duplicate_trade_dates": labels(dates[duplicate]),
        "non_trading_dates": labels(dates[outside_calendar]),
        "historical_partial": bool(len(missing) or outside_calendar.any()),
        "reasons": reasons,
    }
    return valid.sort_values("trade_date").reset_index(drop=True), quality


def _fetch_one_benchmark(
    *,
    key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
    calendar: JiaoyiRili | None = None,
    quality: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    spec = BENCHMARKS[key]
    errors: list[str] = []
    details = quality if quality is not None else {}
    expected_dates, calendar_quality = _benchmark_calendar_dates(start=start, end=end, calendar=calendar)
    details.update(calendar=calendar_quality, attempted_providers=[], degraded=False)
    if expected_dates.empty:
        details.update(status="unavailable", reasons=[calendar_quality["reason"]])
        errors.append(f"{spec['name']} 无法核验指数交易日窗口：{calendar_quality['reason']}")
        return pd.DataFrame(), "unavailable", errors
    request_start = expected_dates[0]
    providers = [source] if source in {"tushare", "akshare"} else ["tushare", "akshare"]
    for provider in providers:
        try:
            if provider == "tushare":
                pro = _tushare_pro()
                raw = pro.index_daily(
                    ts_code=spec["ts_code"],
                    start_date=request_start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )
                fresh = _normalize_index_history(raw, tushare=True)
            else:
                import akshare as ak

                with akshare_zhilian():
                    raw = ak.stock_zh_index_daily(symbol=spec["ak_symbol"])
                fresh = _normalize_index_history(raw, tushare=False)
            fresh, source_quality = _check_benchmark_history(fresh, expected_dates=expected_dates)
            attempt = {"source": provider, **source_quality}
            details["attempted_providers"].append(attempt)
            if not source_quality["accepted"]:
                errors.append(f"{spec['name']} {provider} 日K质量不合格：{', '.join(source_quality['reasons'])}")
                continue
            details.update(
                status="partial" if source_quality["historical_partial"] else "ok",
                source=provider,
                degraded=bool(errors),
                quality=source_quality,
            )
            return fresh, provider, errors
        except Exception as exc:
            details["attempted_providers"].append({
                "source": provider, "accepted": False,
                "reasons": ["source_request_failed"], "error": str(exc),
            })
            errors.append(f"{spec['name']} {provider} 日K失败：{exc}")
    details.update(status="unavailable", source="unavailable", reasons=["no_qualified_source"])
    return pd.DataFrame(), "unavailable", errors


def _benchmark_features(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
    calendar: JiaoyiRili | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged: pd.DataFrame | None = None
    details: dict[str, Any] = {}
    warnings: list[str] = []
    expected_dates, _ = _benchmark_calendar_dates(start=start, end=end, calendar=calendar)
    for key in BENCHMARKS:
        source_meta: dict[str, Any] = {}
        history, provider, errors = _fetch_one_benchmark(
            key=key,
            start=start,
            end=end,
            source=source,
            calendar=calendar,
            quality=source_meta,
        )
        warnings.extend(errors)
        if history.empty:
            details[key] = {**source_meta, "status": "unavailable", "source": provider, "rows": 0}
            continue
        values = history.set_index("trade_date")[["close"]].reindex(expected_dates)
        values.index.name = "trade_date"
        values = values.reset_index()
        close = pd.to_numeric(values["close"], errors="coerce")
        for period in (1, 5, 20):
            complete = close.rolling(period + 1, min_periods=period + 1).count().eq(period + 1)
            values[f"market_{key}_ret_{period}"] = close.pct_change(period, fill_method=None).where(complete)
        values[f"market_{key}_volatility_20"] = (
            close.pct_change(fill_method=None).rolling(20, min_periods=20).std() * math.sqrt(252)
        )
        values = values.drop(columns=["close"])
        merged = values if merged is None else merged.merge(values, on="trade_date", how="outer")
        details[key] = {**source_meta, "source": provider, "rows": int(len(history))}
        if source_meta.get("status") == "partial":
            warnings.append(f"{BENCHMARKS[key]['name']} 较早历史存在无效或缺失观测，受影响窗口的指数指标留空")
    available = merged is not None and not merged.empty
    return (merged if merged is not None else pd.DataFrame()), {
        "status": (
            "ok" if available and all(item.get("status") == "ok" for item in details.values())
            else "partial" if available else "unavailable"
        ),
        "benchmarks": details,
        "warnings": warnings,
        "frequency": "daily_k_only",
        "persistence": "none",
    }


_MINIMUM_INDUSTRY_STOCKS = 5


def _group_snapshot(
    data: pd.DataFrame,
    mask: pd.Series,
    prefix: str,
    *,
    group_columns: tuple[str, ...] = ("trade_date",),
    minimum_stocks: int = 2,
) -> pd.DataFrame:
    aggregations = {
        f"{prefix}_mean_ret_1": ("ret_1", "mean"),
        f"{prefix}_mean_ret_5": ("ret_5", "mean"),
        f"{prefix}_mean_ret_20": ("ret_20", "mean"),
        f"{prefix}_dispersion_ret_5": ("ret_5", "std"),
        f"{prefix}_breadth_above_ma20": (
            "ma_gap_20", lambda values: values.gt(0).where(values.notna()).mean()
        ),
        f"{prefix}_breadth_positive_5d": (
            "ret_5", lambda values: values.gt(0).where(values.notna()).mean()
        ),
    }
    subset = data.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=[*group_columns, *aggregations, f"{prefix}_sample_count"])
    grouped = subset.groupby(list(group_columns), sort=True)
    result = grouped.agg(**aggregations)
    stock_counts = grouped["ts_code"].nunique()
    valid_counts = grouped[list({source for source, _ in aggregations.values()})].count()
    for output, (source, _) in aggregations.items():
        result[output] = result[output].where(
            stock_counts.ge(minimum_stocks) & valid_counts[source].ge(minimum_stocks)
        )
    result[f"{prefix}_sample_count"] = stock_counts
    return result.reset_index()


def _add_industry_context(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按真实当前行业标签分组；单股角色只限定参照成员，不代替行业身份。"""
    result = data.copy()
    industry = result.get("industry", pd.Series("", index=result.index)).fillna("").astype(str).str.strip()
    industry = industry.where(~industry.str.lower().isin({"", "nan", "none", "null", "<na>", "未知", "未知行业"}), "")
    result["industry"] = industry
    roles = result.get("peer_role", pd.Series("", index=result.index)).fillna("").astype(str)
    has_roles = roles.ne("").any()
    target_industry = None
    mismatch_stocks = 0
    if has_roles:
        target_labels = industry[roles.eq("target") & industry.ne("")].drop_duplicates()
        target_industry = str(target_labels.iloc[0]) if len(target_labels) == 1 else None
        role_mask = roles.isin(["target", "same_industry"])
        matching = industry.eq(target_industry) if target_industry else pd.Series(False, index=result.index)
        mask = role_mask & matching & industry.ne("")
        mismatch_stocks = int(result.loc[role_mask & ~matching, "ts_code"].nunique())
        stats = _group_snapshot(result, mask, "industry", minimum_stocks=_MINIMUM_INDUSTRY_STOCKS)
        result = result.merge(stats, on="trade_date", how="left")
        method = "目标股票与真实同业标签一致的角色成员，按交易日计算等权参照"
    else:
        stats = _group_snapshot(
            result, industry.ne(""), "industry",
            group_columns=("trade_date", "industry"), minimum_stocks=_MINIMUM_INDUSTRY_STOCKS,
        )
        result = result.merge(stats, on=["trade_date", "industry"], how="left")
        method = "当前比较池内按交易日与真实行业标签分组的等权参照"
    result["industry_sample_count"] = pd.to_numeric(result["industry_sample_count"], errors="coerce").fillna(0)
    context_columns = [column for column in stats.columns if column.startswith("industry_") and column != "industry_sample_count"]
    available = bool(result[context_columns].notna().any().any()) if context_columns else False
    missing_labels = int(data.loc[industry.eq(""), "ts_code"].nunique())
    insufficient_groups = int((pd.to_numeric(stats["industry_sample_count"], errors="coerce") < _MINIMUM_INDUSTRY_STOCKS).sum())
    warnings: list[str] = []
    if missing_labels:
        warnings.append(f"{missing_labels} 只股票缺少真实行业标签，不以整个比较池代替行业")
    if insufficient_groups:
        warnings.append(f"{insufficient_groups} 个行业交易日分组不足 {_MINIMUM_INDUSTRY_STOCKS} 只股票，行业证据留空")
    if mismatch_stocks:
        warnings.append(f"{mismatch_stocks} 只角色成员缺少或不匹配目标真实行业，未纳入同行统计")
    return result, {
        "status": "ok" if available else "unavailable",
        "method": method,
        "minimum_stocks": _MINIMUM_INDUSTRY_STOCKS,
        "minimum_valid_observations_per_field": _MINIMUM_INDUSTRY_STOCKS,
        "target_industry": target_industry,
        "missing_industry_stocks": missing_labels,
        "excluded_role_stocks": mismatch_stocks,
        "insufficient_group_dates": insufficient_groups,
        "scope": "当前比较池及本次真实标签，不是历史行业完整成分或全市场统计",
        "warnings": warnings,
    }

def _add_market_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    """市场状态来自明确的方向与广度条件，不合成加权分数。"""
    result = data.copy()
    trend = pd.to_numeric(result.get("market_csi300_ret_20"), errors="coerce")
    breadth = pd.to_numeric(result.get("universe_breadth_above_ma20"), errors="coerce")
    valid = trend.notna() & breadth.notna()
    strong = trend.gt(0) & breadth.gt(0.5)
    weak = trend.lt(0) & breadth.lt(0.5)
    result["market_regime_strong"] = strong.astype(float).where(valid)
    result["market_regime_weak"] = weak.astype(float).where(valid)
    result["market_regime_sideways"] = (~strong & ~weak).astype(float).where(valid)
    result["market_regime_trend_breadth_interaction"] = trend * breadth
    result["market_regime_volatility_stress"] = pd.to_numeric(result.get("market_csi300_volatility_20"), errors="coerce")
    return result


def _size_neutral_residual(group: pd.DataFrame, feature: str) -> pd.Series:
    result = pd.Series(np.nan, index=group.index, dtype=float)
    values = pd.to_numeric(group.get(feature), errors="coerce")
    size = pd.to_numeric(group.get("log_circ_mv"), errors="coerce")
    valid = values.notna() & size.notna()
    if int(valid.sum()) < 5 or int(size[valid].nunique()) < 2:
        return result
    x = size[valid].to_numpy(dtype=float)
    y = values[valid].to_numpy(dtype=float)
    x_centered = x - float(np.mean(x))
    denominator = float(np.dot(x_centered, x_centered))
    beta = float(np.dot(x_centered, y - float(np.mean(y))) / denominator) if denominator > 0 else 0.0
    result.loc[valid] = y - (float(np.mean(y)) + beta * x_centered)
    return result


def enrich_daily_factor_panel(
    panel: pd.DataFrame,
    *,
    source: str,
    calendar: JiaoyiRili | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add benchmark, group and size-neutral factors from supplied daily evidence."""
    if panel is None or panel.empty:
        return pd.DataFrame(), {"status": "unavailable", "warnings": ["分析因子面板为空"]}
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["trade_date"]).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    start = pd.Timestamp(data["trade_date"].min()).normalize()
    end = pd.Timestamp(data["trade_date"].max()).normalize()

    if "amount_yuan" not in data.columns:
        data["amount_yuan"] = np.nan
    data["log_amount_yuan"] = np.log1p(pd.to_numeric(data["amount_yuan"], errors="coerce").clip(lower=0))

    # 分析只消费调用方已提供的日频证据；缺失估值字段留空，最新横截面
    # 的换手和市值由分析汇总层补入，不逐股下载历史估值。
    for column in ("turnover_rate_daily", "log_circ_mv", "earnings_yield_ttm", "book_to_price"):
        if column not in data.columns:
            data[column] = np.nan
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    benchmark, benchmark_meta = _benchmark_features(start=start, end=end, source=source, calendar=calendar)
    if not benchmark.empty:
        data = data.merge(benchmark, on="trade_date", how="left")
    for column in BENCHMARK_FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    universe = _group_snapshot(data, pd.Series(True, index=data.index), "universe")
    data = data.merge(universe, on="trade_date", how="left")
    data = _add_market_regime_features(data)
    data, industry_meta = _add_industry_context(data)
    roles = data.get("peer_role", pd.Series("", index=data.index)).fillna("").astype(str)
    market_reference = _group_snapshot(data, roles.eq("market_reference"), "market_reference")
    if not market_reference.empty:
        data = data.merge(market_reference, on="trade_date", how="left")
    for period in [1, 5, 20]:
        market_column = f"market_reference_mean_ret_{period}"
        if market_column not in data.columns:
            data[market_column] = np.nan
        data[f"excess_vs_universe_ret_{period}"] = data[f"ret_{period}"] - data[f"universe_mean_ret_{period}"]
        data[f"excess_vs_industry_ret_{period}"] = data[f"ret_{period}"] - data[f"industry_mean_ret_{period}"]
        data[f"excess_vs_csi300_ret_{period}"] = data[f"ret_{period}"] - data[f"market_csi300_ret_{period}"]

    by_date = data.groupby("trade_date", group_keys=False)
    ranks = {
        "turnover_rate_daily": "rank_turnover_rate_daily",
        "log_circ_mv": "rank_log_circ_mv",
        "earnings_yield_ttm": "rank_earnings_yield_ttm",
        "book_to_price": "rank_book_to_price",
    }
    for source_column, output_column in ranks.items():
        data[output_column] = by_date[source_column].transform(lambda values: values.rank(pct=True))

    neutral_features = {
        "ret_5": "size_neutral_ret_5",
        "ma_gap_20": "size_neutral_ma_gap_20",
        "volume_ratio_5_20": "size_neutral_volume_ratio_5_20",
        "volatility_20": "size_neutral_volatility_20",
        "log_amount_yuan": "size_neutral_log_amount_yuan",
    }
    for source_column, output_column in neutral_features.items():
        if source_column in data.columns:
            residual = pd.Series(np.nan, index=data.index, dtype=float)
            for _, group in data.groupby("trade_date"):
                residual.loc[group.index] = _size_neutral_residual(group, source_column)
            data[output_column] = residual
        else:
            data[output_column] = np.nan

    data = data.replace([np.inf, -np.inf], np.nan)
    data, factor_engineering_meta = describe_factor_coverage(data)
    feature_coverage = {
        column: round(float(data[column].notna().mean()), 4)
        for column in DAILY_FACTOR_FEATURE_COLUMNS
        if column in data.columns
    }
    warnings = list(benchmark_meta.get("warnings", [])) + list(industry_meta["warnings"])
    return data, {
        "status": "ok",
        "frequency": "daily_k_only",
        "market_benchmarks": benchmark_meta,
        "industry_factor_method": industry_meta["method"],
        "industry_factor_quality": industry_meta,
        "universe_factor_scope": "仅本次比较池的日频统计，不代表全市场完整广度",
        "industry_membership_bias": "当前同行或当前板块成分回看历史，不能冒充历史时点成分快照",
        "size_neutralization": "逐交易日用已提供的同日流通市值对指定因子做线性残差化；字段缺失或不足5只有效股票时留空",
        "market_regime_method": "沪深300近20日收益为正且比较池过半位于MA20之上为偏强；两者反向为偏弱；未同时偏强或偏弱为方向分歧或中性，不代表个股横盘。原始波动率单列，不加权",
        "feature_coverage": feature_coverage,
        "factor_engineering": factor_engineering_meta,
        "warnings": warnings,
    }


__all__ = [
    "BENCHMARK_FEATURE_COLUMNS",
    "DAILY_FACTOR_FEATURE_COLUMNS",
    "enrich_daily_factor_panel",
]
