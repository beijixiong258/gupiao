"""Leak-aware daily-frequency factors shared by stock and board models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.gupiao_yanjiu import akshare_zhilian
from src.ashare.shuju_yuan import CACHE_DIR, _tushare_pro


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
] + BENCHMARK_FEATURE_COLUMNS

_DAILY_BASIC_FIELDS = ["turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv"]
_FACTOR_CACHE_DIR = CACHE_DIR / "daily_factor_cache"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path)
        if "trade_date" in data.columns:
            data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
        return data.dropna(subset=["trade_date"]) if "trade_date" in data.columns else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _write_cache(data: pd.DataFrame, csv_path: Path, meta: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    csv_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cache_covers(meta: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    cached_start = pd.to_datetime(meta.get("requested_start"), errors="coerce")
    cached_end = pd.to_datetime(meta.get("requested_end"), errors="coerce")
    return bool(
        pd.notna(cached_start)
        and pd.notna(cached_end)
        and pd.Timestamp(cached_start).normalize() <= start
        and pd.Timestamp(cached_end).normalize() >= end
    )


def _normalize_daily_basic(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date"] + _DAILY_BASIC_FIELDS)
    data = frame.copy()
    if "ts_code" not in data.columns:
        data["ts_code"] = code
    data["ts_code"] = data["ts_code"].astype(str)
    data["trade_date"] = pd.to_datetime(data.get("trade_date"), errors="coerce").dt.normalize()
    for column in _DAILY_BASIC_FIELDS:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    return (
        data[["ts_code", "trade_date"] + _DAILY_BASIC_FIELDS]
        .dropna(subset=["trade_date"])
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def _historical_daily_basic(
    *,
    codes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    cache_hits = 0
    fetched = 0
    try:
        from src.ashare.riping_cangku import load_daily_basic_from_warehouse

        warehouse_data, warehouse_meta = load_daily_basic_from_warehouse(
            codes,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if (
            not warehouse_data.empty
            and float(warehouse_meta.get("calendar_sync_coverage", 0.0)) >= 0.90
        ):
            combined = pd.concat(
                [
                    _normalize_daily_basic(group, str(code))
                    for code, group in warehouse_data.groupby("ts_code")
                ],
                ignore_index=True,
            )
            return combined, {
                **warehouse_meta,
                "status": "ok",
                "stocks_requested": int(len(set(codes))),
                "stocks_with_rows": int(combined["ts_code"].nunique()),
                "merge_rule": "仅按股票代码和同一交易日精确合并，不向前或向后填充",
                "warnings": [],
            }
    except Exception as exc:
        warnings.append(f"全市场日频仓库估值读取失败，改用逐股缓存：{exc}")
    try:
        pro = _tushare_pro()
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "unavailable",
            "source": "tushare_daily_basic",
            "warnings": warnings + [f"历史日频估值不可用：{exc}"],
            "merge_rule": "仅按股票代码和同一交易日精确合并，不向前或向后填充",
        }

    for code in sorted(set(codes)):
        safe_code = code.replace(".", "_")
        cache_path = _FACTOR_CACHE_DIR / "daily_basic" / f"{safe_code}.csv"
        cached = _normalize_daily_basic(_read_csv(cache_path), code)
        meta = _read_meta(cache_path.with_suffix(".json"))
        data = cached
        if _cache_covers(meta, start, end):
            cache_hits += 1
        else:
            try:
                raw = pro.daily_basic(
                    ts_code=code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    fields="ts_code,trade_date,turnover_rate,pe_ttm,pb,total_mv,circ_mv",
                )
                fresh = _normalize_daily_basic(raw, code)
                data = (
                    pd.concat([cached, fresh], ignore_index=True)
                    .drop_duplicates(["ts_code", "trade_date"], keep="last")
                    .sort_values("trade_date")
                    .reset_index(drop=True)
                )
                _write_cache(
                    data,
                    cache_path,
                    {
                        "source": "tushare_daily_basic",
                        "requested_start": start.strftime("%Y-%m-%d"),
                        "requested_end": end.strftime("%Y-%m-%d"),
                        "rows": int(len(data)),
                    },
                )
                fetched += 1
            except Exception as exc:
                warnings.append(f"{code} 历史日频估值获取失败：{exc}")
        if not data.empty:
            frames.append(data[(data["trade_date"] >= start) & (data["trade_date"] <= end)])

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, {
        "status": "ok" if not combined.empty else "unavailable",
        "source": "tushare_daily_basic",
        "stocks_requested": int(len(set(codes))),
        "stocks_with_rows": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "rows": int(len(combined)),
        "cache_hits": int(cache_hits),
        "network_refreshes": int(fetched),
        "merge_rule": "仅按股票代码和同一交易日精确合并，不向前或向后填充",
        "warnings": warnings,
    }


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
    data["trade_date"] = pd.to_datetime(data.get("trade_date"), errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "close" not in data.columns:
        return pd.DataFrame()
    return (
        data.dropna(subset=["trade_date", "close"])
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def _fetch_one_benchmark(
    *,
    key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    spec = BENCHMARKS[key]
    errors: list[str] = []
    providers = [source] if source in {"tushare", "akshare"} else ["tushare", "akshare"]
    for provider in providers:
        cache_path = _FACTOR_CACHE_DIR / "benchmarks" / f"{key}_{provider}.csv"
        cached = _normalize_index_history(_read_csv(cache_path), tushare=provider == "tushare")
        meta = _read_meta(cache_path.with_suffix(".json"))
        if _cache_covers(meta, start, end) and not cached.empty:
            return cached[(cached["trade_date"] >= start) & (cached["trade_date"] <= end)], f"{provider}_cache", errors
        try:
            if provider == "tushare":
                pro = _tushare_pro()
                raw = pro.index_daily(
                    ts_code=spec["ts_code"],
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )
                fresh = _normalize_index_history(raw, tushare=True)
            else:
                import akshare as ak

                with akshare_zhilian():
                    raw = ak.stock_zh_index_daily(symbol=spec["ak_symbol"])
                fresh = _normalize_index_history(raw, tushare=False)
                fresh = fresh[(fresh["trade_date"] >= start) & (fresh["trade_date"] <= end)]
            if fresh.empty:
                raise RuntimeError("返回空日线")
            data = (
                pd.concat([cached, fresh], ignore_index=True)
                .drop_duplicates("trade_date", keep="last")
                .sort_values("trade_date")
                .reset_index(drop=True)
            )
            _write_cache(
                data,
                cache_path,
                {
                    "source": provider,
                    "benchmark": spec["name"],
                    "requested_start": start.strftime("%Y-%m-%d"),
                    "requested_end": end.strftime("%Y-%m-%d"),
                    "rows": int(len(data)),
                },
            )
            return data[(data["trade_date"] >= start) & (data["trade_date"] <= end)], provider, errors
        except Exception as exc:
            errors.append(f"{spec['name']} {provider} 日K失败：{exc}")
            if not cached.empty:
                overlap = cached[(cached["trade_date"] >= start) & (cached["trade_date"] <= end)]
                if not overlap.empty:
                    return overlap, f"{provider}_stale_cache", errors
    return pd.DataFrame(), "unavailable", errors


def _benchmark_features(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged: pd.DataFrame | None = None
    details: dict[str, Any] = {}
    warnings: list[str] = []
    for key in BENCHMARKS:
        history, provider, errors = _fetch_one_benchmark(
            key=key,
            start=start,
            end=end,
            source=source,
        )
        warnings.extend(errors)
        if history.empty:
            details[key] = {"status": "unavailable", "source": provider, "rows": 0}
            continue
        values = history[["trade_date", "close"]].copy()
        close = pd.to_numeric(values["close"], errors="coerce")
        values[f"market_{key}_ret_1"] = close.pct_change(1, fill_method=None)
        values[f"market_{key}_ret_5"] = close.pct_change(5, fill_method=None)
        values[f"market_{key}_ret_20"] = close.pct_change(20, fill_method=None)
        values[f"market_{key}_volatility_20"] = (
            close.pct_change(fill_method=None).rolling(20, min_periods=20).std() * math.sqrt(252)
        )
        values = values.drop(columns=["close"])
        merged = values if merged is None else merged.merge(values, on="trade_date", how="outer")
        details[key] = {"status": "ok", "source": provider, "rows": int(len(history))}
    return (merged if merged is not None else pd.DataFrame()), {
        "status": "ok" if merged is not None and not merged.empty else "unavailable",
        "benchmarks": details,
        "warnings": warnings,
        "frequency": "daily_k_only",
    }


def _group_snapshot(data: pd.DataFrame, mask: pd.Series, prefix: str) -> pd.DataFrame:
    subset = data.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["trade_date"])
    result = subset.groupby("trade_date", as_index=False).agg(
        **{
            f"{prefix}_mean_ret_1": ("ret_1", "mean"),
            f"{prefix}_mean_ret_5": ("ret_5", "mean"),
            f"{prefix}_mean_ret_20": ("ret_20", "mean"),
            f"{prefix}_dispersion_ret_5": ("ret_5", "std"),
            f"{prefix}_breadth_above_ma20": (
                "ma_gap_20",
                lambda values: values.gt(0).where(values.notna()).mean(),
            ),
            f"{prefix}_breadth_positive_5d": (
                "ret_5",
                lambda values: values.gt(0).where(values.notna()).mean(),
            ),
        }
    )
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
    include_historical_valuation: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add exact-date valuation, benchmark, group and size-neutral daily factors."""
    if panel is None or panel.empty:
        return pd.DataFrame(), {"status": "unavailable", "warnings": ["模型面板为空"]}
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["trade_date"]).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    start = pd.Timestamp(data["trade_date"].min()).normalize()
    end = pd.Timestamp(data["trade_date"].max()).normalize()
    codes = data["ts_code"].astype(str).drop_duplicates().tolist()

    if "amount_yuan" not in data.columns:
        data["amount_yuan"] = np.nan
    data["log_amount_yuan"] = np.log1p(pd.to_numeric(data["amount_yuan"], errors="coerce").clip(lower=0))

    valuation_meta: dict[str, Any]
    if include_historical_valuation:
        daily_basic, valuation_meta = _historical_daily_basic(codes=codes, start=start, end=end)
        if not daily_basic.empty:
            daily_basic = daily_basic.rename(
                columns={column: f"daily_basic_{column}" for column in _DAILY_BASIC_FIELDS}
            )
            data = data.merge(daily_basic, on=["ts_code", "trade_date"], how="left")
    else:
        valuation_meta = {
            "status": "disabled",
            "merge_rule": "仅按股票代码和同一交易日精确合并，不向前或向后填充",
            "warnings": [],
        }
    for column in _DAILY_BASIC_FIELDS:
        exact_column = f"daily_basic_{column}"
        if exact_column not in data.columns:
            data[exact_column] = np.nan
        else:
            data[exact_column] = pd.to_numeric(data[exact_column], errors="coerce")
    data["turnover_rate_daily"] = data["daily_basic_turnover_rate"] / 100.0
    data["log_circ_mv"] = np.log(pd.to_numeric(data["daily_basic_circ_mv"], errors="coerce") * 10000.0).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    data["earnings_yield_ttm"] = (
        1.0 / data["daily_basic_pe_ttm"].where(data["daily_basic_pe_ttm"] > 0)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    data["book_to_price"] = (1.0 / data["daily_basic_pb"].where(data["daily_basic_pb"] > 0)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    benchmark, benchmark_meta = _benchmark_features(start=start, end=end, source=source)
    if not benchmark.empty:
        data = data.merge(benchmark, on="trade_date", how="left")
    for column in BENCHMARK_FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    universe = _group_snapshot(data, pd.Series(True, index=data.index), "universe")
    data = data.merge(universe, on="trade_date", how="left")
    if "peer_role" in data.columns and data["peer_role"].notna().any():
        roles = data["peer_role"].fillna("").astype(str)
        industry_mask = roles.isin(["target", "same_industry"])
        market_reference_mask = roles.eq("market_reference")
        industry_method = "目标股票与当前同行的日频等权截面代理"
    else:
        industry_mask = pd.Series(True, index=data.index)
        market_reference_mask = pd.Series(False, index=data.index)
        industry_method = "当前板块成分股日频等权截面代理"
    industry = _group_snapshot(data, industry_mask, "industry")
    market_reference = _group_snapshot(data, market_reference_mask, "market_reference")
    data = data.merge(industry, on="trade_date", how="left")
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
    feature_coverage = {
        column: round(float(data[column].notna().mean()), 4)
        for column in DAILY_FACTOR_FEATURE_COLUMNS
        if column in data.columns
    }
    warnings = list(valuation_meta.get("warnings", [])) + list(benchmark_meta.get("warnings", []))
    return data, {
        "status": "ok",
        "frequency": "daily_k_only",
        "historical_valuation": valuation_meta,
        "market_benchmarks": benchmark_meta,
        "industry_factor_method": industry_method,
        "industry_membership_bias": "当前同行或当前板块成分回看历史，不能冒充历史时点成分快照",
        "size_neutralization": "逐交易日用历史流通市值对指定因子做线性残差化；不足5只有效股票时留空",
        "feature_coverage": feature_coverage,
        "warnings": warnings,
    }


__all__ = [
    "BENCHMARK_FEATURE_COLUMNS",
    "DAILY_FACTOR_FEATURE_COLUMNS",
    "enrich_daily_factor_panel",
]
