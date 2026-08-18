"""预测侧的同行候选池、横截面快照与历史样本准备。"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.gupiao_yanjiu import (
    guifan_you_xian_shuzhi,
    guolv_wanzheng_jiaoyiri_lishi,
    huoqu_rili_xingqing,
)
from src.ashare.shichang_shuju import huoqu_akshare_hengjiemian, huoqu_tushare_hengjiemian


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if np.isfinite(number) else None


def xuanze_tonghang_yangben(
    *,
    code: str,
    name: str,
    industry: str,
    signal_date: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a liquid current peer universe, preferring the same industry."""
    settings = config.get("dangu", {})
    maximum = max(8, int(settings.get("max_peer_stocks", 20)))
    base_same_industry = int(settings.get("same_industry_stocks", 16))
    min_amount = float(settings.get("min_amount_yuan", 30_000_000))
    signal = pd.Timestamp(signal_date).normalize()
    same_industry_limit = max(4, min(maximum - 1, base_same_industry))
    warnings: list[str] = []
    try:
        universe, as_of, source_warnings, stock_master_meta = huoqu_tushare_hengjiemian(signal)
        source = "tushare"
        warnings.extend(source_warnings)
    except Exception as tushare_exc:
        warnings.append(f"Tushare 同行池失败：{tushare_exc}")
        universe, as_of, source_warnings, stock_master_meta = huoqu_akshare_hengjiemian(signal)
        source = "akshare"
        warnings.extend(source_warnings)

    if universe.empty:
        raise RuntimeError("没有可用的 A 股同行横截面")
    if "name" not in universe.columns:
        universe["name"] = ""
    if "industry" not in universe.columns:
        universe["industry"] = ""
    for column in ["latest_price", "amount_yuan"]:
        if column not in universe.columns:
            universe[column] = np.nan
        universe[column] = pd.to_numeric(universe[column], errors="coerce")
    universe["name"] = universe["name"].fillna("").astype(str)
    universe["industry"] = universe["industry"].fillna("").astype(str)

    target_hit = universe[universe["ts_code"] == code]
    resolved_industry = str(industry or "").strip()
    if not resolved_industry and not target_hit.empty:
        resolved_industry = str(target_hit.iloc[0].get("industry") or "").strip()
    if not name and not target_hit.empty:
        name = str(target_hit.iloc[0].get("name") or "")

    eligible = universe[
        universe["latest_price"].between(2.0, 300.0, inclusive="both")
        & universe["amount_yuan"].ge(min_amount)
    ].copy()
    bad_name = eligible["name"].str.upper().str.contains(r"ST|退", regex=True, na=False)
    new_name = eligible["name"].str.upper().str.startswith(("N", "C"), na=False)
    eligible = eligible[~bad_name & ~new_name]
    if "list_date" in eligible.columns:
        listed = pd.to_datetime(eligible["list_date"].astype(str), errors="coerce")
        eligible = eligible[listed.isna() | ((signal - listed.dt.normalize()).dt.days >= 180)]
    eligible = eligible.sort_values("amount_yuan", ascending=False, na_position="last")

    selected_parts: list[pd.DataFrame] = []
    if resolved_industry:
        industry_rows = eligible[(eligible["industry"] == resolved_industry) & (eligible["ts_code"] != code)]
        selected_parts.append(industry_rows.head(same_industry_limit).assign(peer_role="same_industry"))
    selected_codes = {code}
    for part in selected_parts:
        selected_codes.update(part["ts_code"].astype(str))
    room = maximum - 1 - sum(len(part) for part in selected_parts)
    if room > 0:
        references = eligible[~eligible["ts_code"].isin(selected_codes)].head(room)
        selected_parts.append(references.assign(peer_role="market_reference"))

    if not target_hit.empty:
        target_row = target_hit.head(1).copy()
    else:
        target_row = pd.DataFrame([{"ts_code": code, "name": name, "industry": resolved_industry}])
    target_row["peer_role"] = "target"
    selected = pd.concat([target_row] + selected_parts, ignore_index=True, sort=False)
    selected = selected.drop_duplicates("ts_code", keep="first").head(maximum).reset_index(drop=True)
    selected["name"] = selected["name"].fillna("").astype(str)
    selected.loc[selected["ts_code"] == code, "name"] = name or selected.loc[selected["ts_code"] == code, "name"]
    role_counts = Counter(selected["peer_role"].astype(str))
    return selected, {
        "source": source,
        "as_of": as_of,
        "target_industry": resolved_industry or None,
        "selected_stocks": int(len(selected)),
        "role_counts": dict(role_counts),
        "minimum_amount_yuan": min_amount,
        "configured_peer_limit": maximum,
        "applied_peer_limit": maximum,
        "persistence": "none",
        "stock_master_snapshot": stock_master_meta,
        "selection_method": "当前同行优先，再用全市场高流动性股票补足；目标股票始终保留",
        "known_bias": "同行和行业标签来自本次远端请求的当前资料，不能冒充历史时点成分快照",
        "warnings": warnings,
    }


def goujian_tonghang_kuaizhao(peer_table: pd.DataFrame, code: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target = peer_table[peer_table["ts_code"] == code]
    if target.empty:
        return result
    target_row = target.iloc[0]
    mappings = {
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "turnover_rate": "turnover_rate_pct",
        "amount_yuan": "amount_yuan",
        "circulating_market_value_yuan": "circulating_market_value_yuan",
    }
    for source_column, output_column in mappings.items():
        if source_column not in peer_table.columns:
            continue
        values = pd.to_numeric(peer_table[source_column], errors="coerce")
        value = guifan_you_xian_shuzhi(target_row.get(source_column), 4)
        valid = values.dropna()
        if value is None or len(valid) < 5:
            result[output_column] = {"value": value, "peer_percentile": None}
            continue
        if source_column == "pe_ttm":
            valid = valid[valid > 0]
            if value <= 0 or len(valid) < 5:
                result[output_column] = {"value": value, "peer_percentile": None}
                continue
        percentile = float((valid <= value).mean())
        result[output_column] = {
            "value": value,
            "peer_percentile": round(percentile, 4),
            "peer_count": int(len(valid)),
        }
    return result


def huoqu_tonghang_lishi(
    *,
    peer_table: pd.DataFrame,
    target_code: str,
    target_history: pd.DataFrame,
    target_source: str,
    target_adjustment: str,
    signal_date: str,
    source: str,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    settings = config.get("dangu", {})
    minimum_rows = int(settings.get("minimum_history_rows", 180))
    pause_seconds = float(settings.get("request_pause_seconds", 0.08))
    signal = pd.Timestamp(signal_date).normalize()
    start = pd.to_datetime(target_history["trade_date"], errors="coerce").min()
    if pd.isna(start):
        raise RuntimeError("目标股票历史行情缺少有效日期")
    histories: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    sources: Counter[str] = Counter()

    for _, row in peer_table.iterrows():
        peer_code = str(row["ts_code"])
        peer_name = str(row.get("name") or "")
        if peer_code == target_code:
            data = target_history.copy()
            data_source = target_source
            adjustment = target_adjustment
            source_warnings: list[str] = []
            source_errors: list[str] = []
        else:
            fetched = huoqu_rili_xingqing(
                peer_code,
                start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                end_date=signal.strftime("%Y%m%d"),
                source=source,
            )
            if source == "auto" and fetched.adjustment == "raw_unadjusted":
                ak_fallback = huoqu_rili_xingqing(
                    peer_code,
                    start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                    end_date=signal.strftime("%Y%m%d"),
                    source="akshare",
                )
                if not ak_fallback.data.empty and ak_fallback.adjustment != "raw_unadjusted":
                    fetched = ak_fallback
                    warnings.append(f"{peer_code}: 已单独改用 AKShare 前复权行情")
            data, completion_warnings = guolv_wanzheng_jiaoyiri_lishi(
                fetched.data,
                latest_completed_date=signal,
            )
            data_source = fetched.source
            adjustment = fetched.adjustment
            source_warnings = list(fetched.warnings) + completion_warnings
            source_errors = list(fetched.errors)
        data = data[pd.to_datetime(data.get("trade_date"), errors="coerce").dt.normalize() <= signal].copy()
        if data.empty:
            errors.append(f"{peer_code} {peer_name}: 没有可用完整日线；{'；'.join(source_errors)}")
        elif adjustment == "raw_unadjusted":
            errors.append(f"{peer_code} {peer_name}: 未复权行情不进入单股预测模型")
        elif len(data) < minimum_rows:
            errors.append(f"{peer_code} {peer_name}: 有效日线 {len(data)} 行，少于 {minimum_rows}")
        else:
            data["data_source"] = data_source
            data["adjustment"] = adjustment
            data["peer_role"] = str(row.get("peer_role") or "market_reference")
            histories[peer_code] = data
            names[peer_code] = peer_name
            sources[str(data_source)] += 1
            warnings.extend(f"{peer_code}: {value}" for value in source_warnings)
        if peer_code != target_code and pause_seconds > 0:
            time.sleep(pause_seconds)
    return histories, names, {
        "usable_stocks": int(len(histories)),
        "history_sources": dict(sources),
        "minimum_history_rows": minimum_rows,
        "persistence": "none",
        "errors": errors,
        "warnings": warnings,
    }


__all__ = [
    "xuanze_tonghang_yangben",
    "goujian_tonghang_kuaizhao",
    "huoqu_tonghang_lishi",
]
