"""Sector-constrained A-share selection with T+1/T+2/T+3 predictions."""

from __future__ import annotations

import json
import hashlib
import math
import time
from collections import Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.ashare.chengben_huadian import CostScenario
from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    _completed_market_history,
    _json_value,
    _latest_expected_market_date,
    _market_data_freshness,
    _round_optional,
    akshare_zhilian,
    biaozhunhua_daima,
    huoqu_rili_xingqing,
    jiazai_lianghua_peizhi,
    jisuan_tezheng_biao,
    shi_a_gu,
)
from src.ashare.jiaoyi_zhixing import (
    _apply_cost,
    _load_cost_assumption,
    _position_for_budget as _position_for_budget,
    _price_limit_bounds,
    _round_price_tick,
    _roundtrip_cost,
    _stock_roundtrip_cost,
)
from src.ashare.moxing_pinggu import regression_baseline_metrics, signal_evidence_gate
from src.ashare.shuju_zhiliang import (
    build_data_health,
    classify_failure,
    filter_panel_by_membership_snapshots,
)
from src.ashare.shuju_yuan import _latest_tushare_daily, _limit_rate, _load_or_fetch_stock_basic, _tushare_pro
from src.ashare.riping_yinzi import DAILY_FACTOR_FEATURE_COLUMNS, enrich_daily_factor_panel
from src.providers.llm import _ensure_dotenv
from src.tools.path_utils import safe_run_dir


BOARD_FEATURE_COLUMNS = FEATURE_COLUMNS + DAILY_FACTOR_FEATURE_COLUMNS


def _board_name_column(frame: pd.DataFrame) -> str | None:
    for column in ["板块名称", "行业名称", "概念名称", "名称", "name"]:
        if column in frame.columns:
            return column
    return next((str(column) for column in frame.columns if "名称" in str(column)), None)


def _candidate_board_names(ak: Any, query: str, kinds: list[str]) -> tuple[list[tuple[float, str, str]], list[str]]:
    candidates: list[tuple[float, str, str]] = []
    errors: list[str] = []
    for kind in kinds:
        try:
            names = ak.stock_board_industry_name_em() if kind == "hangye" else ak.stock_board_concept_name_em()
            if names is None or names.empty:
                raise RuntimeError("板块列表为空")
            name_column = _board_name_column(names)
            if not name_column:
                raise RuntimeError("板块列表缺少名称列")
            for value in names[name_column].dropna().astype(str):
                if value == query:
                    similarity = 1.0
                elif query in value or value in query:
                    similarity = 0.88
                else:
                    similarity = SequenceMatcher(None, query, value).ratio()
                if similarity >= 0.72:
                    candidates.append((similarity, kind, value))
        except Exception as exc:
            errors.append(f"{kind}板块列表失败：{exc}")

    candidates.sort(key=lambda item: (item[0], item[1] == "hangye"), reverse=True)
    if not candidates:
        errors.append(f"没有找到与“{query}”足够相似的板块（最低相似度0.72）")
        return [], errors
    top_score, _, top_name = candidates[0]
    close_names = list(
        dict.fromkeys(
            name
            for score, _, name in candidates[1:]
            if name != top_name and top_score < 1.0 and top_score - score < 0.03
        )
    )
    if close_names:
        errors.append(f"板块名称“{query}”存在歧义，候选：{top_name}、{'、'.join(close_names[:4])}")
        return [], errors
    return candidates, errors


def _normalize_constituents(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "代码": "ts_code",
        "名称": "name",
        "最新价": "latest_price",
        "涨跌幅": "pct_chg",
        "成交量": "volume",
        "成交额": "amount_yuan",
        "换手率": "turnover_rate",
        "市盈率-动态": "pe_dynamic",
        "市净率": "pb",
        "code": "ts_code",
        "name": "name",
        "trade": "latest_price",
        "changepercent": "pct_chg",
        "volume": "volume",
        "amount": "amount_yuan",
        "turnoverratio": "turnover_rate",
        "per": "pe_dynamic",
    }
    data = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    if "ts_code" not in data.columns or "name" not in data.columns:
        raise ValueError("板块成分接口缺少代码或名称列")
    data["ts_code"] = data["ts_code"].astype(str).str.zfill(6)
    data = data[data["ts_code"].map(shi_a_gu)].copy()
    data["ts_code"] = data["ts_code"].map(biaozhunhua_daima)
    data["name"] = data["name"].fillna("").astype(str)
    for column in ["latest_price", "pct_chg", "volume", "amount_yuan", "turnover_rate", "pe_dynamic", "pb"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def _best_name(query: str, values: list[str]) -> tuple[str, float] | None:
    candidates: list[tuple[float, str]] = []
    for value in values:
        if value == query:
            similarity = 1.0
        elif query in value or value in query:
            similarity = 0.88
        else:
            similarity = SequenceMatcher(None, query, value).ratio()
        candidates.append((similarity, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    similarity, value = candidates[0]
    if similarity < 0.72:
        return None
    ambiguous = [
        candidate
        for score, candidate in candidates[1:]
        if candidate != value and similarity < 1.0 and similarity - score < 0.03
    ]
    if ambiguous:
        raise ValueError(f"名称“{query}”存在歧义，候选：{value}、{'、'.join(ambiguous[:4])}")
    return value, float(similarity)


def _tushare_industry_constituents(query: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    pro = _tushare_pro()
    basic = _load_or_fetch_stock_basic(pro, {})
    if basic.empty or "industry" not in basic.columns:
        raise RuntimeError("Tushare stock_basic 没有行业字段")
    values = sorted(value for value in basic["industry"].dropna().astype(str).unique() if value)
    resolved = _best_name(query, values)
    if not resolved:
        raise RuntimeError(f"Tushare 行业列表中没有与“{query}”匹配的行业")
    industry, similarity = resolved
    members = basic[basic["industry"].fillna("").astype(str) == industry].copy()
    if members.empty:
        raise RuntimeError("Tushare 行业成分为空")

    warnings: list[str] = []
    try:
        _, daily = _latest_tushare_daily(pro, None)
        daily = daily.rename(
            columns={
                "close": "latest_price",
                "vol": "volume",
                "pct_chg": "pct_chg",
            }
        )
        daily["ts_code"] = daily["ts_code"].map(biaozhunhua_daima)
        if "amount" in daily.columns:
            daily["amount_yuan"] = pd.to_numeric(daily["amount"], errors="coerce") * 1000.0
        columns = [
            column
            for column in ["ts_code", "latest_price", "volume", "amount_yuan", "pct_chg"]
            if column in daily.columns
        ]
        members = members.merge(daily[columns].drop_duplicates("ts_code"), on="ts_code", how="left")
    except Exception as exc:
        warnings.append(f"Tushare 最新交易日快照失败，价格和成交额过滤可能无法使用：{exc}")
    return _normalize_constituents(members), {
        "requested_name": query,
        "resolved_name": industry,
        "board_type": "hangye",
        "name_similarity": round(similarity, 4),
        "constituent_source": "tushare_stock_basic",
        "warnings": warnings,
    }


def _sina_constituents(ak: Any, query: str, kinds: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates: list[tuple[float, str, str, str]] = []
    errors: list[str] = []
    indicators = []
    if "hangye" in kinds:
        indicators.extend([("hangye", "行业"), ("hangye", "新浪行业")])
    if "gainian" in kinds:
        indicators.append(("gainian", "概念"))
    for kind, indicator in indicators:
        try:
            table = ak.stock_sector_spot(indicator=indicator)
            if table is None or table.empty or not {"label", "板块"}.issubset(table.columns):
                raise RuntimeError("板块列表为空或字段不完整")
            for _, row in table.iterrows():
                name = str(row["板块"])
                resolved = _best_name(query, [name])
                if resolved:
                    _, similarity = resolved
                    candidates.append((similarity, kind, name, str(row["label"])))
        except Exception as exc:
            errors.append(f"新浪{indicator}列表失败：{exc}")
    ordered_candidates = sorted(candidates, reverse=True)
    if ordered_candidates:
        top_score, _, top_name, _ = ordered_candidates[0]
        close_names = list(
            dict.fromkeys(
                name
                for score, _, name, _ in ordered_candidates[1:]
                if name != top_name and top_score < 1.0 and top_score - score < 0.03
            )
        )
        if close_names:
            raise RuntimeError(f"板块名称“{query}”存在歧义，候选：{top_name}、{'、'.join(close_names[:4])}")
    for similarity, kind, name, label in ordered_candidates:
        try:
            raw = ak.stock_sector_detail(sector=label)
            data = _normalize_constituents(raw)
            if data.empty:
                raise RuntimeError("成分股为空")
            return data, {
                "requested_name": query,
                "resolved_name": name,
                "board_type": kind,
                "name_similarity": round(float(similarity), 4),
                "constituent_source": "akshare_sina",
                "warnings": errors,
            }
        except Exception as exc:
            errors.append(f"新浪 {kind}/{name} 成分股失败：{exc}")
    raise RuntimeError("；".join(errors) or f"新浪板块中找不到：{query}")


def huoqu_bankuai_chengfen(
    bankuai: str,
    *,
    bankuai_leixing: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve an Eastmoney industry/concept board and return its A-share constituents."""
    query = str(bankuai).strip()
    if not query:
        raise ValueError("bankuai 不能为空")
    board_type = bankuai_leixing.strip().lower()
    aliases = {"industry": "hangye", "concept": "gainian", "行业": "hangye", "概念": "gainian"}
    board_type = aliases.get(board_type, board_type)
    if board_type not in {"auto", "hangye", "gainian"}:
        raise ValueError("bankuai_leixing 必须是 auto、hangye 或 gainian")

    import akshare as ak

    kinds = ["hangye", "gainian"] if board_type == "auto" else [board_type]
    errors: list[str] = []
    if "hangye" in kinds:
        try:
            return _tushare_industry_constituents(query)
        except Exception as exc:
            errors.append(f"Tushare 行业成分失败：{exc}")

    with akshare_zhilian():
        try:
            data, meta = _sina_constituents(ak, query, kinds)
            meta["warnings"] = errors + list(meta.get("warnings", []))
            return data, meta
        except Exception as exc:
            errors.append(f"新浪板块成分失败：{exc}")

        candidates, eastmoney_errors = _candidate_board_names(ak, query, kinds)
        errors.extend(eastmoney_errors)
        tried: set[tuple[str, str]] = set()
        for similarity, kind, resolved_name in candidates[:8]:
            key = (kind, resolved_name)
            if key in tried:
                continue
            tried.add(key)
            try:
                raw = (
                    ak.stock_board_industry_cons_em(symbol=resolved_name)
                    if kind == "hangye"
                    else ak.stock_board_concept_cons_em(symbol=resolved_name)
                )
                data = _normalize_constituents(raw)
                if data.empty:
                    raise RuntimeError("成分股为空")
                return data, {
                    "requested_name": query,
                    "resolved_name": resolved_name,
                    "board_type": kind,
                    "name_similarity": round(float(similarity), 4),
                    "constituent_source": "akshare_eastmoney",
                    "warnings": errors,
                }
            except Exception as exc:
                errors.append(f"{kind}/{resolved_name} 成分股失败：{exc}")
    raise RuntimeError("无法取得板块成分股：" + "；".join(errors[-10:]))


def _exclude_name(name: str, keywords: list[str]) -> bool:
    upper = str(name).upper()
    return any(str(keyword).upper() in upper for keyword in keywords)


def _filter_constituents(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    settings = config["guolv"]
    rejected: list[dict[str, Any]] = []
    accepted_rows: list[pd.Series] = []
    keywords = [str(value) for value in settings.get("exclude_name_keywords", [])]
    for _, row in frame.iterrows():
        code = str(row["ts_code"])
        name = str(row.get("name", ""))
        reason = ""
        price = _round_optional(row.get("latest_price"))
        amount = _round_optional(row.get("amount_yuan"))
        pct_chg = _round_optional(row.get("pct_chg"))
        if _exclude_name(name, keywords):
            reason = "名称包含新股、ST 或退市风险标记"
        elif price is None:
            reason = "缺少最新价格，无法应用价格过滤"
        elif price < float(settings.get("min_price", 2.0)):
            reason = "价格低于配置下限"
        elif price > float(settings.get("max_price", 300.0)):
            reason = "价格高于配置上限"
        elif amount is None:
            reason = "缺少最新成交额，无法应用流动性过滤"
        elif amount < float(settings.get("min_amount_yuan", 50_000_000)):
            reason = "最新成交额低于流动性下限"
        elif bool(settings.get("exclude_latest_limit_up", True)) and pct_chg is not None:
            limit_pct = _limit_rate(code, name) * 100.0
            if pct_chg >= limit_pct - 0.15:
                reason = "最新交易日接近或达到涨停，次日可执行性较差"
        if reason:
            rejected.append({"ts_code": code, "name": name, "reason": reason})
        else:
            accepted_rows.append(row)

    accepted = pd.DataFrame(accepted_rows, columns=frame.columns)
    if accepted.empty:
        return accepted, rejected
    if "amount_yuan" in accepted.columns:
        accepted = accepted.sort_values("amount_yuan", ascending=False, na_position="last")
    return accepted.reset_index(drop=True), rejected


def _fetch_histories(
    constituents: pd.DataFrame,
    *,
    source: str,
    history_calendar_days: int,
    minimum_rows: int,
    max_stocks: int,
    pause_seconds: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[str], list[str]]:
    end = _latest_expected_market_date().date()
    start = end - timedelta(days=history_calendar_days)
    histories: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    selected_constituents = constituents.head(max_stocks).copy()
    try:
        from src.ashare.riping_cangku import load_qfq_histories_from_warehouse

        warehouse_histories, warehouse_batch = load_qfq_histories_from_warehouse(
            selected_constituents["ts_code"].astype(str),
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            minimum_rows=minimum_rows,
        )
        name_map = selected_constituents.set_index("ts_code")["name"].astype(str).to_dict()
        for code, warehouse_data in warehouse_histories.items():
            adjustment = str(warehouse_data.attrs.get("adjustment") or "qfq_by_warehouse")
            data = warehouse_data.copy()
            data["data_source"] = str(warehouse_batch.get("source") or "tushare_daily_warehouse_batch")
            data["adjustment"] = adjustment
            histories[code] = data
            names[code] = str(name_map.get(code) or "")
        if warehouse_histories:
            warnings.append(
                f"本地仓库一次批量读取 {len(warehouse_histories)} 只股票，剩余股票才访问外部行情源"
            )
    except Exception as exc:
        warnings.append(f"本地仓库批量读取不可用，改为逐只读取：{exc}")
    remaining_constituents = selected_constituents[
        ~selected_constituents["ts_code"].astype(str).isin(histories)
    ]
    for _, row in remaining_constituents.iterrows():
        code = str(row["ts_code"])
        name = str(row.get("name", ""))
        result = huoqu_rili_xingqing(
            code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            source=source,
            use_cache=True,
        )
        if source == "auto" and result.adjustment == "raw_unadjusted":
            fallback = huoqu_rili_xingqing(
                code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                source="akshare",
                use_cache=True,
            )
            if not fallback.data.empty and fallback.adjustment != "raw_unadjusted":
                warnings.append(f"{code}: Tushare 复权不可用，已对该股票单独改用 AKShare 前复权日线")
                result = fallback
        completed_data, completion_warnings = _completed_market_history(result.data)
        warnings.extend(f"{code}: {item}" for item in completion_warnings)
        if completed_data.empty:
            errors.append(f"{code} {name}: " + "；".join(result.errors))
        elif result.adjustment == "raw_unadjusted":
            errors.append(f"{code} {name}: 板块模型拒绝混入未复权日线")
        elif len(completed_data) < minimum_rows:
            errors.append(f"{code} {name}: 已完成的有效日线只有 {len(completed_data)} 行，少于 {minimum_rows}")
        else:
            data = completed_data.copy()
            data["data_source"] = result.source
            data["adjustment"] = result.adjustment
            histories[code] = data
            names[code] = name
            warnings.extend(f"{code}: {item}" for item in result.warnings)
        if pause_seconds > 0 and "warehouse" not in str(result.source):
            time.sleep(pause_seconds)
    return histories, names, errors, warnings


def _historical_limit_rates(ts_code: str) -> tuple[float, ...]:
    """Return plausible historical price-limit rates without using today's stock name."""
    normalized = biaozhunhua_daima(ts_code)
    digits = normalized.split(".")[0]
    if normalized.endswith(".BJ"):
        return (0.30,)
    if digits.startswith(("688", "689")):
        return (0.20,)
    if digits.startswith(("300", "301")):
        return (0.10, 0.20)
    return (0.05, 0.10)


def _one_price_limit_session(
    prices: pd.DataFrame,
    session_dates: pd.Series,
    *,
    ts_code: str,
    direction: str,
) -> pd.Series:
    """Identify historical one-price limit sessions from their observed bars."""
    high = session_dates.map(pd.to_numeric(prices["high"], errors="coerce"))
    low = session_dates.map(pd.to_numeric(prices["low"], errors="coerce"))
    close = session_dates.map(pd.to_numeric(prices["close"], errors="coerce"))
    if "pct_chg" in prices.columns:
        returns_by_date = pd.to_numeric(prices["pct_chg"], errors="coerce") / 100.0
    else:
        if "pre_close" in prices.columns:
            pre_close = pd.to_numeric(prices["pre_close"], errors="coerce")
        else:
            pre_close = pd.Series(np.nan, index=prices.index, dtype=float)
        pre_close = pre_close.where(pre_close.gt(0), pd.to_numeric(prices["close"], errors="coerce").shift(1))
        returns_by_date = pd.to_numeric(prices["close"], errors="coerce") / pre_close - 1.0
    session_return = session_dates.map(returns_by_date)
    price_tolerance = close.abs().mul(0.0002).clip(lower=0.005)
    one_price = (
        high.notna()
        & low.notna()
        & close.notna()
        & (high.sub(low).abs() <= price_tolerance)
        & (high.sub(close).abs() <= price_tolerance)
        & (low.sub(close).abs() <= price_tolerance)
    )
    limited_session_by_date = pd.Series(np.arange(len(prices)) >= 5, index=prices.index)
    limited_session = session_dates.map(limited_session_by_date).eq(True)
    sign = 1.0 if direction == "up" else -1.0
    at_supported_limit = pd.Series(False, index=session_dates.index, dtype=bool)
    for rate in _historical_limit_rates(ts_code):
        at_supported_limit |= session_return.sub(sign * rate).abs() <= 0.0015
    return one_price & limited_session & at_supported_limit


def goujian_moxing_shuju(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str],
    horizons: list[int],
) -> pd.DataFrame:
    """Build executable labels on a shared market-date calendar.

    A signal is produced after the close of date T.  The assumed entry is the
    next market session's open.  ``T+1`` therefore means the first *sellable*
    close after that entry (the second market session after the signal), with
    ``T+2`` and ``T+3`` following on subsequent market sessions.  A suspended
    stock has no label when it lacks a quote on the required common-market
    entry or exit date; its next observed bar is never silently treated as the
    next market session.
    """
    market_dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for history in histories.values()
            if "trade_date" in history.columns
            for value in pd.to_datetime(history["trade_date"], errors="coerce").dropna()
        }
    )
    market_position = {value: index for index, value in enumerate(market_dates)}
    frames: list[pd.DataFrame] = []
    for code, history in histories.items():
        features = jisuan_tezheng_biao(history)
        features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce").dt.normalize()
        price_columns = ["trade_date", "open", "high", "low", "close"]
        price_columns.extend(column for column in ["pre_close", "pct_chg"] if column in features.columns)
        prices = (
            features[price_columns]
            .dropna(subset=["trade_date"])
            .drop_duplicates("trade_date", keep="last")
            .set_index("trade_date")
        )
        features["ts_code"] = code
        features["name"] = names.get(code, "")
        for horizon in horizons:
            future_dates: list[Any] = []
            entry_dates: list[Any] = []
            exit_dates: list[Any] = []
            for signal_date in features["trade_date"]:
                position = market_position.get(pd.Timestamp(signal_date)) if pd.notna(signal_date) else None
                future_index = position + int(horizon) if position is not None else len(market_dates)
                entry_index = position + 1 if position is not None else len(market_dates)
                exit_index = entry_index + int(horizon)
                future_dates.append(market_dates[future_index] if future_index < len(market_dates) else pd.NaT)
                entry_dates.append(market_dates[entry_index] if entry_index < len(market_dates) else pd.NaT)
                exit_dates.append(market_dates[exit_index] if exit_index < len(market_dates) else pd.NaT)

            future_series = pd.Series(future_dates, index=features.index, dtype="datetime64[ns]")
            entry_series = pd.Series(entry_dates, index=features.index, dtype="datetime64[ns]")
            exit_series = pd.Series(exit_dates, index=features.index, dtype="datetime64[ns]")
            future_close = future_series.map(pd.to_numeric(prices["close"], errors="coerce"))
            features[f"future_date_t{horizon}"] = future_series
            features[f"future_close_t{horizon}"] = future_close
            features[f"future_return_t{horizon}"] = (
                future_close / pd.to_numeric(features["close"], errors="coerce") - 1.0
            )
            entry_open = entry_series.map(pd.to_numeric(prices["open"], errors="coerce"))
            blocked_limit_up = _one_price_limit_session(
                prices,
                entry_series,
                ts_code=code,
                direction="up",
            )
            entry_open = entry_open.mask(blocked_limit_up)
            exit_close = exit_series.map(pd.to_numeric(prices["close"], errors="coerce"))
            blocked_limit_down = _one_price_limit_session(
                prices,
                exit_series,
                ts_code=code,
                direction="down",
            )
            exit_close = exit_close.mask(blocked_limit_down)
            features[f"entry_date_t{horizon}"] = entry_series
            features[f"entry_open_t{horizon}"] = entry_open
            features[f"entry_blocked_limit_up_t{horizon}"] = blocked_limit_up
            features[f"target_date_t{horizon}"] = exit_series
            features[f"target_close_t{horizon}"] = exit_close
            features[f"exit_blocked_limit_down_t{horizon}"] = blocked_limit_down
            features[f"target_t{horizon}"] = exit_close / entry_open - 1.0
        frames.append(features)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _daily_rank_ic(dates: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "actual": actual, "predicted": predicted})
    values: list[float] = []
    for _, group in frame.groupby("date"):
        if len(group) < 5 or group["actual"].nunique() < 2 or group["predicted"].nunique() < 2:
            continue
        value = spearmanr(group["actual"], group["predicted"]).statistic
        if math.isfinite(float(value)):
            values.append(float(value))
    if values:
        return float(np.mean(values)), len(values)
    if len(frame) >= 5 and frame["actual"].nunique() >= 2 and frame["predicted"].nunique() >= 2:
        value = float(spearmanr(frame["actual"], frame["predicted"]).statistic)
        return (value if math.isfinite(value) else 0.0), 1
    return 0.0, 0


def _quality_score(
    *,
    train_count: int,
    direction_accuracy: float,
    rank_ic: float,
    skill_vs_baseline: float,
) -> float:
    sample = min(1.0, math.sqrt(max(train_count, 0) / 2000.0))
    direction = float(np.clip((direction_accuracy - 0.5) / 0.12, 0.0, 1.0))
    rank = float(np.clip(rank_ic / 0.10, 0.0, 1.0))
    skill = float(np.clip(skill_vs_baseline / 0.10, 0.0, 1.0))
    return float(np.clip(sample * (0.45 * direction + 0.35 * rank + 0.20 * skill), 0.0, 1.0))


def _quality_label(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.40:
        return "medium"
    return "low"


class _TrainingQuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to bounds learned only from the active training window."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, values: Any, _target: Any = None) -> "_TrainingQuantileClipper":
        array = np.asarray(values, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper_quantile, axis=0)
        return self

    def transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.minimum(np.maximum(array, self.lower_bounds_), self.upper_bounds_)


def _feature_clipper(model_config: dict[str, Any]) -> _TrainingQuantileClipper:
    quantiles = model_config.get("feature_winsor_quantiles", [0.01, 0.99])
    return _TrainingQuantileClipper(float(quantiles[0]), float(quantiles[1]))


def _build_model_pipeline(model_config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=float(model_config.get("learning_rate", 0.05)),
                    max_iter=int(model_config.get("max_iter", 180)),
                    max_leaf_nodes=int(model_config.get("max_leaf_nodes", 15)),
                    max_depth=int(model_config.get("max_depth", 4)),
                    min_samples_leaf=int(model_config.get("min_samples_leaf", 30)),
                    l2_regularization=float(model_config.get("l2_regularization", 1.0)),
                    random_state=int(model_config.get("random_state", 42)),
                ),
            ),
        ]
    )


def _build_linear_model_pipeline(model_config: dict[str, Any]) -> Pipeline:
    """Build the regularized linear component used to offset tree-model bias."""
    return Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            ("model", Ridge(alpha=float(model_config.get("ridge_alpha", 10.0)))),
        ]
    )


def _fit_model_components(
    *,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    predict_features: pd.DataFrame,
    model_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    tree = _build_model_pipeline(model_config)
    linear = _build_linear_model_pipeline(model_config)
    tree.fit(train_features, train_target)
    linear.fit(train_features, train_target)
    return (
        np.asarray(tree.predict(predict_features), dtype=float),
        np.asarray(linear.predict(predict_features), dtype=float),
    )


def _select_stable_features(
    frame: pd.DataFrame,
    candidate_features: list[str],
    target_column: str,
    model_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Select factors using only the supplied training window."""
    enabled = bool(model_config.get("factor_stability_enabled", True))
    slices = max(2, int(model_config.get("factor_stability_slices", 3)))
    minimum_valid_slices = max(1, int(model_config.get("factor_min_valid_slices", 2)))
    minimum_sign_agreement = float(model_config.get("factor_min_sign_agreement", 0.67))
    minimum_abs_ic = float(model_config.get("factor_min_abs_mean_rank_ic", 0.005))
    minimum_features = max(1, int(model_config.get("factor_min_features", 12)))
    coverage_threshold = float(model_config.get("min_feature_coverage", 0.20))
    coverage = frame[candidate_features].notna().mean()
    eligible = [
        feature
        for feature in candidate_features
        if float(coverage.get(feature, 0.0)) >= coverage_threshold
    ]
    if not enabled:
        return eligible, {
            "status": "disabled",
            "selected_features": eligible,
            "selection_scope": "training_window_only",
        }

    dates = [pd.Timestamp(value) for value in sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique())]
    date_slices = [list(values) for values in np.array_split(np.asarray(dates, dtype=object), slices) if len(values)]
    diagnostics: dict[str, Any] = {}
    ranked: list[tuple[tuple[float, float, float], str]] = []
    selected: list[str] = []
    for feature in eligible:
        values: list[float] = []
        for date_slice in date_slices:
            part = frame[pd.to_datetime(frame["trade_date"]).isin(date_slice)]
            x = pd.to_numeric(part[feature], errors="coerce")
            y = pd.to_numeric(part[target_column], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 30 or int(x[valid].nunique()) < 2 or int(y[valid].nunique()) < 2:
                continue
            correlation = float(spearmanr(x[valid], y[valid]).statistic)
            if math.isfinite(correlation):
                values.append(correlation)
        mean_ic = float(np.mean(values)) if values else 0.0
        sign_agreement = (
            max(sum(value > 0 for value in values), sum(value < 0 for value in values)) / len(values)
            if values
            else 0.0
        )
        is_stable = bool(
            len(values) >= minimum_valid_slices
            and abs(mean_ic) >= minimum_abs_ic
            and sign_agreement >= minimum_sign_agreement
        )
        diagnostics[feature] = {
            "coverage": round(float(coverage[feature]), 4),
            "slice_rank_ic": [round(value, 6) for value in values],
            "mean_rank_ic": round(mean_ic, 6),
            "sign_agreement": round(float(sign_agreement), 4),
            "stable": is_stable,
        }
        ranked.append(((float(is_stable), abs(mean_ic), float(coverage[feature])), feature))
        if is_stable:
            selected.append(feature)
    fallback_used = False
    if len(selected) < min(minimum_features, len(eligible)):
        fallback_used = True
        for _, feature in sorted(ranked, reverse=True):
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= min(minimum_features, len(eligible)):
                break
    selected = [feature for feature in candidate_features if feature in set(selected)]
    return selected, {
        "status": "ok" if selected else "no_eligible_feature",
        "selection_scope": "training_window_only",
        "slices": int(len(date_slices)),
        "candidate_count": int(len(candidate_features)),
        "coverage_eligible_count": int(len(eligible)),
        "selected_count": int(len(selected)),
        "selected_features": selected,
        "fallback_to_strongest_factors": fallback_used,
        "thresholds": {
            "minimum_coverage": coverage_threshold,
            "minimum_valid_slices": minimum_valid_slices,
            "minimum_sign_agreement": minimum_sign_agreement,
            "minimum_abs_mean_rank_ic": minimum_abs_ic,
            "minimum_features": minimum_features,
        },
        "factor_diagnostics": diagnostics,
    }


def _fit_direction_probabilities(
    *,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    predict_features: pd.DataFrame,
    model_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = (np.asarray(train_target, dtype=float) > 0).astype(int)
    if len(np.unique(labels)) < 2:
        probability = float(labels[0]) if len(labels) else 0.5
        return np.full(len(predict_features), probability, dtype=float), {
            "status": "single_training_class",
            "training_positive_rate": probability,
        }
    model = Pipeline(
        [
            ("training_window_winsorizer", _feature_clipper(model_config)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=float(model_config.get("direction_logistic_c", 0.5)),
                    max_iter=int(model_config.get("direction_logistic_max_iter", 500)),
                    random_state=int(model_config.get("random_state", 42)),
                ),
            ),
        ]
    )
    try:
        model.fit(train_features, labels)
        probability = np.asarray(model.predict_proba(predict_features)[:, 1], dtype=float)
        return probability, {
            "status": "ok",
            "training_samples": int(len(labels)),
            "training_positive_rate": round(float(np.mean(labels)), 6),
            "model": "winsorized_robust_scaled_logistic_regression",
        }
    except Exception as exc:
        probability = float(np.mean(labels))
        return np.full(len(predict_features), probability, dtype=float), {
            "status": "fallback_to_training_base_rate",
            "training_samples": int(len(labels)),
            "training_positive_rate": round(probability, 6),
            "error": str(exc),
        }


def _probability_reliability_bins(actual: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = (np.asarray(actual, dtype=float) > 0).astype(int)
    probabilities = np.asarray(probability, dtype=float)
    for lower, upper in zip(np.linspace(0.0, 0.8, 5), np.linspace(0.2, 1.0, 5)):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if not mask.any():
            continue
        rows.append(
            {
                "probability_bin": [round(float(lower), 2), round(float(upper), 2)],
                "samples": int(mask.sum()),
                "mean_predicted_probability": round(float(np.mean(probabilities[mask])), 6),
                "actual_positive_rate": round(float(np.mean(labels[mask])), 6),
            }
        )
    return rows


def _calibrate_direction_probability(
    *,
    actual: np.ndarray,
    raw_probability: np.ndarray,
    dates: pd.Series,
    latest_raw_probability: float,
    model_config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "actual": np.asarray(actual, dtype=float),
            "raw": np.asarray(raw_probability, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    minimum_samples = int(model_config.get("probability_calibration_min_samples", 120))
    evaluation_ratio = float(model_config.get("probability_calibration_evaluation_ratio", 0.30))
    minimum_improvement = float(model_config.get("probability_calibration_min_brier_improvement", 0.0005))
    raw_latest = float(np.clip(latest_raw_probability, 0.0, 1.0))
    labels = (frame["actual"].to_numpy(dtype=float) > 0).astype(int)
    raw = frame["raw"].to_numpy(dtype=float).clip(0.0, 1.0)
    base = {
        "status": "raw_probability_retained",
        "method": "uncalibrated_logistic_probability",
        "oos_samples": int(len(frame)),
        "raw_oos_brier_score": round(float(brier_score_loss(labels, raw)), 6) if len(frame) else None,
        "reliability_bins": _probability_reliability_bins(frame["actual"].to_numpy(dtype=float), raw),
    }
    if len(frame) < minimum_samples:
        base["reason"] = f"样本外方向概率只有{len(frame)}个，少于校准门槛{minimum_samples}"
        return raw_latest, base
    split = max(minimum_samples // 2, int(len(frame) * (1.0 - evaluation_ratio)))
    split = min(split, len(frame) - max(30, minimum_samples // 4))
    train_labels, evaluation_labels = labels[:split], labels[split:]
    if len(np.unique(train_labels)) < 2 or len(np.unique(evaluation_labels)) < 2:
        base["reason"] = "时间校准窗或后段评估窗只有单一方向类别"
        return raw_latest, base
    calibration = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibration.fit(raw[:split], train_labels)
    evaluation_calibrated = np.asarray(calibration.predict(raw[split:]), dtype=float).clip(0.0, 1.0)
    raw_brier = float(brier_score_loss(evaluation_labels, raw[split:]))
    calibrated_brier = float(brier_score_loss(evaluation_labels, evaluation_calibrated))
    base["time_ordered_evaluation"] = {
        "calibration_samples": int(split),
        "evaluation_samples": int(len(frame) - split),
        "raw_brier_score": round(raw_brier, 6),
        "isotonic_brier_score": round(calibrated_brier, 6),
        "improvement": round(raw_brier - calibrated_brier, 6),
    }
    if raw_brier - calibrated_brier < minimum_improvement:
        base["reason"] = "保序校准未在后段时间窗稳定改善Brier分数"
        return raw_latest, base
    production_calibration = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    production_calibration.fit(raw, labels)
    calibrated_latest = float(production_calibration.predict([raw_latest])[0])
    calibrated_oos = np.asarray(production_calibration.predict(raw), dtype=float).clip(0.0, 1.0)
    return calibrated_latest, {
        **base,
        "status": "calibrated",
        "method": "time_ordered_isotonic_then_refit_on_all_oos",
        "latest_raw_probability": round(raw_latest, 6),
        "latest_calibrated_probability": round(calibrated_latest, 6),
        "reliability_bins": _probability_reliability_bins(
            frame["actual"].to_numpy(dtype=float),
            calibrated_oos,
        ),
    }


def _rolling_conformal_interval(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: pd.Series,
    latest_prediction: float,
    model_config: dict[str, Any],
) -> tuple[list[float] | None, dict[str, Any]]:
    coverage = float(model_config.get("conformal_coverage", 0.80))
    minimum_samples = int(model_config.get("conformal_min_samples", 80))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    residuals = np.abs(frame["actual"].to_numpy(dtype=float) - frame["predicted"].to_numpy(dtype=float))
    if len(residuals) < minimum_samples:
        return None, {
            "status": "insufficient_oos_samples",
            "samples": int(len(residuals)),
            "minimum_samples": minimum_samples,
            "target_coverage": coverage,
        }

    def radius(values: np.ndarray) -> float:
        adjusted = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
        return float(np.quantile(values, adjusted, method="higher"))

    hits: list[bool] = []
    for index in range(minimum_samples, len(frame)):
        current_radius = radius(residuals[:index])
        hits.append(bool(residuals[index] <= current_radius))
    latest_radius = radius(residuals)
    interval = [float(latest_prediction - latest_radius), float(latest_prediction + latest_radius)]
    return interval, {
        "status": "ok",
        "method": "rolling_split_conformal_absolute_residual",
        "target_coverage": coverage,
        "calibration_samples": int(len(residuals)),
        "rolling_evaluation_samples": int(len(hits)),
        "rolling_empirical_coverage": round(float(np.mean(hits)), 6) if hits else None,
        "latest_radius": round(latest_radius, 6),
    }


def _fold_stability(folds: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [fold for fold in folds if fold.get("status") == "ok"]
    if not successful:
        return {"status": "unavailable", "folds": 0}
    result: dict[str, Any] = {
        "status": "ok",
        "folds": int(len(successful)),
        "passed_rate": round(float(np.mean([bool(fold.get("fold_passed")) for fold in successful])), 6),
    }
    for field in ["direction_accuracy", "skill_vs_median_baseline", "mean_daily_rank_ic"]:
        values = np.asarray([float(fold[field]) for fold in successful if fold.get(field) is not None], dtype=float)
        result[field] = {
            "mean": round(float(np.mean(values)), 6) if len(values) else None,
            "std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0 if len(values) else None,
            "minimum": round(float(np.min(values)), 6) if len(values) else None,
            "maximum": round(float(np.max(values)), 6) if len(values) else None,
        }
    return result


def _regime_stability(
    oof: pd.DataFrame,
    *,
    regime_column: str,
) -> dict[str, Any]:
    if regime_column not in oof.columns:
        return {"status": "unavailable", "reason": f"缺少{regime_column}"}
    frame = oof.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual", "predicted", regime_column]
    )
    if len(frame) < 90 or frame[regime_column].nunique() < 3:
        return {"status": "unavailable", "samples": int(len(frame))}
    lower, upper = frame[regime_column].quantile([1 / 3, 2 / 3]).tolist()
    groups = {
        "weak_market": frame[frame[regime_column] <= lower],
        "sideways_market": frame[(frame[regime_column] > lower) & (frame[regime_column] < upper)],
        "strong_market": frame[frame[regime_column] >= upper],
    }
    metrics: dict[str, Any] = {}
    for name, group in groups.items():
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        metrics[name] = {
            "samples": int(len(group)),
            "mae": round(float(mean_absolute_error(actual, predicted)), 6) if len(group) else None,
            "direction_accuracy": round(float(np.mean((actual > 0) == (predicted > 0))), 6) if len(group) else None,
            "mean_actual_return": round(float(np.mean(actual)), 6) if len(group) else None,
            "mean_predicted_return": round(float(np.mean(predicted)), 6) if len(group) else None,
        }
    return {
        "status": "ok",
        "regime_factor": regime_column,
        "tercile_cutoffs": [round(float(lower), 6), round(float(upper), 6)],
        "regimes": metrics,
    }


def _experiment_fingerprint(
    *,
    feature_columns: list[str],
    target_definition: str,
    split_method: str,
    model_config: dict[str, Any],
) -> str:
    payload = {
        "contract": "daily_k_quant_research_v3",
        "features": feature_columns,
        "target": target_definition,
        "split": split_method,
        "model_config": model_config,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _select_ensemble_weight(
    actual: np.ndarray,
    tree_prediction: np.ndarray,
    linear_prediction: np.ndarray,
    model_config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Select a convex tree weight using only already out-of-sample observations."""
    enabled = bool(model_config.get("ensemble_enabled", True))
    default_weight = float(model_config.get("ensemble_default_tree_weight", 0.75))
    minimum_samples = int(model_config.get("ensemble_min_calibration_samples", 80))
    arrays = [np.asarray(value, dtype=float) for value in (actual, tree_prediction, linear_prediction)]
    finite = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])
    clean_actual, clean_tree, clean_linear = (value[finite] for value in arrays)
    if not enabled:
        return 1.0, {
            "status": "disabled",
            "selection_method": "tree_only_by_configuration",
            "calibration_samples": int(len(clean_actual)),
            "tree_weight": 1.0,
            "linear_weight": 0.0,
        }
    if len(clean_actual) < minimum_samples:
        return default_weight, {
            "status": "insufficient_oos_calibration_samples",
            "selection_method": "configured_default_until_enough_oos_samples",
            "calibration_samples": int(len(clean_actual)),
            "minimum_calibration_samples": minimum_samples,
            "tree_weight": round(default_weight, 4),
            "linear_weight": round(1.0 - default_weight, 4),
        }

    raw_grid = model_config.get("ensemble_tree_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    grid = sorted({float(value) for value in raw_grid if 0.0 <= float(value) <= 1.0})
    if not grid:
        grid = [default_weight]
    candidates: list[tuple[float, float, np.ndarray]] = []
    for tree_weight in grid:
        blended = tree_weight * clean_tree + (1.0 - tree_weight) * clean_linear
        candidates.append((float(mean_absolute_error(clean_actual, blended)), tree_weight, blended))
    selected_mae, selected_weight, selected_prediction = min(
        candidates,
        key=lambda item: (item[0], abs(item[1] - default_weight)),
    )
    return selected_weight, {
        "status": "selected_from_oos_predictions",
        "selection_method": "minimum_mae_on_prior_oos_grid",
        "calibration_samples": int(len(clean_actual)),
        "tree_weight": round(float(selected_weight), 4),
        "linear_weight": round(float(1.0 - selected_weight), 4),
        "tree_mae": round(float(mean_absolute_error(clean_actual, clean_tree)), 6),
        "linear_mae": round(float(mean_absolute_error(clean_actual, clean_linear)), 6),
        "ensemble_mae": round(float(selected_mae), 6),
        "ensemble_direction_accuracy": round(
            float(np.mean((selected_prediction > 0) == (clean_actual > 0))),
            6,
        ),
        "candidate_tree_weights": grid,
    }


def _blend_component_predictions(
    tree_prediction: np.ndarray,
    linear_prediction: np.ndarray,
    tree_weight: float,
) -> np.ndarray:
    return tree_weight * np.asarray(tree_prediction, dtype=float) + (
        1.0 - tree_weight
    ) * np.asarray(linear_prediction, dtype=float)


def _nested_training_ensemble_weight(
    *,
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_date_column: str,
    clip_low: float,
    clip_high: float,
    model_config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Calibrate a holdout-safe ensemble weight on the tail of the training period."""
    dates = [pd.Timestamp(value) for value in sorted(train["trade_date"].dropna().unique())]
    ratio = float(model_config.get("ensemble_calibration_ratio", 0.15))
    minimum_dates = int(model_config.get("ensemble_min_calibration_dates", 20))
    if len(dates) < minimum_dates * 2:
        return _select_ensemble_weight(np.array([]), np.array([]), np.array([]), model_config)
    calibration_dates = max(minimum_dates, int(len(dates) * ratio))
    calibration_dates = min(calibration_dates, len(dates) - minimum_dates)
    cutoff = dates[-calibration_dates]
    inner_train = train[
        (train["trade_date"] < cutoff)
        & (pd.to_datetime(train[target_date_column]) < cutoff)
    ]
    calibration = train[train["trade_date"] >= cutoff]
    minimum_samples = int(model_config.get("ensemble_min_calibration_samples", 80))
    if len(inner_train) < minimum_samples or len(calibration) < minimum_samples:
        weight, diagnostics = _select_ensemble_weight(
            np.array([]), np.array([]), np.array([]), model_config
        )
        diagnostics.update(
            {
                "nested_train_samples": int(len(inner_train)),
                "nested_calibration_samples": int(len(calibration)),
            }
        )
        return weight, diagnostics
    inner_target = np.clip(
        inner_train[target_column].astype(float).to_numpy(),
        clip_low,
        clip_high,
    )
    tree_prediction, linear_prediction = _fit_model_components(
        train_features=inner_train[feature_columns],
        train_target=inner_target,
        predict_features=calibration[feature_columns],
        model_config=model_config,
    )
    tree_prediction = np.clip(tree_prediction, clip_low, clip_high)
    linear_prediction = np.clip(linear_prediction, clip_low, clip_high)
    weight, diagnostics = _select_ensemble_weight(
        calibration[target_column].astype(float).to_numpy(),
        tree_prediction,
        linear_prediction,
        model_config,
    )
    diagnostics.update(
        {
            "selection_scope": "nested_tail_of_outer_training_only",
            "calibration_start": cutoff.strftime("%Y-%m-%d"),
            "nested_train_samples": int(len(inner_train)),
            "nested_calibration_samples": int(len(calibration)),
        }
    )
    return weight, diagnostics


def _top_n_validation_metrics(
    validation_frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    horizon: int,
    budget_yuan: float,
    scenario: CostScenario,
    top_n: int,
    trading_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_columns = ["trade_date", "ts_code", f"entry_open_t{horizon}"]
    evaluation_columns.extend(
        column for column in ["amount_yuan", "atr_14_pct"] if column in validation_frame.columns
    )
    evaluation = validation_frame[evaluation_columns].copy()
    evaluation["actual"] = actual
    evaluation["predicted"] = predicted
    net_actual: list[float] = []
    net_predicted: list[float] = []
    for _, row in evaluation.iterrows():
        cost_rate, _ = _stock_roundtrip_cost(
            str(row["ts_code"]),
            float(row[f"entry_open_t{horizon}"]),
            budget_yuan,
            scenario,
            daily_amount_yuan=_round_optional(row.get("amount_yuan")),
            atr_pct=_round_optional(row.get("atr_14_pct")),
            trading_settings=trading_settings,
        )
        if cost_rate is None:
            net_actual.append(np.nan)
            net_predicted.append(np.nan)
        else:
            net_actual.append(_apply_cost(float(row["actual"]), cost_rate))
            net_predicted.append(_apply_cost(float(row["predicted"]), cost_rate))
    evaluation["net_actual"] = net_actual
    evaluation["net_predicted"] = net_predicted
    evaluation = evaluation.dropna(subset=["net_actual", "net_predicted"])

    selected_returns: list[float] = []
    excess_returns: list[float] = []
    for _, group in evaluation.groupby("trade_date"):
        if len(group) < 2:
            continue
        selected = group.nlargest(min(int(top_n), len(group)), "net_predicted")
        selected_return = float(selected["net_actual"].mean())
        selected_returns.append(selected_return)
        excess_returns.append(selected_return - float(group["net_actual"].mean()))
    return {
        "top_n": int(top_n),
        "top_n_days": int(len(selected_returns)),
        "top_n_mean_net_return": round(float(np.mean(selected_returns)), 6) if selected_returns else 0.0,
        "top_n_positive_day_rate": round(float(np.mean(np.asarray(selected_returns) > 0)), 6)
        if selected_returns
        else 0.0,
        "top_n_mean_excess_vs_universe": round(float(np.mean(excess_returns)), 6) if excess_returns else 0.0,
    }


def xunlian_yuce_moxing(
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate chronologically, then refit each accepted candidate model on all labels."""
    panel = panel.copy()
    latest = latest.copy()
    for column in BOARD_FEATURE_COLUMNS:
        if column not in panel.columns:
            panel[column] = np.nan
        if column not in latest.columns:
            latest[column] = np.nan
    model_config = config["moxing"]
    horizons = [int(value) for value in model_config["horizons"]]
    minimum_coverage = float(model_config.get("min_feature_coverage", 0.20))
    coverage = panel[BOARD_FEATURE_COLUMNS].notna().mean()
    feature_columns = [
        column
        for column in BOARD_FEATURE_COLUMNS
        if float(coverage.get(column, 0.0)) >= minimum_coverage
    ]
    if not feature_columns:
        raise RuntimeError("板块模型没有达到覆盖率门槛的日频特征")
    dates = sorted(pd.to_datetime(panel["trade_date"].dropna().unique()))
    if len(dates) < 100:
        raise RuntimeError(f"可用交易日期只有 {len(dates)} 个，无法进行可靠的时序验证")
    validation_ratio = float(model_config.get("validation_ratio", 0.2))
    cutoff_index = max(60, min(len(dates) - 20, int(len(dates) * (1.0 - validation_ratio))))
    cutoff = pd.Timestamp(dates[cutoff_index])
    predictions = latest[["ts_code", "name", "trade_date", "close"] + feature_columns].copy()
    validation: dict[str, Any] = {
        "split_method": "chronological_purged_holdout_with_three_oos_stability_subwindows",
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "signal_and_execution": {
            "signal": "T日收盘后",
            "entry": "下一市场交易日（T+1）开盘",
            "T+1": "入场后第1个可卖出交易日收盘，即信号后的第2个市场交易日",
            "T+2": "入场后第2个可卖出交易日收盘",
            "T+3": "入场后第3个可卖出交易日收盘",
        },
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
        "feature_coverage": {column: round(float(coverage[column]), 4) for column in feature_columns},
        "feature_preprocessing": "训练窗口内完成因子稳定筛选和逐特征分位数去极值；Ridge与方向Logistic另做稳健缩放，验证和最新数据不参与拟合",
        "horizons": {},
    }

    minimum_train = int(model_config.get("min_training_samples", 500))
    minimum_validation = int(model_config.get("min_validation_samples", 100))
    quantiles = model_config.get("prediction_clip_quantiles", [0.01, 0.99])
    lower_q, upper_q = float(quantiles[0]), float(quantiles[1])
    cost_scenario_name = str(config.get("jiaoyi", {}).get("cost_scenario", "normal_cost"))
    budget_yuan, cost_scenario, _, _ = _load_cost_assumption(cost_scenario_name)
    validation_top_n = max(1, int(model_config.get("validation_top_n", 3)))

    for horizon in horizons:
        target_column = f"target_t{horizon}"
        target_date_column = f"target_date_t{horizon}"
        entry_date_column = f"entry_date_t{horizon}"
        entry_open_column = f"entry_open_t{horizon}"
        usable = panel.dropna(
            subset=[target_column, target_date_column, entry_date_column, entry_open_column]
        ).copy()
        usable["trade_date"] = pd.to_datetime(usable["trade_date"])
        usable[target_date_column] = pd.to_datetime(usable[target_date_column])
        usable[entry_date_column] = pd.to_datetime(usable[entry_date_column])
        train = usable[(usable["trade_date"] < cutoff) & (usable[target_date_column] < cutoff)]
        validation_frame = usable[usable["trade_date"] >= cutoff]
        if len(train) < minimum_train or len(validation_frame) < minimum_validation:
            raise RuntimeError(
                f"T+{horizon} 样本不足：训练 {len(train)}（至少 {minimum_train}），"
                f"验证 {len(validation_frame)}（至少 {minimum_validation}）"
            )

        y_train_raw = train[target_column].astype(float).to_numpy()
        clip_low = float(np.nanquantile(y_train_raw, lower_q))
        clip_high = float(np.nanquantile(y_train_raw, upper_q))
        y_train = np.clip(y_train_raw, clip_low, clip_high)
        y_validation = validation_frame[target_column].astype(float).to_numpy()
        validation_features, factor_selection = _select_stable_features(
            train,
            feature_columns,
            target_column,
            model_config,
        )
        if not validation_features:
            raise RuntimeError(f"T+{horizon} 训练窗口没有稳定可用因子")
        validation_tree_weight, validation_ensemble = _nested_training_ensemble_weight(
            train=train,
            feature_columns=validation_features,
            target_column=target_column,
            target_date_column=target_date_column,
            clip_low=clip_low,
            clip_high=clip_high,
            model_config=model_config,
        )
        validation_tree_prediction, validation_linear_prediction = _fit_model_components(
            train_features=train[validation_features],
            train_target=y_train,
            predict_features=validation_frame[validation_features],
            model_config=model_config,
        )
        validation_direction_probability, validation_direction_model = _fit_direction_probabilities(
            train_features=train[validation_features],
            train_target=y_train_raw,
            predict_features=validation_frame[validation_features],
            model_config=model_config,
        )
        validation_tree_prediction = np.clip(validation_tree_prediction, clip_low, clip_high)
        validation_linear_prediction = np.clip(validation_linear_prediction, clip_low, clip_high)
        validation_prediction = np.clip(
            _blend_component_predictions(
                validation_tree_prediction,
                validation_linear_prediction,
                validation_tree_weight,
            ),
            clip_low,
            clip_high,
        )
        baseline_metrics = regression_baseline_metrics(
            actual=y_validation,
            predicted=validation_prediction,
            training_target=y_train_raw,
        )
        mae = float(baseline_metrics["model_mae"])
        baseline_value = float(np.median(y_train))
        baseline_mae = float(baseline_metrics["baselines"]["training_median"]["mae"])
        direction_accuracy = float(np.mean((validation_prediction > 0) == (y_validation > 0)))
        rank_ic, rank_ic_days = _daily_rank_ic(
            validation_frame["trade_date"],
            y_validation,
            validation_prediction,
        )
        skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
        quality = _quality_score(
            train_count=len(train),
            direction_accuracy=direction_accuracy,
            rank_ic=rank_ic,
            skill_vs_baseline=skill,
        )
        top_n_metrics = _top_n_validation_metrics(
            validation_frame,
            y_validation,
            validation_prediction,
            horizon=horizon,
            budget_yuan=budget_yuan,
            scenario=cost_scenario,
            top_n=validation_top_n,
            trading_settings=config.get("jiaoyi", {}),
        )
        minimum_rank_ic = float(model_config.get("min_mean_daily_rank_ic", 0.01))
        minimum_skill = float(model_config.get("min_skill_vs_baseline", 0.01))
        minimum_best_naive_skill = float(model_config.get("min_skill_vs_best_naive_baseline", 0.0))
        minimum_direction = float(model_config.get("min_direction_accuracy", 0.52))
        minimum_rank_days = int(model_config.get("min_rank_ic_days", 10))
        minimum_top_n_days = int(model_config.get("min_top_n_days", 10))
        validation_passed = bool(
            direction_accuracy >= minimum_direction
            and rank_ic >= minimum_rank_ic
            and rank_ic_days >= minimum_rank_days
            and skill >= minimum_skill
            and float(baseline_metrics["skill_vs_best_naive_baseline"]) >= minimum_best_naive_skill
            and top_n_metrics["top_n_days"] >= minimum_top_n_days
            and top_n_metrics["top_n_mean_net_return"] > 0
            and top_n_metrics["top_n_mean_excess_vs_universe"] > 0
        )

        # Holdout metrics above stay untouched.  The production forecast is
        # refit on every label whose entry and exit are already known so the
        # freshest labelled observations are not discarded after validation.
        y_full_raw = usable[target_column].astype(float).to_numpy()
        final_clip_low = float(np.nanquantile(y_full_raw, lower_q))
        final_clip_high = float(np.nanquantile(y_full_raw, upper_q))
        production_features, production_factor_selection = _select_stable_features(
            usable,
            feature_columns,
            target_column,
            model_config,
        )
        if not production_features:
            raise RuntimeError(f"T+{horizon} 全量训练窗口没有稳定可用因子")
        production_tree_weight, production_ensemble = _select_ensemble_weight(
            y_validation,
            validation_tree_prediction,
            validation_linear_prediction,
            model_config,
        )
        production_tree_prediction, production_linear_prediction = _fit_model_components(
            train_features=usable[production_features],
            train_target=np.clip(y_full_raw, final_clip_low, final_clip_high),
            predict_features=latest[production_features],
            model_config=model_config,
        )
        production_direction_probability, production_direction_model = _fit_direction_probabilities(
            train_features=usable[production_features],
            train_target=y_full_raw,
            predict_features=latest[production_features],
            model_config=model_config,
        )
        production_tree_prediction = np.clip(
            production_tree_prediction, final_clip_low, final_clip_high
        )
        production_linear_prediction = np.clip(
            production_linear_prediction, final_clip_low, final_clip_high
        )
        latest_prediction = np.clip(
            _blend_component_predictions(
                production_tree_prediction,
                production_linear_prediction,
                production_tree_weight,
            ),
            final_clip_low,
            final_clip_high,
        )
        predictions[f"pred_t{horizon}"] = latest_prediction
        calibrated_probabilities: list[float] = []
        probability_calibration: dict[str, Any] = {}
        conformal_intervals: list[list[float] | None] = []
        conformal_diagnostics: dict[str, Any] = {}
        for latest_index, (raw_probability, point_prediction) in enumerate(
            zip(production_direction_probability, latest_prediction)
        ):
            calibrated, calibration_meta = _calibrate_direction_probability(
                actual=y_validation,
                raw_probability=validation_direction_probability,
                dates=validation_frame["trade_date"],
                latest_raw_probability=float(raw_probability),
                model_config=model_config,
            )
            interval, interval_meta = _rolling_conformal_interval(
                actual=y_validation,
                predicted=validation_prediction,
                dates=validation_frame["trade_date"],
                latest_prediction=float(point_prediction),
                model_config=model_config,
            )
            calibrated_probabilities.append(float(calibrated))
            conformal_intervals.append(interval)
            if latest_index == 0:
                probability_calibration = calibration_meta
                conformal_diagnostics = interval_meta
        predictions[f"direction_prob_t{horizon}"] = calibrated_probabilities
        predictions[f"conformal_low_t{horizon}"] = [
            interval[0] if interval else np.nan for interval in conformal_intervals
        ]
        predictions[f"conformal_high_t{horizon}"] = [
            interval[1] if interval else np.nan for interval in conformal_intervals
        ]

        validation_dates = [pd.Timestamp(value) for value in sorted(validation_frame["trade_date"].unique())]
        temporal_folds: list[dict[str, Any]] = []
        for fold_number, date_slice in enumerate(np.array_split(np.asarray(validation_dates, dtype=object), 3), start=1):
            if not len(date_slice):
                continue
            mask = validation_frame["trade_date"].isin(list(date_slice)).to_numpy()
            fold_actual = y_validation[mask]
            fold_prediction = validation_prediction[mask]
            if not len(fold_actual):
                continue
            fold_baseline = np.full(len(fold_actual), baseline_value)
            fold_baseline_mae = float(mean_absolute_error(fold_actual, fold_baseline))
            fold_mae = float(mean_absolute_error(fold_actual, fold_prediction))
            fold_skill = 1.0 - fold_mae / fold_baseline_mae if fold_baseline_mae > 0 else 0.0
            fold_direction = float(np.mean((fold_prediction > 0) == (fold_actual > 0)))
            fold_rank_ic, _ = _daily_rank_ic(
                validation_frame.loc[mask, "trade_date"],
                fold_actual,
                fold_prediction,
            )
            temporal_folds.append(
                {
                    "fold": fold_number,
                    "status": "ok",
                    "fold_passed": bool(fold_direction >= 0.50 and fold_skill > 0 and fold_rank_ic >= 0),
                    "direction_accuracy": fold_direction,
                    "skill_vs_median_baseline": fold_skill,
                    "mean_daily_rank_ic": fold_rank_ic,
                }
            )
        regime_column = (
            "market_csi300_ret_20"
            if "market_csi300_ret_20" in validation_frame.columns
            and validation_frame["market_csi300_ret_20"].notna().any()
            else "universe_mean_ret_20"
        )
        regime_oof = validation_frame[["trade_date", regime_column]].copy()
        regime_oof["actual"] = y_validation
        regime_oof["predicted"] = validation_prediction
        validation["horizons"][f"T+{horizon}"] = {
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation_frame)),
            "validation_start": validation_frame["trade_date"].min().strftime("%Y-%m-%d"),
            "validation_end": validation_frame["trade_date"].max().strftime("%Y-%m-%d"),
            "mae": round(mae, 6),
            "baseline_mae": round(baseline_mae, 6),
            "skill_vs_median_baseline": round(skill, 6),
            "naive_baseline_comparison": baseline_metrics,
            "direction_accuracy": round(direction_accuracy, 6),
            "mean_daily_rank_ic": round(rank_ic, 6),
            "rank_ic_days": int(rank_ic_days),
            "factor_selection": factor_selection,
            "production_factor_selection": production_factor_selection,
            "experiment_fingerprint": _experiment_fingerprint(
                feature_columns=production_features,
                target_definition=f"next_session_open_to_{horizon}th_sellable_close_return",
                split_method="chronological_purged_holdout_with_three_oos_stability_subwindows",
                model_config=model_config,
            ),
            "direction_model": {
                **validation_direction_model,
                "brier_score": round(
                    float(brier_score_loss((y_validation > 0).astype(int), validation_direction_probability)),
                    6,
                ),
                "classification_accuracy": round(
                    float(np.mean((validation_direction_probability >= 0.5) == (y_validation > 0))),
                    6,
                ),
                "production": production_direction_model,
            },
            "direction_probability_calibration": probability_calibration,
            "conformal_diagnostics": conformal_diagnostics,
            "fold_stability": _fold_stability(temporal_folds),
            "market_regime_stability": _regime_stability(regime_oof, regime_column=regime_column),
            "prediction_clip": [round(clip_low, 6), round(clip_high, 6)],
            "final_prediction_clip": [round(final_clip_low, 6), round(final_clip_high, 6)],
            "final_train_samples": int(len(usable)),
            "final_training_end": usable[target_date_column].max().strftime("%Y-%m-%d"),
            "retrained_on_all_labeled_data": True,
            "model_ensemble": {
                "components": ["HistGradientBoostingRegressor", "Ridge"],
                "outer_validation_weight": validation_ensemble,
                "production_weight": production_ensemble,
                "outer_validation_mean_absolute_disagreement": round(
                    float(np.mean(np.abs(validation_tree_prediction - validation_linear_prediction))),
                    6,
                ),
                "latest_component_predictions": {
                    "tree": [round(float(value), 6) for value in production_tree_prediction],
                    "linear": [round(float(value), 6) for value in production_linear_prediction],
                },
            },
            **top_n_metrics,
            "quality_score": round(quality, 4),
            "quality_label": _quality_label(quality),
            "validation_thresholds": {
                "direction_accuracy": minimum_direction,
                "mean_daily_rank_ic": minimum_rank_ic,
                "rank_ic_days": minimum_rank_days,
                "skill_vs_median_baseline": minimum_skill,
                "skill_vs_best_naive_baseline": minimum_best_naive_skill,
                "top_n_days": minimum_top_n_days,
                "top_n_mean_net_return": "> 0",
                "top_n_mean_excess_vs_universe": "> 0",
            },
            "validation_passed": validation_passed,
        }

    quality_values = [float(item["quality_score"]) for item in validation["horizons"].values()]
    validation["overall_quality_score"] = round(float(np.mean(quality_values)), 4)
    validation["overall_quality_label"] = _quality_label(float(validation["overall_quality_score"]))
    validation["passed_horizons"] = sum(bool(item["validation_passed"]) for item in validation["horizons"].values())
    return predictions, validation


def _prediction_rows(
    predictions: pd.DataFrame,
    constituents: pd.DataFrame,
    validation: dict[str, Any],
    config: dict[str, Any],
    cost_rate: float,
) -> list[dict[str, Any]]:
    horizons = [int(value) for value in config["moxing"]["horizons"]]
    weights = {int(key): float(value) for key, value in config["moxing"]["horizon_weights"].items()}
    passed_horizons = {
        int(label.split("+")[1])
        for label, metrics in validation["horizons"].items()
        if metrics.get("validation_passed")
    }
    budget_yuan, cost_scenario, _, _ = _load_cost_assumption(
        str(config.get("jiaoyi", {}).get("cost_scenario", "normal_cost"))
    )
    constituent_map = constituents.set_index("ts_code").to_dict("index")
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        code = str(row["ts_code"])
        name = str(row["name"])
        close = float(row["close"])
        raw_limit_rate = _limit_rate(code, name)
        limit_rate = float(raw_limit_rate) if raw_limit_rate is not None else None
        _, entry_price_ceiling = _price_limit_bounds(close, limit_rate, 1)
        if not math.isfinite(entry_price_ceiling):
            entry_price_ceiling = close
        stock_cost_rate, position_cost = _stock_roundtrip_cost(
            code,
            entry_price_ceiling,
            budget_yuan,
            cost_scenario,
            daily_amount_yuan=_round_optional(
                constituent_map.get(code, {}).get("amount_yuan", row.get("amount_yuan"))
            ),
            atr_pct=_round_optional(row.get("atr_14_pct")),
            trading_settings=config.get("jiaoyi", {}),
        )
        forecasts: dict[str, Any] = {}
        active_horizons = [
            value
            for value in horizons
            if value in passed_horizons and max(weights.get(value, 0.0), 0.0) > 0
        ]
        active_weight_sum = sum(max(weights.get(value, 0.0), 0.0) for value in active_horizons)
        normalized_weights = {
            value: max(weights.get(value, 0.0), 0.0) / active_weight_sum
            for value in active_horizons
        } if active_weight_sum > 0 else {}
        weighted_net = 0.0 if active_horizons and stock_cost_rate is not None else None
        for horizon in horizons:
            gross = float(row[f"pred_t{horizon}"])
            net = _apply_cost(gross, stock_cost_rate) if stock_cost_rate is not None else None
            quality = validation["horizons"][f"T+{horizon}"]
            direction_probability = _round_optional(row.get(f"direction_prob_t{horizon}"), 6)
            conformal_low = _round_optional(row.get(f"conformal_low_t{horizon}"), 6)
            conformal_high = _round_optional(row.get(f"conformal_high_t{horizon}"), 6)
            # The reference path starts at the signal close.  The first
            # sellable exit spans the T+1 entry session and T+2 exit session,
            # hence horizon + 1 daily-limit steps.
            price_limit_sessions = horizon + 1
            price_lower, price_upper = _price_limit_bounds(close, limit_rate, price_limit_sessions)
            unconstrained_reference_price = _round_price_tick(close * (1.0 + gross))
            constrained_reference_price = _round_price_tick(
                min(max(unconstrained_reference_price, price_lower), price_upper)
            )
            forecasts[f"T+{horizon}"] = {
                "entry_to_exit_gross_return": round(gross, 6),
                "entry_to_exit_gross_return_pct": round(gross * 100.0, 3),
                "estimated_net_return_after_cost": round(net, 6) if net is not None else None,
                "estimated_net_return_after_cost_pct": round(net * 100.0, 3) if net is not None else None,
                "direction_model_positive_probability": direction_probability,
                "direction_probability_method": (
                    quality.get("direction_probability_calibration") or {}
                ).get("method"),
                "conformal_return_interval_80": [conformal_low, conformal_high]
                if conformal_low is not None and conformal_high is not None
                else None,
                "predicted_close": None,
                "predicted_close_unavailable_reason": "实际T+1开盘价尚未知，不能把模型收益可靠换算成目标收盘价",
                "signal_close_reference_price": _round_price_tick(close),
                "model_reference_exit_price_unconstrained": unconstrained_reference_price,
                "model_reference_exit_price_clipped_to_legal_range": constrained_reference_price,
                "reference_exit_price_limit_lower": price_lower,
                "reference_exit_price_limit_upper": price_upper if math.isfinite(price_upper) else None,
                "price_limit_sessions_from_signal_close": int(price_limit_sessions),
                "timing": (
                    f"T日收盘信号，T+1开盘入场；本项为入场后第{horizon}个可卖出交易日收盘"
                ),
                "used_for_ranking": bool(horizon in active_horizons),
                "model_quality": quality["quality_label"],
            }
            if weighted_net is not None and horizon in normalized_weights and net is not None:
                weighted_net += normalized_weights[horizon] * net
        best_horizon = (
            max(
                active_horizons,
                key=lambda value: forecasts[f"T+{value}"]["estimated_net_return_after_cost"],
            )
            if active_horizons and stock_cost_rate is not None
            else None
        )
        annualized_volatility = float(row.get("volatility_20", 0.0) or 0.0)
        atr_pct = float(row.get("atr_14_pct", 0.0) or 0.0)
        holding_sessions = (
            sum(normalized_weights[value] * (value + 1) for value in active_horizons)
            if normalized_weights
            else 1.0
        )
        holding_volatility = max(annualized_volatility, 0.0) / math.sqrt(252.0) * math.sqrt(holding_sessions)
        risk_scale = max(holding_volatility, max(atr_pct, 0.0), 0.01)
        selection_score = weighted_net / risk_scale if weighted_net is not None else None
        current = constituent_map.get(code, {})
        strongest_label = f"T+{best_horizon}" if best_horizon is not None else None
        strongest_forecast = forecasts.get(strongest_label, {}) if strongest_label else {}
        strongest_validation = validation["horizons"].get(strongest_label, {}) if strongest_label else {}
        signal_gate = signal_evidence_gate(
            validation_passed=bool(strongest_validation.get("validation_passed")),
            execution_feasible=bool(position_cost.get("execution_feasible")),
            net_return=_round_optional(strongest_forecast.get("estimated_net_return_after_cost")),
            positive_probability=_round_optional(
                strongest_forecast.get("direction_model_positive_probability")
            ),
            quality_score=_round_optional(strongest_validation.get("quality_score")),
            minimum_net_return=float(config["moxing"].get("abstain_min_net_return", 0.003)),
            minimum_positive_probability=float(
                config["moxing"].get("abstain_min_positive_probability", 0.55)
            ),
            minimum_quality_score=float(config["moxing"].get("abstain_min_quality_score", 0.40)),
        )
        rows.append(
            {
                "ts_code": code,
                "name": name,
                "as_of": pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d"),
                "signal_date": pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d"),
                "latest_close": round(close, 3),
                "selection_score": round(selection_score, 6) if selection_score is not None else None,
                "selection_score_definition": (
                    "通过验证周期的加权成本后收益 / max(持有期历史波动率, ATR占比, 1%)"
                ),
                "selection_expected_holding_sessions": round(float(holding_sessions), 4),
                "selection_risk_scale": round(risk_scale, 6),
                "weighted_expected_net_return": round(weighted_net, 6) if weighted_net is not None else None,
                "ranking_horizons": [f"T+{value}" for value in active_horizons],
                "ranking_horizon_weights": {
                    f"T+{value}": round(weight, 6) for value, weight in normalized_weights.items()
                },
                "strongest_forecast_horizon": f"T+{best_horizon}" if best_horizon is not None else None,
                "strongest_horizon_trading_days": int(best_horizon) if best_horizon is not None else None,
                "strongest_horizon_validation_passed": bool(best_horizon in passed_horizons)
                if best_horizon is not None
                else False,
                "signal_gate": signal_gate,
                "forecast": forecasts,
                "position_and_cost": position_cost,
                "technical_snapshot": {
                    "ret_5": _round_optional(row.get("ret_5"), 6),
                    "ma5_vs_ma20": _round_optional(row.get("ma_trend_5_20"), 6),
                    "rsi_14": _round_optional(row.get("rsi_14"), 2),
                    "atr_14_pct": _round_optional(row.get("atr_14_pct"), 6),
                    "annualized_volatility_20": _round_optional(row.get("volatility_20"), 6),
                    "drawdown_from_20d_high": _round_optional(row.get("drawdown_20"), 6),
                    "position_in_20d_range": _round_optional(row.get("position_20"), 6),
                    "volume_ratio_5_to_20": _round_optional(row.get("volume_ratio_5_20"), 4),
                },
                "board_snapshot": {
                    key: _json_value(current.get(key))
                    for key in ["amount_yuan", "turnover_rate", "pe_dynamic", "pb", "pct_chg"]
                    if key in current
                },
                "a_share_constraints": {
                    "signal_time": "T日收盘后",
                    "assumed_entry_for_analysis": "下一市场交易日（T+1）开盘",
                    "can_sell_earliest": "信号后的第2个市场交易日收盘（输出标记为T+1）",
                    "price_limit_pct": round(limit_rate * 100.0, 2) if limit_rate is not None else None,
                },
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            bool(item["position_and_cost"]["execution_feasible"]),
            float(item["selection_score"]) if item["selection_score"] is not None else -math.inf,
        ),
        reverse=True,
    )


def _source_summary(histories: dict[str, pd.DataFrame]) -> dict[str, Any]:
    sources = Counter(str(frame["data_source"].iloc[-1]) for frame in histories.values())
    adjustments = Counter(str(frame["adjustment"].iloc[-1]) for frame in histories.values())
    return {"history_sources": dict(sources), "adjustments": dict(adjustments)}


def _board_signal_timing(as_of: pd.Timestamp, reference: datetime | None = None) -> dict[str, Any]:
    """Reject a board signal after its assumed next-session opening entry has passed."""
    current = reference or datetime.now()
    signal_date = pd.Timestamp(as_of).normalize()
    expected_date = _latest_expected_market_date(current)
    minute = current.hour * 60 + current.minute
    if current.weekday() >= 5:
        session_status = "non_trading_day"
        timing_valid = signal_date >= expected_date
    elif minute < 9 * 60 + 15:
        session_status = "pre_market"
        timing_valid = signal_date >= expected_date
    elif minute < 9 * 60 + 30:
        session_status = "opening_auction"
        timing_valid = signal_date >= expected_date
    elif minute < 15 * 60 + 5:
        session_status = "entry_already_opened"
        timing_valid = False
    else:
        session_status = "post_close"
        timing_valid = signal_date >= expected_date
    if timing_valid:
        reason = "最近完整收盘信号对应的下一交易日开盘尚未结束"
    elif session_status == "entry_already_opened":
        reason = "当前已进入交易时段，最近完整收盘信号假设的下一交易日开盘入口已经过去"
    else:
        reason = "行情仓库尚未覆盖最近应完成交易日，旧信号假设的开盘入口已经过去"
    return {
        "valid": timing_valid,
        "reason": reason,
        "captured_at": current.strftime("%Y-%m-%d %H:%M:%S"),
        "session_status": session_status,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "expected_latest_completed_date": expected_date.strftime("%Y-%m-%d"),
        "calendar_precision": "工作日和本地时钟判断；法定节假日以交易所日历为准",
    }


def _save_artifacts(run_dir: str, result: dict[str, Any]) -> dict[str, str]:
    run_path = safe_run_dir(run_dir)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "bankuai_xuangu.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = artifact_dir / "bankuai_xuangu.csv"
    flat_rows: list[dict[str, Any]] = []
    for item in result.get("validated_candidates", []):
        row = {
            "ts_code": item["ts_code"],
            "name": item["name"],
            "as_of": item["as_of"],
            "latest_close": item["latest_close"],
            "selection_score": item["selection_score"],
            "strongest_forecast_horizon": item["strongest_forecast_horizon"],
        }
        for horizon in [1, 2, 3]:
            forecast = item["forecast"][f"T+{horizon}"]
            row[f"t{horizon}_entry_to_exit_gross_return"] = forecast["entry_to_exit_gross_return"]
            row[f"t{horizon}_net_return"] = forecast["estimated_net_return_after_cost"]
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {"json": str(json_path), "csv": str(csv_path)}


def _selection_sequence_id(
    *,
    board_meta: dict[str, Any],
    as_of: pd.Timestamp,
    source: str,
    config_path: str,
    eligible: list[dict[str, Any]],
) -> str:
    payload = {
        "board_name": board_meta.get("resolved_name"),
        "board_type": board_meta.get("board_type"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "source": source,
        "config_path": config_path,
        "ordered_codes": [str(item.get("ts_code") or "") for item in eligible],
    }
    return "sel_" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _paginate_candidates(
    eligible: list[dict[str, Any]],
    *,
    offset: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    candidates: list[dict[str, Any]] = []
    for candidate_rank, item in enumerate(eligible[offset : offset + batch_size], start=offset + 1):
        candidate = dict(item)
        candidate["candidate_rank"] = candidate_rank
        candidates.append(candidate)
    next_offset = offset + len(candidates)
    return candidates, next_offset, next_offset < len(eligible)


def bankuai_xuangu(
    *,
    bankuai: str,
    bankuai_leixing: str = "auto",
    top_n: int = 8,
    offset: int = 0,
    selection_id: str | None = None,
    source: str = "auto",
    config_path: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """Select stocks from a specified A-share board and predict only T+1 to T+3."""
    _ensure_dotenv()
    requested_top_n = int(top_n)
    top_n = max(1, min(requested_top_n, 8))
    offset = max(0, int(offset))
    selection_id = str(selection_id or "").strip() or None
    if offset > 0 and selection_id is None:
        return {
            "status": "error",
            "error_code": "selection_id_required",
            "error": "offset大于0时必须传入上一批返回的selection_id，避免候选序列错位或重复",
        }
    source = source.strip().lower()
    if source not in {"auto", "tushare", "akshare"}:
        return {"status": "error", "error": "source 必须是 auto、tushare 或 akshare"}
    try:
        config, resolved_config = jiazai_lianghua_peizhi(config_path)
    except Exception as exc:
        return {
            "status": "error",
            "error_code": "configuration_invalid",
            "failure_stage": "load_configuration",
            "error": f"量化配置无效：{exc}",
        }

    try:
        constituents, board_meta = huoqu_bankuai_chengfen(bankuai, bankuai_leixing=bankuai_leixing)
    except Exception as exc:
        return {
            "status": "error",
            "error_code": classify_failure(exc),
            "failure_stage": "fetch_board_constituents",
            "error": str(exc),
            "board": bankuai,
        }
    try:
        from src.ashare.riping_cangku import (
            board_constituent_history_status,
            snapshot_board_constituents,
        )

        snapshot_result = snapshot_board_constituents(
            constituents,
            board_name=str(board_meta.get("resolved_name") or bankuai),
            board_type=str(board_meta.get("board_type") or bankuai_leixing),
            source=str(board_meta.get("constituent_source") or "unknown"),
            snapshot_date=_latest_expected_market_date().strftime("%Y%m%d"),
        )
        constituent_history = board_constituent_history_status(
            board_name=str(board_meta.get("resolved_name") or bankuai),
            board_type=str(board_meta.get("board_type") or bankuai_leixing),
            source=str(board_meta.get("constituent_source") or "unknown"),
        )
        constituent_history["latest_snapshot_write"] = snapshot_result
    except Exception as exc:
        constituent_history = {
            "status": "snapshot_failed",
            "historical_membership_ready": False,
            "error_category": classify_failure(exc),
            "error": str(exc),
        }
    board_meta["constituent_history"] = constituent_history
    filtered, rejected = _filter_constituents(constituents, config)
    if filtered.empty:
        return {
            "status": "error",
            "error": "板块成分股全部被流动性、价格、风险标记或涨停过滤",
            "board": board_meta,
            "constituent_count": int(len(constituents)),
            "rejected": rejected[:30],
        }

    data_config = config["shuju"]
    history_calendar_days = int(data_config.get("history_calendar_days", 1080))
    base_max_board_stocks = int(data_config.get("max_board_stocks", 24))
    applied_max_board_stocks = base_max_board_stocks
    warehouse_range: dict[str, Any] = {
        "status": "not_checked",
        "ready": False,
        "coverage": 0.0,
    }
    try:
        from src.ashare.riping_cangku import warehouse_range_coverage

        warehouse_end = _latest_expected_market_date()
        warehouse_range = warehouse_range_coverage(
            start_date=(warehouse_end - timedelta(days=history_calendar_days)).strftime("%Y%m%d"),
            end_date=warehouse_end.strftime("%Y%m%d"),
        )
        if warehouse_range.get("ready"):
            applied_max_board_stocks = max(
                applied_max_board_stocks,
                int(data_config.get("warehouse_max_board_stocks", 80)),
            )
    except Exception as exc:
        warehouse_range = {
            "status": "check_failed",
            "ready": False,
            "coverage": 0.0,
            "error_category": classify_failure(exc),
            "error": str(exc),
        }
    histories, names, fetch_errors, fetch_warnings = _fetch_histories(
        filtered,
        source=source,
        history_calendar_days=history_calendar_days,
        minimum_rows=int(data_config.get("minimum_history_rows", 120)),
        max_stocks=applied_max_board_stocks,
        pause_seconds=float(data_config.get("request_pause_seconds", 0.15)),
    )
    if not histories:
        return {
            "status": "error",
            "error": "所有候选股的历史行情都拉取失败",
            "board": board_meta,
            "fetch_errors": fetch_errors,
        }

    horizons = [int(value) for value in config["moxing"]["horizons"]]
    panel = goujian_moxing_shuju(histories, names, horizons)
    if panel.empty:
        return {"status": "error", "error": "无法构造模型样本", "fetch_errors": fetch_errors}
    membership_filter_meta: dict[str, Any] = {
        "applied": False,
        "reason": "真实板块成分快照尚未达到历史训练跨度门槛",
    }
    if constituent_history.get("historical_membership_ready"):
        try:
            from src.ashare.riping_cangku import load_board_membership_history

            membership, membership_meta = load_board_membership_history(
                board_name=str(board_meta.get("resolved_name") or bankuai),
                board_type=str(board_meta.get("board_type") or bankuai_leixing),
                source=str(board_meta.get("constituent_source") or "unknown"),
            )
            panel, membership_filter_meta = filter_panel_by_membership_snapshots(panel, membership)
            membership_filter_meta["history"] = membership_meta
            if panel.empty:
                raise RuntimeError("按历史成分快照过滤后没有可用模型样本")
        except Exception as exc:
            return {
                "status": "error",
                "error_code": "membership_history_filter_failed",
                "failure_category": classify_failure(exc),
                "error": f"板块历史成分快照已经启用，但按时点过滤模型面板失败：{exc}",
                "board": board_meta,
            }
    constituent_history["model_membership_filter"] = membership_filter_meta
    panel, daily_factor_meta = enrich_daily_factor_panel(
        panel,
        source=source,
        include_historical_valuation=True,
    )
    fetch_warnings.extend(str(value) for value in daily_factor_meta.get("warnings", []))
    latest = (
        panel.sort_values("trade_date")
        .groupby("ts_code", as_index=False)
        .tail(1)
        .dropna(subset=FEATURE_COLUMNS)
        .reset_index(drop=True)
    )
    if latest.empty:
        return {"status": "error", "error": "候选股没有足够的最新技术指标"}
    global_as_of = pd.to_datetime(latest["trade_date"]).max()
    market_freshness = _market_data_freshness(global_as_of)
    if market_freshness["status"] == "too_stale":
        return {
            "status": "error",
            "error": (
                f"板块最新可用行情停留在 {global_as_of.strftime('%Y-%m-%d')}，"
                f"距最近应完成交易日已 {market_freshness['business_days_old']} 个工作日；"
                "可能存在停牌或数据源延迟，已停止输出当前预测"
            ),
            "board": board_meta,
            "market_freshness": market_freshness,
            "history_stock_count": int(len(histories)),
            "fetch_errors": fetch_errors[:30],
        }
    signal_timing = _board_signal_timing(global_as_of)
    if not signal_timing["valid"]:
        return {
            "status": "error",
            "error_code": "signal_entry_expired",
            "error": signal_timing["reason"],
            "board": board_meta,
            "as_of": global_as_of.strftime("%Y-%m-%d"),
            "market_freshness": market_freshness,
            "signal_timing": signal_timing,
            "history_stock_count": int(len(histories)),
            "fetch_errors": fetch_errors[:30],
        }
    if market_freshness["status"] == "possibly_stale":
        fetch_warnings.append(
            f"板块最新行情距最近应完成交易日约 {market_freshness['business_days_old']} 个工作日，"
            "可能存在长假或数据接口延迟"
        )
    latest_dates = pd.to_datetime(latest["trade_date"])
    stale_latest = latest[latest_dates != global_as_of][["ts_code", "name", "trade_date"]].copy()
    feature_ready_stock_count = int(len(latest))
    latest = latest[pd.to_datetime(latest["trade_date"]) == global_as_of].reset_index(drop=True)

    try:
        predictions, validation = xunlian_yuce_moxing(panel, latest, config)
        cost_rate, cost_meta = _roundtrip_cost(str(config["jiaoyi"].get("cost_scenario", "normal_cost")))
    except Exception as exc:
        return {
            "status": "error",
            "error_code": "model_training_failed",
            "failure_category": classify_failure(exc),
            "failure_stage": "train_and_validate_models",
            "error": f"T+1/T+2/T+3 模型训练失败：{exc}",
            "board": board_meta,
            "history_stock_count": int(len(histories)),
            "sample_count": int(len(panel)),
            "fetch_errors": fetch_errors[:30],
        }

    ranked = _prediction_rows(predictions, filtered, validation, config, cost_rate)
    passed_horizon_labels = [
        label
        for label, metrics in validation["horizons"].items()
        if metrics.get("validation_passed")
    ]
    eligible = [
        item
        for item in ranked
        if passed_horizon_labels
        and item.get("signal_gate", {}).get("actionable_signal")
        and item["position_and_cost"]["execution_feasible"]
        and item["strongest_horizon_validation_passed"]
        and item["weighted_expected_net_return"] is not None
        and float(item["weighted_expected_net_return"]) > 0
        and item["selection_score"] is not None
        and float(item["selection_score"]) > 0
        and float(item["forecast"][item["strongest_forecast_horizon"]]["estimated_net_return_after_cost"]) > 0
    ]
    current_selection_id = _selection_sequence_id(
        board_meta=board_meta,
        as_of=global_as_of,
        source=source,
        config_path=resolved_config,
        eligible=eligible,
    )
    if selection_id is not None and selection_id != current_selection_id:
        return {
            "status": "selection_expired",
            "error_code": "selection_sequence_changed",
            "error": "板块快照或模型排名已经变化，旧候选序列不能继续顺延；请从新的Top 8重新开始",
            "board": board_meta,
            "as_of": global_as_of.strftime("%Y-%m-%d"),
            "provided_selection_id": selection_id,
            "current_selection_id": current_selection_id,
            "restart_offset": 0,
        }
    validated_candidates, next_offset, has_more = _paginate_candidates(
        eligible,
        offset=offset,
        batch_size=top_n,
    )
    limit_notice = (
        f"单批最多返回8只；请求的{requested_top_n}只已按Top 8执行"
        if requested_top_n > 8
        else None
    )
    daily_source_policy = {
        "auto": "个股日线优先 Tushare，失败后降级 AKShare",
        "tushare": "个股日线固定使用 Tushare，不自动降级 AKShare",
        "akshare": "个股日线固定使用 AKShare（新浪优先，东方财富降级）",
    }[source]
    history_source_summary = _source_summary(histories)
    result_warnings = list(dict.fromkeys(board_meta.get("warnings", []) + fetch_warnings))[:40]
    data_health = build_data_health(
        as_of=global_as_of.strftime("%Y-%m-%d"),
        expected_as_of=str(
            market_freshness.get("expected_latest_date")
            or signal_timing.get("expected_latest_completed_date")
            or ""
        )
        or None,
        freshness=market_freshness,
        sources={
            "board_constituents": board_meta.get("constituent_source"),
            **history_source_summary,
            "daily_factors": daily_factor_meta.get("source") or daily_factor_meta.get("sources"),
        },
        warehouse=warehouse_range,
        warnings=result_warnings,
        errors=fetch_errors,
        constituent_history=constituent_history,
    )
    result: dict[str, Any] = {
        "status": "ok",
        "analysis_type": "board_selection_t3_prediction",
        "board": board_meta,
        "as_of": global_as_of.strftime("%Y-%m-%d"),
        "market_freshness": market_freshness,
        "signal_timing": signal_timing,
        "data_health": data_health,
        "prediction_horizons": ["T+1", "T+2", "T+3"],
        "prediction_horizon_definition": "T+1/T+2/T+3分别指在T+1开盘入场后第1/2/3个可卖出交易日；首个退出日是信号后的第2个市场交易日",
        "constituent_count": int(len(constituents)),
        "post_filter_count": int(len(filtered)),
        "history_stock_count": int(len(histories)),
        "feature_ready_stock_count": feature_ready_stock_count,
        "same_as_of_prediction_stock_count": int(len(latest)),
        "stale_date_excluded_count": int(len(stale_latest)),
        "stale_date_excluded_examples": [
            {
                "ts_code": str(item["ts_code"]),
                "name": str(item["name"]),
                "latest_trade_date": pd.Timestamp(item["trade_date"]).strftime("%Y-%m-%d"),
            }
            for item in stale_latest.to_dict("records")[:20]
        ],
        "model_sample_count": int(len(panel)),
        "validated_candidates": validated_candidates,
        "validated_candidate_count": int(len(validated_candidates)),
        "validated_candidate_total": int(len(eligible)),
        "selection": {
            "selection_id": current_selection_id,
            "offset": offset,
            "requested_top_n": requested_top_n,
            "applied_top_n": top_n,
            "max_batch_size": 8,
            "limit_notice": limit_notice,
            "rank_start": offset + 1 if validated_candidates else None,
            "rank_end": offset + len(validated_candidates) if validated_candidates else None,
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
        },
        "model_ranking": ranked[: max(offset + top_n, 8)],
        "validation": validation,
        "candidate_evidence_gate": {
            "passed_horizons": passed_horizon_labels,
            "rules": [
                "至少一个预测周期通过样本外验证",
                "只有通过验证的预测周期才参与加权收益和排名，剩余权重重新归一化",
                "模型最强预测周期必须是已通过验证的周期",
                "给定测算资金必须覆盖所属板块最低申报数量，且不超过配置的成交额参与率上限",
                "加权成本后收益、最强周期成本后收益、风险调整分数都必须为正",
                "最强周期还必须达到配置的成本后收益、校准上涨概率和样本外质量分弃权门槛",
            ],
            "abstention_thresholds": {
                "minimum_net_return_after_cost": float(
                    config["moxing"].get("abstain_min_net_return", 0.003)
                ),
                "minimum_positive_probability": float(
                    config["moxing"].get("abstain_min_positive_probability", 0.55)
                ),
                "minimum_quality_score": float(
                    config["moxing"].get("abstain_min_quality_score", 0.40)
                ),
            },
        },
        "cost_assumption": cost_meta,
        "data_provenance": {
            "board_constituents": board_meta["constituent_source"],
            **history_source_summary,
            "daily_factors": daily_factor_meta,
            "warehouse_panel_expansion": {
                "configured_limit_without_warehouse": base_max_board_stocks,
                "applied_limit": applied_max_board_stocks,
                "range": warehouse_range,
                "expanded": bool(
                    warehouse_range.get("ready")
                    and applied_max_board_stocks > base_max_board_stocks
                ),
            },
            "source_policy": f"行业成分优先 Tushare，免费降级依次为新浪和东方财富；{daily_source_policy}",
            "config_path": resolved_config,
        },
        "filter_summary": {
            "rejected_count": int(len(rejected)),
            "rejected_examples": rejected[:20],
        },
        "warnings": result_warnings,
        "fetch_errors": fetch_errors[:30],
        "methodology": {
            "model": "日K因子的HistGradientBoostingRegressor + 稳健缩放Ridge小型集成，T+1/T+2/T+3分别训练",
            "ensemble_weighting": "验证预测只使用训练期尾部校准的权重；生产权重再由完整样本外预测选择，禁止用同一外层验证真实值反向优化其自身预测",
            "validation": "按交易日期顺序切分并清除跨越切分点样本；同时要求Top-N扣成本收益为正且优于当日候选池；验证后用全部已知标签重训",
            "target": "T日收盘后生成信号，下一市场交易日开盘入场；预测入场后第1/2/3个可卖出交易日收盘相对入场开盘的收益",
            "calendar_alignment": "入口和出口按板块共同市场日期定位；停牌、缺行情、一字涨停入口或一字跌停退出均不生成收益标签",
            "daily_factor_scope": "同日历史估值精确匹配，叠加上证/沪深300/中证1000日K、板块宽度、相对强弱和流通市值中性化因子",
            "ranking": "只对通过验证的周期归一化加权成本后预测收益，再除以持有期波动率、ATR占比和1%下限中的最大值",
            "survivorship_bias": (
                "每次运行会把真实成分写入本地快照并报告成员变化；积累多个快照后可识别后续变化，"
                "但仓库建立前的历史成分仍不能倒推，当前训练继续明确标记该限制"
            ),
        },
        "a_share_rules": {
            "analysis_timing": "T日完整收盘后分析，且只在下一市场交易日开盘入口尚未过去时发布候选",
            "t_plus_one": "输出T+1指入场后第1个可卖出交易日，实际是信号后的第2个市场交易日；不输出不可执行的同日卖出",
            "max_holding": "最多输出入场后第3个可卖出交易日；系统不会输出更远预测",
            "price_limits": "不输出目标收盘价；仅展示模型未约束参考价和从信号收盘逐日推导、按0.01元取整的合法价格区间，实际T+1开盘确定后必须重算",
        },
        "scope_note": "产品永久只做日K分析与三交易日内预测；结果带验证指标和成本假设，不是价格承诺。验证质量低只表示证据较弱。",
        "execution_policy": "analysis_only：只输出分析候选和预测，永不连接券商、永不提交订单、永不自动交易。",
    }
    if not validated_candidates:
        if offset >= len(eligible) and eligible:
            result["no_validated_candidate_reason"] = "候选序列已经到末尾，没有更多通过证据门槛且未重复的股票。"
        elif not passed_horizon_labels:
            result["no_validated_candidate_reason"] = "T+1、T+2、T+3 均未通过样本外验证，没有形成有效候选证据。"
        else:
            result["no_validated_candidate_reason"] = "程序已弃权：通过验证的周期内，没有股票同时满足动态成本、成交容量、上涨概率和样本外质量门槛。"
    if validation["overall_quality_label"] == "low":
        result["risk_notice"] = "当前样本外验证质量为 low；即使存在正预测，也只能视为低强度分析证据。"
    if run_dir:
        try:
            result["artifacts"] = _save_artifacts(run_dir, result)
        except Exception as exc:
            result["artifact_error"] = str(exc)
    return result
