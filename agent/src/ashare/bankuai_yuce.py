"""Sector-constrained A-share selection with T+1/T+2/T+3 predictions."""

from __future__ import annotations

import json
import hashlib
import math
import time
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.ashare.chengben_huadian import CostScenario
from src.ashare.chengben_huadian import DEFAULT_CONFIG_PATH as COST_CONFIG_PATH
from src.ashare.chengben_huadian import _commission_rate, _load_cost_config
from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    _completed_market_history,
    _json_value,
    _market_data_freshness,
    _round_optional,
    akshare_zhilian,
    biaozhunhua_daima,
    huoqu_rili_xingqing,
    jiazai_lianghua_peizhi,
    jisuan_tezheng_biao,
    shi_a_gu,
)
from src.ashare.shuju_yuan import _latest_tushare_daily, _limit_rate, _load_or_fetch_stock_basic, _tushare_pro
from src.providers.llm import _ensure_dotenv
from src.tools.path_utils import safe_run_dir


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
    end = datetime.now().date()
    start = end - timedelta(days=history_calendar_days)
    histories: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for _, row in constituents.head(max_stocks).iterrows():
        code = str(row["ts_code"])
        name = str(row.get("name", ""))
        result = huoqu_rili_xingqing(
            code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            source=source,
        )
        if source == "auto" and result.adjustment == "raw_unadjusted":
            fallback = huoqu_rili_xingqing(
                code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                source="akshare",
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
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return histories, names, errors, warnings


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
        prices = (
            features[["trade_date", "open", "high", "low", "close"]]
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
            entry_high = entry_series.map(pd.to_numeric(prices["high"], errors="coerce"))
            entry_low = entry_series.map(pd.to_numeric(prices["low"], errors="coerce"))
            limit_rate = _limit_rate(code, names.get(code, ""))
            entry_limit_up = pd.to_numeric(features["close"], errors="coerce").map(
                lambda value: _round_price_tick(float(value) * (1.0 + limit_rate))
                if pd.notna(value)
                else np.nan
            )
            blocked_limit_up = (
                entry_low.notna()
                & entry_high.notna()
                & entry_limit_up.notna()
                & (entry_low >= entry_limit_up - 0.005)
                & (entry_high <= entry_limit_up + 0.005)
            )
            entry_open = entry_open.mask(blocked_limit_up)
            exit_close = exit_series.map(pd.to_numeric(prices["close"], errors="coerce"))
            features[f"entry_date_t{horizon}"] = entry_series
            features[f"entry_open_t{horizon}"] = entry_open
            features[f"entry_blocked_limit_up_t{horizon}"] = blocked_limit_up
            features[f"target_date_t{horizon}"] = exit_series
            features[f"target_close_t{horizon}"] = exit_close
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


def _build_model_pipeline(model_config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
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


def _load_cost_assumption(scenario_name: str) -> tuple[float, CostScenario, str, list[str]]:
    budget, scenarios, path, errors = _load_cost_config(str(COST_CONFIG_PATH))
    scenario = next((item for item in scenarios if item.name == scenario_name), None)
    if scenario is None:
        raise ValueError(f"交易成本配置中不存在场景：{scenario_name}")
    return float(budget), scenario, path, errors


def _buy_order_rule(ts_code: str) -> tuple[int, int, str]:
    """Return minimum shares, share increment and board label for a buy order."""
    normalized = str(ts_code).upper()
    digits = normalized.split(".")[0]
    if normalized.endswith(".BJ"):
        return 100, 1, "beijing"
    if digits.startswith(("688", "689")):
        return 200, 1, "star"
    return 100, 100, "main_or_chinext"


def _position_for_budget(ts_code: str, price: float, budget_yuan: float) -> dict[str, Any]:
    minimum, increment, board = _buy_order_rule(ts_code)
    valid = math.isfinite(float(price)) and float(price) > 0 and float(budget_yuan) > 0
    affordable = int(math.floor(float(budget_yuan) / float(price) + 1e-9)) if valid else 0
    if affordable < minimum:
        shares = 0
    elif increment == 1:
        shares = affordable
    else:
        shares = affordable // increment * increment
    actual_notional = float(shares) * float(price) if shares else 0.0
    return {
        "board": board,
        "minimum_buy_shares": int(minimum),
        "buy_share_increment": int(increment),
        "target_budget_yuan": round(float(budget_yuan), 2),
        "sizing_price": round(float(price), 3) if valid else None,
        "estimated_buy_shares": int(shares),
        "estimated_buy_notional_yuan": round(actual_notional, 2),
        "budget_utilization": round(actual_notional / float(budget_yuan), 6) if budget_yuan > 0 else None,
        "execution_feasible": bool(shares >= minimum),
    }


def _stock_roundtrip_cost(
    ts_code: str,
    price: float,
    budget_yuan: float,
    scenario: CostScenario,
) -> tuple[float | None, dict[str, Any]]:
    position = _position_for_budget(ts_code, price, budget_yuan)
    notional = float(position["estimated_buy_notional_yuan"])
    if not position["execution_feasible"] or notional <= 0:
        return None, {
            **position,
            "cost_scenario": scenario.name,
            "estimated_roundtrip_cost_rate": None,
            "reason": "目标资金不足以按所属板块的最低买入数量建仓",
        }
    buy_commission = _commission_rate(scenario.buy_commission_rate, scenario.min_commission_yuan, notional)
    sell_commission = _commission_rate(scenario.sell_commission_rate, scenario.min_commission_yuan, notional)
    buy_cost = buy_commission + scenario.transfer_fee_buy_rate + scenario.buy_slippage_bps / 10000.0
    sell_cost = (
        sell_commission
        + scenario.transfer_fee_sell_rate
        + scenario.stamp_tax_sell_rate
        + scenario.sell_slippage_bps / 10000.0
    )
    roundtrip = 1.0 - (1.0 - buy_cost) * (1.0 - sell_cost)
    return float(roundtrip), {
        **position,
        "cost_scenario": scenario.name,
        "estimated_roundtrip_cost_rate": round(float(roundtrip), 6),
        "stamp_tax_sell_rate": scenario.stamp_tax_sell_rate,
    }


def _top_n_validation_metrics(
    validation_frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    horizon: int,
    budget_yuan: float,
    scenario: CostScenario,
    top_n: int,
) -> dict[str, Any]:
    evaluation = validation_frame[["trade_date", "ts_code", f"entry_open_t{horizon}"]].copy()
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
    model_config = config["moxing"]
    horizons = [int(value) for value in model_config["horizons"]]
    dates = sorted(pd.to_datetime(panel["trade_date"].dropna().unique()))
    if len(dates) < 100:
        raise RuntimeError(f"可用交易日期只有 {len(dates)} 个，无法进行可靠的时序验证")
    validation_ratio = float(model_config.get("validation_ratio", 0.2))
    cutoff_index = max(60, min(len(dates) - 20, int(len(dates) * (1.0 - validation_ratio))))
    cutoff = pd.Timestamp(dates[cutoff_index])
    predictions = latest[["ts_code", "name", "trade_date", "close"] + FEATURE_COLUMNS].copy()
    validation: dict[str, Any] = {
        "split_method": "chronological_purged_holdout",
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "signal_and_execution": {
            "signal": "T日收盘后",
            "entry": "下一市场交易日（T+1）开盘",
            "T+1": "入场后第1个可卖出交易日收盘，即信号后的第2个市场交易日",
            "T+2": "入场后第2个可卖出交易日收盘",
            "T+3": "入场后第3个可卖出交易日收盘",
        },
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
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
        pipeline = _build_model_pipeline(model_config)
        pipeline.fit(train[FEATURE_COLUMNS], y_train)
        validation_prediction = np.clip(
            pipeline.predict(validation_frame[FEATURE_COLUMNS]),
            clip_low,
            clip_high,
        )
        mae = float(mean_absolute_error(y_validation, validation_prediction))
        baseline_value = float(np.median(y_train))
        baseline_mae = float(mean_absolute_error(y_validation, np.full(len(y_validation), baseline_value)))
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
        )
        minimum_rank_ic = float(model_config.get("min_mean_daily_rank_ic", 0.01))
        minimum_skill = float(model_config.get("min_skill_vs_baseline", 0.01))
        minimum_direction = float(model_config.get("min_direction_accuracy", 0.52))
        minimum_rank_days = int(model_config.get("min_rank_ic_days", 10))
        minimum_top_n_days = int(model_config.get("min_top_n_days", 10))
        validation_passed = bool(
            direction_accuracy >= minimum_direction
            and rank_ic >= minimum_rank_ic
            and rank_ic_days >= minimum_rank_days
            and skill >= minimum_skill
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
        final_pipeline = _build_model_pipeline(model_config)
        final_pipeline.fit(usable[FEATURE_COLUMNS], np.clip(y_full_raw, final_clip_low, final_clip_high))
        latest_prediction = np.clip(
            final_pipeline.predict(latest[FEATURE_COLUMNS]),
            final_clip_low,
            final_clip_high,
        )
        predictions[f"pred_t{horizon}"] = latest_prediction
        validation["horizons"][f"T+{horizon}"] = {
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation_frame)),
            "validation_start": validation_frame["trade_date"].min().strftime("%Y-%m-%d"),
            "validation_end": validation_frame["trade_date"].max().strftime("%Y-%m-%d"),
            "mae": round(mae, 6),
            "baseline_mae": round(baseline_mae, 6),
            "skill_vs_median_baseline": round(skill, 6),
            "direction_accuracy": round(direction_accuracy, 6),
            "mean_daily_rank_ic": round(rank_ic, 6),
            "rank_ic_days": int(rank_ic_days),
            "prediction_clip": [round(clip_low, 6), round(clip_high, 6)],
            "final_prediction_clip": [round(final_clip_low, 6), round(final_clip_high, 6)],
            "final_train_samples": int(len(usable)),
            "final_training_end": usable[target_date_column].max().strftime("%Y-%m-%d"),
            "retrained_on_all_labeled_data": True,
            **top_n_metrics,
            "quality_score": round(quality, 4),
            "quality_label": _quality_label(quality),
            "validation_thresholds": {
                "direction_accuracy": minimum_direction,
                "mean_daily_rank_ic": minimum_rank_ic,
                "rank_ic_days": minimum_rank_days,
                "skill_vs_median_baseline": minimum_skill,
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


def _roundtrip_cost(scenario_name: str) -> tuple[float, dict[str, Any]]:
    """Return a reference cost; stock rows recalculate it after legal lot sizing."""
    notional, scenario, path, errors = _load_cost_assumption(scenario_name)
    buy_commission = _commission_rate(scenario.buy_commission_rate, scenario.min_commission_yuan, notional)
    sell_commission = _commission_rate(scenario.sell_commission_rate, scenario.min_commission_yuan, notional)
    buy_cost = buy_commission + scenario.transfer_fee_buy_rate + scenario.buy_slippage_bps / 10000.0
    sell_cost = (
        sell_commission
        + scenario.transfer_fee_sell_rate
        + scenario.stamp_tax_sell_rate
        + scenario.sell_slippage_bps / 10000.0
    )
    roundtrip = 1.0 - (1.0 - buy_cost) * (1.0 - sell_cost)
    return float(roundtrip), {
        "scenario": scenario.name,
        "config_path": path,
        "notional_yuan": float(notional),
        "reference_roundtrip_cost_rate": round(float(roundtrip), 6),
        "estimated_roundtrip_cost_rate": round(float(roundtrip), 6),
        "estimated_roundtrip_cost_rate_is_reference_only": True,
        "stamp_tax_sell_rate": scenario.stamp_tax_sell_rate,
        "capital_assumption": "每只股票按给定测算资金、T+1最不利参考价格和所属板块最低申报数量重新计算成本，见个股明细",
        "config_errors": errors,
    }


def _apply_cost(gross_return: float, cost_rate: float) -> float:
    return (1.0 + gross_return) * (1.0 - cost_rate) - 1.0


def _round_price_tick(value: float) -> float:
    return float(Decimal(str(max(float(value), 0.01))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _price_limit_bounds(reference_price: float, limit_rate: float | None, sessions: int) -> tuple[float, float]:
    """Apply the daily limit and one-cent tick repeatedly from a reference price."""
    if limit_rate is None or not math.isfinite(float(limit_rate)) or float(limit_rate) <= 0:
        return 0.01, math.inf
    lower = _round_price_tick(reference_price)
    upper = lower
    for _ in range(max(1, int(sessions))):
        lower = _round_price_tick(lower * (1.0 - float(limit_rate)))
        upper = _round_price_tick(upper * (1.0 + float(limit_rate)))
    return lower, upper


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
        return {"status": "error", "error": f"量化配置无效：{exc}"}

    try:
        constituents, board_meta = huoqu_bankuai_chengfen(bankuai, bankuai_leixing=bankuai_leixing)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "board": bankuai}
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
    histories, names, fetch_errors, fetch_warnings = _fetch_histories(
        filtered,
        source=source,
        history_calendar_days=int(data_config.get("history_calendar_days", 540)),
        minimum_rows=int(data_config.get("minimum_history_rows", 120)),
        max_stocks=int(data_config.get("max_board_stocks", 20)),
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
    result: dict[str, Any] = {
        "status": "ok",
        "analysis_type": "board_selection_t3_prediction",
        "board": board_meta,
        "as_of": global_as_of.strftime("%Y-%m-%d"),
        "market_freshness": market_freshness,
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
                "给定测算资金必须覆盖所属板块最低申报数量",
                "加权成本后收益、最强周期成本后收益、风险调整分数都必须为正",
            ],
        },
        "cost_assumption": cost_meta,
        "data_provenance": {
            "board_constituents": board_meta["constituent_source"],
            **_source_summary(histories),
            "source_policy": f"行业成分优先 Tushare，免费降级依次为新浪和东方财富；{daily_source_policy}",
            "config_path": resolved_config,
        },
        "filter_summary": {
            "rejected_count": int(len(rejected)),
            "rejected_examples": rejected[:20],
        },
        "warnings": list(dict.fromkeys(board_meta.get("warnings", []) + fetch_warnings))[:40],
        "fetch_errors": fetch_errors[:30],
        "methodology": {
            "model": "HistGradientBoostingRegressor，T+1/T+2/T+3 分别训练",
            "validation": "按交易日期顺序切分并清除跨越切分点样本；同时要求Top-N扣成本收益为正且优于当日候选池；验证后用全部已知标签重训",
            "target": "T日收盘后生成信号，下一市场交易日开盘入场；预测入场后第1/2/3个可卖出交易日收盘相对入场开盘的收益",
            "calendar_alignment": "入口和出口按板块共同市场日期定位；股票在必需日期停牌或缺行情时不生成该样本",
            "ranking": "只对通过验证的周期归一化加权成本后预测收益，再除以持有期波动率、ATR占比和1%下限中的最大值",
            "survivorship_bias": "模型使用当前板块成分回看历史，验证结果仍可能含当前成分股带来的幸存者偏差",
        },
        "a_share_rules": {
            "analysis_timing": "T日完整收盘后分析，假设下一市场交易日（T+1）开盘作为持有期收益测算基准",
            "t_plus_one": "输出T+1指入场后第1个可卖出交易日，实际是信号后的第2个市场交易日；不输出不可执行的同日卖出",
            "max_holding": "最多输出入场后第3个可卖出交易日；系统不会输出更远预测",
            "price_limits": "不输出目标收盘价；仅展示模型未约束参考价和从信号收盘逐日推导、按0.01元取整的合法价格区间，实际T+1开盘确定后必须重算",
        },
        "scope_note": "预测是带有验证指标和成本假设的研究结果，不是价格承诺，也不回答用户应采取何种交易动作。验证质量低只表示证据较弱。",
        "execution_policy": "analysis_only：只输出分析候选和预测，永不连接券商、永不提交订单、永不自动交易。",
    }
    if not validated_candidates:
        if offset >= len(eligible) and eligible:
            result["no_validated_candidate_reason"] = "候选序列已经到末尾，没有更多通过证据门槛且未重复的股票。"
        elif not passed_horizon_labels:
            result["no_validated_candidate_reason"] = "T+1、T+2、T+3 均未通过样本外验证，没有形成有效候选证据。"
        else:
            result["no_validated_candidate_reason"] = "通过验证的周期内，没有股票同时满足成本、可执行性和风险调整证据门槛。"
    if validation["overall_quality_label"] == "low":
        result["risk_notice"] = "当前样本外验证质量为 low；即使存在正预测，也只能视为低强度分析证据。"
    if run_dir:
        try:
            result["artifacts"] = _save_artifacts(run_dir, result)
        except Exception as exc:
            result["artifact_error"] = str(exc)
    return result
