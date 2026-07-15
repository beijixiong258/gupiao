"""Sector-constrained A-share selection with T+1/T+2/T+3 predictions."""

from __future__ import annotations

import json
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.ashare.chengben_huadian import DEFAULT_CONFIG_PATH as COST_CONFIG_PATH
from src.ashare.chengben_huadian import _commission_rate, _load_cost_config
from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    _json_value,
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
                if similarity >= 0.55:
                    candidates.append((similarity, kind, value))
        except Exception as exc:
            errors.append(f"{kind}板块列表失败：{exc}")

    candidates.sort(key=lambda item: (item[0], item[1] == "hangye"), reverse=True)
    if not candidates:
        candidates = [(0.5, kind, query) for kind in kinds]
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
    similarity, value = max(candidates, key=lambda item: item[0])
    return (value, float(similarity)) if similarity >= 0.55 else None


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
    except Exception:
        pass
    return _normalize_constituents(members), {
        "requested_name": query,
        "resolved_name": industry,
        "board_type": "hangye",
        "name_similarity": round(similarity, 4),
        "constituent_source": "tushare_stock_basic",
        "warnings": [],
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
    for similarity, kind, name, label in sorted(candidates, reverse=True):
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
        elif price is not None and price < float(settings.get("min_price", 2.0)):
            reason = "价格低于配置下限"
        elif price is not None and price > float(settings.get("max_price", 300.0)):
            reason = "价格高于配置上限"
        elif amount is not None and amount < float(settings.get("min_amount_yuan", 50_000_000)):
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
    active_source = source
    for _, row in constituents.head(max_stocks).iterrows():
        code = str(row["ts_code"])
        name = str(row.get("name", ""))
        result = huoqu_rili_xingqing(
            code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            source=active_source,
        )
        if result.data.empty:
            errors.append(f"{code} {name}: " + "；".join(result.errors))
        elif len(result.data) < minimum_rows:
            errors.append(f"{code} {name}: 有效日线只有 {len(result.data)} 行，少于 {minimum_rows}")
        else:
            data = result.data.copy()
            data["data_source"] = result.source
            data["adjustment"] = result.adjustment
            histories[code] = data
            names[code] = name
            warnings.extend(f"{code}: {item}" for item in result.warnings)
            if source == "auto" and result.source == "akshare" and any(
                "adj_factor" in item for item in result.warnings
            ):
                active_source = "akshare"
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return histories, names, errors, warnings


def goujian_moxing_shuju(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str],
    horizons: list[int],
) -> pd.DataFrame:
    """Build pooled panel samples and future-return labels without future features."""
    frames: list[pd.DataFrame] = []
    for code, history in histories.items():
        features = jisuan_tezheng_biao(history)
        features["ts_code"] = code
        features["name"] = names.get(code, "")
        for horizon in horizons:
            features[f"target_t{horizon}"] = features["close"].shift(-horizon) / features["close"] - 1.0
            features[f"target_date_t{horizon}"] = features["trade_date"].shift(-horizon)
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


def xunlian_yuce_moxing(
    panel: pd.DataFrame,
    latest: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train one purged chronological model per horizon and predict latest rows."""
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
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "horizons": {},
    }

    minimum_train = int(model_config.get("min_training_samples", 500))
    minimum_validation = int(model_config.get("min_validation_samples", 100))
    quantiles = model_config.get("prediction_clip_quantiles", [0.01, 0.99])
    lower_q, upper_q = float(quantiles[0]), float(quantiles[1])

    for horizon in horizons:
        target_column = f"target_t{horizon}"
        target_date_column = f"target_date_t{horizon}"
        usable = panel.dropna(subset=[target_column, target_date_column]).copy()
        usable["trade_date"] = pd.to_datetime(usable["trade_date"])
        usable[target_date_column] = pd.to_datetime(usable[target_date_column])
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
        pipeline = Pipeline(
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

        latest_prediction = np.clip(pipeline.predict(latest[FEATURE_COLUMNS]), clip_low, clip_high)
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
            "quality_score": round(quality, 4),
            "quality_label": _quality_label(quality),
            "validation_passed": bool(direction_accuracy >= 0.52 and rank_ic > 0 and skill > 0),
        }

    quality_values = [float(item["quality_score"]) for item in validation["horizons"].values()]
    validation["overall_quality_score"] = round(float(np.mean(quality_values)), 4)
    validation["overall_quality_label"] = _quality_label(float(validation["overall_quality_score"]))
    validation["passed_horizons"] = sum(bool(item["validation_passed"]) for item in validation["horizons"].values())
    return predictions, validation


def _roundtrip_cost(scenario_name: str) -> tuple[float, dict[str, Any]]:
    notional, scenarios, path, errors = _load_cost_config(str(COST_CONFIG_PATH))
    scenario = next((item for item in scenarios if item.name == scenario_name), None)
    if scenario is None:
        raise ValueError(f"交易成本配置中不存在场景：{scenario_name}")
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
        "estimated_roundtrip_cost_rate": round(float(roundtrip), 6),
        "stamp_tax_sell_rate": scenario.stamp_tax_sell_rate,
        "config_errors": errors,
    }


def _apply_cost(gross_return: float, cost_rate: float) -> float:
    return (1.0 + gross_return) * (1.0 - cost_rate) - 1.0


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
    constituent_map = constituents.set_index("ts_code").to_dict("index")
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        code = str(row["ts_code"])
        name = str(row["name"])
        close = float(row["close"])
        forecasts: dict[str, Any] = {}
        weighted_net = 0.0
        for horizon in horizons:
            gross = float(row[f"pred_t{horizon}"])
            net = _apply_cost(gross, cost_rate)
            quality = validation["horizons"][f"T+{horizon}"]
            forecasts[f"T+{horizon}"] = {
                "cumulative_close_return": round(gross, 6),
                "cumulative_close_return_pct": round(gross * 100.0, 3),
                "estimated_net_return_after_cost": round(net, 6),
                "estimated_net_return_after_cost_pct": round(net * 100.0, 3),
                "predicted_close": round(close * (1.0 + gross), 3),
                "model_quality": quality["quality_label"],
            }
            weighted_net += weights.get(horizon, 0.0) * net
        exit_candidates = sorted(passed_horizons) if passed_horizons else horizons
        best_horizon = max(
            exit_candidates,
            key=lambda value: forecasts[f"T+{value}"]["estimated_net_return_after_cost"],
        )
        annualized_volatility = float(row.get("volatility_20", 0.0) or 0.0)
        atr_pct = float(row.get("atr_14_pct", 0.0) or 0.0)
        trend = float(row.get("ma_trend_5_20", 0.0) or 0.0)
        risk_penalty = 0.01 * max(annualized_volatility, 0.0) + 0.10 * max(atr_pct, 0.0)
        trend_bonus = 0.10 * max(min(trend, 0.05), 0.0)
        selection_score = weighted_net - risk_penalty + trend_bonus
        current = constituent_map.get(code, {})
        rows.append(
            {
                "ts_code": code,
                "name": name,
                "as_of": pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d"),
                "latest_close": round(close, 3),
                "selection_score": round(selection_score, 6),
                "weighted_expected_net_return": round(weighted_net, 6),
                "suggested_exit": f"T+{best_horizon}",
                "suggested_holding_trading_days": int(best_horizon),
                "suggested_exit_validation_passed": bool(best_horizon in passed_horizons),
                "forecast": forecasts,
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
                    "can_sell_earliest": "T+1",
                    "price_limit_pct": round(_limit_rate(code, name) * 100.0, 2),
                },
            }
        )
    return sorted(rows, key=lambda item: item["selection_score"], reverse=True)


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
    for item in result.get("recommendations", []):
        row = {
            "ts_code": item["ts_code"],
            "name": item["name"],
            "as_of": item["as_of"],
            "latest_close": item["latest_close"],
            "selection_score": item["selection_score"],
            "suggested_exit": item["suggested_exit"],
        }
        for horizon in [1, 2, 3]:
            forecast = item["forecast"][f"T+{horizon}"]
            row[f"t{horizon}_gross_return"] = forecast["cumulative_close_return"]
            row[f"t{horizon}_net_return"] = forecast["estimated_net_return_after_cost"]
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {"json": str(json_path), "csv": str(csv_path)}


def bankuai_xuangu(
    *,
    bankuai: str,
    bankuai_leixing: str = "auto",
    top_n: int = 3,
    source: str = "auto",
    config_path: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """Select stocks from a specified A-share board and predict only T+1 to T+3."""
    _ensure_dotenv()
    top_n = max(1, min(int(top_n), 10))
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
        and item["suggested_exit_validation_passed"]
        and float(item["weighted_expected_net_return"]) > 0
        and float(item["selection_score"]) > 0
        and float(item["forecast"][item["suggested_exit"]]["estimated_net_return_after_cost"]) > 0
    ]
    recommendations = eligible[:top_n]
    result: dict[str, Any] = {
        "status": "ok",
        "analysis_type": "board_selection_t3_prediction",
        "board": board_meta,
        "as_of": global_as_of.strftime("%Y-%m-%d"),
        "prediction_horizons": ["T+1", "T+2", "T+3"],
        "constituent_count": int(len(constituents)),
        "post_filter_count": int(len(filtered)),
        "history_stock_count": int(len(histories)),
        "model_sample_count": int(len(panel)),
        "recommendations": recommendations,
        "recommendation_count": int(len(recommendations)),
        "watchlist": ranked[: max(top_n, 5)],
        "validation": validation,
        "recommendation_gate": {
            "passed_horizons": passed_horizon_labels,
            "rules": [
                "至少一个预测周期通过样本外验证",
                "建议卖出周期必须是已通过验证的周期",
                "加权成本后收益、建议周期成本后收益、风险调整分数都必须为正",
            ],
        },
        "cost_assumption": cost_meta,
        "data_provenance": {
            "board_constituents": board_meta["constituent_source"],
            **_source_summary(histories),
            "source_policy": "行业成分优先 Tushare；免费降级依次为新浪和东方财富。个股日线优先 Tushare，失败后降级 AKShare",
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
            "validation": "按交易日期顺序切分，并清除目标日期跨越切分点的训练样本",
            "target": "从最新收盘价到未来第 1/2/3 个交易日收盘价的累计收益",
            "ranking": "成本后预测收益加权，扣除 ATR 与波动率风险惩罚",
            "survivorship_bias": "模型使用当前板块成分回看历史，验证结果仍可能含当前成分股带来的幸存者偏差",
        },
        "a_share_rules": {
            "t_plus_one": "当日买入最早下一个交易日卖出，因此不输出 T+0",
            "max_holding": "最多 T+3；系统不会输出更远预测",
            "price_limits": "过滤最新交易日接近涨停的股票，并按主板、ST、创业板/科创板、北交所识别涨跌幅限制",
        },
        "scope_note": "预测是有验证指标和成本假设的概率研究，不是价格承诺。验证质量低时应降低仓位或不交易。",
        "execution_policy": "research_only：只输出研究候选和预测，永不连接券商、永不提交订单、永不自动交易。",
    }
    if not recommendations:
        if not passed_horizon_labels:
            result["no_trade_reason"] = "T+1、T+2、T+3 均未通过样本外验证，系统选择不推荐任何股票。"
        else:
            result["no_trade_reason"] = "通过验证的周期内，没有股票同时覆盖交易成本和风险惩罚，系统选择不推荐。"
    if validation["overall_quality_label"] == "low":
        result["risk_notice"] = "当前样本外验证质量为 low；即使存在正预测，也不应把它当作高确信度交易信号。"
    if run_dir:
        try:
            result["artifacts"] = _save_artifacts(run_dir, result)
        except Exception as exc:
            result["artifact_error"] = str(exc)
    return result
