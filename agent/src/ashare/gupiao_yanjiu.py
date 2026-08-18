"""A-share history, technical indicators, and fundamental data capabilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ashare.shuju_yuan import (
    _tushare_pro,
    biaozhunhua_gupiao_daima,
    huoqu_gupiao_jichu_ziliao,
    huoqu_zhangdieting_guize,
)
from src.ashare.shichang_shuju import akshare_zhilian
from src.ashare.yinzi_gongcheng import (
    RAW_PRICE_VOLUME_FEATURE_COLUMNS as ENGINEERED_RAW_PRICE_VOLUME_FEATURE_COLUMNS,
    add_price_volume_factors,
)
from src.providers.llm import _ensure_dotenv

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "ma_gap_60",
    "ma_trend_5_20",
    "rsi_14",
    "macd_dif_pct",
    "macd_hist_pct",
    "atr_14_pct",
    "volatility_20",
    "drawdown_20",
    "position_20",
    "volume_ratio_5_20",
    "amplitude_1",
]
# ``FEATURE_COLUMNS`` remains the compact, backwards-compatible technical
# contract used by the analysis layer.  The model layer additionally consumes
# the registered continuous price/volume factors below.
RAW_PRICE_VOLUME_FEATURE_COLUMNS = list(ENGINEERED_RAW_PRICE_VOLUME_FEATURE_COLUMNS)
FINANCIAL_CRITICAL_FIELDS = ("roe_pct", "net_profit_yoy_pct", "debt_to_assets_pct")

# Kept for compatibility with callers that imported the old module global.  A
# failed adj_factor request is now isolated to that request and never disables
# adjustment for later stocks.
_ADJ_FACTOR_DISABLED_REASON = ""


@dataclass(frozen=True)
class XingqingJieguo:
    data: pd.DataFrame
    source: str
    adjustment: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def _digits_from_symbol(value: str) -> str:
    raw = str(value).strip().upper()
    raw = re.sub(r"^(SH|SZ|BJ)", "", raw)
    raw = re.sub(r"\.(SH|SZ|BJ)$", "", raw)
    return raw


def biaozhunhua_daima(value: str) -> str:
    """Normalize a mainland A-share stock code to Tushare format."""
    raw = str(value).strip().upper()
    if raw.startswith(("SH", "SZ", "BJ")) and "." not in raw:
        raw = f"{raw[2:]}.{raw[:2]}"
    return biaozhunhua_gupiao_daima(raw)


def shi_a_gu(value: str) -> bool:
    try:
        biaozhunhua_daima(value)
        return True
    except ValueError:
        return False


def _akshare_name_table() -> pd.DataFrame:
    """从 AKShare 实时读取代码名称表，不在本地持久化。"""
    import akshare as ak

    with akshare_zhilian():
        table = ak.stock_info_a_code_name().rename(columns={"code": "ts_code", "name": "name"})
    table = table[["ts_code", "name"]].copy()
    table["ts_code"] = table["ts_code"].astype(str).str.zfill(6)
    table = table[table["ts_code"].map(shi_a_gu)].copy()
    table["ts_code"] = table["ts_code"].map(biaozhunhua_daima)
    return table


def _match_stock_basic(table: pd.DataFrame, query: str) -> dict[str, Any] | None:
    if table is None or table.empty:
        return None
    frame = table.copy()
    if "ts_code" in frame.columns:
        frame["ts_code"] = frame["ts_code"].astype(str).map(
            lambda item: biaozhunhua_daima(item) if shi_a_gu(item) else item
        )
    raw = str(query).strip()
    if shi_a_gu(raw):
        code = biaozhunhua_daima(raw)
        hits = frame[frame.get("ts_code", pd.Series(dtype=str)) == code]
    else:
        if "name" not in frame.columns:
            return None
        names = frame["name"].fillna("").astype(str)
        hits = frame[names == raw]
        if hits.empty:
            hits = frame[names.str.contains(re.escape(raw), regex=True)]
            distinct_hits = hits.drop_duplicates(subset=["ts_code"] if "ts_code" in hits.columns else ["name"])
            if len(distinct_hits) > 1:
                candidates = []
                for _, candidate in distinct_hits.head(8).iterrows():
                    label = str(candidate.get("name") or "未知名称")
                    candidate_code = str(candidate.get("ts_code") or "未知代码")
                    candidates.append(f"{label}（{candidate_code}）")
                suffix = "等" if len(distinct_hits) > len(candidates) else ""
                raise ValueError(
                    f"股票名称“{raw}”匹配到多个候选：{'、'.join(candidates)}{suffix}；"
                    "请使用完整股票名称或 6 位股票代码"
                )
    if hits.empty:
        return None
    row = hits.iloc[0]
    return {str(key): _json_value(value) for key, value in row.items()}


def jiexi_gupiao(gupiao: str, *, source: str = "auto") -> tuple[str, dict[str, Any], list[str]]:
    """Resolve either a stock code or a Chinese stock name."""
    source = source.strip().lower()
    if source not in {"auto", "tushare", "akshare"}:
        raise ValueError("source 必须是 auto、tushare 或 akshare")
    warnings: list[str] = []
    raw = str(gupiao).strip()
    if shi_a_gu(raw):
        code = biaozhunhua_daima(raw)
        resolved: dict[str, Any] = {}
        if source in {"auto", "tushare"}:
            try:
                resolved = _match_stock_basic(
                    huoqu_gupiao_jichu_ziliao(_tushare_pro(), {}),
                    code,
                ) or resolved
            except Exception as exc:
                warnings.append(f"Tushare 股票名称暂不可用：{exc}")
        if not resolved.get("name") and source in {"auto", "akshare"}:
            try:
                resolved = _match_stock_basic(_akshare_name_table(), code) or resolved
                if resolved.get("name"):
                    warnings.append("股票名称来自 AKShare 实时接口")
            except Exception as exc:
                warnings.append(f"股票名称表暂不可用：{exc}")
        return code, resolved, warnings

    errors: list[str] = []
    if source in {"auto", "tushare"}:
        try:
            pro = _tushare_pro()
            table = huoqu_gupiao_jichu_ziliao(pro, {})
            match = _match_stock_basic(table, raw)
            if match and match.get("ts_code"):
                return biaozhunhua_daima(str(match["ts_code"])), match, warnings
            errors.append(f"Tushare 未找到股票名称：{raw}")
        except ValueError:
            raise
        except Exception as exc:
            errors.append(f"Tushare 名称解析失败：{exc}")
            if source == "tushare":
                raise RuntimeError(errors[-1]) from exc

    if source in {"auto", "akshare"}:
        try:
            table = _akshare_name_table()
            match = _match_stock_basic(table, raw)
            if match and match.get("ts_code"):
                if errors:
                    warnings.extend(errors)
                warnings.append("股票名称由 AKShare 免费接口解析")
                return biaozhunhua_daima(str(match["ts_code"])), match, warnings
        except ValueError:
            raise
        except Exception as exc:
            errors.append(f"AKShare 名称解析失败：{exc}")
    raise RuntimeError("；".join(errors) or f"无法识别股票：{gupiao}")


def _normalize_history(frame: pd.DataFrame, *, tushare: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if tushare:
        data = data.rename(columns={"vol": "volume"})
    else:
        data = data.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount_yuan",
                "涨跌幅": "pct_chg",
                "换手率": "turnover_rate",
                "date": "trade_date",
                "turnover": "turnover_rate",
            }
        )
    if "trade_date" not in data.columns:
        raise ValueError("行情缺少 trade_date/日期列")
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount", "amount_yuan", "pct_chg", "turnover_rate"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if tushare and "amount" in data.columns and "amount_yuan" not in data.columns:
        data["amount_yuan"] = data["amount"] * 1000.0
    elif not tushare and "amount" in data.columns and "amount_yuan" not in data.columns:
        data["amount_yuan"] = data["amount"]
    required = ["trade_date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"行情缺少字段：{missing}")
    return (
        data.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .reset_index(drop=True)
    )


def _completed_market_history(
    history: pd.DataFrame,
    *,
    latest_completed_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """按已由交易日历确认的日期排除未完成日线。"""
    if history is None or history.empty:
        return pd.DataFrame(), []
    expected = pd.to_datetime(latest_completed_date, errors="coerce")
    if pd.isna(expected):
        raise ValueError("latest_completed_date 必须是有效交易日")
    expected = pd.Timestamp(expected).normalize()
    dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    keep = dates <= expected
    dropped = int((~keep).sum())
    warnings: list[str] = []
    if dropped:
        warnings.append(f"已忽略 {dropped} 根尚未确认收盘的日线，技术分析只使用完整交易日")
    return history.loc[keep].copy().reset_index(drop=True), warnings


def _apply_qfq(pro: Any, code: str, start_date: str, end_date: str, data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    try:
        factors = pro.adj_factor(ts_code=code, start_date=start_date, end_date=end_date)
        if factors is None or factors.empty:
            return data, "raw_unadjusted", "Tushare adj_factor 返回空值，价格未复权"
        adj = factors[["trade_date", "adj_factor"]].copy()
        adj["trade_date"] = pd.to_datetime(adj["trade_date"], errors="coerce")
        adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        merged = data.merge(adj, on="trade_date", how="left")
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
        latest_factor = float(merged["adj_factor"].iloc[-1])
        if not math.isfinite(latest_factor) or latest_factor <= 0:
            return data, "raw_unadjusted", "Tushare adj_factor 无效，价格未复权"
        ratio = merged["adj_factor"] / latest_factor
        for column in ["open", "high", "low", "close", "pre_close"]:
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce") * ratio
        return merged.drop(columns=["adj_factor"]), "qfq_by_tushare_adj_factor", ""
    except Exception as exc:
        reason = f"Tushare adj_factor 本次请求不可用，使用未复权价格：{exc}"
        return data, "raw_unadjusted", reason


def huoqu_rili_xingqing(
    code: str,
    *,
    start_date: str,
    end_date: str,
    source: str = "auto",
) -> XingqingJieguo:
    """实时获取单股日线；Tushare 优先，失败时降级 AKShare，全程不落盘。"""
    normalized = biaozhunhua_daima(code)
    source = source.strip().lower()
    if source not in {"auto", "tushare", "akshare"}:
        raise ValueError("source 必须是 auto、tushare 或 akshare")
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    errors: list[str] = []
    warnings: list[str] = []
    raw_tushare_fallback: XingqingJieguo | None = None

    if source in {"auto", "tushare"}:
        try:
            pro = _tushare_pro()
            raw = pro.daily(ts_code=normalized, start_date=start, end_date=end)
            data = _normalize_history(raw, tushare=True)
            if data.empty:
                raise RuntimeError("返回空行情")
            data, adjustment, adjustment_warning = _apply_qfq(pro, normalized, start, end, data)
            if adjustment_warning:
                warnings.append(adjustment_warning)
            tushare_result = XingqingJieguo(data, "tushare", adjustment, tuple(warnings), tuple(errors))
            if adjustment == "raw_unadjusted" and source == "auto":
                raw_tushare_fallback = tushare_result
                warnings.append("自动模式要求复权口径，继续尝试 AKShare 前复权行情")
            else:
                return tushare_result
        except Exception as exc:
            errors.append(f"Tushare 日线失败：{exc}")
            if source == "tushare":
                return XingqingJieguo(pd.DataFrame(), "tushare", "unknown", tuple(warnings), tuple(errors))

    try:
        import akshare as ak

        with akshare_zhilian():
            digits, exchange = normalized.split(".")
            raw = pd.DataFrame()
            ak_errors: list[str] = []
            if exchange in {"SH", "SZ"}:
                try:
                    raw = ak.stock_zh_a_daily(
                        symbol=f"{exchange.lower()}{digits}",
                        start_date=start,
                        end_date=end,
                        adjust="qfq",
                    )
                except Exception as exc:
                    ak_errors.append(f"新浪前复权日线失败：{exc}")
            if raw is None or raw.empty:
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=digits,
                        period="daily",
                        start_date=start,
                        end_date=end,
                        adjust="qfq",
                    )
                except Exception as exc:
                    ak_errors.append(f"东方财富前复权日线失败：{exc}")
                    raise RuntimeError("；".join(ak_errors)) from exc
        data = _normalize_history(raw, tushare=False)
        if data.empty:
            raise RuntimeError("返回空行情")
        if errors:
            warnings.extend(errors)
        warnings.append(
            "行情已降级到 AKShare 免费聚合接口"
            if source == "auto"
            else "行情使用 AKShare 免费聚合接口"
        )
        return XingqingJieguo(data, "akshare", "qfq", tuple(warnings), tuple(errors))
    except Exception as exc:
        errors.append(f"AKShare 日线失败：{exc}")
        if raw_tushare_fallback is not None:
            fallback_warnings = list(raw_tushare_fallback.warnings)
            fallback_warnings.append("AKShare 前复权降级失败，只能使用 Tushare 未复权行情")
            return XingqingJieguo(
                raw_tushare_fallback.data,
                raw_tushare_fallback.source,
                raw_tushare_fallback.adjustment,
                tuple(fallback_warnings),
                tuple(errors),
            )
        return XingqingJieguo(pd.DataFrame(), "akshare", "unknown", tuple(warnings), tuple(errors))


def jisuan_tezheng_biao(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate leak-free daily technical features used by analysis and ML."""
    data = history.copy().sort_values("trade_date").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    previous_close = close.shift(1)

    for period in [1, 3, 5, 10, 20]:
        data[f"ret_{period}"] = close.pct_change(period, fill_method=None)
    for period in [5, 10, 20, 60]:
        average = close.rolling(period, min_periods=period).mean()
        data[f"ma_{period}"] = average
        data[f"ma_gap_{period}"] = close / average - 1.0
    data["ma_trend_5_20"] = data["ma_5"] / data["ma_20"] - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    both_flat = gain.eq(0) & loss.eq(0)
    only_gains = gain.gt(0) & loss.eq(0)
    only_losses = gain.eq(0) & loss.gt(0)
    data["rsi_14"] = rsi.mask(both_flat, 50.0).mask(only_gains, 100.0).mask(only_losses, 0.0)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema_12 - ema_26
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    histogram = 2.0 * (dif - dea)
    data["macd_dif"] = dif
    data["macd_dea"] = dea
    data["macd_hist"] = histogram
    data["macd_dif_pct"] = dif / close
    data["macd_hist_pct"] = histogram / close

    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_14_pct"] = data["atr_14"] / close
    daily_return = close.pct_change(fill_method=None)
    data["volatility_20"] = daily_return.rolling(20, min_periods=20).std() * math.sqrt(252)

    rolling_high = high.rolling(20, min_periods=20).max()
    rolling_low = low.rolling(20, min_periods=20).min()
    data["drawdown_20"] = close / rolling_high - 1.0
    spread = (rolling_high - rolling_low).replace(0, np.nan)
    data["position_20"] = ((close - rolling_low) / spread).fillna(0.5)
    data["support_20"] = rolling_low
    data["resistance_20"] = rolling_high

    volume_5 = volume.rolling(5, min_periods=5).mean()
    volume_20 = volume.rolling(20, min_periods=20).mean()
    data["volume_ratio_5_20"] = volume_5 / volume_20.replace(0, np.nan)
    data["amplitude_1"] = (high - low) / previous_close.replace(0, np.nan)
    # Keep all newly introduced factors in the same leak-free daily pipeline.
    # The helper is independent of model labels and works for both standalone
    # technical analysis and the stock/board training panels.
    data = add_price_volume_factors(data)
    return data.replace([np.inf, -np.inf], np.nan)


def _round_optional(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _round_optional(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value) if not isinstance(value, (str, int, bool)) else value


def _technical_score(latest: pd.Series) -> tuple[int, list[str]]:
    score = 50.0
    reasons: list[str] = []
    close = _round_optional(latest.get("close"))
    ma_5 = _round_optional(latest.get("ma_5"))
    ma_20 = _round_optional(latest.get("ma_20"))
    if close is not None and ma_20 is not None:
        if ma_5 is not None and close > ma_5 > ma_20:
            score += 14
            reasons.append("收盘价、MA5、MA20 呈多头顺序")
        elif close > ma_20:
            score += 7
            reasons.append("价格位于 MA20 上方")
        else:
            score -= 8
            reasons.append("价格未站上 MA20")

    ret_5 = _round_optional(latest.get("ret_5"))
    if ret_5 is not None:
        if 0.01 <= ret_5 <= 0.12:
            score += 10
            reasons.append("5 日动量为正且未进入极端区")
        elif ret_5 < -0.05:
            score -= 10
            reasons.append("5 日动量明显偏弱")
        elif ret_5 > 0.18:
            score -= 5
            reasons.append("5 日涨幅过快，短线回撤风险增大")

    rsi = _round_optional(latest.get("rsi_14"))
    if rsi is not None:
        if 45 <= rsi <= 70:
            score += 8
            reasons.append("RSI 位于相对健康区间")
        elif rsi >= 80:
            score -= 9
            reasons.append("RSI 进入高位过热区")
        elif rsi <= 30:
            score -= 5
            reasons.append("RSI 显示弱势超卖，不等于已经反转")

    macd_hist = _round_optional(latest.get("macd_hist"))
    if macd_hist is not None:
        if macd_hist > 0:
            score += 7
            reasons.append("MACD 柱为正")
        elif macd_hist < 0:
            score -= 4
            reasons.append("MACD 柱为负")
        else:
            reasons.append("MACD 柱接近零，本项不加减分")
    volatility = _round_optional(latest.get("volatility_20"))
    if volatility is not None and volatility > 0.55:
        score -= 10
        reasons.append("20 日年化波动率偏高")
    return int(round(max(0, min(100, score)))), reasons


def zongjie_jishu(history: pd.DataFrame) -> dict[str, Any]:
    features = jisuan_tezheng_biao(history)
    usable = features.dropna(subset=["ma_20", "rsi_14", "atr_14_pct", "volatility_20"])
    if usable.empty:
        raise RuntimeError("有效日线不足，至少需要约 21 个交易日")
    latest = usable.iloc[-1]
    score, reasons = _technical_score(latest)
    indicator_warnings: list[str] = []
    if _round_optional(latest.get("ma_60")) is None:
        indicator_warnings.append("历史不足 60 个交易日，MA60 暂不可用且未参与评分")
    if _round_optional(latest.get("macd_hist")) is None:
        indicator_warnings.append("历史不足以形成完整 MACD，MACD 暂不可用且未参与评分")
    return {
        "trade_date": _json_value(latest["trade_date"]),
        "close": _round_optional(latest["close"], 3),
        "returns": {f"{period}d": _round_optional(latest[f"ret_{period}"], 6) for period in [1, 3, 5, 10, 20]},
        "moving_averages": {f"ma{period}": _round_optional(latest[f"ma_{period}"], 3) for period in [5, 10, 20, 60]},
        "rsi_14": _round_optional(latest["rsi_14"], 2),
        "macd": {
            "dif": _round_optional(latest["macd_dif"], 4),
            "dea": _round_optional(latest["macd_dea"], 4),
            "histogram": _round_optional(latest["macd_hist"], 4),
        },
        "atr_14_pct": _round_optional(latest["atr_14_pct"], 6),
        "annualized_volatility_20": _round_optional(latest["volatility_20"], 6),
        "drawdown_from_20d_high": _round_optional(latest["drawdown_20"], 6),
        "position_in_20d_range": _round_optional(latest["position_20"], 6),
        "volume_ratio_5_to_20": _round_optional(latest["volume_ratio_5_20"], 4),
        "support_20": _round_optional(latest["support_20"], 3),
        "resistance_20": _round_optional(latest["resistance_20"], 3),
        "score_0_100": score,
        "score_interpretation": (
            "启发式技术状态分，只表示当前指标组合，不是上涨概率、收益预测或精确目标分"
        ),
        "evidence": reasons,
        "indicator_warnings": indicator_warnings,
    }


def _first_number(row: pd.Series | dict[str, Any], aliases: Iterable[str]) -> float | None:
    items = row.items() if hasattr(row, "items") else []
    normalized = [(str(key).lower(), value) for key, value in items]
    for alias in aliases:
        target = alias.lower()
        for key, value in normalized:
            if key == target or target in key:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(number):
                    return float(number)
    return None


def _akshare_info(code: str) -> tuple[dict[str, Any], list[str]]:
    import akshare as ak

    errors: list[str] = []
    result: dict[str, Any] = {}
    digits = code.split(".")[0]
    try:
        with akshare_zhilian():
            table = ak.stock_individual_info_em(symbol=digits)
        if table is not None and not table.empty and {"item", "value"}.issubset(table.columns):
            result.update({str(row["item"]): _json_value(row["value"]) for _, row in table.iterrows()})
    except Exception as exc:
        errors.append(f"AKShare 个股资料失败：{exc}")

    try:
        with akshare_zhilian():
            spot = ak.stock_zh_a_spot_em()
        hit = spot[spot["代码"].astype(str).str.zfill(6) == digits]
        if not hit.empty:
            row = hit.iloc[0]
            result.update(
                {
                    "名称": _json_value(row.get("名称")),
                    "动态市盈率": _json_value(row.get("市盈率-动态")),
                    "市净率": _json_value(row.get("市净率")),
                    "总市值": _json_value(row.get("总市值")),
                    "流通市值": _json_value(row.get("流通市值")),
                    "换手率": _json_value(row.get("换手率")),
                }
            )
    except Exception as exc:
        errors.append(f"AKShare 实时估值失败：{exc}")
    return result, errors


def _financial_missing_fields(financials: dict[str, Any]) -> list[str]:
    return [field for field in FINANCIAL_CRITICAL_FIELDS if _round_optional(financials.get(field)) is None]


def _akshare_financials(code: str, *, as_of: str | None = None) -> tuple[dict[str, Any], list[str]]:
    import akshare as ak

    errors: list[str] = []
    digits = code.split(".")[0]
    try:
        with akshare_zhilian():
            table = ak.stock_financial_analysis_indicator(symbol=digits, start_year=str(datetime.now().year - 4))
        if table is None or table.empty:
            raise RuntimeError("返回空表")
        date_column = next((column for column in ["日期", "报告期", "date"] if column in table.columns), None)
        if not date_column:
            raise RuntimeError("返回结果缺少报告期，无法保证分析时点一致")
        table = table.assign(_date=pd.to_datetime(table[date_column], errors="coerce")).dropna(subset=["_date"])
        if as_of is not None:
            as_of_date = pd.to_datetime(as_of, errors="coerce")
            if pd.isna(as_of_date):
                raise ValueError(f"无效的分析日期：{as_of}")
            table = table[table["_date"].dt.normalize() <= pd.Timestamp(as_of_date).normalize()]
        if table.empty:
            raise RuntimeError(f"截至 {as_of} 没有可用财务报告")
        table = table.sort_values("_date")
        row = table.iloc[-1]
        financials = {
            "report_date": pd.Timestamp(row["_date"]).strftime("%Y-%m-%d"),
            "announcement_date": None,
            "announcement_date_status": "AKShare 未提供公告日，仅在当前分析时点作为降级数据使用",
            "roe_pct": _first_number(row, ["净资产收益率", "加权净资产收益率", "roe"]),
            "gross_margin_pct": _first_number(row, ["销售毛利率", "毛利率", "grossprofit_margin"]),
            "net_margin_pct": _first_number(row, ["销售净利率", "净利率", "netprofit_margin"]),
            "debt_to_assets_pct": _first_number(row, ["资产负债率", "debt_to_assets"]),
            "revenue_yoy_pct": _first_number(row, ["主营业务收入增长率", "营业收入同比增长", "or_yoy"]),
            "net_profit_yoy_pct": _first_number(row, ["净利润增长率", "净利润同比增长", "netprofit_yoy"]),
            "eps": _first_number(row, ["基本每股收益", "摊薄每股收益", "basic_eps"]),
        }
        financials["missing_fields"] = _financial_missing_fields(financials)
        if financials["missing_fields"]:
            errors.append(f"AKShare 最新财务报告缺少关键字段：{', '.join(financials['missing_fields'])}")
        return financials, errors
    except Exception as exc:
        errors.append(f"AKShare 财务指标失败：{exc}")
        return {}, errors


def huoqu_jibenmian(
    code: str,
    *,
    trade_date: str,
    allow_current_snapshot: bool = False,
) -> dict[str, Any]:
    """Fetch profile, valuation, and financial indicators with explicit provenance."""
    as_of_date = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(as_of_date):
        raise ValueError(f"无效的分析日期：{trade_date}")
    as_of_date = pd.Timestamp(as_of_date).normalize()
    as_of_text = as_of_date.strftime("%Y-%m-%d")
    profile: dict[str, Any] = {}
    valuation: dict[str, Any] = {}
    financials: dict[str, Any] = {}
    sources: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    data_quality: dict[str, Any] = {}

    try:
        pro = _tushare_pro()
        basic_quality: dict[str, Any] = {}
        basic_all = huoqu_gupiao_jichu_ziliao(pro, basic_quality)
        data_quality["stock_basic"] = basic_quality.get("stock_basic", {})
        warnings.extend(str(item) for item in basic_quality.get("warnings", []))
        basic = basic_all[basic_all["ts_code"].astype(str) == code]
        if basic is not None and not basic.empty:
            profile = {str(key): _json_value(value) for key, value in basic.iloc[0].items()}
            sources["profile"] = str(
                data_quality["stock_basic"].get("source") or "tushare_live"
            )
    except Exception as exc:
        errors.append(f"Tushare 基本资料失败：{exc}")

    try:
        pro = _tushare_pro()
        basic_daily = pro.daily_basic(
            ts_code=code,
            trade_date=trade_date.replace("-", ""),
            fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb,total_mv,circ_mv",
        )
        if basic_daily is not None and not basic_daily.empty:
            dated = basic_daily.copy()
            trade_dates = (
                dated["trade_date"]
                if "trade_date" in dated.columns
                else pd.Series(pd.NaT, index=dated.index, dtype="datetime64[ns]")
            )
            dated["_trade_date"] = pd.to_datetime(trade_dates, errors="coerce")
            dated = dated.dropna(subset=["_trade_date"])
            dated = dated[dated["_trade_date"].dt.normalize() <= as_of_date].sort_values("_trade_date")
            if not dated.empty:
                row = dated.iloc[-1]
                valuation_date = pd.Timestamp(row["_trade_date"]).strftime("%Y-%m-%d")
                if valuation_date == as_of_text:
                    valuation = {
                        "as_of": valuation_date,
                        "pe_dynamic": _round_optional(row.get("pe")),
                        "pe_ttm": _round_optional(row.get("pe_ttm")),
                        "pb": _round_optional(row.get("pb")),
                        "total_market_value_yuan": _round_optional(float(row.get("total_mv")) * 10000 if pd.notna(row.get("total_mv")) else None, 2),
                        "circulating_market_value_yuan": _round_optional(float(row.get("circ_mv")) * 10000 if pd.notna(row.get("circ_mv")) else None, 2),
                        "turnover_rate_pct": _round_optional(row.get("turnover_rate")),
                        "volume_ratio": _round_optional(row.get("volume_ratio")),
                    }
                    sources["valuation"] = "tushare"
                else:
                    errors.append(
                        f"Tushare 估值日期为 {valuation_date}，与分析日 {as_of_text} 不一致，已忽略"
                    )
    except Exception as exc:
        errors.append(f"Tushare 估值失败：{exc}")

    try:
        pro = _tushare_pro()
        indicator = pro.fina_indicator(
            ts_code=code,
            fields=(
                "ts_code,ann_date,end_date,roe,roe_dt,grossprofit_margin,netprofit_margin,"
                "debt_to_assets,or_yoy,netprofit_yoy,ocf_to_or,basic_eps"
            ),
        )
        if indicator is not None and not indicator.empty:
            known = indicator.copy()
            announcement_dates = (
                known["ann_date"]
                if "ann_date" in known.columns
                else pd.Series(pd.NaT, index=known.index, dtype="datetime64[ns]")
            )
            report_dates = (
                known["end_date"]
                if "end_date" in known.columns
                else pd.Series(pd.NaT, index=known.index, dtype="datetime64[ns]")
            )
            known["_ann_date"] = pd.to_datetime(announcement_dates, errors="coerce")
            known["_end_date"] = pd.to_datetime(report_dates, errors="coerce")
            known = known.dropna(subset=["_ann_date", "_end_date"])
            known = known[
                (known["_ann_date"].dt.normalize() <= as_of_date)
                & (known["_end_date"].dt.normalize() <= as_of_date)
            ]
            known = known.sort_values(["_end_date", "_ann_date"])
            if known.empty:
                errors.append(f"Tushare 截至 {as_of_text} 没有已公告的财务指标")
            else:
                row = known.iloc[-1]
                roe = _round_optional(row.get("roe"))
                roe_diluted = _round_optional(row.get("roe_dt"))
                if roe is None:
                    roe = roe_diluted
                financials = {
                    "known_as_of": as_of_text,
                    "report_date": pd.Timestamp(row["_end_date"]).strftime("%Y-%m-%d"),
                    "announcement_date": pd.Timestamp(row["_ann_date"]).strftime("%Y-%m-%d"),
                    "roe_pct": roe,
                    "roe_diluted_pct": roe_diluted,
                    "gross_margin_pct": _round_optional(row.get("grossprofit_margin")),
                    "net_margin_pct": _round_optional(row.get("netprofit_margin")),
                    "debt_to_assets_pct": _round_optional(row.get("debt_to_assets")),
                    "revenue_yoy_pct": _round_optional(row.get("or_yoy")),
                    "net_profit_yoy_pct": _round_optional(row.get("netprofit_yoy")),
                    "operating_cashflow_to_revenue_pct": _round_optional(row.get("ocf_to_or")),
                    "eps": _round_optional(row.get("basic_eps")),
                }
                financials["missing_fields"] = _financial_missing_fields(financials)
                if financials["missing_fields"]:
                    errors.append(
                        f"截至 {as_of_text} 的最新已公告财报缺少关键字段："
                        f"{', '.join(financials['missing_fields'])}"
                    )
                sources["financials"] = "tushare"
    except Exception as exc:
        errors.append(f"Tushare 财务指标失败：{exc}")

    need_ak_info = not profile or not valuation
    if need_ak_info:
        try:
            ak_info, ak_errors = _akshare_info(code)
            errors.extend(ak_errors)
            if not profile and ak_info:
                profile = {
                    "ts_code": code,
                    "name": ak_info.get("股票简称") or ak_info.get("名称"),
                    "industry": ak_info.get("行业"),
                    "market": ak_info.get("市场"),
                    "list_date": ak_info.get("上市时间"),
                    "total_share": ak_info.get("总股本"),
                    "circulating_share": ak_info.get("流通股"),
                }
                sources["profile"] = "akshare"
            if not valuation and ak_info and allow_current_snapshot:
                valuation = {
                    "as_of": as_of_text,
                    "as_of_note": "AKShare 快照未提供原始交易日期，仅在最近完成交易日使用",
                    "pe_dynamic": _round_optional(ak_info.get("动态市盈率")),
                    "pe_ttm": None,
                    "pb": _round_optional(ak_info.get("市净率")),
                    "total_market_value_yuan": _round_optional(ak_info.get("总市值"), 2),
                    "circulating_market_value_yuan": _round_optional(ak_info.get("流通市值"), 2),
                    "turnover_rate_pct": _round_optional(ak_info.get("换手率")),
                    "volume_ratio": None,
                }
                sources["valuation"] = "akshare"
            elif not valuation and ak_info:
                errors.append(f"AKShare 实时估值与历史分析日 {as_of_text} 不一致，已忽略该快照")
        except Exception as exc:
            errors.append(f"AKShare 基本面降级失败：{exc}")

    if not financials:
        if allow_current_snapshot:
            try:
                financials, ak_errors = _akshare_financials(code, as_of=as_of_text)
                errors.extend(ak_errors)
                if financials:
                    financials["known_as_of"] = as_of_text
                    sources["financials"] = "akshare"
            except Exception as exc:
                errors.append(f"AKShare 财务指标降级失败：{exc}")
        else:
            errors.append(f"AKShare 财务指标缺少公告日，未用于历史分析日 {as_of_text}")

    return {
        "profile": profile,
        "valuation": valuation,
        "financials": financials,
        "sources": sources,
        "data_quality": data_quality,
        "warnings": warnings,
        "errors": errors,
    }


# 稳定公开接口；保留原私有函数名，避免既有调用方在架构迁移期间失效。
guolv_wanzheng_jiaoyiri_lishi = _completed_market_history
guifan_you_xian_shuzhi = _round_optional
