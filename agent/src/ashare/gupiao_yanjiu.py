"""A-share history, technical indicators, and fundamental data capabilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.ashare.macd_jiegou import yanpan_macd_jiegou
from src.ashare.shuju_yuan import (
    _tushare_pro,
    biaozhunhua_gupiao_daima,
    huoqu_gupiao_jichu_ziliao,
    huoqu_zhangdieting_guize,
)
from src.ashare.shichang_shuju import akshare_zhilian
from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu, WangluoQingqiuYichang
from src.ashare.shuju_zhiliang import VERIFIED_DAILY_VALUATION_SOURCES
from src.ashare.yinzi_gongcheng import (
    RAW_PRICE_VOLUME_FEATURE_COLUMNS as ENGINEERED_RAW_PRICE_VOLUME_FEATURE_COLUMNS,
    add_price_volume_factors,
    continuous_ewm,
    daily_atr,
    factor_definition,
)

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
# 分析层使用紧凑技术字段与统一登记的连续价量因子。
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


def jiexi_gupiao(
    gupiao: str,
    *,
    source: str = "auto",
    stock_basic_loader: Callable[[], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, Any], list[str]]:
    """Resolve either a stock code or a Chinese stock name."""
    source = source.strip().lower()
    if source not in {"auto", "tushare", "akshare"}:
        raise ValueError("source 必须是 auto、tushare 或 akshare")
    warnings: list[str] = []
    def load_stock_basic() -> pd.DataFrame:
        if stock_basic_loader is not None:
            return stock_basic_loader()[0]
        return huoqu_gupiao_jichu_ziliao(_tushare_pro(), {})

    raw = str(gupiao).strip()
    code_query = raw if shi_a_gu(raw) else ""
    if not code_query:
        for match in re.finditer(r"(?<!\d)(\d{6})(?!\d)", raw):
            candidate = match.group(1)
            if shi_a_gu(candidate):
                code_query = candidate
                break
    if code_query:
        code = biaozhunhua_daima(code_query)
        resolved: dict[str, Any] = {}
        if source in {"auto", "tushare"}:
            try:
                resolved = _match_stock_basic(
                    load_stock_basic(),
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
        if not resolved.get("name"):
            raise RuntimeError(f"必要A股身份不可用：{code} 尚未在本次真实股票目录中确认，不能只凭代码形状分析为股票")
        return code, resolved, warnings

    errors: list[str] = []
    if source in {"auto", "tushare"}:
        try:
            table = load_stock_basic()
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
    """计算确定性日线指标，缺失不填充，不跨缺失行拼接窗口。"""
    if history is None or history.empty:
        raise ValueError("日线行情为空")
    if "trade_date" not in history.columns:
        raise ValueError("日线行情缺少 trade_date")
    data = history.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    if data["trade_date"].isna().any() or data["trade_date"].duplicated().any():
        raise ValueError("日线存在无效或重复交易日期，不能跳过或任选记录计算指标")
    data = data.sort_values("trade_date").reset_index(drop=True)
    if data.empty:
        raise ValueError("日线行情没有有效交易日期")
    for column in ("open", "close", "high", "low", "volume", "amount_yuan"):
        values = pd.to_numeric(data.get(column, pd.Series(np.nan, index=data.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        data[column] = values.where(values.ge(0) if column in {"volume", "amount_yuan"} else values.gt(0))
    close, high, low, volume = (data[column] for column in ("close", "high", "low", "volume"))
    valid_range = high.ge(low) & close.between(low, high)
    high, low = high.where(valid_range), low.where(valid_range)
    previous_close = close.shift(1)

    for period in [1, 3, 5, 10, 20]:
        data[f"ret_{period}"] = close.pct_change(period, fill_method=None).where(
            close.rolling(period + 1, min_periods=period + 1).count().eq(period + 1)
        )
    for period in [5, 10, 20, 60]:
        average = close.rolling(period, min_periods=period).mean()
        data[f"ma_{period}"] = average
        data[f"ma_gap_{period}"] = close / average - 1.0
    data["ma_trend_5_20"] = data["ma_5"] / data["ma_20"] - 1.0

    delta = close.diff()
    gain = continuous_ewm(delta.clip(lower=0), alpha=1 / 14, min_periods=14)
    loss = continuous_ewm(-delta.clip(upper=0), alpha=1 / 14, min_periods=14)
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    both_flat = gain.eq(0) & loss.eq(0)
    only_gains = gain.gt(0) & loss.eq(0)
    only_losses = gain.eq(0) & loss.gt(0)
    data["rsi_14"] = rsi.mask(both_flat, 50.0).mask(only_gains, 100.0).mask(only_losses, 0.0)

    ema_12 = continuous_ewm(close, alpha=2 / 13, min_periods=12)
    ema_26 = continuous_ewm(close, alpha=2 / 27, min_periods=26)
    dif = ema_12 - ema_26
    dea = continuous_ewm(dif, alpha=2 / 10, min_periods=9)
    histogram = 2.0 * (dif - dea)
    data["macd_dif"] = dif
    data["macd_dea"] = dea
    data["macd_hist"] = histogram
    valid_close = close.where(close > 0)
    data["macd_dif_pct"] = dif / valid_close
    data["macd_dea_pct"] = dea / valid_close
    data["macd_gap_pct"] = (dif - dea) / valid_close
    data["macd_hist_pct"] = histogram / valid_close
    data["macd_dif_change_pct"] = data["macd_dif_pct"].diff()
    data["macd_dea_change_pct"] = data["macd_dea_pct"].diff()
    data["macd_gap_change_pct"] = data["macd_gap_pct"].diff()
    data["macd_zero_distance_pct"] = pd.concat(
        [data["macd_dif_pct"].abs(), data["macd_dea_pct"].abs()],
        axis=1,
    ).max(axis=1, skipna=False)

    data["atr_14"] = daily_atr(high, low, close)
    data["atr_14_pct"] = data["atr_14"] / close
    daily_return = close.pct_change(fill_method=None)
    data["volatility_20"] = daily_return.rolling(20, min_periods=20).std() * math.sqrt(252)

    rolling_high = high.rolling(20, min_periods=20).max()
    rolling_low = low.rolling(20, min_periods=20).min()
    data["drawdown_20"] = close / rolling_high - 1.0
    spread = rolling_high - rolling_low
    data["position_20"] = ((close - rolling_low) / spread.where(spread > 0)).mask(
        spread.eq(0) & close.eq(rolling_low), 0.5
    )
    data["support_20"] = rolling_low
    data["resistance_20"] = rolling_high

    volume_5 = volume.rolling(5, min_periods=5).mean()
    volume_20 = volume.rolling(20, min_periods=20).mean()
    data["volume_ratio_5_20"] = volume_5 / volume_20.replace(0, np.nan)
    data["amplitude_1"] = (high - low) / previous_close.replace(0, np.nan)
    # 统一计算单股及比较池日线的确定性价量因子。
    data = add_price_volume_factors(data)
    return data.replace([np.inf, -np.inf], np.nan)


def _round_optional(value: Any, digits: int | None = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


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


def _technical_evidence(latest: pd.Series) -> list[str]:

    reasons: list[str] = []
    close = _round_optional(latest.get("close"), None)
    ma_5 = _round_optional(latest.get("ma_5"), None)
    ma_20 = _round_optional(latest.get("ma_20"), None)
    if close is not None and ma_20 is not None:
        if ma_5 is not None and close > ma_5 > ma_20:

            reasons.append("收盘价、MA5、MA20 呈多头顺序")
        elif close > ma_20:

            reasons.append("价格位于 MA20 上方")
        else:

            reasons.append("价格未站上 MA20")

    ret_5 = _round_optional(latest.get("ret_5"), None)
    if ret_5 is not None:
        reasons.append(f"近5个交易日收益 {ret_5:.2%}，需结合波动尺度与成交量理解")

    rsi = _round_optional(latest.get("rsi_14"))
    if rsi is not None:
        reasons.append(f"14日RSI为 {rsi:.2f}，仅描述近期涨跌强弱，不能单独确认反转或买卖时点")

    macd_hist = _round_optional(latest.get("macd_hist"), None)
    if macd_hist is not None:
        if macd_hist > 0:

            reasons.append("MACD 柱为正（不代表刚发生金叉）")
        elif macd_hist < 0:

            reasons.append("MACD 柱为负（不代表刚发生死叉）")
        else:
            reasons.append("MACD 柱接近零")
    volatility = _round_optional(latest.get("volatility_20"), None)
    if volatility is not None and volatility > 0.55:

        reasons.append("20 日年化波动率偏高")
    return reasons


def zongjie_jishu(
    history: pd.DataFrame,
    *,
    macd_structure_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return zongjie_jishu_tezheng(
        jisuan_tezheng_biao(history), macd_structure_config=macd_structure_config,
    )


def zongjie_jishu_tezheng(
    features: pd.DataFrame,
    *,
    macd_structure_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """解释本次已计算的完整日线特征，供因子构造与技术复核共用。"""
    if features.empty:
        return {"status": "unavailable", "outcome": "data_unavailable", "reason": "没有有效日线", "missing_indicators": [], "evidence": []}
    latest = features.iloc[-1]
    reasons = _technical_evidence(latest)
    missing = [key for key in ("ma_5", "ma_10", "ma_20", "ma_60", "rsi_14", "macd_hist", "atr_14_pct", "volatility_20") if _round_optional(latest.get(key)) is None]
    indicator_warnings: list[str] = []
    if _round_optional(latest.get("ma_60")) is None:
        indicator_warnings.append("最近60条有效收盘价不足或有缺失，MA60 暂不可用")
    if _round_optional(latest.get("macd_hist")) is None:
        indicator_warnings.append("连续有效收盘价不足以形成完整 MACD，MACD 暂不可用")
    macd_structure = yanpan_macd_jiegou(features, macd_structure_config)
    if macd_structure.get("status") != "ok":
        reason = str(macd_structure.get("reason") or "MACD 结构研判暂不可用")
        indicator_warnings.append(f"MACD 结构研判未完成：{reason}")
    structure_status = str(macd_structure.get("status") or "error")
    if structure_status == "error":
        summary_status, summary_outcome = "error", "program_error"
    elif structure_status in {"unavailable", "insufficient_data"} or missing:
        summary_status, summary_outcome = "partial", "information_partial"
    else:
        summary_status, summary_outcome = "ok", "analysis_success"
    return {
        "status": summary_status,
        "outcome": summary_outcome,
        "trade_date": _json_value(latest["trade_date"]),
        "raw_indicators": {key: float(latest[key]) if pd.notna(latest[key]) else None for key in (
            "close", "ma_5", "ma_10", "ma_20", "ma_60", "ma_gap_20", "ma_trend_5_20", "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
            "macd_hist", "macd_hist_pct", "volume_ratio_5_20", "volatility_20",
            "directional_efficiency_20", "up_amount_dominance_20",
        )},
        "indicator_definitions": {key: factor_definition(key) for key in (
            "rsi_14", "macd_hist_pct", "atr_14_pct", "volatility_20", "position_20", "drawdown_20",
            "directional_efficiency_20", "up_amount_dominance_20",
        )},
        "calculation_basis": "同一份已核验完整日线；raw_indicators 保留计算精度，显示舍入不参与筛选",
        "close": _round_optional(latest["close"], 3),
        "returns": {f"{period}d": _round_optional(latest[f"ret_{period}"], 6) for period in [1, 3, 5, 10, 20]},
        "moving_averages": {f"ma{period}": _round_optional(latest[f"ma_{period}"], 3) for period in [5, 10, 20, 60]},
        "rsi_14": _round_optional(latest["rsi_14"], 2),
        "macd": {
            "dif": _round_optional(latest["macd_dif"], 4),
            "dea": _round_optional(latest["macd_dea"], 4),
            "histogram": _round_optional(latest["macd_hist"], 4),
            "dif_pct": _round_optional(latest["macd_dif_pct"], 8),
            "dea_pct": _round_optional(latest["macd_dea_pct"], 8),
            "gap_pct": _round_optional(latest["macd_gap_pct"], 8),
            "zero_distance_pct": _round_optional(latest["macd_zero_distance_pct"], 8),
        },
        "macd_structure": macd_structure,
        "atr_14_pct": _round_optional(latest["atr_14_pct"], 6),
        "annualized_volatility_20": _round_optional(latest["volatility_20"], 6),
        "drawdown_from_20d_high": _round_optional(latest["drawdown_20"], 6),
        "position_in_20d_range": _round_optional(latest["position_20"], 6),
        "volume_ratio_5_to_20": _round_optional(latest["volume_ratio_5_20"], 4),
        "support_20": _round_optional(latest["support_20"], 3),
        "resistance_20": _round_optional(latest["resistance_20"], 3),
        "missing_indicators": missing,
        "history_rows": int(len(features)),
        "interpretation": "逐项解释最新日线的实际指标，不计算综合分或强制买卖结论",
        "evidence": reasons,
        "indicator_warnings": indicator_warnings,
    }


def _akshare_info(
    code: str,
    *,
    realtime_loader: Callable[[], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
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
        if realtime_loader is None:
            with akshare_zhilian():
                spot = ak.stock_zh_a_spot_em()
        else:
            shared_spot, metadata = realtime_loader()
            if shared_spot.empty:
                raise RuntimeError(str(metadata.get("error") or "本次实时快照不可用"))
            spot = shared_spot.rename(columns={
                "ts_code": "代码", "name": "名称", "pe_dynamic": "市盈率-动态", "pb": "市净率",
                "total_market_value_yuan": "总市值", "circulating_market_value_yuan": "流通市值",
                "turnover_rate": "换手率",
            }).copy()
            spot["代码"] = spot["代码"].astype(str).str.split(".", regex=False).str[0]
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


def _gongkai_caiwu(code: str, *, as_of: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """公开财务主表：公告日和报告期均不得越过本次分析日，无积分依赖。"""
    cutoff = pd.Timestamp(as_of).normalize()
    metadata: dict[str, Any] = {"source": "eastmoney_financial_main", "persistence": "none"}
    try:
        with GongkaiShujuHTTPKehu("eastmoney_financial_main") as client:
            payload, endpoint = client.qingqiu_json(
                ["https://datacenter.eastmoney.com/securities/api/data/v1/get"],
                params={"reportName": "RPT_F10_FINANCE_MAINFINADATA", "columns": "ALL",
                        "filter": f'(SECUCODE="{code}")', "sortColumns": "REPORT_DATE",
                        "sortTypes": "-1", "pageSize": 32, "pageNumber": 1},
            )
            metadata.update(endpoint=endpoint, attempted_providers=[item.to_dict() for item in client.attempts])
    except WangluoQingqiuYichang as exc:
        metadata.update(status="unavailable", **exc.to_dict())
        return {}, [f"公开财务取数失败：{exc}"], metadata
    result = payload.get("result") or {}
    rows = result.get("data") if isinstance(result, dict) else None
    if payload.get("success") is not True or not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        metadata["status"] = "unavailable"
        return {}, ["公开财务接口没有返回有效明细"], metadata
    table = pd.DataFrame(rows)
    if not {"SECUCODE", "REPORT_DATE", "NOTICE_DATE"}.issubset(table.columns):
        metadata["status"] = "unavailable"
        return {}, ["公开财务缺少证券、报告期或公告日，不能核验时点"], metadata
    table = table[table["SECUCODE"].astype(str).eq(code)].copy()
    table["_report"] = pd.to_datetime(table["REPORT_DATE"], errors="coerce")
    table["_notice"] = pd.to_datetime(table["NOTICE_DATE"], errors="coerce")
    known = table.loc[table["_report"].notna() & table["_notice"].notna()
                      & table["_report"].dt.normalize().le(cutoff)
                      & table["_notice"].dt.normalize().le(cutoff)].sort_values(["_report", "_notice"])
    if known.empty:
        metadata["status"] = "unavailable"
        return {}, [f"截至 {as_of} 没有身份、报告期和公告日均通过核验的公开财务记录"], metadata
    row = known.iloc[-1]
    financials: dict[str, Any] = {
        "known_as_of": as_of,
        "report_date": row["_report"].strftime("%Y-%m-%d"),
        "announcement_date": row["_notice"].strftime("%Y-%m-%d"),
        "source": "eastmoney_financial_main",
    }
    fields = {"roe_pct": "ROEJQ", "gross_margin_pct": "XSMLL", "net_margin_pct": "XSJLL",
              "debt_to_assets_pct": "ZCFZL", "revenue_yoy_pct": "TOTALOPERATEREVETZ",
              "net_profit_yoy_pct": "PARENTNETPROFITTZ", "eps": "EPSJB"}
    for target, origin in fields.items():
        financials[target] = _round_optional(row.get(origin))
    cash_ratio = _round_optional(row.get("JYXJLYYSR"), 12)
    financials.update(
        operating_cashflow_to_revenue_raw_ratio=cash_ratio,
        operating_cashflow_to_revenue_pct=cash_ratio * 100 if cash_ratio is not None else None,
        cashflow_unit_status="verified_ratio_to_percent",
        cashflow_definition="来源JYXJLYYSR为经营活动现金净流量/营业收入（不是营业总收入）的比值；百分数由原值乘100",
    )
    financials["missing_fields"] = _financial_missing_fields(financials)
    metadata.update(status="partial" if financials["missing_fields"] else "ok",
                    report_date=financials["report_date"], announcement_date=financials["announcement_date"],
                    original_fields={**fields, "operating_cashflow_to_revenue_raw_ratio": "JYXJLYYSR"},
                    known_as_of=as_of, fetched_at=datetime.now().astimezone().isoformat(),
                    history_revision_note="按当前来源返回及公告日核验，不声称拥有历史版本财务快照")
    errors = ["公开财务缺少关键字段：" + ", ".join(financials["missing_fields"])] if financials["missing_fields"] else []
    return financials, errors, metadata


def huoqu_jibenmian(
    code: str,
    *,
    trade_date: str,
    allow_current_snapshot: bool = False,
    stock_basic_loader: Callable[[], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
    realtime_loader: Callable[[], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
    daily_basic_loader: Callable[[str], pd.DataFrame] | None = None,
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
        basic_quality: dict[str, Any] = {}
        if stock_basic_loader is None:
            basic_all = huoqu_gupiao_jichu_ziliao(_tushare_pro(), basic_quality)
        else:
            basic_all, basic_quality = stock_basic_loader()
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
        if daily_basic_loader is None:
            basic_daily = _tushare_pro().daily_basic(
                ts_code=code,
                trade_date=trade_date.replace("-", ""),
                fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb,total_mv,circ_mv",
            )
        else:
            basic_daily = daily_basic_loader(as_of_text)
        if basic_daily is not None and not basic_daily.empty:
            dated = basic_daily.copy()
            if "ts_code" not in dated:
                raise ValueError("日终估值缺少证券代码，无法核验目标身份")
            dated = dated[dated["ts_code"].astype(str).eq(code)]
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
                valuation_source = str(row.get("valuation_source") or basic_daily.attrs.get("source")
                                       or ("tushare_daily_basic" if daily_basic_loader is None else "unverified"))
                complete_daily = row.get("valuation_is_complete_daily", valuation_source == "tushare_daily_basic")
                verified_complete = complete_daily is True or isinstance(complete_daily, np.bool_) and bool(complete_daily)
                if valuation_date == as_of_text and valuation_source in VERIFIED_DAILY_VALUATION_SOURCES and verified_complete:
                    valuation = {
                        "as_of": valuation_date,
                        "valuation_trade_date": valuation_date,
                        "valuation_source": valuation_source,
                        "valuation_is_complete_daily": True,
                        "pe": _round_optional(row.get("pe")),
                        "pe_definition": "来源定义为总市值/净利润，保留原始口径，不改称动态或滚动市盈率",
                        "pe_ttm": _round_optional(row.get("pe_ttm")),
                        "pe_static": _round_optional(row.get("pe_static")),
                        "valuation_close": _round_optional(row.get("valuation_close")),
                        "pb": _round_optional(row.get("pb")),
                        "total_market_value_yuan": _round_optional(float(row.get("total_mv")) * 10000 if pd.notna(row.get("total_mv")) else None, 2),
                        "circulating_market_value_yuan": _round_optional(float(row.get("circ_mv")) * 10000 if pd.notna(row.get("circ_mv")) else None, 2),
                        "turnover_rate_pct": _round_optional(row.get("turnover_rate")),
                        "volume_ratio": _round_optional(row.get("volume_ratio")),
                    }
                    sources["valuation"] = valuation_source
                    data_quality["valuation"] = dict(basic_daily.attrs)
                else:
                    errors.append(
                        f"估值日期 {valuation_date}、来源 {valuation_source} 或完整日终标记未通过分析日 {as_of_text} 核验，已忽略"
                    )
    except Exception as exc:
        errors.append(f"Tushare 估值失败：{exc}")

    financials, financial_errors, financial_quality = _gongkai_caiwu(code, as_of=as_of_text)
    errors.extend(financial_errors)
    data_quality["financials"] = financial_quality
    if financials:
        sources["financials"] = "eastmoney_financial_main"
    data_quality["financial_source_policy"] = {
        "source": "eastmoney_financial_main", "requires_tushare_points": False,
        "tushare_fina_indicator": "not_requested",
        "reason": "当前账号已实测无fina_indicator权限，默认财务链路采用带公告日的公开来源",
    }

    need_ak_info = not profile or (not valuation and allow_current_snapshot)
    if need_ak_info:
        try:
            ak_info, ak_errors = (
                _akshare_info(code)
                if realtime_loader is None
                else _akshare_info(code, realtime_loader=realtime_loader)
            )
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
