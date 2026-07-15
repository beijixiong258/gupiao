"""Shared A-share data, technical indicators, and single-stock research."""

from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ashare.shuju_yuan import STOCK_BASIC_CACHE, _limit_rate, _load_or_fetch_stock_basic, _tushare_pro
from src.providers.llm import _ensure_dotenv
from src.tools.path_utils import safe_run_dir

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "lianghua_peizhi.json"
AK_STOCK_NAMES_CACHE = STOCK_BASIC_CACHE.parent / "akshare_stock_names.csv"

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

_ADJ_FACTOR_DISABLED_REASON = ""
_PROXY_ENV_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@dataclass(frozen=True)
class XingqingJieguo:
    data: pd.DataFrame
    source: str
    adjustment: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def jiazai_lianghua_peizhi(config_path: str | None = None) -> tuple[dict[str, Any], str]:
    """Load and validate the external quant configuration."""
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"量化配置文件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("量化配置必须是 JSON 对象")
    horizons = value.get("moxing", {}).get("horizons")
    if horizons != [1, 2, 3]:
        raise ValueError("moxing.horizons 必须严格为 [1, 2, 3]")
    max_holding_days = int(value.get("jiaoyi", {}).get("max_holding_days", 0))
    if max_holding_days != 3:
        raise ValueError("jiaoyi.max_holding_days 必须为 3")
    trading = value.get("jiaoyi", {})
    if (
        trading.get("execution_mode") != "research_only"
        or trading.get("allow_live_trading") is not False
        or trading.get("allow_order_submission") is not False
    ):
        raise ValueError("本程序被硬限制为 research_only，禁止实盘交易和订单提交")
    return value, str(path)


def _akshare_bypass_proxy_enabled() -> bool:
    override = os.getenv("GPYJ_AKSHARE_BYPASS_PROXY", "").strip().lower()
    if override:
        return override not in {"0", "false", "no", "off"}
    try:
        config, _ = jiazai_lianghua_peizhi()
        return bool(config.get("shuju", {}).get("akshare_bypass_proxy", True))
    except Exception:
        return True


@contextmanager
def akshare_zhilian():
    """Temporarily bypass proxy variables for mainland AKShare endpoints."""
    if not _akshare_bypass_proxy_enabled():
        yield
        return
    saved = {name: os.environ[name] for name in _PROXY_ENV_NAMES if name in os.environ}
    try:
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        yield
    finally:
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(saved)


def _digits_from_symbol(value: str) -> str:
    raw = str(value).strip().upper()
    raw = re.sub(r"^(SH|SZ|BJ)", "", raw)
    raw = re.sub(r"\.(SH|SZ|BJ)$", "", raw)
    return raw


def biaozhunhua_daima(value: str) -> str:
    """Normalize a mainland A-share stock code to Tushare format."""
    raw = str(value).strip().upper()
    digits = _digits_from_symbol(raw)
    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"不是有效的 6 位 A 股代码：{value}")

    suffix = ""
    match = re.search(r"\.(SH|SZ|BJ)$", raw)
    if match:
        suffix = match.group(1)
    elif raw.startswith(("SH", "SZ", "BJ")):
        suffix = raw[:2]

    expected = ""
    if digits.startswith(("600", "601", "603", "605", "688", "689")):
        expected = "SH"
    elif digits.startswith(("000", "001", "002", "003", "300", "301")):
        expected = "SZ"
    elif digits.startswith(("4", "8", "9")):
        expected = "BJ"
    if not expected:
        raise ValueError(f"代码不属于本系统支持的 A 股普通股票范围：{value}")
    if suffix and suffix != expected:
        raise ValueError(f"代码与交易所后缀不一致：{value}，应为 .{expected}")
    return f"{digits}.{expected}"


def shi_a_gu(value: str) -> bool:
    try:
        biaozhunhua_daima(value)
        return True
    except ValueError:
        return False


def _stock_basic_cache() -> pd.DataFrame:
    if not STOCK_BASIC_CACHE.is_file():
        return pd.DataFrame()


def _akshare_name_table() -> pd.DataFrame:
    if AK_STOCK_NAMES_CACHE.is_file():
        try:
            cached = pd.read_csv(AK_STOCK_NAMES_CACHE, dtype=str)
            if not cached.empty and {"ts_code", "name"}.issubset(cached.columns):
                return cached
        except Exception:
            pass
    import akshare as ak

    with akshare_zhilian():
        table = ak.stock_info_a_code_name().rename(columns={"code": "ts_code", "name": "name"})
    table = table[["ts_code", "name"]].copy()
    table["ts_code"] = table["ts_code"].astype(str).str.zfill(6)
    table = table[table["ts_code"].map(shi_a_gu)].copy()
    table["ts_code"] = table["ts_code"].map(biaozhunhua_daima)
    AK_STOCK_NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(AK_STOCK_NAMES_CACHE, index=False, encoding="utf-8-sig")
    return table
    try:
        return pd.read_csv(STOCK_BASIC_CACHE, dtype=str)
    except Exception:
        return pd.DataFrame()


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
        cached = _match_stock_basic(_stock_basic_cache(), code) or {}
        if not cached.get("name"):
            try:
                cached = _match_stock_basic(_akshare_name_table(), code) or cached
                if cached.get("name"):
                    warnings.append("股票名称来自 AKShare 本地代码表缓存")
            except Exception as exc:
                warnings.append(f"股票名称表暂不可用：{exc}")
        return code, cached, warnings

    cached = _match_stock_basic(_stock_basic_cache(), raw)
    if cached and cached.get("ts_code"):
        return biaozhunhua_daima(str(cached["ts_code"])), cached, warnings

    errors: list[str] = []
    if source in {"auto", "tushare"}:
        try:
            pro = _tushare_pro()
            table = _load_or_fetch_stock_basic(pro, {})
            match = _match_stock_basic(table, raw)
            if match and match.get("ts_code"):
                return biaozhunhua_daima(str(match["ts_code"])), match, warnings
            errors.append(f"Tushare 未找到股票名称：{raw}")
        except Exception as exc:
            errors.append(f"Tushare 名称解析失败：{exc}")
            if source == "tushare":
                raise RuntimeError(errors[-1]) from exc

    try:
        table = _akshare_name_table()
        match = _match_stock_basic(table, raw)
        if match and match.get("ts_code"):
            if errors:
                warnings.extend(errors)
            warnings.append("股票名称由 AKShare 免费接口解析")
            return biaozhunhua_daima(str(match["ts_code"])), match, warnings
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


def _apply_qfq(pro: Any, code: str, start_date: str, end_date: str, data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    global _ADJ_FACTOR_DISABLED_REASON
    if _ADJ_FACTOR_DISABLED_REASON:
        return data, "raw_unadjusted", _ADJ_FACTOR_DISABLED_REASON
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
        _ADJ_FACTOR_DISABLED_REASON = f"Tushare adj_factor 不可用，使用未复权价格：{exc}"
        return data, "raw_unadjusted", _ADJ_FACTOR_DISABLED_REASON


def huoqu_rili_xingqing(
    code: str,
    *,
    start_date: str,
    end_date: str,
    source: str = "auto",
) -> XingqingJieguo:
    """Fetch one stock's daily bars with Tushare-first fallback semantics."""
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
        warnings.append("行情已降级到 AKShare 免费聚合接口")
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
    data["rsi_14"] = (100.0 - 100.0 / (1.0 + relative_strength)).fillna(100.0)

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
    if latest.get("close", 0) > latest.get("ma_5", math.inf) > latest.get("ma_20", math.inf):
        score += 14
        reasons.append("收盘价、MA5、MA20 呈多头顺序")
    elif latest.get("close", 0) > latest.get("ma_20", math.inf):
        score += 7
        reasons.append("价格位于 MA20 上方")
    else:
        score -= 8
        reasons.append("价格位于 MA20 下方")

    ret_5 = float(latest.get("ret_5", 0) or 0)
    if 0.01 <= ret_5 <= 0.12:
        score += 10
        reasons.append("5 日动量为正且未进入极端区")
    elif ret_5 < -0.05:
        score -= 10
        reasons.append("5 日动量明显偏弱")
    elif ret_5 > 0.18:
        score -= 5
        reasons.append("5 日涨幅过快，短线回撤风险增大")

    rsi = float(latest.get("rsi_14", 50) or 50)
    if 45 <= rsi <= 70:
        score += 8
        reasons.append("RSI 位于相对健康区间")
    elif rsi >= 80:
        score -= 9
        reasons.append("RSI 进入高位过热区")
    elif rsi <= 30:
        score -= 5
        reasons.append("RSI 显示弱势超卖，不等于已经反转")

    if float(latest.get("macd_hist", 0) or 0) > 0:
        score += 7
        reasons.append("MACD 柱为正")
    else:
        score -= 4
        reasons.append("MACD 柱为负")
    volatility = float(latest.get("volatility_20", 0) or 0)
    if volatility > 0.55:
        score -= 10
        reasons.append("20 日年化波动率偏高")
    return int(round(max(0, min(100, score)))), reasons


def zongjie_jishu(history: pd.DataFrame) -> dict[str, Any]:
    features = jisuan_tezheng_biao(history)
    usable = features.dropna(subset=["ma_20", "rsi_14", "atr_14_pct", "volatility_20"])
    if usable.empty:
        raise RuntimeError("有效日线不足，至少需要约 60 个交易日")
    latest = usable.iloc[-1]
    score, reasons = _technical_score(latest)
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
        "evidence": reasons,
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


def _akshare_financials(code: str) -> tuple[dict[str, Any], list[str]]:
    import akshare as ak

    errors: list[str] = []
    digits = code.split(".")[0]
    try:
        with akshare_zhilian():
            table = ak.stock_financial_analysis_indicator(symbol=digits, start_year=str(datetime.now().year - 4))
        if table is None or table.empty:
            raise RuntimeError("返回空表")
        date_column = next((column for column in ["日期", "报告期", "date"] if column in table.columns), None)
        if date_column:
            table = table.assign(_date=pd.to_datetime(table[date_column], errors="coerce")).sort_values("_date")
        row = table.iloc[-1]
        return {
            "report_date": _json_value(row.get(date_column)) if date_column else None,
            "roe_pct": _first_number(row, ["净资产收益率", "加权净资产收益率", "roe"]),
            "gross_margin_pct": _first_number(row, ["销售毛利率", "毛利率", "grossprofit_margin"]),
            "net_margin_pct": _first_number(row, ["销售净利率", "净利率", "netprofit_margin"]),
            "debt_to_assets_pct": _first_number(row, ["资产负债率", "debt_to_assets"]),
            "revenue_yoy_pct": _first_number(row, ["主营业务收入增长率", "营业收入同比增长", "or_yoy"]),
            "net_profit_yoy_pct": _first_number(row, ["净利润增长率", "净利润同比增长", "netprofit_yoy"]),
            "eps": _first_number(row, ["基本每股收益", "摊薄每股收益", "basic_eps"]),
        }, errors
    except Exception as exc:
        errors.append(f"AKShare 财务指标失败：{exc}")
        return {}, errors


def huoqu_jibenmian(code: str, *, trade_date: str) -> dict[str, Any]:
    """Fetch profile, valuation, and financial indicators with explicit provenance."""
    profile: dict[str, Any] = {}
    valuation: dict[str, Any] = {}
    financials: dict[str, Any] = {}
    sources: dict[str, str] = {}
    errors: list[str] = []

    try:
        pro = _tushare_pro()
        basic_all = _load_or_fetch_stock_basic(pro, {})
        basic = basic_all[basic_all["ts_code"].astype(str) == code]
        if basic is not None and not basic.empty:
            profile = {str(key): _json_value(value) for key, value in basic.iloc[0].items()}
            sources["profile"] = "tushare"
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
            row = basic_daily.iloc[0]
            valuation = {
                "pe_dynamic": _round_optional(row.get("pe")),
                "pe_ttm": _round_optional(row.get("pe_ttm")),
                "pb": _round_optional(row.get("pb")),
                "total_market_value_yuan": _round_optional(float(row.get("total_mv")) * 10000 if pd.notna(row.get("total_mv")) else None, 2),
                "circulating_market_value_yuan": _round_optional(float(row.get("circ_mv")) * 10000 if pd.notna(row.get("circ_mv")) else None, 2),
                "turnover_rate_pct": _round_optional(row.get("turnover_rate")),
                "volume_ratio": _round_optional(row.get("volume_ratio")),
            }
            sources["valuation"] = "tushare"
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
            row = indicator.sort_values("end_date").iloc[-1]
            financials = {
                "report_date": _json_value(row.get("end_date")),
                "announcement_date": _json_value(row.get("ann_date")),
                "roe_pct": _round_optional(row.get("roe")),
                "roe_diluted_pct": _round_optional(row.get("roe_dt")),
                "gross_margin_pct": _round_optional(row.get("grossprofit_margin")),
                "net_margin_pct": _round_optional(row.get("netprofit_margin")),
                "debt_to_assets_pct": _round_optional(row.get("debt_to_assets")),
                "revenue_yoy_pct": _round_optional(row.get("or_yoy")),
                "net_profit_yoy_pct": _round_optional(row.get("netprofit_yoy")),
                "operating_cashflow_to_revenue_pct": _round_optional(row.get("ocf_to_or")),
                "eps": _round_optional(row.get("basic_eps")),
            }
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
            if not valuation and ak_info:
                valuation = {
                    "pe_dynamic": _round_optional(ak_info.get("动态市盈率")),
                    "pe_ttm": None,
                    "pb": _round_optional(ak_info.get("市净率")),
                    "total_market_value_yuan": _round_optional(ak_info.get("总市值"), 2),
                    "circulating_market_value_yuan": _round_optional(ak_info.get("流通市值"), 2),
                    "turnover_rate_pct": _round_optional(ak_info.get("换手率")),
                    "volume_ratio": None,
                }
                sources["valuation"] = "akshare"
        except Exception as exc:
            errors.append(f"AKShare 基本面降级失败：{exc}")

    if not financials:
        try:
            financials, ak_errors = _akshare_financials(code)
            errors.extend(ak_errors)
            if financials:
                sources["financials"] = "akshare"
        except Exception as exc:
            errors.append(f"AKShare 财务指标降级失败：{exc}")

    return {
        "profile": profile,
        "valuation": valuation,
        "financials": financials,
        "sources": sources,
        "errors": errors,
    }


def _fundamental_score(fundamentals: dict[str, Any]) -> tuple[int | None, list[str]]:
    financials = fundamentals.get("financials") or {}
    valuation = fundamentals.get("valuation") or {}
    evidence: list[str] = []
    score = 50.0
    observed = 0

    roe = _round_optional(financials.get("roe_pct"))
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

    growth = _round_optional(financials.get("net_profit_yoy_pct"))
    if growth is not None:
        observed += 1
        if growth >= 15:
            score += 12
            evidence.append("净利润同比增长较快")
        elif growth < -15:
            score -= 15
            evidence.append("净利润同比明显下降")

    debt = _round_optional(financials.get("debt_to_assets_pct"))
    if debt is not None:
        observed += 1
        if debt > 75:
            score -= 12
            evidence.append("资产负债率偏高，需结合行业解释")
        elif debt < 45:
            score += 5
            evidence.append("资产负债率相对温和")

    pe = _round_optional(valuation.get("pe_ttm") or valuation.get("pe_dynamic"))
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


def _a_share_rules(code: str, name: str) -> dict[str, Any]:
    rate = _limit_rate(code, name)
    return {
        "settlement": "T+1：当日买入的股票最早下一个交易日卖出",
        "price_limit_pct": round(rate * 100, 2),
        "price_limit_note": "ST、创业板/科创板、北交所的涨跌幅限制不同，结果已按代码和名称归类",
        "buy_lot": "普通竞价买入通常以 100 股整数倍申报；零股卖出按交易所规则处理",
        "prediction_horizon": "本系统只研究 T+1、T+2、T+3，不输出更远预测",
    }


def fenxi_gupiao(
    *,
    gupiao: str,
    source: str = "auto",
    history_calendar_days: int = 540,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """Run a deterministic single-stock fundamental and technical analysis."""
    _ensure_dotenv()
    history_calendar_days = max(180, min(int(history_calendar_days), 1800))
    code, resolved, resolve_warnings = jiexi_gupiao(gupiao, source=source)
    end = datetime.now().date()
    start = end - timedelta(days=history_calendar_days)
    market = huoqu_rili_xingqing(
        code,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        source=source,
    )
    if market.data.empty:
        return {
            "status": "error",
            "error": f"无法取得 {code} 的日线行情",
            "data_errors": list(market.errors),
        }

    technical = zongjie_jishu(market.data)
    fundamentals = huoqu_jibenmian(code, trade_date=str(technical["trade_date"]))
    if resolved and not fundamentals.get("profile"):
        fundamentals["profile"] = resolved
        fundamentals.setdefault("sources", {})["profile"] = "local_cache"
    profile = fundamentals.get("profile") or {}
    name = str(profile.get("name") or resolved.get("name") or "")
    fundamental_score, fundamental_evidence = _fundamental_score(fundamentals)
    risks: list[str] = []
    if market.adjustment == "raw_unadjusted":
        risks.append("行情未复权，历史分红送转可能影响长周期技术指标")
    if technical.get("annualized_volatility_20") and float(technical["annualized_volatility_20"]) > 0.55:
        risks.append("近期波动率较高，T+1 到 T+3 预测误差会放大")
    if any(keyword in name.upper() for keyword in ["ST", "退"]):
        risks.append("股票名称包含 ST/退市风险标记")
    if not fundamentals.get("financials"):
        risks.append("财务指标接口未返回数据，基本面结论不完整")

    result: dict[str, Any] = {
        "status": "ok",
        "analysis_type": "single_stock",
        "stock": {"ts_code": code, "name": name, **{key: value for key, value in profile.items() if key not in {"ts_code", "name"}}},
        "as_of": technical["trade_date"],
        "market_data": {
            "source": market.source,
            "adjustment": market.adjustment,
            "rows": int(len(market.data)),
            "start_date": market.data["trade_date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": market.data["trade_date"].iloc[-1].strftime("%Y-%m-%d"),
            "warnings": list(resolve_warnings) + list(market.warnings),
            "errors": list(market.errors),
        },
        "technical_analysis": technical,
        "fundamental_analysis": {
            **fundamentals,
            "score_0_100": fundamental_score,
            "evidence": fundamental_evidence,
        },
        "a_share_rules": _a_share_rules(code, name),
        "risks": risks,
        "scope_note": "这是基于公开数据的量化研究结果，不是收益保证；LLM 只负责解释工具返回的事实。",
        "execution_policy": "research_only：程序不连接券商、不读取交易账户、不提交委托，所有买卖决定由用户人工完成。",
    }

    if run_dir:
        try:
            run_path = safe_run_dir(run_dir)
            artifact_dir = run_path / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            output = artifact_dir / f"gupiao_fenxi_{code.replace('.', '_')}.json"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["artifact"] = str(output)
        except Exception as exc:
            result["artifact_error"] = str(exc)
    return result
