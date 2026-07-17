"""Shared A-share market-data access used by the two research workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

AGENT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = AGENT_DIR / "cache"
STOCK_BASIC_CACHE = CACHE_DIR / "tushare_stock_basic.csv"
STOCK_BASIC_CACHE_TTL = timedelta(hours=24)
PRICE_LIMIT_RULE_EFFECTIVE_FROM = "2026-07-06"


@dataclass(frozen=True)
class PriceLimitRule:
    """Price-limit decision for one stock and one trading session."""

    status: Literal["limited", "no_limit"]
    limit_rate: float | None
    effective_from: str
    reason: str


def _normalize_code(code: str) -> str:
    value = str(code).strip().upper()
    explicit_suffix = ""
    if value.endswith((".SH", ".SZ", ".BJ")):
        value, explicit_suffix = value.rsplit(".", 1)
    if not value.isdigit() or len(value) > 6:
        raise ValueError(f"无效的 A 股代码：{code}")
    value = value.zfill(6)
    if value.startswith("6"):
        expected_suffix = "SH"
    elif value.startswith(("0", "3")):
        expected_suffix = "SZ"
    elif value.startswith(("43", "83", "87", "88", "920")):
        expected_suffix = "BJ"
    else:
        raise ValueError(f"代码不属于当前支持的沪深北 A 股范围：{code}")
    if explicit_suffix and explicit_suffix != expected_suffix:
        raise ValueError(f"股票代码与交易所后缀不一致：{code}")
    return f"{value}.{expected_suffix}"


def _tushare_pro() -> Any:
    from src.providers.llm import _ensure_dotenv

    _ensure_dotenv()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set")

    import tushare as ts

    ts.set_token(token)
    return ts.pro_api(token)


def _latest_tushare_daily(pro: Any, trade_date: str | None) -> tuple[str, pd.DataFrame]:
    """Return the requested or latest available A-share daily cross-section."""
    if trade_date:
        frame = pro.daily(trade_date=trade_date)
        if frame is not None and not frame.empty:
            return trade_date, frame
        raise RuntimeError(f"Tushare daily returned empty data for {trade_date}")

    for offset in range(20):
        day = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
        frame = pro.daily(trade_date=day)
        if frame is not None and not frame.empty:
            return day, frame
    raise RuntimeError("Tushare daily returned no recent trading day data")


def _load_or_fetch_stock_basic(
    pro: Any,
    quality: dict[str, Any],
    *,
    cache_ttl: timedelta | None = STOCK_BASIC_CACHE_TTL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load the listed-stock table, refreshing an expired cache when possible.

    Existing callers keep the previous behavior and may omit the keyword-only
    options. ``cache_ttl=None`` keeps a valid cache indefinitely, while
    ``force_refresh=True`` always attempts a fresh Tushare request.
    """
    stale_cached: pd.DataFrame | None = None
    if STOCK_BASIC_CACHE.is_file():
        try:
            cached = pd.read_csv(STOCK_BASIC_CACHE, dtype=str)
            if not cached.empty and {"ts_code", "name"}.issubset(cached.columns):
                stale_cached = cached
                age_seconds = max(0.0, datetime.now().timestamp() - STOCK_BASIC_CACHE.stat().st_mtime)
                cache_is_fresh = cache_ttl is None or age_seconds <= max(0.0, cache_ttl.total_seconds())
                if cache_is_fresh and not force_refresh:
                    quality["stock_basic"] = {
                        "source": "cache",
                        "rows": len(cached),
                        "age_hours": round(age_seconds / 3600.0, 2),
                    }
                    return cached
                quality.setdefault("warnings", []).append(
                    "stock_basic cache refresh requested"
                    if force_refresh
                    else "stock_basic cache expired; refreshing from Tushare"
                )
        except Exception as exc:
            quality.setdefault("warnings", []).append(f"stock_basic cache read failed: {exc}")

    try:
        frame = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
    except Exception as exc:
        if stale_cached is None:
            raise
        quality.setdefault("warnings", []).append(
            f"stock_basic refresh failed; using stale cache: {exc}"
        )
        quality["stock_basic"] = {"source": "stale_cache", "rows": len(stale_cached)}
        return stale_cached
    if frame is None or frame.empty:
        if stale_cached is not None:
            quality.setdefault("warnings", []).append(
                "stock_basic refresh returned empty data; using stale cache"
            )
            quality["stock_basic"] = {"source": "stale_cache", "rows": len(stale_cached)}
            return stale_cached
        raise RuntimeError("Tushare stock_basic returned empty data")
    frame = frame.copy()
    frame["ts_code"] = frame["ts_code"].map(_normalize_code)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STOCK_BASIC_CACHE, index=False, encoding="utf-8-sig")
    quality["stock_basic"] = {"source": "tushare", "rows": len(frame)}
    return frame


def _price_limit_rule(
    ts_code: str,
    name: str,
    *,
    price_limit_exempt: bool = False,
) -> PriceLimitRule:
    """Return the current price-limit status and rate for an A-share session.

    Rules reflect the regime effective on 2026-07-06: Shanghai/Shenzhen main
    board shares, including risk-warning shares, use 10%; ChiNext and STAR use
    20%; Beijing Stock Exchange shares use 30%.  IPO/relisting and other exempt
    sessions require an exchange-calendar or trading-status decision upstream;
    callers can represent such a session with ``price_limit_exempt=True``.
    """
    del name  # The 2026 rules no longer require an ST-name override.
    if price_limit_exempt:
        return PriceLimitRule(
            status="no_limit",
            limit_rate=None,
            effective_from=PRICE_LIMIT_RULE_EFFECTIVE_FROM,
            reason="上游交易日历或交易状态标记该交易日无涨跌幅限制",
        )

    normalized_code = _normalize_code(ts_code)
    if normalized_code.endswith(".BJ"):
        rate = 0.30
        reason = "北交所竞价交易股票涨跌幅限制为30%"
    elif normalized_code.startswith(("300", "301", "688", "689")):
        rate = 0.20
        reason = "创业板或科创板股票涨跌幅限制为20%"
    else:
        rate = 0.10
        reason = "沪深主板股票（含风险警示股票）涨跌幅限制为10%"
    return PriceLimitRule(
        status="limited",
        limit_rate=rate,
        effective_from=PRICE_LIMIT_RULE_EFFECTIVE_FROM,
        reason=reason,
    )


def _limit_rate(ts_code: str, name: str) -> float:
    """Return the normal board rate while preserving the historical float API."""
    rate = _price_limit_rule(ts_code, name).limit_rate
    if rate is None:  # Defensive: the default rule request is always limited.
        raise RuntimeError("normal price-limit rule unexpectedly has no numeric rate")
    return rate


__all__ = [
    "PRICE_LIMIT_RULE_EFFECTIVE_FROM",
    "PriceLimitRule",
    "STOCK_BASIC_CACHE",
    "STOCK_BASIC_CACHE_TTL",
]
