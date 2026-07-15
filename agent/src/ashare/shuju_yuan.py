"""Shared A-share market-data access used by the two research workflows."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

AGENT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = AGENT_DIR / "cache"
STOCK_BASIC_CACHE = CACHE_DIR / "tushare_stock_basic.csv"


def _normalize_code(code: str) -> str:
    value = str(code).strip().upper()
    if value.endswith((".SH", ".SZ", ".BJ")):
        return value
    value = value.zfill(6)
    if value.startswith("6"):
        return f"{value}.SH"
    if value.startswith(("0", "3")):
        return f"{value}.SZ"
    return f"{value}.BJ"


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


def _load_or_fetch_stock_basic(pro: Any, quality: dict[str, Any]) -> pd.DataFrame:
    """Load the listed-stock table from cache, otherwise fetch and cache it."""
    if STOCK_BASIC_CACHE.is_file():
        try:
            cached = pd.read_csv(STOCK_BASIC_CACHE, dtype=str)
            if not cached.empty and {"ts_code", "name"}.issubset(cached.columns):
                quality["stock_basic"] = {"source": "cache", "rows": len(cached)}
                return cached
        except Exception as exc:
            quality.setdefault("warnings", []).append(f"stock_basic cache read failed: {exc}")

    frame = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if frame is None or frame.empty:
        raise RuntimeError("Tushare stock_basic returned empty data")
    frame = frame.copy()
    frame["ts_code"] = frame["ts_code"].map(_normalize_code)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STOCK_BASIC_CACHE, index=False, encoding="utf-8-sig")
    quality["stock_basic"] = {"source": "tushare", "rows": len(frame)}
    return frame


def _limit_rate(ts_code: str, name: str) -> float:
    """Return the normal daily price-limit rate for the current A-share board."""
    upper_name = str(name).upper()
    if "ST" in upper_name:
        return 0.05
    if str(ts_code).endswith(".BJ"):
        return 0.30
    if str(ts_code).startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


__all__ = ["STOCK_BASIC_CACHE"]
