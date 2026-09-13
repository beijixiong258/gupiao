"""Shared A-share market-data access used by the two research workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pandas as pd

PRICE_LIMIT_RULE_EFFECTIVE_FROM = "2026-07-06"


def heyan_kuaizhao_shidian(
    snapshot: dict[str, Any], *, expected_trade_date: Any, reference_time: Any = None,
    require_timestamp: bool = False,
) -> dict[str, Any]:
    """核对来源日期和更新时间；取得时间不能代替行情自身的时点。"""
    def timestamp(value: Any) -> pd.Timestamp | None:
        if not isinstance(value, (str, date, datetime)):
            return None
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except (ValueError, TypeError, OverflowError):
            return None
        if pd.isna(parsed):
            return None
        parsed = pd.Timestamp(parsed)
        return parsed.tz_convert("Asia/Shanghai").tz_localize(None) if parsed.tzinfo else parsed

    expected = timestamp(expected_trade_date)
    quote_time = timestamp(snapshot.get("provider_quote_time"))
    provider_date = timestamp(snapshot.get("provider_trade_date"))
    observed = quote_time if quote_time is not None else provider_date
    reference = timestamp(reference_time)
    captured = timestamp(snapshot.get("captured_at"))
    checked_at = max(value for value in (reference, captured) if value is not None) if reference is not None or captured is not None else None
    result = {
        "status": "unavailable", "verified": False,
        "expected_trade_date": expected.strftime("%Y-%m-%d") if expected is not None else None,
        "provider_trade_date": observed.strftime("%Y-%m-%d") if observed is not None else None,
        "provider_quote_time": quote_time.strftime("%Y-%m-%d %H:%M:%S") if quote_time is not None else None,
        "quote_age_seconds": (checked_at - quote_time).total_seconds() if checked_at is not None and quote_time is not None else None,
        "verification_scope": "核对来源日期、更新时间；不保证即时成交，延迟秒数单独披露",
    }
    if expected is None or observed is None:
        result["reason"] = "缺少待核验交易日期或行情来源日期，取得时间不能证明行情日期"
    elif provider_date is not None and quote_time is not None and provider_date.normalize() != quote_time.normalize():
        result.update(status="conflict", reason="行情来源日期与更新时间冲突")
    elif observed.normalize() != expected.normalize():
        result.update(status="stale" if observed.normalize() < expected.normalize() else "future", reason="行情来源日期与待核验交易日不一致")
    elif quote_time is not None and checked_at is not None and quote_time > checked_at:
        result.update(status="future", reason="行情更新时间晚于实际取得和核验时点")
    elif require_timestamp and (quote_time is None or checked_at is None):
        result["reason"] = "盘中缺少来源更新时间或核验时点，不能确认当前行情"
    else:
        result.update(status="verified", verified=True, reason="来源日期与待核验交易日一致，已保留来源更新时间和延迟信息")
    return result


@dataclass(frozen=True)
class PriceLimitRule:
    """Price-limit decision for one stock and one trading session."""

    status: Literal["limited", "no_limit"]
    limit_rate: float | None
    effective_from: str
    reason: str


def biaozhunhua_gupiao_daima(code: str) -> str:
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


def huoqu_gupiao_jichu_ziliao(pro: Any, quality: dict[str, Any]) -> pd.DataFrame:
    """从 Tushare 实时读取当前上市股票资料，不落盘、不读取本地副本。"""
    frame = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date",
    )
    if frame is None or frame.empty:
        raise RuntimeError("Tushare stock_basic returned empty data")
    frame = frame.copy()
    frame["ts_code"] = frame["ts_code"].map(biaozhunhua_gupiao_daima)
    quality["stock_basic"] = {
        "source": "tushare_live",
        "rows": len(frame),
        "persistence": "none",
    }
    return frame


def huoqu_zhangdieting_guize(
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

    normalized_code = biaozhunhua_gupiao_daima(ts_code)
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
    rate = huoqu_zhangdieting_guize(ts_code, name).limit_rate
    if rate is None:  # Defensive: the default rule request is always limited.
        raise RuntimeError("normal price-limit rule unexpectedly has no numeric rate")
    return rate


# 兼容既有内部调用；领域层应使用不带下划线的公开名称。
_normalize_code = biaozhunhua_gupiao_daima
_price_limit_rule = huoqu_zhangdieting_guize


__all__ = [
    "PRICE_LIMIT_RULE_EFFECTIVE_FROM",
    "PriceLimitRule",
    "biaozhunhua_gupiao_daima",
    "huoqu_gupiao_jichu_ziliao",
    "huoqu_zhangdieting_guize",
]
