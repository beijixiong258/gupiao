"""Shared A-share market-data access used by the two research workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal

import pandas as pd

PRICE_LIMIT_RULE_EFFECTIVE_FROM = "2026-07-06"
# 当前快照的工程核验口径；不代表数据源承诺或可保证即时成交。
CURRENT_QUOTE_MAX_ACTIVE_AGE_SECONDS = 120


def _quote_session_phase(checked_at: pd.Timestamp | None) -> str:
    if checked_at is None:
        return "unknown"
    minute = checked_at.hour * 60 + checked_at.minute
    for boundary, phase in (
        (9 * 60 + 15, "pre_open"), (9 * 60 + 25, "opening_auction"), (9 * 60 + 30, "opening_auction_result"),
        (11 * 60 + 30, "trading"), (13 * 60, "midday_break"), (14 * 60 + 57, "trading"),
        (15 * 60, "closing_auction"), (15 * 60 + 5, "close_pending"),
    ):
        if minute < boundary:
            return phase
    return "post_close"


def _quote_active_age_seconds(quote_time: pd.Timestamp, checked_at: pd.Timestamp) -> float:
    """仅计算同日可更新行情的时段，午休等停更时间不积累延迟。"""
    day = checked_at.normalize()
    return sum(
        max(0.0, (min(checked_at, day + pd.Timedelta(minutes=end))
                  - max(quote_time, day + pd.Timedelta(minutes=start))).total_seconds())
        for start, end in ((9 * 60 + 15, 9 * 60 + 25), (9 * 60 + 30, 11 * 60 + 30), (13 * 60, 15 * 60))
    )


def heyan_kuaizhao_shidian(
    snapshot: dict[str, Any], *, expected_trade_date: Any, reference_time: Any = None,
    require_timestamp: bool = False,
) -> dict[str, Any]:
    """核对来源日期；当前行情另核验有效交易时段延迟，不推断交易日身份。"""
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
    acquisition_limit = captured if captured is not None else checked_at
    result = {
        "status": "unavailable", "verified": False,
        "source_date_verified": False,
        "timeliness_status": "unavailable" if require_timestamp else "not_requested",
        "session_phase": _quote_session_phase(checked_at),
        "expected_trade_date": expected.strftime("%Y-%m-%d") if expected is not None else None,
        "provider_trade_date": observed.strftime("%Y-%m-%d") if observed is not None else None,
        "provider_quote_time": quote_time.strftime("%Y-%m-%d %H:%M:%S") if quote_time is not None else None,
        "quote_age_seconds": (checked_at - quote_time).total_seconds() if checked_at is not None and quote_time is not None else None,
        "quote_active_age_seconds": None,
        "maximum_active_age_seconds": CURRENT_QUOTE_MAX_ACTIVE_AGE_SECONDS if require_timestamp else None,
        "verification_scope": (
            f"来源日期与更新时间核验；当前行情有效交易时段延迟最多{CURRENT_QUOTE_MAX_ACTIVE_AGE_SECONDS}秒，扣除竞价后等待、午休与收盘后停更时间；不保证即时成交"
            if require_timestamp else "仅核对来源日期和未来时点，不确认当前时效或即时成交"
        ),
    }
    if expected is None or observed is None:
        result["reason"] = "缺少待核验交易日期或行情来源日期，取得时间不能证明行情日期"
    elif provider_date is not None and quote_time is not None and provider_date.normalize() != quote_time.normalize():
        result.update(status="conflict", reason="行情来源日期与更新时间冲突")
    elif observed.normalize() != expected.normalize():
        result.update(status="stale" if observed.normalize() < expected.normalize() else "future", reason="行情来源日期与待核验交易日不一致")
    elif quote_time is not None and acquisition_limit is not None and quote_time > acquisition_limit:
        result.update(status="future", reason="行情更新时间晚于实际取得和核验时点")
    else:
        result["source_date_verified"] = True
        if not require_timestamp:
            result.update(status="verified", verified=True, reason="来源日期与待核验交易日一致；本次未要求当前时效核验")
        elif quote_time is None or checked_at is None:
            result["reason"] = "盘中缺少来源更新时间或核验时点，不能确认当前行情"
        elif checked_at.normalize() != expected.normalize():
            result.update(status="stale", timeliness_status="stale", reason="核验时点已不在待分析交易日，旧快照不能作为当前行情")
        elif quote_time < quote_time.normalize() + pd.Timedelta(hours=9, minutes=15):
            result.update(status="stale", timeliness_status="stale", reason="来源更新时间仍在当日集合竞价之前，不能确认当前行情")
        else:
            active_age = _quote_active_age_seconds(quote_time, checked_at)
            result["quote_active_age_seconds"] = active_age
            if checked_at >= checked_at.normalize() + pd.Timedelta(hours=15) and quote_time < quote_time.normalize() + pd.Timedelta(hours=15):
                result.update(status="stale", timeliness_status="stale", reason="来源更新时间仍在收盘之前，等待收盘行情核验")
            elif active_age > CURRENT_QUOTE_MAX_ACTIVE_AGE_SECONDS:
                result.update(status="stale", timeliness_status="stale", reason=f"行情有效交易时段延迟 {active_age:.0f} 秒，超过 {CURRENT_QUOTE_MAX_ACTIVE_AGE_SECONDS} 秒核验上限")
            else:
                result.update(status="verified", verified=True, timeliness_status="verified", reason="来源日期一致且有效交易时段延迟在核验上限内；不保证即时成交")
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
    # 深市股票代码段；399指数、36网络投票等不能因首位为3混入个股路径。
    # 主板区间见深交所2021年两板合并通知，创业板区间见2024年数据接口规范。
    elif 1 <= int(value) <= 4999 or 300000 <= int(value) <= 309999:
        expected_suffix = "SZ"
    elif value.startswith(("43", "83", "87", "88", "920")):
        expected_suffix = "BJ"
    else:
        raise ValueError(f"代码不属于当前支持的沪深北 A 股范围：{code}")
    if explicit_suffix and explicit_suffix != expected_suffix:
        raise ValueError(f"股票代码与交易所后缀不一致：{code}")
    return f"{value}.{expected_suffix}"


def guolv_a_gu_jilu(
    frame: pd.DataFrame, *, source: str, code_column: str = "ts_code",
) -> pd.DataFrame:
    """证券代码结构损坏使来源失败；明确非A股的记录剔除并保留数量。"""
    if code_column not in frame:
        raise ValueError(f"{source} 缺少证券代码字段")
    codes = frame[code_column].astype(str).str.strip().str.upper()
    prefixed = codes.str.fullmatch(r"(?:SH|SZ|BJ)\d{6}", na=False)
    codes.loc[prefixed] = codes.loc[prefixed].str[2:] + "." + codes.loc[prefixed].str[:2]
    domestic = codes.str.fullmatch(r"\d{1,6}(?:\.(?:SH|SZ|BJ))?", na=False)
    foreign = codes.str.fullmatch(r"(?:\d{1,6}\.HK|[A-Z][A-Z0-9.-]{0,15}\.(?:US|NYSE|NASDAQ))", na=False)
    malformed = ~(domestic | foreign)
    if malformed.any():
        raise ValueError(f"{source} 含 {int(malformed.sum())} 条证券代码格式损坏，不能作为非A股静默忽略")

    def stock_code(value: str) -> str | None:
        try:
            return biaozhunhua_gupiao_daima(value)
        except ValueError:
            return None

    normalized = codes.map(stock_code)
    accepted = normalized.notna()
    result = frame.loc[accepted].copy()
    result[code_column] = normalized.loc[accepted]
    result.attrs["security_scope"] = {
        "scope": "a_share_stocks_only", "identity_source": source,
        "input_rows": len(frame), "a_share_rows": len(result),
        "excluded_non_a_count": int((~accepted).sum()),
        "excluded_non_a_codes": codes.loc[~accepted].tolist(),
        "verification": "来源股票身份契约与沪深北股票代码范围共同核验；指数、基金、债券等不进入个股池",
    }
    return result


def _tushare_pro() -> Any:
    from src.core.config import ensure_dotenv

    ensure_dotenv()
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


def huoqu_gupiao_jichu_ziliao(
    pro: Any,
    quality: dict[str, Any],
    *,
    public_loader: Callable[[], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    """实时读取身份目录；Tushare受限时使用本次公开目录，不读取本地副本。"""
    attempts: list[dict[str, Any]] = []
    try:
        configured_pro = pro if pro is not None else _tushare_pro()
        frame = configured_pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        if frame is None or frame.empty or not {"ts_code", "name"}.issubset(frame.columns):
            raise RuntimeError("Tushare stock_basic 返回空表或缺少身份字段")
        frame = guolv_a_gu_jilu(frame, source="tushare_stock_basic")
        if frame.empty:
            raise ValueError("Tushare 股票目录排除非A股证券后没有有效记录")
        if frame["ts_code"].duplicated().any() or frame["name"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("Tushare 股票身份目录含重复代码或空名称")
        quality["stock_basic"] = {
            "source": "tushare_live", "rows": len(frame), "persistence": "none",
            "security_scope": dict(frame.attrs.get("security_scope", {})),
        }
        return frame
    except Exception as exc:
        attempts.append({"provider": "tushare_stock_basic", "outcome": "failed", "detail": " ".join(str(exc).split())[:180]})

    if public_loader is None:
        from src.ashare.shichang_shuju import huoqu_shishi_kuaizhao

        public_loader = huoqu_shishi_kuaizhao
    public, meta = public_loader()
    if public is None or public.empty or not {"ts_code", "name"}.issubset(public.columns):
        raise RuntimeError(f"股票身份目录主源与公开备用均不可用：{meta.get('error') or '公开目录为空'}")
    frame = guolv_a_gu_jilu(public, source=str(meta.get("source") or "public_stock_directory"))
    if frame.empty:
        raise ValueError("公开股票目录排除非A股证券后没有有效记录")
    if frame["ts_code"].duplicated().any() or frame["name"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("公开股票身份目录含重复代码或空名称")
    frame["symbol"] = frame["ts_code"].str.split(".", regex=False).str[0]
    # 交易所从经过规范化的代码确定；分类、上市日缺失时不推断填充。
    frame["market"] = frame["ts_code"].str.rsplit(".", n=1).str[-1]
    for column in ("area", "industry", "list_date"):
        if column not in frame.columns:
            frame[column] = None
    frame["list_date"] = pd.to_datetime(frame["list_date"].astype("string"), errors="coerce").dt.strftime("%Y%m%d")
    source = str(meta.get("source") or "public_live")
    quality["stock_basic"] = {
        "source": f"{source}_directory", "rows": len(frame), "persistence": "none",
        "fetched_at": meta.get("fetched_at") or meta.get("captured_at"),
        "as_of": meta.get("as_of"),
        "industry_classification": "来源当前行业标签，分类口径与Tushare分开",
        "area_definition": "来源地域板块名称，保留原值",
        "missing_fields": {column: int(frame[column].isna().sum()) for column in ("area", "industry", "list_date") if frame[column].isna().any()},
        "attempted_providers": attempts + list(meta.get("attempted_providers") or []),
        "security_scope": dict(frame.attrs.get("security_scope", {})),
    }
    columns = ["ts_code", "symbol", "name", "area", "industry", "market", "list_date"]
    columns.extend(column for column in (
        "listing_market_code", "listing_market_source", "listing_market_as_of",
    ) if column in frame.columns)
    return frame[columns].reset_index(drop=True)


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
    "guolv_a_gu_jilu",
    "huoqu_gupiao_jichu_ziliao",
    "huoqu_zhangdieting_guize",
]
