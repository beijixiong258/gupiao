"""从调用方指定的 RQAlpha 日线数据包按需读取历史交易证据。"""

from __future__ import annotations

import pickle
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ashare.lishi_jiaoyi import huizong_lishi_jiaoyi, jiaoyi_riqi, zhengli_jiaoyi_zhengju, zhankai_shengxiao_yinzi
from ashare.lishi_jiaoyi_yuan import huoqu_jiaoyi_gonggao
from ashare.shuju_yuan import biaozhunhua_gupiao_daima


_EXCHANGE = {"SH": "XSHG", "SZ": "XSHE"}
_BOARD = {"MainBoard": "main_board", "GEM": "chinext", "KSH": "star"}
_DAILY_COLUMNS = {
    "open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close",
    "prev_close": "limit_reference_price", "total_turnover": "amount_yuan",
    "limit_up": "limit_up_price", "limit_down": "limit_down_price",
}


class _PlainBundleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise ValueError(f"RQAlpha 证券目录包含非普通数据对象 {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise ValueError("RQAlpha 证券目录包含外部对象引用")


def _bundle_dates(values: Any, *, with_time: bool = False, unordered: bool = False) -> pd.DatetimeIndex:
    encoded = np.asarray(values, dtype=np.int64)
    if with_time:
        if (encoded % 1_000_000 != 0).any():
            raise ValueError("RQAlpha 日线日期包含非零时间")
        encoded = encoded // 1_000_000
    dates = pd.to_datetime(encoded.astype(str), format="%Y%m%d", errors="coerce")
    if dates.isna().any() or (not unordered and (not dates.is_unique or not dates.is_monotonic_increasing)):
        raise ValueError("RQAlpha 日期无效、重复或顺序异常")
    return dates.unique().sort_values() if unordered else dates


def _actions(tables: Mapping[str, Any], symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    result = []
    for row in tables["dividends"].get(symbol, ()):
        ex_date = jiaoyi_riqi(int(row["ex_dividend_date"]))
        announcement = jiaoyi_riqi(int(row["announcement_date"]))
        shares = float(row["round_lot"])
        cash = float(row["dividend_cash_before_tax"])
        if pd.notna(announcement) and announcement <= end and start <= ex_date <= end and shares > 0 and np.isfinite(cash):
            result.append({"kind": "cash_dividend", "ex_date": ex_date, "announcement_date": announcement,
                           "record_date": jiaoyi_riqi(int(row["book_closure_date"])), "cash_per_10_shares": cash * 10 / shares,
                           "payable_date": jiaoyi_riqi(int(row["payable_date"])), "source": "rqalpha_dividends"})
    for row in tables["split_factor"].get(symbol, ()):
        ex_date = jiaoyi_riqi(int(row["ex_date"]) // 1_000_000)
        values = [float(row[field]) for field in ("split_factor", "split_coefficient_from", "split_coefficient_to")]
        if start <= ex_date <= end and all(np.isfinite(value) and value > 0 for value in values):
            result.append({"kind": "share_count_change", "ex_date": ex_date, "split_factor": values[0],
                           "shares_before": values[1], "shares_after": values[2], "source": "rqalpha_split_factor"})
    return sorted(result, key=lambda item: item["ex_date"])


def _read_stock(
    tables: Mapping[str, Any], symbol: str, dates: pd.DatetimeIndex, bundle_dates: pd.DatetimeIndex,
    profile: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cutoff: pd.Timestamp,
    notices: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    raw = pd.DataFrame.from_records(tables["stocks"][symbol][:]).rename(columns=_DAILY_COLUMNS)
    raw.index = _bundle_dates(raw.pop("datetime"), with_time=True)
    raw = raw.loc[(raw.index <= end) & raw.index.isin(bundle_dates)].rename_axis("trade_date")
    prior = raw.loc[(raw.index < start) & raw["volume"].gt(0)].tail(1)
    if not prior.empty:
        raw = raw.loc[raw.index >= prior.index[0]]
    else:
        raw = raw.loc[raw.index >= start]
    suspended = _bundle_dates(tables["suspended_days"].get(symbol, ()))
    st_dates = _bundle_dates(tables["st_stock_days"].get(symbol, ()), unordered=True)
    index = raw.index.union(suspended[(suspended >= (prior.index[0] if not prior.empty else start)) & (suspended <= min(end, cutoff))])
    raw = raw.assign(raw_price_source="rqalpha_stocks", limit_reference_source="rqalpha_stocks.prev_close",
                     limit_source="rqalpha_stocks").reindex(index).rename_axis("trade_date")
    raw["is_suspended"] = raw.index.isin(suspended)
    raw["suspension_source"] = "rqalpha_suspended_days"
    if symbol in tables["st_stock_days"]:
        raw["is_st"] = raw.index.isin(st_dates)
        raw["st_source"] = "rqalpha_st_stock_days"
    events = tables["ex_cum_factor"].get(symbol)
    if events is not None:
        factors = pd.DataFrame.from_records(events[:]).rename(columns={"ex_cum_factor": "adj_factor"})
        starts = factors.pop("start_date")
        factors["trade_date"] = _bundle_dates(starts.mask(starts.eq(0), 19000101000000), with_time=True)
        raw["adj_factor"] = zhankai_shengxiao_yinzi(factors, raw.index)
        raw["factor_source"] = raw["adj_factor"].notna().map({True: "rqalpha_ex_cum_factor", False: None})
    elif symbol not in tables["dividends"] and symbol not in tables["split_factor"]:
        # RQAlpha 对包内股票没有复权事件记录时直接返回原价。
        raw["adj_factor"] = 1.0
        raw["factor_source"] = "rqalpha_no_adjustment_events"
    return zhengli_jiaoyi_zhengju(raw.reset_index(), dates, profile, start=start, end=end, notices=notices)


def duqu_rqalpha_lishi(
    bundle_dir: str | Path,
    codes: Iterable[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    trading_calendar: Iterable[pd.Timestamp],
    notice_urls: Mapping[str, Iterable[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """使用外部数据快照和显式市场日历，保留缺行、原值与来源。"""
    import h5py
    import hdf5plugin  # noqa: F401  注册 RQAlpha 使用的 HDF5 压缩过滤器

    path = Path(bundle_dir).expanduser().resolve(strict=True)
    codes = tuple(dict.fromkeys(biaozhunhua_gupiao_daima(code) for code in codes))
    dates = pd.DatetimeIndex(sorted(set(trading_calendar)))
    if pd.isna(start) or pd.isna(end) or start > end or dates.empty or dates.hasnans or not dates.equals(dates.normalize()):
        raise ValueError("历史数据包需要有效日期区间和显式市场交易日历")
    bundle_dates = _bundle_dates(np.load(path / "trading_dates.npy", allow_pickle=False))
    with (path / "instruments.pk").open("rb") as source:
        instruments = _PlainBundleUnpickler(source).load()
    if not isinstance(instruments, list):
        raise ValueError("RQAlpha 证券目录结构无效")
    common = {row["order_book_id"]: row for row in instruments if row["type"] == "CS" and row["exchange"] in _EXCHANGE.values()}
    frames, sources, skipped = [], {}, {}
    with ExitStack() as stack:
        tables = {name: stack.enter_context(h5py.File(path / f"{name}.h5", "r")) for name in (
            "stocks", "suspended_days", "st_stock_days", "ex_cum_factor", "dividends", "split_factor",
        )}
        stock_keys = {key for key in set(tables["stocks"]) & common.keys() if len(tables["stocks"][key])}
        if not stock_keys:
            raise ValueError("RQAlpha 数据包没有经目录核验的沪深个股")
        cutoff = jiaoyi_riqi(max(int(tables["stocks"][key][-1]["datetime"]) for key in stock_keys) // 1_000_000)
        if cutoff not in bundle_dates:
            raise ValueError("RQAlpha 日线截止日未在其交易日历中核验")
        first_checked = max(start, bundle_dates[0])
        last_checked = min(end, cutoff)
        external_days = dates[(dates >= first_checked) & (dates <= last_checked)]
        source_days = bundle_dates[(bundle_dates >= first_checked) & (bundle_dates <= last_checked)]
        if not external_days.equals(source_days):
            remote_only = len(external_days.difference(source_days))
            bundle_only = len(source_days.difference(external_days))
            raise ValueError(f"市场日历与 RQAlpha 数据包历史开市日有冲突（仅市场日历 {remote_only} 日，仅包内日历 {bundle_only} 日）")
        for code in codes:
            digits, exchange = code.split(".")
            symbol = f"{digits}.{_EXCHANGE.get(exchange, '')}"
            if symbol not in stock_keys:
                skipped[code] = "RQAlpha 日线包没有核验到该沪深 A 股；该公开包不覆盖北交所" if exchange == "BJ" else "RQAlpha 目录或日线缺少该 A 股"
                continue
            item = common[symbol]
            listed, delisted = (jiaoyi_riqi(item[key]) for key in ("listed_date", "de_listed_date"))
            if pd.isna(listed) or (pd.notna(delisted) and listed >= delisted):
                skipped[code] = "RQAlpha 上市及摘牌日期无效"
                continue
            profile = {"ts_code": code, "snapshot_name": item["symbol"], "board": _BOARD.get(item["board_type"]),
                       "list_date": listed, "delist_date": delisted, "identity_source": "rqalpha_instruments",
                       "listing_status": "delisted" if pd.notna(delisted) else "listed"}
            notices, attempts = huoqu_jiaoyi_gonggao((notice_urls or {}).get(code, ()), code, end=end, delist_date=delisted)
            actions = _actions(tables, symbol, start, min(end, cutoff))
            frame, conflicts = _read_stock(tables, symbol, dates, bundle_dates, profile, start, end, cutoff, notices)
            frames.append(frame)
            evidence = {"profile": profile, "corporate_actions": actions, "notice_events": notices,
                        "conflicts": conflicts, "attempts": attempts}
            sources[code] = {**huizong_lishi_jiaoyi(frame, evidence, end=end), "attempts": attempts}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), {
        "source": "rqalpha_bundle", "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bundle_cutoff": cutoff, "bundle_stock_count": len(stock_keys), "requested_range": [start, end],
        "bundle_calendar_start": bundle_dates[0], "by_code": sources, "unavailable_codes": skipped,
        "persistence": "none", "intraday_coverage": "exchange_notices_only",
    }
