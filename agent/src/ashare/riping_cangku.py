"""Persistent full-market daily warehouse for reproducible A-share research."""

from __future__ import annotations

import math
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ashare.shuju_yuan import CACHE_DIR, _normalize_code, _tushare_pro


DAILY_WAREHOUSE_PATH = CACHE_DIR / "a_share_daily_warehouse.sqlite3"
WAREHOUSE_SCHEMA_VERSION = 2
FULL_MARKET_MIN_COMPLETE_SESSIONS = 500


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"无效日期：{value}")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _display_date(value: str | None) -> str | None:
    return datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d") if value else None


def _connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    if not create and not path.is_file():
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_daily_warehouse(path: Path | str = DAILY_WAREHOUSE_PATH) -> str:
    resolved = Path(path).expanduser().resolve()
    with _connect(resolved) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS warehouse_meta (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_calendar (
                cal_date TEXT PRIMARY KEY,
                is_open INTEGER NOT NULL,
                pretrade_date TEXT,
                exchange TEXT,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stock_snapshots (
                snapshot_date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                symbol TEXT,
                name TEXT,
                area TEXT,
                industry TEXT,
                market TEXT,
                exchange TEXT,
                list_status TEXT,
                list_date TEXT,
                delist_date TEXT,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (snapshot_date, ts_code)
            );

            CREATE TABLE IF NOT EXISTS daily_bars (
                trade_date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change_value REAL,
                pct_chg REAL,
                volume REAL,
                amount_yuan REAL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (trade_date, ts_code)
            );

            CREATE TABLE IF NOT EXISTS daily_basic (
                trade_date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                close REAL,
                turnover_rate REAL,
                turnover_rate_f REAL,
                volume_ratio REAL,
                pe REAL,
                pe_ttm REAL,
                pb REAL,
                ps REAL,
                ps_ttm REAL,
                dv_ratio REAL,
                dv_ttm REAL,
                total_share REAL,
                float_share REAL,
                free_share REAL,
                total_mv REAL,
                circ_mv REAL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (trade_date, ts_code)
            );

            CREATE TABLE IF NOT EXISTS adj_factors (
                trade_date TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                adj_factor REAL NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (trade_date, ts_code)
            );

            CREATE TABLE IF NOT EXISTS sync_status (
                trade_date TEXT PRIMARY KEY,
                bars_status TEXT NOT NULL DEFAULT 'pending',
                basic_status TEXT NOT NULL DEFAULT 'pending',
                adj_status TEXT NOT NULL DEFAULT 'pending',
                bars_rows INTEGER NOT NULL DEFAULT 0,
                basic_rows INTEGER NOT NULL DEFAULT 0,
                adj_rows INTEGER NOT NULL DEFAULT 0,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS board_snapshots (
                snapshot_date TEXT NOT NULL,
                board_type TEXT NOT NULL,
                board_name TEXT NOT NULL,
                source TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (snapshot_date, board_type, board_name, source)
            );

            CREATE TABLE IF NOT EXISTS board_members (
                snapshot_date TEXT NOT NULL,
                board_type TEXT NOT NULL,
                board_name TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                name TEXT,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (snapshot_date, board_type, board_name, ts_code, source)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_bars_code_date
                ON daily_bars (ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_daily_basic_code_date
                ON daily_basic (ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_adj_factors_code_date
                ON adj_factors (ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_stock_snapshots_code_date
                ON stock_snapshots (ts_code, snapshot_date);
            CREATE INDEX IF NOT EXISTS idx_board_members_board_date
                ON board_members (board_type, board_name, snapshot_date);
            """
        )
        sync_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sync_status)").fetchall()
        }
        if "attempt_count" not in sync_columns:
            connection.execute(
                "ALTER TABLE sync_status ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
            )
        if "last_attempt_at" not in sync_columns:
            connection.execute("ALTER TABLE sync_status ADD COLUMN last_attempt_at TEXT")
        connection.execute(
            "INSERT INTO warehouse_meta(meta_key, meta_value) VALUES('schema_version', ?) "
            "ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value",
            (str(WAREHOUSE_SCHEMA_VERSION),),
        )
    return str(resolved)


def _normalise_codes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    data["ts_code"] = data["ts_code"].astype(str).map(_normalize_code)
    return data


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) and np.isfinite(float(parsed)) else None


def _upsert_calendar(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    source: str = "tushare_trade_cal",
) -> None:
    if frame is None or frame.empty:
        raise RuntimeError("Tushare trade_cal 返回空数据")
    fetched_at = _now_text()
    rows = []
    for row in frame.to_dict("records"):
        pretrade_date = row.get("pretrade_date")
        rows.append(
            (
                _date_text(row.get("cal_date")),
                int(_number(row.get("is_open")) or 0),
                _date_text(pretrade_date)
                if pretrade_date is not None and pd.notna(pretrade_date) and str(pretrade_date).strip()
                else None,
                str(row.get("exchange") or "SSE"),
                source,
                fetched_at,
            )
        )
    connection.executemany(
        """
        INSERT INTO trade_calendar(cal_date,is_open,pretrade_date,exchange,source,fetched_at)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(cal_date) DO UPDATE SET
            is_open=excluded.is_open,
            pretrade_date=excluded.pretrade_date,
            exchange=excluded.exchange,
            source=excluded.source,
            fetched_at=excluded.fetched_at
        """,
        rows,
    )


def _akshare_trade_calendar(start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    frame = ak.tool_trade_date_hist_sina()
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        raise RuntimeError("AKShare 交易日历返回空数据")
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if dates.empty:
        raise RuntimeError(f"AKShare 交易日历不覆盖 {start} 至 {end}")
    result = pd.DataFrame({"cal_date": dates.dt.strftime("%Y%m%d")})
    result["is_open"] = 1
    result["pretrade_date"] = result["cal_date"].shift(1)
    result["exchange"] = "SSE"
    return result


def _ensure_trade_calendar(
    connection: sqlite3.Connection,
    provider: Any,
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    meta_rows = connection.execute(
        "SELECT meta_key,meta_value FROM warehouse_meta "
        "WHERE meta_key IN ('calendar_coverage_start','calendar_coverage_end')"
    ).fetchall()
    coverage = {str(row["meta_key"]): str(row["meta_value"]) for row in meta_rows}
    cached_start = coverage.get("calendar_coverage_start")
    cached_end = coverage.get("calendar_coverage_end")
    if cached_start and cached_end and cached_start <= start and cached_end >= end:
        return {"source": "warehouse_cache", "warnings": []}

    warnings: list[str] = []
    source = "tushare_trade_cal"
    try:
        frame = provider.trade_cal(
            exchange="SSE",
            start_date=start,
            end_date=end,
            fields="exchange,cal_date,is_open,pretrade_date",
        )
    except Exception as exc:
        warnings.append(f"Tushare trade_cal 不可用，改用 AKShare 交易日历：{exc}")
        frame = _akshare_trade_calendar(start, end)
        source = "akshare_sina_trade_calendar"

    with connection:
        _upsert_calendar(connection, frame, source=source)
        new_start = min(value for value in [cached_start, start] if value)
        new_end = max(value for value in [cached_end, end] if value)
        connection.executemany(
            "INSERT INTO warehouse_meta(meta_key,meta_value) VALUES(?,?) "
            "ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value",
            [
                ("calendar_coverage_start", new_start),
                ("calendar_coverage_end", new_end),
                ("calendar_source", source),
            ],
        )
    return {"source": source, "warnings": warnings}


def _snapshot_stock_master(
    connection: sqlite3.Connection,
    pro: Any,
    *,
    snapshot_date: str,
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    fields = "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
    for status in ["L", "D", "P"]:
        try:
            frame = pro.stock_basic(exchange="", list_status=status, fields=fields)
            if frame is not None and not frame.empty:
                frames.append(frame)
        except Exception as exc:
            warnings.append(f"stock_basic list_status={status} 失败：{exc}")
    if not frames:
        return {"status": "unavailable", "rows": 0, "warnings": warnings}
    data = _normalise_codes(pd.concat(frames, ignore_index=True))
    data = data.drop_duplicates("ts_code", keep="first")
    fetched_at = _now_text()
    rows = []
    for row in data.to_dict("records"):
        rows.append(
            (
                snapshot_date,
                str(row.get("ts_code")),
                str(row.get("symbol") or "") or None,
                str(row.get("name") or "") or None,
                str(row.get("area") or "") or None,
                str(row.get("industry") or "") or None,
                str(row.get("market") or "") or None,
                str(row.get("exchange") or "") or None,
                str(row.get("list_status") or "") or None,
                _date_text(row.get("list_date")) if row.get("list_date") else None,
                _date_text(row.get("delist_date")) if row.get("delist_date") else None,
                "tushare_stock_basic_snapshot",
                fetched_at,
            )
        )
    connection.executemany(
        """
        INSERT INTO stock_snapshots(
            snapshot_date,ts_code,symbol,name,area,industry,market,exchange,
            list_status,list_date,delist_date,source,fetched_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(snapshot_date,ts_code) DO UPDATE SET
            symbol=excluded.symbol,name=excluded.name,area=excluded.area,
            industry=excluded.industry,market=excluded.market,exchange=excluded.exchange,
            list_status=excluded.list_status,list_date=excluded.list_date,
            delist_date=excluded.delist_date,source=excluded.source,fetched_at=excluded.fetched_at
        """,
        rows,
    )
    return {
        "status": "ok",
        "snapshot_date": snapshot_date,
        "rows": int(len(rows)),
        "listed": int((data.get("list_status") == "L").sum()) if "list_status" in data else None,
        "delisted": int((data.get("list_status") == "D").sum()) if "list_status" in data else None,
        "warnings": warnings,
    }


def snapshot_board_constituents(
    frame: pd.DataFrame,
    *,
    board_name: str,
    board_type: str,
    source: str,
    snapshot_date: str | None = None,
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> dict[str, Any]:
    """Persist the exact membership returned by a board provider on a fetch date."""
    resolved = Path(path).expanduser().resolve()
    initialize_daily_warehouse(resolved)
    captured_date = _date_text(snapshot_date or date.today().isoformat())
    normalized = _normalise_codes(frame)
    if normalized.empty or "ts_code" not in normalized.columns:
        return {"status": "empty", "snapshot_date": _display_date(captured_date), "rows": 0}
    normalized = normalized.drop_duplicates("ts_code", keep="first")
    board_key = str(board_name).strip()
    type_key = str(board_type).strip().lower()
    source_key = str(source).strip()
    fetched_at = _now_text()
    with _connect(resolved) as connection:
        previous_row = connection.execute(
            """
            SELECT MAX(snapshot_date) AS value FROM board_snapshots
            WHERE board_type=? AND board_name=? AND source=? AND snapshot_date<?
            """,
            (type_key, board_key, source_key, captured_date),
        ).fetchone()
        previous_date = previous_row["value"] if previous_row else None
        previous_codes: set[str] = set()
        if previous_date:
            previous_codes = {
                str(row["ts_code"])
                for row in connection.execute(
                    """
                    SELECT ts_code FROM board_members
                    WHERE snapshot_date=? AND board_type=? AND board_name=? AND source=?
                    """,
                    (previous_date, type_key, board_key, source_key),
                ).fetchall()
            }
        connection.execute(
            "DELETE FROM board_members WHERE snapshot_date=? AND board_type=? AND board_name=? AND source=?",
            (captured_date, type_key, board_key, source_key),
        )
        rows = [
            (
                captured_date,
                type_key,
                board_key,
                str(row.get("ts_code")),
                str(row.get("name") or "") or None,
                source_key,
                fetched_at,
            )
            for row in normalized.to_dict("records")
        ]
        connection.executemany(
            """
            INSERT INTO board_members(
                snapshot_date,board_type,board_name,ts_code,name,source,fetched_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            rows,
        )
        connection.execute(
            """
            INSERT INTO board_snapshots(
                snapshot_date,board_type,board_name,source,member_count,fetched_at
            ) VALUES(?,?,?,?,?,?)
            ON CONFLICT(snapshot_date,board_type,board_name,source) DO UPDATE SET
                member_count=excluded.member_count,fetched_at=excluded.fetched_at
            """,
            (captured_date, type_key, board_key, source_key, len(rows), fetched_at),
        )
    current_codes = set(normalized["ts_code"].astype(str))
    return {
        "status": "ok",
        "snapshot_date": _display_date(captured_date),
        "rows": int(len(rows)),
        "previous_snapshot_date": _display_date(previous_date),
        "added_since_previous": int(len(current_codes - previous_codes)) if previous_date else None,
        "removed_since_previous": int(len(previous_codes - current_codes)) if previous_date else None,
        "warehouse_path": str(resolved),
    }


def board_constituent_history_status(
    *,
    board_name: str,
    board_type: str,
    source: str,
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {"status": "not_initialized", "historical_membership_ready": False}
    with _connect(resolved, create=False) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS snapshots,MIN(snapshot_date) AS first_date,
                   MAX(snapshot_date) AS last_date,MAX(member_count) AS maximum_members
            FROM board_snapshots
            WHERE board_type=? AND board_name=? AND source=?
            """,
            (str(board_type).strip().lower(), str(board_name).strip(), str(source).strip()),
        ).fetchone()
    snapshots = int(row["snapshots"] or 0)
    first_date = row["first_date"]
    last_date = row["last_date"]
    coverage_days = (
        int((pd.Timestamp(last_date) - pd.Timestamp(first_date)).days)
        if first_date and last_date
        else 0
    )
    change_tracking_ready = snapshots >= 2
    historical_training_ready = snapshots >= 12 and coverage_days >= 180
    return {
        "status": (
            "historical_training_ready"
            if historical_training_ready
            else "change_tracking_available"
            if change_tracking_ready
            else "collecting"
        ),
        "change_tracking_ready": change_tracking_ready,
        "historical_membership_ready": historical_training_ready,
        "snapshot_count": snapshots,
        "snapshot_date_range": [_display_date(first_date), _display_date(last_date)],
        "coverage_calendar_days": coverage_days,
        "historical_training_minimum": {"snapshots": 12, "coverage_calendar_days": 180},
        "maximum_members": int(row["maximum_members"] or 0),
        "usage": (
            "快照跨度已达到历史训练门槛，可按真实快照日期构造成分区间；仓库建立前仍无法倒推"
            if historical_training_ready
            else "已有多个真实抓取日快照，可识别成员变化，但跨度尚不足以替代当前成分历史训练"
            if change_tracking_ready
            else "正在积累真实抓取日成分快照；当前历史训练仍需明确标记当前成分偏差"
        ),
    }


def load_board_membership_history(
    *,
    board_name: str,
    board_type: str,
    source: str,
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    status = board_constituent_history_status(
        board_name=board_name,
        board_type=board_type,
        source=source,
        path=resolved,
    )
    if not resolved.is_file():
        return pd.DataFrame(), status
    with _connect(resolved, create=False) as connection:
        rows = connection.execute(
            """
            SELECT snapshot_date,ts_code,name
            FROM board_members
            WHERE board_type=? AND board_name=? AND source=?
            ORDER BY snapshot_date,ts_code
            """,
            (str(board_type).strip().lower(), str(board_name).strip(), str(source).strip()),
        ).fetchall()
    data = pd.DataFrame([dict(row) for row in rows])
    if not data.empty:
        data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce").dt.normalize()
    return data, {**status, "rows": int(len(data))}


def load_stock_snapshot_asof(
    as_of: str,
    *,
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the latest stock-master snapshot that was actually captured by an as-of date."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return pd.DataFrame(), {"status": "warehouse_not_initialized"}
    target = _date_text(as_of)
    with _connect(resolved, create=False) as connection:
        row = connection.execute(
            "SELECT MAX(snapshot_date) AS value FROM stock_snapshots WHERE snapshot_date<=?",
            (target,),
        ).fetchone()
        snapshot_date = row["value"] if row else None
        if not snapshot_date:
            return pd.DataFrame(), {"status": "no_snapshot_on_or_before_asof"}
        rows = connection.execute(
            """
            SELECT ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date
            FROM stock_snapshots
            WHERE snapshot_date=?
            ORDER BY ts_code
            """,
            (snapshot_date,),
        ).fetchall()
    data = pd.DataFrame([dict(value) for value in rows])
    target_date = pd.Timestamp(target)
    if not data.empty:
        listed = pd.to_datetime(data["list_date"], errors="coerce")
        delisted = pd.to_datetime(data["delist_date"], errors="coerce")
        data = data[
            (listed.isna() | (listed <= target_date))
            & (delisted.isna() | (delisted > target_date))
        ].reset_index(drop=True)
    age_days = int((pd.Timestamp(target) - pd.Timestamp(snapshot_date)).days)
    return data, {
        "status": "ok",
        "source": "warehouse_stock_snapshot",
        "snapshot_date": _display_date(snapshot_date),
        "requested_as_of": _display_date(target),
        "snapshot_age_calendar_days": age_days,
        "rows": int(len(data)),
        "historical_precision": "使用不晚于分析日的真实抓取快照，不把更新快照倒填到更早日期",
    }


def _upsert_daily_bars(connection: sqlite3.Connection, frame: pd.DataFrame, trade_date: str) -> int:
    data = _normalise_codes(frame)
    if data.empty:
        raise RuntimeError(f"{trade_date} daily 返回空数据")
    fetched_at = _now_text()
    rows = []
    for row in data.to_dict("records"):
        row_date = _date_text(row.get("trade_date") or trade_date)
        rows.append(
            (
                row_date,
                str(row["ts_code"]),
                _number(row.get("open")),
                _number(row.get("high")),
                _number(row.get("low")),
                _number(row.get("close")),
                _number(row.get("pre_close")),
                _number(row.get("change")),
                _number(row.get("pct_chg")),
                _number(row.get("vol")),
                (_number(row.get("amount")) * 1000.0) if _number(row.get("amount")) is not None else None,
                "tushare_daily",
                fetched_at,
            )
        )
    connection.executemany(
        """
        INSERT INTO daily_bars(
            trade_date,ts_code,open,high,low,close,pre_close,change_value,
            pct_chg,volume,amount_yuan,source,fetched_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(trade_date,ts_code) DO UPDATE SET
            open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,
            pre_close=excluded.pre_close,change_value=excluded.change_value,
            pct_chg=excluded.pct_chg,volume=excluded.volume,amount_yuan=excluded.amount_yuan,
            source=excluded.source,fetched_at=excluded.fetched_at
        """,
        rows,
    )
    return len(rows)


_DAILY_BASIC_COLUMNS = [
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]


def _upsert_daily_basic(connection: sqlite3.Connection, frame: pd.DataFrame, trade_date: str) -> int:
    data = _normalise_codes(frame)
    if data.empty:
        raise RuntimeError(f"{trade_date} daily_basic 返回空数据")
    fetched_at = _now_text()
    rows = []
    for row in data.to_dict("records"):
        rows.append(
            (
                _date_text(row.get("trade_date") or trade_date),
                str(row["ts_code"]),
                *[_number(row.get(column)) for column in _DAILY_BASIC_COLUMNS],
                "tushare_daily_basic",
                fetched_at,
            )
        )
    placeholders = ",".join("?" for _ in range(2 + len(_DAILY_BASIC_COLUMNS) + 2))
    update_columns = ",".join(f"{column}=excluded.{column}" for column in _DAILY_BASIC_COLUMNS)
    connection.executemany(
        f"""
        INSERT INTO daily_basic(
            trade_date,ts_code,{','.join(_DAILY_BASIC_COLUMNS)},source,fetched_at
        ) VALUES({placeholders})
        ON CONFLICT(trade_date,ts_code) DO UPDATE SET
            {update_columns},source=excluded.source,fetched_at=excluded.fetched_at
        """,
        rows,
    )
    return len(rows)


def _upsert_adj_factors(connection: sqlite3.Connection, frame: pd.DataFrame, trade_date: str) -> int:
    data = _normalise_codes(frame)
    if data.empty:
        raise RuntimeError(f"{trade_date} adj_factor 返回空数据")
    fetched_at = _now_text()
    rows = []
    for row in data.to_dict("records"):
        factor = _number(row.get("adj_factor"))
        if factor is None or factor <= 0:
            continue
        rows.append(
            (
                _date_text(row.get("trade_date") or trade_date),
                str(row["ts_code"]),
                factor,
                "tushare_adj_factor",
                fetched_at,
            )
        )
    if not rows:
        raise RuntimeError(f"{trade_date} adj_factor 没有有效复权因子")
    connection.executemany(
        """
        INSERT INTO adj_factors(trade_date,ts_code,adj_factor,source,fetched_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(trade_date,ts_code) DO UPDATE SET
            adj_factor=excluded.adj_factor,source=excluded.source,fetched_at=excluded.fetched_at
        """,
        rows,
    )
    return len(rows)


def _set_sync_result(
    connection: sqlite3.Connection,
    *,
    trade_date: str,
    endpoint: str,
    status: str,
    rows: int,
    error: str | None,
) -> None:
    if endpoint not in {"bars", "basic", "adj"}:
        raise ValueError(f"未知同步端点：{endpoint}")
    connection.execute(
        "INSERT OR IGNORE INTO sync_status(trade_date,updated_at) VALUES(?,?)",
        (trade_date, _now_text()),
    )
    connection.execute(
        f"UPDATE sync_status SET {endpoint}_status=?, {endpoint}_rows=?, "
        "last_error=CASE WHEN ? IS NULL THEN last_error ELSE ? END, updated_at=? WHERE trade_date=?",
        (status, int(rows), error, error, _now_text(), trade_date),
    )


def _sync_one_session(connection: sqlite3.Connection, pro: Any, trade_date: str, *, force: bool) -> dict[str, Any]:
    existing = connection.execute(
        "SELECT bars_status,basic_status,adj_status FROM sync_status WHERE trade_date=?",
        (trade_date,),
    ).fetchone()
    statuses = dict(existing) if existing is not None else {}
    with connection:
        connection.execute(
            "INSERT OR IGNORE INTO sync_status(trade_date,updated_at) VALUES(?,?)",
            (trade_date, _now_text()),
        )
        connection.execute(
            """
            UPDATE sync_status
            SET last_error=NULL,attempt_count=attempt_count+1,last_attempt_at=?,updated_at=?
            WHERE trade_date=?
            """,
            (_now_text(), _now_text(), trade_date),
        )
    fields = (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
        "dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
    )
    endpoints = {
        "bars": lambda: pro.daily(trade_date=trade_date),
        "basic": lambda: pro.daily_basic(trade_date=trade_date, fields=fields),
        "adj": lambda: pro.adj_factor(trade_date=trade_date),
    }
    writers = {
        "bars": _upsert_daily_bars,
        "basic": _upsert_daily_basic,
        "adj": _upsert_adj_factors,
    }
    result: dict[str, Any] = {"trade_date": trade_date, "endpoints": {}, "errors": []}
    for endpoint, fetch in endpoints.items():
        if not force and statuses.get(f"{endpoint}_status") == "complete":
            result["endpoints"][endpoint] = {"status": "already_complete"}
            continue
        try:
            frame = fetch()
            with connection:
                row_count = writers[endpoint](connection, frame, trade_date)
                _set_sync_result(
                    connection,
                    trade_date=trade_date,
                    endpoint=endpoint,
                    status="complete",
                    rows=row_count,
                    error=None,
                )
            result["endpoints"][endpoint] = {"status": "complete", "rows": int(row_count)}
        except Exception as exc:
            error = str(exc)
            with connection:
                _set_sync_result(
                    connection,
                    trade_date=trade_date,
                    endpoint=endpoint,
                    status="failed",
                    rows=0,
                    error=error,
                )
            result["endpoints"][endpoint] = {"status": "failed", "error": error}
            result["errors"].append(f"{endpoint}: {error}")
    result["status"] = "complete" if all(
        value.get("status") in {"complete", "already_complete"}
        for value in result["endpoints"].values()
    ) else "partial"
    if result["status"] == "complete":
        with connection:
            connection.execute(
                "UPDATE sync_status SET last_error=NULL,updated_at=? WHERE trade_date=?",
                (_now_text(), trade_date),
            )
    return result


def rebuild_derived_adj_factors(
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> dict[str, Any]:
    """Rebuild a consistent relative adjustment chain from daily pre-close values."""
    resolved = Path(path).expanduser().resolve()
    initialize_daily_warehouse(resolved)
    read_connection = _connect(resolved, create=False)
    write_connection = _connect(resolved, create=False)
    cursor = read_connection.execute(
        "SELECT ts_code,trade_date,close,pre_close FROM daily_bars ORDER BY ts_code,trade_date"
    )
    inserted = 0
    current_code: str | None = None
    previous_close: float | None = None
    factor = 1.0
    pending: list[tuple[str, str, float, str, str]] = []

    def flush() -> None:
        nonlocal inserted
        if not pending:
            return
        with write_connection:
            write_connection.executemany(
                """
                INSERT INTO adj_factors(trade_date,ts_code,adj_factor,source,fetched_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(trade_date,ts_code) DO UPDATE SET
                    adj_factor=excluded.adj_factor,
                    source=excluded.source,
                    fetched_at=excluded.fetched_at
                """,
                pending,
            )
        inserted += len(pending)
        pending.clear()

    try:
        for row in cursor:
            code = str(row["ts_code"])
            close = _number(row["close"])
            pre_close = _number(row["pre_close"])
            if code != current_code:
                current_code = code
                previous_close = None
                factor = 1.0
            elif previous_close is not None and pre_close is not None and pre_close > 0:
                ratio = previous_close / pre_close
                if math.isfinite(ratio) and ratio > 0:
                    factor *= ratio
            if not math.isfinite(factor) or factor <= 0:
                factor = 1.0
            pending.append(
                (
                    str(row["trade_date"]),
                    code,
                    float(factor),
                    "derived_from_daily_pre_close",
                    _now_text(),
                )
            )
            if len(pending) >= 20_000:
                flush()
            if close is not None and close > 0:
                previous_close = close
        flush()
        with write_connection:
            write_connection.execute(
                """
                UPDATE sync_status
                SET adj_status='complete',
                    adj_rows=(
                        SELECT COUNT(*) FROM adj_factors a
                        WHERE a.trade_date=sync_status.trade_date
                    ),
                    updated_at=?
                WHERE bars_status='complete'
                  AND EXISTS(
                      SELECT 1 FROM adj_factors a
                      WHERE a.trade_date=sync_status.trade_date
                  )
                """,
                (_now_text(),),
            )
    finally:
        read_connection.close()
        write_connection.close()
    return {
        "status": "ok",
        "warehouse_path": str(resolved),
        "rows": int(inserted),
        "source": "derived_from_daily_pre_close",
        "definition": "相邻复权因子比=上一实际收盘价/当日pre_close；仅确定相对比例",
    }


def sync_price_warehouse(
    *,
    start_date: str,
    end_date: str | None = None,
    max_sessions: int = 20,
    newest_first: bool = True,
    force: bool = False,
    pause_seconds: float = 0.08,
    workers: int = 1,
    path: Path | str = DAILY_WAREHOUSE_PATH,
    pro: Any | None = None,
) -> dict[str, Any]:
    """Sync full-market daily prices and derive adjustment factors for low-quota accounts."""
    start = _date_text(start_date)
    end = _date_text(end_date or date.today().isoformat())
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    if max_sessions < 0:
        raise ValueError("max_sessions 不能小于0；0表示本次处理全部待同步交易日")
    if not 0 <= pause_seconds <= 10:
        raise ValueError("pause_seconds 必须在0到10秒之间")
    if not 1 <= int(workers) <= 8:
        raise ValueError("workers 必须在1到8之间")
    resolved = Path(path).expanduser().resolve()
    initialize_daily_warehouse(resolved)
    live_provider = pro is None
    provider = pro or _tushare_pro()
    warnings: list[str] = [
        "低额度模式只同步全市场日线并重建相对复权链；不把缺失的历史daily_basic伪造成完整数据"
    ]
    with _connect(resolved) as connection:
        calendar_result = _ensure_trade_calendar(connection, provider, start=start, end=end)
        warnings.extend(calendar_result.get("warnings", []))
        with connection:
            snapshot = _snapshot_stock_master(
                connection,
                provider,
                snapshot_date=date.today().strftime("%Y%m%d"),
            )
        warnings.extend(snapshot.get("warnings", []))
        rows = connection.execute(
            """
            SELECT c.cal_date,COALESCE(s.bars_status,'pending') AS bars_status
            FROM trade_calendar c
            LEFT JOIN sync_status s ON s.trade_date=c.cal_date
            WHERE c.is_open=1 AND c.cal_date BETWEEN ? AND ?
            ORDER BY c.cal_date
            """,
            (start, end),
        ).fetchall()
        pending_dates = [
            str(row["cal_date"])
            for row in rows
            if force or row["bars_status"] != "complete"
        ]
        if newest_first:
            pending_dates.reverse()
        selected = pending_dates if max_sessions == 0 else pending_dates[:max_sessions]
        sessions: list[dict[str, Any]] = []
        with connection:
            for trade_date in selected:
                connection.execute(
                    "INSERT OR IGNORE INTO sync_status(trade_date,updated_at) VALUES(?,?)",
                    (trade_date, _now_text()),
                )
                connection.execute(
                    """
                    UPDATE sync_status
                    SET attempt_count=attempt_count+1,last_attempt_at=?,updated_at=?
                    WHERE trade_date=?
                    """,
                    (_now_text(), _now_text(), trade_date),
                )
        throttle_lock = Lock()
        next_request_at = [time.monotonic()]

        def fetch_one(trade_date: str) -> tuple[str, pd.DataFrame | None, str | None]:
            try:
                if live_provider and workers > 1:
                    with throttle_lock:
                        now = time.monotonic()
                        wait_seconds = max(0.0, next_request_at[0] - now)
                        if wait_seconds > 0:
                            time.sleep(wait_seconds)
                        next_request_at[0] = max(now, next_request_at[0]) + (60.0 / 30.0)
                frame = provider.daily(trade_date=trade_date)
                if pause_seconds > 0:
                    time.sleep(pause_seconds)
                return trade_date, frame, None
            except Exception as exc:
                return trade_date, None, str(exc)

        if workers == 1:
            fetched = (fetch_one(trade_date) for trade_date in selected)
        else:
            executor = ThreadPoolExecutor(max_workers=int(workers))
            futures = [executor.submit(fetch_one, trade_date) for trade_date in selected]
            fetched = (future.result() for future in as_completed(futures))
        try:
            for trade_date, frame, fetch_error in fetched:
                try:
                    if fetch_error is not None:
                        raise RuntimeError(fetch_error)
                    with connection:
                        assert frame is not None
                        count = _upsert_daily_bars(connection, frame, trade_date)
                        _set_sync_result(
                            connection,
                            trade_date=trade_date,
                            endpoint="bars",
                            status="complete",
                            rows=count,
                            error=None,
                        )
                        connection.execute(
                            "UPDATE sync_status SET last_error=NULL,updated_at=? WHERE trade_date=?",
                            (_now_text(), trade_date),
                        )
                    sessions.append({"trade_date": trade_date, "status": "complete", "rows": count})
                except Exception as exc:
                    error = str(exc)
                    with connection:
                        _set_sync_result(
                            connection,
                            trade_date=trade_date,
                            endpoint="bars",
                            status="failed",
                            rows=0,
                            error=error,
                        )
                    sessions.append({"trade_date": trade_date, "status": "failed", "error": error})
        finally:
            if workers != 1:
                executor.shutdown(wait=True)
    with _connect(resolved) as connection:
        with connection:
            connection.execute(
                "INSERT INTO warehouse_meta(meta_key,meta_value) VALUES('adjustment_mode',?) "
                "ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value",
                ("derived_on_load_from_daily_pre_close",),
            )
            connection.execute(
                """
                UPDATE sync_status
                SET adj_status='complete',adj_rows=0,updated_at=?
                WHERE bars_status='complete'
                """,
                (_now_text(),),
            )
    adjustment = {
        "status": "ok",
        "source": "derived_on_load_from_daily_pre_close",
        "rows_persisted": 0,
        "definition": "读取单股历史时递推：相邻复权因子比=上一实际收盘价/当日pre_close",
    }
    status = daily_warehouse_status(resolved)
    return {
        "status": "ok" if all(item["status"] == "complete" for item in sessions) else "partial",
        "mode": "price_only_low_quota",
        "warehouse_path": str(resolved),
        "requested_range": [_display_date(start), _display_date(end)],
        "trading_sessions_in_range": int(len(rows)),
        "pending_before_run": int(len(pending_dates)),
        "attempted_sessions": int(len(sessions)),
        "remaining_after_run": max(0, int(len(pending_dates) - len(sessions))),
        "order": "newest_first" if newest_first else "oldest_first",
        "calendar_source": calendar_result.get("source"),
        "stock_snapshot": snapshot,
        "adjustment_rebuild": adjustment,
        "sessions": sessions,
        "warnings": warnings,
        "warehouse_status": status,
        "resume": {
            "supported": True,
            "next_run_skips_completed_sessions": True,
            "failed_sessions_this_run": [
                item["trade_date"] for item in sessions if item.get("status") != "complete"
            ],
        },
    }


def sync_daily_warehouse(
    *,
    start_date: str,
    end_date: str | None = None,
    max_sessions: int = 20,
    newest_first: bool = True,
    force: bool = False,
    pause_seconds: float = 0.08,
    path: Path | str = DAILY_WAREHOUSE_PATH,
    pro: Any | None = None,
) -> dict[str, Any]:
    """Incrementally sync full-market daily data with per-endpoint resume markers."""
    start = _date_text(start_date)
    end = _date_text(end_date or date.today().isoformat())
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    if max_sessions < 0:
        raise ValueError("max_sessions 不能小于0；0表示本次处理全部待同步交易日")
    if not 0 <= pause_seconds <= 10:
        raise ValueError("pause_seconds 必须在0到10秒之间")
    resolved = Path(path).expanduser().resolve()
    initialize_daily_warehouse(resolved)
    provider = pro or _tushare_pro()
    warnings: list[str] = []
    with _connect(resolved) as connection:
        calendar_result = _ensure_trade_calendar(
            connection,
            provider,
            start=start,
            end=end,
        )
        warnings.extend(calendar_result.get("warnings", []))
        with connection:
            snapshot = _snapshot_stock_master(
                connection,
                provider,
                snapshot_date=date.today().strftime("%Y%m%d"),
            )
        warnings.extend(snapshot.get("warnings", []))
        rows = connection.execute(
            """
            SELECT c.cal_date,
                   COALESCE(s.bars_status,'pending') AS bars_status,
                   COALESCE(s.basic_status,'pending') AS basic_status,
                   COALESCE(s.adj_status,'pending') AS adj_status
            FROM trade_calendar c
            LEFT JOIN sync_status s ON s.trade_date=c.cal_date
            WHERE c.is_open=1 AND c.cal_date BETWEEN ? AND ?
            ORDER BY c.cal_date
            """,
            (start, end),
        ).fetchall()
        pending = [
            str(row["cal_date"])
            for row in rows
            if force or not all(row[field] == "complete" for field in ["bars_status", "basic_status", "adj_status"])
        ]
        if newest_first:
            pending.reverse()
        selected = pending if max_sessions == 0 else pending[:max_sessions]
        sessions: list[dict[str, Any]] = []
        for index, trade_date in enumerate(selected):
            sessions.append(_sync_one_session(connection, provider, trade_date, force=force))
            if pause_seconds > 0 and index + 1 < len(selected):
                time.sleep(pause_seconds)
    status = daily_warehouse_status(resolved)
    return {
        "status": "ok" if all(item["status"] == "complete" for item in sessions) else "partial",
        "warehouse_path": str(resolved),
        "requested_range": [_display_date(start), _display_date(end)],
        "trading_sessions_in_range": int(len(rows)),
        "pending_before_run": int(len(pending)),
        "attempted_sessions": int(len(sessions)),
        "remaining_after_run": max(0, int(len(pending) - len(sessions))),
        "order": "newest_first" if newest_first else "oldest_first",
        "force": force,
        "calendar_source": calendar_result.get("source"),
        "stock_snapshot": snapshot,
        "sessions": sessions,
        "warnings": warnings,
        "warehouse_status": status,
        "resume": {
            "supported": True,
            "next_run_skips_completed_endpoints": True,
            "failed_sessions_this_run": [
                item["trade_date"] for item in sessions if item.get("status") != "complete"
            ],
        },
    }


def enrich_latest_daily_basic(
    path: Path | str = DAILY_WAREHOUSE_PATH,
    pro: Any | None = None,
) -> dict[str, Any]:
    """Fill one newest missing daily_basic session for low-quota accounts."""
    resolved = Path(path).expanduser().resolve()
    initialize_daily_warehouse(resolved)
    provider = pro or _tushare_pro()
    fields = (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
        "dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
    )
    with _connect(resolved) as connection:
        pending = connection.execute(
            """
            SELECT trade_date
            FROM sync_status
            WHERE bars_status='complete'
              AND COALESCE(basic_status,'pending')<>'complete'
            ORDER BY trade_date DESC
            LIMIT 1
            """
        ).fetchone()
        if pending is None:
            return {
                "status": "ok",
                "enrich_status": "up_to_date",
                "attempted_sessions": 0,
                "message": "已有日线交易日的 daily_basic 已全部补齐",
                "warehouse_status": daily_warehouse_status(resolved),
            }
        trade_date = str(pending["trade_date"])
        with connection:
            connection.execute(
                """
                UPDATE sync_status
                SET last_error=NULL,attempt_count=attempt_count+1,last_attempt_at=?,updated_at=?
                WHERE trade_date=?
                """,
                (_now_text(), _now_text(), trade_date),
            )
        try:
            frame = provider.daily_basic(trade_date=trade_date, fields=fields)
            with connection:
                row_count = _upsert_daily_basic(connection, frame, trade_date)
                _set_sync_result(
                    connection,
                    trade_date=trade_date,
                    endpoint="basic",
                    status="complete",
                    rows=row_count,
                    error=None,
                )
                connection.execute(
                    "UPDATE sync_status SET last_error=NULL,updated_at=? WHERE trade_date=?",
                    (_now_text(), trade_date),
                )
            session = {
                "trade_date": _display_date(trade_date),
                "status": "complete",
                "rows": int(row_count),
            }
            status = "ok"
        except Exception as exc:
            error = str(exc)
            with connection:
                _set_sync_result(
                    connection,
                    trade_date=trade_date,
                    endpoint="basic",
                    status="failed",
                    rows=0,
                    error=error,
                )
            session = {
                "trade_date": _display_date(trade_date),
                "status": "failed",
                "error": error,
            }
            status = "partial"
    return {
        "status": status,
        "enrich_status": "updated" if status == "ok" else "rate_limited_or_failed",
        "attempted_sessions": 1,
        "session": session,
        "message": "每次只补一个最近交易日，适合低额度账号重复渐进执行",
        "warehouse_status": daily_warehouse_status(resolved),
    }


def daily_warehouse_status(path: Path | str = DAILY_WAREHOUSE_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "status": "not_initialized",
            "warehouse_path": str(resolved),
            "daily_frequency_only": True,
            "full_market_training_ready": False,
        }
    initialize_daily_warehouse(resolved)
    with _connect(resolved, create=False) as connection:
        schema = connection.execute(
            "SELECT meta_value FROM warehouse_meta WHERE meta_key='schema_version'"
        ).fetchone()
        session = connection.execute(
            """
            SELECT COUNT(*) AS tracked,
                   SUM(CASE WHEN bars_status='complete' AND basic_status='complete' AND adj_status='complete' THEN 1 ELSE 0 END) AS complete,
                   SUM(CASE WHEN bars_status='complete' AND adj_status='complete' THEN 1 ELSE 0 END) AS price_complete,
                   SUM(CASE WHEN basic_status='complete' THEN 1 ELSE 0 END) AS basic_complete,
                   SUM(CASE WHEN last_error IS NOT NULL AND last_error<>'' THEN 1 ELSE 0 END) AS failed,
                   MAX(attempt_count) AS maximum_attempts,
                   MIN(CASE WHEN bars_status='complete' AND basic_status='complete' AND adj_status='complete' THEN trade_date END) AS min_complete,
                   MAX(CASE WHEN bars_status='complete' AND basic_status='complete' AND adj_status='complete' THEN trade_date END) AS max_complete,
                   MIN(CASE WHEN bars_status='complete' AND adj_status='complete' THEN trade_date END) AS min_price_complete,
                   MAX(CASE WHEN bars_status='complete' AND adj_status='complete' THEN trade_date END) AS max_price_complete
            FROM sync_status
            """
        ).fetchone()
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in [
                "daily_bars",
                "daily_basic",
                "adj_factors",
                "trade_calendar",
                "stock_snapshots",
                "board_snapshots",
                "board_members",
            ]
        }
        recent_failures = [
            dict(row)
            for row in connection.execute(
                """
                SELECT trade_date,attempt_count,last_attempt_at,last_error
                FROM sync_status
                WHERE last_error IS NOT NULL AND last_error<>''
                ORDER BY updated_at DESC
                LIMIT 20
                """
            ).fetchall()
        ]
        latest_snapshot = connection.execute(
            "SELECT MAX(snapshot_date) AS value FROM stock_snapshots"
        ).fetchone()["value"]
        adjustment_row = connection.execute(
            "SELECT meta_value FROM warehouse_meta WHERE meta_key='adjustment_mode'"
        ).fetchone()
        adjustment_mode = str(adjustment_row["meta_value"]) if adjustment_row else "persisted_factor"
    complete = int(session["complete"] or 0)
    price_complete = int(session["price_complete"] or 0)
    basic_complete = int(session["basic_complete"] or 0)
    tracked = int(session["tracked"] or 0)
    failed = int(session["failed"] or 0)
    low_quota_mode = adjustment_mode == "derived_on_load_from_daily_pre_close"
    reported_complete = price_complete if low_quota_mode else complete
    reported_min = session["min_price_complete"] if low_quota_mode else session["min_complete"]
    reported_max = session["max_price_complete"] if low_quota_mode else session["max_complete"]
    return {
        "status": "ok",
        "warehouse_path": str(resolved),
        "schema_version": int(schema["meta_value"]) if schema else None,
        "size_mb": round(resolved.stat().st_size / (1024 * 1024), 2),
        "tracked_sessions": tracked,
        "complete_sessions": reported_complete,
        "complete_ratio": round(reported_complete / tracked, 4) if tracked else 0.0,
        "complete_date_range": [
            _display_date(reported_min),
            _display_date(reported_max),
        ],
        "all_endpoints_complete_sessions": complete,
        "all_endpoints_complete_ratio": round(complete / tracked, 4) if tracked else 0.0,
        "price_complete_sessions": price_complete,
        "price_complete_date_range": [
            _display_date(session["min_price_complete"]),
            _display_date(session["max_price_complete"]),
        ],
        "daily_basic_complete_sessions": basic_complete,
        "failed_sessions": failed,
        "maximum_attempts_for_one_session": int(session["maximum_attempts"] or 0),
        "recent_failures": [
            {
                **item,
                "trade_date": _display_date(str(item["trade_date"])),
            }
            for item in recent_failures
        ],
        "resume": {
            "supported": True,
            "behavior": "再次执行相同区间的 warehouse sync 会跳过完整交易日，只重试缺失或失败端点",
            "remaining_sessions": max(0, tracked - reported_complete),
        },
        "adjustment_mode": adjustment_mode,
        "row_counts": counts,
        "latest_stock_snapshot": (
            _display_date(latest_snapshot)
        ),
        "daily_frequency_only": True,
        "full_market_training_ready": price_complete >= FULL_MARKET_MIN_COMPLETE_SESSIONS,
        "full_market_training_min_complete_sessions": FULL_MARKET_MIN_COMPLETE_SESSIONS,
        "pit_scope": (
            "日线和交易日历按交易日保存；相对前复权在读取单股时由pre_close递推；"
            "低额度模式不伪造daily_basic；股票资料按实际抓取日留快照，不会把当前行业标签倒填为历史标签"
            if adjustment_mode == "derived_on_load_from_daily_pre_close"
            else "日线、复权因子、daily_basic和交易日历按交易日保存；股票资料按实际抓取日留快照，"
            "不会把当前行业标签倒填为历史标签"
        ),
        "remaining_bias": "数据源不能倒推出仓库建立前的历史行业/板块成分快照；低额度模式不保证历史daily_basic完整",
    }


def warehouse_range_coverage(
    *,
    start_date: str,
    end_date: str,
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> dict[str, Any]:
    """Report whether all model inputs are complete for a requested daily range."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "status": "warehouse_not_initialized",
            "coverage": 0.0,
            "ready": False,
        }
    start = _date_text(start_date)
    end = _date_text(end_date)
    with _connect(resolved, create=False) as connection:
        calendar_rows = connection.execute(
            "SELECT meta_key,meta_value FROM warehouse_meta "
            "WHERE meta_key IN ('calendar_coverage_start','calendar_coverage_end')"
        ).fetchall()
        calendar_meta = {str(row["meta_key"]): str(row["meta_value"]) for row in calendar_rows}
        calendar_start = calendar_meta.get("calendar_coverage_start")
        calendar_end = calendar_meta.get("calendar_coverage_end")
        calendar_covers_request = bool(
            calendar_start
            and calendar_end
            and calendar_start <= start
            and calendar_end >= end
        )
        expected = int(
            connection.execute(
                "SELECT COUNT(*) FROM trade_calendar WHERE is_open=1 AND cal_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()[0]
        )
        price_complete = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sync_status
                WHERE trade_date BETWEEN ? AND ?
                  AND bars_status='complete' AND adj_status='complete'
                """,
                (start, end),
            ).fetchone()[0]
        )
        basic_complete = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sync_status
                WHERE trade_date BETWEEN ? AND ? AND basic_status='complete'
                """,
                (start, end),
            ).fetchone()[0]
        )
    coverage = price_complete / expected if expected else 0.0
    basic_coverage = basic_complete / expected if expected else 0.0
    return {
        "status": "ok" if calendar_covers_request else "calendar_range_incomplete",
        "requested_range": [_display_date(start), _display_date(end)],
        "calendar_coverage_range": [
            _display_date(calendar_start),
            _display_date(calendar_end),
        ],
        "calendar_covers_requested_range": calendar_covers_request,
        "expected_sessions": expected,
        "complete_sessions": price_complete,
        "coverage": round(coverage, 4),
        "ready": bool(calendar_covers_request and expected >= 60 and coverage >= 0.90),
        "minimum_coverage": 0.90,
        "coverage_definition": "本地交易日历先完整覆盖请求区间，再计算全市场日线与相对复权链完整交易日覆盖率",
        "daily_basic_complete_sessions": basic_complete,
        "daily_basic_coverage": round(basic_coverage, 4),
    }


def _stock_lifetime(connection: sqlite3.Connection, code: str) -> tuple[str | None, str | None]:
    row = connection.execute(
        """
        SELECT list_date,delist_date
        FROM stock_snapshots
        WHERE ts_code=?
        ORDER BY snapshot_date DESC
        LIMIT 1
        """,
        (code,),
    ).fetchone()
    return (row["list_date"], row["delist_date"]) if row else (None, None)


def load_qfq_history_from_warehouse(
    code: str,
    *,
    start_date: str,
    end_date: str,
    path: Path | str = DAILY_WAREHOUSE_PATH,
    minimum_sync_coverage: float = 0.90,
    minimum_rows: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one stock's QFQ history only when the warehouse range is sufficiently complete."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return pd.DataFrame(), {"status": "warehouse_not_initialized", "warehouse_path": str(resolved)}
    normalized = _normalize_code(code)
    start = _date_text(start_date)
    end = _date_text(end_date)
    with _connect(resolved, create=False) as connection:
        adjustment_row = connection.execute(
            "SELECT meta_value FROM warehouse_meta WHERE meta_key='adjustment_mode'"
        ).fetchone()
        adjustment_mode = str(adjustment_row["meta_value"]) if adjustment_row else "persisted_factor"
        list_date, delist_date = _stock_lifetime(connection, normalized)
        effective_start = max(start, list_date) if list_date else start
        effective_end = min(end, delist_date) if delist_date else end
        calendar_rows = connection.execute(
            "SELECT meta_key,meta_value FROM warehouse_meta "
            "WHERE meta_key IN ('calendar_coverage_start','calendar_coverage_end')"
        ).fetchall()
        calendar_meta = {str(row["meta_key"]): str(row["meta_value"]) for row in calendar_rows}
        calendar_start = calendar_meta.get("calendar_coverage_start")
        calendar_end = calendar_meta.get("calendar_coverage_end")
        calendar_covers_request = bool(
            calendar_start
            and calendar_end
            and calendar_start <= effective_start
            and calendar_end >= effective_end
        )
        if not calendar_covers_request:
            return pd.DataFrame(), {
                "status": "calendar_range_incomplete",
                "requested_range": [_display_date(effective_start), _display_date(effective_end)],
                "calendar_coverage_range": [
                    _display_date(calendar_start),
                    _display_date(calendar_end),
                ],
                "sync_coverage": 0.0,
            }
        expected = int(
            connection.execute(
                "SELECT COUNT(*) FROM trade_calendar WHERE is_open=1 AND cal_date BETWEEN ? AND ?",
                (effective_start, effective_end),
            ).fetchone()[0]
        )
        complete = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sync_status
                WHERE trade_date BETWEEN ? AND ?
                  AND bars_status='complete' AND adj_status='complete'
                """,
                (effective_start, effective_end),
            ).fetchone()[0]
        )
        sync_coverage = complete / expected if expected else 0.0
        if expected == 0 or sync_coverage < minimum_sync_coverage:
            return pd.DataFrame(), {
                "status": "insufficient_warehouse_coverage",
                "expected_sessions": expected,
                "complete_bar_and_factor_sessions": complete,
                "sync_coverage": round(sync_coverage, 4),
                "minimum_sync_coverage": minimum_sync_coverage,
            }
        if adjustment_mode == "derived_on_load_from_daily_pre_close":
            rows = connection.execute(
                """
                SELECT b.trade_date,b.open,b.high,b.low,b.close,b.pre_close,b.pct_chg,
                       b.volume,b.amount_yuan,NULL AS adj_factor,d.turnover_rate
                FROM daily_bars b
                LEFT JOIN daily_basic d ON d.trade_date=b.trade_date AND d.ts_code=b.ts_code
                WHERE b.ts_code=? AND b.trade_date BETWEEN ? AND ?
                ORDER BY b.trade_date
                """,
                (normalized, effective_start, effective_end),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT b.trade_date,b.open,b.high,b.low,b.close,b.pre_close,b.pct_chg,
                       b.volume,b.amount_yuan,a.adj_factor,d.turnover_rate
                FROM daily_bars b
                JOIN adj_factors a ON a.trade_date=b.trade_date AND a.ts_code=b.ts_code
                LEFT JOIN daily_basic d ON d.trade_date=b.trade_date AND d.ts_code=b.ts_code
                WHERE b.ts_code=? AND b.trade_date BETWEEN ? AND ?
                ORDER BY b.trade_date
                """,
                (normalized, effective_start, effective_end),
            ).fetchall()
    data = pd.DataFrame([dict(row) for row in rows])
    if len(data) < minimum_rows:
        return pd.DataFrame(), {
            "status": "insufficient_stock_rows",
            "rows": int(len(data)),
            "minimum_rows": minimum_rows,
            "sync_coverage": round(sync_coverage, 4),
        }
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    for column in [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "volume",
        "amount_yuan",
        "adj_factor",
        "turnover_rate",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if adjustment_mode == "derived_on_load_from_daily_pre_close":
        previous_close = data["close"].shift(1)
        step = previous_close / data["pre_close"]
        step = step.where(step.gt(0) & np.isfinite(step), 1.0).fillna(1.0)
        relative_factor = step.cumprod()
        ratio = relative_factor / float(relative_factor.iloc[-1])
        adjustment_label = "qfq_derived_on_load_from_daily_pre_close"
    else:
        if data["adj_factor"].isna().any() or not (data["adj_factor"] > 0).all():
            return pd.DataFrame(), {"status": "invalid_adjustment_factor"}
        latest_factor = float(data["adj_factor"].iloc[-1])
        ratio = data["adj_factor"] / latest_factor
        adjustment_label = "qfq_by_warehouse_adj_factor"
    for column in ["open", "high", "low", "close", "pre_close"]:
        data[column] = data[column] * ratio
    data = data.drop(columns=["adj_factor"])
    return data, {
        "status": "ok",
        "source": "tushare_daily_warehouse",
        "adjustment": adjustment_label,
        "rows": int(len(data)),
        "sync_coverage": round(sync_coverage, 4),
        "warehouse_path": str(resolved),
    }


def load_qfq_histories_from_warehouse(
    codes: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    path: Path | str = DAILY_WAREHOUSE_PATH,
    minimum_sync_coverage: float = 0.90,
    minimum_rows: int = 60,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load several adjusted histories with one SQLite query."""
    resolved = Path(path).expanduser().resolve()
    normalized = sorted({_normalize_code(code) for code in codes})
    if not resolved.is_file() or not normalized:
        return {}, {"status": "warehouse_not_initialized_or_no_codes"}
    start = _date_text(start_date)
    end = _date_text(end_date)
    coverage = warehouse_range_coverage(start_date=start, end_date=end, path=resolved)
    if not coverage.get("ready") or float(coverage.get("coverage", 0.0)) < minimum_sync_coverage:
        return {}, {
            "status": "insufficient_warehouse_coverage",
            "range": coverage,
            "minimum_sync_coverage": minimum_sync_coverage,
        }
    placeholders = ",".join("?" for _ in normalized)
    with _connect(resolved, create=False) as connection:
        adjustment_row = connection.execute(
            "SELECT meta_value FROM warehouse_meta WHERE meta_key='adjustment_mode'"
        ).fetchone()
        adjustment_mode = str(adjustment_row["meta_value"]) if adjustment_row else "persisted_factor"
        factor_join = (
            "LEFT JOIN adj_factors a ON a.trade_date=b.trade_date AND a.ts_code=b.ts_code"
            if adjustment_mode == "derived_on_load_from_daily_pre_close"
            else "JOIN adj_factors a ON a.trade_date=b.trade_date AND a.ts_code=b.ts_code"
        )
        rows = connection.execute(
            f"""
            SELECT b.ts_code,b.trade_date,b.open,b.high,b.low,b.close,b.pre_close,b.pct_chg,
                   b.volume,b.amount_yuan,a.adj_factor,d.turnover_rate
            FROM daily_bars b
            {factor_join}
            LEFT JOIN daily_basic d ON d.trade_date=b.trade_date AND d.ts_code=b.ts_code
            WHERE b.ts_code IN ({placeholders}) AND b.trade_date BETWEEN ? AND ?
            ORDER BY b.ts_code,b.trade_date
            """,
            [*normalized, start, end],
        ).fetchall()
    combined = pd.DataFrame([dict(row) for row in rows])
    if combined.empty:
        return {}, {"status": "no_rows", "range": coverage}
    combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce")
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "volume",
        "amount_yuan",
        "adj_factor",
        "turnover_rate",
    ]
    for column in numeric_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    histories: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for code, group in combined.groupby("ts_code", sort=False):
        data = group.sort_values("trade_date").reset_index(drop=True).copy()
        if len(data) < minimum_rows:
            skipped[str(code)] = f"rows={len(data)}<minimum_rows={minimum_rows}"
            continue
        if adjustment_mode == "derived_on_load_from_daily_pre_close":
            step = data["close"].shift(1) / data["pre_close"]
            step = step.where(step.gt(0) & np.isfinite(step), 1.0).fillna(1.0)
            relative_factor = step.cumprod()
            ratio = relative_factor / float(relative_factor.iloc[-1])
            adjustment_label = "qfq_derived_on_load_from_daily_pre_close"
        else:
            if data["adj_factor"].isna().any() or not (data["adj_factor"] > 0).all():
                skipped[str(code)] = "invalid_adjustment_factor"
                continue
            ratio = data["adj_factor"] / float(data["adj_factor"].iloc[-1])
            adjustment_label = "qfq_by_warehouse_adj_factor"
        for column in ["open", "high", "low", "close", "pre_close"]:
            data[column] = data[column] * ratio
        data = data.drop(columns=["adj_factor"])
        data.attrs["adjustment"] = adjustment_label
        histories[str(code)] = data
    missing = sorted(set(normalized) - set(histories))
    return histories, {
        "status": "ok" if histories else "no_usable_stocks",
        "source": "tushare_daily_warehouse_batch",
        "requested_stocks": int(len(normalized)),
        "loaded_stocks": int(len(histories)),
        "missing_stocks": missing,
        "skipped": skipped,
        "rows": int(sum(len(frame) for frame in histories.values())),
        "range": coverage,
        "warehouse_path": str(resolved),
    }


def load_daily_basic_from_warehouse(
    codes: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    path: Path | str = DAILY_WAREHOUSE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    normalized = sorted({_normalize_code(code) for code in codes})
    if not resolved.is_file() or not normalized:
        return pd.DataFrame(), {"status": "warehouse_not_initialized_or_no_codes"}
    start = _date_text(start_date)
    end = _date_text(end_date)
    placeholders = ",".join("?" for _ in normalized)
    with _connect(resolved, create=False) as connection:
        rows = connection.execute(
            f"""
            SELECT ts_code,trade_date,turnover_rate,pe_ttm,pb,total_mv,circ_mv
            FROM daily_basic
            WHERE ts_code IN ({placeholders}) AND trade_date BETWEEN ? AND ?
            ORDER BY ts_code,trade_date
            """,
            [*normalized, start, end],
        ).fetchall()
        calendar_sessions = int(
            connection.execute(
                "SELECT COUNT(*) FROM trade_calendar WHERE is_open=1 AND cal_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()[0]
        )
        complete_sessions = int(
            connection.execute(
                "SELECT COUNT(*) FROM sync_status WHERE trade_date BETWEEN ? AND ? AND basic_status='complete'",
                (start, end),
            ).fetchone()[0]
        )
    data = pd.DataFrame([dict(row) for row in rows])
    if not data.empty:
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    coverage = complete_sessions / calendar_sessions if calendar_sessions else 0.0
    return data, {
        "status": "ok" if not data.empty else "no_rows",
        "rows": int(len(data)),
        "stocks": int(data["ts_code"].nunique()) if not data.empty else 0,
        "calendar_sync_coverage": round(coverage, 4),
        "source": "tushare_daily_warehouse",
    }


__all__ = [
    "DAILY_WAREHOUSE_PATH",
    "board_constituent_history_status",
    "daily_warehouse_status",
    "enrich_latest_daily_basic",
    "initialize_daily_warehouse",
    "load_daily_basic_from_warehouse",
    "load_board_membership_history",
    "load_qfq_histories_from_warehouse",
    "load_qfq_history_from_warehouse",
    "load_stock_snapshot_asof",
    "rebuild_derived_adj_factors",
    "snapshot_board_constituents",
    "sync_daily_warehouse",
    "sync_price_warehouse",
    "warehouse_range_coverage",
]
