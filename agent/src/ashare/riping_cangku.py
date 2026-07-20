"""Persistent full-market daily warehouse for reproducible A-share research."""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ashare.shuju_yuan import CACHE_DIR, _normalize_code, _tushare_pro


DAILY_WAREHOUSE_PATH = CACHE_DIR / "a_share_daily_warehouse.sqlite3"
WAREHOUSE_SCHEMA_VERSION = 1
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
                last_error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_daily_bars_code_date
                ON daily_bars (ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_daily_basic_code_date
                ON daily_basic (ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_adj_factors_code_date
                ON adj_factors (ts_code, trade_date);
            CREATE INDEX IF NOT EXISTS idx_stock_snapshots_code_date
                ON stock_snapshots (ts_code, snapshot_date);
            """
        )
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


def _upsert_calendar(connection: sqlite3.Connection, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        raise RuntimeError("Tushare trade_cal 返回空数据")
    fetched_at = _now_text()
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            (
                _date_text(row.get("cal_date")),
                int(_number(row.get("is_open")) or 0),
                _date_text(row.get("pretrade_date")) if row.get("pretrade_date") else None,
                str(row.get("exchange") or "SSE"),
                "tushare_trade_cal",
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
            "UPDATE sync_status SET last_error=NULL, updated_at=? WHERE trade_date=?",
            (_now_text(), trade_date),
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
    return result


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
        calendar = provider.trade_cal(
            exchange="SSE",
            start_date=start,
            end_date=end,
            fields="exchange,cal_date,is_open,pretrade_date",
        )
        with connection:
            _upsert_calendar(connection, calendar)
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
        "stock_snapshot": snapshot,
        "sessions": sessions,
        "warnings": warnings,
        "warehouse_status": status,
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
    with _connect(resolved, create=False) as connection:
        schema = connection.execute(
            "SELECT meta_value FROM warehouse_meta WHERE meta_key='schema_version'"
        ).fetchone()
        session = connection.execute(
            """
            SELECT COUNT(*) AS tracked,
                   SUM(CASE WHEN bars_status='complete' AND basic_status='complete' AND adj_status='complete' THEN 1 ELSE 0 END) AS complete,
                   MIN(CASE WHEN bars_status='complete' AND basic_status='complete' AND adj_status='complete' THEN trade_date END) AS min_complete,
                   MAX(CASE WHEN bars_status='complete' AND basic_status='complete' AND adj_status='complete' THEN trade_date END) AS max_complete
            FROM sync_status
            """
        ).fetchone()
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ["daily_bars", "daily_basic", "adj_factors", "trade_calendar", "stock_snapshots"]
        }
        latest_snapshot = connection.execute(
            "SELECT MAX(snapshot_date) AS value FROM stock_snapshots"
        ).fetchone()["value"]
    complete = int(session["complete"] or 0)
    tracked = int(session["tracked"] or 0)
    return {
        "status": "ok",
        "warehouse_path": str(resolved),
        "schema_version": int(schema["meta_value"]) if schema else None,
        "size_mb": round(resolved.stat().st_size / (1024 * 1024), 2),
        "tracked_sessions": tracked,
        "complete_sessions": complete,
        "complete_ratio": round(complete / tracked, 4) if tracked else 0.0,
        "complete_date_range": [
            _display_date(session["min_complete"]),
            _display_date(session["max_complete"]),
        ],
        "row_counts": counts,
        "latest_stock_snapshot": (
            _display_date(latest_snapshot)
        ),
        "daily_frequency_only": True,
        "full_market_training_ready": complete >= FULL_MARKET_MIN_COMPLETE_SESSIONS,
        "full_market_training_min_complete_sessions": FULL_MARKET_MIN_COMPLETE_SESSIONS,
        "pit_scope": (
            "日线、复权因子、daily_basic和交易日历按交易日保存；股票资料按实际抓取日留快照，"
            "不会把当前行业标签倒填为历史标签"
        ),
        "remaining_bias": "数据源不能倒推出仓库建立前的历史行业/板块成分快照",
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
        expected = int(
            connection.execute(
                "SELECT COUNT(*) FROM trade_calendar WHERE is_open=1 AND cal_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()[0]
        )
        complete = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sync_status
                WHERE trade_date BETWEEN ? AND ?
                  AND bars_status='complete' AND basic_status='complete' AND adj_status='complete'
                """,
                (start, end),
            ).fetchone()[0]
        )
    coverage = complete / expected if expected else 0.0
    return {
        "status": "ok",
        "requested_range": [_display_date(start), _display_date(end)],
        "expected_sessions": expected,
        "complete_sessions": complete,
        "coverage": round(coverage, 4),
        "ready": bool(expected >= 60 and coverage >= 0.90),
        "minimum_coverage": 0.90,
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
        list_date, delist_date = _stock_lifetime(connection, normalized)
        effective_start = max(start, list_date) if list_date else start
        effective_end = min(end, delist_date) if delist_date else end
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
    if data["adj_factor"].isna().any() or not (data["adj_factor"] > 0).all():
        return pd.DataFrame(), {"status": "invalid_adjustment_factor"}
    latest_factor = float(data["adj_factor"].iloc[-1])
    ratio = data["adj_factor"] / latest_factor
    for column in ["open", "high", "low", "close", "pre_close"]:
        data[column] = data[column] * ratio
    data = data.drop(columns=["adj_factor"])
    return data, {
        "status": "ok",
        "source": "tushare_daily_warehouse",
        "adjustment": "qfq_by_warehouse_adj_factor",
        "rows": int(len(data)),
        "sync_coverage": round(sync_coverage, 4),
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
    "daily_warehouse_status",
    "initialize_daily_warehouse",
    "load_daily_basic_from_warehouse",
    "load_qfq_history_from_warehouse",
    "sync_daily_warehouse",
    "warehouse_range_coverage",
]
