"""Uniform data-health and failure summaries for user-facing research results."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import pandas as pd


def classify_failure(value: Any) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ["429", "rate limit", "限流", "频率", "too many"]):
        return "rate_limited"
    if any(token in text for token in ["401", "403", "token", "权限", "积分", "认证"]):
        return "authentication_or_quota"
    if any(token in text for token in ["timeout", "timed out", "超时"]):
        return "timeout"
    if any(token in text for token in ["proxy", "connection", "network", "dns", "连接", "网络"]):
        return "network"
    if any(token in text for token in ["empty", "no rows", "没有", "为空", "无数据"]):
        return "empty_data"
    if any(token in text for token in ["cache", "缓存", "sqlite", "database"]):
        return "cache_or_warehouse"
    if any(token in text for token in ["field", "column", "字段", "格式", "schema"]):
        return "schema_or_format"
    return "unexpected"


def summarize_failures(messages: Iterable[Any], *, limit: int = 20) -> dict[str, Any]:
    values = [str(value) for value in messages if str(value or "").strip()]
    categories = Counter(classify_failure(value) for value in values)
    return {
        "count": int(len(values)),
        "category_counts": dict(categories),
        "examples": [
            {"category": classify_failure(value), "message": value}
            for value in values[: max(0, int(limit))]
        ],
    }


def build_data_health(
    *,
    as_of: str | None,
    expected_as_of: str | None,
    freshness: dict[str, Any] | None,
    sources: dict[str, Any],
    warehouse: dict[str, Any] | None,
    warnings: Iterable[Any],
    errors: Iterable[Any],
    constituent_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warning_summary = summarize_failures(warnings)
    error_summary = summarize_failures(errors)
    freshness_status = str((freshness or {}).get("status") or "unknown")
    warehouse_ready = bool((warehouse or {}).get("ready"))
    degraded = bool(error_summary["count"] or freshness_status in {"possibly_stale", "too_stale"})
    return {
        "status": "degraded" if degraded else "ready",
        "as_of": as_of,
        "expected_latest_completed_date": expected_as_of,
        "freshness": freshness or {},
        "sources": sources,
        "warehouse": {
            "ready": warehouse_ready,
            **(warehouse or {}),
        },
        "cache_usage_detected": any("cache" in str(value).lower() or "缓存" in str(value) for value in warnings),
        "constituent_history": constituent_history or {"status": "not_applicable"},
        "warnings": warning_summary,
        "errors": error_summary,
    }


def filter_panel_by_membership_snapshots(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep rows whose stock belonged to the latest board snapshot known on that date."""
    if panel.empty or membership.empty:
        return panel, {"applied": False, "reason": "membership_history_empty"}
    required_panel = {"trade_date", "ts_code"}
    required_membership = {"snapshot_date", "ts_code"}
    if not required_panel.issubset(panel.columns) or not required_membership.issubset(membership.columns):
        return panel, {"applied": False, "reason": "membership_history_schema_incomplete"}
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    history = membership.copy()
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], errors="coerce").dt.normalize()
    snapshot_dates = sorted(history["snapshot_date"].dropna().unique())
    keep = pd.Series(False, index=data.index, dtype=bool)
    for index, snapshot_date in enumerate(snapshot_dates):
        start = pd.Timestamp(snapshot_date)
        end = pd.Timestamp(snapshot_dates[index + 1]) if index + 1 < len(snapshot_dates) else None
        codes = set(history.loc[history["snapshot_date"] == start, "ts_code"].astype(str))
        interval = data["trade_date"].ge(start)
        if end is not None:
            interval &= data["trade_date"].lt(end)
        keep |= interval & data["ts_code"].astype(str).isin(codes)
    filtered = data.loc[keep].copy().reset_index(drop=True)
    return filtered, {
        "applied": True,
        "definition": "每个交易日只保留当日之前最近一次真实板块快照中的成员",
        "snapshot_count": int(len(snapshot_dates)),
        "input_rows": int(len(data)),
        "output_rows": int(len(filtered)),
        "excluded_rows": int(len(data) - len(filtered)),
        "usable_date_range": [
            pd.Timestamp(snapshot_dates[0]).strftime("%Y-%m-%d") if snapshot_dates else None,
            data["trade_date"].max().strftime("%Y-%m-%d") if not data.empty else None,
        ],
    }


__all__ = [
    "build_data_health",
    "classify_failure",
    "filter_panel_by_membership_snapshots",
    "summarize_failures",
]
