"""范围选股的显式市值条件；只消费本次请求取得的同日日终估值。"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.ashare.shuju_yuan import biaozhunhua_gupiao_daima


def jiexi_shizhi_tiaojian(value: Any) -> dict[str, Any] | None:
    """解析结构化条件，不从“小盘”等自然语言猜测阈值。"""
    if value is None:
        return None  # 保留程序直接调用的无市值限制用法。
    question = "请明确市值口径和上限，例如“流通市值低于100亿元”或“总市值不超过50亿元”。"
    if not isinstance(value, dict) or set(value) - {"mode", "basis", "max_yi", "inclusive"}:
        raise ValueError(question)
    if value.get("mode") == "none" and all(item is None for key, item in value.items() if key != "mode"):
        return None
    if value.get("mode") != "upper_limit" or value.get("basis") not in {"total", "circulating"}:
        raise ValueError(question)
    maximum = value.get("max_yi")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise ValueError(question)
    try:
        maximum_yuan = float(maximum) * 100_000_000
    except (OverflowError, ValueError):
        raise ValueError(question) from None
    if not math.isfinite(maximum_yuan) or maximum_yuan <= 0:
        raise ValueError(question)
    inclusive = value.get("inclusive", False)
    if not isinstance(inclusive, bool):
        raise ValueError(question)
    label = "总市值" if value["basis"] == "total" else "流通市值"
    return {
        "basis": value["basis"], "basis_label": label,
        "maximum_yuan": maximum_yuan, "maximum_yi": float(maximum),
        "operator": "le" if inclusive else "lt",
        "description": f"{label}{'不超过' if inclusive else '低于'}{maximum:g}亿元",
        "time_basis": "最新完整交易日日终市值",
    }


def guolv_shizhi(
    data: pd.DataFrame,
    condition: dict[str, Any],
    valuation: pd.DataFrame,
    *,
    analysis_date: pd.Timestamp,
    fetched_at: str | None,
    source_error: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """先核验股票、日期、口径、单位及冲突，再过滤；缺失不等于未达标。"""
    target = pd.Timestamp(analysis_date).normalize()
    field = "total_mv" if condition["basis"] == "total" else "circ_mv"
    daily = valuation.copy()
    if not {"ts_code", "trade_date", field}.issubset(daily.columns):
        daily = pd.DataFrame(columns=["ts_code", "trade_date", field])

    def normalize_code(value: Any) -> str | None:
        try:
            return biaozhunhua_gupiao_daima(str(value))
        except ValueError:
            return None

    daily["ts_code"] = daily["ts_code"].map(normalize_code)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    # Tushare daily_basic 的 total_mv/circ_mv 单位为万元，公开证据统一为元。
    daily["value_yuan"] = pd.to_numeric(daily[field], errors="coerce") * 10_000.0
    grouped = {code: frame for code, frame in daily.groupby("ts_code", sort=False)}
    checks: list[dict[str, Any]] = []
    accepted: list[int] = []
    work = data.copy().reset_index(drop=True)
    for index, row in work.iterrows():
        code = str(row["ts_code"])
        rows = grouped.get(code)
        reason = None
        actual = None
        if rows is None:
            reason = source_error or "未取得该股票可核验的日终市值；快照市值不能代替日终证据"
        elif not rows["trade_date"].eq(target).all():
            reason = "市值来源日期缺失或与分析日不一致"
        elif not rows["value_yuan"].map(lambda value: pd.notna(value) and math.isfinite(value) and value > 0).all():
            reason = "市值缺失、非有限数或非正数"
        elif rows["value_yuan"].nunique() != 1:
            reason = "同一股票同日日终市值存在冲突"
        else:
            actual = float(rows.iloc[0]["value_yuan"])
        met = actual is not None and (
            actual <= condition["maximum_yuan"] if condition["operator"] == "le"
            else actual < condition["maximum_yuan"]
        )
        check = {
            "ts_code": code, "name": str(row.get("name") or ""),
            "status": "unavailable" if reason else "met" if met else "unmet",
            "condition": condition, "actual_yuan": actual, "unit": "元",
            "source": "tushare_daily_basic", "as_of": target.strftime("%Y-%m-%d"),
            "source_dates": sorted(rows["trade_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist()) if rows is not None else [],
            "fetched_at": fetched_at,
            "reason": reason or ("符合市值条件" if met else "超过上限或不满足上限边界"),
        }
        checks.append(check)
        if met:
            accepted.append(index)
    work["market_cap_check"] = checks
    unavailable = sum(item["status"] == "unavailable" for item in checks)
    unmet = sum(item["status"] == "unmet" for item in checks)
    summary = {
        "status": "partial" if unavailable else "ok", "condition": condition,
        "source": "tushare_daily_basic", "as_of": target.strftime("%Y-%m-%d"),
        "fetched_at": fetched_at, "source_error": source_error,
        "input_count": len(work), "verified_count": len(work) - unavailable,
        "matched_count": len(accepted), "unmet_count": unmet, "unavailable_count": unavailable,
        "unmet_examples": [item for item in checks if item["status"] == "unmet"][:20],
        "unavailable_examples": [item for item in checks if item["status"] == "unavailable"][:20],
        "application_stage": "基础检查后、流动性抽样和历史行情请求前",
        "missing_evidence": "未核验市值的股票不进入候选；不能认定这些股票超过市值上限",
    }
    return work.loc[accepted].reset_index(drop=True), summary
