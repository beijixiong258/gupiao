"""用本次已取得的财务字段补充解释，不发起请求、不产生分数或买卖门槛。"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any


_CASHFLOW_APPLICABILITY = "金融类公司的现金流结构特殊，需结合业务理解；本项不使用跨行业的好坏阈值。"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def _date(value: Any) -> date | None:
    text = value.isoformat() if isinstance(value, (datetime, date)) else _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None


def _sign(value: float | None) -> str | None:
    if value is None:
        return None
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _block(key: str, label: str, status: str, summary: str, metrics: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "summary": summary,
        "metrics": metrics,
        "missing_reason": "；".join(missing) if missing else None,
    }


def _report_timeliness(
    financials: dict[str, Any],
    *,
    source: str | None,
    as_of_raw: Any,
) -> tuple[dict[str, Any], list[str]]:
    as_of = _date(as_of_raw)
    dates = {
        "report_date": _date(financials.get("report_date")),
        "announcement_date": _date(financials.get("announcement_date")),
        "known_as_of": _date(financials.get("known_as_of")),
    }
    labels = {"report_date": "报告期", "announcement_date": "公告日", "known_as_of": "已知数据截止日"}
    missing = [f"{labels[key]}缺失或日期无效" for key, value in dates.items() if value is None]
    if as_of is None:
        missing.append("分析日缺失或日期无效，无法核对财报时点")
    if source is None:
        missing.append("财务数据来源未记录，无法确认核验来源")
    announcement_note = _text(financials.get("announcement_date_status"))
    if dates["announcement_date"] is None and announcement_note:
        missing.append(announcement_note)
    conflicts = [
        f"{labels[key]} {value.isoformat()} 晚于分析日 {as_of.isoformat()}，不可用于当前诊断"
        for key, value in dates.items()
        if as_of is not None and value is not None and value > as_of
    ]
    report = dates["report_date"]
    announcement = dates["announcement_date"]
    if report is not None and announcement is not None and announcement < report:
        conflicts.append("公告日早于报告期，日期关系异常，不能核验本次财报时点")
    known_as_of = dates["known_as_of"]
    if known_as_of is not None and announcement is not None and announcement > known_as_of:
        conflicts.append("公告日晚于已知数据截止日，不能作为该截止日已知财报使用")
    verified = not missing and not conflicts
    metrics = {
        **{key: value.isoformat() if value is not None else None for key, value in dates.items()},
        "analysis_date": as_of.isoformat() if as_of is not None else None,
        "source": source,
        "calendar_days_since_report": (as_of - report).days if as_of and report else None,
        "calendar_days_since_announcement": (as_of - announcement).days if as_of and announcement else None,
        "announcement_lag_calendar_days": (announcement - report).days if announcement and report else None,
        "point_in_time_verified": verified,
        "date_conflicts": conflicts,
    }
    facts: list[str] = []
    for key, value in dates.items():
        if value is not None:
            facts.append(f"{labels[key]} {value.isoformat()}")
    if source:
        facts.append(f"来源 {source}")
    if as_of and report and report <= as_of:
        facts.append(f"分析日距报告期末 {(as_of - report).days} 个自然日")
    if as_of and announcement and announcement <= as_of:
        facts.append(f"距公告日 {(as_of - announcement).days} 个自然日")
    if conflicts:
        facts.extend(conflicts)
    elif verified:
        facts.append("记录中的报告期和公告日均不晚于分析日，财报时点核对通过")
    else:
        facts.append("财报时点信息不完整，不能标为已核验")
    status = "unavailable" if conflicts or not any(dates.values()) else "ok" if verified else "partial"
    return _block("report_timeliness", "财报时点与来源", status, "；".join(facts) + "。", metrics, [*missing, *conflicts]), conflicts


def _growth(financials: dict[str, Any], date_conflicts: list[str]) -> dict[str, Any]:
    revenue = _number(financials.get("revenue_yoy_pct"))
    profit = _number(financials.get("net_profit_yoy_pct"))
    metrics = {
        "revenue_yoy_pct": revenue,
        "net_profit_yoy_pct": profit,
        "revenue_growth_sign": _sign(revenue),
        "net_profit_growth_sign": _sign(profit),
    }
    missing = [label for value, label in ((revenue, "营收同比缺失或非有限数"), (profit, "净利润同比缺失或非有限数")) if value is None]
    if date_conflicts:
        return _block("financial_growth", "营收与利润增长", "unavailable", "财报日期核对未通过，所列增长字段不可用于当前诊断。", metrics, [*missing, *date_conflicts])
    facts: list[str] = []
    if revenue is not None:
        facts.append(f"营业收入同比 {revenue:.6g}%")
    if profit is not None:
        facts.append(f"净利润同比 {profit:.6g}%")
    if revenue is not None and profit is not None:
        revenue_label = "营收增长" if revenue > 0 else "营收下降" if revenue < 0 else "营收同比持平"
        profit_label = "利润增长" if profit > 0 else "利润下降" if profit < 0 else "利润同比持平"
        facts.append(f"本期表现为{revenue_label}、{profit_label}")
        facts.append("单期同比不能证明趋势改善或增长持续")
    else:
        facts.append("缺少完整的营收与利润同比组合，无法核对两者是否同步")
    status = "ok" if not missing else "unavailable" if revenue is None and profit is None else "partial"
    return _block("financial_growth", "营收与利润增长", status, "；".join(facts) + "。", metrics, missing)


def _cashflow(financials: dict[str, Any], industry: str | None, date_conflicts: list[str]) -> dict[str, Any]:
    margin = _number(financials.get("net_margin_pct"))
    cashflow = _number(financials.get("operating_cashflow_to_revenue_pct"))
    margin_sign = _sign(margin)
    cashflow_sign = _sign(cashflow)
    metrics = {
        "net_margin_pct": margin,
        "operating_cashflow_to_revenue_raw": cashflow,
        "cashflow_unit_status": "unverified",
        "cashflow_evidence_basis": "sign_only",
        "net_margin_sign": margin_sign,
        "operating_cashflow_sign": cashflow_sign,
        "industry": industry,
        "applicability_note": _CASHFLOW_APPLICABILITY,
    }
    missing = [label for value, label in ((margin, "净利率缺失或非有限数"), (cashflow, "经营现金流指标缺失或非有限数")) if value is None]
    if date_conflicts:
        return _block("cashflow_quality", "盈利与经营现金流核对", "unavailable", "财报日期核对未通过，所列盈利和现金流字段不可用于当前诊断。" + _CASHFLOW_APPLICABILITY, metrics, [*missing, *date_conflicts])
    facts: list[str] = []
    if margin is not None:
        facts.append(f"本期净利率 {margin:.6g}%")
    if cashflow is not None:
        facts.append("经营现金流指标" + {"positive": "为正", "negative": "为负", "zero": "为零"}[cashflow_sign])
    if margin_sign is not None and cashflow_sign is not None:
        if margin_sign == "positive" and cashflow_sign == "negative":
            facts.append("净利率为正但现金流指标为负，两项符号背离，需进一步核对回款和支出")
        elif margin_sign == "negative" and cashflow_sign == "positive":
            facts.append("净利率为负而现金流指标为正，经营现金流符号不能代替盈利判断")
        elif margin_sign == "zero" or cashflow_sign == "zero":
            facts.append("为零的指标按实际零值展示，不代表缺失，也不代表已经改善")
        else:
            facts.append("两项符号一致，不能单凭这一组合认定经营质量")
    else:
        facts.append("字段不齐，无法核对盈利与经营现金流符号")
    facts.append("现金流字段单位尚未核验，仅核对正负和零")
    status = "ok" if not missing else "unavailable" if margin is None and cashflow is None else "partial"
    return _block("cashflow_quality", "盈利与经营现金流核对", status, "；".join(facts) + "。" + _CASHFLOW_APPLICABILITY, metrics, missing)


def goujian_caiwu_zhenduan(fundamental: dict, *, as_of_date: str | None = None) -> list[dict]:
    """解释已有财务响应；兼容直接结果、合成基本面和字段不足时的深度结果。"""
    root = _mapping(fundamental)
    deep = _mapping(root.get("deep_analysis"))
    financials = _mapping(root.get("financials")) or _mapping(deep.get("financials"))
    profile = _mapping(root.get("profile")) or _mapping(deep.get("profile"))
    source = _text(_mapping(root.get("sources")).get("financials")) or _text(_mapping(deep.get("sources")).get("financials"))
    if source and source.lower() in {"unknown", "unavailable", "none"}:
        source = None
    as_of_raw = as_of_date if as_of_date is not None else root.get("as_of") or deep.get("as_of") or financials.get("known_as_of")
    timeliness, date_conflicts = _report_timeliness(financials, source=source, as_of_raw=as_of_raw)
    return [
        _growth(financials, date_conflicts),
        _cashflow(financials, _text(profile.get("industry")), date_conflicts),
        timeliness,
    ]


__all__ = ["goujian_caiwu_zhenduan"]
