"""单股分析：尽可能保留已取得的证据，独立披露缺口，不作自动买入决策。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.dangu_lianghua import yunxing_dangu_tongyi_lianghua
from src.ashare.dangu_zhaiyao import goujian_dangu_zhaiyao
from src.ashare.gupiao_yanjiu import zongjie_jishu
from src.ashare.shichang_shuju import FenxiShujuShangxiawen
from src.ashare.xuangu_guize import goujian_kejiaoyixing_zhaiyao


DANGU_ANALYSIS_TYPE = "single_stock_analysis"
DANGU_TOOL_CONTRACT_VERSION = 9


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat(sep=" ")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _texts(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return list(dict.fromkeys(" ".join(str(value).split()) for value in values if value))


def _gap(code: str, component: str, reason: Any) -> dict[str, str]:
    return {
        "code": code, "component": component, "reason": " ".join(str(reason).split()),
        "impact": "该部分信息不足，不能据此断言有利或不利；其他已取得证据仍然有效",
        "reassessment_condition": f"补齐并核验{component}所需信息后重新分析",
    }


def _identity(code: str, *profiles: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"ts_code": code}
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for key in ("name", "industry", "market", "list_date"):
            value = profile.get(key)
            if value is not None and not pd.isna(value) and str(value).strip() and not result.get(key):
                result[key] = value
    return _json_safe(result)


def _failure(query: str, *, status: str, outcome: str, code: str, stage: str, error: Any, stock=None, provenance=None) -> dict[str, Any]:
    return {
        "status": status, "outcome": outcome, "analysis_type": DANGU_ANALYSIS_TYPE,
        "tool_contract_version": DANGU_TOOL_CONTRACT_VERSION, "query": query,
        "error_code": code, "stage": stage, "error": " ".join(str(error).split())[:500],
        "stock": stock, "selected_stock": stock, "recommendation_available": False,
        "primary": None, "alternatives": [], "data_provenance": provenance or {},
        "retryable": status == "unavailable",
        "next_action": "补充明确的股票身份后重试" if status == "clarification_required" else "待数据恢复或错误修复后重新分析",
        "research_scope": "公开数据单股分析，不连接证券账户或执行交易",
    }


def _fundamental_block(raw: dict[str, Any], as_of: str) -> dict[str, Any]:
    financials, valuation = raw.get("financials") or {}, raw.get("valuation") or {}
    if financials and valuation:
        status = "ok"
    elif financials or valuation:
        status = "partial"
    else:
        status = "unavailable"
    return {
        **raw, "status": status, "as_of": as_of,
        "profile": raw.get("profile") or {}, "financials": financials, "valuation": valuation,
        "sources": raw.get("sources") or {}, "errors": _texts(raw.get("errors")),
        "warnings": _texts(raw.get("warnings")),
        "available_fields": {"profile": bool(raw.get("profile")), "financials": bool(financials), "valuation": bool(valuation)},
        "interpretation": "逐项保留本次已取得的估值与财务数据；缺失字段不代表基本面差",
    }


def _collect_gaps(technical, fundamentals, snapshot, unified) -> list[dict[str, str]]:
    gaps = list(unified.get("evidence_gaps") or [])
    for warning in technical.get("indicator_warnings") or []:
        gaps.append(_gap("technical_indicator_missing", "技术指标", warning))
    if technical.get("status") in {"partial", "unavailable", "insufficient_data", "error"}:
        reason = technical.get("reason") or technical.get("error") or (technical.get("macd_structure") or {}).get("reason") or "部分技术结构或长窗口指标尚不可用"
        gaps.append(_gap("technical_partial", "技术结构", reason))
    if fundamentals.get("status") != "ok":
        gaps.append(_gap("fundamental_partial", "基本面", "；".join(fundamentals.get("errors") or []) or "财务或估值字段尚未完整取得"))
    if snapshot.get("status") != "ok":
        gaps.append(_gap("snapshot_unavailable", "当前行情", snapshot.get("error") or "实时行情尚未取得"))
    factor = unified.get("factor_analysis") or {}
    if factor.get("status") in {"unavailable", "error"}:
        gaps.append(_gap("factor_unavailable", "八组日线指标", factor.get("reason") or "日线指标暂未形成"))
    for key, group in (factor.get("groups") or {}).items():
        if isinstance(group, dict) and group.get("missing_fields"):
            labels = group.get("metric_definitions") or group.get("field_metadata") or {}
            missing = [str((labels.get(field) or {}).get("label") or field) for field in group["missing_fields"]]
            gaps.append(_gap("factor_group_" + key, str(group.get("label") or key), "缺少：" + "、".join(missing)))
    for block in (unified.get("supplemental_diagnostics") or {}).get("blocks") or []:
        if block.get("status") != "ok":
            gaps.append(_gap("supplemental_" + str(block.get("key")), str(block.get("label") or "补充指标"), block.get("missing_reason") or block.get("summary")))
    seen = set()
    result = []
    for gap in gaps:
        identity = (gap.get("code"), gap.get("reason"))
        if identity not in seen:
            seen.add(identity)
            result.append(gap)
    return result


def fenxi_dangu(*, gupiao: str, config: dict[str, Any], context: FenxiShujuShangxiawen) -> dict[str, Any]:
    query = " ".join(str(gupiao or "").split())
    if not query:
        return _failure(query, status="clarification_required", outcome="clarification_required", code="stock_query_missing", stage="request_validation", error="请提供股票名称或代码")
    try:
        code, resolved_profile, resolution_warnings = context.jiexi_gupiao(query)
    except ValueError as exc:
        return _failure(query, status="clarification_required", outcome="clarification_required", code="stock_query_ambiguous", stage="stock_resolution", error=exc)
    except RuntimeError as exc:
        message = str(exc).lower()
        source_failed = any(marker in message for marker in ("超时", "timeout", "connection", "连接", "限频", "频率", "权限", "permission", "不可用", "失败"))
        return _failure(
            query, status="unavailable" if source_failed else "clarification_required",
            outcome="data_unavailable" if source_failed else "clarification_required",
            code="stock_resolution_unavailable" if source_failed else "stock_not_found",
            stage="stock_resolution", error=exc,
        )
    stock = _identity(code, resolved_profile or {})
    try:
        clock = context.shichang_shizhong()
        requested_date = pd.Timestamp(context.zuixin_wanzheng_jiaoyiri()).normalize()
        if pd.isna(requested_date):
            raise RuntimeError("最近完整交易日期无效")
    except Exception as exc:
        return _failure(query, status="unavailable", outcome="data_unavailable", code="calendar_unavailable", stage="calendar", error=exc, stock=stock)
    start_date = (requested_date - pd.Timedelta(days=int((config.get("dangu") or {}).get("history_calendar_days", 1440)))).strftime("%Y%m%d")
    end_date = requested_date.strftime("%Y%m%d")
    try:
        histories, history_meta = context.piliang_lishi([code], start_date=start_date, end_date=end_date, minimum_rows=1)
    except Exception as exc:
        return _failure(query, status="unavailable", outcome="data_unavailable", code="single_stock_history_request_failed", stage="history_data", error=exc, stock=stock)
    history = histories.get(code)
    if history is None or history.empty:
        return _failure(query, status="unavailable", outcome="data_unavailable", code="single_stock_history_unavailable", stage="history_data", error=history_meta.get("error") or "没有取得目标股票的可用完整日线", stock=stock, provenance={"history": history_meta})
    if "trade_date" not in history:
        return _failure(query, status="unavailable", outcome="data_unavailable", code="single_stock_history_invalid", stage="history_data", error="目标日线缺少交易日期", stock=stock, provenance={"history": history_meta})
    dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        return _failure(query, status="unavailable", outcome="data_unavailable", code="single_stock_history_invalid", stage="history_data", error="目标日线日期无效或重复，不能任意跳过坏行分析", stock=stock, provenance={"history": history_meta})
    history = history.assign(trade_date=dates).loc[dates.le(requested_date)].sort_values("trade_date").reset_index(drop=True)
    if history.empty:
        return _failure(query, status="unavailable", outcome="data_unavailable", code="single_stock_history_unavailable", stage="history_data", error="分析日及之前没有可用日线", stock=stock, provenance={"history": history_meta})
    as_of = history.iloc[-1]["trade_date"].strftime("%Y-%m-%d")
    extra_gaps: list[dict[str, Any]] = []
    if as_of != requested_date.strftime("%Y-%m-%d"):
        extra_gaps.append(_gap("latest_daily_missing", "最新完整日线", f"实际日线停留在 {as_of}，没有取得请求日期 {requested_date:%Y-%m-%d} 的日线"))
    try:
        technical = zongjie_jishu(history, macd_structure_config=(config.get("fenxi") or {}).get("macd_structure"))
    except RuntimeError as exc:
        technical = {"status": "partial", "reason": str(exc), "trade_date": as_of, "close": history.iloc[-1].get("close")}
    except Exception as exc:
        technical = {"status": "error", "outcome": "program_error", "error": str(exc), "trade_date": as_of}
    try:
        snapshot = context.dangu_kuaizhao(code)
    except Exception as exc:
        snapshot = {"status": "unavailable", "source": "remote_live_snapshot", "error": str(exc)}
    allow_current = bool(clock.get("session_status") == "post_close" and pd.Timestamp(context.reference.date()).normalize() == pd.Timestamp(as_of))
    try:
        fundamentals = _fundamental_block(context.jibenmian(code, trade_date=as_of, allow_current_snapshot=allow_current), as_of)
    except Exception as exc:
        fundamentals = _fundamental_block({"errors": [str(exc)]}, as_of)
    stock = _identity(code, resolved_profile or {}, fundamentals.get("profile") or {}, snapshot)
    name = str(stock.get("name") or code)
    realtime_required = clock.get("session_status") in {"opening_auction", "trading", "midday_break", "close_pending"}
    try:
        tradability = goujian_kejiaoyixing_zhaiyao(
            code=code, name=name, snapshot=snapshot, history=history,
            minimum_amount=float((config.get("fenxi") or {}).get("min_amount_yuan", 50_000_000)),
            realtime_required=realtime_required,
            reference_time=context.reference,
        )
    except Exception as exc:
        tradability = {"status": "unavailable", "cautions": [str(exc)]}
        extra_gaps.append(_gap("tradability_unavailable", "成交条件", exc))
    if realtime_required and not tradability.get("current_quote_verified"):
        extra_gaps.append(_gap("current_quote_unverified", "当前行情时点与成交条件",
                               tradability.get("current_quote_reason") or "实时行情未通过核验"))
    unified = yunxing_dangu_tongyi_lianghua(
        code=code, name=name, industry=str(stock.get("industry") or ""), history=history,
        analysis_date=pd.Timestamp(as_of), snapshot=snapshot, clock=clock,
        technical=technical, fundamentals=fundamentals, tradability=tradability,
        config=config, context=context,
    )
    stock = _identity(code, stock, unified.get("stock_identity") or {})
    name = str(stock.get("name") or name)
    fundamentals = unified.get("fundamental_analysis") or fundamentals
    gaps = [*extra_gaps, *_collect_gaps(technical, fundamentals, snapshot, unified)]
    risks = _texts([
        *((technical.get("macd_structure") or {}).get("risk_warnings") or []),
        *(tradability.get("hard_blocks") or []), *(tradability.get("cautions") or []),
    ])
    if "ST" in name.upper():
        risks.append("股票简称含ST风险标记，需结合风险警示说明理解；本次仍保留完整分析")
    if "退" in name:
        risks.append("股票简称含退市风险标记，需核对退市安排")
    summary = goujian_dangu_zhaiyao(
        technical=technical, fundamentals=fundamentals, risks=risks, evidence_gaps=gaps, as_of=as_of,
        factor_analysis=unified.get("factor_analysis"), pattern=unified.get("limit_up_pullback_pattern"),
        late=unified.get("late_session_analysis"), supplemental=unified.get("supplemental_diagnostics"),
    )
    status = "error" if technical.get("status") == "error" else "partial" if gaps else "ok"
    confirmation = "intraday_provisional" if clock.get("session_status") in {"opening_auction", "trading", "midday_break"} else "close_pending" if clock.get("session_status") == "close_pending" else "completed_daily_close"
    generated_at = context.reference.strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "status": status, "outcome": "program_error" if status == "error" else "information_partial" if status == "partial" else "analysis_success",
        "analysis_type": DANGU_ANALYSIS_TYPE, "tool_contract_version": DANGU_TOOL_CONTRACT_VERSION,
        "query": query, "stock": stock, "selected_stock": stock,
        "as_of": as_of, "generated_at": generated_at, "market_clock": clock,
        "result_confirmation": confirmation, "recommendation_available": False,
        "primary": None, "alternatives": [],
        "diagnosis_summary": summary, "plain_language_summary": summary["summary"],
        "reassessment_conditions": summary["reassessment_conditions"],
        "technical_summary": technical, "daily_factor_analysis": unified.get("factor_analysis"),
        "fundamental_analysis": fundamentals,
        "limit_up_pullback_pattern": unified.get("limit_up_pullback_pattern"),
        "late_session_analysis": unified.get("late_session_analysis"),
        "supplemental_diagnostics": unified.get("supplemental_diagnostics"),
        "realtime_snapshot": snapshot, "tradability": tradability, "evidence_gaps": gaps,
        "risks": risks, "warnings": _texts([*(resolution_warnings or []), *(history_meta.get("warnings") or []), *(fundamentals.get("warnings") or [])]),
        "diagnosis_validity": {
            "daily_data_as_of": as_of, "generated_at": generated_at,
            "session_status": clock.get("session_status"), "result_confirmation": confirmation,
            "realtime_required": realtime_required,
            "realtime_status": tradability.get("current_quote_status") or snapshot.get("status"),
            "explanation": f"日线证据截至 {as_of}，生成于 {generated_at}。盘中行情会变化，缺失来源已单独列明；新日线或财报发布后需重新分析。",
            "reassess_when": summary["reassessment_conditions"],
        },
        "data_analysis": {
            "latest_daily_bar": history.iloc[-1].to_dict(), "resolved_profile": resolved_profile,
            "comparison_profile": unified.get("comparison_profile"),
            "history_summary": {"rows": len(history), "first_trade_date": history.iloc[0]["trade_date"].strftime("%Y-%m-%d"), "last_trade_date": as_of, "requested_start_date": start_date, "requested_end_date": end_date},
        },
        "data_provenance": {
            "stock_resolution": {"persistence": "none", "warnings": resolution_warnings},
            "history": history_meta,
            "realtime_snapshot": {key: snapshot.get(key) for key in ("status", "source", "captured_at", "provider_trade_date", "timeliness", "error") if key in snapshot},
            "fundamentals": {"sources": fundamentals.get("sources") or {}, "as_of": as_of, "persistence": "none"},
            **(unified.get("data_provenance") or {}),
        },
        "analysis_stage": {"status": status, "scope": "股票已有行情、指标、财务、形态、尾盘及来源证据的完整呈现"},
        "research_scope": "公开数据单股分析，不连接证券账户或执行交易",
    }
    if status == "error":
        result.update(error_code="single_stock_technical_error", error=technical.get("error") or technical.get("reason") or "技术分析程序错误", stage="technical_analysis")
    return _json_safe(result)


__all__ = ["DANGU_ANALYSIS_TYPE", "DANGU_TOOL_CONTRACT_VERSION", "fenxi_dangu"]
