"""单只 A 股量化分析编排。

单股路径保留独立的证券身份和数据边界，但复用统一选股的八组因子、基本面、
形态、尾盘、风险扣分、综合评分与推荐门槛，最终明确回答是否建议买入。
它仍不产生预测资格。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.ashare.dangu_lianghua import yunxing_dangu_tongyi_lianghua
from src.ashare.gupiao_yanjiu import (
    huoqu_jibenmian,
    jiexi_gupiao,
    zongjie_jishu,
)
from src.ashare.shichang_shuju import FenxiShujuShangxiawen
from src.ashare.xuangu_guize import (
    goujian_kejiaoyixing_zhaiyao,
    zhuan_you_xian_shuzhi,
)


DANGU_ANALYSIS_TYPE = "single_stock_analysis"
DANGU_TOOL_CONTRACT_VERSION = 9


def _json_safe(value: Any) -> Any:
    """递归转换第三方数据，确保结果不携带 NaN、Timestamp 或 numpy 标量。"""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _dedupe(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text not in result:
            result.append(text)
    return result


def _identity(
    code: str,
    resolved_profile: dict[str, Any] | None,
    fundamental_profile: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """按远端来源优先级合并股票身份，绝不从用户文字猜行业。"""

    result: dict[str, Any] = {"ts_code": code}

    def has_value(value: Any) -> bool:
        if value is None or value is pd.NA:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        try:
            return not bool(pd.isna(value))
        except (TypeError, ValueError):
            return True

    for profile in (resolved_profile, fundamental_profile, snapshot):
        if not isinstance(profile, dict):
            continue
        for key in ("name", "industry", "market", "list_date"):
            value = profile.get(key)
            if has_value(value) and not has_value(result.get(key)):
                result[key] = value
    return _json_safe(result)


def _result_confirmation(clock: dict[str, Any]) -> str:
    status = str(clock.get("session_status") or "")
    if status in {"trading", "midday_break", "opening_auction"}:
        return "intraday_provisional"
    if status == "close_pending":
        return "close_pending"
    return "completed_daily_close"


def _source_error_result(
    *,
    query: str,
    status: str,
    outcome: str,
    error_code: str,
    error: str,
    next_action: str,
    stage: str,
    stock: dict[str, Any] | None = None,
    data_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "outcome": outcome,
        "tool_contract_version": DANGU_TOOL_CONTRACT_VERSION,
        "analysis_type": DANGU_ANALYSIS_TYPE,
        "query": query,
        "error_code": error_code,
        "stage": stage,
        "source": "remote_market_data",
        "retryable": status == "unavailable",
        "error": " ".join(str(error).split())[:320],
        "next_action": next_action,
        "stock": _json_safe(stock),
        "selected_stock": _json_safe(stock),
        "recommendation_available": False,
        "primary": None,
        "alternatives": [],
        "data_provenance": _json_safe(data_provenance or {}),
        "research_scope": "公开数据单股量化研究，不连接证券账户、不提交委托、不自动交易",
    }


def _is_resolution_unavailable(error: str) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "超时",
            "timeout",
            "connection",
            "连接",
            "限频",
            "频率",
            "权限",
            "permission",
            "不可用",
            "失败",
        )
    )


def _fundamental_block(raw: dict[str, Any], *, as_of: str) -> dict[str, Any]:
    profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else {}
    valuation = raw.get("valuation") if isinstance(raw.get("valuation"), dict) else {}
    financials = raw.get("financials") if isinstance(raw.get("financials"), dict) else {}
    errors = _dedupe(raw.get("errors"))
    warnings = _dedupe(raw.get("warnings"))
    if financials or valuation:
        status = "ok"
        outcome = "analysis_success"
    else:
        status = "unavailable"
        outcome = "data_unavailable"
    available_fields = {
        "profile": bool(profile),
        "valuation": bool(valuation),
        "financials": bool(financials),
    }
    return {
        "status": status,
        "outcome": outcome,
        "as_of": as_of,
        "profile": _json_safe(profile),
        "valuation": _json_safe(valuation),
        "financials": _json_safe(financials),
        "available_fields": available_fields,
        "sources": _json_safe(raw.get("sources") or {}),
        "data_quality": _json_safe(raw.get("data_quality") or {}),
        "warnings": warnings,
        "errors": errors,
        "interpretation": (
            "基本面证据来自截至分析日已知的数据；缺失字段会降低完整度。"
            if status == "ok"
            else "基本面数据当前不可用，不能据此断言基本面好坏。"
        ),
    }


def _technical_risks(technical: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    structure = technical.get("macd_structure") if isinstance(technical, dict) else {}
    if isinstance(structure, dict):
        risks.extend(_dedupe(structure.get("risk_warnings")))
        risks.extend(_dedupe(structure.get("counter_evidence")))
        classification = structure.get("structure_classification")
        if isinstance(classification, dict) and classification.get("code") in {
            "momentum_weakening",
            "top_risk",
        }:
            label = str(classification.get("label") or "结构动能偏弱")
            risks.append(label)
    volatility = zhuan_you_xian_shuzhi(technical.get("annualized_volatility_20"))
    if volatility is not None and volatility > 0.55:
        risks.append(f"20 日年化波动率约 {volatility * 100:.1f}%，短线波动风险较高")
    drawdown = zhuan_you_xian_shuzhi(technical.get("drawdown_from_20d_high"))
    if drawdown is not None and drawdown < -0.1:
        risks.append(f"较近 20 日高点回撤约 {abs(drawdown) * 100:.1f}%")
    return list(dict.fromkeys(risks))


def fenxi_dangu(
    *,
    gupiao: str,
    config: dict[str, Any],
    context: FenxiShujuShangxiawen,
) -> dict[str, Any]:
    """分析一只股票；所有行情只在 ``context`` 生命周期内复用。"""

    query = " ".join(str(gupiao or "").split())
    if not query:
        return _source_error_result(
            query=query,
            status="clarification_required",
            outcome="clarification_required",
            error_code="stock_query_missing",
            error="请提供要分析的股票名称或 6 位股票代码",
            next_action="补充完整股票名称或代码",
            stage="request_validation",
        )

    try:
        code, resolved_profile, resolution_warnings = jiexi_gupiao(query, source="auto")
    except ValueError as exc:
        return _source_error_result(
            query=query,
            status="clarification_required",
            outcome="clarification_required",
            error_code="stock_query_ambiguous",
            error=str(exc),
            next_action="请使用完整股票名称或 6 位股票代码",
            stage="stock_resolution",
        )
    except RuntimeError as exc:
        error = str(exc)
        if _is_resolution_unavailable(error):
            return _source_error_result(
                query=query,
                status="unavailable",
                outcome="data_unavailable",
                error_code="stock_resolution_unavailable",
                error=error,
                next_action="稍后重试；程序不会使用旧本地股票资料代替当前结果",
                stage="stock_resolution",
            )
        return _source_error_result(
            query=query,
            status="clarification_required",
            outcome="clarification_required",
            error_code="stock_not_found",
            error=error,
            next_action="请检查股票名称或代码后重试",
            stage="stock_resolution",
        )

    settings = config.get("dangu") if isinstance(config.get("dangu"), dict) else {}
    try:
        clock = context.shichang_shizhong()
        calendar = context.jiaoyi_rili()
        requested_date = context.zuixin_wanzheng_jiaoyiri()
    except Exception as exc:
        return _source_error_result(
            query=query,
            status="unavailable",
            outcome="data_unavailable",
            error_code="calendar_unavailable",
            error=f"无法确认最近完整交易日：{exc}",
            next_action="稍后重试；没有可靠交易日历时不继续技术分析",
            stage="calendar",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
        )

    history_days = int(settings.get("history_calendar_days", 1440))
    minimum_rows = int(settings.get("minimum_history_rows", 180))
    start_date = (requested_date - pd.Timedelta(days=history_days)).strftime("%Y%m%d")
    end_date = requested_date.strftime("%Y%m%d")
    try:
        histories, history_meta = context.piliang_lishi(
            [code],
            start_date=start_date,
            end_date=end_date,
            minimum_rows=minimum_rows,
        )
    except Exception as exc:
        return _source_error_result(
            query=query,
            status="unavailable",
            outcome="data_unavailable",
            error_code="single_stock_history_request_failed",
            error=f"单股历史行情获取失败：{exc}",
            next_action="稍后重新获取远端完整日线",
            stage="history_data",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta if "history_meta" in locals() else {}},
        )

    history = histories.get(code)
    if history is None or history.empty:
        incomplete = history_meta.get("incomplete_examples") or []
        known_incomplete = any(str(item.get("ts_code")) == code for item in incomplete if isinstance(item, dict))
        if known_incomplete:
            return _source_error_result(
                query=query,
                status="insufficient_data",
                outcome="information_insufficient",
                error_code="single_stock_history_insufficient",
                error=f"{code} 的完整日线少于 {minimum_rows} 个交易日或覆盖率不足",
                next_action="当前只能说明信息不足，不能把短历史当成可靠结构结论",
                stage="history_data",
                stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
                data_provenance={"history": history_meta},
            )
        return _source_error_result(
            query=query,
            status="unavailable",
            outcome="data_unavailable",
            error_code="single_stock_history_unavailable",
            error=str(history_meta.get("error") or "没有取得该股票的完整远端日线"),
            next_action="稍后重新获取远端完整日线；不使用本地旧行情代替",
            stage="history_data",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta},
        )

    history = history.copy()
    history["trade_date"] = pd.to_datetime(history.get("trade_date"), errors="coerce").dt.normalize()
    history = (
        history.dropna(subset=["trade_date"])
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .reset_index(drop=True)
    )
    history = history[history["trade_date"] <= requested_date.normalize()].reset_index(drop=True)
    if len(history) < minimum_rows:
        return _source_error_result(
            query=query,
            status="insufficient_data",
            outcome="information_insufficient",
            error_code="single_stock_history_insufficient",
            error=f"有效完整日线只有 {len(history)} 个交易日，低于要求的 {minimum_rows} 个",
            next_action="当前只能说明信息不足，不能据此给出稳定结构判断",
            stage="history_data",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta},
        )

    as_of = pd.Timestamp(history.iloc[-1]["trade_date"]).strftime("%Y-%m-%d")
    if as_of != requested_date.strftime("%Y-%m-%d"):
        history_meta = {
            **history_meta,
            "analysis_date_adjustment": f"交易日线实际截至 {as_of}，未沿用缺失的请求日期 {requested_date:%Y-%m-%d}",
        }

    try:
        technical = zongjie_jishu(
            history,
            macd_structure_config=(config.get("fenxi") or {}).get("macd_structure"),
        )
    except RuntimeError as exc:
        return _source_error_result(
            query=query,
            status="insufficient_data",
            outcome="information_insufficient",
            error_code="single_stock_technical_insufficient",
            error=f"技术指标有效历史不足：{exc}",
            next_action="补齐更长的完整日线后再分析",
            stage="technical_analysis",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta},
        )
    except Exception as exc:
        return _source_error_result(
            query=query,
            status="error",
            outcome="program_error",
            error_code="single_stock_technical_error",
            error=f"单股技术分析程序错误：{exc}",
            next_action="请记录运行编号并检查程序日志",
            stage="technical_analysis",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta},
        )

    technical_status = str(technical.get("status") or "")
    if technical_status == "error":
        return _source_error_result(
            query=query,
            status="error",
            outcome="program_error",
            error_code="single_stock_technical_error",
            error=str(technical.get("error") or technical.get("macd_structure", {}).get("reason") or "技术结构研判发生程序错误"),
            next_action="请记录运行编号并检查程序日志",
            stage="technical_analysis",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta},
        )
    if technical_status in {"unavailable", "insufficient_data"}:
        return _source_error_result(
            query=query,
            status="insufficient_data",
            outcome="information_insufficient",
            error_code="single_stock_technical_insufficient",
            error=str(technical.get("reason") or "最新日线缺少完整技术指标，不能沿用旧状态"),
            next_action="补齐完整日线或等待数据源确认后再分析",
            stage="technical_analysis",
            stock={"ts_code": code, "name": resolved_profile.get("name") if isinstance(resolved_profile, dict) else None},
            data_provenance={"history": history_meta},
        )

    # 单股请求至多读取一次实时全市场快照；它只是当前可交易性参考，不能回填历史收盘。
    try:
        snapshot = context.dangu_kuaizhao(code)
    except Exception as exc:
        snapshot = {
            "status": "unavailable",
            "error": f"实时快照请求失败：{exc}",
            "source": "remote_live_snapshot",
        }
    name_hint = str((resolved_profile or {}).get("name") or snapshot.get("name") or code)
    try:
        tradability = goujian_kejiaoyixing_zhaiyao(
            code=code,
            name=name_hint,
            snapshot=snapshot,
            history=history,
            minimum_amount=float((config.get("fenxi") or {}).get("min_amount_yuan", 50_000_000)),
        )
    except Exception as exc:
        tradability = {
            "status": "unavailable",
            "basic_execution_feasible": False,
            "hard_blocks": [],
            "cautions": [f"可交易性检查失败：{exc}"],
        }

    reference = context.reference
    allow_current_snapshot = (
        str(clock.get("session_status")) == "post_close"
        and pd.Timestamp(reference.date()).normalize() == pd.Timestamp(as_of).normalize()
    )
    try:
        raw_fundamentals = huoqu_jibenmian(
            code,
            trade_date=as_of,
            allow_current_snapshot=allow_current_snapshot,
        )
    except Exception as exc:
        raw_fundamentals = {
            "profile": {},
            "valuation": {},
            "financials": {},
            "sources": {},
            "data_quality": {},
            "warnings": [],
            "errors": [f"基本面请求失败：{exc}"],
        }
    fundamentals = _fundamental_block(raw_fundamentals, as_of=as_of)
    stock = _identity(
        code,
        resolved_profile,
        fundamentals.get("profile"),
        snapshot,
    )
    if stock.get("name") in (None, ""):
        stock["name"] = name_hint

    try:
        unified = yunxing_dangu_tongyi_lianghua(
            code=code,
            name=str(stock.get("name") or name_hint),
            industry=str(stock.get("industry") or ""),
            history=history,
            analysis_date=pd.Timestamp(as_of),
            snapshot=snapshot,
            clock=clock,
            technical=technical,
            fundamentals=fundamentals,
            tradability=tradability,
            config=config,
            context=context,
        )
    except RuntimeError as exc:
        return _source_error_result(
            query=query,
            status="unavailable",
            outcome="data_unavailable",
            error_code="single_stock_unified_quantitative_unavailable",
            error=f"单股统一量化分析未完成：{exc}",
            next_action="稍后重新获取横截面和比较池日线；不使用旧市场数据降级",
            stage="unified_quantitative_analysis",
            stock=stock,
            data_provenance={"history": history_meta},
        )
    fundamentals = unified["fundamental_analysis"]
    tradability = unified["tradability"]
    buy_decision = unified["buy_decision"]
    unified_identity = unified.get("stock_identity")
    if isinstance(unified_identity, dict):
        for key in ("name", "industry", "market", "list_date"):
            value = unified_identity.get(key)
            current = stock.get(key)
            if value not in (None, "") and (
                current in (None, "") or (key == "name" and str(current) == code)
            ):
                stock[key] = value
    recommended = bool(buy_decision.get("meets_recommendation_threshold"))

    warnings = _dedupe(
        [
            *(resolution_warnings or []),
            *(history_meta.get("warnings") or []),
            *(fundamentals.get("warnings") or []),
        ]
    )
    risks = _technical_risks(technical)
    risks.extend(_dedupe(unified.get("risks")))
    risks.extend(_dedupe(tradability.get("hard_blocks")))
    risks.extend(_dedupe(tradability.get("cautions")))
    if fundamentals.get("status") != "ok":
        risks.append("基本面数据不可用或不完整，不能据此断言基本面差")
    if snapshot.get("status") != "ok":
        risks.append("实时快照不可用，当前价格和可交易性只能按最近完整日线参考")
    risks = list(dict.fromkeys(risks))

    result = {
        "status": "ok",
        "outcome": "recommendation" if recommended else "no_recommendation",
        "tool_contract_version": DANGU_TOOL_CONTRACT_VERSION,
        "analysis_type": DANGU_ANALYSIS_TYPE,
        "query": query,
        "stock": stock,
        "selected_stock": stock,
        "as_of": as_of,
        "generated_at": context.reference.strftime("%Y-%m-%d %H:%M:%S"),
        "market_clock": _json_safe(clock),
        "result_confirmation": _result_confirmation(clock),
        "recommendation_available": recommended,
        "primary": None,
        "alternatives": [],
        "buy_decision": _json_safe(buy_decision),
        "daily_factor_analysis": _json_safe(unified.get("factor_analysis")),
        "limit_up_pullback_pattern": _json_safe(unified.get("limit_up_pullback_pattern")),
        "late_session_analysis": _json_safe(unified.get("late_session_analysis")),
        "ranking_details": _json_safe(unified.get("ranking_details")),
        "technical_summary": _json_safe(technical),
        "fundamental_analysis": fundamentals,
        "tradability": _json_safe(tradability),
        "risks": risks,
        "warnings": warnings,
        "data_provenance": {
            "stock_resolution": {
                "source": "remote_stock_basic_or_akshare",
                "warnings": list(resolution_warnings or []),
                "persistence": "none",
            },
            "history": _json_safe(history_meta),
            "realtime_snapshot": _json_safe(
                {
                    key: snapshot.get(key)
                    for key in (
                        "status",
                        "source",
                        "captured_at",
                        "provider_trade_date",
                        "timeliness",
                        "error",
                    )
                    if key in snapshot
                }
            ),
            "fundamentals": {
                "sources": fundamentals.get("sources", {}),
                "as_of": as_of,
                "persistence": "none",
            },
            **_json_safe(unified.get("data_provenance") or {}),
        },
        "analysis_stage": {
            "status": "completed",
            "scope": "单股身份、八组日K因子、基本面、形态、尾盘、风险扣分、综合评分和买入门槛复核已完成",
            "prediction_status": "not_available",
            "prediction_confirmation_required": False,
            "confirmation_timing": "not_applicable",
            "initial_preapproval_counts": False,
            "affirmative_reply_defaults_to": None,
            "prediction_data_policy": "fresh_remote_download_without_local_market_cache",
            "next_step": "当前结果已明确是否建议买入；单股结论不产生自动预测资格",
        },
        "research_scope": "公开数据单股量化研究建议，不连接证券账户、不提交委托、不自动交易",
    }
    return _json_safe(result)


__all__ = ["DANGU_ANALYSIS_TYPE", "DANGU_TOOL_CONTRACT_VERSION", "fenxi_dangu"]
