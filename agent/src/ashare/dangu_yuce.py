"""合格 A 股的日 K 预测业务编排。

本模块是预测子系统的稳定门面：协调同行样本、因子面板、模型训练、交易日历和
结果解释；具体模型训练实现在 yuce_xunlian 中。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ashare.fenxi_yinzi import zengjia_hengjiemian_yinzi
from src.ashare.gupiao_yanjiu import guifan_you_xian_shuzhi
from src.ashare.jiaoyi_zhixing import (
    guifan_gujia_jingdu,
    jiazai_chengben_jiashe,
    jisuan_gupiao_wangfan_chengben,
    jisuan_zhangdieting_bianjie,
    koujian_jiaoyi_chengben,
)
from src.ashare.moxing_gongju import goujian_moxing_shuju
from src.ashare.moxing_pinggu import signal_evidence_gate
from src.ashare.riping_yinzi import enrich_daily_factor_panel
from src.ashare.shichang_shuju import huoqu_jiaoyi_rili, shichang_shizhong
from src.ashare.tonghang_yangben import (
    goujian_tonghang_kuaizhao,
    huoqu_tonghang_lishi,
    xuanze_tonghang_yangben,
)
from src.ashare.yuce_xunlian import (
    SINGLE_STOCK_FEATURE_COLUMNS,
    xunlian_chiyouqi_yuce_moxing,
    xunlian_weilai_shoupan_yuce_moxing,
)


def _next_market_sessions(signal_date: str, count: int = 4) -> dict[str, Any]:
    """使用统一交易日历生成预测日期；日历不可用时不伪造工作日。"""
    signal = pd.Timestamp(signal_date).normalize()
    calendar = huoqu_jiaoyi_rili()
    clock = shichang_shizhong(calendar=calendar)
    sessions = sorted(day for day in calendar.open_dates if day > signal)[:count]
    warnings = list(calendar.warnings)
    if len(sessions) < count:
        warnings.append(f"交易日历只取得 {len(sessions)}/{count} 个后续交易日，停止生成预测日期")
        return {
            "status": "unavailable",
            "source": calendar.source,
            "signal_date": signal.strftime("%Y-%m-%d"),
            "future_session_dates": {},
            "assumed_entry_date": None,
            "scenario_exit_dates": {},
            "market_clock": clock,
            "warnings": warnings,
        }
    return {
        "status": "ok",
        "source": calendar.source,
        "signal_date": signal.strftime("%Y-%m-%d"),
        "future_session_dates": {
            f"T+{horizon}": sessions[horizon - 1].strftime("%Y-%m-%d")
            for horizon in [1, 2, 3]
        },
        "assumed_entry_date": sessions[0].strftime("%Y-%m-%d"),
        "scenario_exit_dates": {f"T+{horizon}": sessions[horizon].strftime("%Y-%m-%d") for horizon in [1, 2, 3]},
        "market_clock": clock,
        "warnings": warnings,
    }


def _future_schedule_unavailable_reason(
    schedule: dict[str, Any],
    tradability: dict[str, Any],
) -> str:
    """Reject a so-called future horizon whose first target session has already ended."""
    clock = tradability.get("market_clock") or {}
    captured_at = pd.to_datetime(clock.get("captured_at"), errors="coerce")
    first_target = pd.to_datetime(
        (schedule.get("future_session_dates") or {}).get("T+1"),
        errors="coerce",
    )
    if pd.isna(first_target):
        return "无法确定未来第一个交易日"
    if pd.isna(captured_at):
        return "缺少分析时间，无法确认预测日期仍属于未来"
    captured_day = pd.Timestamp(captured_at).normalize()
    target_day = pd.Timestamp(first_target).normalize()
    session_status = str(clock.get("session_status") or "")
    if target_day < captured_day or (
        target_day == captured_day
        and session_status in {"close_pending", "post_close", "non_trading_day"}
    ):
        return (
            f"行情源最新完整日线对应的首个预测日 {target_day.strftime('%Y-%m-%d')} 已经结束；"
            "需要等待数据源更新到最新完整收盘后重新预测"
        )
    return ""


def _fundamental_risk_flags(fundamentals: dict[str, Any]) -> list[str]:
    financials = fundamentals.get("financials") or {}
    valuation = fundamentals.get("valuation") or {}
    profile = fundamentals.get("profile") or {}
    industry = str(profile.get("industry") or profile.get("所属行业") or "")
    financial_industry = any(value in industry for value in ("银行", "保险", "证券", "多元金融"))
    flags: list[str] = []
    roe = guifan_you_xian_shuzhi(financials.get("roe_pct"))
    growth = guifan_you_xian_shuzhi(financials.get("net_profit_yoy_pct"))
    debt = guifan_you_xian_shuzhi(financials.get("debt_to_assets_pct"))
    pe = guifan_you_xian_shuzhi(valuation.get("pe_ttm"))
    if pe is None:
        pe = guifan_you_xian_shuzhi(valuation.get("pe_dynamic"))
    if roe is not None and roe < 0:
        flags.append("最新已公告口径 ROE 为负")
    if growth is not None and growth < -30:
        flags.append("最新已公告口径净利润同比下降超过30%")
    if debt is not None and debt > 80 and not financial_industry:
        flags.append("非金融企业资产负债率超过80%")
    if pe is not None and pe <= 0:
        flags.append("当前市盈率口径为负")
    return flags


def _build_analysis_assessment(
    *,
    holding_days: int,
    forecast: dict[str, Any],
    validation: dict[str, Any],
    tradability: dict[str, Any],
    technical: dict[str, Any],
    fundamentals: dict[str, Any],
    schedule: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config.get("dangu", {})
    minimum_net = float(settings.get("assessment_min_net_return", 0.003))
    minimum_probability = float(settings.get("assessment_min_positive_probability", 0.55))
    evidence_label = "证据不足"
    reasons: list[str] = []
    requested = forecast.get(f"T+{holding_days}") or {}
    metrics = validation.get("horizons", {}).get(f"T+{holding_days}") or {}
    fundamental_flags = _fundamental_risk_flags(fundamentals)
    entry_timing_valid = bool(tradability.get("model_entry_timing_valid", True))
    entry_timing_reason = str(tradability.get("model_entry_timing_reason") or "模型假设的开盘测算基准可用")
    assumed_entry = pd.to_datetime(schedule.get("assumed_entry_date"), errors="coerce")
    captured_at = pd.to_datetime(
        (tradability.get("market_clock") or {}).get("captured_at"),
        errors="coerce",
    )
    if pd.notna(assumed_entry) and pd.notna(captured_at):
        assumed_entry = pd.Timestamp(assumed_entry).normalize()
        captured_day = pd.Timestamp(captured_at).normalize()
        captured_minute = pd.Timestamp(captured_at).hour * 60 + pd.Timestamp(captured_at).minute
        if assumed_entry < captured_day or (assumed_entry == captured_day and captured_minute >= 9 * 60 + 30):
            entry_timing_valid = False
            entry_timing_reason = "该收盘信号对应的假设开盘入口已经过去，当前价格不属于模型训练的入口口径"
        elif assumed_entry > captured_day or captured_minute < 9 * 60 + 30:
            entry_timing_valid = True
            entry_timing_reason = "模型假设的开盘入口尚未发生，情景时点仍有效"

    if not tradability.get("basic_execution_feasible"):
        evidence_label = "证据偏负面"
        reasons.extend(str(value) for value in tradability.get("hard_blocks", []))
    elif not entry_timing_valid:
        evidence_label = "证据不足"
        reasons.append(entry_timing_reason)
    elif not requested or requested.get("entry_to_exit_gross_return") is None:
        reasons.append("指定持有期没有可用模型预测")
    elif not metrics.get("validation_passed"):
        reasons.append("指定持有期模型没有通过滚动样本外门槛")
    elif not requested.get("position_and_cost", {}).get("execution_feasible"):
        evidence_label = "证据偏负面"
        reasons.append(
            str(
                requested.get("position_and_cost", {}).get("reason")
                or "按目标资金测算，无法满足最低买入数量或成交容量约束"
            )
        )
    else:
        net = float(requested.get("estimated_net_return_after_cost") or 0.0)
        probability = requested.get("direction_model_positive_probability")
        if probability is None:
            probability = requested.get("empirical_positive_probability")
        probability_value = float(probability) if probability is not None else 0.5
        if net <= -minimum_net or probability_value < 0.45:
            evidence_label = "证据偏负面"
            reasons.append("通过验证的成本后期望或样本外方向概率明显不利")
        elif net >= minimum_net and probability_value >= minimum_probability:
            evidence_label = "证据偏正面"
            reasons.append("指定期限模型通过滚动样本外验证，成本后期望和校准方向概率同时达到分析门槛")
        else:
            evidence_label = "证据中性"
            reasons.append("模型有效，但成本后优势或样本外方向概率没有形成明显方向")

    rsi = guifan_you_xian_shuzhi(technical.get("rsi_14"))
    ret_5 = guifan_you_xian_shuzhi((technical.get("returns") or {}).get("5d"))
    if evidence_label == "证据偏正面" and (
        (rsi is not None and rsi >= 80) or (ret_5 is not None and ret_5 >= 0.18)
    ):
        evidence_label = "证据中性"
        reasons.append("短线指标处于过热区，正面模型证据受到追高风险削弱")
    if evidence_label == "证据偏正面" and fundamental_flags:
        evidence_label = "证据中性"
        reasons.append("基本面存在明显风险项，短线量化证据不足以覆盖这些不确定性")
    reasons.extend(fundamental_flags)
    requested_probability = requested.get("direction_model_positive_probability") if requested else None
    if requested_probability is None and requested:
        requested_probability = requested.get("empirical_positive_probability")
    signal_gate = signal_evidence_gate(
        validation_passed=bool(metrics.get("validation_passed")),
        execution_feasible=bool((requested.get("position_and_cost") or {}).get("execution_feasible"))
        if requested
        else False,
        net_return=guifan_you_xian_shuzhi(requested.get("estimated_net_return_after_cost")) if requested else None,
        positive_probability=guifan_you_xian_shuzhi(requested_probability),
        quality_score=guifan_you_xian_shuzhi(metrics.get("quality_score")),
        minimum_net_return=minimum_net,
        minimum_positive_probability=minimum_probability,
        minimum_quality_score=float(config.get("moxing", {}).get("abstain_min_quality_score", 0.40)),
    )
    if evidence_label == "证据偏正面" and not signal_gate["actionable_signal"]:
        evidence_label = "证据中性"
        reasons.extend(value for value in signal_gate["reasons"] if value not in reasons)
    if evidence_label != "证据偏正面":
        signal_gate["actionable_signal"] = False
        signal_gate["decision"] = "abstain"
        if not signal_gate["reasons"]:
            signal_gate["reasons"] = ["综合技术、基本面或时点约束后没有形成明确正面证据"]
    summary_detail = reasons[0] if reasons else "指定期限证据不足"
    return {
        "evidence_label": evidence_label,
        "requested_horizon": f"T+{holding_days}",
        "summary": f"{evidence_label}：{summary_detail}",
        "confidence": metrics.get("quality_label", "low") if metrics.get("validation_passed") else "insufficient",
        "reasons": reasons,
        "signal_gate": signal_gate,
        "assessment_thresholds": {
            "minimum_net_return_after_cost": minimum_net,
            "minimum_oos_direction_positive_probability": minimum_probability,
            "model_validation_must_pass": True,
            "execution_constraints_must_be_clear": True,
        },
        "scenario_timing": {
            "valid": entry_timing_valid,
            "reason": entry_timing_reason,
            "assumed_entry_date": schedule.get("assumed_entry_date"),
            "scenario_exit_date": schedule.get("scenario_exit_dates", {}).get(f"T+{holding_days}"),
        },
        "fundamental_risk_flags": fundamental_flags,
        "responsibility_note": "这是分析证据汇总，不是买入、卖出或持有指令；最终决定由用户自行作出。",
    }


def yanjiu_dangu_yuce(
    *,
    code: str,
    name: str,
    industry: str,
    target_history: pd.DataFrame,
    target_source: str,
    target_adjustment: str,
    source: str,
    signal_date: str,
    config: dict[str, Any],
    technical: dict[str, Any],
    fundamentals: dict[str, Any],
    tradability: dict[str, Any],
) -> dict[str, Any]:
    """Run the model, evidence summary, and fixed three-session forecast."""
    holding_days = 2
    budget_yuan = None
    configured_budget, cost_scenario, cost_source, cost_errors = jiazai_chengben_jiashe("research_reference")
    actual_budget = float(budget_yuan) if budget_yuan is not None else float(configured_budget)
    schedule = _next_market_sessions(signal_date)
    tradability = {
        **tradability,
        "market_clock": schedule.get("market_clock") or tradability.get("market_clock") or {},
    }
    peer_table, peer_meta = xuanze_tonghang_yangben(
        code=code,
        name=name,
        industry=industry,
        signal_date=signal_date,
        config=config,
    )
    peer_snapshot = goujian_tonghang_kuaizhao(peer_table, code)
    histories, names, history_meta = huoqu_tonghang_lishi(
        peer_table=peer_table,
        target_code=code,
        target_history=target_history,
        target_source=target_source,
        target_adjustment=target_adjustment,
        signal_date=signal_date,
        source=source,
        config=config,
    )
    minimum_peers = int(config.get("dangu", {}).get("minimum_peer_stocks", 8))
    if code not in histories or len(histories) < minimum_peers:
        reason = (
            f"可用同行历史只有 {len(histories)} 只，至少需要 {minimum_peers} 只"
            if code in histories
            else "目标股票没有可用于模型的前复权历史"
        )
        history_errors = [str(value) for value in history_meta.get("errors", [])]
        unadjusted_count = sum("未复权行情" in value for value in history_errors)
        if unadjusted_count:
            reason += f"；另有 {unadjusted_count} 只股票仅取得未复权行情，不能进入模型"
            if source == "tushare":
                reason += "；严格 Tushare 模式不会自动降级到 AKShare"
        return {
            "status": "unavailable",
            "requested_horizon": f"T+{holding_days}",
            "schedule": schedule,
            "peer_universe": {**peer_meta, "history_fetch": history_meta, "relative_snapshot": peer_snapshot},
            "forecast": {},
            "future_3_trading_days": {
                "status": "unavailable",
                "signal_date": signal_date,
                "forecast": {},
                "error": reason,
            },
            "validation": {"horizons": {}, "passed_horizons": 0},
            "analysis_assessment": {
                "evidence_label": "证据不足" if tradability.get("basic_execution_feasible") else "证据偏负面",
                "requested_horizon": f"T+{holding_days}",
                "summary": (
                    f"证据不足：{reason}"
                    if tradability.get("basic_execution_feasible")
                    else "证据偏负面：存在明显可交易性约束"
                ),
                "reasons": [reason] + list(tradability.get("hard_blocks", [])),
                "signal_gate": {
                    "actionable_signal": False,
                    "decision": "abstain",
                    "reasons": [reason] + list(tradability.get("hard_blocks", [])),
                },
                "responsibility_note": "这是分析证据汇总，不是交易指令；最终决定由用户自行作出。",
            },
            "limitations": ["同行历史不足时不退化为单股票自拟合模型，也不把启发式技术分冒充预测"],
        }

    panel = goujian_moxing_shuju(histories, names, [1, 2, 3])
    panel = zengjia_hengjiemian_yinzi(panel)
    panel, daily_factor_meta = enrich_daily_factor_panel(
        panel,
        source=source,
        include_historical_valuation=True,
    )
    target_rows = panel[
        (panel["ts_code"] == code)
        & (pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize() == pd.Timestamp(signal_date).normalize())
    ]
    if target_rows.empty:
        target_rows = panel[panel["ts_code"] == code].sort_values("trade_date").tail(1)
    if target_rows.empty:
        raise RuntimeError("模型面板中没有目标股票的最新特征")
    predictions, validation = xunlian_chiyouqi_yuce_moxing(
        panel=panel,
        latest=target_rows.tail(1),
        config=config,
        budget_yuan=actual_budget,
    )
    future_schedule_error = _future_schedule_unavailable_reason(schedule, tradability)
    try:
        if future_schedule_error:
            raise RuntimeError(future_schedule_error)
        future_predictions, future_validation = xunlian_weilai_shoupan_yuce_moxing(
            panel=panel,
            latest=target_rows.tail(1),
            config=config,
        )
        future_error = ""
    except Exception as exc:
        future_predictions = pd.DataFrame()
        future_validation = {
            "horizons": {},
            "passed_horizons": 0,
            "overall_quality_score": 0.0,
            "overall_quality_label": "low",
        }
        future_error = str(exc)
    prediction_row = predictions.iloc[0]
    analysis_price = guifan_you_xian_shuzhi(tradability.get("analysis_price"), 3) or guifan_you_xian_shuzhi(prediction_row.get("close"), 3)
    latest_model_row = target_rows.tail(1).iloc[0]
    stock_cost_rate, position_cost = jisuan_gupiao_wangfan_chengben(
        code,
        float(analysis_price),
        actual_budget,
        cost_scenario,
        daily_amount_yuan=guifan_you_xian_shuzhi(latest_model_row.get("amount_yuan")),
        atr_pct=guifan_you_xian_shuzhi(latest_model_row.get("atr_14_pct")),
        trading_settings=None,
    )
    forecast: dict[str, Any] = {}
    for horizon in [1, 2, 3]:
        gross = guifan_you_xian_shuzhi(prediction_row.get(f"pred_t{horizon}"))
        metrics = validation["horizons"].get(f"T+{horizon}", {})
        net = koujian_jiaoyi_chengben(float(gross), stock_cost_rate) if gross is not None and stock_cost_rate is not None else None
        interval = metrics.get("prediction_interval_80")
        conformal_interval = metrics.get("conformal_prediction_interval_80")
        quantile_interval = metrics.get("quantile_prediction_interval_80")
        cqr_interval = metrics.get("conformalized_quantile_prediction_interval_80")
        preferred_interval = (
            metrics.get("preferred_prediction_interval_80")
            or conformal_interval
            or interval
        )
        net_interval = [
            round(koujian_jiaoyi_chengben(float(value), stock_cost_rate), 6) for value in interval
        ] if interval and stock_cost_rate is not None else None
        conformal_net_interval = [
            round(koujian_jiaoyi_chengben(float(value), stock_cost_rate), 6) for value in conformal_interval
        ] if conformal_interval and stock_cost_rate is not None else None
        preferred_net_interval = [
            round(koujian_jiaoyi_chengben(float(value), stock_cost_rate), 6) for value in preferred_interval
        ] if preferred_interval and stock_cost_rate is not None else None
        forecast[f"T+{horizon}"] = {
            "entry_to_exit_gross_return": round(float(gross), 6) if gross is not None else None,
            "entry_to_exit_gross_return_pct": round(float(gross) * 100.0, 3) if gross is not None else None,
            "estimated_net_return_after_cost": round(float(net), 6) if net is not None else None,
            "estimated_net_return_after_cost_pct": round(float(net) * 100.0, 3) if net is not None else None,
            "empirical_positive_probability": metrics.get("latest_empirical_positive_probability"),
            "direction_model_positive_probability": metrics.get("latest_direction_positive_probability"),
            "direction_probability_method": metrics.get("direction_probability_method"),
            "empirical_return_interval_80": interval,
            "empirical_net_return_interval_80": net_interval,
            "conformal_return_interval_80": conformal_interval,
            "conformal_net_return_interval_80": conformal_net_interval,
            "quantile_return_interval_80": quantile_interval,
            "quantile_median_return": metrics.get("quantile_median_prediction"),
            "conformalized_quantile_return_interval_80": cqr_interval,
            "preferred_return_interval_80": preferred_interval,
            "preferred_net_return_interval_80": preferred_net_interval,
            "preferred_return_interval_method": metrics.get(
                "preferred_prediction_interval_method"
            ),
            "validation_passed": bool(metrics.get("validation_passed")),
            "model_quality": metrics.get("quality_label", "low"),
            "position_and_cost": position_cost,
            "timing": f"{signal_date} 收盘后信号，假设下一交易日开盘作为测算基准；比较入场后第{horizon}个可卖出交易日收盘",
            "assumed_entry_date": schedule.get("assumed_entry_date"),
            "scenario_exit_date": schedule.get("scenario_exit_dates", {}).get(f"T+{horizon}"),
            "predicted_close": None,
            "predicted_close_unavailable_reason": "入场开盘价尚未知，模型预测的是入场到退出收益，不能伪造精确目标价",
        }

    future_forecast: dict[str, Any] = {}
    signal_close = guifan_you_xian_shuzhi(prediction_row.get("close"), 3)
    future_status = "ok" if not future_predictions.empty and signal_close is not None else "unavailable"
    if future_status == "ok":
        future_row = future_predictions.iloc[0]
        limit_pct = guifan_you_xian_shuzhi(tradability.get("price_limit_pct"))
        limit_rate = limit_pct / 100.0 if limit_pct is not None else None
        for horizon in [1, 2, 3]:
            predicted_return = guifan_you_xian_shuzhi(future_row.get(f"future_pred_t{horizon}"))
            metrics = future_validation.get("horizons", {}).get(f"T+{horizon}", {})
            interval = metrics.get("prediction_interval_80")
            conformal_interval = metrics.get("conformal_prediction_interval_80")
            quantile_interval = metrics.get("quantile_prediction_interval_80")
            cqr_interval = metrics.get("conformalized_quantile_prediction_interval_80")
            preferred_interval = (
                metrics.get("preferred_prediction_interval_80")
                or conformal_interval
                or interval
            )
            lower_price, upper_price = jisuan_zhangdieting_bianjie(signal_close, limit_rate, horizon)
            predicted_close = None
            predicted_close_interval = None
            if predicted_return is not None:
                raw_close = signal_close * (1.0 + predicted_return)
                predicted_close = guifan_gujia_jingdu(min(max(raw_close, lower_price), upper_price))
            if preferred_interval and len(preferred_interval) == 2:
                interval_prices = [signal_close * (1.0 + float(value)) for value in preferred_interval]
                predicted_close_interval = [
                    guifan_gujia_jingdu(min(max(value, lower_price), upper_price))
                    for value in interval_prices
                ]
            future_forecast[f"T+{horizon}"] = {
                "target_trade_date": schedule.get("future_session_dates", {}).get(f"T+{horizon}"),
                "cumulative_return_from_signal_close": (
                    round(float(predicted_return), 6) if predicted_return is not None else None
                ),
                "cumulative_return_from_signal_close_pct": (
                    round(float(predicted_return) * 100.0, 3) if predicted_return is not None else None
                ),
                "predicted_close_reference": predicted_close,
                "predicted_close_interval_80": predicted_close_interval,
                "predicted_close_interval_method": metrics.get(
                    "preferred_prediction_interval_method",
                    "nearest_oos_empirical_quantile",
                ),
                "empirical_return_interval_80": interval,
                "conformal_return_interval_80": conformal_interval,
                "quantile_return_interval_80": quantile_interval,
                "quantile_median_return": metrics.get("quantile_median_prediction"),
                "conformalized_quantile_return_interval_80": cqr_interval,
                "preferred_return_interval_80": preferred_interval,
                "empirical_positive_probability": metrics.get("latest_empirical_positive_probability"),
                "direction_model_positive_probability": metrics.get("latest_direction_positive_probability"),
                "direction_probability_method": metrics.get("direction_probability_method"),
                "validation_passed": bool(metrics.get("validation_passed")),
                "model_quality": metrics.get("quality_label", "low"),
                "direction": (
                    "up" if predicted_return is not None and predicted_return > 0
                    else "down" if predicted_return is not None and predicted_return < 0
                    else "flat_or_unavailable"
                ),
            }

    future_three_days = {
        "status": future_status,
        "signal_date": signal_date,
        "signal_close": signal_close,
        "definition": "以最近完整收盘日为T，预测未来第1、2、3个市场交易日收盘相对T收盘的累计收益",
        "forecast": future_forecast,
        "validation": future_validation,
        "error": future_error or None,
        "interpretation": (
            "预测收盘价是模型参考值，不是目标价或成交承诺；未通过样本外验证的周期只作观察。"
        ),
    }
    analysis_assessment = _build_analysis_assessment(
        holding_days=holding_days,
        forecast=forecast,
        validation=validation,
        tradability=tradability,
        technical=technical,
        fundamentals=fundamentals,
        schedule=schedule,
        config=config,
    )
    return {
        "status": "ok",
        "requested_horizon": f"T+{holding_days}",
        "requested_holding_trading_days": holding_days,
        "schedule": schedule,
        "peer_universe": {
            **peer_meta,
            "history_fetch": history_meta,
            "relative_snapshot": peer_snapshot,
        },
        "daily_factor_data": daily_factor_meta,
        "forecast": forecast,
        "future_3_trading_days": future_three_days,
        "validation": validation,
        "analysis_assessment": analysis_assessment,
        "cost_assumption": {
            "scenario": cost_scenario.name,
            "source": cost_source,
            "budget_yuan": round(actual_budget, 2),
            "estimated_roundtrip_cost_rate": round(float(stock_cost_rate), 6) if stock_cost_rate is not None else None,
            "assumption": "程序内固定研究成本假设，不提供外部交易成本配置",
            "assumption_errors": cost_errors,
        },
        "methodology": {
            "model": "HistGradientBoostingRegressor + 稳健缩放Ridge的小型集成",
            "ensemble_weighting": "每个滚动折只使用更早折的样本外预测选权重；最终生产权重使用全部滚动样本外预测",
            "training_universe": "目标股票的当前同行优先，加少量全市场高流动性参考股票",
            "validation": "六折扩展窗口、标签跨界清除的滚动样本外验证，最后一折作为最终保留测试窗口",
            "signal": "只使用已确认收盘的日线、同日精确匹配的历史估值、市场指数日K和当日横截面特征",
            "execution_scenario": "假设下一交易日开盘作为收益测算基准，并按A股T+1比较指定T+1/T+2/T+3收盘",
            "future_forecast": "另行预测从最近完整收盘到未来第1/2/3个交易日收盘，两类结果都只用于分析",
            "llm_boundary": "数值、验证与证据标签均由程序生成；LLM只能解释，不能改写或作交易决定",
        },
        "limitations": [
            "当前同行池用于历史训练，仍存在当前成分与幸存者偏差",
            "产品永久只做日K；不能模拟集合竞价排队、盘口深度和突发公告冲击",
            "经验区间和上涨比例来自历史样本外相似预测，不是收益保证",
        ],
    }


__all__ = [
    "SINGLE_STOCK_FEATURE_COLUMNS",
    "yanjiu_dangu_yuce",
]
