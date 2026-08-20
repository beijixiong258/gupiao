"""量化分析结果的安全降级展示。

正常回答仍由智能体结合上下文组织；这里只在模型输出混入异常语言字符时，根据已计算的
结构化结果生成一份保守中文答案，避免展示层把坏文本交给用户。
"""

from __future__ import annotations

from typing import Any, Iterable


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _items(values: Any, maximum: int) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
        if len(result) >= maximum:
            break
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _percent(value: Any, digits: int = 1) -> str | None:
    number = _number(value)
    return f"{number * 100:.{digits}f}%" if number is not None else None


def _fundamental_unavailable_reasons(fundamentals: dict[str, Any]) -> list[str]:
    """把底层错误归纳为用户能核验的数据可用性原因。"""

    errors = _items(fundamentals.get("errors"), 20)
    combined = " ".join(errors).lower()
    reasons: list[str] = []
    if any(marker in combined for marker in ("频率", "限频", "rate limit")):
        reasons.append("主数据源的基本资料或估值接口触发限频")
    if any(marker in combined for marker in ("权限", "permission", "access denied")):
        reasons.append("财务指标接口权限不足")
    if any(
        marker in combined
        for marker in ("connection", "连接", "remote end", "timed out", "timeout")
    ):
        reasons.append("备用公开数据源连接失败")
    if "公告日" in combined:
        reasons.append("备用财报缺少公告日，无法证明在分析日当时已经公开，因此未采用")
    if any(marker in combined for marker in ("与分析日", "与历史分析日")) and "不一致" in combined:
        reasons.append("备用估值快照与分析日不一致，因此未采用")
    if not reasons and errors:
        reasons.append("远端基本面接口本次没有返回可按分析日核验的数据")
    return reasons


def _fundamental_evidence(fundamentals: dict[str, Any]) -> list[str]:
    valuation = fundamentals.get("valuation")
    financials = fundamentals.get("financials")
    if not isinstance(valuation, dict):
        valuation = {}
    if not isinstance(financials, dict):
        financials = {}
    evidence: list[str] = []
    valuation_labels = (
        ("pe_ttm", "市盈率（TTM）"),
        ("pe_dynamic", "动态市盈率"),
        ("pb", "市净率"),
    )
    for key, label in valuation_labels:
        value = _number(valuation.get(key))
        if value is not None:
            evidence.append(f"{label} {value:.2f}")
    report_date = _text(financials.get("report_date"))
    if report_date:
        evidence.append(f"最新可核验报告期 {report_date}")
    financial_labels = (
        ("roe_pct", "净资产收益率"),
        ("gross_margin_pct", "毛利率"),
        ("net_margin_pct", "净利率"),
        ("debt_to_assets_pct", "资产负债率"),
        ("revenue_yoy_pct", "营收同比"),
        ("net_profit_yoy_pct", "净利润同比"),
    )
    for key, label in financial_labels:
        value = _number(financials.get(key))
        if value is not None:
            evidence.append(f"{label} {value:.2f}%")
    evidence.extend(_items(fundamentals.get("evidence"), 8))
    return list(dict.fromkeys(evidence))


def _divergence_summary(structure: dict[str, Any]) -> str | None:
    divergences = structure.get("divergences")
    if not isinstance(divergences, dict):
        return None
    active: list[str] = []
    inactive_count = 0
    for values in divergences.values():
        if not isinstance(values, dict):
            continue
        for signal in values.values():
            if not isinstance(signal, dict):
                continue
            status = _text(signal.get("status"))
            label = f"{_text(signal.get('indicator_label'))}{_text(signal.get('kind_label'))}"
            if status in {"active", "confirmed"} and label:
                active.append(label)
            elif status in {"expired", "invalidated"}:
                inactive_count += 1
    if active:
        return "当前仍有效的背离证据：" + "、".join(active) + "。"
    if inactive_count:
        return f"当前没有仍有效的背离信号；记录到的 {inactive_count} 项背离均已失效或过期。"
    return None


def _single_stock_fallback(payload: dict[str, Any]) -> str:
    """生成包含可追溯理由的单股买入结论，供展示保护逻辑使用。"""

    stock = payload.get("stock") or payload.get("selected_stock") or {}
    if not isinstance(stock, dict):
        stock = {}
    name = _text(stock.get("name")) or _text(payload.get("query")) or "这只股票"
    code = _text(stock.get("ts_code"))
    identity = f"{name}（{code}）" if code else name
    as_of = _text(payload.get("as_of")) or "最近完整交易日"
    technical = payload.get("technical_summary")
    if not isinstance(technical, dict):
        technical = {}
    structure = technical.get("macd_structure")
    if not isinstance(structure, dict):
        structure = {}
    classification = structure.get("structure_classification")
    if not isinstance(classification, dict):
        classification = {}
    classification_label = _text(classification.get("label")) or "当前结构没有形成明确方向"
    score = _number(technical.get("score_0_100"))
    close = _number(technical.get("close"))
    returns = technical.get("returns") if isinstance(technical.get("returns"), dict) else {}
    moving_averages = (
        technical.get("moving_averages")
        if isinstance(technical.get("moving_averages"), dict)
        else {}
    )
    buy_decision = payload.get("buy_decision")
    if not isinstance(buy_decision, dict):
        buy_decision = {}
    decision_label = _text(buy_decision.get("label")) or "暂不建议买入"
    ranking_score = _number(buy_decision.get("score_0_100"))
    decision_confidence = _number(buy_decision.get("confidence"))

    conclusion = f"结论：对{identity}的当前量化结论是“{decision_label}”"
    if ranking_score is not None:
        conclusion += f"，统一综合分为 {ranking_score:.2f}/100"
    if decision_confidence is not None:
        conclusion += f"，证据完整度约 {decision_confidence * 100:.0f}%"
    conclusion += f"；MACD 结构归类为“{classification_label}”。"
    lines = [conclusion]

    confirmation = _text(payload.get("result_confirmation"))
    generated_at = _text(payload.get("generated_at"))
    if confirmation == "intraday_provisional":
        tradability_basis = payload.get("tradability")
        amount_is_completed = bool(
            isinstance(tradability_basis, dict)
            and tradability_basis.get("amount_basis") == "latest_completed_daily_bar"
        )
        if amount_is_completed:
            lines.append(
                f"数据口径：技术结构使用截至 {as_of} 的完整日线；本次生成于"
                f"{generated_at or '盘中'}，盘中参考价为暂定，用于流动性检查的成交额来自 {as_of} 完整日线。"
            )
        else:
            lines.append(
                f"数据口径：技术结构使用截至 {as_of} 的完整日线；本次生成于"
                f"{generated_at or '盘中'}，盘中参考价格和可交易性证据均为暂定。"
            )
    elif confirmation == "close_pending":
        lines.append(f"数据口径：完整日线截至 {as_of}，收盘数据仍在确认中。")
    else:
        lines.append(f"数据口径：完整日线截至 {as_of}。")

    price_parts: list[str] = []
    if close is not None:
        price_parts.append(f"收盘价 {close:.2f} 元")
    for period in (1, 3, 5, 10, 20):
        value = _percent(returns.get(f"{period}d"))
        if value:
            price_parts.append(f"近 {period} 日 {value}")
    if price_parts:
        lines.append("价格与趋势：" + "；".join(price_parts) + "。")

    ma_parts: list[str] = []
    below: list[str] = []
    above: list[str] = []
    for period in (5, 10, 20, 60):
        value = _number(moving_averages.get(f"ma{period}"))
        if value is None:
            continue
        ma_parts.append(f"MA{period} {value:.2f}")
        if close is not None:
            (below if close < value else above).append(f"MA{period}")
    if ma_parts:
        position = ""
        if below:
            position += "，价格低于" + "、".join(below)
        if above:
            position += "，价格高于或等于" + "、".join(above)
        lines.append("均线位置：" + "；".join(ma_parts) + position + "。")

    rsi = _number(technical.get("rsi_14"))
    evidence = _items(technical.get("evidence"), 5)
    technical_parts = []
    if score is not None:
        technical_parts.append(f"技术状态分 {score:.0f}/100（不是上涨概率）")
    if rsi is not None:
        technical_parts.append(f"RSI(14) {rsi:.2f}")
    if evidence:
        technical_parts.append("程序依据为" + "、".join(evidence))
    if technical_parts:
        lines.append("技术状态：" + "；".join(technical_parts) + "。")

    factor = buy_decision.get("daily_factor_analysis")
    if not isinstance(factor, dict):
        factor = payload.get("daily_factor_analysis")
    groups = factor.get("groups") if isinstance(factor, dict) else {}
    if isinstance(groups, dict):
        group_parts: list[str] = []
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            label = _text(group.get("label")) or "因子组"
            group_score = _number(group.get("score_0_100"))
            group_parts.append(
                f"{label} {group_score:.1f}/100" if group_score is not None else f"{label}信息有限"
            )
        if group_parts:
            lines.append("八组日K因子：" + "；".join(group_parts[:8]) + "。")

        momentum = groups.get("momentum_reversal")
        momentum_values = momentum.get("values") if isinstance(momentum, dict) else {}
        if isinstance(momentum_values, dict):
            momentum_parts: list[str] = []
            for key, label in (
                ("ret_5", "5日动量"),
                ("ret_10", "10日动量"),
                ("ret_20", "20日动量"),
                ("macd_dif_pct", "MACD快线强度"),
                ("macd_hist_pct", "MACD柱强度"),
            ):
                value = _number(momentum_values.get(key))
                if value is not None:
                    momentum_parts.append(f"{label} {value * 100:.2f}%")
            if momentum_parts:
                lines.append("动量细项：" + "；".join(momentum_parts) + "。")

    ranking = buy_decision.get("ranking_details")
    if not isinstance(ranking, dict):
        ranking = payload.get("ranking_details")
    components = ranking.get("components") if isinstance(ranking, dict) else {}
    if isinstance(components, dict):
        component_labels = {
            "daily_factors": "八组日K因子",
            "fundamental": "基本面",
            "pattern": "涨停回马枪形态",
            "late_session": "尾盘证据",
        }
        component_parts: list[str] = []
        for key, label in component_labels.items():
            item = components.get(key)
            component_score = _number(item.get("score_0_100")) if isinstance(item, dict) else None
            component_parts.append(
                f"{label} {component_score:.1f}/100" if component_score is not None else f"{label}未计分"
            )
        penalty = _number(ranking.get("risk_penalty"))
        if penalty is not None:
            component_parts.append(f"风险扣分 {penalty:.1f}")
        lines.append("综合评分构成：" + "；".join(component_parts) + "。")

    positives = _items(buy_decision.get("positive_evidence"), 6)
    blockers = _items(buy_decision.get("blocking_conditions"), 8)
    unmet = [
        value
        for value in _items(buy_decision.get("unmet_conditions"), 8)
        if value not in blockers
    ]
    if positives:
        lines.append("支持条件：" + "；".join(positives) + "。")
    if blockers:
        lines.append("未通过条件：" + "；".join(blockers) + "。")
    if unmet:
        lines.append("其余未满足或需关注项：" + "；".join(unmet) + "。")

    lines.append(f"MACD 结构：{classification_label}。")
    macd = technical.get("macd") if isinstance(technical.get("macd"), dict) else {}
    support = _items(structure.get("supporting_evidence"), 4)
    cross = structure.get("latest_cross") if isinstance(structure.get("latest_cross"), dict) else {}
    if cross.get("status") == "active" and _text(cross.get("label")):
        cross_text = (
            f"{_text(cross.get('event_date'))} 在{_text(cross.get('region_label'))}形成"
            f"{_text(cross.get('label'))}，距今 {cross.get('age_trading_sessions')} 个交易日"
        )
        if cross_text not in " ".join(support):
            support.append(cross_text)
    histogram = _number(macd.get("histogram"))
    if histogram is not None and histogram > 0:
        support.append(f"MACD 柱为正（{histogram:.4f}），只作为修复证据")
    support = list(dict.fromkeys(item for item in support if _text(item)))[:5]
    counter = _items(structure.get("counter_evidence"), 5)
    if support:
        lines.append("短线修复证据：" + "；".join(support) + "。")
    if counter:
        lines.append("反向证据：" + "；".join(counter) + "。")
    if support and counter:
        lines.append(
            "证据冲突：正向结构或短线修复证据与动能衰减等反向风险同时存在；"
            "单一金叉或正柱不能覆盖价格趋势和风险证据，"
            f"因此仍按“{classification_label}”处理。"
        )
    divergence_line = _divergence_summary(structure)
    if divergence_line:
        lines.append("背离检查：" + divergence_line)

    volatility = _percent(technical.get("annualized_volatility_20"))
    drawdown_number = _number(technical.get("drawdown_from_20d_high"))
    risk_metrics = []
    if volatility:
        risk_metrics.append(f"20 日年化波动率 {volatility}")
    if drawdown_number is not None:
        risk_metrics.append(f"较近 20 日高点回撤 {abs(drawdown_number) * 100:.1f}%")
    if risk_metrics:
        lines.append("波动与回撤：" + "；".join(risk_metrics) + "。")

    fundamentals = payload.get("fundamental_analysis")
    if isinstance(fundamentals, dict):
        if fundamentals.get("status") == "ok":
            fundamental_evidence = _fundamental_evidence(fundamentals)
            if fundamental_evidence:
                lines.append("基本面证据：" + "；".join(fundamental_evidence[:8]) + "。")
            else:
                lines.append("基本面：只取得部分字段，未取得可展示的核心估值或财务指标。")
        else:
            reasons = _fundamental_unavailable_reasons(fundamentals)
            reason_text = "；".join(reasons) if reasons else "远端接口本次没有返回可核验字段"
            available_fields = fundamentals.get("available_fields")
            profile_available = bool(
                isinstance(available_fields, dict) and available_fields.get("profile")
            )
            availability_text = (
                "已取得公司基本资料，但没有取得可按分析日核验的估值和财务指标"
                if profile_available
                else "没有取得可按分析日核验的基本资料、估值和财务指标"
            )
            lines.append(
                "基本面：" + availability_text + "，原因是" + reason_text
                + "。这是数据可用性问题，不能据此断言公司基本面好或差。"
            )

    tradability = payload.get("tradability")
    if isinstance(tradability, dict):
        tradability_parts: list[str] = []
        analysis_price = _number(tradability.get("analysis_price"))
        if analysis_price is not None:
            basis = "实时行情快照" if tradability.get("analysis_price_basis") == "realtime_snapshot" else "完整日线"
            tradability_parts.append(f"{basis}参考价 {analysis_price:.2f} 元")
        amount = _number(tradability.get("amount_yuan"))
        if amount is not None:
            amount_trade_date = _text(tradability.get("amount_trade_date"))
            if tradability.get("amount_basis") == "latest_completed_daily_bar":
                amount_label = (
                    f"{amount_trade_date} 完整交易日成交额"
                    if amount_trade_date
                    else "最近完整交易日成交额"
                )
            else:
                amount_label = "成交额"
            tradability_parts.append(f"{amount_label}约 {amount / 100_000_000:.2f} 亿元")
        limit_pct = _number(tradability.get("price_limit_pct"))
        if limit_pct is not None:
            tradability_parts.append(f"适用 {limit_pct:.0f}% 涨跌幅限制")
        status_text = (
            "基础执行性检查通过，但不代表适合交易"
            if tradability.get("basic_execution_feasible")
            else "存在基础执行限制"
        )
        lines.append("可交易性：" + status_text + ("；" + "；".join(tradability_parts) if tradability_parts else "") + "。")

    provenance = payload.get("data_provenance")
    history = provenance.get("history") if isinstance(provenance, dict) else {}
    if isinstance(history, dict):
        quality_parts: list[str] = []
        actual_range = history.get("actual_range")
        if isinstance(actual_range, list) and len(actual_range) >= 2:
            quality_parts.append(f"完整日线实际覆盖 {actual_range[0]} 至 {actual_range[1]}")
        coverage = history.get("session_coverage")
        coverage_minimum = _number(coverage.get("minimum")) if isinstance(coverage, dict) else None
        if coverage_minimum is not None:
            quality_parts.append(f"远端日历可核验区间内交易日覆盖率最低 {coverage_minimum * 100:.0f}%")
        reliability = structure.get("evidence_reliability")
        reliability_label = _text(reliability.get("label")) if isinstance(reliability, dict) else ""
        if reliability_label:
            quality_parts.append(reliability_label)
        if quality_parts:
            lines.append("数据质量：" + "；".join(quality_parts) + "。")

    already_shown = " ".join(lines)
    extra_risks: list[str] = []
    for risk in _items(payload.get("risks"), 8):
        if risk in already_shown:
            continue
        if "波动率" in risk and volatility:
            continue
        if "回撤" in risk and drawdown_number is not None:
            continue
        if "基本面" in risk and isinstance(fundamentals, dict):
            continue
        if any(fragment in risk for fragment in counter):
            continue
        extra_risks.append(risk)
    if extra_risks:
        lines.append("其他风险：" + "；".join(extra_risks[:5]) + "。")
    plain_summary = _text(buy_decision.get("plain_language_summary"))
    if plain_summary:
        lines.append("收束总结：" + plain_summary)
    lines.append(
        f"最终落点：{decision_label}。这是程序按固定量化门槛形成的研究建议，不是收益概率、"
        "目标价、收益承诺或自动交易指令，也不产生自动预测资格。"
    )
    return "\n\n".join(lines)


def _scope_name(payload: dict[str, Any]) -> str:
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        return "当前请求"
    return _text(scope.get("canonical_name") or scope.get("scope_name") or "当前请求")


def _candidate_line(candidate: dict[str, Any], role: str) -> list[str]:
    name = _text(candidate.get("name")) or "未命名股票"
    code = _text(candidate.get("ts_code"))
    identity = f"{name}（{code}）" if code else name
    score = candidate.get("ranking_score_0_100")
    confidence = candidate.get("confidence")
    score_text = f"，综合排名分 {float(score):.2f}/100" if isinstance(score, (int, float)) else ""
    confidence_text = (
        f"，证据完整度约 {float(confidence) * 100:.0f}%"
        if isinstance(confidence, (int, float))
        else ""
    )
    lines = [
        f"{role}：{identity}{score_text}{confidence_text}。排名分和证据完整度都不是上涨概率。"
    ]
    positives = _items(candidate.get("positive_evidence"), 4)
    risks = _items(candidate.get("risks"), 3)
    unmet = _items(candidate.get("unmet_conditions"), 3)
    if positives:
        lines.append("主要依据：" + "；".join(positives) + "。")
    if risks or unmet:
        lines.append("主要风险或未满足条件：" + "；".join([*risks, *unmet][:5]) + "。")
    risk_price = candidate.get("risk_reference_price")
    if isinstance(risk_price, (int, float)):
        lines.append(f"形态风险参考价：{float(risk_price):.2f} 元。")
    return lines


def goujian_fenxi_anquan_huitui(payload: dict[str, Any] | None) -> str:
    """根据分析业务状态生成不含内部参数的中文安全回答。"""
    if not isinstance(payload, dict):
        return "本轮回答出现了异常语言内容，已阻止展示。请重试一次。"
    status = _text(payload.get("status"))
    if status == "clarification_required":
        lines = [_text(payload.get("error")) or "实时数据源里有多个可能范围，需要你确认一次。"]
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            labels = [_text(item.get("user_label")) for item in candidates if isinstance(item, dict)]
            labels = [value for value in labels if value]
            if labels:
                lines.append("可选范围：" + "；".join(labels[:4]) + "。")
        lines.append("请选择最符合原意的一项，也可以换一种日常说法；不需要了解专业分类或内部代码。")
        return "\n\n".join(lines)
    if status == "unavailable":
        reason = _text(payload.get("error")) or "远端市场数据当前不可用"
        action = _text(payload.get("next_action")) or "请稍后重试"
        return f"这次没有完成量化分析：{reason}。\n\n{action}。程序没有使用旧本地市场数据冒充当前结果。"
    if status == "insufficient_data":
        reason = _text(payload.get("error")) or "当前完整数据不足"
        action = _text(payload.get("next_action")) or "补齐完整数据后再试"
        return f"这次没有形成可靠的量化结论：{reason}。\n\n{action}。程序没有把信息不足伪装成中性或成功。"
    if status in {"error", "failed"}:
        return f"量化分析没有完成：{_text(payload.get('error')) or '程序发生内部错误'}。"
    if payload.get("analysis_type") == "single_stock_analysis":
        return _single_stock_fallback(payload)
    scope = _scope_name(payload)
    as_of = _text(payload.get("as_of") or payload.get("generated_at"))
    if payload.get("recommendation_available") is not True:
        reason = _text(payload.get("no_recommendation_reason")) or "当前没有候选同时达到排名分和可信度门槛"
        suffix = f"，数据截至 {as_of}" if as_of else ""
        return f"按数据源当前的“{scope}”范围分析{suffix}，这次不建议勉强选股：{reason}。"
    primary = payload.get("primary")
    if not isinstance(primary, dict):
        return "量化分析标记为完成，但没有返回可展示的首选；本轮不生成推荐。"
    time_text = f"，数据截至 {as_of}" if as_of else ""
    lines = [f"按数据源当前的“{scope}”范围完成量化分析{time_text}。"]
    lines.extend(_candidate_line(primary, "首选"))
    alternatives = payload.get("alternatives")
    if isinstance(alternatives, list):
        for index, candidate in enumerate(alternatives[:4], start=1):
            if isinstance(candidate, dict):
                lines.extend(_candidate_line(candidate, f"备选 {index}")[:1])
    lines.append("预测需要重新下载远端数据并训练模型，耗时明显更长；分析结果展示后再由你决定是否继续。")
    return "\n\n".join(lines)


__all__ = ["goujian_fenxi_anquan_huitui"]
