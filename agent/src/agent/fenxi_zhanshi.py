"""完整原始证据报告：共享展示与漏项补全，不生成评分或买卖结论。"""

from __future__ import annotations

import math
import re
from typing import Any

from src.ashare.yinzi_gongcheng import factor_definition


# 展示词典只翻译已经取得的字段；不参与选股范围理解、指标计算或条件判断。
_LABELS = {
    "status": "数据状态", "outcome": "结果状态", "reason": "原因", "error": "错误原因",
    "name": "名称", "ts_code": "证券代码", "code": "代码", "industry": "行业", "market": "市场",
    "trade_date": "交易日", "as_of": "数据截止日", "generated_at": "分析生成时间", "captured_at": "取得时间",
    "open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价", "pre_close": "前收盘价",
    "volume": "成交量", "vol": "成交量", "amount_yuan": "成交额（元）", "amount": "源成交额",
    "returns": "区间收益", "moving_averages": "移动平均线", "rsi_14": "14日相对强弱指标",
    "macd": "指数平滑异同移动平均线", "dif": "快线", "dea": "慢线", "histogram": "柱值",
    "dif_pct": "快线相对价格比例", "dea_pct": "慢线相对价格比例", "gap_pct": "双线间距比例",
    "zero_distance_pct": "零轴距离比例", "macd_structure": "MACD结构证据", "structure_classification": "结构归类",
    "atr_14_pct": "14日真实波幅相对价格比例", "annualized_volatility_20": "20日年化波动率（比例）",
    "drawdown_from_20d_high": "距20日高点回撤（比例）", "position_in_20d_range": "20日价格区间位置（比例）",
    "volume_ratio_5_to_20": "5日与20日平均成交量之比", "support_20": "20日支撑参考价", "resistance_20": "20日阻力参考价",
    "evidence": "观察依据", "supporting_evidence": "支持证据", "counter_evidence": "反向证据", "risk_warnings": "风险提示",
    "indicator_warnings": "指标缺口", "warnings": "注意事项", "risks": "风险", "errors": "来源异常",
    "summary": "综合解释", "reassessment_conditions": "重新评估条件", "evidence_gaps": "证据缺口",
    "missing_reason": "缺失原因", "missing_fields": "缺失字段", "missing_conditions": "无法核验条件",
    "unmet_conditions": "未满足条件", "failure_reasons": "未形成原因", "unavailable_items": "不可用项",
    "state": "形态状态", "state_label": "形态说明", "eligible": "条件是否满足", "stage": "时段", "stage_label": "时段说明",
    "confirmation_level": "确认状态", "conditions": "条件核验", "actuals": "实际观测值", "metrics": "原始指标",
    "baseline_date": "基准日", "shrink_date": "缩量日", "breakout_date": "突破日", "risk_reference_price": "风险参考价格",
    "rule_semantics": "形态规则口径", "session_origin": "交易日计数起点",
    "shrink_window_start_session": "缩量窗口起始交易日序号", "shrink_window_end_session": "缩量窗口截止交易日序号",
    "shrink_window_sessions": "缩量窗口交易日数", "breakout_deadline_sessions": "突破截止交易日序号",
    "breakout_deadline_session": "突破截止交易日序号", "breakout_after_shrink_required": "突破是否必须晚于缩量",
    "breakout_deadline_inclusive": "截止交易日是否包含在内",
    "shrink_session_after_baseline": "缩量发生在基准日后第几个交易日",
    "breakout_session_after_baseline": "突破发生在基准日后第几个交易日",
    "profile": "公司资料", "valuation": "估值数据", "financials": "财务数据", "sources": "数据来源",
    "pe_ttm": "滚动市盈率（倍）", "pe_dynamic": "动态市盈率（倍）", "pe": "市盈率（总市值/净利润，倍）", "pe_definition": "市盈率来源口径",
    "pe_unspecified": "市盈率（来源未明确口径，倍）", "pb": "市净率（倍）",
    "report_date": "报告期", "announcement_date": "公告日", "known_as_of": "财务已知数据截止日",
    "roe_pct": "净资产收益率（%）", "gross_margin_pct": "毛利率（%）", "net_margin_pct": "净利率（%）",
    "debt_to_assets_pct": "资产负债率（%）", "revenue_yoy_pct": "营收同比（%）", "net_profit_yoy_pct": "净利润同比（%）",
    "operating_cashflow_to_revenue_pct": "经营现金流与营收指标（沿用来源口径）",
    "total_market_value_yuan": "总市值（元）", "circulating_market_value_yuan": "流通市值（元）",
    "turnover_rate_pct": "换手率（%）", "turnover_rate": "换手率（来源单位）", "volume_ratio": "量比（倍）",
    "pe_percentile": "市盈率原始分位", "pb_percentile": "市净率原始分位", "valuation_percentiles": "估值分位",
    "valuation_context": "估值比较背景", "sample_size": "比较样本数量", "available_fields": "字段可用性",
    "valuation_time_verification": "估值时点核验", "valuation_trade_date": "估值对应交易日",
    "valuation_source": "估值来源", "valuation_is_complete_daily": "估值是否确认为完整日终数据",
    "excluded_unverified_or_other_date_count": "因时点未核验或日期不同排除的样本数",
    "last_price": "最新价", "latest_price": "最新价", "pct_change": "涨跌幅（%）", "pct_chg": "涨跌幅（%）",
    "basic_execution_feasible": "本次适用的基础条件是否满足", "hard_blocks": "执行限制", "cautions": "注意事项",
    "realtime_required": "当前是否需要实时行情", "current_quote_verified": "当前行情时效及价量是否已核验",
    "execution_check_scope": "基础条件检查范围", "execution_note": "成交条件解释",
    "current_volume": "快照成交量（股）", "current_amount_yuan": "快照成交额（元）",
    "historical_quote_verified": "历史快照日期及价量是否已核验", "market_session_status": "权威日历确认时段",
    "current_quote_status": "当前行情核验状态",
    "analysis_price": "诊断参考价格", "analysis_price_basis": "价格依据", "amount_basis": "成交额依据",
    "amount_trade_date": "成交额对应交易日", "price_limit_pct": "涨跌幅限制（%）", "data_quality": "数据质量",
    "source": "来源", "history": "历史日线", "history_summary": "日线覆盖概况", "rows": "有效行数",
    "first_trade_date": "首个交易日", "last_trade_date": "最后交易日", "requested_start_date": "请求起始日",
    "requested_end_date": "请求结束日", "actual_range": "实际覆盖区间", "requested_range": "请求区间",
    "session_coverage": "交易日覆盖情况", "minimum": "最小值", "maximum": "最大值", "mean": "均值",
    "latest_daily_bar": "最近完整日线", "resolved_profile": "已核验身份资料", "comparison_profile": "本次比较池资料",
    "comparison_pool": "比较池来源", "factor_panel": "指标面板来源", "realtime_snapshot": "行情快照来源",
    "feature_coverage": "指标有效观测覆盖情况", "source_warnings": "来源降级记录",
    "data_as_of": "证据截止日", "daily_data_as_of": "日线截止日", "explanation": "口径说明", "reassess_when": "何时更新诊断",
    "result_confirmation": "结果确认状态", "session_status": "交易时段", "realtime_status": "实时核验状态",
    "selection_analysis": "选股条件核验", "meets_selection_conditions": "是否通过选股条件",
    "pareto_front": "非支配比较层次", "ranking_basis": "候选展示次序依据", "values": "观测值",
    "label": "说明", "key": "指标标识", "economic_meaning": "经济含义", "unit": "单位", "meaning": "含义",
    "definition": "定义", "window": "观察窗口", "method": "计算口径", "period": "观察周期",
    "canonical_source": "同源指标标识", "input_requirement": "所需数据", "frequency": "数据频率",
    "availability": "可用时点", "group": "证据类别", "role": "指标用途",
    "quote_time_verification": "行情时点核验", "provider_quote_time": "来源更新时间",
    "provider_trade_date": "来源交易日期", "expected_trade_date": "待核验交易日期",
    "quote_age_seconds": "来源更新时间距核验时点（秒）", "verification_scope": "核验范围",
    "current_quote_reason": "当前行情核验说明",
    "scope_input": "输入范围股票数", "after_hard_filter": "基础检查通过数", "after_prefilter": "分批请求候选数",
    "after_market_cap_filter": "市值条件通过数", "market_cap_unverified": "市值尚未核验数",
    "market_cap_check": "市值条件核验", "condition": "指定条件", "basis_label": "市值口径",
    "maximum_yuan": "市值上限（元）", "maximum_yi": "市值上限（亿元）", "actual_yuan": "已核验市值（元）",
    "operator": "比较边界", "time_basis": "时点口径", "source_dates": "来源原始日期",
    "fetched_at": "来源取得时间", "verified_count": "已核验数量", "matched_count": "满足条件数量",
    "unmet_count": "未满足条件数量", "unavailable_count": "证据不足数量",
    "unmet_examples": "未满足条件示例", "unavailable_examples": "证据不足示例", "application_stage": "条件执行阶段",
    "history_ready": "完整日线可用数", "factor_ready": "指标可用数", "after_factor_limit": "进入本地复核范围数",
    "technical_reviewed": "已完成本地技术复核", "deep_reviewed": "已整理完整报告", "qualified": "已确认合格数",
    "unverified": "必需条件尚无法核验数", "displayed": "实际展示候选数", "display_target": "本次展示目标数",
    "maximum_full_reports": "完整报告数量上限", "maximum_technical_reviews": "本地技术复核数量上限",
    "maximum_minute_requests": "分钟请求数量上限", "maximum_alternatives": "后续候选数量上限",
    "review_stop_reason": "复核停止原因", "selection_outcome": "筛选结果",
    "source_date_verified": "来源日期是否有效", "timeliness_status": "行情时效状态", "session_phase": "时段",
    "quote_active_age_seconds": "有效交易时段延迟（秒）", "maximum_active_age_seconds": "有效交易时段延迟上限（秒）",
    "main_judgment": "主要判断", "research_question": "待验证观点", "question": "原始问题",
    "verdict": "核验结论", "verdict_label": "结论说明", "checks": "可观测命题核验",
    "supporting_checks": "支持的命题", "counter_checks": "存在反证的命题", "missing_checks": "缺少证据的命题",
    "unverifiable_parts": "尚不能核验的部分", "observation": "实际观测", "reference": "参照观测",
    "metric": "指标标识", "reference_metric": "参照指标", "value": "原始值", "threshold": "比较界限（原始单位）",
    "display_scale": "原值到展示单位的倍数", "family": "证据来源类别", "evidence_families": "相关证据归组",
    "intraday_effect": "盘中变化对日线判断的影响", "effect": "实际变化", "ranking_effect": "对排序的适用边界",
    "reference_note": "价格比较口径", "ma20_reference": "已完成日线MA20参考值（元）",
    "change_from_previous_close": "较前收盘价变化（比例，1表示100%）", "quote_time": "行情更新时间",
    "comparison_values": "排序原始值", "comparison_rank": "比较次序", "comparison_pool_size": "合格比较样本数",
    "comparison_to_next": "与后一候选的比较", "current_values": "本股原始值", "next_values": "后一候选原始值",
    "next_pareto_front": "后一候选非支配层次", "absolute_performance": "绝对表现",
    "history_requested": "请求历史行情数", "sampled": "是否抽样", "complete": "比较是否完成",
    "technical_review_planned": "本次计划技术复核数", "limitation": "覆盖边界", "best_definition": "本次优选口径",
    "equivalent_performance_count": "两项表现完全相同的并列对象数",
    "size_matched_stocks": "行业与规模均可核验的同行数", "target_size_verified": "本股规模是否核验",
}
_STATUS = {
    "lt": "低于（不含上限）", "le": "不超过（含上限）", "not_evaluated": "尚未核验",
    "ok": "可用", "partial": "部分可用", "unavailable": "不可用", "insufficient_data": "数据不足",
    "error": "程序错误", "program_error": "程序错误", "failed": "未完成", "not_applicable": "当前不适用", "not_used": "未使用",
    "met": "已满足", "unmet": "未满足", "verified": "已核验", "pending": "待确认",
    "analysis_success": "分析完成", "information_partial": "部分证据缺失", "analysis_completed": "分析完成",
    "intraday_provisional": "盘中暂定", "completed_daily_close": "完整日线确认", "close_pending": "收盘待确认",
    "latest_completed_daily_bar": "最近完整交易日", "realtime_snapshot": "实时行情快照", "diagnostic_only": "补充观察",
    "evidence_unavailable": "必需证据不足，尚无法确认是否合格", "no_recommendation": "没有已确认合格候选",
    "recommendation": "已有通过条件的研究候选", "display_target_reached": "已达到本次展示目标",
    "candidate_pool_exhausted": "已检查完本次可复核候选", "stale": "行情已过期", "future": "行情时点超前",
    "historical_reference": "历史日线基础条件通过，仅作研究参考",
    "conditions_verified": "本次行情基础条件已核验，不代表即时成交",
    "historical_daily": "最近完整日线", "current_quote": "当前行情",
    "not_required": "本次不要求", "not_requested": "本次未请求",
    "non_trading_day": "休市", "pre_open": "开市前", "post_close": "收盘后",
    "latest_completed_qfq_close": "最近完整日线前复权收盘价",
    "supported": "支持所列命题", "partly_supported": "部分支持", "not_supported": "存在直接反证",
    "insufficient_evidence": "证据不足", "gt": "大于", "gte": "大于等于", "lte": "小于等于", "eq": "等于",
    "all_scope_members_in_batches": "范围成分分批全量请求",
    "comparison_reference": "比较参照对象，不增加推荐数量",
}
_METADATA_CONTAINERS = {"data_provenance", "data_quality", "sources", "method", "configuration"}
_HIDDEN_KEYS = {
    "analysis_id", "tool_contract_version", "buy_decision", "ranking_details", "ranking_score_0_100",
    "ranking_score_definition", "score", "score_0_100", "score_definition", "score_interpretation",
    "confidence", "component_weights", "factor_group_weights", "risk_penalty", "scoring_effect",
    "independently_scored", "independent_scoring_factor_count", "scored_factor_count", "scoring_source",
    "scoring_exclusions", "oriented_factor_count", "available_oriented_factor_count", "score_summary",
    "weights", "weight", "direction_rule", "market_regime_score",
}
_LEGACY_TEXT = re.compile(r"评分|综合分|技术状态分|技术分|得分|分数|置信度|权重|排名分|适配分|(?:暂不)?建议买入")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def qingli_jiu_pingfen_biaoshu(content: str) -> str:
    """剔除旧评价体系的模型句子；原始证据由完整报告补齐。"""
    parts = re.split(r"(?<=[。！？\n])", str(content or ""))
    return "".join(part for part in parts if not _LEGACY_TEXT.search(part)).strip()


def _label(key: str) -> str:
    definition = factor_definition(key)
    if definition.get("label") and definition["label"] != key:
        return definition["label"]
    if key in _LABELS:
        return _LABELS[key]
    if re.fullmatch(r"ma_?\d+", key):
        return f"{re.sub(r'[^0-9]', '', key)}日均线（元）"
    if re.fullmatch(r"\d+d", key):
        return f"近{key[:-1]}日收益（比例，1表示100%）"
    return key.replace("_", " ")


def _value(value: Any, *, field: str | None = None) -> str:
    if value is None:
        return "缺失（未取得或无法计算）"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "缺失（不是有效有限数）"
        return str(value) if isinstance(value, int) else f"{value:.12g}"
    text = _text(value)
    if field == "freshness":
        return {"fresh": "新近信号", "recent": "近期信号", "stale": "陈旧信号"}.get(text, text) or "未提供"
    if field == "timeliness_status" and text == "not_requested":
        return "本次未核验当前时效"
    if field in {"current_quote_status", "realtime_status"} and text == "not_required":
        return "本次不要求当前行情核验"
    return _STATUS.get(text, text) or "未提供"


def _tree(value: Any, prefix: str = "", metric_key: str | None = None, *, value_kind: str = "observation") -> list[str]:
    """完整展开已返回结构，不截断列表；仅屏蔽已取消的评价字段。"""
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            key = str(key)
            if key in _HIDDEN_KEYS or key.startswith("_") or key in {"field_metadata", "metric_definitions"}:
                continue
            path = f"{prefix} / {_label(key)}" if prefix else _label(key)
            child_kind = "coverage" if key == "feature_coverage" else "metadata" if key in _METADATA_CONTAINERS else value_kind
            lines.extend(_tree(child, path, key, value_kind=child_kind))
        return lines or ([f"- {prefix}：未返回可用字段。"] if prefix else [])
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"- {prefix}：未列出项目。"]
        lines = []
        for index, child in enumerate(value, start=1):
            lines.extend(_tree(child, f"{prefix} {index}", metric_key, value_kind=value_kind))
        return lines
    if value_kind == "coverage":
        # 覆盖率与指标共享字段标识，但不是指标数值，不能套用收益、波动率或分位的单位与释义。
        displayed = value * 100 if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        return [f"- {prefix} / 有效观测覆盖率（%）：{_value(displayed)}"]
    definition = factor_definition(metric_key or "")
    known_metric = bool(value_kind == "observation" and metric_key and definition.get("label") != metric_key)
    displayed = value
    if known_metric and isinstance(value, (int, float)) and not isinstance(value, bool):
        scale = definition.get("display_scale", 1)
        if isinstance(scale, (int, float)):
            displayed = value * scale
    text = _value(displayed, field=metric_key)
    unit = definition.get("unit") if known_metric else None
    meaning = definition.get("meaning") if known_metric else None
    if unit:
        prefix += f"（{unit}）"
    if meaning:
        text += f"；含义：{meaning}"
    return [f"- {prefix}：{text}"]


def _table_cell(value: Any) -> str:
    return _value(value).replace("|", "／").replace("\n", " ")


def _factor_sections(factor: Any) -> list[tuple[str, str]]:
    if not isinstance(factor, dict):
        return [("八组日K证据", "本次未取得可用的日K指标组。")]
    groups = factor.get("groups")
    if not isinstance(groups, dict):
        return [("八组日K证据", "\n".join(_tree(factor)))]
    sections: list[tuple[str, str]] = []
    for key, group in groups.items():
        if not isinstance(group, dict):
            continue
        title = _text(group.get("label")) or _label(str(key))
        meaning = _text(group.get("economic_meaning"))
        values = group.get("values") if isinstance(group.get("values"), dict) else {}
        missing = group.get("missing_fields") or []
        missing_keys = list(missing) if isinstance(missing, (dict, list, tuple)) else []
        metadata = group.get("field_metadata") or group.get("metric_definitions") or {}
        if isinstance(metadata, list):
            metadata = {item.get("feature") or item.get("key"): item for item in metadata if isinstance(item, dict)}
        if not isinstance(metadata, dict):
            metadata = {}
        fields = list(dict.fromkeys([*metadata, *values, *missing_keys]))
        lines = [meaning] if meaning else []
        lines.append(f"已有字段 {group.get('available_factor_count', len(values))} / {group.get('factor_count', len(fields))}；缺失不作为零值。")
        lines.extend(["", "| 指标 | 当前值 | 单位 | 中文含义与缺失原因 |", "| --- | --- | --- | --- |"])
        for field in fields:
            if field in _HIDDEN_KEYS:
                continue
            meta = metadata.get(field) if isinstance(metadata.get(field), dict) else {}
            if meta.get("definition_ref") == field:
                meta = {**factor_definition(field), **meta}
            label = meta.get("label") or meta.get("name") or _label(str(field))
            unit = (meta["unit"] or "无量纲") if "unit" in meta else "原始口径（未标注单位）"
            explanation = meta.get("meaning") or meta.get("description") or meta.get("economic_meaning") or "来源未提供此字段的独立释义"
            for key, explanation_title in (("formula", "公式"), ("window", "窗口"), ("minimum_observations", "所需样本"), ("valid_observations", "本次有效样本数")):
                if meta.get(key) is not None:
                    explanation += f"；{explanation_title}：{meta[key]}"
            if field in missing_keys:
                cause = missing.get(field) if isinstance(missing, dict) else None
                explanation += f"；缺失：{_text(cause) or meta.get('missing_rule') or '所需输入未齐全'}"
            displayed = values.get(field)
            scale = meta.get("display_scale", 1)
            if isinstance(displayed, (int, float)) and not isinstance(displayed, bool) and isinstance(scale, (int, float)):
                displayed *= scale
            lines.append("| " + " | ".join(_table_cell(item) for item in (label, displayed, unit, explanation)) + " |")
        extra = {key: value for key, value in group.items() if key not in {"label", "economic_meaning", "values", "field_metadata", "metric_definitions", "missing_fields", "available_factor_count", "factor_count"}}
        lines.extend(_tree(extra))
        sections.append(("日K证据 / " + title, "\n".join(lines)))
    return sections


def _identity(candidate: dict[str, Any], default: str) -> str:
    name = _text(candidate.get("name"))
    code = _text(candidate.get("ts_code"))
    return f"{name}（{code}）" if name and code else name or code or default


def _candidate_sections(candidate: dict[str, Any], identity: str, *, single: bool) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    summary = candidate.get("diagnosis_summary")
    if candidate.get("research_question"):
        sections.append((identity + " / 用户观点核验", "\n".join(_tree(candidate["research_question"]))))
    if isinstance(summary, dict):
        sections.append((identity + " / 综合判断、证据与复评条件", "\n".join(_tree({
            key: value for key, value in summary.items() if key != "research_assessment"
        }))))
    if "selection_analysis" in candidate:
        sections.append((identity + " / 选股条件与比较依据", "\n".join(_tree(candidate["selection_analysis"]))))
    for key, title in (
        ("stock", "股票身份"), ("technical_summary", "技术指标与结构"),
        ("market_cap_check", "市值条件与原始值"),
        ("fundamental_analysis", "公司基本面与估值"),
        ("limit_up_pullback_pattern", "涨停回马枪形态"), ("late_session_analysis", "尾盘证据"),
        ("supplemental_diagnostics", "补充诊断"), ("tradability", "可交易性与执行限制"),
        ("realtime_snapshot", "本次行情快照"), ("snapshot", "本次行情快照"), ("data_analysis", "原始行情与比较资料"),
        ("evidence_gaps", "证据缺口"), ("risks", "风险"), ("warnings", "注意事项"),
        ("supporting_evidence", "支持证据"), ("positive_evidence", "正向证据"),
        ("counter_evidence", "反向证据"), ("unmet_conditions", "未满足条件"),
        ("reassessment_conditions", "重新评估条件"),
        ("data_quality", "数据质量"), ("diagnosis_validity", "诊断时点与更新条件"),
        ("data_provenance", "来源与核验记录"),
    ):
        if key in candidate:
            kind = "metadata" if key in _METADATA_CONTAINERS else "observation"
            sections.append((identity + " / " + title, "\n".join(_tree(candidate[key], value_kind=kind)) or "当前没有可展示的字段。"))
    if "daily_factor_analysis" in candidate:
        sections.extend((identity + " / " + title, body) for title, body in _factor_sections(candidate["daily_factor_analysis"]))
    return sections


def _report_sections(payload: dict[str, Any]) -> list[tuple[str, str]]:
    if payload.get("analysis_type") == "single_stock_analysis":
        stock = payload.get("stock") or payload.get("selected_stock") or {}
        identity = _identity(stock if isinstance(stock, dict) else {}, _text(payload.get("query")) or "当前股票")
        return _candidate_sections(payload, identity, single=True)
    sections: list[tuple[str, str]] = []
    for key, title in (("scope", "选股范围及核验"), ("candidate_counts", "本次筛选与复核数量"),
                       ("market_cap_filter", "市值筛选条件及核验"),
                       ("user_conditions", "用户明确选股条件"), ("comparison_coverage", "实际比较覆盖"),
                       ("selection_limits", "复核范围与停止原因"), ("selection_outcome", "筛选结果"),
                       ("selection_methodology", "选股方法"), ("diagnosis_validity", "诊断时点"),
                       ("data_provenance", "公共数据来源"), ("filter_summary", "范围过滤记录")):
        if key in payload:
            kind = "metadata" if key in _METADATA_CONTAINERS else "observation"
            sections.append((title, "\n".join(_tree(payload[key], value_kind=kind))))
    targets: list[tuple[dict[str, Any], str]] = []
    if isinstance(payload.get("primary"), dict):
        targets.append((payload["primary"], "研究候选 1"))
    alternatives = payload.get("alternatives")
    if isinstance(alternatives, list):
        targets.extend((candidate, f"研究候选 {index}") for index, candidate in enumerate(alternatives, start=2) if isinstance(candidate, dict))
    if not targets and isinstance(payload.get("diagnostic_candidates"), list):
        targets.extend((candidate, "观察对象（未通过选股条件）") for candidate in payload["diagnostic_candidates"][:1] if isinstance(candidate, dict))
    for candidate, role in targets:
        sections.extend(_candidate_sections(candidate, role + "：" + _identity(candidate, "未命名股票"), single=False))
    return sections


def _render_section(title: str, body: str) -> str:
    return f"### {title}\n\n{body}"


def _single_stock_error_has_evidence(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "error" or payload.get("analysis_type") != "single_stock_analysis":
        return False
    metadata_only = {"status", "outcome", "reason", "error", "error_code", "stage", "warnings", "errors", "source", "sources", "trade_date", "as_of", "generated_at", "captured_at"}
    for key in ("data_analysis", "daily_factor_analysis", "technical_summary", "fundamental_analysis"):
        component = payload.get(key)
        if isinstance(component, dict) and any(
            value is not None and value != "" and value != {} and value != []
            for field, value in component.items() if field not in metadata_only
        ):
            return True
    return False


def _clean_prose_preserving_sections(content: str, sections: list[str]) -> str:
    """只清理模型文字；已由当前载荷生成的完整章节保留原始事实和释义。"""
    if not sections:
        return qingli_jiu_pingfen_biaoshu(content)
    known_sections = set(sections)
    pieces = re.split("(" + "|".join(re.escape(section) for section in sections) + ")", str(content or ""))
    return "\n\n".join(
        cleaned for piece in pieces
        if (cleaned := piece if piece in known_sections else qingli_jiu_pingfen_biaoshu(piece))
    )


def buquan_zhengju_baogao(content: str, payload: dict[str, Any] | None) -> str:
    """按完整章节补回遗漏证据，不用数值关键词验证或更改业务结果。"""
    if not isinstance(payload, dict):
        return qingli_jiu_pingfen_biaoshu(content)
    error_with_evidence = _single_stock_error_has_evidence(payload)
    if payload.get("status") not in {"ok", "partial"} and not error_with_evidence:
        return qingli_jiu_pingfen_biaoshu(content)
    sections = [_render_section(title, body) for title, body in _report_sections(payload)]
    if error_with_evidence:
        reason = _text(payload.get("error") or payload.get("reason")) or "技术计算发生程序错误"
        lead = f"本次分析未完成（程序错误）：{reason}。以下保留已经取得的证据，错误部分需要修复后重新分析。"
        return "\n\n".join([lead, *sections])
    clean = _clean_prose_preserving_sections(content, sections)
    existing = "".join(clean.split())
    missing: list[str] = []
    for section in sections:
        normalized = "".join(section.split())
        if normalized not in existing:
            missing.append(section)
            existing += normalized
    return "\n\n".join([part for part in [clean, *missing] if part]).strip()


def goujian_fenxi_anquan_huitui(payload: dict[str, Any] | None) -> str:
    """来源部分可用时完整展示余下证据，主链失败时保留真实状态。"""
    if not isinstance(payload, dict):
        return "本轮未取得可核验的分析结果。"
    status = payload.get("status")
    if _single_stock_error_has_evidence(payload):
        return buquan_zhengju_baogao("", payload)
    if status not in {"ok", "partial"}:
        reason = _text(payload.get("error") or payload.get("reason")) or "当前缺少完成分析所需的数据"
        action = _text(payload.get("next_action"))
        return f"本次分析未完成：{reason}。" + (f"\n\n{action}" if action else "")
    if payload.get("analysis_type") == "single_stock_analysis":
        summary = payload.get("diagnosis_summary")
        lead = _text(summary.get("summary")) if isinstance(summary, dict) else "已整理该股票本次能够取得的分析证据。"
        if status == "partial":
            lead += "部分来源或指标缺失，以下保留可用证据和具体缺口。"
    elif payload.get("recommendation_available") is True:
        lead = "以下为通过用户条件和数据核验的优选候选：按20日、5日已实现表现分层，同层优先20日表现。具体比较依据与覆盖范围见报告。"
        if status == "partial":
            lead += "部分来源或补充证据缺失，报告保留具体缺口和实际复核范围。"
    else:
        lead = _text(payload.get("no_recommendation_reason")) or "本次没有候选通过全部选股条件。"
    return buquan_zhengju_baogao(lead, payload)


# 原补充诊断入口继续指向完整报告，避免调用方只保护七个补充块。
buquan_buchong_zhenduan = buquan_zhengju_baogao

__all__ = ["buquan_zhengju_baogao", "buquan_buchong_zhenduan", "goujian_fenxi_anquan_huitui", "qingli_jiu_pingfen_biaoshu"]
