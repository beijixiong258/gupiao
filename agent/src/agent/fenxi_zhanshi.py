"""完整原始证据报告：消费计算层推荐指数与原始证据，不重算或生成买卖结论。"""

from __future__ import annotations

import math
import re
from typing import Any

from src.ashare.yinzi_gongcheng import factor_definition


# 展示词典只翻译已经取得的字段；不参与选股范围理解、指标计算或条件判断。
_LABELS = {
    "status": "数据状态", "outcome": "结果状态", "reason": "原因", "error": "错误原因",
    "ownership_analysis": "持仓分析（通达信控盘指标）", "raw_value": "来源指标原值",
    "pe_static": "静态市盈率（来源年报口径）", "valuation_close": "估值来源日终收盘价",
    "operating_cashflow_to_revenue_raw_ratio": "经营现金净流量相对营业收入原始比值",
    "operating_cashflow_to_revenue_pct": "经营现金净流量相对营业收入（%）", "cashflow_definition": "现金流比值口径",
    "unit_verified": "指标单位是否核实", "formula_known": "指标公式是否核实", "denominator_known": "指标分母是否核实",
    "realtime_verified": "指标实时性是否核验", "date_verified": "来源观测日是否核验",
    "provider_server_time": "来源服务器时钟（非指标更新时间）", "provider_updated_at": "来源指标实际更新时间",
    "source_data": "控盘来源原始证据", "requested_at": "请求发起时间", "fetched_at": "实际取得时间",
    "stock_code": "证券代码", "series": "来源逐日序列", "changes": "连续交易间隔变化",
    "sample_range": "本次观测区间", "price_context": "同日价格与成交对照",
    "price_return": "同窗口股价收益", "control_change": "同窗口控盘原值变化", "window_comparisons": "同窗口价量对照",
    "direction_relationship": "股价与控盘变化关系", "date_aligned": "控盘与正式日线日期是否对齐",
    "range_position": "样本区间位置", "minimum": "本次样本最小原值", "maximum": "本次样本最大原值",
    "sample_count": "实际观测数", "volume_ratio": "成交量之比（倍）", "amount_yuan_ratio": "成交额之比（倍）",
    "all_zero_unverified": "全零含义尚未核实", "calendar_dates": "本次权威交易日历片段",
    "calendar_source": "交易日历来源", "observation_count": "来源观测数", "upstream_reference": "协议参考来源",
    "interpretation_limits": "可解释范围与限制", "from_date": "窗口起日", "to_date": "窗口止日",
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
    "recommendation_index_analysis": "上涨概率、条件涨幅与期望净收益明细", "raw_values": "计算原始值",
    "percentiles": "共同有效池中的名次分位", "price_group": "价格组分P（0–1）",
    "volume_group": "量价组分V（0–1）", "pool_size": "共同有效比较股票数N",
    "equivalent_index_count": "期望净收益精确相同的并列对象数",
    "recommendation_index_ready": "推荐指数有效数量", "recommendation_index_unavailable": "推荐指数证据不足数量",
    "recommendation_index_status": "推荐指数计算状态", "recommendation_index_missing_fields": "推荐指数缺失原始量",
    "up_probability": "上涨概率估计", "conditional_upside_mean": "上涨条件下平均涨幅估计",
    "conditional_downside_mean": "非上涨条件下平均亏损幅度估计", "expected_gross_return": "期望毛参考收益",
    "expected_net_return": "期望净参考收益", "composite_score_bps": "综合评分（期望净收益，基点）",
    "current_state": "当前固定价量状态", "state_key": "价量状态标识", "statistics": "同状态成熟事件统计",
    "candidate_estimates": "逐股同状态估计", "state_statistics": "各固定状态历史统计",
    "known_count": "结果已知的成熟事件数", "unknown_count": "结果未知的成熟事件数",
    "up_count": "已知事件中上涨次数", "non_up_count": "已知事件中非上涨次数",
    "date_count": "成熟事件涉及信号日期数", "known_signal_dates": "有已知结果的不同信号日期数",
    "sample_support": "样本支持状态", "probability_missing_bounds": "历史未知标签所对应频率可能范围（非未来区间）",
    "uncertainty": "按信号日期考虑相关性的估计不确定性", "date_summaries": "逐信号日期统计",
    "event_counts": "参考事件状态计数", "signal_events": "完整历史参考事件",
    "observed_summary": "历史已知样本描述（工程支持不足时不可用于推荐）",
    "signal_dates": "涉及信号日期数", "signal_start": "成熟信号起日", "signal_end": "成熟信号止日",
    "known_signal_start": "已知结果信号起日", "known_signal_end": "已知结果信号止日",
    "unknown_reason_counts": "未知结果原因计数", "execution_limitation_counts": "参考成交限制计数",
    "minimum_distinct_known_signal_dates": "公开估计所需不同已知信号日期数（工程门槛）",
    "supported": "工程样本支持是否满足", "reference_coverage": "参考结果覆盖率", "return_coverage": "结果覆盖率",
    "up_probability_95_interval": "上涨频率历史样本探索区间（非未来概率保证）",
    "expected_net_return_95_interval": "期望净收益历史样本探索区间（非未来收益范围）",
    "chronological_subperiods": "按时间顺序划分的四个子期", "historical_events": "成熟历史事件数",
    "reference_known_events": "参考结果已知事件数", "reference_unknown_events": "参考结果未知事件数",
    "current_state_valid": "当前价量状态有效股数", "estimate_available": "同状态估计可用股数",
    "estimate_unavailable": "同状态估计不可用股数", "historical_state_missing_events": "历史状态输入不足事件数",
    "scheduled_signal_dates": "计划信号日期数", "mature_signal_dates": "已成熟信号日期数",
    "immature_signal_dates": "尚未成熟信号日期数", "roundtrip_reference_cost_rate": "参考往返损耗比例",
    "notional_yuan": "参考名义金额（元）", "actual_fill_or_broker_cost_verified": "实际成交与券商费用是否核实",
    "historical_identity_or_limit_rules_verified": "历史证券身份与涨跌停规则是否核实",
    "cost_assumption": "参考成本假设", "methodology": "固定状态与统计口径", "coverage": "输入及事件覆盖",
    "expected_gross_return_difference": "本股减下一候选的期望毛收益差",
    "composite_score_bps_difference": "本股减下一候选的期望净收益综合分差（基点）",
    "probability_estimation": "概率与期望收益样本口径",
    "reference_only": "仅为参考价格统计", "execution_verified": "实际成交是否核验",
    "future_probability_calibrated": "未来概率是否经过校准验证",
    "historical_pool": "历史统计样本范围（当前存续且当前条件有效，非历史完整市场）",
    "expected_net_return_difference": "本股减下一候选的期望净收益差",
    "up_probability_difference": "本股减下一候选的上涨概率差",
    "conditional_upside_mean_difference": "本股减下一候选的上涨条件均幅差",
    "conditional_downside_mean_difference": "本股减下一候选的非上涨条件均亏差",
    "direction": "最近方向组合", "position": "相对均线位置", "amount_change": "成交额变化状态",
    "atr14": "14日真实波幅（元）",
    "atr_distance": "距MA20的ATR倍数", "amount_ratio_5_15": "最近5日与此前15日均成交额之比",
    "missing_reasons": "无法计算的原因",
    "current_index": "本股期望净收益综合分（基点）", "next_index": "下一候选期望净收益综合分（基点）", "index_difference": "期望净收益综合分差（基点）",
    "current_percentiles": "本股原始量名次分位", "next_percentiles": "下一候选原始量名次分位",
    "current_groups": "本股价格与量价组分", "next_groups": "下一候选价格与量价组分",
    "raw_advantages": "本股领先的原始量", "raw_tradeoffs": "本股落后的原始量",
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
    "evidence_context": "现有证据详解", "evidence_conflicts": "具体证据分歧", "interpretation_basis": "判断依据与解释边界",
    "contexts": "对应证据维度",
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
    "up_up": "近5日上涨且最近1日上涨", "up_nonup": "近5日上涨但最近1日非上涨",
    "nonup_up": "近5日非上涨但最近1日上涨", "nonup_nonup": "近5日和最近1日均非上涨",
    "at_or_below_ma20": "价格不高于MA20", "above_ma20_within_2_atr": "高于MA20且不超过2ATR",
    "above_ma20_over_2_atr": "高于MA20超过2ATR", "increased": "近期成交额增加", "not_increased": "近期成交额未增加",
    "basis_points": "基点", "next_session_open_to_fifth_session_close": "次交易日开盘至第5交易日收盘",
    "current_request_currently_eligible_stocks_not_reconstructed_historical_full_market": "本次取得的当前存续且当前条件有效样本，未重建历史完整市场",
    "reference_known": "参考价格结果已知", "reference_unknown": "参考价格结果未知",
    "engineering_support_gate_not_return_effectiveness_or_calibration": "只满足工程支持门槛，不代表收益有效或未来概率已校准",
    "circular_signal_date_block_bootstrap": "按信号日期连续分块重抽样（历史样本探索）",
    "exploratory_observed_sample_distribution_not_future_guarantee": "描述已观测样本的探索性不确定性，不保证未来结果",
    "observed_mature_event_unknown_label_bounds_not_future_confidence_interval": "历史成熟事件未知标签的可能范围，不是未来概率区间",
    "expected_net_return_descending_exact_ties": "未舍入期望净收益降序，精确同值并列",
    "fewer_than_ten_distinct_known_signal_dates": "已知结果不足10个不同信号日期，尚不支持公开估计",
    "probability_interval_degenerate_boundary_all_known_labels_equal": "已知结果全涨或全非涨，重抽样概率区间退化，不发布零宽区间",
    "probability_interval_degenerate_empirical_resamples": "历史重抽样上涨频率无变化，探索区间不可用",
    "net_return_interval_degenerate_empirical_resamples": "历史重抽样净收益无变化，探索区间不可用",
    "lt": "低于（不含上限）", "le": "不超过（含上限）", "not_evaluated": "尚未核验",
    "ok": "可用", "success": "可用", "partial": "部分可用", "unavailable": "不可用", "insufficient_data": "数据不足",
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
_INDEX_SCORE_TERM = re.compile(r"推荐指数(?:的)?(?:评分|得分|分数)?")
_NET_RETURN_SCORE_TERM = re.compile(r"期望净收益(?:的)?(?:综合评分|综合分|得分|分数)|(?:综合评分|综合分)(?=（期望净收益[，、]基点）)|(?:综合评分|综合分)(?=[^。！？\n]*(?:为|等于|=)\s*10000\s*[×*]\s*期望净收益)|(?:综合评分|综合分)(?=（基点）|\s*(?:为|是|[:：])?\s*[+-]?\d+(?:\.\d+)?\s*基点)")
_INDEX_WEIGHT_TERM = re.compile(r"(?:固定(?:公式)?(?:组分)?|组分|价格组(?:内部)?|量价组(?:内部)?)(?:的)?权重")
_NEGATED_LEGACY_TERM = re.compile(
    r"(?:不是|并非|不代表|不等于|不作为|不产生|不输出|不发布|不计算|不使用)"
    r"(?:收益率[、，或和与及\s]*)?(?:上涨概率[、，或和与及\s]*)?"
    r"(?:置信度(?:分)?|评分|综合分|得分|分数|权重)"
)
_PERCENTILE_CONTAINERS = {"percentiles", "current_percentiles", "next_percentiles"}
_VALIDATION_TITLES = {
    "assessment": "分项收益结论与输入覆盖", "strategies": "各固定策略的参考收益统计",
    "paired_comparisons": "同日配对比较", "full_pool_missing_impact": "完整池未知收益的敏感性",
    "index_group_analysis": "固定五组的收益梯度与覆盖", "four_time_subperiods": "四个时间子期",
    "execution_counts": "成交假设与未知事件数量", "date_results": "逐信号日原始统计与事件",
    "limitations": "研究限制",
}
_VALIDATION_LABELS = {
    "mathematical_sorting": "数学排序核验", "input_coverage": "输入及信号池覆盖",
    "future_reference_coverage": "后续参考收益覆盖", "signal_codes": "本组信号日完整成员代码",
    "signal_valid_stock_events": "信号池有效股票事件数", "minimum_signal_pool_size": "最小信号池股票数",
    "full_pool_known_dates": "完整池参考收益全已知日期数",
    "reference_known_stock_events": "参照池收益已知股票事件数", "reference_unknown_stock_events": "参照池收益未知股票事件数",
    "paired_comparisons": "同日配对比较", "strategies": "固定策略统计",
    "absolute_reference_return": "绝对参考收益", "relative_reference_returns": "相对参考收益",
    "index_group_gradient": "指数分组梯度", "index": "推荐指数TopN", "old_rule": "旧20／5日规则TopN",
    "ret20_only": "仅20日收益TopN", "ret5_only": "仅5日收益TopN",
    "reference_full_pool": "严格完整池参照", "reference_available_subset": "可计算子集参照",
    "index_vs_old_rule": "指数减旧规则", "index_vs_full_pool": "指数减严格完整池",
    "index_vs_available_subset": "指数减可计算子集", "index_vs_ret20_only": "指数减20日单项",
    "index_vs_ret5_only": "指数减5日单项", "signal_dates": "信号日期数", "known_dates": "收益已知日期数",
    "unknown_dates": "收益未知日期数", "paired_dates": "同日配对数", "unpaired_dates": "未能配对日期数",
    "mean_reference_return": "平均参考收益", "median_reference_return": "参考收益中位数",
    "positive_reference_date_fraction": "正参考收益日期占比", "mean_paired_difference": "同日配对平均收益差",
    "positive_difference_date_fraction": "正收益差日期占比", "bootstrap_95_interval": "分块重采样95%收益差区间",
    "paired_reference_coverage": "配对日期参照覆盖", "zero_return_comparison": "与零参考收益的比较",
    "mean_full_pool_lower_bound": "完整池均值下界L", "upper_bound_mean_index_minus_full_pool": "指数减完整池的平均超额上界",
    "mean_index_return": "同日指数平均参考收益", "mean_known_return_mass": "已知收益对完整池的平均贡献K",
    "mean_missing_weight": "完整池平均未知权重w", "missing_weighted_mean_return_to_tie": "使完整池与指数持平的临界未知加权收益",
    "threshold_relation": "临界值的比较含义", "unknown_stock_events": "收益未知股票事件数",
    "known_stock_events": "收益已知股票事件数", "groups": "固定分组", "available_subset": "本组可计算子集",
    "complete_group": "严格完整组", "common_date_available_subset": "五组共同可计算日期的子集统计",
    "minimum_date_coverage": "逐日最低覆盖率", "mean_date_coverage": "逐日平均覆盖率",
    "common_available_group_dates": "五组共同可计算日期数",
    "aggregate_common_date_means_nondecreasing": "共同日期各组平均收益是否依次不降",
    "aggregate_common_date_means_strictly_increasing": "共同日期各组平均收益是否严格递增",
    "nondecreasing_date_fraction": "逐组收益依次不降的日期占比",
    "highest_vs_lowest_available_subset": "最高组减最低组的同日子集比较",
}
_VALIDATION_STATES = {
    "not_reassessed_here": "本研究未重新核验数学排序", "positive_sample_mean": "已知样本平均参考收益为正",
    "non_positive_sample_mean": "已知样本未显示正的平均参考收益",
    "exploratory_supported": "同日配对存在探索性支持", "inconclusive": "当前配对结果不能确定相对优势",
    "not_supported": "未显示正的配对收益差（区间非正）",
    "fully_observed": "同日完整池收益均已知", "available": "可计算",
    "index_cannot_outperform_even_at_missing_lower_bound": "未知取最低收益仍无法领先完整池",
    "index_can_at_most_tie_at_missing_lower_bound": "未知取最低收益至多打平完整池",
    "unresolved_without_missing_upper_bound": "缺失仍可能改变结论，不能确认领先",
    "not_applicable_no_unknown_returns": "没有未知收益，不适用临界值",
    "below_long_only_lower_bound": "临界未知收益低于纯多头最低收益",
    "at_long_only_lower_bound": "临界未知收益位于纯多头最低收益",
    "above_long_only_lower_bound": "临界未知收益高于纯多头最低收益",
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def qingli_jiu_pingfen_biaoshu(content: str) -> str:
    """剔除旧评价体系；保护明确绑定期望净收益基点的综合分与否定声明。"""
    parts = re.split(r"(?<=[。！？\n])", str(content or ""))
    retained: list[str] = []
    for part in parts:
        checked = _NET_RETURN_SCORE_TERM.sub("", _NEGATED_LEGACY_TERM.sub("", part))
        if "推荐指数" in part or "固定公式" in part:
            checked = _INDEX_SCORE_TERM.sub("", checked)
            checked = _INDEX_WEIGHT_TERM.sub("", checked)
        if not _LEGACY_TEXT.search(checked):
            retained.append(part)
    return "".join(retained).strip()


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
    if field in {"raw_advantages", "raw_tradeoffs"}:
        return _label(text)
    return _STATUS.get(text, text) or "未提供"


def _tree(value: Any, prefix: str = "", metric_key: str | None = None, *, value_kind: str = "observation") -> list[str]:
    """完整展开已返回结构，不截断列表；仅屏蔽已取消的评价字段。"""
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            key = str(key)
            if key in _HIDDEN_KEYS or key.startswith("_") or key in {"field_metadata", "metric_definitions", "metric_key"}:
                continue
            path = f"{prefix} / {_label(key)}" if prefix else _label(key)
            child_kind = (
                "percentile" if key in _PERCENTILE_CONTAINERS
                else "historical_frequency_bound" if key == "probability_missing_bounds"
                else "sample_probability_interval" if key == "up_probability_95_interval"
                else "sample_return_interval" if key == "expected_net_return_95_interval"
                else "fraction_metadata" if key in {"return_coverage", "reference_coverage", "roundtrip_reference_cost_rate"}
                else "coverage" if key == "feature_coverage"
                else "metadata" if key in _METADATA_CONTAINERS
                else value_kind
            )
            observation_key = key
            if key in {"raw_value", "minimum", "maximum"} and str(value.get("metric_key", "")).startswith("tdx_control_"):
                observation_key = "tdx_control_raw"
            elif key in {"value", "range_position"} and value.get("metric_key"):
                observation_key = str(value["metric_key"])
            elif key in {"price_return", "control_change"} and value.get("window") in {1, 5, 20}:
                observation_key = f"ret_{value['window']}" if key == "price_return" else f"tdx_control_change_{value['window']}d"
            elif key in {"recommendation_index", "current_index", "next_index", "index_difference"}:
                observation_key = "composite_score_bps"
            lines.extend(_tree(child, path, observation_key, value_kind=child_kind))
        return lines or ([f"- {prefix}：未返回可用字段。"] if prefix else [])
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"- {prefix}：未列出项目。"]
        lines = []
        for index, child in enumerate(value, start=1):
            lines.extend(_tree(child, f"{prefix} {index}", metric_key, value_kind=value_kind))
        return lines
    if value_kind in {"historical_frequency_bound", "sample_probability_interval", "sample_return_interval", "fraction_metadata"}:
        displayed = value * 100 if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        return [f"- {prefix}（%）：{_value(displayed)}"]
    if value_kind == "coverage":
        # 覆盖率与指标共享字段标识，但不是指标数值，不能套用收益、波动率或分位的单位与释义。
        displayed = value * 100 if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        return [f"- {prefix} / 有效观测覆盖率（%）：{_value(displayed)}"]
    if value_kind == "percentile":
        # 此处是名次分位，不是原指标值；不能乘收益显示倍数或沿用金额等单位。
        displayed = value * 100 if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        return [f"- {prefix} / 名次百分位（%）：{_value(displayed)}"]
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
    estimate_fields = {key: candidate[key] for key in (
        "up_probability", "conditional_upside_mean", "conditional_downside_mean",
        "expected_gross_return", "expected_net_return", "composite_score_bps",
    ) if key in candidate}
    if estimate_fields:
        sections.append((identity + " / 上涨概率、条件涨幅与综合次序", "\n".join(_tree(estimate_fields))))
    summary = candidate.get("diagnosis_summary")
    if candidate.get("research_question"):
        sections.append((identity + " / 用户观点核验", "\n".join(_tree(candidate["research_question"]))))
    elif candidate.get("research_assessment"):
        sections.append((identity + " / 用户观点核验", "\n".join(_tree(candidate["research_assessment"]))))
    if isinstance(summary, dict):
        sections.append((identity + " / 综合判断、证据与复评条件", "\n".join(_tree({
            key: value for key, value in summary.items() if key not in {"research_assessment", "evidence_context"}
        }))))
        evidence_context = summary.get("evidence_context")
        if isinstance(evidence_context, dict):
            details = []
            for dimension, facts in evidence_context.items():
                # 只展开摘要已整理的事实，不在展示层计算指标或补造方向判断。
                if isinstance(facts, (list, tuple)) and all(isinstance(fact, str) for fact in facts):
                    body = "\n".join(f"- {_text(fact)}" for fact in facts) or "本次未返回该维度的可用说明。"
                else:
                    body = "\n".join(_tree(facts))
                details.append(f"#### {_text(dimension)}\n\n{body}")
            if details:
                sections.append((identity + " / 现有证据详解", "\n\n".join(details)))
        elif evidence_context is not None:
            sections.append((identity + " / 现有证据详解", "\n".join(_tree(evidence_context))))
    if "selection_analysis" in candidate:
        sections.append((identity + " / 选股条件与比较依据", "\n".join(_tree(candidate["selection_analysis"]))))
    for key, title in (
        ("stock", "股票身份"), ("technical_summary", "技术指标与结构"),
        ("market_cap_check", "市值条件与原始值"),
        ("fundamental_analysis", "公司基本面与估值"),
        ("ownership_analysis", "持仓分析（通达信控盘指标）"),
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
            if key == "ownership_analysis" and isinstance(candidate[key], dict) and candidate[key].get("metric_definitions"):
                sections.append((identity + " / 控盘指标计算口径", "\n".join(_tree(candidate[key]["metric_definitions"], value_kind="metadata"))))
    if "daily_factor_analysis" in candidate:
        sections.extend((identity + " / " + title, body) for title, body in _factor_sections(candidate["daily_factor_analysis"]))
    return sections


def _report_sections(payload: dict[str, Any]) -> list[tuple[str, str]]:
    if payload.get("analysis_type") == "recommendation_index_validation":
        return _validation_sections(payload)
    if payload.get("analysis_type") == "single_stock_analysis":
        stock = payload.get("stock") or payload.get("selected_stock") or {}
        identity = _identity(stock if isinstance(stock, dict) else {}, _text(payload.get("query")) or "当前股票")
        return _candidate_sections(payload, identity, single=True)
    sections: list[tuple[str, str]] = []
    for key, title in (("scope", "选股范围及核验"), ("candidate_counts", "本次筛选与复核数量"),
                       ("market_cap_filter", "市值筛选条件及核验"),
                       ("user_conditions", "用户明确选股条件"), ("comparison_coverage", "实际比较覆盖"),
                       ("selection_limits", "复核范围与停止原因"), ("selection_outcome", "筛选结果"),
                       ("selection_methodology", "选股方法"), ("probability_estimation", "概率与期望收益样本口径"), ("diagnosis_validity", "诊断时点"),
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


def _validation_tree(value: Any, prefix: str = "") -> list[str]:
    """原样展开研究统计与覆盖；不套用技术指标单位、不重算或删除未知项。"""
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            label = _VALIDATION_LABELS.get(str(key), _label(str(key)))
            path = f"{prefix} / {label}" if prefix else label
            if key == "signal_codes" and isinstance(child, (list, tuple)) and child:
                lines.append(f"- {path}：完整成员在结构化逐日结果中可访问。")
                continue
            lines.extend(_validation_tree(child, path))
        return lines or [f"- {prefix}：未返回字段。"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"- {prefix}：未列出项目。"]
        return [line for index, child in enumerate(value, start=1)
                for line in _validation_tree(child, f"{prefix} {index}")]
    text = _VALIDATION_STATES.get(value, _value(value)) if isinstance(value, str) else _value(value)
    return [f"- {prefix}：{text}"]


def _validation_sections(payload: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = {key: value for key, value in payload.items()
                if key not in _VALIDATION_TITLES and key != "analysis_type"}
    sections = [("历史研究口径与输出覆盖", "收益和收益差均保留计算层小数比例：0.01分别表示1%收益、1个百分点收益差；"
                 "覆盖率和日期占比也以0–1保存。收益差不是推荐指数点。顶层数据状态只表示输出覆盖，收益判断见各分项。\n\n"
                 + "\n".join(_validation_tree(metadata)))]
    for key, title in _VALIDATION_TITLES.items():
        if key not in payload:
            continue
        value = payload[key]
        if key == "assessment" and isinstance(value, dict):
            # 同一返回结构同时提供便捷分项和独立章节时，只展开一次完整统计。
            repeated = {"relative_reference_returns": "paired_comparisons",
                        "full_pool_missing_impact": "full_pool_missing_impact", "index_group_gradient": "index_group_analysis"}
            value = {field: child for field, child in value.items()
                     if field not in repeated or repeated[field] not in payload or payload[repeated[field]] != child}
        sections.append((title, "\n".join(_validation_tree(value))))
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
    validation = payload.get("analysis_type") == "recommendation_index_validation"
    if payload.get("status") not in {"ok", "partial"} and not error_with_evidence and not validation:
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
    if payload.get("analysis_type") == "recommendation_index_validation":
        assessment = payload.get("assessment")
        absolute = assessment.get("absolute_reference_return") if isinstance(assessment, dict) else None
        summary = _text(absolute.get("summary")) if isinstance(absolute, dict) else ""
        lead = f"以下为固定推荐指数的历史参考研究，输出覆盖状态：{_value(status)}。" + summary
        return buquan_zhengju_baogao(lead, payload)
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
        lead = "以下为通过用户条件和数据核验的候选：按同状态历史事件估计的未舍入期望净参考收益降序排列。上涨概率与上涨条件均幅分别展示，期望净收益综合分仅换算为基点；同状态可并列，估计不代表已知未来结果。样本、日期、成本及缺口见报告。"
        if status == "partial":
            lead += "部分来源或补充证据缺失，报告保留具体缺口和实际复核范围。"
    else:
        lead = _text(payload.get("no_recommendation_reason")) or "本次没有候选通过全部选股条件。"
    return buquan_zhengju_baogao(lead, payload)


# 原补充诊断入口继续指向完整报告，避免调用方只保护七个补充块。
buquan_buchong_zhenduan = buquan_zhengju_baogao

__all__ = ["buquan_zhengju_baogao", "buquan_buchong_zhenduan", "goujian_fenxi_anquan_huitui", "qingli_jiu_pingfen_biaoshu"]
