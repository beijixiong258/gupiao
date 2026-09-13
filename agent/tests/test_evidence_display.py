"""完整原始证据展示、部分可用状态与规则选股展示的离线回归。"""

from __future__ import annotations

import copy
import json

import pytest

from src.agent.context import ContextBuilder, _ANALYSIS_TOOL_CONTRACT_VERSION
from src.agent.fenxi_zhanshi import buquan_zhengju_baogao, goujian_fenxi_anquan_huitui
from src.agent.loop import AgentLoop, _analysis_run_status
from src.agent.memory import WorkspaceMemory
from src.agent.tools import BaseTool, ToolRegistry
from src.ashare.yinzi_gongcheng import FACTOR_REGISTRY, factor_definition
from src.providers.chat import LLMResponse, ToolCallRequest
from src.tools.gupiao_analysis_state import analysis_session_store
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool


def _single(status="ok"):
    groups = {}
    for index in range(8):
        definitions = {
            f"metric_{index}_{n}": {"label": f"指标{index}-{n}", "unit": "%", "meaning": f"该项描述变化{index}-{n}", "display_scale": 100}
            for n in range(12)
        }
        groups[f"group_{index}"] = {
            "label": f"证据组{index}", "economic_meaning": f"经济含义{index}",
            "values": {key: n / 100 for n, key in enumerate(definitions) if n != 11},
            "missing_fields": {f"metric_{index}_11": "实际输入缺失"},
            "metric_definitions": definitions, "available_factor_count": 11, "factor_count": 12,
        }
    return {
        "status": status, "outcome": "information_partial" if status == "partial" else "analysis_success",
        "analysis_type": "single_stock_analysis", "tool_contract_version": _ANALYSIS_TOOL_CONTRACT_VERSION,
        "analysis_id": "fx_evidence", "analysis_stage": {"status": "completed"},
        "recommendation_available": False, "primary": None, "alternatives": [],
        "stock": {"name": "样本股份", "ts_code": "600001.SH"},
        "diagnosis_summary": {"summary": "价格回升但仍有反向证据，需结合已知事实继续观察。",
                              "supporting_evidence": ["近5日回升"], "counter_evidence": ["成交额尚不稳定"],
                              "reassessment_conditions": ["下一完整交易日重新核验"]},
        "technical_summary": {"status": "ok", "close": 12.34, "returns": {"5d": 0.025},
                              "rsi_14": 57.3, "macd_structure": {"state_label": "零轴附近修复"}},
        "daily_factor_analysis": {"groups": groups},
        "fundamental_analysis": {"status": "partial", "valuation": {"pe_ttm": 18.2, "pb": 2.4},
                                 "financials": {"revenue_yoy_pct": 5.2, "net_profit_yoy_pct": 3.4,
                                                "operating_cashflow_to_revenue_pct": None},
                                 "extra_fields": {f"财务字段{i}": i for i in range(12)},
                                 "errors": ["现金流字段本次未返回"]},
        "limit_up_pullback_pattern": {"state_label": "等待突破", "actuals": {"volume_ratio": 0.7},
                                      "conditions": {"突破已确认": False}},
        "late_session_analysis": {"status": "not_applicable", "reason": "当前尚未进入尾盘时段"},
        "supplemental_diagnostics": {"status": "partial", "blocks": [
            {"key": f"block_{i}", "label": f"补充块{i}", "summary": f"实际证据短句{i}",
             "metrics": {"window": 20}, "missing_reason": "报告公告日缺失" if i == 6 else None}
            for i in range(7)
        ]},
        "tradability": {"basic_execution_feasible": True, "amount_basis": "latest_completed_daily_bar",
                        "amount_trade_date": "2026-09-11", "amount_yuan": 45678000},
        "realtime_snapshot": {"status": "unavailable", "error": "实时源超时"},
        "data_analysis": {"latest_daily_bar": {"trade_date": "2026-09-11", "close": 12.34},
                          "comparison_profile": {"流通股本": 7654321}},
        "evidence_gaps": [{"reason": "实时源超时"}], "risks": ["价格变化可能影响原有判断"],
        "warnings": ["当前仅部分证据可用"],
        "data_provenance": {"history": {"source": "样本日线源", "actual_range": ["2026-01-01", "2026-09-11"]}},
    }


def test_complete_report_retains_every_group_metric_missing_field_and_component():
    result = goujian_fenxi_anquan_huitui(_single())
    for group in range(8):
        assert f"证据组{group}" in result and f"经济含义{group}" in result
        assert f"指标{group}-10 | 10 | % | 该项描述变化{group}-10" in result
        assert f"指标{group}-11 | 缺失（未取得或无法计算）" in result
        assert "实际输入缺失" in result
    for text in ("财务字段11", "近5日回升", "成交额尚不稳定", "下一完整交易日重新核验", "等待突破",
                 "当前尚未进入尾盘时段", "实际证据短句6", "报告公告日缺失", "45678000", "最近完整交易日",
                 "实时源超时", "7654321", "样本日线源"):
        assert text in result
    assert "建议买入" not in result and "暂不建议买入" not in result


def test_partial_analysis_keeps_all_available_evidence_and_truthful_outer_status():
    payload = _single("partial")
    result = goujian_fenxi_anquan_huitui(payload)
    assert "部分来源或指标缺失" in result
    assert "收盘价：12.34" in result
    assert "现金流字段本次未返回" in result
    assert ContextBuilder.is_compatible_analysis_result(payload)
    assert _analysis_run_status(payload) == "information_partial"


def test_cancelled_scoring_fields_and_model_sentences_do_not_reenter_report():
    payload = _single()
    payload["technical_summary"].update(score_0_100=99, confidence=0.88, evidence=["RSI参考区间0-100"])
    payload["fundamental_analysis"]["score_0_100"] = 87
    original = "综合评分99分，建议买入。\n价格存在回升迹象。"
    result = buquan_zhengju_baogao(original, payload)
    assert "价格存在回升迹象" in result
    assert all(word not in result for word in ("综合评分99分", "得分", "置信度", "score 0 100", "建议买入"))
    assert "RSI参考区间0-100" in result


def test_complete_report_completion_is_idempotent_and_preserves_valid_llm_prose():
    payload = _single()
    original = "这段综合解释来自当前有效证据。"
    first = buquan_zhengju_baogao(original, payload)
    assert first.startswith(original)
    assert buquan_zhengju_baogao(first, payload) == first


def test_real_registry_definitions_and_raw_rsi_survive_repeated_completion():
    payload = _single()
    groups = {}
    for index, row in enumerate(FACTOR_REGISTRY):
        group = groups.setdefault(row["group"], {"label": row["group"], "values": {}, "metric_definitions": {}, "missing_fields": []})
        group["values"][row["feature"]] = 57.3 if row["feature"] == "rsi_14" else index / 100
        group["metric_definitions"][row["feature"]] = factor_definition(row["feature"])
    payload["daily_factor_analysis"] = {"groups": groups}
    first = buquan_zhengju_baogao("按全部原始指标解释。", payload)
    for row in FACTOR_REGISTRY:
        assert row["label"] in first
        assert row["meaning"] in first
    assert "14日相对强弱指标RSI：57.3；含义：" in first
    assert "14日相对强弱指标RSI | 57.3 | 无量纲 |" in first
    assert "不是股票评分" in first and "不是评分或上涨概率" in first
    assert buquan_zhengju_baogao(first, payload) == first
    fallback = goujian_fenxi_anquan_huitui(payload)
    assert buquan_zhengju_baogao(fallback, payload) == fallback


def test_factor_units_distinguish_dimensionless_and_unspecified():
    payload = _single()
    payload["daily_factor_analysis"] = {"groups": {"units": {
        "label": "单位测试", "values": {"known": 1, "unknown": 2},
        "metric_definitions": {"known": {"label": "已知无量纲", "unit": "", "meaning": "明确的比值"},
                               "unknown": {"label": "单位未确认", "meaning": "沿用来源值"}},
    }}}
    result = goujian_fenxi_anquan_huitui(payload)
    assert "已知无量纲 | 1 | 无量纲 | 明确的比值" in result
    assert "单位未确认 | 2 | 原始口径（未标注单位）" in result


def _technical_error():
    payload = _single("error")
    payload.update(outcome="program_error", error="MACD结构计算异常", error_code="single_stock_technical_error")
    payload["technical_summary"] = {"status": "error", "outcome": "program_error", "error": payload["error"], "trade_date": "2026-09-11"}
    return payload


def test_single_stock_program_error_retains_other_evidence_without_claiming_completion():
    payload = _technical_error()
    result = goujian_fenxi_anquan_huitui(payload)
    assert result.startswith("本次分析未完成（程序错误）：MACD结构计算异常")
    for text in ("指标7-11", "财务字段11", "实际证据短句6", "收盘价：12.34", "样本日线源"):
        assert text in result
    assert _analysis_run_status(payload) == "failed"
    assert payload["status"] == "error" and payload["outcome"] == "program_error"
    assert buquan_zhengju_baogao("分析已经完成。", payload) == result
    assert buquan_zhengju_baogao(result, payload) == result


def test_error_without_obtained_evidence_keeps_existing_failure_message():
    payload = {"status": "error", "analysis_type": "single_stock_analysis", "error": "目标日线读取失败",
               "technical_summary": {"status": "error", "error": "计算未执行", "trade_date": "2026-09-11"}}
    result = goujian_fenxi_anquan_huitui(payload)
    assert result == "本次分析未完成：目标日线读取失败。"
    assert "###" not in result


def _selection(observation=False):
    candidate = {key: value for key, value in _single().items() if key in {
        "technical_summary", "daily_factor_analysis", "fundamental_analysis", "supplemental_diagnostics", "tradability",
    }}
    candidate.update(name="样本股份", ts_code="600001.SH", meets_selection_conditions=not observation,
                     selection_analysis={"eligible": not observation, "conditions": [
                         {"key": "momentum", "label": "短期动量", "status": "unmet" if observation else "met",
                          "values": {"原始5日收益": -0.02 if observation else 0.02}, "reason": "按原始收益核验"}],
                         "unmet_conditions": ["短期动量不足"] if observation else [], "missing_conditions": [],
                         "pareto_front": 1, "ranking_basis": "同层仅按成交额及证券代码稳定展示"})
    return {"status": "ok", "analysis_type": "unified_stock_selection", "recommendation_available": not observation,
            "primary": None if observation else candidate, "alternatives": [],
            "diagnostic_candidates": [candidate] if observation else [],
            "no_recommendation_reason": "没有候选通过全部条件" if observation else None}


@pytest.mark.parametrize("observation", [False, True])
def test_rule_selection_report_keeps_conditions_raw_values_and_all_evidence(observation):
    text = goujian_fenxi_anquan_huitui(_selection(observation))
    assert "短期动量" in text and "按原始收益核验" in text
    assert "非支配比较层次：1" in text
    assert "同层仅按成交额及证券代码稳定展示" in text
    assert "指标7-11" in text and "实际证据短句6" in text
    assert ("观察对象（未通过选股条件）" in text) is observation
    assert "综合分" not in text


def test_tool_preserves_partial_analysis_and_adds_live_session_identifier(monkeypatch):
    analysis_session_store.clear()
    payload = _single("partial")
    payload.pop("analysis_id")
    monkeypatch.setattr("src.ashare.xuangu_fenxi.fenxi_xuangu", lambda **kwargs: payload)
    result = json.loads(GupiaoFenxiTool().execute(fanwei="single_stock", gupiao="样本股份"))
    assert result["status"] == "partial" and result["analysis_id"].startswith("fx_")
    assert result["diagnosis_summary"] == payload["diagnosis_summary"]
    assert analysis_session_store.contains(result["analysis_id"])
    analysis_session_store.clear()


@pytest.mark.parametrize("status", ["partial", "error"])
def test_loop_preserves_available_evidence_and_partial_or_error_status(tmp_path, monkeypatch, status):
    payload = _technical_error() if status == "error" else _single("partial")
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")

    class Tool(BaseTool):
        name = "gupiao_fenxi"
        def execute(self, **kwargs):
            return json.dumps(payload, ensure_ascii=False)

    class LLM:
        responses = [LLMResponse(tool_calls=[ToolCallRequest(id="analysis", name="gupiao_fenxi", arguments={})]),
                     LLMResponse(content="当前存在支持和反向证据，以下以已取得数据为准。")]
        def stream_chat(self, messages, tools=None, on_text_chunk=None):
            return self.responses.pop(0)

    registry = ToolRegistry()
    registry.register(Tool())
    result = AgentLoop(registry=registry, llm=LLM(), max_iterations=3).run("分析样本股份全部内容")
    assert result["status"] == ("failed" if status == "error" else "information_partial")
    assert result["content"].startswith("本次分析未完成（程序错误）" if status == "error" else "当前存在支持和反向证据")
    assert "指标7-11" in result["content"]
    assert "建议买入" not in result["content"]


def test_prompt_separates_stock_evidence_and_rule_based_selection():
    prompt = ContextBuilder(ToolRegistry(), WorkspaceMemory()).build_system_prompt()
    assert "recommendation_available=false is a compatibility flag" in prompt
    assert "Show every returned daily_factor_analysis group" in prompt
    assert "five raw dimensions for non-dominated comparison" in prompt
    assert "Do not convert missing values to zero" in prompt
    assert "buy_decision.label" not in prompt


def test_legacy_scored_history_is_not_compatible_with_current_analysis():
    payload = _single()
    payload["buy_decision"] = {"label": "建议买入"}
    assert not ContextBuilder.is_compatible_analysis_result(payload)
