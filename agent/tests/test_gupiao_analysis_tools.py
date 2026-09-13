"""仅分析产品的工具契约与进程内会话边界。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.agent.context import ContextBuilder, _ANALYSIS_TOOL_CONTRACT_VERSION
from src.agent.fenxi_zhanshi import goujian_fenxi_anquan_huitui
from src.agent.memory import WorkspaceMemory
from src.agent.tools import ToolRegistry
from src.tools.gupiao_analysis_state import AnalysisSessionStore, analysis_session_store
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool


@pytest.fixture(autouse=True)
def _empty_analysis_state():
    analysis_session_store.clear()
    yield
    analysis_session_store.clear()


def _selection_result() -> dict:
    primary = {
        "ts_code": "600001.SH", "name": "样本股份",
        "selection_analysis": {"eligible": True, "conditions": [], "pareto_front": 1},
        "meets_selection_conditions": True,
        "evidence_gaps": [{"reason": "财务指标尚未核验", "impact": "降低证据完整度"}],
        "reassessment_conditions": ["财务指标补齐并通过核验后重新分析"],
    }
    return {
        "status": "ok", "tool_contract_version": _ANALYSIS_TOOL_CONTRACT_VERSION,
        "analysis_type": "unified_stock_selection", "as_of": "2026-09-11",
        "recommendation_available": True, "primary": primary,
        "alternatives": [], "reviewed_candidates": [primary],
        "_request_internal": {"identity": "private"},
        "diagnosis_validity": {
            "daily_data_as_of": "2026-09-11", "generated_at": "2026-09-12T10:00:00",
            "explanation": "判断仅代表本次取得的数据", "reassess_when": ["下一完整交易日数据形成后"],
        },
    }


def test_analysis_finishes_without_forecast_handoff(monkeypatch):
    monkeypatch.setattr("src.ashare.xuangu_fenxi.fenxi_xuangu", lambda **kwargs: _selection_result())
    result = json.loads(GupiaoFenxiTool().execute(fanwei="named_scope", mingcheng="电子"))

    assert ContextBuilder.is_compatible_analysis_result(result)
    assert not any("prediction" in key for key in result["analysis_stage"])
    assert "reviewed_candidates" not in result
    assert "_request_internal" not in result
    assert result["reviewed_candidate_count"] == 1
    stored = analysis_session_store.get(result["analysis_id"])
    assert stored is not None and not hasattr(stored, "prediction_context")
    assert stored.result["primary"]["name"] == "样本股份"


def test_analysis_state_rejects_nested_market_tables():
    store = AnalysisSessionStore()
    with pytest.raises(ValueError, match="DataFrame"):
        store.save({"nested": [{"history": pd.DataFrame({"close": [10.0]})}]})


def test_retired_prediction_history_is_obsolete_and_hides_old_forecast():
    history = [
        {"role": "tool", "name": "gupiao_yuce", "tool_call_id": "old", "content": json.dumps({"forecast": {"T+3": 88}})},
        {"role": "assistant", "content": "旧预测认为未来收盘价为 88 元，可继续预测。"},
    ]
    messages = ContextBuilder(ToolRegistry(), WorkspaceMemory()).build_messages("继续", history)
    marker = json.loads(messages[1]["content"])
    assert marker["status"] == "obsolete_history_result"
    assert marker["outcome"] == "feature_removed"
    assert "forecast" not in marker
    assert "88" not in messages[2]["content"]


def test_analysis_fallback_shows_evidence_gaps_and_reassessment_without_opt_in():
    text = goujian_fenxi_anquan_huitui(_selection_result())
    assert "财务指标尚未核验" in text
    assert "财务指标补齐并通过核验后重新分析" in text
    assert "日线截止日：2026-09-11" in text
    assert "下一完整交易日数据形成后" in text
    assert "预测" not in text


def test_legacy_analysis_contract_with_forecast_fields_is_not_reusable():
    result = {**_selection_result(), "analysis_id": "fx_legacy", "analysis_stage": {
        "status": "completed", "prediction_confirmation_required": True,
    }}
    assert not ContextBuilder.is_compatible_analysis_result(result)
