"""量化分析优先、预测二次确认的两阶段契约测试。"""

from __future__ import annotations

import json

import pytest

from src.tools import gupiao_yuce_tool
from src.tools.gupiao_analysis_state import analysis_session_store
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
from src.tools.gupiao_yuce_tool import GupiaoYuceTool


def _prediction_context() -> dict:
    return {
        "primary_code": "600001.SH",
        "candidates": {
            "600001.SH": {
                "code": "600001.SH",
                "name": "样本股份",
                "industry": "电子",
                "source": "auto",
                "signal_date": "2026-07-17",
                "config": {"dangu": {"history_calendar_days": 1440}},
                "technical": {},
                "fundamentals": {},
                "tradability": {"basic_execution_feasible": True},
            }
        },
    }


def _selection_result() -> dict:
    primary = {
        "rank": 1,
        "ts_code": "600001.SH",
        "name": "样本股份",
        "industry": "电子",
        "ranking_score_0_100": 72.0,
        "confidence": 0.74,
        "meets_recommendation_threshold": True,
    }
    return {
        "status": "ok",
        "tool_contract_version": 7,
        "analysis_type": "unified_stock_selection",
        "as_of": "2026-07-17",
        "recommendation_available": True,
        "primary": primary,
        "alternatives": [],
        "reviewed_candidates": [primary],
        "_prediction_context": _prediction_context(),
    }


def _quantitative_prediction() -> dict:
    forecasts = {
        f"T+{horizon}": {
            "target_trade_date": f"2026-07-{20 + horizon}",
            "direction": "up" if horizon != 3 else "flat_or_unavailable",
            "cumulative_return_from_signal_close": 0.01 * horizon,
            "cumulative_return_from_signal_close_pct": 1.0 * horizon,
            "predicted_close_reference": 10.0 + horizon * 0.1,
            "predicted_close_interval_80": [9.5, 10.8 + horizon * 0.1],
            "direction_model_positive_probability": 0.56 + horizon * 0.02,
            "validation_passed": horizon != 3,
            "model_quality": "medium" if horizon != 3 else "low",
        }
        for horizon in (1, 2, 3)
    }
    return {
        "future_3_trading_days": {
            "status": "ok",
            "signal_date": "2026-07-17",
            "signal_close": 10.0,
            "forecast": forecasts,
        },
        "analysis_assessment": {},
    }


@pytest.fixture(autouse=True)
def _empty_analysis_state():
    analysis_session_store.clear()
    yield
    analysis_session_store.clear()


def _run_analysis(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(
        "src.ashare.xuangu_fenxi.fenxi_xuangu",
        lambda **_kwargs: _selection_result(),
    )
    return json.loads(GupiaoFenxiTool().execute(fanwei="named_scope", mingcheng="电子"))


def test_first_stage_returns_confirmation_policy_without_private_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_analysis(monkeypatch)

    assert result["status"] == "ok"
    assert result["tool_contract_version"] == 7
    assert result["analysis_id"].startswith("fx_")
    assert result["analysis_stage"]["prediction_status"] == "not_requested"
    assert result["analysis_stage"]["prediction_confirmation_required"] is True
    assert result["analysis_stage"]["confirmation_timing"] == (
        "later_user_turn_after_analysis_result"
    )
    assert result["analysis_stage"]["initial_preapproval_counts"] is False
    assert result["analysis_stage"]["affirmative_reply_defaults_to"] == "primary"
    assert result["analysis_stage"]["prediction_data_policy"] == (
        "fresh_remote_download_without_local_market_cache"
    )
    assert "_prediction_context" not in result
    assert "reviewed_candidates" not in result
    assert result["reviewed_candidate_count"] == 1
    stored = analysis_session_store.get(result["analysis_id"])
    assert stored is not None
    assert len(stored.result["reviewed_candidates"]) == 1


def test_second_stage_returns_all_horizons_after_explicit_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnosis = _run_analysis(monkeypatch)
    monkeypatch.setattr(
        gupiao_yuce_tool,
        "_xiazai_moxing_lishi",
        lambda context: dict(context),
    )
    monkeypatch.setattr(
        "src.ashare.dangu_yuce.yanjiu_dangu_yuce",
        lambda **_kwargs: _quantitative_prediction(),
    )

    result = json.loads(GupiaoYuceTool().execute(analysis_id=diagnosis["analysis_id"]))

    assert result["status"] == "ok"
    assert result["tool_contract_version"] == 6
    assert set(result["forecast"]) == {"T+1", "T+2", "T+3"}
    assert result["forecast"]["T+1"]["predicted_close"] == pytest.approx(10.1)
    assert result["forecast"]["T+2"]["positive_probability"] == pytest.approx(0.60)
    assert "不读取或写入本地市场数据缓存" in result["data_policy"]


def test_second_stage_requires_live_analysis_session() -> None:
    result = json.loads(GupiaoYuceTool().execute(analysis_id="fx_missing"))

    assert result["status"] == "reanalysis_required"
    assert result["outcome"] == "reanalysis_required"
    assert result["error_code"] == "analysis_not_found"
    assert result["market_data_persistence"] == "none"
