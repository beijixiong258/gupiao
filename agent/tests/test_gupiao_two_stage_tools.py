"""Contract tests for diagnosis-first, request-specific single-stock forecasts."""

from __future__ import annotations

import copy
import json

import pytest

from src.tools.gupiao_analysis_cache import clear_analysis_cache
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
from src.tools.gupiao_yuce_tool import GupiaoYuceTool


def _full_result() -> dict:
    return {
        "status": "ok",
        "tool_contract_version": 3,
        "stock": {"ts_code": "600001.SH", "name": "样本股份"},
        "as_of": "2026-07-17",
        "generated_at": "2026-07-20 16:00:00",
        "market_data": {"source": "akshare", "adjustment": "qfq"},
        "current_quote": {"status": "ok", "last_price": 10.2},
        "technical_analysis": {"close": 10.0, "score_0_100": 55},
        "fundamental_analysis": {"profile": {"industry": "测试行业"}},
        "tradability": {"basic_execution_feasible": True},
        "risks": ["样本仅用于测试"],
        "quantitative_analysis": {
            "status": "ok",
            "peer_universe": {"selected_stock_count": 12, "relative_snapshot": {"rank_ret_5": 0.62}},
            "methodology": {"model": "HistGradientBoostingRegressor"},
            "limitations": ["同行样本存在当前成分偏差"],
            "cost_assumption": {
                "scenario": "normal_cost",
                "estimated_roundtrip_cost_rate": 0.001,
            },
            "validation": {
                "horizons": {
                    "T+2": {
                        "validation_passed": True,
                        "quality_score": 0.72,
                        "quality_label": "medium",
                        "direction_accuracy": 0.58,
                        "skill_vs_median_baseline": 0.06,
                        "oos_samples": 120,
                        "production_model_ensemble": {
                            "components": ["HistGradientBoostingRegressor", "Ridge"],
                            "weight_selection": {"tree_weight": 0.5, "linear_weight": 0.5},
                            "latest_component_predictions": {"tree": 0.04, "linear": 0.06},
                        },
                    }
                }
            },
            "forecast": {
                "T+2": {
                    "assumed_entry_date": "2026-07-20",
                    "scenario_exit_date": "2026-07-22",
                    "entry_to_exit_gross_return": 0.03,
                    "entry_to_exit_gross_return_pct": 3.0,
                    "estimated_net_return_after_cost": 0.028,
                    "estimated_net_return_after_cost_pct": 2.8,
                    "empirical_positive_probability": 0.61,
                    "empirical_net_return_interval_80": [-0.02, 0.07],
                    "position_and_cost": {"execution_feasible": True},
                }
            },
            "analysis_assessment": {"scenario_timing": {"valid": True}},
        },
        "future_3_trading_days": {
            "status": "ok",
            "validation": {
                "horizons": {
                    "T+2": {
                        "validation_passed": True,
                        "quality_score": 0.7,
                        "quality_label": "medium",
                        "direction_accuracy": 0.57,
                        "skill_vs_median_baseline": 0.05,
                        "oos_samples": 118,
                        "production_model_ensemble": {
                            "components": ["HistGradientBoostingRegressor", "Ridge"],
                            "weight_selection": {"tree_weight": 0.5, "linear_weight": 0.5},
                            "latest_component_predictions": {"tree": 0.04, "linear": 0.06},
                        },
                    }
                }
            },
            "forecast": {
                "T+2": {
                    "target_trade_date": "2026-07-21",
                    "cumulative_return_from_signal_close": 0.05,
                    "cumulative_return_from_signal_close_pct": 5.0,
                    "predicted_close_reference": 10.5,
                    "predicted_close_interval_80": [9.7, 11.1],
                    "empirical_positive_probability": 0.63,
                    "empirical_return_interval_80": [-0.03, 0.09],
                }
            },
        },
        "analysis_assessment": {"evidence_label": "证据偏正面"},
    }


@pytest.fixture(autouse=True)
def _empty_analysis_cache():
    clear_analysis_cache()
    yield
    clear_analysis_cache()


def test_first_stage_hides_forecasts_and_returns_analysis_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ashare.gupiao_yanjiu.fenxi_gupiao", lambda **_kwargs: _full_result())

    result = json.loads(GupiaoFenxiTool().execute(gupiao="样本股份"))

    assert result["status"] == "ok"
    assert result["tool_contract_version"] == 4
    assert result["analysis_id"].startswith("fx_")
    assert result["analysis_stage"]["status"] == "completed"
    assert result["peer_analysis"]["peer_universe"]["selected_stock_count"] == 12
    assert "quantitative_analysis" not in result
    assert "future_3_trading_days" not in result
    assert "analysis_assessment" not in result


def test_second_stage_publishes_only_validated_horizon_and_position_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.ashare.gupiao_yanjiu.fenxi_gupiao", lambda **_kwargs: _full_result())
    diagnosis = json.loads(GupiaoFenxiTool().execute(gupiao="样本股份"))

    result = json.loads(
        GupiaoYuceTool().execute(
            analysis_id=diagnosis["analysis_id"],
            horizon=2,
            mode="future_close",
            intent="sell_upside",
            buy_price=9.0,
            shares=1_000,
        )
    )

    assert result["forecast_status"] == "validated"
    assert result["forecast"]["historical_similar_sample_positive_rate"] == 0.63
    assert result["forecast"]["estimated_return_after_roundtrip_cost"] == pytest.approx(0.04895)
    assert result["model_ensemble"]["weight_selection"]["tree_weight"] == 0.5
    assert "latest_component_predictions" not in result["model_ensemble"]
    assert result["position_return_analysis"]["status"] == "ok"
    assert result["position_return_analysis"]["buy_cost"]["commission_yuan"] == 5.0
    assert result["position_return_analysis"]["current_exit_estimate"]["net_profit_yuan"] > 0
    assert result["position_return_analysis"]["projected_exit_estimate"]["net_profit_yuan"] > 0


def test_second_stage_publishes_unvalidated_model_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    full = copy.deepcopy(_full_result())
    full["future_3_trading_days"]["validation"]["horizons"]["T+2"]["validation_passed"] = False
    full["future_3_trading_days"]["forecast"]["T+2"]["cumulative_return_from_signal_close"] = 9.99
    monkeypatch.setattr("src.ashare.gupiao_yanjiu.fenxi_gupiao", lambda **_kwargs: full)
    diagnosis = json.loads(GupiaoFenxiTool().execute(gupiao="样本股份"))

    result = json.loads(
        GupiaoYuceTool().execute(
            analysis_id=diagnosis["analysis_id"],
            horizon=2,
            mode="future_close",
        )
    )

    assert result["status"] == "ok"
    assert result["forecast_status"] == "model_estimate"
    assert result["forecast"]["cumulative_return_from_signal_close"] == 9.99
    assert result["forecast"]["prediction_type"] == "unvalidated_model_estimate"
    assert result["forecast"]["validation_passed"] is False
    assert "仍发布当前模型估计" in result["forecast_notice"]


def test_second_stage_requires_live_analysis_id() -> None:
    result = json.loads(
        GupiaoYuceTool().execute(
            analysis_id="fx_missing",
            horizon=2,
            mode="future_close",
        )
    )

    assert result["status"] == "error"
    assert result["error_code"] == "analysis_not_found"
