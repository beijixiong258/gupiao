"""Contract checks for the diagnosis-first, fixed three-day workflow."""

from __future__ import annotations

import copy
import json

import pytest

from src.tools.gupiao_analysis_cache import clear_analysis_cache
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
from src.tools.gupiao_yuce_tool import GupiaoYuceTool


def _full_result() -> dict:
    forecasts = {
        f"T+{horizon}": {
            "target_trade_date": f"2026-07-{20 + horizon}",
            "direction": "up" if horizon != 3 else "flat_or_unavailable",
            "cumulative_return_from_signal_close": 0.01 * horizon,
            "cumulative_return_from_signal_close_pct": 1.0 * horizon,
            "predicted_close_reference": 10.0 + horizon * 0.1,
            "predicted_close_interval_80": [9.5, 10.8 + horizon * 0.1],
            "empirical_positive_probability": 0.55 + horizon * 0.02,
            "direction_model_positive_probability": 0.56 + horizon * 0.02,
            "direction_probability_method": "calibrated_direction_model",
            "validation_passed": horizon != 3,
            "model_quality": "medium" if horizon != 3 else "low",
        }
        for horizon in (1, 2, 3)
    }
    validation = {
        "horizons": {
            f"T+{horizon}": {
                "validation_passed": horizon != 3,
                "quality_score": 0.7 if horizon != 3 else 0.2,
                "quality_label": "medium" if horizon != 3 else "low",
            }
            for horizon in (1, 2, 3)
        }
    }
    return {
        "status": "ok",
        "tool_contract_version": 4,
        "stock": {"ts_code": "600001.SH", "name": "样本股份"},
        "as_of": "2026-07-17",
        "generated_at": "2026-07-20 16:00:00",
        "market_data": {"source": "akshare", "adjustment": "qfq"},
        "technical_analysis": {"close": 10.0, "score_0_100": 55},
        "fundamental_analysis": {"profile": {"industry": "测试行业"}},
        "tradability": {"basic_execution_feasible": True},
        "risks": ["样本仅用于测试"],
        "quantitative_analysis": {
            "status": "ok",
            "peer_universe": {"selected_stock_count": 12},
            "daily_factor_data": {"status": "ok"},
            "methodology": {"model": "HistGradientBoostingRegressor"},
            "limitations": ["同行样本存在当前成分偏差"],
            "analysis_assessment": {
                "evidence_label": "证据偏正面",
                "confidence": "descriptive_factor_evidence",
                "reasons": ["当前因子证据偏正面"],
            },
            "factor_analysis": {
                "status": "ok",
                "model_status": "not_run",
                "direction": "偏上涨",
                "evidence_label": "证据偏正面",
                "evidence": ["当前因子证据偏正面"],
            },
        },
        "future_3_trading_days": {
            "status": "ok",
            "signal_date": "2026-07-17",
            "signal_close": 10.0,
            "validation": validation,
            "forecast": forecasts,
        },
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
    assert result["direction_analysis"]["horizon"] == "当前因子状态"
    assert result["direction_analysis"]["positive_probability"] is None
    assert result["direction_analysis"]["negative_probability"] is None
    assert result["direction_analysis"]["probability_method"] is None
    assert "quantitative_analysis" not in result
    assert "future_3_trading_days" not in result
    assert "analysis_assessment" not in result


def test_second_stage_returns_all_three_horizons_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ashare.gupiao_yanjiu.fenxi_gupiao", lambda **_kwargs: _full_result())
    diagnosis = json.loads(GupiaoFenxiTool().execute(gupiao="样本股份"))

    result = json.loads(GupiaoYuceTool().execute(analysis_id=diagnosis["analysis_id"]))

    assert result["status"] == "ok"
    assert set(result["forecast"]) == {"T+1", "T+2", "T+3"}
    assert result["forecast"]["T+1"]["predicted_close"] == pytest.approx(10.1)
    assert result["forecast"]["T+2"]["positive_probability"] == pytest.approx(0.60)
    assert result["forecast"]["T+3"]["validation_passed"] is False
    assert "position" not in json.dumps(result, ensure_ascii=False)
    assert "cost" not in json.dumps(result, ensure_ascii=False)


def test_second_stage_publishes_unvalidated_model_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    full = copy.deepcopy(_full_result())
    full["future_3_trading_days"]["forecast"]["T+2"]["cumulative_return_from_signal_close"] = 9.99
    full["future_3_trading_days"]["forecast"]["T+2"]["cumulative_return_from_signal_close_pct"] = 999.0
    full["future_3_trading_days"]["forecast"]["T+2"]["validation_passed"] = False
    monkeypatch.setattr("src.ashare.gupiao_yanjiu.fenxi_gupiao", lambda **_kwargs: full)
    diagnosis = json.loads(GupiaoFenxiTool().execute(gupiao="样本股份"))

    result = json.loads(GupiaoYuceTool().execute(analysis_id=diagnosis["analysis_id"]))

    assert result["status"] == "ok"
    assert result["forecast"]["T+2"]["predicted_return"] == 9.99
    assert result["forecast"]["T+2"]["validation_passed"] is False


def test_second_stage_requires_live_analysis_id() -> None:
    result = json.loads(GupiaoYuceTool().execute(analysis_id="fx_missing"))

    assert result["status"] == "error"
    assert result["error_code"] == "analysis_not_found"
