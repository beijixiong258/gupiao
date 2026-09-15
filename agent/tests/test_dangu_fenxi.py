"""单股诊断入口的契约与数据边界回归测试。"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent.context import ContextBuilder
from src.agent.fenxi_zhanshi import goujian_fenxi_anquan_huitui
from src.agent.loop import (
    _remove_removed_forecast_offer,
    _replace_single_stock_classification_code,
    _removed_forecast_offer,
)
from src.ashare import dangu_fenxi, dangu_lianghua, xuangu_fenxi
from src.ashare.dangu_fenxi import fenxi_dangu
from src.ashare.xuangu_fenxi import xianzhi_xuangu_jieguo
from src.ashare.xuangu_fanwei import FanweiLeixing, FenxiFanwei
from src.tools.gupiao_analysis_state import analysis_session_store
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool, _mianxiang_zhinengti_jieguo


def _history(rows: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-08-19", periods=rows)
    close = 20 + np.linspace(0, 4, rows) + np.sin(np.arange(rows) / 5)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(rows, 2_000_000.0),
            "amount_yuan": np.full(rows, 80_000_000.0),
        }
    )


class _Context:
    reference = datetime(2026, 8, 20, 8, 30)

    def jiexi_gupiao(self, query):
        return "000021.SZ", {"name": "深科技", "industry": "电子"}, ["live"]

    def jibenmian(self, code, *, trade_date, allow_current_snapshot=False):
        return {
            "profile": {"name": "深科技", "industry": "电子"},
            "valuation": {"pe_ttm": 20},
            "financials": {"roe_pct": 18, "net_profit_yoy_pct": 20, "debt_to_assets_pct": 30},
            "sources": {"valuation": "test"},
            "data_quality": {},
            "warnings": [],
            "errors": [],
        }

    def jiaoyi_rili(self):
        return SimpleNamespace()

    def shichang_shizhong(self):
        return {
            "session_status": "pre_market",
            "captured_at": "2026-08-20 08:30:00",
            "calendar_source": "test_calendar",
        }

    def zuixin_wanzheng_jiaoyiri(self):
        return pd.Timestamp("2026-08-19")

    def piliang_lishi(self, codes, *, start_date, end_date, minimum_rows):
        del start_date, end_date, minimum_rows
        return {code: _history() for code in codes}, {
            "status": "ok",
            "source": "test_remote_history",
            "warnings": [],
            "persistence": "none",
        }

    def zuixin_hengjiemian(self):
        return pd.DataFrame(
            [
                {
                    "ts_code": f"{code:06d}.SZ",
                    "name": "深科技" if code == 21 else f"参照{code}",
                    "industry": "电子",
                    "latest_price": 24.0,
                    "previous_close": 23.8,
                    "open": 23.9,
                    "high": 24.2,
                    "low": 23.7,
                    "volume": 2_000_000,
                    "amount_yuan": 80_000_000,
                    "pe_ttm": 20.0,
                    "pb": 2.0,
                }
                for code in range(21, 29)
            ]
        ), {"source": "tushare_daily_cross_section", "as_of": "2026-08-19"}

    def dangu_kuaizhao(self, code):
        return {
            "status": "ok",
            "source": "test_live_snapshot",
            "captured_at": "2026-08-20 08:30:00",
            "name": "深科技",
            "last_price": 24.0,
            "previous_close": 23.8,
            "open": 23.9,
            "high": 24.2,
            "low": 23.7,
            "amount_yuan": 80_000_000,
        }


def _config() -> dict:
    return {
        "dangu": {"history_calendar_days": 1440, "minimum_history_rows": 180, "min_amount_yuan": 30_000_000},
        "fenxi": {
            "macd_structure": {},
            "component_weights": {
                "daily_factors": 0.55,
                "fundamental": 0.20,
                "pattern": 0.15,
                "late_session": 0.10,
            },
        },
    }


def test_single_stock_value_object_keeps_query_outside_named_scope() -> None:
    request = FenxiFanwei.create("single_stock", None, "深科技")
    assert request.leixing is FanweiLeixing.DANGU_GUPIAO
    assert request.gupiao == "深科技"














def test_tool_exposes_single_stock_contract_and_keeps_session_non_predictive(monkeypatch) -> None:
    analysis_session_store.clear()
    monkeypatch.setattr(
        "src.ashare.xuangu_fenxi.fenxi_xuangu",
        lambda **_kwargs: {
            "status": "ok",
            "outcome": "no_recommendation",
            "analysis_type": "single_stock_analysis",
            "stock": {"ts_code": "000021.SZ", "name": "深科技"},
            "selected_stock": {"ts_code": "000021.SZ", "name": "深科技"},
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "analysis_stage": {
                "status": "completed",
            },
        },
    )
    result = json.loads(GupiaoFenxiTool().execute(fanwei="single_stock", gupiao="深科技"))
    assert result["analysis_type"] == "single_stock_analysis"
    assert "prediction_confirmation_required" not in result["analysis_stage"]
    assert result["analysis_id"].startswith("fx_")
    stored = analysis_session_store.get(result["analysis_id"])
    assert stored is not None and not hasattr(stored, "prediction_context")
    analysis_session_store.clear()


def test_tool_count_argument_limits_public_selection(monkeypatch) -> None:
    analysis_session_store.clear()
    primary = {"ts_code": "000001.SZ", "name": "甲"}
    alternatives = [
        {"ts_code": "000002.SZ", "name": "乙"},
        {"ts_code": "000003.SZ", "name": "丙"},
    ]
    raw_result = {
        "status": "ok",
        "outcome": "recommendation",
        "tool_contract_version": 7,
        "analysis_type": "unified_stock_selection",
        "recommendation_available": True,
        "primary": primary,
        "alternatives": alternatives,
        "reviewed_candidates": [primary, *alternatives],
        "_prediction_context": {
            "primary_code": "000001.SZ",
            "candidates": {
                item["ts_code"]: {"code": item["ts_code"]}
                for item in [primary, *alternatives]
            },
        },
    }
    monkeypatch.setattr(
        "src.ashare.xuangu_fenxi.fenxi_xuangu",
        lambda **kwargs: xianzhi_xuangu_jieguo(raw_result, kwargs.get("shuliang")),
    )
    result = json.loads(
        GupiaoFenxiTool().execute(
            fanwei="named_scope",
            mingcheng="电子板块",
            shuliang=2,
        )
    )
    assert result["requested_candidate_count"] == 2
    assert result["reviewed_candidate_count"] == 3
    assert result["displayed_candidate_count"] == 2
    assert [result["primary"]["ts_code"], result["alternatives"][0]["ts_code"]] == [
        "000001.SZ",
        "000002.SZ",
    ]
    stored = analysis_session_store.get(result["analysis_id"])
    assert stored is not None
    assert "_prediction_context" not in stored.result
    analysis_session_store.clear()


def test_persisted_single_stock_history_requests_single_stock_reanalysis() -> None:
    payload = {
        "status": "ok",
        "tool_contract_version": dangu_fenxi.DANGU_TOOL_CONTRACT_VERSION,
        "analysis_id": "fx_test",
        "analysis_type": "single_stock_analysis",
        "query": "深科技",
        "stock": {"ts_code": "000021.SZ", "name": "深科技"},
        "analysis_stage": {"status": "completed"},
    }
    message = {
        "role": "tool",
        "name": "gupiao_fenxi",
        "content": json.dumps(payload, ensure_ascii=False),
    }
    sanitized = ContextBuilder._sanitize_historical_message(message)
    restored = json.loads(sanitized["content"])
    assert restored["status"] == "reanalysis_required"
    assert restored["scope_request"] == {"fanwei": "single_stock", "mingcheng": None, "gupiao": "深科技"}


def test_single_stock_answer_guard_rejects_prediction_invitation_but_allows_boundary_note() -> None:
    assert _removed_forecast_offer("如需预测，请确认后再继续 T+1/T+2/T+3。") is True
    assert _removed_forecast_offer("单股诊断不产生预测资格。") is False


def test_single_stock_answer_guard_only_removes_prediction_invitation_paragraph() -> None:
    answer = (
        "结论：价格仍低于四条均线。\n\n"
        "理由：零轴下方金叉只表示弱势修复，20 日回撤为 21.4%。\n\n"
        "如需预测，我可以继续运行 T+1/T+2/T+3。"
    )
    cleaned = _remove_removed_forecast_offer(answer)
    assert cleaned == (
        "结论：价格仍低于四条均线。\n\n"
        "理由：零轴下方金叉只表示弱势修复，20 日回撤为 21.4%。"
    )


def test_single_stock_internal_classification_code_is_replaced_by_returned_label() -> None:
    payload = {
        "technical_summary": {
            "macd_structure": {
                "structure_classification": {
                    "code": "weak_rebound",
                    "label": "零轴下方出现修复证据，仍属于弱势反弹观察",
                }
            }
        }
    }
    answer = _replace_single_stock_classification_code(
        "结构分类为 weak_rebound。",
        payload,
    )
    assert "weak_rebound" not in answer
    assert "零轴下方出现修复证据，仍属于弱势反弹观察" in answer






def test_selection_count_limit_hides_unrequested_candidates() -> None:
    primary = {"ts_code": "000001.SZ", "name": "甲"}
    alternatives = [
        {"ts_code": "000002.SZ", "name": "乙"},
        {"ts_code": "000003.SZ", "name": "丙"},
    ]
    result = {
        "status": "ok",
        "analysis_type": "unified_stock_selection",
        "recommendation_available": True,
        "primary": primary,
        "alternatives": alternatives,
        "reviewed_candidates": [primary, *alternatives],
    }
    limited = xianzhi_xuangu_jieguo(result, 2)
    assert [item["ts_code"] for item in [limited["primary"], *limited["alternatives"]]] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert limited["requested_candidate_count"] == 2
