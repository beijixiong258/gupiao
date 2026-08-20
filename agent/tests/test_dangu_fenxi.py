"""单股诊断入口的契约与数据边界回归测试。"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.agent.context import ContextBuilder
from src.agent.fenxi_zhanshi import goujian_fenxi_anquan_huitui
from src.agent.loop import (
    _remove_single_stock_prediction_offer,
    _replace_single_stock_classification_code,
    _single_stock_answer_has_verifiable_reasons,
    _single_stock_prediction_offer,
)
from src.ashare import dangu_fenxi
from src.ashare.dangu_fenxi import fenxi_dangu
from src.ashare.xuangu_fenxi import xianzhi_xuangu_jieguo
from src.ashare.xuangu_fanwei import FanweiLeixing, FenxiFanwei
from src.tools.gupiao_analysis_state import analysis_session_store
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool


def _history(rows: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2025-10-01", periods=rows)
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
        del codes, start_date, end_date, minimum_rows
        return {"000021.SZ": _history()}, {
            "status": "ok",
            "source": "test_remote_history",
            "warnings": [],
            "persistence": "none",
        }

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
        "fenxi": {"macd_structure": {}},
    }


def test_single_stock_value_object_keeps_query_outside_named_scope() -> None:
    request = FenxiFanwei.create("single_stock", None, "深科技")
    assert request.leixing is FanweiLeixing.DANGU_GUPIAO
    assert request.gupiao == "深科技"


def test_single_stock_analysis_returns_diagnosis_without_ranking(monkeypatch) -> None:
    monkeypatch.setattr(
        dangu_fenxi,
        "jiexi_gupiao",
        lambda _query, source="auto": ("000021.SZ", {"name": "深科技", "industry": "电子"}, ["live"]),
    )
    monkeypatch.setattr(
        dangu_fenxi,
        "huoqu_jibenmian",
        lambda *_args, **_kwargs: {
            "profile": {"name": "深科技", "industry": "电子"},
            "valuation": {"pe_ttm": 20},
            "financials": {"roe_pct": 8},
            "sources": {"valuation": "test"},
            "data_quality": {},
            "warnings": [],
            "errors": [],
        },
    )
    result = fenxi_dangu(gupiao="深科技", config=_config(), context=_Context())

    assert result["status"] == "ok"
    assert result["outcome"] == "single_stock_analysis"
    assert result["analysis_type"] == "single_stock_analysis"
    assert result["stock"]["ts_code"] == "000021.SZ"
    assert result["recommendation_available"] is False
    assert result["primary"] is None
    assert result["analysis_stage"]["prediction_confirmation_required"] is False
    assert result["tradability"]["amount_basis"] == "latest_completed_daily_bar"
    expected_amount_date = _history().iloc[-1]["trade_date"].strftime("%Y-%m-%d")
    assert result["tradability"]["amount_trade_date"] == expected_amount_date
    assert "history" not in result["stock"]
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_single_stock_history_shortage_is_information_insufficient(monkeypatch) -> None:
    monkeypatch.setattr(
        dangu_fenxi,
        "jiexi_gupiao",
        lambda _query, source="auto": ("000021.SZ", {"name": "深科技"}, []),
    )

    class ShortContext(_Context):
        def piliang_lishi(self, codes, *, start_date, end_date, minimum_rows):
            del codes, start_date, end_date, minimum_rows
            return {}, {
                "status": "unavailable",
                "incomplete_examples": [{"ts_code": "000021.SZ", "rows": 12}],
                "warnings": [],
            }

    result = fenxi_dangu(gupiao="深科技", config=_config(), context=ShortContext())
    assert result["status"] == "insufficient_data"
    assert result["outcome"] == "information_insufficient"
    assert result["recommendation_available"] is False


def test_tool_exposes_single_stock_contract_and_keeps_session_non_predictive(monkeypatch) -> None:
    analysis_session_store.clear()
    monkeypatch.setattr(
        "src.ashare.xuangu_fenxi.fenxi_xuangu",
        lambda **_kwargs: {
            "status": "ok",
            "outcome": "single_stock_analysis",
            "analysis_type": "single_stock_analysis",
            "stock": {"ts_code": "000021.SZ", "name": "深科技"},
            "selected_stock": {"ts_code": "000021.SZ", "name": "深科技"},
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "analysis_stage": {"status": "completed"},
        },
    )
    result = json.loads(GupiaoFenxiTool().execute(fanwei="single_stock", gupiao="深科技"))
    assert result["analysis_type"] == "single_stock_analysis"
    assert result["analysis_stage"]["prediction_confirmation_required"] is False
    assert result["analysis_id"].startswith("fx_")
    stored = analysis_session_store.get(result["analysis_id"])
    assert stored is not None and stored.prediction_context is None
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
    assert result["reviewed_candidate_count"] == 2
    assert [result["primary"]["ts_code"], result["alternatives"][0]["ts_code"]] == [
        "000001.SZ",
        "000002.SZ",
    ]
    stored = analysis_session_store.get(result["analysis_id"])
    assert stored is not None
    assert set(stored.prediction_context["candidates"]) == {"000001.SZ", "000002.SZ"}
    analysis_session_store.clear()


def test_persisted_single_stock_history_requests_single_stock_reanalysis() -> None:
    payload = {
        "status": "ok",
        "tool_contract_version": 7,
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
    assert _single_stock_prediction_offer("如需预测，请确认后再继续 T+1/T+2/T+3。") is True
    assert _single_stock_prediction_offer("单股诊断不产生预测资格。") is False


def test_single_stock_answer_guard_only_removes_prediction_invitation_paragraph() -> None:
    answer = (
        "结论：价格仍低于四条均线。\n\n"
        "理由：零轴下方金叉只表示弱势修复，20 日回撤为 21.4%。\n\n"
        "如需预测，我可以继续运行 T+1/T+2/T+3。"
    )
    cleaned = _remove_single_stock_prediction_offer(answer)
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


def test_single_stock_safe_summary_keeps_verifiable_reasons_and_source_failures() -> None:
    payload = {
        "status": "ok",
        "analysis_type": "single_stock_analysis",
        "query": "新易盛",
        "stock": {"ts_code": "300502.SZ", "name": "新易盛"},
        "as_of": "2026-08-19",
        "generated_at": "2026-08-20 09:32:24",
        "result_confirmation": "intraday_provisional",
        "technical_summary": {
            "close": 411.5,
            "returns": {"1d": -0.090025, "3d": -0.081637, "20d": -0.191234},
            "moving_averages": {"ma5": 441.674, "ma10": 429.429, "ma20": 431.01, "ma60": 502.552},
            "rsi_14": 42.98,
            "score_0_100": 39,
            "evidence": ["价格未站上 MA20", "MACD 柱为正", "20 日年化波动率偏高"],
            "annualized_volatility_20": 1.095366,
            "drawdown_from_20d_high": -0.21444,
            "macd": {"histogram": 10.105},
            "macd_structure": {
                "structure_classification": {
                    "code": "weak_rebound",
                    "label": "零轴下方出现修复证据，仍属于弱势反弹观察",
                },
                "latest_cross": {
                    "status": "active",
                    "label": "金叉",
                    "event_date": "2026-08-13",
                    "region_label": "零轴下方",
                    "age_trading_sessions": 4,
                },
                "supporting_evidence": [],
                "counter_evidence": ["快线和慢线均在零轴下方，当前仍处于弱势趋势背景"],
                "divergences": {
                    "bottom": {"dif": {"status": "invalidated"}},
                    "top": {"dif": {"status": "expired"}},
                },
                "evidence_reliability": {"label": "有效历史较完整，结构证据可靠性较高"},
            },
        },
        "fundamental_analysis": {
            "status": "unavailable",
            "available_fields": {"profile": True, "valuation": False, "financials": False},
            "errors": [
                "Tushare 估值失败：接口频率超限",
                "Tushare 财务指标失败：没有接口访问权限",
                "AKShare 个股资料失败：Remote end closed connection",
                "AKShare 实时估值与历史分析日 2026-08-19 不一致",
                "AKShare 财务指标缺少公告日，未用于历史分析日",
            ],
        },
        "tradability": {
            "basic_execution_feasible": True,
            "analysis_price": 414.02,
            "analysis_price_basis": "realtime_snapshot",
            "amount_yuan": 22_898_868_700.0,
            "amount_basis": "latest_completed_daily_bar",
            "amount_trade_date": "2026-08-19",
            "price_limit_pct": 20.0,
        },
        "data_provenance": {
            "history": {
                "actual_range": ["2023-12-27", "2026-08-19"],
                "session_coverage": {"minimum": 1.0},
            }
        },
        "risks": [],
    }
    answer = goujian_fenxi_anquan_huitui(payload)
    for expected in (
        "411.50 元",
        "近 3 日 -8.2%",
        "近 20 日 -19.1%",
        "MA5 441.67",
        "39/100（不是上涨概率）",
        "2026-08-13 在零轴下方形成金叉",
        "证据冲突",
        "20 日年化波动率 109.5%",
        "较近 20 日高点回撤 21.4%",
        "接口触发限频",
        "财务指标接口权限不足",
        "备用公开数据源连接失败",
        "缺少公告日",
        "已取得公司基本资料，但没有取得可按分析日核验的估值和财务指标",
        "备用估值快照与分析日不一致",
        "2026-08-19 完整交易日成交额约 228.99 亿元",
        "远端日历可核验区间内交易日覆盖率最低 100%",
    ):
        assert expected in answer
    assert "weak_rebound" not in answer
    assert _single_stock_answer_has_verifiable_reasons(answer, payload) is True
    assert "今日盘中成交额" not in answer


def test_single_stock_reason_check_rejects_polished_but_incomplete_answer() -> None:
    payload = {
        "result_confirmation": "intraday_provisional",
        "technical_summary": {
            "close": 411.5,
            "returns": {"20d": -0.191234},
            "moving_averages": {"ma20": 431.01},
            "score_0_100": 39,
            "annualized_volatility_20": 1.095366,
            "drawdown_from_20d_high": -0.21444,
            "macd_structure": {
                "counter_evidence": ["快线和慢线均在零轴下方"],
                "latest_cross": {"status": "active", "event_date": "2026-08-13"},
            },
        },
        "fundamental_analysis": {
            "status": "unavailable",
            "available_fields": {"profile": True, "valuation": False, "financials": False},
            "errors": [
                "接口频率超限",
                "没有访问权限",
                "Remote end closed connection",
                "估值与历史分析日不一致",
                "缺少公告日",
            ],
        },
        "data_provenance": {"history": {"session_coverage": {"minimum": 1.0}}},
    }
    incomplete = (
        "截至盘中，新易盛收盘价 411.5 元，近 20 日 -19.1%，MA20 为 431.01。"
        "当前仍属弱势，2026-08-13 出现金叉。基本面因限频、权限和连接中断而不可用，"
        "这是数据可用性问题，不能据此判断好坏。回撤 21.4%。"
    )
    assert _single_stock_answer_has_verifiable_reasons(incomplete, payload) is False


def test_selection_count_limit_hides_unrequested_candidates_and_prediction_context() -> None:
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
        "_prediction_context": {
            "primary_code": "000001.SZ",
            "candidates": {item["ts_code"]: {"code": item["ts_code"]} for item in [primary, *alternatives]},
        },
    }
    limited = xianzhi_xuangu_jieguo(result, 2)
    assert [item["ts_code"] for item in [limited["primary"], *limited["alternatives"]]] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert set(limited["_prediction_context"]["candidates"]) == {"000001.SZ", "000002.SZ"}
    assert limited["requested_candidate_count"] == 2
