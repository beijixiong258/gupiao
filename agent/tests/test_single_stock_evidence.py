"""无评分单股分析：短历史和可选来源失败仍保留本股已有证据。"""

from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest

from src.ashare import dangu_fenxi, dangu_lianghua, fenxi_yinzi
from src.ashare.dangu_fenxi import fenxi_dangu


def _history(rows=100):
    close = 10 + np.linspace(0, 2, rows)
    return pd.DataFrame({
        "trade_date": pd.bdate_range(end="2026-09-11", periods=rows),
        "open": close - 0.1, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": 1_000_000.0, "amount_yuan": 80_000_000.0,
    })


class Context:
    reference = datetime(2026, 9, 12, 9, 0)

    def __init__(self, rows=100, name="样本股票"):
        self.history = _history(rows)
        self.name = name
        self.history_requests = []
        self.fundamental_requests = 0

    def jiexi_gupiao(self, query):
        return "000021.SZ", {"name": self.name, "industry": "电子", "market": "主板"}, []

    def shichang_shizhong(self):
        return {"session_status": "non_trading_day", "is_trading_day": False, "captured_at": "2026-09-12 09:00:00"}

    def zuixin_wanzheng_jiaoyiri(self):
        return pd.Timestamp("2026-09-11")

    def piliang_lishi(self, codes, *, start_date, end_date, minimum_rows):
        self.history_requests.append({"codes": list(codes), "minimum_rows": minimum_rows})
        return {code: self.history.copy() for code in codes}, {"status": "ok", "source": "offline_history"}

    def dangu_kuaizhao(self, code):
        return {"status": "ok", "name": self.name, "latest_price": 12.0, "previous_close": 11.9, "open": 11.9, "high": 12.2, "low": 11.8, "amount_yuan": 80_000_000.0, "source": "offline_snapshot"}

    def jibenmian(self, code, *, trade_date, allow_current_snapshot=False):
        self.fundamental_requests += 1
        return {
            "profile": {"name": self.name, "industry": "电子"},
            "valuation": {"as_of": trade_date, "pe_ttm": 20, "pb": 2.5},
            "financials": {"report_date": "2026-06-30", "announcement_date": "2026-08-20", "known_as_of": trade_date, "roe_pct": 12, "revenue_yoy_pct": 10, "net_profit_yoy_pct": -5, "debt_to_assets_pct": 30, "net_margin_pct": 5, "operating_cashflow_to_revenue_pct": 0.1},
            "sources": {"financials": "offline_test", "valuation": "offline_test"},
        }

    def zuixin_hengjiemian(self):
        raise RuntimeError("同行横截面本次超时")


def _config():
    return {"dangu": {"history_calendar_days": 1440}, "fenxi": {"min_amount_yuan": 50_000_000}, "weipan": {}, "xingtai": {}}


@pytest.fixture(autouse=True)
def offline_optional_market(monkeypatch):
    def unavailable_index(panel, **kwargs):
        raise RuntimeError("指数源本次不可用")
    monkeypatch.setattr(fenxi_yinzi, "enrich_daily_factor_panel", unavailable_index)


def _forbidden_keys(value):
    forbidden = {"score_0_100", "ranking_score_0_100", "confidence", "weight", "weights", "ranking_details", "buy_decision", "meets_recommendation_threshold", "minimum_recommendation_score"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                yield key
            yield from _forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _forbidden_keys(item)


def test_optional_peer_and_index_failures_preserve_own_eight_groups_without_scores():
    context = Context()
    result = fenxi_dangu(gupiao="000021", config=_config(), context=context)
    assert result["status"] == "partial"
    assert result["outcome"] == "information_partial"
    assert result["technical_summary"]["close"] == 12.0
    assert len(result["daily_factor_analysis"]["groups"]) == 8
    trend = result["daily_factor_analysis"]["groups"]["trend_structure"]
    assert trend["values"]["ma_gap_20"] is not None
    assert result["fundamental_analysis"]["financials"]["net_profit_yoy_pct"] == -5
    assert any("同行横截面本次超时" in gap["reason"] for gap in result["evidence_gaps"])
    assert any("指数源本次不可用" in gap["reason"] for gap in result["evidence_gaps"])
    assert not list(_forbidden_keys(result))
    assert result["recommendation_available"] is False
    assert result["primary"] is None and result["alternatives"] == []
    assert result["diagnosis_summary"]["summary"] == result["plain_language_summary"]
    assert context.history_requests == [{"codes": ["000021.SZ"], "minimum_rows": 1}]
    json.dumps(result, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("rows", [1, 5, 20])
def test_short_history_keeps_available_indicators_and_financials(rows):
    context = Context(rows=rows)
    result = fenxi_dangu(gupiao="样本股票", config=_config(), context=context)
    assert result["status"] == "partial"
    assert result["data_analysis"]["history_summary"]["rows"] == rows
    assert result["data_analysis"]["latest_daily_bar"]["close"] == context.history.close.iloc[-1]
    assert context.fundamental_requests == 1
    assert result["fundamental_analysis"]["financials"]["roe_pct"] == 12
    assert len(result["daily_factor_analysis"]["groups"]) == 8
    if rows < 20:
        assert result["technical_summary"]["moving_averages"]["ma20"] is None
    if rows >= 5:
        assert result["technical_summary"]["moving_averages"]["ma5"] is not None
    assert result["evidence_gaps"]
    assert not list(_forbidden_keys(result))
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_st_stock_and_unavailable_optional_financials_do_not_block_reading():
    class MissingOptional(Context):
        def jibenmian(self, *args, **kwargs):
            raise RuntimeError("财务接口超时")

        def dangu_kuaizhao(self, code):
            raise RuntimeError("实时接口超时")

    result = fenxi_dangu(gupiao="ST样本", config=_config(), context=MissingOptional(name="ST样本"))
    assert result["status"] == "partial"
    assert result["stock"]["name"] == "ST样本"
    assert result["technical_summary"]["close"] == 12
    assert result["daily_factor_analysis"]["groups"]["trend_structure"]["values"]["ma_gap_20"] is not None
    assert result["fundamental_analysis"]["status"] == "unavailable"
    assert any("ST风险标记" in risk for risk in result["risks"])
    assert "财务接口超时" in str(result["evidence_gaps"])
    assert "实时接口超时" in str(result["evidence_gaps"])


def test_technical_program_error_preserves_other_evidence_and_error_status(monkeypatch):
    monkeypatch.setattr(dangu_fenxi, "zongjie_jishu", lambda *args, **kwargs: {"status": "error", "outcome": "program_error", "error": "离线模拟技术错误"})
    result = fenxi_dangu(gupiao="000021", config=_config(), context=Context())
    assert result["status"] == "error"
    assert result["outcome"] == "program_error"
    assert result["error"] == "离线模拟技术错误"
    assert result["fundamental_analysis"]["financials"]["roe_pct"] == 12
    assert len(result["daily_factor_analysis"]["groups"]) == 8


def test_missing_essential_history_keeps_unavailable_instead_of_success():
    class EmptyHistory(Context):
        def piliang_lishi(self, *args, **kwargs):
            return {}, {"status": "unavailable", "error": "主行情源无日线"}

    result = fenxi_dangu(gupiao="000021", config=_config(), context=EmptyHistory())
    assert result["status"] == "unavailable"
    assert result["outcome"] == "data_unavailable"
    assert result["error"] == "主行情源无日线"
    assert result["recommendation_available"] is False


def test_minute_failure_keeps_daily_and_basic_late_session_evidence():
    class MissingMinute(Context):
        def shichang_shizhong(self):
            return {"session_status": "trading", "is_trading_day": True, "captured_at": "2026-09-12 14:50:00"}

        def fenzhong_xingqing(self, code):
            raise RuntimeError("分钟行情本次超时")

    result = fenxi_dangu(gupiao="000021", config=_config(), context=MissingMinute())
    assert result["status"] == "partial"
    assert result["late_session_analysis"]["minute_data"]["status"] == "unavailable"
    assert "分钟行情本次超时" in str(result["evidence_gaps"])
    assert result["technical_summary"]["close"] == 12
    assert result["fundamental_analysis"]["financials"]["roe_pct"] == 12


@pytest.mark.parametrize("message,status", [("Tushare 未找到股票名称：不存在；AKShare 未找到股票名称：不存在", "clarification_required"), ("名称解析失败：连接超时", "unavailable")])
def test_unknown_stock_name_is_distinct_from_resolution_source_failure(message, status):
    class Unresolved(Context):
        def jiexi_gupiao(self, query):
            raise RuntimeError(message)

    result = fenxi_dangu(gupiao="不存在", config=_config(), context=Unresolved())
    assert result["status"] == status
    assert result["error"] == message
