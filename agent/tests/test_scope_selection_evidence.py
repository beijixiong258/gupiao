"""条件选股的离线整合：真实条件判断、非支配分层、原始证据与来源失败。"""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from src.ashare import xuangu_fenxi
from src.ashare.xuangu_fanwei import FenxiFanwei


def _history():
    close = np.linspace(10, 12, 220)
    return pd.DataFrame({"trade_date": pd.bdate_range(end="2026-09-11", periods=220), "open": close - 0.1, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1_000_000.0, "amount_yuan": 80_000_000.0})


class ScopeContext:
    reference = datetime(2026, 9, 14, 9, 40)

    def __init__(self, mode):
        self.mode = mode
        self.codes = ["000021.SZ", "000022.SZ", "000023.SZ"]
        self.frame = pd.DataFrame([{"ts_code": code, "name": f"样本{code}", "industry": "电子", "latest_price": 12.0, "previous_close": 11.9, "open": 11.9, "high": 12.2, "low": 11.8, "volume": 1_000_000.0, "amount_yuan": 80_000_000.0, "pe_ttm": 20.0, "pb": 2.0, "list_date": "2000-01-01"} for code in self.codes])

    def zuixin_wanzheng_jiaoyiri(self):
        return pd.Timestamp("2026-09-11")

    def shichang_shizhong(self):
        return {"session_status": "trading", "is_trading_day": True, "captured_at": "2026-09-14 14:50:00" if self.mode == "minute_failure" else "2026-09-14 09:40:00"}

    def piliang_lishi(self, codes, **kwargs):
        return {code: _history() for code in codes}, {"status": "ok", "source": "offline_history"}

    def shishi_kuaizhao(self):
        if self.mode == "snapshot_exception":
            raise RuntimeError("实时源请求抛出异常")
        if self.mode == "missing_quote":
            return pd.DataFrame(), {"status": "unavailable", "error": "实时源本次无数据"}
        return self.frame.assign(provider_quote_time="2026-09-14 09:39:30", provider_trade_date="2026-09-14"), {"status": "ok", "source": "offline_snapshot", "captured_at": "2026-09-14 09:40:00"}

    def fenzhong_xingqing(self, code):
        raise RuntimeError("分钟源请求抛出异常")

    def jibenmian(self, code, **kwargs):
        if self.mode == "financial_failure":
            raise RuntimeError("财务源请求抛出异常")
        return {"profile": {"name": "样本", "industry": "电子"}, "financials": {"report_date": "2026-06-30", "announcement_date": "2026-08-20", "known_as_of": "2026-09-11", "roe_pct": 12, "net_profit_yoy_pct": 8, "revenue_yoy_pct": 10, "debt_to_assets_pct": 30, "net_margin_pct": 5, "operating_cashflow_to_revenue_pct": 0.2}, "valuation": {"pe_ttm": 20, "pb": 2}, "sources": {"financials": "offline_test"}}


def _config():
    return {"fenxi": {"prefilter_limit": 3, "history_calendar_days": 420, "minimum_history_rows": 80, "min_amount_yuan": 50_000_000, "minimum_listing_calendar_days": 180, "factor_candidate_limit": 3, "deep_analysis_limit": 3, "backup_limit": 2, "high_volatility_threshold": 0.55}, "xingtai": {}, "weipan": {"minute_candidate_limit": 3}}


def _forbidden_keys(value):
    forbidden = {"score_0_100", "ranking_score_0_100", "confidence", "weight", "weights", "ranking_details", "component_weights", "buy_decision", "minimum_recommendation_score"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                yield key
            yield from _forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _forbidden_keys(item)


def _run(monkeypatch, mode):
    context = ScopeContext(mode)
    pool = SimpleNamespace(data=context.frame.copy(), metadata={"source": "tushare_daily_cross_section", "as_of": "2026-09-11"})
    monkeypatch.setattr(xuangu_fenxi, "huoqu_houxuanchi_celue", lambda scope: SimpleNamespace(goujian=lambda *args: pool))
    monkeypatch.setattr(xuangu_fenxi, "goujian_fenxi_yinzi_mianban", lambda histories, profiles, **kwargs: (profiles.assign(trade_date=pd.Timestamp("2026-09-11")), {"status": "ok", "source": "offline_panel"}))
    factor = {"status": "ok", "groups": {
        "trend_structure": {"values": {"ma_gap_20": 0.04, "ma_trend_5_20": 0.02}},
        "momentum_reversal": {"values": {"ret_5": 0.05, "ret_20": 0.1, "macd_hist_pct": 0.01}},
        "relative_strength": {"values": {"excess_vs_csi300_ret_20": 0.05}},
        "price_volume_confirmation": {"values": {"volume_ratio_5_20": 1.2}},
        "risk_liquidity": {"values": {"volatility_20": 0.2}},
    }}
    if mode == "no_qualified":
        factor["groups"]["momentum_reversal"]["values"]["ret_5"] = -0.03
    if mode == "missing_index":
        factor["groups"]["relative_strength"]["values"]["excess_vs_csi300_ret_20"] = None
    monkeypatch.setattr(xuangu_fenxi, "huizong_houxuan_yinzi", lambda panel, **kwargs: {code: deepcopy(factor) for code in panel.ts_code})
    monkeypatch.setattr(xuangu_fenxi, "fenxi_zhangting_huimaqiang", lambda *args, **kwargs: {"status": "ok", "state": "close_confirmed", "state_label": "收盘确认"})
    monkeypatch.setattr(xuangu_fenxi, "fenxi_weipan", lambda *args, **kwargs: {"status": "ok", "conditions": [], "actuals": {"available_daily_evidence": True}})

    def technical(*args, **kwargs):
        if mode == "program_error":
            raise TypeError("真实离线技术异常")
        return {"status": "partial" if mode == "partial_technical" else "ok", "outcome": "information_partial" if mode == "partial_technical" else "analysis_success", "macd": {"histogram": 0.1}, "close": 12.0, "returns": {"5d": 0.05, "20d": 0.1}}

    monkeypatch.setattr(xuangu_fenxi, "zongjie_jishu", technical)
    result = xuangu_fenxi.XuanguFenxiFuWu(config=_config(), context=context).fenxi(FenxiFanwei.create("all_market", None))
    return result, context


@pytest.mark.parametrize("mode", ["ready", "no_qualified", "missing_quote", "program_error", "missing_index", "partial_technical", "financial_failure", "minute_failure", "snapshot_exception"])
def test_scope_condition_selection_retains_evidence_and_has_no_scores(monkeypatch, mode):
    result, context = _run(monkeypatch, mode)
    assert len(result["reviewed_candidates"]) == 3
    assert not list(_forbidden_keys(result))
    for candidate in result["reviewed_candidates"]:
        assert len(candidate["supplemental_diagnostics"]["blocks"]) == 7
        assert candidate["daily_factor_analysis"]["groups"]["trend_structure"]["values"]["ma_gap_20"] == 0.04
        assert "selection_analysis" in candidate
        assert "conditions" in candidate["selection_analysis"]
    if mode == "program_error":
        assert result["status"] == "error"
        assert result["outcome"] == "program_error"
        assert result["recommendation_available"] is False
        assert all(error["reason"] == "真实离线技术异常" for error in result["technical_review_errors"])
    elif mode in {"no_qualified", "missing_quote", "missing_index", "snapshot_exception"}:
        assert result["recommendation_available"] is False
        assert result["primary"] is None
        assert result["diagnostic_candidates"][0]["diagnostic_role"] == "observation_only"
        assert all(not candidate["meets_selection_conditions"] for candidate in result["reviewed_candidates"])
    else:
        assert result["recommendation_available"] is True
        assert result["primary"]["ts_code"] == context.codes[0]
    if mode == "minute_failure":
        for candidate in result["reviewed_candidates"]:
            assert candidate["late_session_analysis"]["actuals"]["available_daily_evidence"] is True
            assert candidate["late_session_analysis"]["minute_data"]["status"] == "unavailable"
    if mode == "financial_failure":
        assert all("财务源请求抛出异常" in str(candidate["fundamental_analysis"]) for candidate in result["reviewed_candidates"])
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_equal_pareto_dimensions_have_one_front_and_only_stable_code_order(monkeypatch):
    result, context = _run(monkeypatch, "ready")
    candidates = result["reviewed_candidates"]
    assert [candidate["ts_code"] for candidate in candidates] == context.codes
    assert [candidate["selection_analysis"]["pareto_front"] for candidate in candidates] == [1, 1, 1]
    assert all("不代表上涨概率" in candidate["selection_analysis"]["ranking_basis"] for candidate in candidates)
