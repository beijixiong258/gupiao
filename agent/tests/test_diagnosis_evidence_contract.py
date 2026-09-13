"""诊断证据语义回归：行业、估值、行情时点以及技术证据之间的分歧。"""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from src.agent.fenxi_zhanshi import buquan_zhengju_baogao
from src.ashare.dangu_zhaiyao import goujian_dangu_zhaiyao
from src.ashare.fanwei_faxian import BankuaiLeixing, ShichangFanwei
from src.ashare.fenxi_weipan import fenxi_weipan
from src.ashare.fenxi_yinzi import huizong_houxuan_yinzi, jisuan_hengjiemian_jibenmian
from src.ashare.shichang_shuju import FenxiShujuShangxiawen, _normalize_constituents, _normalize_snapshot
from src.ashare.xuangu_fanwei import FenxiFanwei, MingmingFanweiHouXuanChi, YijiexiFenxiFanwei
from src.ashare.xuangu_guize import goujian_houxuan_zhaiyao, goujian_kejiaoyixing_zhaiyao, goujian_kuaizhao_jilu


@pytest.mark.parametrize("kind,verified,expected", [
    (BankuaiLeixing.GAINIAN, True, ""), (BankuaiLeixing.HANGYE, False, ""),
    (BankuaiLeixing.HANGYE, True, "目录范围"),
])
def test_scope_members_do_not_gain_a_fabricated_industry(kind, verified, expected):
    scope = ShichangFanwei("用户范围", "目录范围", "BK0001", kind, "fixture", "2026-09-11", 1, "exact", "verified",
                          verification={"verified": verified})
    members = pd.DataFrame({"ts_code": ["000021.SZ", "000022.SZ"], "industry": [None, "已知行业"]})
    context = SimpleNamespace(bankuai_chengfen=lambda scope: (members, {}))
    result = MingmingFanweiHouXuanChi().goujian(YijiexiFenxiFanwei(FenxiFanwei.create("named_scope", "用户范围"), scope), context)
    assert result.data.industry.tolist() == [expected, "已知行业"]
    assert pd.isna(members.iloc[0].industry)
    assert result.metadata["industry_missing_rows"] == (0 if expected else 1)


def test_dynamic_pe_never_enters_ttm_comparison():
    raw = pd.DataFrame({"代码": [f"00000{i}" for i in range(1, 7)], "名称": ["样本"] * 6,
                        "市盈率-动态": [10, 20, 30, 40, 50, 60], "市净率": [1] * 6})
    normalized = _normalize_snapshot(raw)
    assert normalized.pe_dynamic.tolist() == [10, 20, 30, 40, 50, 60]
    assert "pe_ttm" not in normalized
    report = jisuan_hengjiemian_jibenmian(normalized)["000001.SZ"]
    assert report["values"]["pe_dynamic"] == 10
    assert report["relative_valuation"]["pe_ttm"]["percentile"] is None
    unspecified = _normalize_constituents(pd.DataFrame({"code": ["000001"], "name": ["样本"], "per": [15]}))
    assert unspecified.iloc[0].pe_unspecified == 15
    assert "pe_ttm" not in unspecified


def test_provider_timestamp_survives_request_memory_and_candidate_projection():
    raw = pd.DataFrame({"代码": ["000021"], "名称": ["样本"], "最新价": [12], "f124": [1789114301]})
    data = _normalize_snapshot(raw)
    context = FenxiShujuShangxiawen(reference=datetime(2026, 9, 12))
    context._memo["shishi_kuaizhao"] = (data, {"provider_trade_date": None, "source": "fixture", "status": "ok"})
    single = context.dangu_kuaizhao("000021.SZ")
    selected = goujian_kuaizhao_jilu(data.iloc[0], {"provider_trade_date": None})
    for snapshot in (single, selected):
        assert snapshot["provider_quote_time"] == "2026-09-11 16:11:41"
        assert snapshot["provider_trade_date"] == "2026-09-11"


def _quote(timestamp):
    return {"status": "ok", "last_price": 12, "previous_close": 11, "open": 11.5, "high": 12.1, "low": 11.2,
            "captured_at": "2026-09-14 09:40:00", "provider_quote_time": timestamp}


@pytest.mark.parametrize("timestamp,verified,state", [
    (None, False, "unavailable"), ("2026-09-11 15:00:00", False, "stale"),
    ("2026-09-15 09:35:00", False, "future"), ("2026-09-14 09:50:00", False, "future"),
    ("2026-09-14 09:39:30", True, "verified"),
])
def test_current_tradability_needs_provider_time_not_just_positive_prices(timestamp, verified, state):
    history = pd.DataFrame([{"trade_date": "2026-09-11", "close": 10, "amount_yuan": 1e8}])
    result = goujian_kejiaoyixing_zhaiyao(code="000021.SZ", name="样本", snapshot=_quote(timestamp), history=history,
                                       minimum_amount=5e7, realtime_required=True, reference_time="2026-09-14 09:40:00")
    assert result["current_quote_verified"] is verified
    assert result["basic_execution_feasible"] is verified
    assert result["quote_time_verification"]["status"] == state
    assert result["analysis_price"] == (12 if verified else 10)
    if verified:
        assert result["quote_time_verification"]["quote_age_seconds"] == 30


def test_stale_quote_cannot_produce_late_session_confirmation():
    result = fenxi_weipan(pd.DataFrame(), snapshot=_quote("2026-09-11 15:00:00"),
                         clock={"is_trading_day": True, "captured_at": "2026-09-14 14:35:00"}, config={})
    assert result["status"] == "unavailable"
    assert result["confirmation_level"] == "not_confirmed"
    assert not result.get("conditions")


def test_preclose_quote_is_not_published_as_a_completed_close():
    result = fenxi_weipan(pd.DataFrame(), snapshot=_quote("2026-09-14 14:58:00"),
                         clock={"is_trading_day": True, "captured_at": "2026-09-14 15:10:00"}, config={})
    assert result["status"] == "unavailable"
    assert result["confirmation_level"] == "not_confirmed"
    assert "收盘之前" in result["reason"]


def test_ai_receives_canonical_source_and_input_requirements():
    report = huizong_houxuan_yinzi(pd.DataFrame([{"ts_code": "000021.SZ", "ret_5": 0.1}]), config={})["000021.SZ"]
    definitions = report["groups"]["relative_strength"]["metric_definitions"]
    assert definitions["excess_ret_5"]["canonical_source"] == definitions["excess_vs_universe_ret_5"]["canonical_source"]
    assert definitions["excess_ret_5"]["frequency"] == "daily_k"
    assert "input_requirement" in definitions["excess_ret_5"]


def test_summary_preserves_price_volume_and_market_conflicts_and_active_reassessment():
    technical = {"close": 12, "moving_averages": {"ma20": 11, "ma60": 13}, "returns": {"5d": 0.1, "20d": 0.04},
                 "volume_ratio_5_to_20": 0.8,
                 "macd_structure": {"status": "ok", "latest_cross": {"status": "expired", "invalidation_condition": "旧交叉条件"},
                                    "invalidation_conditions": ["当前背离失效条件"], "supporting_evidence": ["动能改善"]}}
    summary = goujian_dangu_zhaiyao(technical=technical, fundamentals={}, risks=[], evidence_gaps=[], as_of="2026-09-11",
                                   factor_analysis={"groups": {"market_context": {"values": {"market_regime_weak": 1}}}},
                                   pattern={"eligible": False, "state": "none", "risk_reference_price": 1})
    assert len(summary["evidence_conflicts"]) == 3
    assert "20日均线" in summary["summary"] and "60日均线" in summary["summary"]
    assert "均量" in summary["summary"] and "市场" in summary["summary"]
    assert "当前背离失效条件" in summary["reassessment_conditions"]
    assert "旧交叉条件" not in summary["reassessment_conditions"]
    assert not any("风险参考价" in text for text in summary["reassessment_conditions"])
    report = buquan_zhengju_baogao("", {"analysis_type": "single_stock_analysis", "status": "partial",
                                "stock": {"name": "样本", "ts_code": "000021.SZ"}, "diagnosis_summary": summary,
                                "technical_summary": technical})
    assert "技术证据与分歧" in report and "均量" in report


def test_inapplicable_pattern_is_not_a_candidate_risk():
    candidate = goujian_houxuan_zhaiyao({"ts_code": "688001.SH", "name": "样本",
        "technical": {}, "pattern": {"eligible": False, "state": "none", "state_label": "不适用", "failure_reasons": ["不适用正常主板形态"]},
        "data_quality": {"as_of": "2026-09-11"}}, rank=1)
    assert "不适用正常主板形态" not in candidate["risks"]
    assert candidate["limit_up_pullback_pattern"]["failure_reasons"] == ["不适用正常主板形态"]
