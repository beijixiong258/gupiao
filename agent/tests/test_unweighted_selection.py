"""Offline contracts for raw evidence and condition/Pareto stock selection."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.ashare.fenxi_yinzi import huizong_houxuan_yinzi, jisuan_hengjiemian_jibenmian, zengjia_hengjiemian_yinzi
from src.ashare.xuangu_guize import (
    choushu_liudongxing_houxuan,
    goujian_houxuan_zhaiyao,
    goujian_kejiaoyixing_zhaiyao,
    guolv_lishi_wanzhengxing,
    hebing_jibenmian_zhengju,
    jichu_ying_guolv,
    paixu_shangzhang_houxuan,
    pinggu_shangzhang_tiaojian,
    shishi_ying_guolv,
)
from src.ashare.yinzi_gongcheng import FACTOR_GROUPS


CONFIG = {"fenxi": {"high_volatility_threshold": 0.55}}


def _candidate(code="600001.SH", *, excess=0.03, ret5=0.02, ret20=0.06, volatility=0.2, amount=1e8):
    return {
        "ts_code": code,
        "name": "测试股票",
        "factor": {"groups": {
            "trend_structure": {"values": {"ma_gap_20": 0.02, "ma_trend_5_20": 0.01}},
            "momentum_reversal": {"values": {"ret_5": ret5, "ret_20": ret20, "macd_hist_pct": 0.001}},
            "relative_strength": {"values": {"excess_vs_csi300_ret_20": excess}},
            "price_volume_confirmation": {"values": {"volume_ratio_5_20": 1.2}},
            "risk_liquidity": {"values": {"volatility_20": volatility}},
        }},
        "history": pd.DataFrame([{"trade_date": pd.Timestamp("2026-09-11"), "amount_yuan": amount}]),
        "technical": {"status": "ok", "macd": {"histogram": 0.03}, "macd_structure": {"status": "ok"}},
        "tradability": {"basic_execution_feasible": True, "realtime_required": True, "current_quote_verified": True},
        "fundamental": {"status": "partial", "financials": {"roe_pct": 12.0}, "valuation": {"pe_ttm": 15.0}},
        "late": {"status": "not_applicable"},
        "pattern": {"state": "no_pattern"},
    }


def _assert_no_scores(value):
    if isinstance(value, dict):
        for key, child in value.items():
            assert not any(token in key.lower() for token in ("score", "weight", "confidence", "probability")), key
            _assert_no_scores(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_scores(child)


def test_all_registered_raw_values_and_missing_fields_survive_without_rounding_or_scores():
    original = 0.0000000123456789
    panel = pd.DataFrame([{"ts_code": "600001.SH", "ret_5": original, "wvma_20": 0.3, "ma_gap_20": np.nan}])
    result = huizong_houxuan_yinzi(panel, config={})["600001.SH"]
    assert result["status"] == "partial"
    assert result["available_factor_count"] == 2
    assert result["total_factor_count"] == sum(map(len, FACTOR_GROUPS.values()))
    for group, members in FACTOR_GROUPS.items():
        assert list(result["groups"][group]["values"]) == list(members)
        assert result["groups"][group]["factor_count"] == len(members)
    assert result["groups"]["momentum_reversal"]["values"]["ret_5"] == original
    assert "ma_gap_20" in result["groups"]["trend_structure"]["missing_fields"]
    assert result["groups"]["price_volume_confirmation"]["values"]["wvma_20"] == 0.3
    assert result == huizong_houxuan_yinzi(panel, config={"factor_group_weights": {"momentum_reversal": 1e9}})["600001.SH"]
    _assert_no_scores(result)


def test_single_stock_legacy_comparison_background_is_missing():
    panel = pd.DataFrame([{
        "ts_code": "600001.SH", "trade_date": pd.Timestamp("2026-09-11"), "open": 10, "high": 11,
        "low": 9, "close": 10.5, "amount_yuan": 1e8, "ret_1": 0.01, "ret_5": 0.03,
        "ret_20": 0.06, "ma_gap_20": 0.02, "volume_ratio_5_20": 1.2, "volatility_20": 0.2,
    }])
    result = zengjia_hengjiemian_yinzi(panel)
    assert result[["peer_mean_ret_5", "excess_ret_5", "peer_breadth_above_ma20", "rank_ret_5"]].isna().all().all()
    assert result.loc[0, "ret_5"] == 0.03


def test_relative_valuation_is_plain_percentile_and_preserves_missing_sources():
    panel = pd.DataFrame({"ts_code": [f"60000{i}.SH" for i in range(6)], "industry": [""] * 6,
                          "pe_ttm": [10, 20, 30, 40, 50, 60], "pb": [1, 2, 3, 4, 5, 6], "source": "fixture"})
    result = jisuan_hengjiemian_jibenmian(panel)["600000.SH"]
    assert result["values"]["pe_ttm"] == 10
    assert result["relative_valuation"]["pe_ttm"]["percentile"] == pytest.approx(100 / 6, abs=0.0001)
    assert result["relative_valuation"]["pe_ttm"]["comparison_source"] == "current_comparison_pool"
    assert result["sources"]["source"] == "fixture"
    single = jisuan_hengjiemian_jibenmian(panel.iloc[:1])["600000.SH"]
    assert single["relative_valuation"]["pe_ttm"]["percentile"] is None
    deep = {"status": "error", "outcome": "program_error", "errors": ["upstream failure"], "financials": {"roe_pct": 12}}
    merged = hebing_jibenmian_zhengju(result, deep)
    assert merged["status"] == "error" and merged["errors"] == ["upstream failure"]
    assert merged["financials"] == deep["financials"]
    _assert_no_scores(merged)


def test_all_bullish_conditions_are_explicit_and_configured_volatility_is_a_limit():
    item = _candidate()
    result = pinggu_shangzhang_tiaojian(item, config=CONFIG)
    assert result["eligible"] is True
    assert len(result["conditions"]) == 5
    assert {condition["status"] for condition in result["conditions"]} == {"met"}
    assert result["unmet_conditions"] == result["missing_conditions"] == []
    constrained = pinggu_shangzhang_tiaojian(item, config={"fenxi": {"high_volatility_threshold": 0.1}})
    assert constrained["unmet_conditions"] == ["bounded_volatility"]
    _assert_no_scores(result)


@pytest.mark.parametrize("group,field,key", [
    ("trend_structure", "ma_gap_20", "trend_alignment"),
    ("trend_structure", "ma_trend_5_20", "trend_alignment"),
    ("momentum_reversal", "ret_5", "positive_momentum"),
    ("momentum_reversal", "ret_20", "positive_momentum"),
    ("momentum_reversal", "macd_hist_pct", "positive_momentum"),
    ("relative_strength", "excess_vs_csi300_ret_20", "outperform_csi300"),
    ("price_volume_confirmation", "volume_ratio_5_20", "volume_confirmation"),
    ("risk_liquidity", "volatility_20", "bounded_volatility"),
])
def test_each_required_missing_value_blocks_eligibility(group, field, key):
    item = _candidate()
    item["factor"]["groups"][group]["values"][field] = np.nan
    result = pinggu_shangzhang_tiaojian(item, config=CONFIG)
    assert result["eligible"] is False
    assert result["missing_conditions"] == [key]
    assert result["unmet_conditions"] == []


def test_zero_is_not_positive_but_equal_volume_and_volatility_limits_pass():
    item = _candidate(excess=0)
    item["factor"]["groups"]["price_volume_confirmation"]["values"]["volume_ratio_5_20"] = 1
    item["factor"]["groups"]["risk_liquidity"]["values"]["volatility_20"] = 0.55
    result = pinggu_shangzhang_tiaojian(item, config=CONFIG)
    assert result["eligible"] is False
    assert result["unmet_conditions"] == ["outperform_csi300"]


def test_deep_review_preserves_program_error_and_never_passes_missing_current_quote():
    item = _candidate()
    item["technical"] = {"status": "error", "outcome": "program_error", "error": "actual MACD failure"}
    item["tradability"]["current_quote_verified"] = False
    result = pinggu_shangzhang_tiaojian(item, config=CONFIG, deep_reviewed=True)
    assert not result["eligible"]
    assert result["status"] == "error" and result["outcome"] == "program_error"
    assert result["missing_conditions"] == ["technical_review", "tradability"]
    assert next(condition for condition in result["conditions"] if condition["key"] == "technical_review")["reason"] == "actual MACD failure"


def test_partial_technical_with_required_macd_can_pass_but_missing_histogram_cannot():
    item = _candidate()
    item["technical"]["status"] = "partial"
    assert pinggu_shangzhang_tiaojian(item, config=CONFIG, deep_reviewed=True)["eligible"]
    item["technical"]["macd"]["histogram"] = None
    assert pinggu_shangzhang_tiaojian(item, config=CONFIG, deep_reviewed=True)["missing_conditions"] == ["technical_review"]


def test_pareto_layers_keep_tradeoffs_and_ignore_input_order_without_mutation():
    dominant = _candidate("600001.SH", excess=0.06, ret5=0.05, ret20=0.08, volatility=0.1, amount=2e8)
    dominated = _candidate("600002.SH", excess=0.02, ret5=0.03, ret20=0.06, volatility=0.2, amount=1e8)
    tradeoff = _candidate("600003.SH", excess=0.1, ret5=0.04, ret20=0.08, volatility=0.2, amount=1.5e8)
    input_items = [dominated, tradeoff, dominant]
    old_values = deepcopy(dominant["factor"])
    actual = paixu_shangzhang_houxuan(input_items, config=CONFIG)
    assert [item["ts_code"] for item in actual] == ["600001.SH", "600003.SH", "600002.SH"]
    assert [item["selection"]["pareto_front"] for item in actual] == [1, 1, 2]
    assert all("selection" not in item for item in input_items)
    assert dominant["factor"] == old_values
    assert actual is not input_items and actual[0] is not dominant
    assert [item["ts_code"] for item in paixu_shangzhang_houxuan(list(reversed(input_items)), config=CONFIG)] == [item["ts_code"] for item in actual]


def test_eligibility_precedes_pareto_and_incomplete_comparisons_are_last_without_fake_amount():
    passing = _candidate("600001.SH")
    failing = _candidate("600002.SH", excess=0.3, ret5=0.5, ret20=0.9, volatility=0.1, amount=3e8)
    failing["factor"]["groups"]["trend_structure"]["values"]["ma_gap_20"] = -0.01
    incomplete = _candidate("600003.SH", excess=0.5, ret5=0.8, ret20=0.9, amount=np.nan)
    incomplete["factor"]["groups"]["risk_liquidity"]["values"]["log_amount_yuan"] = 99
    actual = paixu_shangzhang_houxuan([incomplete, failing, passing], config=CONFIG)
    assert [item["ts_code"] for item in actual] == ["600001.SH", "600002.SH", "600003.SH"]
    assert actual[-1]["selection"]["pareto_front"] is None
    assert actual[-1]["selection"]["comparison_missing_fields"] == ["amount_yuan"]
    assert actual[-1]["selection"]["comparison_values"]["amount_yuan"] is None


def test_identical_vectors_use_code_only_as_stable_display_and_no_score_output():
    actual = paixu_shangzhang_houxuan([_candidate("600002.SH"), _candidate("600001.SH")], config=CONFIG, deep_reviewed=True)
    assert [item["ts_code"] for item in actual] == ["600001.SH", "600002.SH"]
    assert [item["selection"]["pareto_front"] for item in actual] == [1, 1]
    summary = goujian_houxuan_zhaiyao(actual[0], rank=1)
    assert summary["technical_summary"] == actual[0]["technical"]
    assert summary["fundamental_analysis"] == actual[0]["fundamental"]
    assert summary["meets_selection_conditions"]
    assert "不代表上涨概率优劣" in summary["selection_analysis"]["ranking_basis"]
    _assert_no_scores(summary)


def test_liquidity_sampling_preserves_industry_rounds_without_weighted_prefilter():
    panel = pd.DataFrame({"ts_code": ["A1", "A2", "B1", "B2"], "industry": ["A", "A", "B", "B"],
                          "amount_yuan": [10, 9, 2, 1]})
    result = choushu_liudongxing_houxuan(panel, 2)
    assert result["ts_code"].tolist() == ["A1", "B1"]
    assert result.columns.tolist() == panel.columns.tolist()


def test_realtime_one_price_limit_block_is_preserved_and_rechecked_each_time():
    quote = {"previous_close": 10, "last_price": 11, "open": 11, "high": 11, "low": 11}
    assert shishi_ying_guolv(quote, code="600001.SH", name="测试股票", config=CONFIG)
    quote.update(last_price=10.7, low=10.3)
    assert shishi_ying_guolv(quote, code="600001.SH", name="测试股票", config=CONFIG) == []


def test_completed_daily_hard_filter_keeps_name_volume_and_liquidity_blocks():
    row = pd.Series({"ts_code": "600001.SH", "name": "ST测试", "latest_price": 10,
                     "volume": 0, "amount_yuan": 1e6, "list_date": "2020-01-01"})
    reasons = jichu_ying_guolv(row, analysis_date=pd.Timestamp("2026-09-11"), config=CONFIG, quote_is_completed=True)
    assert len(reasons) == 3
    assert any("ST" in reason for reason in reasons)
    assert any("成交量" in reason for reason in reasons)
    assert any("成交额" in reason for reason in reasons)


def test_realtime_required_cannot_use_daily_history_to_assert_current_tradability():
    history = pd.DataFrame([{"trade_date": "2026-09-11", "amount_yuan": 1e8, "close": 10}])
    trade = goujian_kejiaoyixing_zhaiyao(code="600001.SH", name="测试股票",
                                      snapshot={"status": "unavailable", "error": "fixture outage"},
                                      history=history, minimum_amount=5e7, realtime_required=True)
    assert trade["basic_execution_feasible"] is False
    assert trade["current_quote_verified"] is False
    item = _candidate()
    item["tradability"] = trade
    assert "tradability" in pinggu_shangzhang_tiaojian(item, config=CONFIG, deep_reviewed=True)["missing_conditions"]


def test_stale_daily_history_cannot_enter_candidate_pool():
    history = pd.DataFrame({"trade_date": pd.bdate_range("2026-01-01", periods=180), "amount_yuan": 1e8})
    accepted, rejected = guolv_lishi_wanzhengxing({"600001.SH": history}, analysis_date=pd.Timestamp("2026-09-11"),
                                               minimum_rows=180, minimum_amount=5e7)
    assert not accepted
    assert rejected and "最新日线停留" in rejected[0]["reason"]
