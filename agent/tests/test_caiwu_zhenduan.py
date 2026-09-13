"""已有财务字段补充诊断的离线回归；不调用任何数据源。"""

from copy import deepcopy
from datetime import date, datetime
import json

import pandas as pd
import pytest

from src.ashare.caiwu_zhenduan import goujian_caiwu_zhenduan


def _fundamental():
    return {
        "as_of": "2026-08-20",
        "profile": {"industry": "电子"},
        "sources": {"financials": "tushare"},
        "financials": {
            "report_date": "2026-06-30",
            "announcement_date": "2026-08-10",
            "known_as_of": "2026-08-20",
            "revenue_yoy_pct": 12.5,
            "net_profit_yoy_pct": -8.5,
            "net_margin_pct": 7.2,
            "operating_cashflow_to_revenue_pct": -0.025,
        },
        "score_0_100": 68.5,
        "confidence": 0.675,
    }


def _blocks(fundamental, **kwargs):
    return {block["key"]: block for block in goujian_caiwu_zhenduan(fundamental, **kwargs)}


def test_existing_financial_evidence_is_explained_without_mutation_or_score():
    fundamental = _fundamental()
    before = deepcopy(fundamental)
    blocks = _blocks(fundamental)

    assert fundamental == before
    assert set(blocks) == {"financial_growth", "cashflow_quality", "report_timeliness"}
    assert all(set(block) == {"key", "label", "status", "summary", "metrics", "missing_reason"} for block in blocks.values())
    assert all("score" not in str(block) and "confidence" not in str(block) for block in blocks.values())
    growth = blocks["financial_growth"]
    assert growth["status"] == "ok"
    assert "12.5%" in growth["summary"] and "-8.5%" in growth["summary"]
    assert "营收增长、利润下降" in growth["summary"]
    assert "不能证明趋势改善" in growth["summary"]
    json.dumps(list(blocks.values()), ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("revenue,profit,revenue_sign,profit_sign", [
    (3, 2, "positive", "positive"),
    (3, -2, "positive", "negative"),
    (-3, 2, "negative", "positive"),
    (-3, -2, "negative", "negative"),
    (0, 0, "zero", "zero"),
    (0, 2, "zero", "positive"),
    (-3, 0, "negative", "zero"),
])
def test_growth_combinations_keep_zero_separate_from_missing(revenue, profit, revenue_sign, profit_sign):
    fundamental = _fundamental()
    fundamental["financials"].update(revenue_yoy_pct=revenue, net_profit_yoy_pct=profit)
    growth = _blocks(fundamental)["financial_growth"]
    assert growth["status"] == "ok"
    assert growth["metrics"]["revenue_growth_sign"] == revenue_sign
    assert growth["metrics"]["net_profit_growth_sign"] == profit_sign
    assert growth["missing_reason"] is None
    if revenue == 0 or profit == 0:
        assert "同比持平" in growth["summary"]


def test_cashflow_is_sign_only_and_keeps_financial_industry_limitation():
    fundamental = _fundamental()
    fundamental["profile"]["industry"] = "银行"
    cash = _blocks(fundamental)["cashflow_quality"]
    assert cash["status"] == "ok"
    assert cash["metrics"]["operating_cashflow_to_revenue_raw"] == -0.025
    assert cash["metrics"]["cashflow_unit_status"] == "unverified"
    assert cash["metrics"]["cashflow_evidence_basis"] == "sign_only"
    assert cash["metrics"]["industry"] == "银行"
    assert "7.2%" in cash["summary"]
    assert "经营现金流指标为负" in cash["summary"]
    assert "两项符号背离" in cash["summary"]
    assert "-0.025" not in cash["summary"]
    assert "金融类公司的现金流结构特殊" in cash["summary"]
    assert "现金利润比" not in str(cash)


@pytest.mark.parametrize("margin,cashflow,sign", [(0, 0, "zero"), (-2, 0, "zero"), (-2, 0.1, "positive"), (2, 0.1, "positive")])
def test_cashflow_zero_and_sign_combinations(margin, cashflow, sign):
    fundamental = _fundamental()
    fundamental["financials"].update(net_margin_pct=margin, operating_cashflow_to_revenue_pct=cashflow)
    cash = _blocks(fundamental)["cashflow_quality"]
    assert cash["status"] == "ok"
    assert cash["metrics"]["operating_cashflow_sign"] == sign
    assert cash["missing_reason"] is None
    if cashflow == 0:
        assert "为零" in cash["summary"]


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf"), "--", True])
def test_missing_or_nonfinite_numbers_are_never_zero(invalid):
    fundamental = _fundamental()
    fundamental["financials"].update(revenue_yoy_pct=invalid, operating_cashflow_to_revenue_pct=invalid)
    blocks = _blocks(fundamental)
    growth, cash = blocks["financial_growth"], blocks["cashflow_quality"]
    assert growth["status"] == cash["status"] == "partial"
    assert growth["metrics"]["revenue_yoy_pct"] is None
    assert growth["metrics"]["revenue_growth_sign"] is None
    assert cash["metrics"]["operating_cashflow_to_revenue_raw"] is None
    assert cash["metrics"]["operating_cashflow_sign"] is None
    assert growth["missing_reason"] and cash["missing_reason"]
    json.dumps(list(blocks.values()), ensure_ascii=False, allow_nan=False)


def test_completely_missing_financials_remain_unavailable():
    blocks = _blocks({}, as_of_date="2026-08-20")
    assert all(block["status"] == "unavailable" for block in blocks.values())
    assert all(block["missing_reason"] for block in blocks.values())
    assert blocks["report_timeliness"]["metrics"]["point_in_time_verified"] is False


def test_report_calendar_day_intervals_are_exact_without_age_threshold():
    report = _blocks(_fundamental())["report_timeliness"]
    assert report["status"] == "ok"
    assert report["metrics"]["calendar_days_since_report"] == 51
    assert report["metrics"]["calendar_days_since_announcement"] == 10
    assert report["metrics"]["announcement_lag_calendar_days"] == 41
    assert report["metrics"]["point_in_time_verified"] is True
    assert "51 个自然日" in report["summary"]
    assert "距公告日 10 个自然日" in report["summary"]


@pytest.mark.parametrize("missing_field", ["announcement_date", "source", "known_as_of"])
def test_incomplete_report_provenance_is_never_marked_verified(missing_field):
    fundamental = _fundamental()
    if missing_field == "source":
        fundamental["sources"] = {}
    else:
        fundamental["financials"].pop(missing_field)
    if missing_field == "announcement_date":
        fundamental["sources"]["financials"] = "akshare"
        fundamental["financials"]["announcement_date_status"] = "备用来源没有公告日，仅为当前降级参考"
    report = _blocks(fundamental)["report_timeliness"]
    assert report["status"] == "partial"
    assert report["metrics"]["point_in_time_verified"] is False
    assert report["missing_reason"]
    if missing_field == "announcement_date":
        assert "备用来源没有公告日" in report["missing_reason"]


@pytest.mark.parametrize("future_field", ["report_date", "announcement_date", "known_as_of"])
def test_future_dated_financials_are_not_used_for_current_diagnosis(future_field):
    fundamental = _fundamental()
    fundamental["financials"][future_field] = "2026-08-21"
    blocks = _blocks(fundamental)
    assert all(block["status"] == "unavailable" for block in blocks.values())
    assert "晚于分析日" in blocks["report_timeliness"]["summary"]
    assert blocks["report_timeliness"]["metrics"]["point_in_time_verified"] is False
    assert blocks["financial_growth"]["metrics"]["revenue_yoy_pct"] == 12.5
    assert "不可用于当前诊断" in blocks["financial_growth"]["summary"]
    assert "不可用于当前诊断" in blocks["cashflow_quality"]["summary"]


def test_date_order_conflicts_are_exposed():
    fundamental = _fundamental()
    fundamental["financials"]["announcement_date"] = "2026-06-20"
    report = _blocks(fundamental)["report_timeliness"]
    assert report["status"] == "unavailable"
    assert report["metrics"]["announcement_lag_calendar_days"] == -10
    assert "公告日早于报告期" in report["summary"]


def test_missing_analysis_date_does_not_use_today():
    fundamental = _fundamental()
    fundamental.pop("as_of")
    fundamental["financials"].pop("known_as_of")
    report = _blocks(fundamental)["report_timeliness"]
    assert report["status"] == "partial"
    assert report["metrics"]["analysis_date"] is None
    assert report["metrics"]["calendar_days_since_report"] is None
    assert report["metrics"]["point_in_time_verified"] is False


def test_deep_fallback_shape_and_existing_datetime_types():
    fundamental = _fundamental()
    financials = fundamental["financials"]
    financials.update(report_date=date(2026, 6, 30), announcement_date=pd.Timestamp("2026-08-10"), known_as_of=datetime(2026, 8, 20, 15, 5))
    nested = {"deep_analysis": fundamental, "sources": {"profile": "tushare"}}
    report = _blocks(nested)["report_timeliness"]
    assert report["status"] == "ok"
    assert report["metrics"]["source"] == "tushare"
    financials["announcement_date"] = pd.NaT
    report = _blocks(nested)["report_timeliness"]
    assert report["status"] == "partial"
    assert report["metrics"]["announcement_date"] is None
