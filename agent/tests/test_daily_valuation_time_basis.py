"""日终估值补充只消费明确同日的完整横截面，不混用实时原值。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ashare import shichang_shuju
from src.ashare.fenxi_yinzi import zengjia_dangri_guzhi_yinzi


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["600001.SH", "600002.SH"],
        "trade_date": pd.to_datetime(["2026-09-11", "2026-09-11"]),
        "turnover_rate_daily": [np.nan, np.nan],
        "log_circ_mv": [np.nan, np.nan],
    })


def _daily_profiles() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["600001.SH", "600002.SH"],
        "trade_date": pd.to_datetime(["2026-09-11", "2026-09-11"]),
        "valuation_trade_date": pd.to_datetime(["2026-09-11", "2026-09-11"]),
        "valuation_source": ["tushare_daily_basic", "tushare_daily_basic"],
        "valuation_is_complete_daily": [True, True],
        "turnover_rate": [2.0, 4.0],
        "circulating_market_value_yuan": [1e9, 2e9],
    })


def test_verified_daily_valuation_fills_missing_values_and_recomputes_ranks() -> None:
    panel, profiles = _panel(), _daily_profiles()
    original_panel, original_profiles = panel.copy(deep=True), profiles.copy(deep=True)

    result = zengjia_dangri_guzhi_yinzi(panel, profiles)

    assert result["turnover_rate_daily"].tolist() == [0.02, 0.04]
    np.testing.assert_allclose(result["log_circ_mv"], np.log([1e9, 2e9]))
    assert result["rank_turnover_rate_daily"].tolist() == [0.5, 1.0]
    assert result["rank_log_circ_mv"].tolist() == [0.5, 1.0]
    assert not any(column.startswith("_snapshot_") for column in result.columns)
    pd.testing.assert_frame_equal(panel, original_panel)
    pd.testing.assert_frame_equal(profiles, original_profiles)


@pytest.mark.parametrize("case", ["unknown_time", "intraday", "incomplete_day", "other_day", "valuation_date_conflict", "missing_date"])
def test_unverified_or_different_day_valuation_cannot_enter_daily_factors(case: str) -> None:
    panel, profiles = _panel(), _daily_profiles()
    if case == "unknown_time":
        profiles = profiles.drop(columns=["valuation_trade_date", "valuation_is_complete_daily"])
    elif case == "intraday":
        profiles["valuation_source"] = "eastmoney_live_a_share_snapshot"
    elif case == "incomplete_day":
        profiles["valuation_is_complete_daily"] = False
    elif case == "other_day":
        profiles["trade_date"] = pd.Timestamp("2026-09-14")
        profiles["valuation_trade_date"] = pd.Timestamp("2026-09-14")
    elif case == "valuation_date_conflict":
        profiles["valuation_trade_date"] = pd.Timestamp("2026-09-10")
    else:
        profiles["valuation_trade_date"] = pd.NaT
    original_profiles = profiles.copy(deep=True)

    result = zengjia_dangri_guzhi_yinzi(panel, profiles)

    assert result["turnover_rate_daily"].isna().all()
    assert result["log_circ_mv"].isna().all()
    pd.testing.assert_frame_equal(profiles, original_profiles)


def test_values_match_both_stock_code_and_daily_date() -> None:
    panel = _panel().iloc[[0, 0]].reset_index(drop=True)
    panel["trade_date"] = pd.to_datetime(["2026-09-10", "2026-09-11"])
    profiles = _daily_profiles().iloc[[0, 0]].reset_index(drop=True)
    profiles["trade_date"] = panel["trade_date"]
    profiles["valuation_trade_date"] = panel["trade_date"]
    profiles["turnover_rate"] = [2.0, 8.0]

    result = zengjia_dangri_guzhi_yinzi(panel, profiles)

    assert len(result) == 2
    assert result["turnover_rate_daily"].tolist() == [0.02, 0.08]


def test_existing_valid_history_is_kept_when_snapshot_differs_or_has_missing_values() -> None:
    panel, profiles = _panel(), _daily_profiles()
    panel["turnover_rate_daily"] = [0.0, 0.08]
    panel["log_circ_mv"] = np.log([3e9, 4e9])
    profiles.loc[1, ["turnover_rate", "circulating_market_value_yuan"]] = np.nan

    result = zengjia_dangri_guzhi_yinzi(panel, profiles)

    assert result["turnover_rate_daily"].tolist() == [0.0, 0.08]
    np.testing.assert_allclose(result["log_circ_mv"], panel["log_circ_mv"])


def test_invalid_supplements_stay_missing_and_real_zero_turnover_is_allowed() -> None:
    panel, profiles = _panel(), _daily_profiles()
    profiles["turnover_rate"] = [0.0, -1.0]
    profiles["circulating_market_value_yuan"] = [0.0, np.inf]

    result = zengjia_dangri_guzhi_yinzi(panel, profiles)

    assert result.loc[0, "turnover_rate_daily"] == 0.0
    assert pd.isna(result.loc[1, "turnover_rate_daily"])
    assert result["log_circ_mv"].isna().all()
    assert result["rank_turnover_rate_daily"].isna().all()


def test_cross_section_keeps_row_dates_and_rejects_mismatched_valuation_day(monkeypatch) -> None:
    daily = pd.DataFrame({
        "ts_code": ["600001.SH", "600002.SH", "600003.SH"],
        "trade_date": ["20260911"] * 3,
        "close": [10.0] * 3,
    })
    valuation = pd.DataFrame({
        "ts_code": daily["ts_code"],
        "trade_date": ["20260911", "20260910", None],
        "turnover_rate": [2.0, 3.0, 4.0],
        "circ_mv": [100000.0, 200000.0, 300000.0],
    })
    monkeypatch.setattr(shichang_shuju, "_tushare_pro", lambda: SimpleNamespace(daily_basic=lambda **kwargs: valuation))
    monkeypatch.setattr(shichang_shuju, "_latest_tushare_daily", lambda pro, date: ("20260911", daily))

    profiles, metadata = shichang_shuju.huoqu_zuixin_hengjiemian(
        pd.Timestamp("2026-09-11"),
        stock_basic_loader=lambda: (pd.DataFrame({"ts_code": daily["ts_code"], "name": ["样本"] * 3}), {}),
    )
    panel = _panel()
    result = zengjia_dangri_guzhi_yinzi(panel, profiles)

    assert metadata["source"] == "tushare_daily_cross_section"
    assert profiles["trade_date"].eq(pd.Timestamp("2026-09-11")).all()
    assert profiles["valuation_is_complete_daily"].tolist() == [True, False, False]
    assert profiles["turnover_rate"].tolist() == [2.0, 3.0, 4.0]
    assert profiles["circulating_market_value_yuan"].tolist() == [1e9, 2e9, 3e9]
    assert result.loc[0, "turnover_rate_daily"] == 0.02
    assert pd.isna(result.loc[1, "turnover_rate_daily"])
    assert any("2 只股票" in warning for warning in metadata["warnings"])


def test_live_fallback_retains_raw_values_without_claiming_daily_valuation(monkeypatch) -> None:
    def fail_tushare():
        raise RuntimeError("离线日终源不可用")

    realtime = pd.DataFrame({
        "ts_code": ["600001.SH"],
        "provider_trade_date": ["2026-09-14"],
        "provider_quote_time": ["2026-09-14 10:00:00"],
        "turnover_rate": [2.0],
        "circulating_market_value_yuan": [1e9],
    })
    original = realtime.copy(deep=True)
    monkeypatch.setattr(shichang_shuju, "_tushare_pro", fail_tushare)

    profiles, metadata = shichang_shuju.huoqu_zuixin_hengjiemian(
        pd.Timestamp("2026-09-11"),
        realtime_loader=lambda: (realtime, {"source": "eastmoney_live_a_share_snapshot"}),
    )
    result = zengjia_dangri_guzhi_yinzi(_panel(), profiles)

    assert metadata["status"] == "degraded"
    pd.testing.assert_frame_equal(profiles, original)
    assert result["turnover_rate_daily"].isna().all()
    assert result["log_circ_mv"].isna().all()
