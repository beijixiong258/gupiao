"""Offline contracts for raw factor evidence and genuine industry context."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ashare import riping_yinzi
from src.ashare.fenxi_yinzi import huizong_houxuan_yinzi
from src.ashare.yinzi_gongcheng import (
    FACTOR_GROUPS,
    FACTOR_ALIASES,
    add_price_volume_factors,
    factor_registry_rows,
    independent_factor_features,
)


def _history() -> pd.DataFrame:
    returns = np.tile(np.array([-0.009, 0.004, 0.012, -0.003, 0.007]), 8)
    close = 20 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2025-01-02", periods=len(close)),
            "ts_code": "600001.SH",
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
            "amount_yuan": 1e8 * np.exp(np.cumsum(-2 * returns)),
            "turnover_rate": 10 + np.cumsum(100 * returns),
        }
    )


def test_turnover_uses_real_daily_observations_instead_of_amount_proxy() -> None:
    factors = add_price_volume_factors(_history())
    latest = factors.iloc[-1]
    assert latest["price_turnover_corr_20"] == pytest.approx(1.0)
    assert latest["return_amount_corr_20"] < -0.99


@pytest.mark.parametrize("current_snapshot_only", [False, True])
def test_missing_turnover_history_stays_unavailable(current_snapshot_only: bool) -> None:
    history = _history().drop(columns="turnover_rate")
    if current_snapshot_only:
        history["turnover_rate"] = np.nan
        history.loc[history.index[-1], "turnover_rate"] = 5.0
    factors = add_price_volume_factors(history)
    assert factors["return_amount_corr_20"].notna().any()
    assert factors["price_turnover_corr_20"].isna().all()


def _evidence_panel() -> pd.DataFrame:
    panel = pd.DataFrame({"ts_code": [f"60000{index}.SH" for index in range(1, 7)]})
    for members in FACTOR_GROUPS.values():
        for feature in members:
            panel[feature] = np.arange(1, 7, dtype=float)
    return panel


def test_all_aliases_and_directionless_volatility_remain_raw_evidence() -> None:
    panel = _evidence_panel()
    for feature in [*FACTOR_ALIASES, "wvma_20"]:
        panel[feature] = np.array([1000, 500, 100, -100, -500, -1000], dtype=float)
    actual = huizong_houxuan_yinzi(panel, config={})
    for code, result in actual.items():
        for group, members in FACTOR_GROUPS.items():
            raw = result["groups"][group]
            assert set(raw["values"]) == set(members)
            assert raw["available_factor_count"] == len(members)
            assert raw["missing_fields"] == []
            assert "score_0_100" not in raw
        assert "confidence" not in result
        assert "score_0_100" not in result
    price_volume = actual["600001.SH"]["groups"]["price_volume_confirmation"]
    assert price_volume["values"]["wvma_20"] == 1000
    assert price_volume["metric_definitions"]["wvma_20"]["display_scale"] == 100


def test_registry_describes_raw_evidence_and_keeps_distinct_correlated_factors() -> None:
    rows = {row["feature"]: row for row in factor_registry_rows()}
    assert all(row["use"] == "observed_evidence" for row in rows.values())
    assert all(row["label"] and row["meaning"] and "unit" in row for row in rows.values())
    assert rows["rank_log_amount"]["canonical_source"] == "log_amount_yuan"
    assert "无方向波动" in rows["wvma_20"]["meaning"]
    assert "5 表示 5%" in rows["price_turnover_corr_20"]["input_requirement"]
    assert {"ret_1", "ret_3", "ret_5", "ret_10", "ret_20"}.issubset(
        independent_factor_features(FACTOR_GROUPS["momentum_reversal"])
    )


def _industry_panel(labels: list[str], returns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.Timestamp("2025-04-01"),
            "ts_code": [f"600{index:03d}.SH" for index in range(len(labels))],
            "industry": labels,
            "ret_1": returns,
            "ret_5": returns,
            "ret_20": returns,
            "ma_gap_20": returns,
        }
    )


def test_range_industry_context_groups_actual_labels_and_preserves_market_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        riping_yinzi, "_benchmark_features",
        lambda **kwargs: (pd.DataFrame(), {"status": "unavailable", "warnings": []}),
    )
    panel = _industry_panel(["甲行业"] * 5 + ["乙行业"] * 5, [0.1] * 5 + [-0.1] * 5)
    enriched, meta = riping_yinzi.enrich_daily_factor_panel(panel, source="auto")
    assert enriched["universe_mean_ret_5"].eq(0).all()
    assert enriched.loc[enriched["industry"].eq("甲行业"), "industry_mean_ret_5"].eq(0.1).all()
    assert enriched.loc[enriched["industry"].eq("乙行业"), "industry_mean_ret_5"].eq(-0.1).all()
    assert enriched["excess_vs_industry_ret_5"].eq(0).all()
    assert meta["industry_factor_quality"]["minimum_stocks"] == 5
    assert "比较池" in meta["universe_factor_scope"]


def test_industry_context_rejects_small_unknown_and_field_incomplete_groups() -> None:
    panel = _industry_panel(["甲行业"] * 5 + ["乙行业"] * 4 + [""], [0.1] * 10)
    panel.loc[0, "ret_20"] = np.nan
    result, meta = riping_yinzi._add_industry_context(panel)
    assert result.loc[result["industry"].eq("甲行业"), "industry_mean_ret_5"].notna().all()
    assert result.loc[result["industry"].eq("甲行业"), "industry_mean_ret_20"].isna().all()
    assert result.loc[~result["industry"].eq("甲行业"), "industry_mean_ret_5"].isna().all()
    assert meta["missing_industry_stocks"] == 1
    assert meta["insufficient_group_dates"] == 1


def test_single_stock_roles_require_actual_matching_industry_labels() -> None:
    panel = _industry_panel(["甲行业"] * 5 + ["乙行业", "丙行业"], [0.1] * 5 + [-0.3, -0.4])
    panel["peer_role"] = ["target", *["same_industry"] * 5, "market_reference"]
    result, meta = riping_yinzi._add_industry_context(panel)
    assert result["industry_mean_ret_5"].eq(0.1).all()
    assert result["industry_sample_count"].eq(5).all()
    assert meta["excluded_role_stocks"] == 1
    panel.loc[0, "industry"] = ""
    unavailable, quality = riping_yinzi._add_industry_context(panel)
    assert unavailable["industry_mean_ret_5"].isna().all()
    assert quality["status"] == "unavailable"


def test_industry_statistics_do_not_mix_dates() -> None:
    first = _industry_panel(["甲行业"] * 5, [0.1] * 5)
    second = first.assign(trade_date=pd.Timestamp("2025-04-02"), ret_1=-0.2, ret_5=-0.2, ret_20=-0.2)
    result, _ = riping_yinzi._add_industry_context(pd.concat([first, second], ignore_index=True))
    assert result.loc[result["trade_date"].eq(pd.Timestamp("2025-04-01")), "industry_mean_ret_5"].eq(0.1).all()
    assert result.loc[result["trade_date"].eq(pd.Timestamp("2025-04-02")), "industry_mean_ret_5"].eq(-0.2).all()

def test_legacy_alias_keeps_its_own_observation_without_inventing_missing_canonical_values() -> None:
    panel = pd.DataFrame({"ts_code": ["600001.SH"], "excess_ret_5": [0.01], "peer_mean_ret_5": [0.03]})
    result = huizong_houxuan_yinzi(panel, config={})["600001.SH"]
    values = result["groups"]["relative_strength"]["values"]
    assert values["excess_ret_5"] == 0.01
    assert values["peer_mean_ret_5"] == 0.03
    assert values["excess_vs_universe_ret_5"] is None
    assert result["groups"]["market_context"]["values"]["universe_mean_ret_5"] is None
