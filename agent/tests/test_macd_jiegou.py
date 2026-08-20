"""MACD 结构研判的定向边界验证。"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.ashare import peizhi
from src.ashare.gupiao_yanjiu import FEATURE_COLUMNS, jisuan_tezheng_biao, zongjie_jishu
from src.ashare.macd_jiegou import MacdJiegouPeizhi, yanpan_macd_jiegou
from src.ashare.xuangu_guize import goujian_houxuan_zhaiyao, jisuan_fengxian_koufen
from src.ashare.yinzi_gongcheng import FACTOR_GROUPS, factor_group


def _structure_features(
    dif_pct: list[float],
    gap_pct: list[float],
    *,
    close: list[float] | None = None,
    low: list[float] | None = None,
    high: list[float] | None = None,
) -> pd.DataFrame:
    rows = len(dif_pct)
    close_values = np.asarray(close if close is not None else [100.0] * rows, dtype=float)
    dif_values = np.asarray(dif_pct, dtype=float)
    gap_values = np.asarray(gap_pct, dtype=float)
    dea_values = dif_values - gap_values
    histogram_values = 2.0 * gap_values
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2025-01-02", periods=rows),
            "close": close_values,
            "high": np.asarray(high if high is not None else close_values + 1.0, dtype=float),
            "low": np.asarray(low if low is not None else close_values - 1.0, dtype=float),
            "macd_dif": dif_values * close_values,
            "macd_dea": dea_values * close_values,
            "macd_hist": histogram_values * close_values,
            "macd_dif_pct": dif_values,
            "macd_dea_pct": dea_values,
            "macd_gap_pct": gap_values,
            "macd_hist_pct": histogram_values,
            "macd_zero_distance_pct": np.maximum(np.abs(dif_values), np.abs(dea_values)),
        }
    )


def _bottom_divergence_features() -> pd.DataFrame:
    rows = 18
    close = [100.0] * rows
    low = [98.0] * rows
    low[5] = 90.0
    low[12] = 85.0
    dif = [-0.01] * rows
    dif[5] = -0.03
    dif[12] = -0.015
    gap = [-0.002] * rows
    gap[5] = -0.01
    gap[12] = -0.004
    return _structure_features(dif, gap, close=close, low=low)


def _top_divergence_features() -> pd.DataFrame:
    rows = 18
    close = [100.0] * rows
    high = [102.0] * rows
    high[5] = 110.0
    high[12] = 115.0
    dif = [0.01] * rows
    dif[5] = 0.03
    dif[12] = 0.015
    gap = [0.002] * rows
    gap[5] = 0.01
    gap[12] = 0.004
    return _structure_features(dif, gap, close=close, high=high)


def _flat_history(rows: int = 30, *, scale: float = 1.0) -> pd.DataFrame:
    close = np.full(rows, 10.0 * scale)
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2025-01-02", periods=rows),
            "open": close,
            "high": close + 0.1 * scale,
            "low": close - 0.1 * scale,
            "close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )


def test_zero_axis_states_are_distinguished() -> None:
    cases = [
        ([0.01, 0.012], [0.002, 0.002], "above"),
        ([-0.01, -0.012], [-0.002, -0.002], "below"),
        ([0.002, 0.002], [0.004, 0.004], "mixed"),
    ]
    for dif, gap, expected in cases:
        result = yanpan_macd_jiegou(_structure_features(dif, gap))
        assert result["status"] == "ok"
        assert result["zero_axis"]["code"] == expected


def test_cross_is_recorded_only_on_the_actual_trading_day_and_gets_stale() -> None:
    gaps = [-0.002, 0.001, 0.002, 0.003, 0.004, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001]
    result = yanpan_macd_jiegou(_structure_features([0.02] * len(gaps), gaps))
    cross = result["latest_cross"]
    assert cross["signal_type"] == "golden"
    assert cross["event_date"] == "2025-01-03"
    assert cross["age_trading_sessions"] == 9
    assert cross["freshness"] == "stale"
    assert cross["region"] == "above"


def test_cross_region_distinguishes_below_near_zero_and_mixed() -> None:
    cases = [
        ([-0.02, -0.02], [-0.002, 0.001], "below"),
        ([0.001, 0.001], [-0.001, 0.001], "near_zero"),
        ([0.01, 0.01], [-0.001, 0.02], "mixed"),
    ]
    for dif, gap, expected_region in cases:
        cross = yanpan_macd_jiegou(_structure_features(dif, gap))["latest_cross"]
        assert cross["signal_type"] == "golden"
        assert cross["region"] == expected_region


def test_positive_histogram_without_a_new_cross_is_not_described_as_new() -> None:
    gaps = [-0.001, 0.001] + [0.002] * 10
    result = yanpan_macd_jiegou(_structure_features([0.02] * len(gaps), gaps))
    cross = result["latest_cross"]
    assert cross["event_date"] != result["as_of"]
    assert cross["freshness"] == "stale"


def test_bottom_divergence_appears_only_on_the_right_side_confirmation_date() -> None:
    features = _bottom_divergence_features()
    before_confirmation = yanpan_macd_jiegou(features.iloc[:14])
    on_confirmation = yanpan_macd_jiegou(features.iloc[:15])
    before = before_confirmation["divergences"]["bottom"]["dif"]
    confirmed = on_confirmation["divergences"]["bottom"]["dif"]
    assert before["status"] != "confirmed"
    assert confirmed["status"] == "confirmed"
    assert confirmed["second_price_pivot"]["date"] == "2025-01-20"
    assert confirmed["confirmation_date"] == "2025-01-22"
    assert on_confirmation["divergences"]["bottom"]["histogram"]["status"] == "confirmed"


def test_top_divergence_is_separate_for_dif_and_histogram_and_can_invalidate() -> None:
    features = _top_divergence_features()
    confirmed = yanpan_macd_jiegou(features.iloc[:15])
    assert confirmed["divergences"]["top"]["dif"]["status"] == "confirmed"
    assert confirmed["divergences"]["top"]["histogram"]["status"] == "confirmed"
    assert confirmed["risk_warnings"]
    features.loc[15, "high"] = 120.0
    invalidated = yanpan_macd_jiegou(features)
    signal = invalidated["divergences"]["top"]["dif"]
    assert signal["status"] == "invalidated"
    assert signal["invalidation_reason"] == "价格突破背离高点"


def test_small_change_and_close_pivots_do_not_create_divergence() -> None:
    features = _bottom_divergence_features()
    features.loc[12, "low"] = 89.5
    result = yanpan_macd_jiegou(features)
    assert result["divergences"]["bottom"]["dif"]["status"] == "no_signal"


def test_price_scaling_keeps_normalized_structure_conclusion() -> None:
    rows = 100
    base = 10.0 + np.linspace(0.0, 3.0, rows) + np.sin(np.linspace(0.0, 12.0, rows))
    history = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=rows),
            "open": base * 0.998,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.full(rows, 1_000_000.0),
        }
    )
    scaled = history.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] *= 10.0
    original_result = yanpan_macd_jiegou(jisuan_tezheng_biao(history))
    scaled_result = yanpan_macd_jiegou(jisuan_tezheng_biao(scaled))
    assert scaled_result["zero_axis"] == original_result["zero_axis"]
    assert scaled_result["latest_cross"]["signal_type"] == original_result["latest_cross"]["signal_type"]
    assert scaled_result["latest_cross"]["event_date"] == original_result["latest_cross"]["event_date"]
    assert scaled_result["structure_classification"] == original_result["structure_classification"]


def test_short_history_and_invalid_latest_data_degrade_explicitly() -> None:
    summary = zongjie_jishu(_flat_history())
    assert summary["macd_structure"]["status"] == "insufficient_data"
    features = _structure_features([0.01] * 40, [0.001] * 40)
    features.loc[39, "macd_dea"] = np.nan
    result = yanpan_macd_jiegou(features)
    assert result["status"] == "unavailable"
    assert result["outcome"] == "data_unavailable"


def test_duplicate_dates_are_deduplicated_and_disclosed() -> None:
    features = _structure_features([0.01, 0.02], [-0.001, 0.001])
    features = pd.concat([features, features.iloc[[-1]]], ignore_index=True)
    result = yanpan_macd_jiegou(features)
    assert result["status"] == "ok"
    assert any("重复交易日" in warning for warning in result["warnings"])


def test_flat_zero_volume_session_history_does_not_create_false_signal() -> None:
    history = _flat_history(80)
    history.loc[35, "volume"] = 0.0
    summary = zongjie_jishu(history)
    structure = summary["macd_structure"]
    assert structure["status"] == "ok"
    assert structure["latest_cross"]["status"] == "no_signal"
    assert structure["divergences"]["bottom"]["dif"]["status"] == "no_signal"
    assert structure["divergences"]["top"]["dif"]["status"] == "no_signal"


def test_missing_high_low_makes_structure_unavailable_without_creating_risk() -> None:
    features = _structure_features([0.01] * 40, [0.001] * 40).drop(columns=["high", "low"])
    result = yanpan_macd_jiegou(features)
    assert result["status"] == "unavailable"
    assert result["outcome"] == "data_unavailable"
    assert result["risk_warnings"] == []


def test_structure_evidence_does_not_expand_factor_budget_or_add_risk_penalty() -> None:
    momentum_members = FACTOR_GROUPS["momentum_reversal"]
    assert "macd_dif_pct" in momentum_members
    assert "macd_hist_pct" in momentum_members
    assert not any("divergence" in feature or "cross" in feature for feature in momentum_members)
    assert not any("cross" in feature or "divergence" in feature for feature in FEATURE_COLUMNS)
    technical = {
        "macd_structure": {
            "status": "ok",
            "risk_warnings": ["已确认顶部背离，仅作风险提示"],
        }
    }
    penalties, risks = jisuan_fengxian_koufen(
        code="600000.SH",
        name="样本",
        snapshot={},
        factor={"confidence": 1.0},
        pattern={},
        late={},
        config={"fenxi": {"risk_penalty_max": 30}},
        technical=technical,
    )
    assert penalties == []
    assert risks == ["已确认顶部背离，仅作风险提示"]


def test_moving_average_gap_factor_has_an_accurate_name_and_legacy_alias() -> None:
    features = jisuan_tezheng_biao(_flat_history(80))
    assert "ma_5_20_gap_change" in FACTOR_GROUPS["trend_structure"]
    assert "golden_cross_speed" not in FACTOR_GROUPS["trend_structure"]
    assert features["ma_5_20_gap_change"].equals(features["golden_cross_speed"])
    assert factor_group("golden_cross_speed") == "trend_structure"


def test_candidate_summary_exposes_structure_support_counterevidence_and_risk() -> None:
    structure = {
        "status": "ok",
        "outcome": "analysis_success",
        "reason": "",
        "supporting_evidence": ["零轴上方趋势背景"],
        "counter_evidence": ["正柱连续收窄"],
        "risk_warnings": ["快线顶部背离仅作风险提示"],
    }
    summary = goujian_houxuan_zhaiyao(
        {
            "ts_code": "600000.SH",
            "name": "样本",
            "industry": "样本行业",
            "ranking": {"score_0_100": 70.0, "confidence": 0.8, "definition": "研究排序"},
            "factor": {"groups": {}},
            "fundamental": {"evidence": []},
            "pattern": {},
            "late": {},
            "technical": {"macd_structure": structure},
            "tradability": {"basic_execution_feasible": True},
            "data_quality": {},
            "risks": ["已有风险"],
        },
        rank=1,
        minimum_score=60.0,
        minimum_confidence=0.6,
    )
    assert summary["positive_evidence"][0] == "零轴上方趋势背景"
    assert summary["unmet_conditions"][0] == "正柱连续收窄"
    assert summary["risks"][0] == "快线顶部背离仅作风险提示"
    assert summary["technical_summary"]["status"] == "ok"
    assert summary["technical_summary"]["outcome"] == "analysis_success"
    assert summary["technical_summary"]["reason"] in ("", None)
    assert summary["technical_summary"]["macd_structure"] == structure


def test_default_thresholds_are_fixed_research_parameters_not_score_weights() -> None:
    settings = MacdJiegouPeizhi()
    assert settings.zero_near_threshold_pct > 0
    assert settings.minimum_price_change_pct > 0
    assert settings.minimum_indicator_change_pct > 0


def test_structure_config_rejects_nonincreasing_freshness_windows() -> None:
    config, _ = peizhi.jiazai_lianghua_peizhi()
    invalid = copy.deepcopy(config)
    invalid["fenxi"]["macd_structure"]["cross_recent_sessions"] = 21
    invalid["fenxi"]["macd_structure"]["cross_max_age_sessions"] = 20
    with pytest.raises(ValueError, match="交叉新鲜度窗口必须递增"):
        peizhi._xiaoyan_fenxi_peizhi(invalid)
