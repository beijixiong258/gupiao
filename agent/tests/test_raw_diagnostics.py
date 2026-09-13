"""无综合评分的原始指标、状态和配置契约。"""

import numpy as np
import pandas as pd

from src.ashare.fenxi_weipan import fenxi_weipan
from src.ashare.fenxi_xingtai import fenxi_zhangting_huimaqiang
from src.ashare.gupiao_yanjiu import zongjie_jishu
from src.ashare.macd_huifang import _market_regime
from src.ashare.peizhi import jiazai_lianghua_peizhi
from src.ashare.riping_yinzi import _add_market_regime_features
from src.ashare.yinzi_gongcheng import FACTOR_REGISTRY, describe_factor_coverage
from src.ashare.xuangu_guize import goujian_kejiaoyixing_zhaiyao, shishi_ying_guolv


def _history(rows=80):
    close = 20 + np.linspace(0, 4, rows)
    return pd.DataFrame({"trade_date": pd.bdate_range(end="2026-09-10", periods=rows),
                         "open": close - 0.1, "close": close, "high": close + 0.3, "low": close - 0.3,
                         "volume": np.full(rows, 2_000_000.0), "amount_yuan": np.full(rows, 80_000_000.0)})


def _assert_no_scoring(value):
    if isinstance(value, dict):
        assert not set(value).intersection({"score", "score_0_100", "component_scores", "component_weights", "factor_group_weights", "risk_penalty", "buy_decision"})
        for child in value.values():
            _assert_no_scoring(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_scoring(child)


def test_short_history_keeps_latest_known_values_without_scoring():
    data = _history(5)
    result = zongjie_jishu(data)
    assert result["status"] == "partial"
    assert result["close"] == 24
    assert result["moving_averages"]["ma20"] is None
    assert "ma_20" in result["missing_indicators"]
    assert result["trade_date"] == "2026-09-10"
    _assert_no_scoring(result)


def test_latest_incomplete_indicator_never_uses_earlier_day_as_current():
    data = _history()
    data.loc[data.index[-1], "close"] = np.nan
    result = zongjie_jishu(data)
    assert result["trade_date"] == "2026-09-10"
    assert result["close"] is None


def test_pattern_and_late_session_expose_states_and_actuals_without_scores():
    pattern = fenxi_zhangting_huimaqiang(_history(), code="000021.SZ", name="测试", config={})
    late = fenxi_weipan(_history(), snapshot={"last_price": 25, "pct_change": 2, "volume_ratio": 1.2, "provider_quote_time": "2026-09-11 14:34:50"},
                       clock={"is_trading_day": True, "captured_at": "2026-09-11 14:35:00"}, config={})
    assert "state" in pattern
    assert late["actuals"]["pct_change"] == 2
    assert any(item["key"] == "positive_return" and item["status"] == "met" for item in late["conditions"])
    assert "换手率" in late["unavailable_items"]
    _assert_no_scoring([pattern, late])


def test_raw_registry_has_chinese_definitions_and_does_not_make_composite_columns():
    assert all(row["label"] != row["feature"] for row in FACTOR_REGISTRY)
    assert all(row["meaning"] and "display_scale" in row for row in FACTOR_REGISTRY)
    data = pd.DataFrame({"trade_date": ["2026-09-10"], "ret_5": [0.1]})
    result, _ = describe_factor_coverage(data)
    assert list(result.columns) == list(data.columns)
    assert not any("composite" in row["feature"] or row["feature"] == "market_regime_score" for row in FACTOR_REGISTRY)


def test_market_regime_is_explicit_direction_and_breadth_not_weighted_score():
    data = pd.DataFrame({"market_csi300_ret_20": [0.1, -0.1, 0.1, np.nan],
                         "universe_breadth_above_ma20": [0.8, 0.2, 0.2, 0.9],
                         "market_csi300_volatility_20": [0.2] * 4})
    result = _add_market_regime_features(data)
    assert result.loc[0, "market_regime_strong"] == 1
    assert result.loc[1, "market_regime_weak"] == 1
    assert result.loc[2, "market_regime_sideways"] == 1
    assert pd.isna(result.loc[3, "market_regime_strong"])
    assert "market_regime_score" not in result
    assert [_market_regime(row) for _, row in result.iterrows()] == ["strong", "weak", "sideways", "unavailable"]


def test_configuration_has_no_scoring_or_forced_single_stock_minimum_gates():
    config, _ = jiazai_lianghua_peizhi()
    _assert_no_scoring(config)
    assert "minimum_recommendation_score" not in config["fenxi"]
    assert "minimum_confidence" not in config["fenxi"]
    assert "state_scores" not in config["xingtai"]
    assert "minimum_history_rows" not in config["dangu"]
    assert "minimum_peer_stocks" not in config["dangu"]


def test_realtime_hard_filter_and_missing_quote_checks_survive_score_removal():
    config, _ = jiazai_lianghua_peizhi()
    blocked = shishi_ying_guolv({"status": "ok", "last_price": 11, "previous_close": 10,
                               "open": 11, "high": 11, "low": 11}, code="000021.SZ", name="测试", config=config)
    assert blocked
    missing = goujian_kejiaoyixing_zhaiyao(code="000021.SZ", name="测试", snapshot={"status": "unavailable"},
                                         history=_history(), minimum_amount=50_000_000, realtime_required=True)
    assert missing["basic_execution_feasible"] is False
    closed = goujian_kejiaoyixing_zhaiyao(code="000021.SZ", name="测试", snapshot={"status": "unavailable"},
                                        history=_history(), minimum_amount=50_000_000, realtime_required=False)
    assert closed["basic_execution_feasible"] is True
    assert closed["current_quote_verified"] is False
