"""补充指标的公式、时点和缺失数据边界；不访问远端行情。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.ashare.buchong_zhenduan import goujian_buchong_zhenduan
from src.ashare.gupiao_yanjiu import jisuan_tezheng_biao


def _history(rows=80):
    close = np.full(rows, 100.0)
    return pd.DataFrame({
        "trade_date": pd.bdate_range(end="2026-09-11", periods=rows),
        "close": close, "open": close, "high": close + 1, "low": close - 3,
        "volume": np.full(rows, 10_000.0), "amount_yuan": np.full(rows, 80_000_000.0),
    })


def _result(data, *, cutoff="2026-09-11"):
    return goujian_buchong_zhenduan(data, {}, as_of_date=cutoff, minimum_amount_yuan=50_000_000)


def _blocks(data):
    return {block["key"]: block for block in _result(data)["blocks"]}


def test_downside_uses_all_twenty_returns_and_drawdown_keeps_recovered_loss():
    data = _history()
    data.loc[data.index[-40], "close"] = 80
    data.loc[data.index[-2], "close"] = 90
    risk = _blocks(data)["downside_risk"]
    assert risk["metrics"]["downside_deviation_20"] == pytest.approx(np.sqrt(0.1 ** 2 / 20 * 252))
    assert risk["metrics"]["max_drawdown_60"] == pytest.approx(0.2)
    assert risk["status"] == "ok"
    # 最后一日已回到最高价，不能把窗口最大回撤误算成当前距最高价的跌幅。
    assert data.close.iloc[-1] == data.close.max()


def test_pressure_formula_and_liquidity_count_are_observed_not_proxy():
    data = _history()
    data.loc[data.index[-3:], "amount_yuan"] = 40_000_000
    data.loc[data.index[-4], "amount_yuan"] = 50_000_000
    blocks = _blocks(data)
    assert blocks["volume_pressure"]["metrics"]["cmf_20"] == pytest.approx(0.5)
    assert blocks["liquidity_stability"]["metrics"]["amount_median_20_yuan"] == 80_000_000
    assert blocks["liquidity_stability"]["metrics"]["below_minimum_days"] == 3
    absent = _blocks(data.drop(columns="amount_yuan"))["liquidity_stability"]
    assert absent["status"] == "unavailable"
    assert absent["metrics"]["amount_median_20_yuan"] is None


def test_atr_reuses_technical_formula_and_price_scale_does_not_change_distance():
    data = _history()
    data.loc[:, "close"] += np.linspace(0, 20, len(data))
    data.loc[:, "high"] = data.close + 2
    data.loc[:, "low"] = data.close - 2
    distance = _blocks(data)["atr_distance"]["metrics"]
    features = jisuan_tezheng_biao(data)
    expected = (features.close.iloc[-1] - features.ma_20.iloc[-1]) / features.atr_14.iloc[-1]
    assert distance["distance_atr"] == pytest.approx(expected, abs=1e-6)
    scaled = data.copy()
    scaled[["open", "high", "low", "close"]] *= 100
    assert _blocks(scaled)["atr_distance"]["metrics"]["distance_atr"] == distance["distance_atr"]


def test_future_rows_are_excluded_and_inputs_are_not_mutated():
    data = _history()
    before = data.copy(deep=True)
    future = data.iloc[-1:].copy()
    future["trade_date"] = pd.Timestamp("2026-09-14")
    future[["open", "high", "low", "close"]] *= 10
    result = _result(pd.concat([data, future], ignore_index=True))
    assert result == _result(data)
    pd.testing.assert_frame_equal(data, before)
    assert "score_0_100" not in result


def test_atr_does_not_use_a_rejected_bar_as_previous_close():
    data = _history()
    data.loc[data.index[-20:], "close"] = np.arange(100, 120, dtype=float)
    data.loc[data.index[-20:], "high"] = data.close.tail(20) + 1
    data.loc[data.index[-20:], "low"] = data.close.tail(20) - 3
    data.loc[data.index[-21], "close"] = 1000  # 价格超出该日高低范围，整条日线不能参与。
    assert _blocks(data)["atr_distance"]["metrics"] == _blocks(data.tail(20))["atr_distance"]["metrics"]


@pytest.mark.parametrize("rows", [0, 1, 19, 20, 21, 59])
def test_partial_windows_do_not_silently_shrink_required_sample(rows):
    blocks = _blocks(_history(rows))
    assert blocks["downside_risk"]["metrics"]["max_drawdown_60"] is None
    assert (blocks["downside_risk"]["metrics"]["downside_deviation_20"] is not None) == (rows >= 21)
    assert (blocks["volume_pressure"]["status"] == "ok") == (rows >= 20)
    json.dumps(list(blocks.values()), allow_nan=False)


def test_zero_observation_is_different_from_missing_and_flat_prices_are_disclosed():
    data = _history()
    assert _blocks(data)["downside_risk"]["metrics"]["downside_deviation_20"] == 0.0
    data.loc[:, "high"] = data.close
    data.loc[:, "low"] = data.close
    blocks = _blocks(data)
    assert blocks["atr_distance"]["metrics"]["distance_atr"] is None
    assert blocks["volume_pressure"]["metrics"]["cmf_20"] is None
    assert blocks["volume_pressure"]["status"] == "unavailable"
    data.loc[data.index[-1], "high"] += 1
    assert _blocks(data)["volume_pressure"]["status"] == "partial"
    assert "19日" in _blocks(data)["volume_pressure"]["missing_reason"]


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -1])
def test_bad_recent_price_does_not_bridge_missing_returns(value):
    data = _history()
    data.loc[data.index[-3], "close"] = value
    blocks = _blocks(data)
    assert blocks["downside_risk"]["metrics"]["downside_deviation_20"] is None
    assert blocks["atr_distance"]["status"] == "unavailable"
    assert blocks["volume_pressure"]["status"] == "unavailable"
    json.dumps(_result(data), allow_nan=False)


@pytest.mark.parametrize("problem", ["duplicate", "invalid_date", "missing_date"])
def test_ambiguous_dates_are_not_silently_dropped(problem):
    data = _history()
    if problem == "duplicate":
        data = pd.concat([data, data.iloc[-1:]], ignore_index=True)
    elif problem == "invalid_date":
        data.loc[data.index[-1], "trade_date"] = pd.NaT
    else:
        data = data.drop(columns="trade_date")
    result = _result(data)
    assert all(block["status"] == "unavailable" for block in result["blocks"][:4])
    assert result["daily_data_as_of"] is None


def test_invalid_analysis_date_cannot_publish_dated_metrics():
    result = _result(_history(), cutoff="bad-date")
    assert result["data_as_of"] is None
    assert all(block["status"] == "unavailable" for block in result["blocks"][:4])
