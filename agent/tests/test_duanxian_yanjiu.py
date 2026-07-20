"""Offline tests for the two A-share T+3 business workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ashare import gupiao_yanjiu
from src.ashare.bankuai_yuce import (
    _filter_constituents,
    goujian_moxing_shuju,
    xunlian_yuce_moxing,
)
from src.ashare.dangu_yuce import _future_schedule_unavailable_reason
from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    _a_share_rules,
    akshare_zhilian,
    biaozhunhua_daima,
    jiazai_lianghua_peizhi,
    jisuan_tezheng_biao,
    zongjie_jishu,
)
from src.tools import build_registry


def _history(seed: int, rows: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=rows)
    returns = rng.normal(0.0005, 0.015, rows)
    close = 20.0 * np.exp(np.cumsum(returns))
    open_price = close * (1.0 + rng.normal(0.0, 0.004, rows))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.02, rows))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.02, rows))
    volume = rng.integers(1_000_000, 8_000_000, rows).astype(float)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600519", "600519.SH"),
        ("sz000001", "000001.SZ"),
        ("430047.BJ", "430047.BJ"),
    ],
)
def test_a_share_code_normalization(raw: str, expected: str) -> None:
    assert biaozhunhua_daima(raw) == expected


def test_non_stock_code_is_rejected() -> None:
    with pytest.raises(ValueError):
        biaozhunhua_daima("510300.SH")


def test_technical_features_use_only_current_and_past_rows() -> None:
    original = _history(1)
    baseline = jisuan_tezheng_biao(original)
    changed = original.copy()
    changed.loc[changed.index[-1], "close"] *= 10
    recalculated = jisuan_tezheng_biao(changed)

    pd.testing.assert_series_equal(
        baseline.loc[baseline.index[-2], FEATURE_COLUMNS],
        recalculated.loc[recalculated.index[-2], FEATURE_COLUMNS],
        check_names=False,
    )
    summary = zongjie_jishu(original)
    assert 0 <= summary["score_0_100"] <= 100
    assert summary["trade_date"] == original["trade_date"].iloc[-1].strftime("%Y-%m-%d")


def test_external_config_hard_caps_horizons_at_t3() -> None:
    config, path = jiazai_lianghua_peizhi()
    assert path.endswith("lianghua_peizhi.json")
    assert config["moxing"]["horizons"] == [1, 2, 3]
    assert config["jiaoyi"]["max_holding_days"] == 3
    assert config["shuju"]["akshare_bypass_proxy"] is True
    assert config["jiaoyi"]["execution_mode"] == "research_only"
    assert config["jiaoyi"]["allow_live_trading"] is False
    assert config["jiaoyi"]["allow_order_submission"] is False
    assert "source" not in config["shuju"]
    assert "min_list_days" not in config["guolv"]


def test_config_cannot_enable_order_submission(tmp_path: Path) -> None:
    config, _ = jiazai_lianghua_peizhi()
    config["jiaoyi"]["allow_order_submission"] = True
    config_path = tmp_path / "unsafe.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="禁止实盘交易和订单提交"):
        jiazai_lianghua_peizhi(str(config_path))


def test_akshare_direct_context_restores_proxy_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    with akshare_zhilian():
        assert "HTTP_PROXY" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_agent_exposes_research_tools_only() -> None:
    assert build_registry().tool_names == ["gupiao_fenxi", "gupiao_yuce", "bankuai_xuangu"]


def test_filter_removes_st_illiquid_and_limit_up_rows() -> None:
    config, _ = jiazai_lianghua_peizhi()
    frame = pd.DataFrame(
        [
            {"ts_code": "600001.SH", "name": "正常股份", "latest_price": 10, "amount_yuan": 90_000_000, "pct_chg": 1},
            {"ts_code": "600002.SH", "name": "ST测试", "latest_price": 10, "amount_yuan": 90_000_000, "pct_chg": 1},
            {"ts_code": "600003.SH", "name": "低流动", "latest_price": 10, "amount_yuan": 1_000_000, "pct_chg": 1},
            {"ts_code": "600004.SH", "name": "涨停股", "latest_price": 10, "amount_yuan": 90_000_000, "pct_chg": 10},
            {"ts_code": "600005.SH", "name": "N新股", "latest_price": 10, "amount_yuan": 90_000_000, "pct_chg": 1},
            {"ts_code": "600006.SH", "name": "C新股", "latest_price": 10, "amount_yuan": 90_000_000, "pct_chg": 1},
            {"ts_code": "600007.SH", "name": "缺价格", "latest_price": None, "amount_yuan": 90_000_000, "pct_chg": 1},
            {"ts_code": "600008.SH", "name": "缺成交额", "latest_price": 10, "amount_yuan": None, "pct_chg": 1},
        ]
    )
    accepted, rejected = _filter_constituents(frame, config)
    assert accepted["ts_code"].tolist() == ["600001.SH"]
    assert len(rejected) == 7
    assert {item["reason"] for item in rejected} >= {
        "缺少最新价格，无法应用价格过滤",
        "缺少最新成交额，无法应用流动性过滤",
    }


def test_stock_basic_cache_reads_existing_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "stock_basic.csv"
    pd.DataFrame([{"ts_code": "600519.SH", "name": "贵州茅台"}]).to_csv(cache_path, index=False)
    monkeypatch.setattr(gupiao_yanjiu, "STOCK_BASIC_CACHE", cache_path)

    cached = gupiao_yanjiu._stock_basic_cache()

    assert cached.to_dict("records") == [{"ts_code": "600519.SH", "name": "贵州茅台"}]


def test_single_stock_rules_describe_supported_t3_forecast() -> None:
    rules = _a_share_rules("600519.SH", "贵州茅台")
    assert "单股工具支持 T+1/T+2/T+3" in rules["prediction_horizon"]
    assert "不伪造" in rules["prediction_horizon"]


def test_model_panel_builds_literal_next_three_session_close_labels() -> None:
    history = _history(9, rows=100)
    panel = goujian_moxing_shuju(
        {"600519.SH": history},
        {"600519.SH": "贵州茅台"},
        [1, 2, 3],
    )
    signal_index = len(history) - 4
    signal_date = pd.Timestamp(history.iloc[signal_index]["trade_date"]).normalize()
    row = panel.loc[panel["trade_date"] == signal_date].iloc[0]
    signal_close = float(history.iloc[signal_index]["close"])

    for horizon in [1, 2, 3]:
        expected = history.iloc[signal_index + horizon]
        assert row[f"future_date_t{horizon}"] == pd.Timestamp(expected["trade_date"]).normalize()
        assert row[f"future_close_t{horizon}"] == pytest.approx(float(expected["close"]))
        assert row[f"future_return_t{horizon}"] == pytest.approx(float(expected["close"]) / signal_close - 1.0)


def test_future_forecast_rejects_first_target_that_already_closed() -> None:
    schedule = {"future_session_dates": {"T+1": "2026-07-20"}}
    post_close = {
        "market_clock": {
            "captured_at": "2026-07-20 15:52:00",
            "session_status": "post_close",
        }
    }
    during_session = {
        "market_clock": {
            "captured_at": "2026-07-20 10:00:00",
            "session_status": "trading",
        }
    }

    assert "已经结束" in _future_schedule_unavailable_reason(schedule, post_close)
    assert _future_schedule_unavailable_reason(schedule, during_session) == ""


def test_models_train_with_purged_time_split_and_only_t3_outputs() -> None:
    codes = ["600001.SH", "600002.SH", "600003.SH", "600004.SH", "600005.SH"]
    histories = {code: _history(index + 10) for index, code in enumerate(codes)}
    names = {code: f"样本{index}" for index, code in enumerate(codes)}
    config, _ = jiazai_lianghua_peizhi()
    panel = goujian_moxing_shuju(histories, names, [1, 2, 3])
    latest = (
        panel.sort_values("trade_date")
        .groupby("ts_code", as_index=False)
        .tail(1)
        .dropna(subset=FEATURE_COLUMNS)
        .reset_index(drop=True)
    )

    predictions, validation = xunlian_yuce_moxing(panel, latest, config)

    assert {"pred_t1", "pred_t2", "pred_t3"}.issubset(predictions.columns)
    assert "pred_t5" not in predictions.columns
    assert set(validation["horizons"]) == {"T+1", "T+2", "T+3"}
    cutoff = pd.Timestamp(validation["cutoff_date"])
    for horizon in [1, 2, 3]:
        target_date = pd.to_datetime(panel[f"target_date_t{horizon}"])
        train_mask = (pd.to_datetime(panel["trade_date"]) < cutoff) & (target_date < cutoff)
        assert not (target_date[train_mask] >= cutoff).any()
        metrics = validation["horizons"][f"T+{horizon}"]
        assert metrics["train_samples"] >= config["moxing"]["min_training_samples"]
        assert metrics["validation_samples"] >= config["moxing"]["min_validation_samples"]
