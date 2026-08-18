"""Daily-K contract tests for the two supported A-share workflows."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import requests

from src.ashare.dangu_yuce import _future_schedule_unavailable_reason
from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    biaozhunhua_daima,
    jisuan_tezheng_biao,
    zongjie_jishu,
)
from src.ashare.moxing_gongju import goujian_moxing_shuju
from src.ashare.peizhi import jiazai_lianghua_peizhi
from src.ashare.shichang_shuju import akshare_zhilian
from src.ashare.yuce_xunlian import xunlian_chiyouqi_yuce_moxing
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


def test_internal_config_hard_caps_horizons_at_t3() -> None:
    config, path = jiazai_lianghua_peizhi()
    assert path.endswith("lianghua_peizhi.json")
    assert config["moxing"]["horizons"] == [1, 2, 3]
    assert config["wangluo"]["domestic_connection_mode"] == "direct"
    assert "akshare_bypass_proxy" not in config["shuju"]
    assert "jiaoyi" not in config
    assert "source" not in config["shuju"]
    assert "guolv" not in config


def test_trading_configuration_is_not_loaded() -> None:
    config, _ = jiazai_lianghua_peizhi()
    assert "jiaoyi" not in config


def test_akshare_direct_context_restores_proxy_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "localhost")
    with akshare_zhilian():
        assert "HTTP_PROXY" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
        assert os.environ["NO_PROXY"] == "*"
        # Windows 的 requests 还会读取系统代理；NO_PROXY=* 是确保其
        # 内部新建 Session 也真正直连的关键回归断言。
        assert requests.utils.get_environ_proxies("https://example.com") == {}
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["NO_PROXY"] == "localhost"


def test_agent_exposes_research_tools_only() -> None:
    assert build_registry().tool_names == ["gupiao_fenxi", "gupiao_yuce"]


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
    # This contract test covers the purged split and fixed T+1/T+2/T+3
    # outputs.  Five random synthetic stocks are not a meaningful universe
    # for the production Rank-IC stability gate, which has dedicated tests.
    config["moxing"]["factor_stability_enabled"] = False
    panel = goujian_moxing_shuju(histories, names, [1, 2, 3])
    latest = (
        panel.sort_values("trade_date")
        .groupby("ts_code", as_index=False)
        .tail(1)
        .dropna(subset=FEATURE_COLUMNS)
        .reset_index(drop=True)
    )

    predictions, validation = xunlian_chiyouqi_yuce_moxing(
        panel=panel,
        latest=latest,
        config=config,
        budget_yuan=100_000.0,
    )

    assert {"pred_t1", "pred_t2", "pred_t3"}.issubset(predictions.columns)
    assert "pred_t5" not in predictions.columns
    assert set(validation["horizons"]) == {"T+1", "T+2", "T+3"}
    for horizon in [1, 2, 3]:
        metrics = validation["horizons"][f"T+{horizon}"]
        target_date = pd.to_datetime(panel[f"target_date_t{horizon}"])
        trade_date = pd.to_datetime(panel["trade_date"])
        assert metrics["folds"]
        for fold in metrics["folds"]:
            validation_start = pd.Timestamp(fold["validation_start"])
            train_mask = (trade_date < validation_start) & (target_date < validation_start)
            assert not (target_date[train_mask] >= validation_start).any()
            if fold["status"] == "ok":
                assert fold["train_samples"] >= config["dangu"]["min_fold_training_samples"]
                assert fold["validation_samples"] >= config["dangu"]["min_fold_validation_samples"]
