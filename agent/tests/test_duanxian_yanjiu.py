"""Daily-K contracts for stock analysis and diagnosis."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import requests

from src.ashare.gupiao_yanjiu import (
    FEATURE_COLUMNS,
    biaozhunhua_daima,
    jisuan_tezheng_biao,
    zongjie_jishu,
)
from src.ashare.peizhi import jiazai_lianghua_peizhi
from src.ashare.shichang_shuju import akshare_zhilian
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
    assert "score_0_100" not in summary
    assert summary["evidence"]
    assert summary["trade_date"] == original["trade_date"].iloc[-1].strftime("%Y-%m-%d")


def test_internal_config_contains_analysis_settings_only() -> None:
    config, path = jiazai_lianghua_peizhi()
    assert path.endswith("lianghua_peizhi.json")
    assert "moxing" not in config
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
    assert build_registry().tool_names == ["gupiao_fenxi"]

def test_analysis_factor_pipeline_survives_without_training_or_historical_valuation(monkeypatch) -> None:
    from src.ashare import riping_yinzi
    from src.ashare.fenxi_yinzi import (
        goujian_fenxi_yinzi_mianban,
        huizong_houxuan_yinzi,
        zengjia_dangri_guzhi_yinzi,
    )
    from src.ashare.yinzi_gongcheng import FACTOR_GROUPS

    monkeypatch.setattr(
        riping_yinzi,
        "_benchmark_features",
        lambda **kwargs: (pd.DataFrame(), {"status": "unavailable", "warnings": ["离线样本"]}),
    )

    def reject_remote_valuation():
        raise AssertionError("分析因子不应逐股请求历史估值")

    monkeypatch.setattr(riping_yinzi, "_tushare_pro", reject_remote_valuation)
    codes = [f"60000{index}.SH" for index in range(1, 7)]
    histories = {code: _history(index, rows=100) for index, code in enumerate(codes)}
    profiles = pd.DataFrame(
        {
            "ts_code": codes,
            "name": [f"样本{index}" for index in range(len(codes))],
            "industry": ["样本行业"] * len(codes),
            "turnover_rate": np.arange(1.0, len(codes) + 1),
            "circulating_market_value_yuan": np.arange(1, len(codes) + 1) * 1e9,
        }
    )
    panel, metadata = goujian_fenxi_yinzi_mianban(histories, profiles, source="auto")
    latest = panel[panel["trade_date"].eq(panel["trade_date"].max())]
    latest = zengjia_dangri_guzhi_yinzi(latest, profiles)
    config, _ = jiazai_lianghua_peizhi()
    results = huizong_houxuan_yinzi(latest, config=config["fenxi"])

    assert metadata["status"] == "ok"
    assert set(results) == set(codes)
    for result in results.values():
        assert result["status"] == "partial"  # 离线指数缺失，保留其余实际指标。
        assert set(result["groups"]) == set(FACTOR_GROUPS)
        assert "score_0_100" not in result
        assert "confidence" not in result
        assert result["available_factor_count"] > 0
