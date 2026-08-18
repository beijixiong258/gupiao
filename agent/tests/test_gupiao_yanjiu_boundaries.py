"""Regression tests for single-stock data-quality boundaries."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ashare import gupiao_yanjiu
from src.ashare import peizhi
from src.ashare.fenxi_yinzi import jisuan_shendu_jibenmian
from src.ashare.shuju_yuan import huoqu_zhangdieting_guize


def _flat_history(rows: int = 30) -> pd.DataFrame:
    close = np.full(rows, 10.0)
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2025-01-02", periods=rows),
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )


def test_quant_config_requires_minute_bars_only_for_late_session_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    config = json.loads(peizhi.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["shuju"]["minute_bars_enabled"] = False
    path = tmp_path / "no_minute_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(peizhi, "DEFAULT_CONFIG_PATH", path)
    with pytest.raises(ValueError, match="尾盘复核需要少量 5 分钟行情"):
        peizhi.jiazai_lianghua_peizhi()


def test_short_history_keeps_unformed_ma60_and_macd_neutral() -> None:
    summary = gupiao_yanjiu.zongjie_jishu(_flat_history())

    assert summary["rsi_14"] == 50.0
    assert summary["moving_averages"]["ma60"] is None
    assert summary["macd"]["histogram"] is None
    assert not any(reason == "MACD 柱为负" for reason in summary["evidence"])
    assert any("MA60 暂不可用" in warning for warning in summary["indicator_warnings"])
    assert any("MACD 暂不可用" in warning for warning in summary["indicator_warnings"])


def test_partial_name_with_multiple_candidates_is_rejected_but_exact_name_works() -> None:
    table = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "平安银行"},
            {"ts_code": "600036.SH", "name": "招商银行"},
        ]
    )

    exact = gupiao_yanjiu._match_stock_basic(table, "平安银行")
    assert exact and exact["ts_code"] == "000001.SZ"
    with pytest.raises(ValueError, match="多个候选.*平安银行.*招商银行"):
        gupiao_yanjiu._match_stock_basic(table, "银行")


def test_explicit_akshare_name_resolution_never_calls_tushare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gupiao_yanjiu,
        "_akshare_name_table",
        lambda: pd.DataFrame([{"ts_code": "600519.SH", "name": "贵州茅台"}]),
    )
    monkeypatch.setattr(
        gupiao_yanjiu,
        "_tushare_pro",
        lambda: (_ for _ in ()).throw(AssertionError("explicit akshare must not call tushare")),
    )

    code, profile, warnings = gupiao_yanjiu.jiexi_gupiao("贵州茅台", source="akshare")

    assert code == "600519.SH"
    assert profile["name"] == "贵州茅台"
    assert any("AKShare" in warning for warning in warnings)


def test_explicit_tushare_name_resolution_never_calls_akshare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gupiao_yanjiu, "_tushare_pro", lambda: object())
    monkeypatch.setattr(
        gupiao_yanjiu,
        "huoqu_gupiao_jichu_ziliao",
        lambda _pro, _quality: pd.DataFrame([{"ts_code": "600519.SH", "name": "贵州茅台"}]),
    )
    monkeypatch.setattr(
        gupiao_yanjiu,
        "_akshare_name_table",
        lambda: (_ for _ in ()).throw(AssertionError("explicit tushare must not call akshare")),
    )

    code, profile, warnings = gupiao_yanjiu.jiexi_gupiao("贵州茅台", source="tushare")

    assert code == "600519.SH"
    assert profile["name"] == "贵州茅台"
    assert warnings == []


def test_only_known_920_segment_is_inferred_from_a_code_starting_with_nine() -> None:
    assert gupiao_yanjiu.biaozhunhua_daima("920001") == "920001.BJ"
    with pytest.raises(ValueError, match="不属于.*A 股"):
        gupiao_yanjiu.biaozhunhua_daima("900901")
    with pytest.raises(ValueError, match="不属于.*A 股"):
        gupiao_yanjiu.biaozhunhua_daima("400001")


def test_price_limit_exemption_is_an_explicit_upstream_fact_not_a_name_guess() -> None:
    exempt = huoqu_zhangdieting_guize(
        "688001.SH",
        "N样本",
        price_limit_exempt=True,
    )
    normal = huoqu_zhangdieting_guize("688001.SH", "N样本")

    assert exempt.status == "no_limit"
    assert exempt.limit_rate is None
    assert normal.status == "limited"
    assert normal.limit_rate == 0.20


def test_akshare_name_table_always_fetches_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fetch_names() -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame([{"code": "600000", "name": "新名称"}])

    monkeypatch.setattr(gupiao_yanjiu, "akshare_zhilian", nullcontext)
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_info_a_code_name=fetch_names))

    result = gupiao_yanjiu._akshare_name_table()

    assert calls["count"] == 1
    assert result.to_dict("records") == [{"ts_code": "600000.SH", "name": "新名称"}]


def test_adj_factor_failure_is_limited_to_one_request() -> None:
    class Pro:
        def __init__(self) -> None:
            self.calls = 0

        def adj_factor(self, **_kwargs) -> pd.DataFrame:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            return pd.DataFrame(
                [
                    {"trade_date": "20250102", "adj_factor": 1.0},
                    {"trade_date": "20250103", "adj_factor": 2.0},
                ]
            )

    data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "open": [10.0, 20.0],
            "high": [10.0, 20.0],
            "low": [10.0, 20.0],
            "close": [10.0, 20.0],
        }
    )
    pro = Pro()

    _, first_adjustment, _ = gupiao_yanjiu._apply_qfq(pro, "600000.SH", "20250102", "20250103", data)
    adjusted, second_adjustment, _ = gupiao_yanjiu._apply_qfq(
        pro, "600000.SH", "20250102", "20250103", data
    )

    assert first_adjustment == "raw_unadjusted"
    assert second_adjustment == "qfq_by_tushare_adj_factor"
    assert pro.calls == 2
    assert adjusted["close"].tolist() == [5.0, 20.0]


def test_financials_use_only_announcements_known_by_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    class Pro:
        def daily_basic(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "trade_date": "20240628",
                        "pe": 12,
                        "pe_ttm": 11,
                        "pb": 1.2,
                        "total_mv": 100,
                        "circ_mv": 80,
                        "turnover_rate": 1.1,
                        "volume_ratio": 0.9,
                    }
                ]
            )

        def fina_indicator(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "ann_date": "20240420",
                        "end_date": "20231231",
                        "roe": 10,
                        "roe_dt": 9,
                        "grossprofit_margin": 20,
                        "netprofit_margin": 8,
                        "debt_to_assets": 45,
                        "or_yoy": 5,
                        "netprofit_yoy": np.nan,
                        "ocf_to_or": 7,
                        "basic_eps": 0.5,
                    },
                    {
                        "ann_date": "20240720",
                        "end_date": "20240630",
                        "roe": 99,
                        "roe_dt": 99,
                        "grossprofit_margin": 99,
                        "netprofit_margin": 99,
                        "debt_to_assets": 99,
                        "or_yoy": 99,
                        "netprofit_yoy": 99,
                        "ocf_to_or": 99,
                        "basic_eps": 99,
                    },
                ]
            )

    pro = Pro()
    profile = pd.DataFrame([{"ts_code": "600000.SH", "name": "浦发银行"}])
    monkeypatch.setattr(gupiao_yanjiu, "_tushare_pro", lambda: pro)
    monkeypatch.setattr(gupiao_yanjiu, "huoqu_gupiao_jichu_ziliao", lambda _pro, _quality: profile)

    result = gupiao_yanjiu.huoqu_jibenmian("600000.SH", trade_date="2024-06-28")

    assert result["valuation"]["as_of"] == "2024-06-28"
    assert result["financials"]["announcement_date"] == "2024-04-20"
    assert result["financials"]["roe_pct"] == 10.0
    assert result["financials"]["missing_fields"] == ["net_profit_yoy_pct"]


def test_profile_live_source_quality_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pro:
        def daily_basic(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

        def fina_indicator(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    def load(_pro, quality: dict[str, object]) -> pd.DataFrame:
        quality["stock_basic"] = {"source": "tushare_live", "rows": 1, "persistence": "none"}
        return pd.DataFrame([{"ts_code": "600000.SH", "name": "实时名称"}])

    monkeypatch.setattr(gupiao_yanjiu, "_tushare_pro", lambda: Pro())
    monkeypatch.setattr(gupiao_yanjiu, "huoqu_gupiao_jichu_ziliao", load)
    monkeypatch.setattr(gupiao_yanjiu, "_akshare_info", lambda _code: ({}, []))

    result = gupiao_yanjiu.huoqu_jibenmian("600000.SH", trade_date="2024-06-28")

    assert result["sources"]["profile"] == "tushare_live"
    assert result["data_quality"]["stock_basic"]["source"] == "tushare_live"
    assert result["data_quality"]["stock_basic"]["persistence"] == "none"


def test_completed_history_uses_calendar_confirmed_date_instead_of_weekday_guess() -> None:
    history = _flat_history(30)
    history.loc[len(history)] = {
        "trade_date": pd.Timestamp("2025-01-20"),
        "open": 10,
        "high": 10.1,
        "low": 9.9,
        "close": 10,
        "volume": 1_000_000,
    }

    completed, warnings = gupiao_yanjiu.guolv_wanzheng_jiaoyiri_lishi(
        history,
        latest_completed_date="2025-01-17",
    )

    assert completed["trade_date"].max() < pd.Timestamp("2025-01-20")
    assert any("尚未确认收盘" in warning for warning in warnings)


def test_financial_industry_debt_ratio_is_not_scored_like_an_industrial_company() -> None:
    fundamentals = {
        "profile": {"industry": "银行"},
        "financials": {
            "roe_pct": 10,
            "net_profit_yoy_pct": 5,
            "debt_to_assets_pct": 92,
        },
        "valuation": {"pe_ttm": 8},
    }

    _, evidence = jisuan_shendu_jibenmian(fundamentals)

    assert "金融行业资产负债率口径特殊，本项只展示、不加减分" in evidence
    assert "资产负债率偏高，需结合行业解释" not in evidence


def test_quant_config_rejects_out_of_range_validation_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    config = json.loads(peizhi.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["moxing"]["validation_ratio"] = 0.9
    path = tmp_path / "invalid_quant.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(peizhi, "DEFAULT_CONFIG_PATH", path)
    with pytest.raises(ValueError, match="validation_ratio"):
        peizhi.jiazai_lianghua_peizhi()
