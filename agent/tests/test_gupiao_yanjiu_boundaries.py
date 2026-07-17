"""Regression tests for single-stock data-quality boundaries."""

from __future__ import annotations

import json
import sys
import os
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ashare import gupiao_yanjiu


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


def test_only_known_920_segment_is_inferred_from_a_code_starting_with_nine() -> None:
    assert gupiao_yanjiu.biaozhunhua_daima("920001") == "920001.BJ"
    with pytest.raises(ValueError, match="不属于.*A 股"):
        gupiao_yanjiu.biaozhunhua_daima("900901")
    with pytest.raises(ValueError, match="不属于.*A 股"):
        gupiao_yanjiu.biaozhunhua_daima("400001")


def test_single_stock_rules_recognize_n_and_c_no_limit_markers() -> None:
    n_rules = gupiao_yanjiu._a_share_rules("688001.SH", "N样本")
    c_rules = gupiao_yanjiu._a_share_rules("300001.SZ", "C样本")
    normal_rules = gupiao_yanjiu._a_share_rules("688001.SH", "样本")

    assert n_rules["price_limit_status"] == "no_limit"
    assert n_rules["price_limit_pct"] is None
    assert c_rules["price_limit_status"] == "no_limit"
    assert normal_rules["price_limit_status"] == "limited"
    assert normal_rules["price_limit_pct"] == 20.0


def test_expired_akshare_name_cache_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    old = datetime.now().timestamp() - 7200
    calls = {"count": 0}

    class FakePath:
        parent: "FakePath"

        def __init__(self) -> None:
            self.parent = self

        def is_file(self) -> bool:
            return True

        def stat(self):
            return SimpleNamespace(st_mtime=old)

        def mkdir(self, **_kwargs) -> None:
            return None

    def fetch_names() -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame([{"code": "600000", "name": "新名称"}])

    monkeypatch.setattr(gupiao_yanjiu, "AK_STOCK_NAMES_CACHE", FakePath())
    monkeypatch.setattr(gupiao_yanjiu, "AK_STOCK_NAMES_CACHE_TTL_SECONDS", 60)
    monkeypatch.setattr(gupiao_yanjiu, "akshare_zhilian", nullcontext)
    monkeypatch.setattr(
        gupiao_yanjiu.pd,
        "read_csv",
        lambda *_args, **_kwargs: pd.DataFrame([{"ts_code": "600000.SH", "name": "旧名称"}]),
    )
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_info_a_code_name=fetch_names))

    result = gupiao_yanjiu._akshare_name_table()

    assert calls["count"] == 1
    assert result.to_dict("records") == [{"ts_code": "600000.SH", "name": "新名称"}]


def test_expired_tushare_name_cache_is_not_used_as_current_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "stock_basic.csv"
    pd.DataFrame([{"ts_code": "600000.SH", "name": "旧名称"}]).to_csv(cache, index=False)
    old = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(cache, (old, old))
    monkeypatch.setattr(gupiao_yanjiu, "STOCK_BASIC_CACHE", cache)

    assert gupiao_yanjiu._stock_basic_cache().empty


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
    monkeypatch.setattr(gupiao_yanjiu, "_load_or_fetch_stock_basic", lambda _pro, _cache: profile)

    result = gupiao_yanjiu.huoqu_jibenmian("600000.SH", trade_date="2024-06-28")

    assert result["valuation"]["as_of"] == "2024-06-28"
    assert result["financials"]["announcement_date"] == "2024-04-20"
    assert result["financials"]["roe_pct"] == 10.0
    assert result["financials"]["missing_fields"] == ["net_profit_yoy_pct"]


def test_profile_cache_quality_and_refresh_warning_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pro:
        def daily_basic(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

        def fina_indicator(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    def load(_pro, quality: dict[str, object]) -> pd.DataFrame:
        quality["stock_basic"] = {"source": "stale_cache", "rows": 1}
        quality["warnings"] = ["stock_basic refresh failed; using stale cache"]
        return pd.DataFrame([{"ts_code": "600000.SH", "name": "旧名称"}])

    monkeypatch.setattr(gupiao_yanjiu, "_tushare_pro", lambda: Pro())
    monkeypatch.setattr(gupiao_yanjiu, "_load_or_fetch_stock_basic", load)
    monkeypatch.setattr(gupiao_yanjiu, "_akshare_info", lambda _code: ({}, []))

    result = gupiao_yanjiu.huoqu_jibenmian("600000.SH", trade_date="2024-06-28")

    assert result["sources"]["profile"] == "tushare_stale_cache"
    assert result["data_quality"]["stock_basic"]["source"] == "stale_cache"
    assert any("refresh failed" in warning for warning in result["warnings"])


def test_market_freshness_uses_completed_weekdays_and_marks_old_data() -> None:
    during_session = datetime(2025, 1, 20, 10, 0)
    history = _flat_history(30)
    history.loc[len(history)] = {
        "trade_date": pd.Timestamp("2025-01-20"),
        "open": 10,
        "high": 10.1,
        "low": 9.9,
        "close": 10,
        "volume": 1_000_000,
    }

    completed, warnings = gupiao_yanjiu._completed_market_history(history, reference=during_session)
    freshness = gupiao_yanjiu._market_data_freshness("2025-01-03", reference=datetime(2025, 1, 20, 16, 0))

    assert completed["trade_date"].max() < pd.Timestamp("2025-01-20")
    assert any("尚未确认收盘" in warning for warning in warnings)
    assert freshness["status"] == "too_stale"
    assert freshness["business_days_old"] > gupiao_yanjiu.MARKET_DATA_STALE_ERROR_BUSINESS_DAYS


def test_undated_akshare_snapshot_is_not_used_during_the_trading_session() -> None:
    monday_morning = datetime(2025, 1, 20, 10, 0)
    monday_closed = datetime(2025, 1, 20, 16, 0)
    saturday = datetime(2025, 1, 18, 10, 0)

    assert not gupiao_yanjiu._can_use_current_akshare_snapshot(
        "2025-01-17", reference=monday_morning
    )
    assert gupiao_yanjiu._can_use_current_akshare_snapshot(
        "2025-01-20", reference=monday_closed
    )
    assert gupiao_yanjiu._can_use_current_akshare_snapshot(
        "2025-01-17", reference=saturday
    )


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

    _, evidence = gupiao_yanjiu._fundamental_score(fundamentals)

    assert "金融行业资产负债率口径特殊，本项只展示、不加减分" in evidence
    assert "资产负债率偏高，需结合行业解释" not in evidence


def test_quant_config_rejects_out_of_range_validation_settings(tmp_path: Path) -> None:
    config = json.loads(gupiao_yanjiu.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["moxing"]["validation_ratio"] = 0.9
    path = tmp_path / "invalid_quant.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="validation_ratio"):
        gupiao_yanjiu.jiazai_lianghua_peizhi(str(path))
