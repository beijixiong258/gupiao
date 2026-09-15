"""指数来源质量和交易日窗口的离线契约，不访问行情接口。"""

from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from src.ashare import riping_yinzi
from src.ashare.shichang_shuju import JiaoyiRili


@pytest.fixture
def calendar() -> JiaoyiRili:
    # 显式给定日历，包含清明休市缺口；不以行情日期或星期推断交易日。
    dates = pd.to_datetime([
        "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07",
        "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14",
        "2025-03-17", "2025-03-18", "2025-03-19", "2025-03-20", "2025-03-21",
        "2025-03-24", "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28",
        "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-07",
        "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11", "2025-04-14",
        "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18", "2025-04-21",
        "2025-04-22", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-28",
        "2025-04-29", "2025-04-30",
    ])
    return JiaoyiRili(
        open_dates=frozenset(dates), source="tushare_trade_cal",
        start_date="2025-03-01", end_date="2025-04-30", fetched_at="2025-04-30T16:00:00+08:00",
    )


def _history(calendar: JiaoyiRili, *, step: float = 1.0) -> pd.DataFrame:
    dates = sorted(calendar.open_dates)
    return pd.DataFrame({"trade_date": dates, "close": 100.0 + np.arange(len(dates)) * step})


def _providers(monkeypatch, primary: pd.DataFrame, fallback: pd.DataFrame) -> list[dict]:
    requests: list[dict] = []

    def tushare(**kwargs):
        requests.append({"source": "tushare", **kwargs})
        return primary.copy()

    def akshare(**kwargs):
        requests.append({"source": "akshare", **kwargs})
        return fallback.copy()

    monkeypatch.setattr(riping_yinzi, "_tushare_pro", lambda: SimpleNamespace(index_daily=tushare))
    monkeypatch.setattr(riping_yinzi, "akshare_zhilian", nullcontext)
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_zh_index_daily=akshare))
    return requests


def _fetch(calendar, *, source="auto"):
    quality: dict = {}
    history, provider, warnings = riping_yinzi._fetch_one_benchmark(
        key="csi300", start=min(calendar.open_dates), end=max(calendar.open_dates),
        source=source, calendar=calendar, quality=quality,
    )
    return history, provider, warnings, quality


@pytest.mark.parametrize("defect", ["missing_latest", "missing_middle", "zero", "infinite", "non_numeric", "duplicate"])
def test_current_window_quality_rejects_primary_and_uses_whole_fallback(monkeypatch, calendar, defect):
    primary = _history(calendar)
    bad_index = len(primary) - 5
    if defect == "missing_latest":
        bad_index = len(primary) - 1
        primary = primary.drop(index=bad_index)
    elif defect == "missing_middle":
        primary = primary.drop(index=bad_index)
    elif defect == "duplicate":
        primary = pd.concat([primary, primary.iloc[[bad_index]]], ignore_index=True)
    else:
        primary["close"] = primary["close"].astype(object)
        primary.loc[bad_index, "close"] = {"zero": 0, "infinite": np.inf, "non_numeric": "无"}[defect]
    fallback = _history(calendar, step=3.0)
    requests = _providers(monkeypatch, primary, fallback)

    history, provider, warnings, quality = _fetch(calendar)

    assert [item["source"] for item in requests] == ["tushare", "akshare"]
    assert provider == "akshare"
    pd.testing.assert_frame_equal(history[["trade_date", "close"]], fallback)
    assert warnings and quality["degraded"] is True
    failed, accepted = quality["attempted_providers"]
    assert failed["accepted"] is False and accepted["accepted"] is True
    assert fallback.iloc[bad_index]["trade_date"].strftime("%Y-%m-%d") in failed["required_window_missing_dates"]
    if defect in {"zero", "infinite", "non_numeric"}:
        assert "invalid_close_in_required_window" in failed["reasons"]
        assert failed["invalid_close_dates"]
    if defect == "duplicate":
        assert failed["duplicate_trade_dates"]


def test_valid_primary_does_not_request_fallback_and_holiday_is_not_a_missing_session(monkeypatch, calendar):
    expected = _history(calendar)
    requests = _providers(monkeypatch, expected, pd.DataFrame())

    history, provider, warnings, quality = _fetch(calendar)

    assert [item["source"] for item in requests] == ["tushare"]
    assert provider == "tushare" and not warnings
    assert len(history) == len(expected)
    assert quality["quality"]["missing_trade_dates"] == []
    assert quality["calendar"]["fetched_at"] == calendar.fetched_at


def test_no_qualified_source_returns_unavailable_and_preserves_both_failures(monkeypatch, calendar):
    primary = _history(calendar).iloc[:-1]
    fallback = _history(calendar).drop(index=25)
    _providers(monkeypatch, primary, fallback)

    history, provider, warnings, quality = _fetch(calendar)

    assert history.empty and provider == "unavailable"
    assert quality["status"] == "unavailable"
    assert len(warnings) == 2
    assert all(not attempt["accepted"] for attempt in quality["attempted_providers"])
    assert quality["attempted_providers"][0]["required_window_missing_dates"] == ["2025-04-30"]
    assert quality["attempted_providers"][1]["required_window_missing_dates"] == ["2025-04-08"]


def test_explicit_source_preserves_provider_boundary(monkeypatch, calendar):
    requests = _providers(monkeypatch, _history(calendar).iloc[:-1], _history(calendar))
    history, provider, _, _ = _fetch(calendar, source="tushare")
    assert history.empty and provider == "unavailable"
    assert [item["source"] for item in requests] == ["tushare"]


def test_missing_calendar_does_not_fetch_or_infer_trading_days(monkeypatch, calendar):
    requests = _providers(monkeypatch, _history(calendar), _history(calendar))
    quality: dict = {}
    history, provider, warnings = riping_yinzi._fetch_one_benchmark(
        key="csi300", start=min(calendar.open_dates), end=max(calendar.open_dates),
        source="auto", quality=quality,
    )
    assert history.empty and provider == "unavailable" and warnings
    assert requests == []
    assert quality["reasons"] == ["calendar_unavailable"]


def test_short_stock_history_still_requests_complete_benchmark_window(monkeypatch, calendar):
    expected = _history(calendar)
    requests = _providers(monkeypatch, expected, pd.DataFrame())
    history, provider, warnings = riping_yinzi._fetch_one_benchmark(
        key="csi300", start=expected.iloc[-3]["trade_date"], end=max(calendar.open_dates),
        source="auto", calendar=calendar,
    )
    assert provider == "tushare" and not warnings
    assert requests[0]["start_date"] == expected.iloc[-21]["trade_date"].strftime("%Y%m%d")
    assert len(history) == 21


def test_older_gap_masks_every_affected_window_without_shifting_trading_periods(monkeypatch, calendar):
    expected = _history(calendar)
    gap_index = 8
    _providers(monkeypatch, expected.drop(index=gap_index), pd.DataFrame())

    features, meta = riping_yinzi._benchmark_features(
        start=min(calendar.open_dates), end=max(calendar.open_dates), source="auto", calendar=calendar,
    )

    assert meta["status"] == "partial"
    values = features.set_index("trade_date")
    # 端点均有效但窗口中间缺了一天，5 日收益也必须留空。
    assert pd.isna(values.loc[expected.iloc[gap_index + 2]["trade_date"], "market_csi300_ret_5"])
    assert pd.isna(values.loc[expected.iloc[gap_index + 20]["trade_date"], "market_csi300_ret_20"])
    assert pd.isna(values.loc[expected.iloc[gap_index + 20]["trade_date"], "market_csi300_volatility_20"])
    latest = values.loc[expected.iloc[-1]["trade_date"]]
    assert latest["market_csi300_ret_5"] == pytest.approx(expected.iloc[-1]["close"] / expected.iloc[-6]["close"] - 1)
    assert latest["market_csi300_ret_20"] == pytest.approx(expected.iloc[-1]["close"] / expected.iloc[-21]["close"] - 1)
    assert meta["benchmarks"]["csi300"]["quality"]["missing_trade_dates"] == ["2025-03-13"]


def test_one_missing_benchmark_is_partial_when_other_indices_are_available(monkeypatch, calendar):
    expected = _history(calendar)

    def primary(**kwargs):
        return pd.DataFrame() if kwargs["ts_code"] == "000300.SH" else expected.copy()

    monkeypatch.setattr(riping_yinzi, "_tushare_pro", lambda: SimpleNamespace(index_daily=primary))
    features, meta = riping_yinzi._benchmark_features(
        start=min(calendar.open_dates), end=max(calendar.open_dates), source="tushare", calendar=calendar,
    )
    assert not features.empty and meta["status"] == "partial"
    assert meta["benchmarks"]["csi300"]["status"] == "unavailable"
    assert "market_csi300_ret_20" not in features.columns
    assert "market_shanghai_ret_20" in features.columns
