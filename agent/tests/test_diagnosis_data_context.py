"""诊断数据边界的离线回归：长窗口、初始化降级和请求内复用。"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ashare import gupiao_yanjiu as research
from src.ashare import shichang_shuju as market


def _config():
    return {
        "dangu": {"history_calendar_days": 1440},
        "fenxi": {"history_calendar_days": 420, "minimum_history_session_coverage": 0.9},
        "wangluo": {"tushare_max_attempts": 1, "retry_backoff_seconds": 0, "history_fallback_max_stocks": 0},
        "shuju": {"request_pause_seconds": 0},
    }, None


def test_tushare_initialization_failure_uses_remote_calendar_fallback(monkeypatch):
    monkeypatch.setattr(market, "jiazai_lianghua_peizhi", _config)
    monkeypatch.setattr(market, "_tushare_pro", lambda: (_ for _ in ()).throw(RuntimeError("TUSHARE_TOKEN not set")))
    requested = []

    def fallback(*, start, end):
        requested.append((start, end))
        return frozenset([pd.Timestamp("2026-09-11")]), {"fetched_at": "2026-09-12 10:00:00"}

    monkeypatch.setattr(market, "_huoqu_xinlang_jiaoyi_rili", fallback)
    calendar = market.huoqu_jiaoyi_rili(datetime(2026, 9, 12), start_date="2022-10-03")

    assert calendar.source == "sina_exchange_calendar"
    assert requested[0][0] == pd.Timestamp("2022-10-03")
    assert calendar.attempted_providers[0]["outcome"] == "failed"
    assert "TUSHARE_TOKEN" in calendar.attempted_providers[0]["detail"]


def test_requested_history_is_not_truncated_to_550_days(monkeypatch):
    monkeypatch.setattr(market, "jiazai_lianghua_peizhi", _config)
    requests = []

    def trade_cal(**kwargs):
        requests.append(kwargs)
        dates = pd.date_range(kwargs["start_date"], kwargs["end_date"], freq="D")
        return pd.DataFrame({"cal_date": dates.strftime("%Y%m%d"), "is_open": (dates.dayofweek < 5).astype(int)})

    monkeypatch.setattr(market, "_tushare_pro", lambda: SimpleNamespace(trade_cal=trade_cal))
    dates = pd.bdate_range("2022-10-03", "2026-09-11")
    history = pd.DataFrame({"trade_date": dates, "close": 10.0})
    monkeypatch.setattr(market, "_qfq_history_from_tencent", lambda *args, **kwargs: ({"000001.SZ": history}, {}))
    context = market.FenxiShujuShangxiawen(reference=datetime(2026, 9, 12))
    context.shichang_shizhong()
    ready, metadata = context.piliang_lishi(["000001.SZ"], start_date="20221003", end_date="20260911", minimum_rows=180)

    assert len(requests) == 1
    assert ready["000001.SZ"]["trade_date"].min() == dates.min()
    assert len(ready["000001.SZ"]) == len(dates)
    assert metadata["status"] == "ok"


def test_source_calendar_must_actually_cover_requested_history(monkeypatch):
    monkeypatch.setattr(market, "jiazai_lianghua_peizhi", _config)
    short_calendar = pd.DataFrame({"cal_date": ["20260911", "20260912"], "is_open": [1, 0]})
    monkeypatch.setattr(market, "_tushare_pro", lambda: SimpleNamespace(trade_cal=lambda **kwargs: short_calendar))
    fallback_calls = []

    def fallback(*, start, end):
        fallback_calls.append(start)
        return frozenset(pd.bdate_range(start, end)), {}

    monkeypatch.setattr(market, "_huoqu_xinlang_jiaoyi_rili", fallback)
    calendar = market.huoqu_jiaoyi_rili(datetime(2026, 9, 12), start_date="2022-10-03")
    assert calendar.source == "sina_exchange_calendar"
    assert fallback_calls == [pd.Timestamp("2022-10-03")]
    assert "未覆盖请求区间" in calendar.attempted_providers[0]["detail"]


def test_explicit_short_calendar_cannot_silently_trim_history(monkeypatch):
    def must_not_download(*args, **kwargs):
        pytest.fail("日历边界不完整时不应下载日线")

    monkeypatch.setattr(market, "_qfq_history_from_tencent", must_not_download)
    calendar = market.JiaoyiRili(
        open_dates=frozenset(pd.bdate_range("2026-01-01", "2026-09-11")),
        source="test", start_date="2026-01-01", end_date="2026-09-12",
    )
    ready, metadata = market.huoqu_piliang_qfq_lishi(
        ["000001.SZ"], start_date="20221003", end_date="20260911", minimum_rows=180, calendar=calendar,
    )
    assert ready == {}
    assert metadata["status"] == "unavailable"
    assert metadata["error_code"] == "calendar_coverage_insufficient"


def test_identity_cross_section_and_fundamentals_share_market_requests(monkeypatch):
    import sys

    counts = {"basic": 0, "snapshot": 0}
    basic = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"], "name": ["样本甲", "样本乙"],
        "industry": ["样本行业", "样本行业"], "list_date": ["20000101", "20000101"],
    })

    def stock_basic(**kwargs):
        counts["basic"] += 1
        return basic.copy()

    pro = SimpleNamespace(stock_basic=stock_basic, daily_basic=lambda **kwargs: pd.DataFrame(), fina_indicator=lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(market, "_tushare_pro", lambda: pro)
    monkeypatch.setattr(research, "_tushare_pro", lambda: pro)
    daily = pd.DataFrame({"ts_code": basic["ts_code"], "close": 10.0, "open": 9.9, "high": 10.1, "low": 9.8,
                          "pre_close": 9.9, "pct_chg": 1.0, "vol": 10000.0, "amount": 10000.0})
    monkeypatch.setattr(market, "_latest_tushare_daily", lambda *args: ("20260911", daily))

    def snapshot(*args):
        counts["snapshot"] += 1
        return pd.DataFrame({"ts_code": basic["ts_code"], "name": basic["name"], "pe_dynamic": [12.0, 13.0], "pb": 1.2}), {"status": "ok", "source": "test"}

    monkeypatch.setattr(market, "huoqu_shishi_kuaizhao", snapshot)
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_individual_info_em=lambda **kwargs: pd.DataFrame()))
    monkeypatch.setattr(research, "akshare_zhilian", nullcontext)
    monkeypatch.setattr(research, "_akshare_financials", lambda *args, **kwargs: ({}, []))
    context = market.FenxiShujuShangxiawen(reference=datetime(2026, 9, 12))
    context._memo["zuixin_wanzheng_jiaoyiri"] = pd.Timestamp("2026-09-11")

    assert context.jiexi_gupiao("000001")[1]["name"] == "样本甲"
    assert len(context.zuixin_hengjiemian()[0]) == 2
    context.shishi_kuaizhao()
    first = context.jibenmian("000001.SZ", trade_date="2026-09-11", allow_current_snapshot=True)
    second = context.jibenmian("000002.SZ", trade_date="2026-09-11", allow_current_snapshot=True)
    assert first["valuation"]["pe_dynamic"] == 12.0
    assert second["valuation"]["pe_dynamic"] == 13.0
    first["profile"]["name"] = "不可污染复用结果"
    assert context.jibenmian("000001.SZ", trade_date="2026-09-11", allow_current_snapshot=True)["profile"]["name"] == "样本甲"
    assert counts == {"basic": 1, "snapshot": 1}


def test_failed_stock_basic_request_is_not_repeated_per_candidate(monkeypatch):
    calls = []

    def unavailable():
        calls.append(True)
        raise RuntimeError("资料接口暂不可用")

    monkeypatch.setattr(market, "_tushare_pro", unavailable)
    context = market.FenxiShujuShangxiawen()
    for _ in range(3):
        with pytest.raises(RuntimeError, match="资料接口暂不可用"):
            context.gupiao_ziliao()
    assert len(calls) == 1


def test_disallowed_current_valuation_does_not_fetch_unused_snapshot(monkeypatch):
    pro = SimpleNamespace(daily_basic=lambda **kwargs: pd.DataFrame(), fina_indicator=lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(research, "_tushare_pro", lambda: pro)

    def must_not_fetch(*args, **kwargs):
        pytest.fail("已有身份且禁止使用当前估值时不应再获取实时快照")

    monkeypatch.setattr(research, "_akshare_info", must_not_fetch)
    result = research.huoqu_jibenmian(
        "000001.SZ", trade_date="2026-09-11", allow_current_snapshot=False,
        stock_basic_loader=lambda: (pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["样本甲"]}), {}),
    )
    assert result["profile"]["name"] == "样本甲"
    assert result["valuation"] == {}
