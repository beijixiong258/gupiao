"""Offline tests for the full-market daily warehouse."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from src.ashare import dangu_yuce, gupiao_yanjiu, riping_cangku, riping_yinzi


class FakePro:
    def __init__(self, *, fail_basic_once: bool = False) -> None:
        self.calls: Counter[str] = Counter()
        self.fail_basic_once = fail_basic_once

    def trade_cal(self, **_kwargs) -> pd.DataFrame:
        self.calls["trade_cal"] += 1
        return pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240101", "is_open": 0, "pretrade_date": "20231229"},
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1, "pretrade_date": "20231229"},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 1, "pretrade_date": "20240102"},
            ]
        )

    def stock_basic(self, *, list_status: str, **_kwargs) -> pd.DataFrame:
        self.calls[f"stock_basic_{list_status}"] += 1
        if list_status == "L":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "600001.SH",
                        "symbol": "600001",
                        "name": "样本一",
                        "area": "上海",
                        "industry": "电子",
                        "market": "主板",
                        "exchange": "SSE",
                        "list_status": "L",
                        "list_date": "20200101",
                        "delist_date": None,
                    }
                ]
            )
        if list_status == "D":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "600002.SH",
                        "symbol": "600002",
                        "name": "退市样本",
                        "area": "上海",
                        "industry": "电子",
                        "market": "主板",
                        "exchange": "SSE",
                        "list_status": "D",
                        "list_date": "20100101",
                        "delist_date": "20231231",
                    }
                ]
            )
        return pd.DataFrame()

    def daily(self, *, trade_date: str, **_kwargs) -> pd.DataFrame:
        self.calls[f"daily_{trade_date}"] += 1
        close = 10.0 if trade_date == "20240102" else 20.0
        return pd.DataFrame(
            [
                {
                    "ts_code": "600001.SH",
                    "trade_date": trade_date,
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 2.0,
                    "close": close,
                    "pre_close": close - 0.5,
                    "change": 0.5,
                    "pct_chg": 1.0,
                    "vol": 1000.0,
                    "amount": 100.0,
                }
            ]
        )

    def daily_basic(self, *, trade_date: str, **_kwargs) -> pd.DataFrame:
        self.calls[f"daily_basic_{trade_date}"] += 1
        if self.fail_basic_once:
            self.fail_basic_once = False
            raise RuntimeError("temporary daily_basic failure")
        return pd.DataFrame(
            [
                {
                    "ts_code": "600001.SH",
                    "trade_date": trade_date,
                    "close": 10.0,
                    "turnover_rate": 2.0,
                    "turnover_rate_f": 3.0,
                    "volume_ratio": 1.1,
                    "pe": 12.0,
                    "pe_ttm": 13.0,
                    "pb": 2.0,
                    "ps": 1.0,
                    "ps_ttm": 1.1,
                    "dv_ratio": 0.5,
                    "dv_ttm": 0.6,
                    "total_share": 1000.0,
                    "float_share": 800.0,
                    "free_share": 700.0,
                    "total_mv": 100000.0,
                    "circ_mv": 80000.0,
                }
            ]
        )

    def adj_factor(self, *, trade_date: str, **_kwargs) -> pd.DataFrame:
        self.calls[f"adj_factor_{trade_date}"] += 1
        factor = 1.0 if trade_date == "20240102" else 2.0
        return pd.DataFrame(
            [{"ts_code": "600001.SH", "trade_date": trade_date, "adj_factor": factor}]
        )


def test_daily_warehouse_sync_resume_and_qfq_read(tmp_path) -> None:
    database = tmp_path / "daily.sqlite3"
    pro = FakePro()

    result = riping_cangku.sync_daily_warehouse(
        start_date="2024-01-01",
        end_date="2024-01-03",
        max_sessions=0,
        newest_first=False,
        pause_seconds=0,
        path=database,
        pro=pro,
    )
    history, metadata = riping_cangku.load_qfq_history_from_warehouse(
        "600001.SH",
        start_date="2024-01-01",
        end_date="2024-01-03",
        path=database,
        minimum_rows=1,
    )

    assert result["status"] == "ok"
    assert result["warehouse_status"]["complete_sessions"] == 2
    assert result["warehouse_status"]["row_counts"]["daily_bars"] == 2
    assert result["stock_snapshot"]["delisted"] == 1
    assert metadata["adjustment"] == "qfq_by_warehouse_adj_factor"
    assert history["close"].tolist() == pytest.approx([5.0, 20.0])
    assert history["amount_yuan"].tolist() == pytest.approx([100000.0, 100000.0])
    coverage = riping_cangku.warehouse_range_coverage(
        start_date="2024-01-01",
        end_date="2024-01-03",
        path=database,
    )
    assert coverage["coverage"] == 1.0
    assert coverage["ready"] is False

    endpoint_calls = {
        key: value
        for key, value in pro.calls.items()
        if key.startswith(("daily_", "adj_factor_"))
    }
    second = riping_cangku.sync_daily_warehouse(
        start_date="2024-01-01",
        end_date="2024-01-03",
        max_sessions=0,
        pause_seconds=0,
        path=database,
        pro=pro,
    )
    assert second["attempted_sessions"] == 0
    assert {
        key: value
        for key, value in pro.calls.items()
        if key.startswith(("daily_", "adj_factor_"))
    } == endpoint_calls


def test_daily_warehouse_retries_only_failed_endpoint(tmp_path) -> None:
    database = tmp_path / "retry.sqlite3"
    pro = FakePro(fail_basic_once=True)
    first = riping_cangku.sync_daily_warehouse(
        start_date="2024-01-02",
        end_date="2024-01-02",
        max_sessions=1,
        pause_seconds=0,
        path=database,
        pro=pro,
    )
    assert first["status"] == "partial"
    assert first["sessions"][0]["endpoints"]["bars"]["status"] == "complete"
    assert first["sessions"][0]["endpoints"]["basic"]["status"] == "failed"

    second = riping_cangku.sync_daily_warehouse(
        start_date="2024-01-02",
        end_date="2024-01-02",
        max_sessions=1,
        pause_seconds=0,
        path=database,
        pro=pro,
    )
    assert second["status"] == "ok"
    assert second["sessions"][0]["endpoints"]["bars"]["status"] == "already_complete"
    assert second["sessions"][0]["endpoints"]["adj"]["status"] == "already_complete"
    assert second["sessions"][0]["endpoints"]["basic"]["status"] == "complete"
    assert pro.calls["daily_20240102"] == 1
    assert pro.calls["adj_factor_20240102"] == 1
    assert pro.calls["daily_basic_20240102"] == 2


def test_stock_history_fetch_prefers_complete_warehouse(monkeypatch: pytest.MonkeyPatch) -> None:
    history = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=60),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "pre_close": 10.0,
            "pct_chg": 5.0,
            "volume": 1000.0,
            "amount_yuan": 100000.0,
            "turnover_rate": 2.0,
        }
    )
    monkeypatch.setattr(gupiao_yanjiu, "_load_daily_bar_cache", lambda **_kwargs: None)
    monkeypatch.setattr(
        riping_cangku,
        "load_qfq_history_from_warehouse",
        lambda *_args, **_kwargs: (
            history,
            {
                "status": "ok",
                "source": "tushare_daily_warehouse",
                "adjustment": "qfq_by_warehouse_adj_factor",
                "sync_coverage": 1.0,
            },
        ),
    )
    monkeypatch.setattr(
        gupiao_yanjiu,
        "_tushare_pro",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse hit must avoid network")),
    )

    result = gupiao_yanjiu.huoqu_rili_xingqing(
        "600001.SH",
        start_date="20240101",
        end_date="20240401",
        source="auto",
        use_cache=True,
    )

    assert result.source == "tushare_daily_warehouse"
    assert result.adjustment == "qfq_by_warehouse_adj_factor"
    assert len(result.data) == 60


def test_daily_factor_loader_uses_complete_warehouse_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "trade_date": "2024-01-02",
                "turnover_rate": 2.0,
                "pe_ttm": 13.0,
                "pb": 2.0,
                "total_mv": 100000.0,
                "circ_mv": 80000.0,
            }
        ]
    )
    monkeypatch.setattr(
        riping_cangku,
        "load_daily_basic_from_warehouse",
        lambda *_args, **_kwargs: (
            warehouse,
            {"status": "ok", "calendar_sync_coverage": 1.0, "source": "warehouse"},
        ),
    )
    monkeypatch.setattr(
        riping_yinzi,
        "_tushare_pro",
        lambda: (_ for _ in ()).throw(AssertionError("complete warehouse must avoid network")),
    )

    result, metadata = riping_yinzi._historical_daily_basic(
        codes=["600001.SH"],
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-02"),
    )

    assert len(result) == 1
    assert metadata["source"] == "warehouse"
    assert metadata["merge_rule"].startswith("仅按股票代码")


def test_complete_warehouse_expands_daily_peer_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = pd.DataFrame(
        [
            {
                "ts_code": f"{600000 + index:06d}.SH",
                "name": f"样本{index}",
                "industry": "电子" if index <= 50 else "机械",
                "latest_price": 10.0 + index / 10,
                "amount_yuan": 1_000_000_000.0 - index,
            }
            for index in range(1, 71)
        ]
    )
    monkeypatch.setattr(
        riping_cangku,
        "warehouse_range_coverage",
        lambda **_kwargs: {"status": "ok", "coverage": 1.0, "ready": True},
    )
    monkeypatch.setattr(
        dangu_yuce,
        "_latest_tushare_cross_section",
        lambda _signal: (universe, "2024-12-31", []),
    )

    selected, metadata = dangu_yuce.xuanze_tonghang_yangben(
        code="600001.SH",
        name="样本1",
        industry="电子",
        signal_date="2024-12-31",
        config={
            "dangu": {
                "history_calendar_days": 1440,
                "max_peer_stocks": 20,
                "same_industry_stocks": 16,
                "warehouse_max_peer_stocks": 60,
                "warehouse_same_industry_stocks": 45,
                "min_amount_yuan": 0,
            }
        },
    )

    assert len(selected) == 60
    assert metadata["configured_peer_limit_without_warehouse"] == 20
    assert metadata["applied_peer_limit"] == 60
    assert metadata["warehouse_expansion_applied"] is True
