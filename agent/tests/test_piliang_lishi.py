"""批量日线逐源质量检查和受限备用链的离线回归。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.ashare import shichang_shuju as market


_SESSIONS = pd.bdate_range("2026-08-03", "2026-08-28")
_CODE = "000001.SZ"


def _history(dates: Any = _SESSIONS, *, price: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": dates, "close": price})


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    primary: dict[str, pd.DataFrame],
    secondary: dict[str, pd.DataFrame] | None = None,
    tertiary: dict[str, pd.DataFrame] | None = None,
    *,
    fallback_limit: int = 2,
) -> dict[str, list[tuple[str, ...]]]:
    calls: dict[str, list[tuple[str, ...]]] = {"primary": [], "secondary": [], "tertiary": []}

    def source_loader(source: str, histories: dict[str, pd.DataFrame]):
        def load(codes: tuple[str, ...], **kwargs: Any):
            calls[source].append(codes)
            result = {code: histories[code] for code in codes if code in histories}
            metadata = [] if source == "tertiary" else {"loaded_stocks": len(result)}
            return result, metadata

        return load

    monkeypatch.setattr(market, "_qfq_history_from_tencent", source_loader("primary", primary))
    monkeypatch.setattr(market, "_qfq_history_from_eastmoney", source_loader("secondary", secondary or {}))
    monkeypatch.setattr(market, "_qfq_history_from_tushare", source_loader("tertiary", tertiary or {}))
    monkeypatch.setattr(
        market,
        "jiazai_lianghua_peizhi",
        lambda: (
            {
                "fenxi": {"minimum_history_session_coverage": 0.9},
                "wangluo": {"history_fallback_max_stocks": fallback_limit},
                "shuju": {"request_pause_seconds": 0.0},
            },
            None,
        ),
    )
    return calls


def _load(codes: tuple[str, ...] = (_CODE,), *, end_date: str = "20260828"):
    calendar = market.JiaoyiRili(
        open_dates=frozenset(_SESSIONS),
        source="test_calendar",
        start_date="2026-08-03",
        end_date="2026-08-31",
    )
    return market.huoqu_piliang_qfq_lishi(
        codes,
        start_date="20260803",
        end_date=end_date,
        minimum_rows=10,
        calendar=calendar,
    )


@pytest.mark.parametrize(
    ("dates", "reason"),
    [
        (_SESSIONS[-5:], "insufficient_rows"),
        (_SESSIONS[::2].union(_SESSIONS[-1:]), "insufficient_session_coverage"),
        (_SESSIONS[:-1], "latest_session_missing"),
    ],
)
def test_incomplete_nonempty_primary_history_uses_qualified_secondary(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex, reason: str
) -> None:
    calls = _install_sources(monkeypatch, {_CODE: _history(dates)}, {_CODE: _history(price=20.0)})

    ready, metadata = _load()

    assert calls["secondary"] == [(_CODE,)]
    assert calls["tertiary"] == []
    assert len(ready[_CODE]) == len(_SESSIONS)
    assert ready[_CODE]["close"].eq(20.0).all()
    assert metadata["status"] == "ok"
    assert metadata["incomplete_count"] == 0
    assert metadata["quality_failure_count"] == 1
    assert reason in metadata["quality_failures"][0]["reasons"]
    assert metadata["source_counts"] == {
        "tencent_qfq_daily": 0,
        "eastmoney_qfq_fallback": 1,
        "tushare_qfq_fallback": 0,
    }
    assert metadata["persistence"] == "none"


def test_secondary_quality_failure_uses_tertiary_without_double_counting_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_sources(
        monkeypatch,
        {_CODE: _history(_SESSIONS[-5:])},
        {_CODE: _history(_SESSIONS[:-1], price=20.0)},
        {_CODE: _history(price=30.0)},
    )

    ready, metadata = _load()

    assert calls["tertiary"] == [(_CODE,)]
    assert ready[_CODE]["close"].eq(30.0).all()
    assert metadata["quality_failure_count"] == 2
    assert metadata["source_counts"] == {
        "tencent_qfq_daily": 0,
        "eastmoney_qfq_fallback": 0,
        "tushare_qfq_fallback": 1,
    }
    assert metadata["source"] == "tushare_qfq_fallback"


def test_all_sources_must_pass_the_same_latest_session_check(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = {_CODE: _history(_SESSIONS[:-1])}
    _install_sources(monkeypatch, stale, stale, stale)

    ready, metadata = _load()

    assert ready == {}
    assert metadata["status"] == "unavailable"
    assert metadata["incomplete_count"] == 1
    assert metadata["quality_failure_count"] == 3
    assert all(item["reasons"] == ["latest_session_missing"] for item in metadata["quality_failures"])
    assert metadata["incomplete_examples"][0]["required_latest_date"] == "2026-08-28"
    assert sum(metadata["source_counts"].values()) == 0


def test_incomplete_histories_share_existing_fallback_request_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")
    primary = {code: _history(_SESSIONS[-5:]) for code in codes}
    primary[codes[0]] = _history()
    calls = _install_sources(
        monkeypatch,
        primary,
        {code: _history(price=20.0) for code in codes},
        fallback_limit=1,
    )

    ready, metadata = _load(codes)

    assert set(ready) == set(codes[:2])
    assert calls["secondary"] == [(codes[1],)]
    assert calls["tertiary"] == []
    assert metadata["fallback_attempted_stocks"] == 1
    assert metadata["incomplete_count"] == 2
    assert any("按上限只尝试 1 只" in warning for warning in metadata["warnings"])


def test_good_primary_needs_no_fallback_and_weekend_end_uses_last_open_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_sources(monkeypatch, {_CODE: _history()})

    ready, metadata = _load(end_date="20260830")

    assert len(ready[_CODE]) == 20
    assert calls["secondary"] == []
    assert calls["tertiary"] == []
    assert metadata["actual_range"] == ["2026-08-03", "2026-08-28"]
    assert metadata["quality_failure_count"] == 0


def test_duplicate_dates_do_not_inflate_minimum_history_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    repeated = pd.concat([_history(_SESSIONS[-5:])] * 4, ignore_index=True)
    calls = _install_sources(monkeypatch, {_CODE: repeated}, {_CODE: _history()})

    ready, metadata = _load()

    assert calls["secondary"] == [(_CODE,)]
    assert len(ready[_CODE]) == 20
    assert metadata["quality_failures"][0]["rows"] == 5
    assert metadata["quality_failures"][0]["reasons"] == ["insufficient_rows"]
