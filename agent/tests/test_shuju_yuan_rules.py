"""Focused tests for shared A-share market rules and reference-data caching."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.ashare import shuju_yuan


@pytest.mark.parametrize(
    ("ts_code", "name", "expected"),
    [
        ("600000.SH", "*ST浦发", 0.10),
        ("000001.SZ", "ST平安", 0.10),
        ("300001.SZ", "*ST创业", 0.20),
        ("301001.SZ", "创业板", 0.20),
        ("688001.SH", "ST科创", 0.20),
        ("689001.SH", "科创板", 0.20),
        ("830001.BJ", "ST北交", 0.30),
        ("920001", "北交新代码", 0.30),
    ],
)
def test_limit_rate_uses_current_board_rules(ts_code: str, name: str, expected: float) -> None:
    assert shuju_yuan._limit_rate(ts_code, name) == expected


def test_price_limit_rule_can_represent_a_session_without_price_limit() -> None:
    rule = shuju_yuan._price_limit_rule("688001.SH", "新股", price_limit_exempt=True)

    assert rule.status == "no_limit"
    assert rule.limit_rate is None
    assert rule.effective_from == "2026-07-06"


def test_shared_code_normalization_does_not_treat_all_nine_prefixes_as_beijing() -> None:
    assert shuju_yuan._normalize_code("920001") == "920001.BJ"
    with pytest.raises(ValueError, match="A 股范围"):
        shuju_yuan._normalize_code("900901")
    with pytest.raises(ValueError, match="A 股范围"):
        shuju_yuan._normalize_code("400001")
    with pytest.raises(ValueError, match="后缀不一致"):
        shuju_yuan._normalize_code("920001.SH")


class _FakePro:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def stock_basic(self, **_: object) -> pd.DataFrame:
        self.calls += 1
        return self.frame.copy()


def _fresh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "symbol": "000001",
                "name": "平安银行",
                "area": "深圳",
                "industry": "银行",
                "market": "主板",
                "list_date": "19910403",
            }
        ]
    )


@pytest.fixture
def cache_path(monkeypatch: pytest.MonkeyPatch):
    path = Path(__file__).with_name(f".stock_basic_{uuid4().hex}.csv")
    monkeypatch.setattr(shuju_yuan, "STOCK_BASIC_CACHE", path)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_stock_basic_cache_uses_fresh_file_without_network(
    cache_path: Path,
) -> None:
    _fresh_frame().to_csv(cache_path, index=False)
    pro = _FakePro(pd.DataFrame())
    quality: dict[str, object] = {}

    result = shuju_yuan._load_or_fetch_stock_basic(pro, quality)

    assert pro.calls == 0
    assert result.iloc[0]["ts_code"] == "000001.SZ"
    assert quality["stock_basic"]["source"] == "cache"  # type: ignore[index]


def test_stock_basic_cache_refreshes_after_ttl(
    cache_path: Path,
) -> None:
    pd.DataFrame([{"ts_code": "600000.SH", "name": "旧名称"}]).to_csv(cache_path, index=False)
    old_time = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(cache_path, (old_time, old_time))
    pro = _FakePro(_fresh_frame())
    quality: dict[str, object] = {}

    result = shuju_yuan._load_or_fetch_stock_basic(pro, quality, cache_ttl=timedelta(hours=24))

    assert pro.calls == 1
    assert result.iloc[0]["ts_code"] == "000001.SZ"
    assert quality["stock_basic"]["source"] == "tushare"  # type: ignore[index]


def test_stock_basic_cache_supports_forced_refresh(
    cache_path: Path,
) -> None:
    pd.DataFrame([{"ts_code": "600000.SH", "name": "旧名称"}]).to_csv(cache_path, index=False)
    pro = _FakePro(_fresh_frame())

    result = shuju_yuan._load_or_fetch_stock_basic(pro, {}, force_refresh=True)

    assert pro.calls == 1
    assert result.iloc[0]["ts_code"] == "000001.SZ"
