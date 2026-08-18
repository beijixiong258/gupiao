"""公开行情适配器的字段口径与事实核验测试。"""

from __future__ import annotations

import pytest

from src.ashare.shichang_shuju import _jiexi_tengxun_rili
from src.ashare.tengxun_api import (
    TENGXUN_FQKLINE_ENDPOINTS,
    tengxun_qfq_rili_duqu,
)


def _identity(*, amount: float = 1954663791.0) -> dict[str, object]:
    return {
        "quote_timestamp": "20260818161439",
        "latest_close": "39.06",
        "latest_volume_source_units": "500943",
        "latest_amount_yuan": amount,
        "volume_unit": "hands",
    }


def test_current_tencent_endpoint_precedes_obsolete_compatibility_endpoint() -> None:
    assert TENGXUN_FQKLINE_ENDPOINTS[0] == (
        "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    )


def test_tencent_daily_uses_exact_amount_field_and_cross_checks_latest_quote() -> None:
    rows = [
        ["2026-08-17", "36.00", "37.00", "38.00", "35.50", "100000", {}, "5.0", "37000.00", ""],
        ["2026-08-18", "37.64", "39.06", "42.00", "36.28", "500943", {}, "28.37", "195466.38", ""],
    ]

    data = _jiexi_tengxun_rili(rows, _identity())

    assert data.iloc[0]["amount_yuan"] == pytest.approx(370_000_000.0)
    assert data.iloc[-1]["amount_yuan"] == pytest.approx(1_954_663_791.0)
    assert data.iloc[-1]["volume"] == pytest.approx(50_094_300.0)


def test_tencent_daily_rejects_amount_conflict_with_same_response_quote() -> None:
    rows = [
        ["2026-08-18", "37.64", "39.06", "42.00", "36.28", "500943", {}, "28.37", "100000.00", ""],
    ]

    with pytest.raises(ValueError, match="最新成交额冲突"):
        _jiexi_tengxun_rili(rows, _identity())


def test_tencent_day_fallback_uses_share_unit_for_qfq_request() -> None:
    quote = [""] * 37
    quote[1] = "样本公司"
    quote[2] = "688981"
    quote[3] = "134.98"
    quote[6] = "57424091"
    quote[30] = "20260818161439"
    quote[35] = "134.98/57424091/7748492525"
    quote[36] = "57424091"
    payload = {
        "code": 0,
        "data": {
            "sh688981": {
                "day": [
                    ["2026-08-18", "136.00", "134.98", "137.48", "132.47", "57424091", {}, "2.87", "774849.25"]
                ],
                "qt": {"sh688981": quote},
                "version": "18",
            }
        },
    }

    class _Client:
        def qingqiu_json(self, *_args, **_kwargs):
            return payload, TENGXUN_FQKLINE_ENDPOINTS[0]

    rows, identity, _ = tengxun_qfq_rili_duqu(
        _Client(),
        code="688981.SH",
        start_date="20250624",
        end_date="20260818",
    )
    data = _jiexi_tengxun_rili(rows, identity)

    assert identity["series_key"] == "day"
    assert identity["volume_unit"] == "shares"
    assert data.iloc[-1]["volume"] == pytest.approx(57_424_091.0)
    assert data.iloc[-1]["amount_yuan"] == pytest.approx(7_748_492_525.0)
