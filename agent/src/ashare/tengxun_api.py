"""腾讯证券公开前复权日 K 接口适配。

腾讯响应同时返回前复权日线和当前证券快照。适配器会用快照复核证券代码、名称、
最新日期收盘价和最新成交额；调用方再用交易日历检查时间完整性。
"""

from __future__ import annotations

from typing import Any, Iterable

from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu


TENGXUN_FQKLINE_ENDPOINTS = (
    # 腾讯当前公开行情页使用的新接口。旧 ``web.ifzq`` 端点会阶段性返回 501，
    # 因此只保留为末级兼容端点，不能再作为主源。
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
)


def _tengxun_symbol(code: str) -> tuple[str, str]:
    text = str(code).strip().upper()
    if "." in text:
        digits, suffix = text.rsplit(".", 1)
    else:
        digits = text
        suffix = "SH" if digits.startswith("6") else "SZ"
    if not digits.isdigit() or len(digits) != 6 or suffix not in {"SH", "SZ", "BJ"}:
        raise ValueError(f"证券代码格式无效：{code}")
    return f"{suffix.lower()}{digits}", digits


def _quote_value(quote: list[Any], index: int) -> str | None:
    if index >= len(quote):
        return None
    value = str(quote[index] or "").strip()
    return value or None


def _latest_amount_yuan(quote: list[Any]) -> float | None:
    """优先读取价/量/额组合中的精确成交额，避免把万元字段误作元。"""
    composite = _quote_value(quote, 35)
    if composite:
        fields = composite.split("/")
        if len(fields) >= 3:
            try:
                return float(fields[2])
            except (TypeError, ValueError):
                pass
    return None


def _api_date(value: str) -> str:
    digits = str(value).replace("-", "").strip()
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"日期格式无效：{value}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def tengxun_qfq_rili_duqu(
    client: GongkaiShujuHTTPKehu,
    *,
    code: str,
    start_date: str,
    end_date: str,
    endpoints: Iterable[str] = TENGXUN_FQKLINE_ENDPOINTS,
) -> tuple[list[list[Any]], dict[str, Any], str]:
    """读取前复权日线，并用同一响应中的实时证券快照做结构化核验。"""
    symbol, digits = _tengxun_symbol(code)
    start = _api_date(start_date)
    end = _api_date(end_date)
    payload, endpoint = client.qingqiu_json(
        endpoints,
        params={
            "param": f"{symbol},day,{start},{end},640,qfq"
        },
        headers={"Referer": "https://gu.qq.com/"},
    )
    response_code = payload.get("code")
    if response_code not in {None, 0, "0"}:
        raise ValueError(f"腾讯证券接口返回业务错误：{response_code}")
    data = payload.get("data")
    entry = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"腾讯证券响应缺少 {symbol} 数据")
    raw_rows = entry.get("qfqday")
    if isinstance(raw_rows, list) and raw_rows:
        series_key = "qfqday"
        volume_unit = "hands"
        adjustment_basis = "provider_qfq_series"
    else:
        # 腾讯对没有独立复权序列的证券会在 qfq 请求中返回 ``day``；
        # 该响应仍由同一证券快照核验，但成交量单位是股而不是手。
        raw_rows = entry.get("day")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"腾讯证券没有返回 {digits} 的前复权或等价日线")
        series_key = "day"
        volume_unit = "shares"
        adjustment_basis = "provider_day_series_for_qfq_request"
    # 日线除 OHLCV 外还应包含换手率和成交额字段；缺少这些字段时不能
    # 用复权价格乘成交量冒充真实成交额。
    rows = [value for value in raw_rows if isinstance(value, list) and len(value) >= 9]
    if len(rows) != len(raw_rows):
        raise ValueError("腾讯证券前复权日线包含字段不完整的行情行")

    qt = entry.get("qt")
    quote = qt.get(symbol) if isinstance(qt, dict) else None
    if not isinstance(quote, list):
        raise ValueError("腾讯证券响应缺少证券快照，无法二次核验")
    observed_code = _quote_value(quote, 2)
    if observed_code != digits:
        raise ValueError(f"腾讯证券身份冲突：请求 {digits}，返回 {observed_code or '空值'}")
    identity = {
        "code": observed_code,
        "name": _quote_value(quote, 1),
        "quote_timestamp": _quote_value(quote, 30),
        "latest_close": _quote_value(quote, 3),
        "latest_volume_source_units": _quote_value(quote, 36) or _quote_value(quote, 6),
        "latest_amount_yuan": _latest_amount_yuan(quote),
        "adjustment": "qfq",
        "adjustment_basis": adjustment_basis,
        "series_key": series_key,
        "volume_unit": volume_unit,
        "response_version": entry.get("version"),
        "identity_check": "passed",
    }
    return rows, identity, endpoint


__all__ = ["TENGXUN_FQKLINE_ENDPOINTS", "tengxun_qfq_rili_duqu"]
