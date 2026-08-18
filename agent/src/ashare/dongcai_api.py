"""东方财富公开接口的共享传输能力。

这里只处理公开接口分页、证券标识和响应结构，不在这一层解释选股业务规则。
"""

from __future__ import annotations

from typing import Any, Iterable

from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu


DONGCAI_CLIST_ENDPOINTS = (
    "https://push2delay.eastmoney.com/api/qt/clist/get",
    "https://push2.eastmoney.com/api/qt/clist/get",
    "https://17.push2.eastmoney.com/api/qt/clist/get",
    "https://79.push2.eastmoney.com/api/qt/clist/get",
    "https://82.push2.eastmoney.com/api/qt/clist/get",
)
DONGCAI_PUBLIC_TOKEN = "bd1d9ddb04089700cf9c27f6f7426281"
# 与当前 AKShare 官方适配器使用的公开行情令牌保持一致；接口能力仍由响应结构校验。
DONGCAI_KLINE_TOKEN = "7eea3edcaed734bea9cbfc24409ed989"
DONGCAI_KLINE_ENDPOINTS = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://7.push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://33.push2his.eastmoney.com/api/qt/stock/kline/get",
)


def _diff_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("东方财富响应缺少 data")
    raw_rows = data.get("diff") or []
    if isinstance(raw_rows, dict):
        rows = [value for value in raw_rows.values() if isinstance(value, dict)]
    elif isinstance(raw_rows, list):
        rows = [value for value in raw_rows if isinstance(value, dict)]
    else:
        raise ValueError("东方财富响应中的 diff 类型无效")
    try:
        total = int(data.get("total") or len(rows))
    except (TypeError, ValueError) as exc:
        raise ValueError("东方财富响应中的 total 无效") from exc
    return rows, max(total, len(rows))


def dongcai_fenye_duqu(
    client: GongkaiShujuHTTPKehu,
    *,
    base_params: dict[str, Any],
    endpoints: Iterable[str] = DONGCAI_CLIST_ENDPOINTS,
    referer: str = "https://quote.eastmoney.com/",
    maximum_pages: int = 100,
) -> tuple[list[dict[str, Any]], str, int]:
    """完整分页，首个可用子域确定后优先复用，防止静默截断为 100 条。"""
    endpoint_candidates = tuple(endpoints)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    preferred_endpoint: str | None = None
    expected_total: int | None = None
    for page in range(1, maximum_pages + 1):
        page_endpoints = (
            (preferred_endpoint, *endpoint_candidates)
            if preferred_endpoint
            else endpoint_candidates
        )
        payload, used_endpoint = client.qingqiu_json(
            page_endpoints,
            params={**base_params, "pn": page, "pz": 100},
            headers={"Referer": referer},
        )
        preferred_endpoint = used_endpoint
        page_rows, page_total = _diff_rows(payload)
        expected_total = page_total if expected_total is None else max(expected_total, page_total)
        added = 0
        for item in page_rows:
            identity = (str(item.get("f12") or ""), str(item.get("f14") or ""))
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(item)
            added += 1
        if len(rows) >= expected_total:
            break
        if not page_rows or added == 0:
            break
    if expected_total is None or not rows:
        raise ValueError("东方财富列表为空")
    if len(rows) < expected_total:
        raise ValueError(f"东方财富分页不完整：应有 {expected_total} 条，实际取得 {len(rows)} 条")
    return rows, str(preferred_endpoint), expected_total


def _dongcai_secid(code: str) -> tuple[str, str]:
    """把已规范化或纯数字证券代码转换为东方财富证券标识。"""
    text = str(code).strip().upper()
    if "." in text:
        digits, suffix = text.rsplit(".", 1)
    else:
        digits = text
        suffix = "SH" if digits.startswith("6") else "SZ"
    if not digits.isdigit() or len(digits) != 6:
        raise ValueError(f"证券代码格式无效：{code}")
    market = "1" if suffix == "SH" else "0"
    return f"{market}.{digits}", digits


def dongcai_rili_kxian_duqu(
    client: GongkaiShujuHTTPKehu,
    *,
    code: str,
    start_date: str,
    end_date: str,
    qfq: bool = True,
    endpoints: Iterable[str] = DONGCAI_KLINE_ENDPOINTS,
) -> tuple[list[str], dict[str, Any], str]:
    """读取一只股票或指数的日 K 原始行，并校验响应证券身份。

    ``qfq`` 只控制数据源的复权参数；调用方仍需按所需字段和交易日历验证完整性。
    """
    secid, digits = _dongcai_secid(code)
    payload, endpoint = client.qingqiu_json(
        endpoints,
        params={
            "secid": secid,
            "ut": DONGCAI_KLINE_TOKEN,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "klt": 101,
            "fqt": 1 if qfq else 0,
            "beg": str(start_date).replace("-", ""),
            "end": str(end_date).replace("-", ""),
        },
        headers={"Referer": "https://quote.eastmoney.com/"},
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("东方财富日 K 响应缺少 data")
    observed_code = str(data.get("code") or "").strip()
    if observed_code != digits:
        raise ValueError(f"东方财富日 K 证券身份冲突：请求 {digits}，返回 {observed_code or '空值'}")
    raw_rows = data.get("klines")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"东方财富日 K 未返回 {digits} 的行情")
    rows = [str(value) for value in raw_rows if isinstance(value, str) and value.strip()]
    if len(rows) != len(raw_rows):
        raise ValueError("东方财富日 K 包含非文本行情行")
    identity = {
        "code": observed_code,
        "name": str(data.get("name") or "").strip(),
        "market": data.get("market"),
        "decimal": data.get("decimal"),
        "adjustment": "qfq" if qfq else "raw",
    }
    return rows, identity, endpoint


__all__ = [
    "DONGCAI_CLIST_ENDPOINTS",
    "DONGCAI_KLINE_ENDPOINTS",
    "DONGCAI_KLINE_TOKEN",
    "DONGCAI_PUBLIC_TOKEN",
    "dongcai_fenye_duqu",
    "dongcai_rili_kxian_duqu",
]
