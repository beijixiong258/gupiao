"""沪深京交易所的当前风险警示和退市整理名册。"""

from __future__ import annotations

import json
import re
import ssl
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any

import pandas as pd

from ashare.lishi_jiaoyi import RISK_STATUS_FIELDS, shi_fengxian_jingshi
from ashare.wangluo_kehu import GongkaiShujuHTTPKehu


_BEIJING = timezone(timedelta(hours=8))
_SSE_PAGES = ("https://www.sse.com.cn/disclosure/listedinfo/riskplate/",
              "https://star.sse.com.cn/disclosure/listedinfo/starriskplate/")
_SZSE_PAGE = "https://investor.szse.cn/disclosure/listed/warn/index.html"
_BSE_PAGE = "https://www.bse.cn/disclosure/risk_warning_board.html"
_LIST_PAGES = {"SH": _SSE_PAGES, "SZ": (_SZSE_PAGE,), "BJ": (_BSE_PAGE,)}
_DATE_BASIS = {"SH": "exchange_last_trading_date", "SZ": "exchange_report_date", "BJ": "current_exchange_directory"}


def _members(rows: list[dict[str, Any]], code_field: str, name_field: str, exchange: str) -> dict[str, str]:
    members = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("风险名单证券行的格式无效")
        digits, label = row.get(code_field), row.get(name_field)
        if not isinstance(digits, str) or not re.fullmatch(r"[0-9]{6}", digits) or not isinstance(label, str):
            raise ValueError("风险名单的证券代码或简称无效")
        code = f"{digits}.{exchange}"
        name = unescape(re.sub(r"<[^>]*>", "", label)).strip()
        if not name or code in members:
            raise ValueError("风险名单出现重复证券或空简称")
        members[code] = name
    return members


def _sse_members(client: GongkaiShujuHTTPKehu, metric: str) -> dict[str, str]:
    members = {}
    for segment, page in zip(("main", "star"), _SSE_PAGES):
        if segment == "main":
            params = {"sqlId": "PL_SSGSXX_FXJSBGPLB", "productType": "0",
                      "domesticIndicator": "S" if metric == "is_st" else "P"}
            code_field, name_field = "INSTRUMENT_ID", "INSTRUMENT_SHORT"
        else:
            params = {"sqlId": "SSE_PL_SSGSXX_KCBFXTS_L", "type": "0" if metric == "is_st" else "1"}
            code_field, name_field = "secCode", "secNameCn"
        payload, _ = client.qingqiu_json(("https://query.sse.com.cn/commonSoaQuery.do",), params=params,
                                         headers={"Referer": page, "Cache-Control": "no-cache"})
        rows, meta = payload.get("result"), payload.get("pageHelp") or {}
        if (payload.get("sqlId") != params["sqlId"] or payload.get("actionErrors") or payload.get("fieldErrors")
                or not isinstance(rows, list) or meta.get("total") != len(rows) or meta.get("pageCount") not in (0, 1)):
            raise ValueError("上交所风险名单的返回状态或覆盖数量未核验")
        part = _members(rows, code_field, name_field, "SH")
        if members.keys() & part.keys():
            raise ValueError("上交所不同交易板的风险名单证券重复")
        members.update(part)
    return members


def _szse_report(client: GongkaiShujuHTTPKehu, tab: str, page: int) -> dict[str, dict[str, Any]]:
    endpoints = tuple(host + "/api/report/ShowReport/data" for host in ("https://investor.szse.cn", "https://www.szse.cn"))
    raw, _ = client.qingqiu_wenben(endpoints, params={
        "SHOWTYPE": "JSON", "CATALOGID": "fxjsb", "TABKEY": tab, "PAGENO": page,
    }, headers={"Referer": _SZSE_PAGE, "Cache-Control": "no-cache"})
    payload = json.loads(raw)
    reports = {}
    if not isinstance(payload, list):
        raise ValueError("深交所风险名单响应格式无效")
    for group in payload:
        if not isinstance(group, dict) or group.get("error") or not isinstance(group.get("data"), list):
            raise ValueError("深交所风险名单返回错误或缺少列表")
        meta = group.get("metadata") or {}
        group_tab = meta.get("tabkey")
        if meta.get("catalogid") != "fxjsb" or group_tab in reports or meta.get("pageno") != page:
            raise ValueError("深交所风险名单标签或页号不一致")
        reports[group_tab] = group
    if reports.keys() != {"tab1", "tab2"}:
        raise ValueError("深交所风险名单缺少所需报告")
    return reports


def _szse_members(client: GongkaiShujuHTTPKehu, first: dict[str, Any], metric: str, expected_date: str) -> dict[str, str]:
    tab = "tab1" if metric == "is_st" else "tab2"
    if tab == "tab2":
        first = _szse_report(client, tab, 1)
    meta = first[tab]["metadata"]
    reported_date = str(meta.get("subname") or "").strip()
    pages, total = meta.get("pagecount"), meta.get("recordcount")
    if (type(pages) is not int or type(total) is not int or not 0 <= pages <= 20 or total < 0
            or (pages > 0) != (total > 0) or (reported_date and reported_date != expected_date)
            or (total > 0 and not reported_date)):
        raise ValueError("深交所风险名单分页或日期未核验")
    rows = []
    for page in range(1, max(1, pages) + 1):
        group = first[tab] if page == 1 else _szse_report(client, tab, page)[tab]
        current = group["metadata"]
        if any(current.get(key) != meta.get(key) for key in ("pagecount", "recordcount", "subname")):
            raise ValueError("读取深交所风险名单期间来源数据发生变化")
        rows.extend(group["data"])
    if len(rows) != total:
        raise ValueError("深交所风险名单分页不完整")
    return _members(rows, "zqdm", "gsjc", "SZ")


def _bse_members(client: GongkaiShujuHTTPKehu, metric: str) -> dict[str, str]:
    rows = []
    count = pages = None
    for page in range(20):
        endpoint = "https://www.bse.cn/nqxxController/getRiskWarningStock.do"
        raw, _ = client.qingqiu_wenben((endpoint,), method="POST", headers={"Referer": _BSE_PAGE, "Cache-Control": "no-cache"},
                                      params={"page": page, "pageSize": 20, "type": 0 if metric == "is_st" else 1,
                                              "callback": "riskReport"})
        response = re.fullmatch(r"\s*riskReport\(([\s\S]+)\)\s*;?\s*", raw)
        payload = json.loads(response[1]) if response else None
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("北交所风险名单响应格式无效")
        group = payload[0]
        total, page_count, content = group.get("totalElements"), group.get("totalPages"), group.get("content")
        if (type(total) is not int or total < 0 or type(page_count) is not int or not 0 <= page_count <= 20
                or (page_count > 0) != (total > 0) or not isinstance(content, list)
                or group.get("number") != page or group.get("numberOfElements") != len(content)):
            raise ValueError("北交所风险名单分页信息无效")
        if page == 0:
            count, pages = total, page_count
        if (total, page_count) != (count, pages):
            raise ValueError("读取北交所风险名单期间来源数据发生变化")
        rows.extend(content)
        if page + 1 >= pages:
            break
    if len(rows) != count:
        raise ValueError("北交所风险名单分页不完整")
    return _members(rows, "xxzqdm", "xxzqjc", "BJ")


def _market_members(market: str, metrics: Iterable[str], day: str) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    members = {}
    records = []
    with GongkaiShujuHTTPKehu(f"{market.lower()}_exchange_risk_board", ssl_context=ssl.create_default_context()) as client:
        try:
            if market == "SH":
                text, _ = client.qingqiu_wenben(("https://www.sse.com.cn/js/common/systemDate_global.js",))
                date = re.search(r'\blastTradeDate_global\s*=\s*"(\d{4}-\d{2}-\d{2})"', text)
                if date is None or date[1] != day:
                    raise ValueError("上交所最近交易日与本次分析日不一致")
            first_report = {}
            if market == "SZ":
                first_report = _szse_report(client, "tab1", 1)
                report_date = str(first_report["tab1"]["metadata"].get("subname") or "").strip()
                if report_date != day:
                    raise ValueError("深交所风险名单的报告日期与本次分析日不一致")
        except (OSError, RuntimeError, ValueError) as exc:
            return members, [{"exchange": market, "metric": metric, "status": "unavailable",
                              "sources": _LIST_PAGES[market], "as_of": day,
                              "reason": str(exc).split("?", 1)[0][:240]} for metric in sorted(metrics)]
        for metric in sorted(metrics):
            audit = {"exchange": market, "metric": metric, "sources": _LIST_PAGES[market], "as_of": day,
                     "date_basis": _DATE_BASIS[market]}
            try:
                if market == "SH":
                    observed = _sse_members(client, metric)
                elif market == "SZ":
                    observed = _szse_members(client, first_report, metric, day)
                else:
                    observed = _bse_members(client, metric)
                members[metric] = observed
                audit.update(status="ok", reported_members=len(observed))
            except (OSError, RuntimeError, ValueError) as exc:
                audit.update(status="unavailable", reason=str(exc).split("?", 1)[0][:240])
            records.append(audit)
    return members, records


def huoqu_fengxian_jingshi(
    profiles: Mapping[str, str], metrics: Iterable[str], *, target: pd.Timestamp,
    current_completed_date: pd.Timestamp, allow_current_list: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """完整名单支持否定核验，同日身份或名册无法确认时保留未知。"""
    requested = sorted(set(metrics).intersection(RISK_STATUS_FIELDS))
    day = target.strftime("%Y-%m-%d")
    if not requested:
        return {}, {"status": "not_requested"}
    if not profiles:
        return {}, {"status": "unavailable", "as_of": day, "reason": "缺少可核验的候选证券身份", "persistence": "none"}
    if target != current_completed_date or not allow_current_list:
        return {}, {"status": "unavailable", "as_of": day, "reason": "当前名册的适用时点与完整分析日不一致",
                    "persistence": "none"}
    membership = {}
    sources = []
    markets = sorted({code.rsplit(".", 1)[-1] for code in profiles} & _LIST_PAGES.keys())
    with ThreadPoolExecutor(max_workers=len(_LIST_PAGES), thread_name_prefix="exchange-risk") as executor:
        jobs = {market: executor.submit(_market_members, market, requested, day) for market in markets}
        for market, job in jobs.items():
            membership[market], records = job.result()
            sources.extend(records)
    fetched_at = datetime.now(_BEIJING).isoformat(timespec="seconds")
    observations = {}
    for code, name in profiles.items():
        market = code.rsplit(".", 1)[-1]
        values: dict[str, Any] = {"trade_date": day, "fetched_at": fetched_at}
        for metric in requested:
            members = membership.get(market, {}).get(metric)
            if members is None:
                continue
            listed_name = members.get(code)
            source_key = "st_source" if metric == "is_st" else "delisting_status_source"
            values.update({source_key: list(_LIST_PAGES[market]), f"{metric}_date_basis": _DATE_BASIS[market]})
            if listed_name is None:
                # 只按完整正式名单判断。若当前身份显示相反标记，保留冲突。
                suspect = shi_fengxian_jingshi(name) if metric == "is_st" else name.startswith("退市") or name.endswith("退")
                values[metric] = None if suspect else False
                if suspect:
                    values[f"{metric}_conflict"] = {"candidate_name": name, "in_directory": False}
            elif listed_name == name:
                values[metric] = True
            else:
                values[f"{metric}_conflict"] = {"candidate_name": name, "source_name": listed_name}
        observations[code] = values
    count = sum(value.get(metric) is not None for value in observations.values() for metric in requested)
    expected = len(profiles) * len(requested)
    return observations, {"status": "ok" if count == expected else "partial" if count else "unavailable",
                         "as_of": day, "fetched_at": fetched_at, "verified_checks": count,
                         "requested_checks": expected, "sources": sources, "persistence": "none"}
