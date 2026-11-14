"""公开历史交易证据的在线适配，按证券和请求区间取数。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from html import unescape
from io import BytesIO
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import pandas as pd

from ashare.baostock_api import bao_gupiao_shuju
from ashare.dongcai_api import dongcai_rili_kxian_duqu, dongcai_rizhong_guzhi_duqu
from ashare.lishi_jiaoyi import (
    RAW_FIELDS,
    jiaoyi_riqi,
    lishi_shuzhi,
    shi_fengxian_jingshi,
    zhengli_jiaoyi_zhengju,
    zhengli_lishi_rili,
    zhankai_shengxiao_yinzi,
    zuihou_shiji_chengjiao,
)
from ashare.shuju_yuan import biaozhunhua_gupiao_daima
from ashare.tengxun_api import tengxun_qfq_rili_duqu
from ashare.wangluo_kehu import GongkaiShujuHTTPKehu, WangluoQingqiuYichang


_SSE_STOP = "https://query.sse.com.cn/commonSoaQuery.do"
_SSE_STOP_PAGE = "https://www.sse.com.cn/disclosure/dealinstruc/suspension/stock/"
_SSE_DELIST_PAGE = "https://www.sse.com.cn/assortment/stock/list/delisting/"
_EASTMONEY_DATA = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_EASTMONEY_PROFILE = "https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/CompanySurveyAjax"
_BSE_CODES = "https://www.bse.cn/service/code_mapping.html"
_MARKET_BOARD = {"主板": "main_board", "中小板": "main_board", "创业板": "chinext", "科创板": "star", "北交所": "bse"}
_TIMEZONE = timezone(timedelta(hours=8))
LIQUIDITY_HISTORY_SESSIONS = 60
_TURNOVER_FIELDS = ["trade_date", "reported_turnover_rate_pct", "turnover_source", "turnover_volume_shares"]


def _day(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _bao_daily(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    if not rows or any(row["code"] != symbol or row["tradestatus"] not in {"0", "1"} for row in rows):
        raise ValueError("BaoStock 日线为空或证券、状态字段冲突")
    frame = pd.DataFrame(rows).rename(columns={
        "date": "trade_date", "open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close",
        "preclose": "limit_reference_price", "amount": "amount_yuan",
    })
    frame["is_suspended"] = frame["tradestatus"].eq("0")
    frame["is_st"] = frame["isST"].map({"0": False, "1": True})
    frame["suspension_source"] = "baostock_daily"
    frame["st_source"] = frame["is_st"].map({True: "baostock_daily", False: "baostock_daily"})
    frame["raw_price_source"] = "baostock_daily"
    frame["limit_reference_source"] = "baostock_daily.preclose"
    for field in (*RAW_FIELDS, "limit_reference_price", "volume", "amount_yuan"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame.drop(columns=["code", "tradestatus", "isST"])


def _tushare_daily(pro: Any, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pro.daily(ts_code=code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"}
    if frame is None or frame.empty or not required.issubset(frame) or not frame["ts_code"].eq(code).all():
        raise ValueError("Tushare 原始日线缺失、身份冲突或字段不足")
    data = frame.rename(columns={
        "open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close",
        "pre_close": "limit_reference_price",
    })[["trade_date", *RAW_FIELDS, "limit_reference_price", "vol", "amount"]].copy()
    data["volume"] = pd.to_numeric(data.pop("vol"), errors="coerce").mul(100).round(6)
    data["amount_yuan"] = pd.to_numeric(data.pop("amount"), errors="coerce") * 1_000
    data["raw_price_source"] = "tushare_daily"
    data["limit_reference_source"] = "tushare_daily.pre_close"
    return data


def _eastmoney_daily(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    with GongkaiShujuHTTPKehu("eastmoney_raw_daily") as client:
        rows, _, _ = dongcai_rili_kxian_duqu(client, code=code, start_date=_day(start), end_date=_day(end), qfq=False)
    frame = pd.DataFrame([row.split(",")[:7] for row in rows], columns=[
        "trade_date", "raw_open", "raw_close", "raw_high", "raw_low", "volume", "amount_yuan",
    ])
    for field in (*RAW_FIELDS, "amount_yuan"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * 100
    # 该接口的涨跌额在除息日可能仍使用昨日实收，不能据此反推限价基准。
    return frame.assign(raw_price_source="eastmoney_raw_daily")


def _public_turnover(code: str, start: pd.Timestamp, end: pd.Timestamp, *, tencent: bool) -> pd.DataFrame:
    source = "tencent_daily_turnover" if tencent else "eastmoney_daily_turnover"
    with GongkaiShujuHTTPKehu(source) as client:
        if tencent:
            rows, identity, _ = tengxun_qfq_rili_duqu(client, code=code, start_date=_day(start), end_date=_day(end))
            multiplier = 100 if identity["volume_unit"] == "hands" else 1
            values = [(row[0], row[7], row[5]) for row in rows]
        else:
            rows, _, _ = dongcai_rili_kxian_duqu(client, code=code, start_date=_day(start), end_date=_day(end), qfq=False)
            multiplier = 100
            values = [(row[0], row[10], row[5]) for record in rows if len(row := record.split(",")) >= 11]
            if len(values) != len(rows):
                raise ValueError("东方财富日线缺少换手率字段位置")
    frame = pd.DataFrame(values, columns=["trade_date", "reported_turnover_rate_pct", "turnover_volume_shares"])
    frame["turnover_volume_shares"] = pd.to_numeric(frame["turnover_volume_shares"], errors="coerce") * multiplier
    frame["turnover_source"] = source
    return zhengli_lishi_rili(frame).reset_index()[_TURNOVER_FIELDS]


def _public_historical_valuation(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    with GongkaiShujuHTTPKehu("eastmoney_historical_valuation") as client:
        rows = dongcai_rizhong_guzhi_duqu(
            client, filter_text=f'(SECUCODE="{code}")(TRADE_DATE>=\'{_day(start)}\')(TRADE_DATE<=\'{_day(end)}\')',
            sort_column="TRADE_DATE", page_size=100,
        )
    frame = pd.DataFrame(rows)
    required = {"SECUCODE", "TRADE_DATE", "CLOSE_PRICE", "NOTLIMITED_MARKETCAP_A"}
    if not required.issubset(frame) or not frame["SECUCODE"].eq(code).all():
        raise ValueError("东方财富历史估值字段或证券身份未核验")
    values = zhengli_lishi_rili(frame.rename(columns={
        "TRADE_DATE": "trade_date", "CLOSE_PRICE": "reported_valuation_close",
        "NOTLIMITED_MARKETCAP_A": "reported_circulating_market_value_yuan",
    })[["trade_date", "reported_valuation_close", "reported_circulating_market_value_yuan"]]).reset_index()
    if not values["trade_date"].between(start, end).all():
        raise ValueError("东方财富历史估值日期超出请求区间")
    return values


def _merge_liquidity(
    daily: pd.DataFrame, turnover_sources: Iterable[pd.DataFrame], valuation: pd.DataFrame, sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """按开市日绑定原值。换手率和市值只有通过相应的成交核验才进入指标。"""
    result = daily.loc[daily["trade_date"].isin(sessions)].copy()
    turnover = pd.DataFrame(index=pd.Index(result["trade_date"], name="trade_date"), columns=_TURNOVER_FIELDS[1:])
    traded = result.set_index("trade_date")
    traded_volume = traded["volume"].where(traded["raw_bar_valid"])
    selected_verified = pd.Series(False, index=turnover.index)
    for observations in turnover_sources:
        if observations.empty:
            continue
        observed = observations.set_index("trade_date")[_TURNOVER_FIELDS[1:]].reindex(turnover.index)
        rates = pd.to_numeric(observed["reported_turnover_rate_pct"], errors="coerce")
        volumes = pd.to_numeric(observed["turnover_volume_shares"], errors="coerce")
        accepted = rates.between(0, 100) & volumes.gt(0) & (volumes - traded_volume).abs().le(50)
        supply = (accepted & ~selected_verified) | (rates.notna() & turnover["reported_turnover_rate_pct"].isna())
        turnover.loc[supply] = observed.loc[supply].astype(object)
        selected_verified |= accepted
    result = result.merge(turnover.reset_index(), on="trade_date", how="left", validate="one_to_one")
    if not valuation.empty:
        result = result.merge(valuation, on="trade_date", how="left", validate="one_to_one")
    for field in ("reported_turnover_rate_pct", "turnover_volume_shares", "reported_valuation_close",
                  "reported_circulating_market_value_yuan"):
        values = pd.to_numeric(result.get(field, pd.Series(index=result.index, dtype=float)), errors="coerce")
        result[field] = values.replace([float("inf"), float("-inf")], float("nan"))
    if "turnover_source" not in result:
        result["turnover_source"] = None
    valid = result["raw_bar_valid"]
    # 腾讯及东方财富部分日 K 以整手报告，独立日线的股数可有手数舍入差。
    source_volume = result["turnover_volume_shares"]
    comparable_volume = (source_volume - result["volume"]).abs().le(50) & source_volume.gt(0)
    rates = result["reported_turnover_rate_pct"]
    result["turnover_rate"] = rates.where(valid & comparable_volume & rates.between(0, 100))
    market_value = result["reported_circulating_market_value_yuan"]
    reported_close = result["reported_valuation_close"]
    close_matches = (reported_close - result["raw_close"]).abs().le(0.005000001) & reported_close.gt(0)
    result["circulating_market_value_yuan"] = market_value.where(valid & close_matches & market_value.gt(0))
    result["valuation_source"] = "eastmoney_daily_valuation" if not valuation.empty else None
    result["turnover_status"] = "unavailable"
    result["valuation_status"] = "unavailable"
    for field, raw, eligible, comparable in (
        ("turnover_status", rates, result["turnover_rate"], comparable_volume),
        ("valuation_status", market_value, result["circulating_market_value_yuan"], close_matches),
    ):
        result.loc[raw.notna() & ~valid, field] = "no_verified_trade"
        result.loc[raw.notna() & valid & ~comparable, field] = "conflict"
        result.loc[raw.notna() & valid & comparable & eligible.isna(), field] = "invalid_reported_value"
        result.loc[eligible.notna(), field] = "verified"
    result.loc[market_value.isna(), "valuation_source"] = None
    cols = ["trade_date", "turnover_rate", "turnover_status", "turnover_source", "turnover_volume_shares",
            "reported_turnover_rate_pct", "circulating_market_value_yuan", "reported_circulating_market_value_yuan",
            "reported_valuation_close", "valuation_status", "valuation_source"]
    return daily.merge(result[cols], on="trade_date", how="left", validate="one_to_one")


def _bao_factors(rows: list[dict[str, Any]], symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
    if any(row["code"] != symbol for row in rows):
        raise ValueError("BaoStock 复权因子与请求证券不一致")
    events = pd.DataFrame(rows).rename(columns={"dividOperateDate": "trade_date", "backAdjustFactor": "adj_factor"})
    # 接口从市场成立时开始请求，上市首日至首个权息事件前因子为 1。
    if not rows:
        return pd.Series(1.0, index=dates)
    events = pd.concat([pd.DataFrame([{"trade_date": "1990-12-18", "adj_factor": 1.0}]), events], ignore_index=True)
    return zhankai_shengxiao_yinzi(events, dates)


def _tushare_factors(pro: Any, code: str, dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    frame = pro.adj_factor(ts_code=code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    if frame is None or frame.empty or not {"ts_code", "trade_date", "adj_factor"}.issubset(frame):
        raise ValueError("Tushare 逐日复权因子缺失")
    if not frame["ts_code"].eq(code).all():
        raise ValueError("Tushare 复权因子与请求证券不一致")
    values = pd.to_numeric(zhengli_lishi_rili(frame)["adj_factor"], errors="coerce").reindex(dates)
    return values.where(values.gt(0) & values.lt(float("inf")))


def _tushare_names(pro: Any, code: str, end: pd.Timestamp) -> list[dict[str, Any]]:
    table = pro.namechange(ts_code=code, fields="ts_code,name,start_date,end_date,ann_date,change_reason")
    if table is None or table.empty:
        return []
    if not table["ts_code"].eq(code).all():
        raise ValueError("Tushare 曾用名与请求证券不一致")
    names = []
    for row in table.drop_duplicates().to_dict("records"):
        start, ann = (jiaoyi_riqi(row[key]) for key in ("start_date", "ann_date"))
        if pd.notna(start) and pd.notna(ann) and start <= end and ann <= end and pd.notna(row["name"]) and str(row["name"]).strip():
            names.append({"name": str(row["name"]), "start_date": start, "end_date": jiaoyi_riqi(row["end_date"]),
                          "announcement_date": ann, "reason": row.get("change_reason"), "source": "tushare_namechange"})
    return sorted(names, key=lambda row: row["start_date"])


def _sse_suspensions(code: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    events = []
    digits = code.split(".")[0]
    with GongkaiShujuHTTPKehu("sse_historical_suspension") as client:
        for page in range(1, 21):
            data, _ = client.qingqiu_json(
                (_SSE_STOP,), headers={"Referer": _SSE_STOP_PAGE},
                params={"sqlId": "GW_PL_JYTS_TFPXX", "productCode": digits, "keyWords": "",
                        "startStopDate": start.strftime("%Y%m%d"), "endStopDate": end.strftime("%Y%m%d"),
                        "isPagination": "true", "pageHelp.pageSize": 25, "pageHelp.pageNo": page,
                        "pageHelp.beginPage": page, "pageHelp.endPage": page, "pageHelp.cacheSize": 1},
            )
            meta, rows = data.get("pageHelp"), data.get("result")
            if not isinstance(meta, dict) or not isinstance(rows, list) or int(meta["pageNo"]) != page:
                raise ValueError("上交所停复牌分页未核验")
            for item in rows:
                if item.get("productCode") != digits:
                    raise ValueError("上交所停复牌记录与请求证券不一致")
                first, last = (jiaoyi_riqi(item.get(key)) for key in ("startStopDate", "endStopDate"))
                if pd.isna(first) or first > end or (pd.notna(last) and last < start):
                    continue
                period = item.get("stopTime")
                full = period == "WH" or (item.get("type") == "LXTP" and not period)
                kind = "full_day" if full else "intraday"
                span = {"930": ("09:30:00", "10:30:00"), "AM": ("09:30:00", "11:30:00"),
                        "PM": ("13:00:00", "15:00:00")}.get(period)
                if not full and (span is None or first != last):
                    continue
                # 未公布结束日的连续停牌仅确认起始日，后续日期由逐日证据确认。
                if pd.isna(last):
                    last = first
                events.append({"kind": kind, "start_date": first, "end_date": last, "source": _SSE_STOP_PAGE,
                               "reason": item.get("stopReason"), "end_reason": item.get("endStopReason"),
                               **({"start_time": span[0], "resume_time": span[1], "through_close": period == "PM"} if span else {})})
            if page >= int(meta["pageCount"]):
                return events
    raise ValueError("上交所停复牌记录超过请求分页上限")


def huoqu_bse_daima_duiying(code: str) -> dict[str, Any]:
    with GongkaiShujuHTTPKehu("bse_security_code_mapping") as client:
        body, _ = client.qingqiu_wenben((_BSE_CODES,), headers={"Referer": "https://www.bse.cn/service/"})
    digits = code.split(".")[0]
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", body, flags=re.S | re.I):
        cells = [unescape(re.sub(r"<[^>]+>", "", cell)).strip()
                 for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", row, flags=re.S | re.I)]
        if len(cells) == 5 and digits in cells[3:5] and all(re.fullmatch(r"\d{6}", value) for value in cells[3:5]):
            listed = jiaoyi_riqi(cells[2])
            # 北交所说明精选层平移股票在表中可能列原精选层挂牌日。
            return {"ts_code": cells[4] + ".BJ", "old_ts_code": cells[3] + ".BJ", "board": "bse", "snapshot_name": cells[1],
                    "list_date": max(listed, pd.Timestamp("2021-11-15")) if pd.notna(listed) else None,
                    "code_change_date": pd.Timestamp("2025-10-09"), "identity_source": _BSE_CODES}
    return {}


def huoqu_sse_tuishi_shenfen(code: str) -> dict[str, Any]:
    """按原 A 股代码核对上交所已终止上市目录。"""
    digits, exchange = biaozhunhua_gupiao_daima(code).split(".")
    if exchange != "SH":
        return {}
    with GongkaiShujuHTTPKehu("sse_delisted_stock_directory") as client:
        data, _ = client.qingqiu_json(("https://query.sse.com.cn/commonQuery.do",), headers={"Referer": _SSE_DELIST_PAGE},
                                     params={"sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_ZZGP_L", "STOCK_CODE": digits,
                                             "CSRC_CODE": "", "REG_PROVINCE": "", "STOCK_TYPE": "1,8",
                                             "COMPANY_STATUS": "3", "type": "inParams", "isPagination": "true",
                                             "pageHelp.cacheSize": 1, "pageHelp.beginPage": 1,
                                             "pageHelp.pageSize": 25, "pageHelp.pageNo": 1})
    rows = data.get("result")
    meta = data.get("pageHelp") or {}
    if not isinstance(rows, list) or int(meta.get("total", -1)) != len(rows):
        raise ValueError("上交所终止上市记录范围未核验")
    if not rows:
        return {}
    if len(rows) != 1 or rows[0].get("A_STOCK_CODE") != digits or not rows[0].get("COMPANY_ABBR"):
        raise ValueError("上交所终止上市记录身份未核验")
    row = rows[0]
    listed, delisted = (jiaoyi_riqi(row.get(key)) for key in ("LIST_DATE", "DELIST_DATE"))
    if pd.isna(listed) or pd.isna(delisted) or listed >= delisted:
        raise ValueError("上交所终止上市记录日期无效")
    return {"ts_code": code, "name": row["COMPANY_ABBR"], "list_date": listed, "delist_date": delisted,
            "market": {"1": "主板", "8": "科创板"}.get(row["LIST_BOARD"]),
            "listing_status": "delisted", "identity_source": _SSE_DELIST_PAGE}


def _eastmoney_profile(code: str) -> dict[str, Any]:
    digits, exchange = code.split(".")
    with GongkaiShujuHTTPKehu("eastmoney_stock_profile") as client:
        data, _ = client.qingqiu_json((_EASTMONEY_PROFILE,), params={"code": exchange + digits},
                                     headers={"Referer": "https://emweb.securities.eastmoney.com/"})
    basic = data.get("jbzl") or {}
    offering = data.get("fxxg") or {}
    listed = jiaoyi_riqi(offering.get("ssrq"))
    if (data.get("SecuCode") != code or data.get("CodeType") != "ABStock"
            or data.get("Market") != exchange or pd.isna(listed)):
        raise ValueError("东方财富公司资料未核验 A 股代码及上市日期")
    category = basic.get("zqlb")
    board = {"上交所主板A股": "main_board", "深交所主板A股": "main_board", "深交所创业板A股": "chinext",
             "上交所科创板A股": "star"}.get(category)
    return {"name": data.get("SecurityShortName"), "list_date": listed, "board": board,
            "listing_status": "listed" if basic.get("agdm") == digits and board else None,
            "identity_source": "eastmoney_company_survey"}


def _tushare_delisted_profile(pro: Any, code: str) -> dict[str, Any]:
    rows = pro.stock_basic(ts_code=code, list_status="D", fields="ts_code,name,market,list_date,delist_date,list_status")
    if rows is None or rows.empty:
        return {}
    if len(rows) != 1 or rows.iloc[0]["ts_code"] != code or rows.iloc[0]["list_status"] != "D":
        raise ValueError("Tushare 退市资料与请求证券不一致")
    row = rows.iloc[0]
    listed, delisted = (jiaoyi_riqi(row[key]) for key in ("list_date", "delist_date"))
    if pd.isna(listed) or pd.isna(delisted) or listed >= delisted:
        raise ValueError("Tushare 退市资料日期无效")
    return {"name": row["name"], "list_date": listed, "delist_date": delisted, "board": _MARKET_BOARD.get(row["market"]),
            "listing_status": "delisted", "identity_source": "tushare_stock_basic_delisted"}


def _matching_snapshot_limits(code: str, day: pd.Timestamp, daily: pd.DataFrame, reference: datetime | None) -> dict[str, Any]:
    digits, exchange = code.split(".")
    symbol = exchange.lower() + digits
    with GongkaiShujuHTTPKehu("tencent_daily_limit_snapshot") as client:
        body, _ = client.qingqiu_wenben((f"https://qt.gtimg.cn/q={symbol}",), headers={"Referer": "https://gu.qq.com/"})
    match = re.fullmatch(rf'v_{symbol}="([^"\r\n]+)";?', body.strip())
    fields = match[1].split("~") if match else []
    if len(fields) < 49 or fields[2] != digits or not re.fullmatch(r"\d{14}", fields[30]) or jiaoyi_riqi(fields[30][:8]) != day:
        raise ValueError("腾讯涨跌停价快照的代码或交易日未核验")
    quote_time = datetime.strptime(fields[30], "%Y%m%d%H%M%S").replace(tzinfo=_TIMEZONE)
    if reference is not None and quote_time > reference.astimezone(_TIMEZONE):
        raise ValueError("腾讯涨跌停价快照晚于本次请求时点")
    row = daily.loc[daily["trade_date"].eq(day)].iloc[0]
    observed = [lishi_shuzhi(fields[index]) for index in (5, 33, 34, 3, 4, 47, 48)]
    if any(value is None or value <= 0 for value in observed):
        raise ValueError("腾讯涨跌停价快照缺少有效价量参考")
    expected = [lishi_shuzhi(row[field]) for field in RAW_FIELDS]
    reference = lishi_shuzhi(row.get("limit_reference_price"))
    if (any(value is None or abs(value - actual) > 1e-8 for value, actual in zip(expected, observed[:4]))
            or (reference is not None and abs(reference - observed[4]) > 1e-8)):
        raise ValueError("腾讯涨跌停价快照与同日原始日线不一致")
    return {"limit_up_price": observed[5], "limit_down_price": observed[6], "limit_source": "tencent_limit_snapshot",
            "limit_reference_price": observed[4], "limit_reference_source": row.get("limit_reference_source") or "tencent_limit_snapshot",
            "limit_quote_time": fields[30]}


def _eastmoney_dividends(code: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    with GongkaiShujuHTTPKehu("eastmoney_corporate_actions") as client:
        data, _ = client.qingqiu_json(
            (_EASTMONEY_DATA,), headers={"Referer": "https://data.eastmoney.com/yjfp/"},
            params={"reportName": "RPT_SHAREBONUS_DET", "columns": "ALL", "source": "WEB", "client": "WEB",
                    "pageNumber": 1, "pageSize": 100, "sortColumns": "EX_DIVIDEND_DATE", "sortTypes": "1",
                    "filter": f'(SECUCODE="{code}")(EX_DIVIDEND_DATE>=\'{_day(start)}\')(EX_DIVIDEND_DATE<=\'{_day(end)}\')(ASSIGN_PROGRESS="实施分配")'},
        )
    if data.get("message") == "返回数据为空" and not data.get("result"):
        return []
    block = data.get("result") or {}
    rows = block.get("data")
    if not isinstance(rows, list) or len(rows) != int(block["count"]):
        raise ValueError("东方财富权益分派范围或分页未核验")
    result = []
    for row in rows:
        day, announced = (jiaoyi_riqi(row[key]) for key in ("EX_DIVIDEND_DATE", "NOTICE_DATE"))
        if row.get("SECUCODE") != code or pd.isna(day) or not start <= day <= end:
            raise ValueError("东方财富权益分派身份或日期与请求冲突")
        if pd.notna(announced) and announced <= end:
            result.append({"ex_date": day, "announcement_date": announced,
                           "record_date": jiaoyi_riqi(row.get("EQUITY_RECORD_DATE")),
                           "cash_per_10_shares": lishi_shuzhi(row.get("PRETAX_BONUS_RMB")),
                           "bonus_shares_per_10": lishi_shuzhi(row.get("BONUS_RATIO")),
                           "capitalization_shares_per_10": lishi_shuzhi(row.get("IT_RATIO")),
                           "distribution_text": row.get("IMPL_PLAN_PROFILE"), "source": "eastmoney_sharebonus"})
    return result


def _eastmoney_full_day_halts(code: str, days: pd.DatetimeIndex) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """公开日历仅补核少量缺失日，单一供应商记录保留为待交叉核验。"""
    attempts, evidence = [], []
    digits = code.split(".")[0]
    with GongkaiShujuHTTPKehu("eastmoney_historical_suspension") as client:
        for day in days[-5:]:
            try:
                data, _ = client.qingqiu_json(
                    (_EASTMONEY_DATA,), headers={"Referer": "https://data.eastmoney.com/tfpxx/"},
                    params={"reportName": "RPT_CUSTOM_SUSPEND_DATA_INTERFACE", "columns": "ALL", "source": "WEB",
                            "client": "WEB", "pageNumber": 1, "pageSize": 50,
                            "filter": f'(MARKET="全部")(DATETIME=\'{_day(day)}\')(SECURITY_CODE="{digits}")'},
                )
                block = data.get("result") or {}
                rows = block.get("data") or []
                for row in rows:
                    if row.get("SECURITY_CODE") != digits or jiaoyi_riqi(row.get("DATETIME")) != day:
                        continue
                    evidence.append({"trade_date": day, "source": "eastmoney_suspension_calendar",
                                     "stop_type": row.get("SUSPEND_TYPE"), "reason": row.get("SUSPEND_REASON"),
                                     "source_fields": {key: row[key] for key in row if "DATE" in key or "TIME" in key or "TYPE" in key}})
            except WangluoQingqiuYichang as exc:
                attempts.append({"source": "eastmoney_historical_suspension", "status": "unavailable", "trade_date": day,
                                 "reason": str(exc)})
    return evidence, attempts


_NOTICE_PATHS = {
    "www.sse.com.cn": ("/disclosure/announcement/", "/disclosure/listedinfo/announcement/"),
    "static.sse.com.cn": ("/disclosure/listedinfo/announcement/",),
    "big5.sse.com.cn": ("/site/cht/www.sse.com.cn/disclosure/listedinfo/announcement/",),
    "www.szse.cn": ("/disclosure/notice/",),
    "disc.static.szse.cn": ("/download/disc/", "/disc/disk03/finalpage/"),
    "www.bse.cn": ("/disclosure/",),
    "static.cninfo.com.cn": ("/finalpage/",),
}
_NOTICE_DATE = r"\d{4}年\d{1,2}月\d{1,2}日"
_NOTICE_TIME = r"(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2})时(\d{1,2})分(?:(\d{1,2})秒)?"


def _notice_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%Y年%m月%d日"))


def _issuer_notice(url: str, code: str) -> list[dict[str, Any]]:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != parsed.hostname or parsed.query or parsed.fragment
            or not parsed.path.startswith(_NOTICE_PATHS.get(parsed.hostname, ())) or ".." in parsed.path.split("/")):
        raise ValueError("公告链接须为交易所或巨潮资讯的公开 HTTPS 公告")
    with GongkaiShujuHTTPKehu("official_trading_notice") as client:
        body, _ = client.qingqiu_erjinzhi((url,), headers={"Referer": url})
    if parsed.path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        if len(body) > 8_000_000 or not body.startswith(b"%PDF-"):
            raise ValueError("公告 PDF 内容或大小不符")
        reader = PdfReader(BytesIO(body))
        text = "".join(page.extract_text() for page in reader.pages[:3])
    else:
        text = unescape(re.sub(r"<[^>]+>", " ", body.decode("utf-8", errors="replace")))
    text = re.sub(r"\s+", "", text)
    digits = code.split(".")[0]
    if not re.search(rf"(?<!\d){digits}(?!\d)", text):
        raise ValueError("公告正文未核实请求证券代码")
    announced = re.search(r"/(\d{4}-\d{2}-\d{2})/", parsed.path)
    ann_date = jiaoyi_riqi(announced[1]) if announced else pd.NaT
    events = []
    span = _NOTICE_TIME + r".{0,60}?(?:暂停|实施临时停牌).{0,110}?" + _NOTICE_TIME + r".{0,12}?(?:恢复交易|复牌)"
    for match in re.finditer(span, text):
        start, resume = (datetime(*(int(value or 0) for value in group))
                         for group in (match.groups()[:6], match.groups()[6:]))
        if start.date() != resume.date() or start >= resume:
            raise ValueError("公告中的盘中停复牌时间不一致")
        events.append({"kind": "intraday", "start_date": pd.Timestamp(start.date()), "end_date": pd.Timestamp(start.date()),
                       "start_time": start.strftime("%H:%M:%S"), "resume_time": resume.strftime("%H:%M:%S")})
    stop_dates = re.findall(rf"({_NOTICE_DATE})(?:（[^）]*）)?(?:开市(?:时)?起)?停牌(?:[1一]天)?", text)
    resume_dates = re.findall(rf"({_NOTICE_DATE})(?:（[^）]*）)?(?:开市(?:时)?起|起)?复牌", text)
    if len(set(stop_dates)) == len(set(resume_dates)) == 1:
        stop, resume = map(_notice_date, (stop_dates[0], resume_dates[0]))
        if stop < resume and pd.notna(ann_date):
            events.append({"kind": "full_day", "start_date": stop, "end_date": resume - pd.Timedelta(days=1),
                           "resume_date": resume})
    change = re.search(r"(?:简称由|简称：由)[“\"]([^”\"]+)[”\"]变更为[“\"]([^”\"]+)[”\"]", text)
    effective = re.search(rf"(?:实施|撤销)(?:其他|退市)?风险警示的起始日[：:]({_NOTICE_DATE})"
                          rf"|(?:实施|撤销)起始日为({_NOTICE_DATE})"
                          rf"|({_NOTICE_DATE})开市起(?:被实施|撤销)[“\"]?(?:其他|退市)?风险警示", text)
    if change and effective and pd.notna(ann_date):
        day = _notice_date(next(group for group in effective.groups() if group is not None))
        if ann_date <= day and shi_fengxian_jingshi(change[1]) != shi_fengxian_jingshi(change[2]):
            events.append({"kind": "risk_warning", "start_date": day, "end_date": day,
                           "previous_name": change[1], "new_name": change[2], "is_st": shi_fengxian_jingshi(change[2])})
    delisting = re.search(rf"({_NOTICE_DATE})(?:复牌并|起)?进入退市整理期", text)
    delisting_rule = re.search(r"首(?:个交易|)日不实行(?:价格)?涨跌幅限制，此后(?:每个交易日|每日)的?涨跌幅限制为(\d+(?:\.\d+)?)%", text)
    if delisting and delisting_rule and pd.notna(ann_date):
        day = _notice_date(delisting[1])
        if ann_date <= day:
            events.append({"kind": "delisting_period", "start_date": day, "end_date": day,
                           "first_session_no_limit": True, "limit_rate": float(delisting_rule[1]) / 100})
    removal = re.search(rf"(?:终止上市暨摘牌(?:日|日期)(?:为|[：:])|将于)({_NOTICE_DATE})(?:终止上市暨摘牌|(?=[。；]))", text)
    if removal and pd.notna(ann_date):
        day = _notice_date(removal[1])
        if ann_date <= day:
            events.append({"kind": "delisting", "start_date": day, "end_date": day})
    if not events:
        raise ValueError("公告未解析到明确停复牌时点、风险警示生效日或退市整理期规则")
    return [{**event, "source": url, "announcement_date": ann_date} for event in events]


def huoqu_jiaoyi_gonggao(
    urls: Iterable[str], code: str, *, end: pd.Timestamp, delist_date: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """核对显式提供的公告。ST 只确认生效日及公告到生效前的旧简称。"""
    events, attempts = [], []
    urls = tuple(dict.fromkeys(urls))
    if not urls:
        return events, attempts
    from pypdf.errors import PdfReadError

    for url in dict.fromkeys(urls):
        try:
            records = _issuer_notice(url, code)
            for item in records:
                announced = item["announcement_date"]
                # 尚未发生的摘牌安排只保留在公告原文，历史状态不提前生效。
                if item["start_date"] > end or (pd.notna(announced) and announced > end):
                    continue
                if item["kind"] == "delisting_period" and pd.notna(delist_date) and delist_date > item["start_date"]:
                    item["end_date"] = min(end, delist_date - pd.Timedelta(days=1))
                events.append(item)
            attempts.append({"source": url, "status": "ok"})
        except (ImportError, ValueError, OSError, PdfReadError, WangluoQingqiuYichang) as exc:
            attempts.append({"source": url, "status": "unavailable", "reason": " ".join(str(exc).split())[:260]})
    return events, attempts


def huoqu_gupiao_lishi_jiaoyi(
    code: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    calendar: Iterable[pd.Timestamp],
    profile: dict[str, Any],
    tushare: Callable[[], Any],
    bao_timeout: float,
    notice_urls: Iterable[str] = (),
    snapshot_day: pd.Timestamp | None = None,
    reference: datetime | None = None,
    include_liquidity: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """逐源核验单股证据，同一份收益价格只使用配对的原始价与因子。"""
    attempts: list[dict[str, Any]] = []

    def attempt(source: str, load: Callable[[], Any], default: Any) -> Any:
        try:
            value = load()
            empty = value.empty if isinstance(value, (pd.DataFrame, pd.Series)) else not value
            attempts.append({"source": source, "status": "no_records" if empty else "ok"})
            return value
        except Exception as exc:  # Tushare SDK 直接抛出 Exception 表示接口限频或不可用。
            reason = " ".join(str(exc).split()).split("?", 1)[0][:240]
            attempts.append({"source": source, "status": "unavailable", "reason": reason})
            return default

    code = biaozhunhua_gupiao_daima(code)
    digits, exchange = code.split(".")
    identity = {"ts_code": code, "name": profile.get("name"), "list_date": jiaoyi_riqi(profile.get("list_date")),
                "delist_date": jiaoyi_riqi(profile.get("delist_date")), "board": _MARKET_BOARD.get(profile.get("market")),
                "identity_source": profile.get("identity_source"), "listing_status": profile.get("listing_status")}
    dates = pd.DatetimeIndex(sorted(set(calendar)))
    # 多取上一小段交易日，只有连贯真实行情或已确认停牌才能追溯昨收。
    lookback = dates[dates < start][-20:]
    fetch_start = lookback[0] if len(lookback) else start
    if pd.notna(identity["delist_date"]) and identity["delist_date"] < start:
        previous = dates[dates < identity["delist_date"]][-60:]
        if len(previous):
            fetch_start = min(fetch_start, previous[0])
    if exchange == "BJ":
        identity["board"] = "bse"
        if profile.get("old_ts_code") and profile.get("code_change_date"):
            identity.update(old_ts_code=profile["old_ts_code"], code_change_date=jiaoyi_riqi(profile["code_change_date"]),
                            code_mapping_source=profile.get("code_mapping_source"))
        else:
            mapping = attempt("bse_security_code_mapping", lambda: huoqu_bse_daima_duiying(code), {})
            if mapping:
                identity.update(mapping)
                identity["code_mapping_source"] = _BSE_CODES
        code = identity["ts_code"]
    elif any(pd.isna(identity[key]) for key in ("list_date", "board", "listing_status")):
        public = attempt("eastmoney_company_survey", lambda: _eastmoney_profile(code), {})
        identity.update({key: value for key, value in public.items() if pd.notna(value) and pd.isna(identity.get(key))})
    daily = pd.DataFrame()
    if exchange in {"SH", "SZ"}:
        symbol = exchange.lower() + "." + digits
        bao = attempt("baostock", lambda: bao_gupiao_shuju(symbol, _day(fetch_start), _day(end), timeout=bao_timeout), {})
        for field, reason in (bao.get("errors") or {}).items():
            attempts.append({"source": "baostock_" + field, "status": "unavailable", "reason": reason})
        listed = bao.get("profile") or []
        if len(listed) == 1 and listed[0].get("code") == symbol and listed[0].get("type") == "1":
            first, last = (jiaoyi_riqi(listed[0].get(key)) for key in ("ipoDate", "outDate"))
            if pd.notna(first) and (pd.isna(last) or first < last):
                identity.update(list_date=first, delist_date=last, identity_source="baostock_stock_basic",
                                listing_status={"0": "delisted", "1": "listed"}.get(listed[0].get("status")))
        if bao.get("daily"):
            daily = attempt("baostock_daily", lambda: zhengli_lishi_rili(_bao_daily(bao["daily"], symbol)).reset_index(), pd.DataFrame())
            if not daily.empty and "factors" in bao:
                indexes = pd.DatetimeIndex(daily["trade_date"])
                factor = attempt("baostock_factors", lambda: _bao_factors(bao.get("factors") or [], symbol, indexes), pd.Series(index=indexes, dtype=float))
                daily["adj_factor"] = factor.to_numpy()
                daily["factor_source"] = factor.notna().map({True: "baostock_adjust_factor", False: None}).to_numpy()
    if identity["listing_status"] is None and pd.isna(identity["delist_date"]) and exchange != "BJ":
        if exchange == "SH":
            retired = attempt("sse_delisted_stock_directory", lambda: huoqu_sse_tuishi_shenfen(code), {})
            if retired:
                identity.update({**retired, "board": _MARKET_BOARD.get(retired["market"])})
        if pd.isna(identity["delist_date"]):
            identity.update(attempt("tushare_stock_basic_delisted", lambda: _tushare_delisted_profile(tushare(), code), {}))
    effective_end = min(end, identity["delist_date"] - pd.Timedelta(days=1)) if pd.notna(identity["delist_date"]) else end
    expected = dates[(dates >= fetch_start) & (dates <= effective_end)]
    bao_daily = daily
    bao_latest = daily["trade_date"].max() if not daily.empty else pd.NaT
    bao_factor_missing = not daily.empty and ("adj_factor" not in daily or not daily["adj_factor"].notna().any())
    if daily.empty or bao_factor_missing or (len(expected) and (pd.isna(bao_latest) or bao_latest < expected[-1])):
        quote_code = code
        fetch_end = max(fetch_start, effective_end)
        tushare_daily = attempt("tushare_daily", lambda: zhengli_lishi_rili(_tushare_daily(tushare(), quote_code, fetch_start, fetch_end)).reset_index(), pd.DataFrame())
        if tushare_daily.empty and identity.get("old_ts_code") and end < identity["code_change_date"]:
            quote_code = identity["old_ts_code"]
            tushare_daily = attempt("tushare_daily_old_code", lambda: zhengli_lishi_rili(_tushare_daily(tushare(), quote_code, fetch_start, fetch_end)).reset_index(), pd.DataFrame())
        if not tushare_daily.empty and (daily.empty or tushare_daily["trade_date"].max() >= bao_latest):
            indexes = pd.DatetimeIndex(tushare_daily["trade_date"])
            factors = attempt("tushare_adj_factor", lambda: _tushare_factors(tushare(), quote_code, indexes, fetch_start, fetch_end), pd.Series(index=indexes, dtype=float))
            if daily.empty or tushare_daily["trade_date"].max() > bao_latest or (bao_factor_missing and factors.notna().any()):
                daily = tushare_daily.assign(adj_factor=factors.to_numpy(), source_ts_code=quote_code)
                daily["factor_source"] = factors.notna().map({True: "tushare_adj_factor", False: None}).to_numpy()
        elif daily.empty:
            daily = attempt("eastmoney_raw_daily", lambda: zhengli_lishi_rili(_eastmoney_daily(code, fetch_start, fetch_end)).reset_index(), pd.DataFrame())
    if not bao_daily.empty and daily is not bao_daily:
        # 更近的整段价格可由 Tushare 提供，BaoStock 只补同日独立的状态观测。
        status_fields = ["is_suspended", "suspension_source", "is_st", "st_source"]
        statuses = bao_daily.loc[bao_daily["trade_date"].isin(daily["trade_date"]) | bao_daily["is_suspended"].eq(True),
                                 ["trade_date", *status_fields]]
        daily = daily.merge(statuses, on="trade_date", how="outer", validate="one_to_one")
    if daily.empty:
        daily = pd.DataFrame(columns=["trade_date"])
    if snapshot_day is not None and start <= snapshot_day <= end and daily["trade_date"].eq(snapshot_day).any():
        limits = attempt("tencent_limit_snapshot", lambda: _matching_snapshot_limits(code, snapshot_day, daily, reference), {})
        for field, value in limits.items():
            daily.loc[daily["trade_date"].eq(snapshot_day), field] = value
    names = attempt("tushare_namechange", lambda: _tushare_names(tushare(), code, end), [])
    # 停牌可能在请求区间之前开始，上交所接口按重叠区间返回记录。
    notices = attempt("sse_historical_suspension", lambda: _sse_suspensions(code, fetch_start, effective_end), []) if exchange == "SH" else []
    if exchange == "SH" and not daily.empty:
        last_observed = daily["trade_date"].max()
        trailing_days = dates[(dates > last_observed) & (dates <= effective_end)]
        if len(trailing_days) and not any(event["kind"] == "full_day" and event["end_date"] >= trailing_days[-1] for event in notices):
            narrow_start = max(fetch_start, min(last_observed, trailing_days[-1] - pd.Timedelta(days=45)))
            if narrow_start > fetch_start:
                extra = attempt("sse_historical_suspension_latest_gap", lambda: _sse_suspensions(code, narrow_start, effective_end), [])
                notices.extend(event for event in extra if event not in notices)
    official, notice_attempts = huoqu_jiaoyi_gonggao(notice_urls, code, end=end, delist_date=identity["delist_date"])
    notices.extend(official)
    attempts.extend(notice_attempts)
    removal = [event for event in official if event["kind"] == "delisting"]
    if len(removal) == 1:
        identity.update(delist_date=removal[0]["start_date"], listing_status="delisted", delist_date_source=removal[0]["source"])
    dividends = attempt("eastmoney_sharebonus", lambda: _eastmoney_dividends(code, start, end), [])
    source_range_conflicts = daily.loc[~daily["trade_date"].isin(dates), "trade_date"]
    daily, conflicts = zhengli_jiaoyi_zhengju(daily, dates, identity, start=fetch_start, end=end, names=names, notices=notices)
    other_halts = []
    if exchange == "BJ" and not daily.empty:
        unknown = pd.DatetimeIndex(daily.loc[daily["trade_status"].eq("unknown") & daily["trade_date"].ge(start), "trade_date"])
        if len(unknown):
            other_halts, eastmoney_attempts = _eastmoney_full_day_halts(code, unknown)
            attempts.extend(eastmoney_attempts)
    final_trade = zuihou_shiji_chengjiao(daily, identity, end=end)
    if pd.notna(final_trade):
        identity["last_actual_trade_date"] = final_trade
    if not source_range_conflicts.empty:
        attempts.append({"source": "calendar_cross_check", "status": "unavailable", "reason": "有来源行情日期不在本次已核验日历内",
                         "dates": source_range_conflicts.tolist()})
    daily = daily.loc[daily["trade_date"].ge(start)].reset_index(drop=True)
    conflicts = [item for item in conflicts if item["trade_date"] >= _day(start)]
    liquidity_meta: dict[str, Any] | None = None
    if include_liquidity and not daily.empty:
        daily, liquidity_meta = _historical_liquidity(code, daily, bao_daily, attempt)
    return daily, {
        "profile": identity, "name_intervals": names, "notice_events": notices, "corporate_actions": dividends,
        "conflicts": conflicts, "attempts": attempts, "unconfirmed_suspension_records": other_halts,
        "liquidity": liquidity_meta,
        "fetched_at": datetime.now(_TIMEZONE).isoformat(timespec="seconds"),
        "persistence": "none", "intraday_coverage": "exchange_notices_only",
    }


def _historical_liquidity(
    code: str, daily: pd.DataFrame, bao_daily: pd.DataFrame, attempt: Callable[[str, Callable, Any], Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sessions = pd.DatetimeIndex(daily["trade_date"].tail(LIQUIDITY_HISTORY_SESSIONS))
    start, end = sessions[0], sessions[-1]
    turnover_sources = []
    if not bao_daily.empty and "turn" in bao_daily:
        bao_turn = bao_daily.loc[bao_daily["trade_date"].isin(sessions) & bao_daily["is_suspended"].eq(False), ["trade_date", "turn", "volume"]].rename(
            columns={"turn": "reported_turnover_rate_pct", "volume": "turnover_volume_shares"})
        turnover_sources.append(bao_turn.assign(turnover_source="baostock_daily"))
    valuation = attempt("eastmoney_historical_valuation", lambda: _public_historical_valuation(code, start, end), pd.DataFrame())
    traded_days = set(daily.loc[daily["raw_bar_valid"] & daily["trade_date"].isin(sessions), "trade_date"])
    merged = _merge_liquidity(daily, turnover_sources, valuation, sessions)
    for source, tencent in (("tencent_daily_turnover", True), ("eastmoney_daily_turnover", False)):
        if traded_days.issubset(merged.loc[merged["turnover_rate"].notna(), "trade_date"]):
            break
        observations = attempt(source, lambda: _public_turnover(code, start, end, tencent=tencent), pd.DataFrame())
        if not observations.empty:
            turnover_sources.append(observations)
            merged = _merge_liquidity(daily, turnover_sources, valuation, sessions)
    return merged, {"start_date": start, "end_date": end, "requested_sessions": len(sessions)}
