"""通达信每日主力控盘观测及同股量化对照；不推算账户持仓或投资者身份。

请求协议与多表字段参考 electkismet/eltdx，固定提交：
59d4615d3dafed4ecbdc480a34f96807cb132b92。
Copyright (c) 2026 ELTDX contributors；ELTDX Research-Only License。
https://github.com/electkismet/eltdx/blob/59d4615d3dafed4ecbdc480a34f96807cb132b92/LICENSE
本项目仅提取个人研究所需协议，以既有网络层访问，不安装或执行上游程序。
控盘算法位于服务端；公式、分母、原值单位和实际更新频率均未核验。
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.ashare.shuju_yuan import biaozhunhua_gupiao_daima
from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu, WangluoQingqiuYichang
from src.ashare.yinzi_gongcheng import factor_definition

_ENDPOINT = "http://static.tdx.com.cn:7615/TQLEX"
_REQ_ID = "200745"
_SOURCE = "tongdaxin_daily_control_ratio"
_UPSTREAM = "https://github.com/electkismet/eltdx/tree/59d4615d3dafed4ecbdc480a34f96807cb132b92"
_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
_LIMITS = [
    "服务端每日主力控盘指标，公式、分母及原值单位尚未核验；原值差不称为持股百分点",
    "没有机构、个人账户身份与库存，不能用原值或其补数推算机构/个人持股比例或数量",
    "返回仅有观测日期，服务器时间和取得时间不代表指标更新时间，未核验实时性",
    "指标与价量的同向或分歧只描述观测，不直接证明加仓、出货或预测后续收益",
]


class _LaiyuanShujuYichang(ValueError):
    """已取得的远端载荷不满足本来源契约。"""


def _beijing_time(value: Any = None) -> pd.Timestamp:
    result = pd.Timestamp(value if value is not None else datetime.now(_TZ))
    if pd.isna(result):
        raise ValueError("控盘分析核验时间无效")
    return result.tz_convert("Asia/Shanghai").tz_localize(None) if result.tzinfo else result


def _date(value: Any) -> pd.Timestamp:
    """显式处理来源 YYYYMMDD，避免整数被 pandas 当作纳秒。"""
    raw = str(value).strip()
    try:
        parsed = pd.to_datetime(raw, format="%Y%m%d", errors="raise") if raw.isdigit() and len(raw) == 8 else pd.Timestamp(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise _LaiyuanShujuYichang(f"控盘日期无效：{raw[:32]}") from exc
    if pd.isna(parsed):
        raise _LaiyuanShujuYichang("控盘日期缺失")
    result = pd.Timestamp(parsed)
    if result.tzinfo:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    if result != result.normalize():
        raise _LaiyuanShujuYichang("控盘观测日期包含未约定的时分秒")
    return result.normalize()


def _calendar_dates(calendar: Any) -> tuple[list[pd.Timestamp], str | None]:
    values = getattr(calendar, "open_dates", None)
    source = getattr(calendar, "source", None)
    if isinstance(calendar, dict):
        values = calendar.get("open_dates")
        source = calendar.get("source")
    if values is None and calendar is not None and not isinstance(calendar, (str, bytes, dict)):
        values = calendar
    if values is None:
        return [], source
    return sorted({_date(value) for value in values}), source or "caller_provided_trade_calendar"


def _table_rows(table: Any) -> list[dict[str, Any]]:
    if not isinstance(table, dict):
        raise _LaiyuanShujuYichang("控盘结果表不是对象")
    columns = table.get("ColName")
    if columns is None:
        descriptions = table.get("ColDes")
        if not isinstance(descriptions, list):
            raise _LaiyuanShujuYichang("控盘结果表缺少字段定义")
        columns = [item.get("Name") if isinstance(item, dict) else None for item in descriptions]
    if not isinstance(columns, list) or not columns or any(not isinstance(item, str) or not item.strip() for item in columns):
        raise _LaiyuanShujuYichang("控盘结果表字段定义无效")
    columns = [item.strip() for item in columns]
    if len(set(columns)) != len(columns):
        raise _LaiyuanShujuYichang("控盘结果表字段重复")
    content = table.get("Content")
    if not isinstance(content, list):
        raise _LaiyuanShujuYichang("控盘结果表内容不是数组")
    for field, actual in (("ColNum", len(columns)), ("RowNum", len(content))):
        if field in table and str(table[field]) != str(actual):
            raise _LaiyuanShujuYichang(f"控盘结果表 {field} 与实际表形状不一致")
    rows: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, list) or len(item) != len(columns):
            raise _LaiyuanShujuYichang("控盘结果表行宽与字段定义不一致")
        rows.append(dict(zip(columns, item)))
    return rows


def _parse_payload(payload: dict[str, Any], *, calendar_dates: list[pd.Timestamp], reference: pd.Timestamp) -> dict[str, Any]:
    if str(payload.get("ErrorCode")) != "0":
        raise _LaiyuanShujuYichang(f"通达信业务错误：{str(payload.get('ErrorInfo') or payload.get('ErrorCode'))[:160]}")
    tables = payload.get("ResultSets")
    if not isinstance(tables, list) or not tables:
        raise _LaiyuanShujuYichang("控盘响应没有结果表")
    metadata: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] | None = None
    for table in tables:
        rows = _table_rows(table)
        names = table.get("ColName") or [item.get("Name") for item in table.get("ColDes", [])]
        if {"N001", "N002"}.issubset(names):
            if observations is not None:
                raise _LaiyuanShujuYichang("控盘响应存在多个观测表，不能任意选取")
            observations = rows
        elif "ans" in names:
            metadata.extend(rows)
    if len(metadata) != 1 or str(metadata[0].get("ans")) != _REQ_ID:
        raise _LaiyuanShujuYichang("控盘响应 ReqId 与 200745 不一致或缺失")
    if not observations:
        raise _LaiyuanShujuYichang("控盘观测表为空")
    open_dates = set(calendar_dates)
    previous: pd.Timestamp | None = None
    series: list[dict[str, Any]] = []
    for row in observations:
        day = _date(row.get("N001"))
        if day > reference.normalize():
            raise _LaiyuanShujuYichang("控盘观测日期晚于本次核验日期")
        if day not in open_dates:
            raise _LaiyuanShujuYichang(f"控盘日期 {day:%Y-%m-%d} 不在本次权威交易日历内")
        if previous is not None and day <= previous:
            raise _LaiyuanShujuYichang("控盘日期重复或未按时间递增")
        try:
            raw_value = row.get("N002")
            if isinstance(raw_value, bool):
                raise ValueError("布尔值不是控盘原值")
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _LaiyuanShujuYichang("控盘原值不是有效数值") from exc
        if not math.isfinite(value):
            raise _LaiyuanShujuYichang("控盘原值不是有限数值")
        series.append({"trade_date": day.strftime("%Y-%m-%d"), "raw_value": value})
        previous = day
    meta = metadata[0]
    server_time = None
    if meta.get("date") and meta.get("time"):
        raw_time = str(meta["time"]).zfill(6)
        try:
            server = pd.to_datetime(f"{meta['date']} {raw_time}", format="%Y%m%d %H%M%S", errors="raise")
        except (ValueError, TypeError, OverflowError) as exc:
            raise _LaiyuanShujuYichang("控盘服务器时间格式异常") from exc
        server_time = server.strftime("%Y-%m-%d %H:%M:%S")
    return {"series": series, "metadata": meta, "provider_server_time": server_time}


def huoqu_kongpan_shuju(
    stock_code: str, *, trade_calendar: Any, now: Any = None, observation_count: int = 30,
) -> dict[str, Any]:
    """按明确沪深身份取得本次观测；不要求 Tushare 积分，不写入市场数据缓存。"""
    code = biaozhunhua_gupiao_daima(stock_code)
    reference = _beijing_time(now)
    if isinstance(observation_count, bool) or not isinstance(observation_count, int) or not 1 <= observation_count <= 30:
        raise ValueError("控盘观测条数必须是 1 至 30 的整数")
    result: dict[str, Any] = {
        "status": "unavailable", "source": _SOURCE, "stock_code": code,
        "upstream_reference": _UPSTREAM, "req_id": _REQ_ID,
        "requested_at": reference.strftime("%Y-%m-%d %H:%M:%S"), "fetched_at": None,
        "as_of": None, "latest": None, "series": [], "requested_observations": observation_count,
        "unit": "provider_raw_value", "unit_verified": False, "formula_known": False, "denominator_known": False,
        "date_verified": False, "realtime_verified": False, "provider_updated_at": None,
        "provider_server_time": None, "confirmation": "unavailable", "attempted_providers": [],
        "interpretation_limits": list(_LIMITS), "missing_reason": None,
        "identity_verification": "请求使用明确沪深代码和交易所；响应未回传股票代码，不单独宣称来源身份交叉核验",
    }
    if code.endswith(".BJ"):
        result.update(status="unsupported", missing_reason="北交所控盘源覆盖未核验，本次不请求或把全零当作真实控盘")
        return result
    dates, calendar_source = _calendar_dates(trade_calendar)
    result["calendar_source"] = calendar_source
    if not dates:
        result["missing_reason"] = "权威交易日历不可用，无法核验控盘观测日期"
        return result
    expected_dates = [day for day in dates if day <= reference.normalize()]
    if reference.time() < datetime.strptime("09:15", "%H:%M").time():
        expected_dates = [day for day in expected_dates if day < reference.normalize()]
    expected = expected_dates[-1] if expected_dates else None
    result["expected_trade_date"] = expected.strftime("%Y-%m-%d") if expected is not None else None
    request = [{
        "ReqId": _REQ_ID, "setcode": 1 if code.endswith(".SH") else 0,
        "code": code.split(".")[0], "Page": -1, "PageSize": "10",
        "modname": "mod_tcihq.dll", "zq_num": str(observation_count),
    }]
    try:
        with GongkaiShujuHTTPKehu(_SOURCE) as client:
            payload, endpoint = client.qingqiu_json(
                [_ENDPOINT], params={"Entry": "HQServ.hq_nlp_tcihq"}, method="POST", json_body=request,
            )
            result["attempted_providers"] = [item.to_dict() for item in client.attempts]
            result["endpoint"] = endpoint
        result["fetched_at"] = _beijing_time().strftime("%Y-%m-%d %H:%M:%S")
        parsed = _parse_payload(payload, calendar_dates=dates, reference=reference)
        if len(parsed["series"]) > observation_count:
            raise _LaiyuanShujuYichang("控盘返回观测条数超过本次明确请求上限")
    except WangluoQingqiuYichang as exc:
        result.update(exc.to_dict(), missing_reason=str(exc))
        return result
    except _LaiyuanShujuYichang as exc:
        result.update(error_code="source_payload_invalid", missing_reason=str(exc))
        return result
    result.update(parsed)
    first_observation = _date(parsed["series"][0]["trade_date"])
    result["calendar_dates"] = [day.strftime("%Y-%m-%d") for day in dates if first_observation <= day <= reference.normalize()]
    result["latest"] = dict(parsed["series"][-1])
    result["as_of"] = result["latest"]["trade_date"]
    result["date_verified"] = True
    latest_day = _date(result["as_of"])
    all_zero = all(row["raw_value"] == 0 for row in parsed["series"])
    stale = expected is not None and latest_day < expected
    intraday = latest_day == reference.normalize() and reference.hour * 60 + reference.minute < 15 * 60 + 5
    result.update(
        status="partial" if all_zero or stale else "ok",
        quality={"all_zero_unverified": all_zero, "stale": stale, "observation_count": len(parsed["series"])},
        confirmation="intraday_unconfirmed" if intraday else "daily_observation_update_time_unverified",
        missing_reason=("来源返回全部零值，尚未核验其真实含义；不据此推算持仓" if all_zero else
                        "最新控盘观测早于本次应有交易日，保留历史观察" if stale else None),
    )
    return result


def _finite(value: Any, *, positive: bool = False, nonnegative: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        return None
    return number


def _analysis_gaps(reasons: list[str]) -> list[dict[str, Any]]:
    return [{
        "code": "ownership_interpretation_partial", "component": "持仓与控盘观察", "reason": reason,
        "impact": "该项不能确认实际机构或个人持股，其余已核验证据和技术选股条件继续保留",
        "reassessment_condition": "补齐同日来源、连续窗口及指标定义和更新时间后复评",
        "blocks_selection": False,
    } for reason in dict.fromkeys(reasons) if reason]


def fenxi_kongpan_shuju(source_data: dict[str, Any], history: pd.DataFrame, *, analysis_date: Any = None) -> dict[str, Any]:
    """只计算连续交易日原值差及同日价量，不把相关观测计为独立投票。"""
    source = copy.deepcopy(source_data)
    metric_keys = ("tdx_control_raw", "tdx_control_change_1d", "tdx_control_change_5d",
                   "tdx_control_change_20d", "tdx_control_range_position")
    result: dict[str, Any] = {
        "key": "ownership_analysis", "label": "持仓分析（通达信主力控盘指标）",
        "status": source.get("status", "unavailable"), "source": source.get("source", _SOURCE),
        "stock_code": source.get("stock_code"), "as_of": source.get("as_of"),
        "latest": None, "changes": [], "sample_range": None, "price_context": {},
        "summary": "主力控盘观测暂不可用", "interpretation": [], "evidence_gaps": [],
        "missing_reason": source.get("missing_reason"), "interpretation_limits": list(_LIMITS),
        "realtime_verified": False, "unit_verified": False, "source_data": source,
        "metric_definitions": {key: factor_definition(key) for key in metric_keys},
    }
    if source.get("status") in {"unavailable", "unsupported", "error"} or not source.get("series"):
        result["evidence_gaps"] = _analysis_gaps([source.get("missing_reason") or "控盘观测不可用"])
        return result
    if not source.get("date_verified"):
        result.update(status="unavailable", missing_reason="控盘日期尚未经过权威交易日历核验")
        result["evidence_gaps"] = _analysis_gaps([result["missing_reason"]])
        return result
    series = source["series"]
    cutoff = _date(analysis_date) if analysis_date is not None else None
    # 盘中最新控盘与正式日线分别保留；不能用不同日期的值声称同日对照。
    latest = dict(series[-1])
    latest_day = _date(latest["trade_date"])
    result["latest"] = {**latest, "metric_key": "tdx_control_raw", "unit": factor_definition("tdx_control_raw")["unit"]}
    gaps: list[str] = []
    if source.get("missing_reason"):
        gaps.append(str(source["missing_reason"]))
    if source.get("quality", {}).get("all_zero_unverified"):
        result.update(status="partial", summary="来源全部返回零值，无法确认真实控盘含义；保留原始序列")
        result["evidence_gaps"] = _analysis_gaps(gaps or ["来源全部返回零值，含义未核实"])
        return result
    days = [_date(value) for value in source.get("calendar_dates", [])]
    by_day = {_date(row["trade_date"]): float(row["raw_value"]) for row in series}
    through_latest = [day for day in days if day <= latest_day]
    history_frame = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    history_rows: dict[pd.Timestamp, dict[str, Any]] = {}
    if not history_frame.empty and "trade_date" in history_frame:
        history_dates = pd.to_datetime(history_frame["trade_date"], errors="coerce")
        if history_dates.notna().all() and not history_dates.dt.normalize().duplicated().any():
            for day, (_, row) in zip(history_dates.dt.normalize(), history_frame.iterrows()):
                raw = row.to_dict()
                row_code = raw.get("ts_code")
                if row_code is not None and str(row_code) != str(source.get("stock_code")):
                    continue
                history_rows[pd.Timestamp(day)] = raw
        else:
            gaps.append("价量日线含无效或重复日期，未用于同日对照")
    aligned = latest_day in history_rows and (cutoff is None or latest_day <= cutoff)
    price_context: dict[str, Any] = {
        "status": "ok" if aligned else "unavailable", "trade_date": latest["trade_date"],
        "analysis_date": cutoff.strftime("%Y-%m-%d") if cutoff is not None else None,
        "date_aligned": aligned, "window_comparisons": [],
    }
    if aligned:
        row = history_rows[latest_day]
        price_context.update(
            close=_finite(row.get("close"), positive=True), volume=_finite(row.get("volume"), nonnegative=True),
            amount_yuan=_finite(row.get("amount_yuan"), nonnegative=True),
            turnover_rate_pct=_finite(row.get("turnover_rate"), nonnegative=True),
        )
    else:
        reason = "最新控盘观测与已核验正式日线日期未对齐，未计算当日价量关系"
        price_context["missing_reason"] = reason
        gaps.append(reason)
    for window in (1, 5, 20):
        expected = through_latest[-(window + 1):]
        enough = len(expected) == window + 1 and expected[-1] == latest_day and all(day in by_day for day in expected)
        metric_key = f"tdx_control_change_{window}d"
        change = {"window": window, "metric_key": metric_key, "status": "ok" if enough else "unavailable",
                  "value": None, "unit": factor_definition(metric_key)["unit"], "from_date": None,
                  "to_date": latest["trade_date"], "sample_count": len(expected), "missing_reason": None}
        if enough:
            delta = by_day[latest_day] - by_day[expected[0]]
            change.update(value=delta, from_date=expected[0].strftime("%Y-%m-%d"))
        else:
            change["missing_reason"] = f"需要权威日历内连续{window + 1}个交易日的有效控盘原值，不跨缺失日计算"
            gaps.append(change["missing_reason"])
        result["changes"].append(change)
        comparison: dict[str, Any] = {"window": window, "status": "unavailable", "price_return": None,
                                      "control_change": change["value"], "direction_relationship": None}
        prices = [_finite(history_rows.get(day, {}).get("close"), positive=True) for day in expected]
        if enough and aligned and all(day in history_rows for day in expected) and all(value is not None for value in prices):
            price_return = prices[-1] / prices[0] - 1.0
            delta = change["value"]
            relationship = ("同向上升" if delta > 0 and price_return > 0 else "同向下降" if delta < 0 and price_return < 0
                            else "方向分歧" if delta * price_return < 0 else "至少一项未变化")
            comparison.update(status="ok", from_date=change["from_date"], to_date=latest["trade_date"],
                              price_return=price_return, direction_relationship=relationship)
            if window == 1:
                previous_row = history_rows[expected[0]]
                for field in ("volume", "amount_yuan"):
                    before = _finite(previous_row.get(field), positive=True)
                    current = _finite(history_rows[latest_day].get(field), nonnegative=True)
                    comparison[f"{field}_ratio"] = current / before if before is not None and current is not None else None
        else:
            comparison["missing_reason"] = "控盘和价格需要同日、同股且连续的有效窗口；本次未满足"
        price_context["window_comparisons"].append(comparison)
    values = [float(row["raw_value"]) for row in series]
    low, high = min(values), max(values)
    result["sample_range"] = {
        "metric_key": "tdx_control_range_position",
        "sample_count": len(values), "from_date": series[0]["trade_date"], "to_date": latest["trade_date"],
        "minimum": low, "maximum": high,
        "range_position": (latest["raw_value"] - low) / (high - low) if high > low else None,
        "missing_reason": "样本原值全部相同，区间位置分母为零" if high == low else None,
    }
    result["price_context"] = price_context
    interpretations = [f"{latest['trade_date']} 主力控盘原值 {latest['raw_value']:.2f}，保留来源原值口径"]
    for item in result["changes"]:
        if item["status"] == "ok":
            interpretations.append(f"相隔{item['window']}个交易日原值变化 {item['value']:+.2f}，不是已核验持股百分点")
    for item in price_context["window_comparisons"]:
        if item["status"] == "ok":
            interpretations.append(f"同一{item['window']}交易日窗口股价变化 {item['price_return']:.2%}，与控盘指标{item['direction_relationship']}")
    result.update(
        status="partial", summary="；".join(interpretations), interpretation=interpretations,
        evidence_gaps=_analysis_gaps(gaps + ["指标单位、公式、分母和真实更新频率未核验，不能完成机构/个人真实持仓比例分析"]),
        missing_reason="；".join(gaps) or "指标观测可用；真实持仓身份、公式及更新频率尚未核验",
    )
    return result


__all__ = ["huoqu_kongpan_shuju", "fenxi_kongpan_shuju"]
