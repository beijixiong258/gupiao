"""本次候选的申万2021二级行业环境；只保存用户允许复用的行业参考。

公开来源的当前行业归类可用于历史近似分组，不冒充历史当时归属。
行业代码的父子关系来自官方分类表与公开指数目录，绝不按代码前缀推断。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO, StringIO
import json
import re
import ssl
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ashare.wangluo_kehu import GongkaiShujuHTTPKehu, WangluoQingqiuYichang
from ashare.yinzi_gongcheng import interval_return
from config import INDUSTRY_REFERENCE_PATH


_CLASSIFICATION_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
_STANDARD_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/2014to2021.xlsx"
_DIRECTORY_URL = "https://tushare.pro/document/2?doc_id=181"
_HISTORY_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/trend/"
_ANALYSIS_URL = "https://www.swsresearch.com/institute-sw/api/index_analysis/index_analysis_report/"
_HEADERS = {"Referer": "https://www.swsresearch.com/"}
_LIMITATIONS = [
    "按用户要求复用有公开来源的当前行业参考；历史信号沿用该归类，是近似分组，不是当时行业快照。",
    "行业参考保留来源与查证时间；归类可能变化，新的公开依据或人工纠正可更新记录，不以缺少历史归属阻断统计。",
    "采用申万2021二级行业价格指数；个股前复权收益与行业价格指数的收益口径分别保留。",
    "行业指数是独立环境参照，不增加候选股票或消除当前资格池、事后入榜偏差。",
]


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _client(provider: str) -> GongkaiShujuHTTPKehu:
    # 新来源沿用用户当前环境代理及 Windows 系统证书，不修改持久配置。
    return GongkaiShujuHTTPKehu(
        provider, ssl_context=ssl.create_default_context(), trust_env=True,
        settings={"domestic_connection_mode": "system_proxy", "connect_timeout_seconds": 6,
                  "read_timeout_seconds": 20, "max_attempts_per_endpoint": 2,
                  "retry_backoff_seconds": 0.35},
    )


def _code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text if re.fullmatch(r"\d{6}", text) else ""


def _load_directory(client: GongkaiShujuHTTPKehu) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    standard, standard_url = client.qingqiu_erjinzhi([_STANDARD_URL], headers=_HEADERS)
    source = pd.read_excel(BytesIO(standard), sheet_name="新旧对比版本2", header=1, dtype=str).fillna("")
    if source.shape[1] != 8 or list(source.columns)[4:7] != ["新版一级行业", "新版二级行业", "新版三级行业"]:
        raise ValueError("申万2021官方分类表结构改变，未推断列含义")
    official: dict[str, set[tuple[str, str, str]]] = {}
    official_second: dict[str, set[tuple[str, str]]] = {}
    first = second = ""
    for row in source.itertuples(index=False, name=None):
        one, two, three, raw_code = (str(value).strip() for value in row[4:8])
        category = _code(raw_code)
        if not category:
            continue
        if one:
            first, second = one, ""
        if two:
            second = two
            official_second.setdefault(category, set()).add((first, second))
        if three and first and second:
            official.setdefault(category, set()).add((first, second, three))

    html, directory_url = client.qingqiu_wenben([_DIRECTORY_URL])
    required = {"行业代码", "指数代码", "一级行业", "二级行业", "三级行业", "指数类别"}
    tables = [frame for frame in pd.read_html(StringIO(html), keep_default_na=False) if required.issubset(frame.columns)]
    if len(tables) != 1:
        raise ValueError("公开申万2021指数目录表未唯一确定")
    directory = tables[0].copy()
    for column in required:
        directory[column] = directory[column].astype(str).str.strip()
    second_indices: dict[tuple[str, str], set[str]] = {}
    for row in directory[directory["指数类别"].eq("二级行业")].to_dict("records"):
        category, index = _code(row["行业代码"]), _code(row["指数代码"])
        path = (row["一级行业"], row["二级行业"])
        if index and official_second.get(category) == {path}:
            second_indices.setdefault(path, set()).add(index)
    mapping: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    for row in directory[directory["指数类别"].eq("三级行业")].to_dict("records"):
        category = _code(row["行业代码"])
        path = (row["一级行业"], row["二级行业"], row["三级行业"])
        indices = second_indices.get(path[:2], set())
        if not category or official.get(category) != {path} or len(indices) != 1:
            conflicts.append(category or str(row["行业代码"]))
            continue
        value = {"industry_code": next(iter(indices)), "industry_name": path[1],
                 "industry_level_1": path[0], "industry_level_3": path[2]}
        if category in mapping and mapping[category] != value:
            conflicts.append(category)
        else:
            mapping[category] = value
    for category in conflicts:
        mapping.pop(category, None)
    if not mapping:
        raise ValueError("申万官方分类与公开指数目录没有核验一致的映射")
    return mapping, {"standard_source": standard_url, "index_mapping_source": directory_url,
                     "classification": "SW2021", "industry_level": "L2", "fetched_at": _now(),
                     "verified_third_level_mappings": len(mapping), "unverified_classification_codes": sorted(set(conflicts)),
                     "mapping_method": "官方新版分类显式父子名称与公开目录行业代码逐项匹配，不按代码前缀推断"}


def _load_memberships(client: GongkaiShujuHTTPKehu, symbols: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    content, url = client.qingqiu_erjinzhi([_CLASSIFICATION_URL], headers=_HEADERS)
    data = pd.read_excel(BytesIO(content), dtype={"股票代码": str, "行业代码": str})
    required = {"股票代码", "计入日期", "行业代码", "更新日期"}
    if not required.issubset(data.columns):
        raise ValueError("申万行业历史归属缺少代码、计入日期或更新日期")
    count = len(data)
    data = data.rename(columns={"股票代码": "symbol", "计入日期": "start_date",
                                "行业代码": "classification_code", "更新日期": "update_date"})
    data["symbol"] = data["symbol"].map(_code)
    data = data[data["symbol"].isin(symbols)].copy()
    data["classification_code"] = data["classification_code"].map(_code)
    for column in ("start_date", "update_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    return data.drop_duplicates(), {"source": url, "fetched_at": _now(), "source_rows": count,
                                    "candidate_rows": len(data), "point_in_time_verified": False}


def _valid_reference(value: Any) -> bool:
    """只接受有出处和查证日期的归类笔记，不把未知来源的标签当行业事实。"""
    return (
        isinstance(value, dict) and bool(_code(value.get("industry_code")))
        and isinstance(value.get("industry_name"), str) and bool(value["industry_name"].strip())
        and value.get("classification") == "SW2021" and value.get("industry_level") == "L2"
        and isinstance(value.get("source"), str) and value["source"].startswith(("https://", "http://"))
        and pd.notna(pd.to_datetime(value.get("fetched_at"), errors="coerce"))
        and pd.notna(pd.to_datetime(value.get("reference_date"), errors="coerce"))
        and isinstance(value.get("approximate"), bool)
    )


def huoqu_hangye_cankao(codes: list[str], *, refresh: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """复用行业笔记，只查询尚无记录的股票；显式刷新时重查公开来源。"""
    requested = list(dict.fromkeys(codes))
    stored: dict[str, Any] = {}
    meta: dict[str, Any] = {"reference_path": str(INDUSTRY_REFERENCE_PATH), "warnings": [],
                            "reused_codes": [], "fetched_codes": [], "attempted_providers": []}
    readable = True
    if INDUSTRY_REFERENCE_PATH.is_file():
        try:
            payload = json.loads(INDUSTRY_REFERENCE_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
                raise ValueError("行业参考笔记结构无效")
            stored = payload["records"]
        except (OSError, ValueError) as exc:
            readable = False
            meta["warnings"].append(f"行业参考笔记读取失败，保留原文件：{exc}")
    references = {code: dict(stored[code]) for code in requested if _valid_reference(stored.get(code))}
    missing = requested if refresh else [code for code in requested if code not in references]
    meta["reused_codes"] = sorted(references)
    fresh: dict[str, dict[str, Any]] = {}
    if missing:
        with _client("swsresearch_classification") as client:
            try:
                mapping, directory_meta = _load_directory(client)
                rows, source_meta = _load_memberships(client, {_code(code.split(".")[0]) for code in missing})
                meta.update(directory=directory_meta, membership_source=source_meta)
                fetched_at = source_meta["fetched_at"]
                today = pd.Timestamp(fetched_at).tz_localize(None).normalize()
                for code in missing:
                    local = rows[rows["symbol"].eq(_code(code.split(".")[0]))]
                    local = local[local["start_date"].le(today)]
                    if local.empty:
                        continue
                    latest = local[local["start_date"].eq(local["start_date"].max())]
                    categories = latest["classification_code"].unique().tolist()
                    if len(categories) != 1 or categories[0] not in mapping:
                        continue
                    updated = latest["update_date"].max()
                    fresh[code] = {**mapping[categories[0]], "classification": "SW2021", "industry_level": "L2",
                        "source": source_meta["source"], "fetched_at": fetched_at,
                        "reference_date": today.strftime("%Y-%m-%d"), "approximate": False,
                        "standard_source": directory_meta["standard_source"],
                        "index_mapping_source": directory_meta["index_mapping_source"],
                        "source_effective_date": latest["start_date"].max().strftime("%Y-%m-%d"),
                        "source_update_date": updated.strftime("%Y-%m-%d") if pd.notna(updated) else None,
                        "classification_note": "公开资料最新行业归属；历史统计沿用当前分类，不代表历史当时归属"}
            except (WangluoQingqiuYichang, ValueError, OSError) as exc:
                meta["warnings"].append(f"公开行业归属查询失败：{str(exc)[:240]}")
            meta["attempted_providers"] = [attempt.to_dict() for attempt in client.attempts]
    references.update(fresh)
    if fresh and readable:
        try:
            INDUSTRY_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            INDUSTRY_REFERENCE_PATH.write_text(json.dumps(
                {"purpose": "用户允许跨请求复用的公开行业归属参考，不保存行情或计算结果", "records": {**stored, **fresh}},
                ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except OSError as exc:
            meta["warnings"].append(f"行业归属已取得，但参考笔记保存失败：{exc}")
    meta.update(fetched_codes=sorted(fresh), reused_codes=sorted(set(references) - set(fresh)),
                missing_codes=[code for code in requested if code not in references])
    return references, meta


def _membership_panel(reference: dict[str, Any] | None, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame({"trade_date": dates, "industry_code": None, "industry_name": None,
                          "membership_verified": False, "membership_usable": False,
                          "membership_approximate": False, "membership_basis": "unavailable"})
    if reference is None:
        return frame, {"missing_reasons": ["public_membership_reference_missing"], "industry_reference": None}
    frame["industry_code"] = reference["industry_code"]
    frame["industry_name"] = reference["industry_name"]
    frame["membership_usable"] = True
    frame["membership_basis"] = "public_current_reference"
    frame["membership_approximate"] = bool(reference["approximate"]) | frame["trade_date"].ne(pd.Timestamp(reference["reference_date"]))
    frame["membership_verified"] = ~frame["membership_approximate"]
    frame["membership_reference_date"] = reference["reference_date"]
    return frame, {"missing_reasons": [], "industry_reference": dict(reference),
                   "classification_source": reference["source"], "fetched_at": reference["fetched_at"],
                   "verified_membership_dates": int(frame["membership_verified"].sum()),
                   "approximate_membership_dates": int(frame["membership_approximate"].sum()),
                   "membership_policy": "public_current_reference", "point_in_time_verified": False}


def _index_returns(payload: dict[str, Any], index_code: str, dates: pd.DatetimeIndex,
                   *, close_only: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("申万指数历史返回空数据或结构改变")
    source = pd.DataFrame(rows)
    required = {"swindexcode", "bargaindate", "closeindex"}
    if not required.issubset(source.columns):
        raise ValueError("申万指数历史缺少身份、日期或所需价格字段")
    if source["swindexcode"].map(_code).ne(index_code).any():
        raise ValueError("申万指数历史返回了其他指数身份")
    # 备用源返回带 +08:00 的时间，统一成北京时间交易日后再与无时区日历对齐。
    source["trade_date"] = (
        pd.to_datetime(source["bargaindate"], errors="coerce", format="mixed", utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    )
    duplicate = source["trade_date"].duplicated(keep=False)
    numeric = source[["closeindex"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric).all(axis=1) & numeric.gt(0).all(axis=1)
    valid &= source["trade_date"].notna() & ~duplicate
    clean = source.loc[valid, ["trade_date"]].copy()
    clean["close"] = numeric.loc[valid, "closeindex"]
    clean = clean.set_index("trade_date").reindex(dates)
    frame = pd.DataFrame({"trade_date": dates})
    close = clean["close"]
    endpoints: dict[str, dict[str, Any]] = {}
    for window in (5, 20):
        values = interval_return(close, window)
        frame[f"industry_ret_{window}"] = values.to_numpy()
        start_price = close.iloc[-window - 1] if len(close) > window else np.nan
        end_price = close.iloc[-1]
        endpoints[str(window)] = {
            "start_date": dates[-window - 1].strftime("%Y-%m-%d") if len(dates) > window else None,
            "end_date": dates[-1].strftime("%Y-%m-%d"),
            "start_close": float(start_price) if pd.notna(start_price) else None,
            "end_close": float(end_price) if pd.notna(end_price) else None,
            "usable": bool(pd.notna(values.iloc[-1])),
        }
    source_dates, calendar_dates = set(source["trade_date"].dropna()), set(dates)
    detail = {"source_rows": len(source), "invalid_or_duplicate_rows": int((~valid).sum()),
              "duplicate_date_rows": int(duplicate.sum()), "valid_calendar_rows": int(close.notna().sum()),
              "missing_calendar_dates": [value.strftime("%Y-%m-%d") for value in dates if value not in source_dates],
              "invalid_dates_in_requested_calendar": sorted({value.strftime("%Y-%m-%d") for value in source.loc[~valid, "trade_date"].dropna() if value in calendar_dates}),
              "duplicate_dates_in_requested_calendar": sorted({value.strftime("%Y-%m-%d") for value in source.loc[duplicate, "trade_date"].dropna() if value in calendar_dates}),
              "latest_21_session_missing_or_invalid_dates": close.iloc[-21:][close.iloc[-21:].isna()].index.strftime("%Y-%m-%d").tolist(),
              "source_latest_date": source["trade_date"].max().strftime("%Y-%m-%d") if source["trade_date"].notna().any() else None,
              "last_usable_date": close.dropna().index.max().strftime("%Y-%m-%d") if close.notna().any() else None,
              "current_return_endpoints": endpoints,
              "date_timezone": "Asia/Shanghai",
              "return_definition": "same_index_close / close_n_authoritative_sessions_earlier - 1; valid_endpoints",
              "price_quality_scope": "unique_valid_dates_and_positive_finite_close; return_endpoints_required",
              "ohlc_provided": not close_only and {"openindex", "maxindex", "minindex"}.issubset(source.columns),
              "ohlc_used": False,
              "adjustment": "official_price_index"}
    return frame, detail


def _analysis_history(client: GongkaiShujuHTTPKehu, index_code: str, dates: pd.DatetimeIndex) -> tuple[dict[str, Any], dict[str, Any]]:
    """只请求目标指数，核对稳定总数和完整分页，不拼接主源缺行。"""
    rows: list[dict[str, Any]] = []
    expected: int | None = None
    page_size, maximum_pages = 100, 12
    for page in range(1, maximum_pages + 1):
        payload, _ = client.qingqiu_json(
            [_ANALYSIS_URL], headers=_HEADERS,
            params={"page": page, "page_size": page_size, "index_type": "二级行业", "type": "DAY",
                    "start_date": dates[0].strftime("%Y-%m-%d"), "end_date": dates[-1].strftime("%Y-%m-%d"),
                    "swindexcode": index_code},
        )
        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("申万指数分析备用源分页结构改变")
        try:
            count = int(data.get("count", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("申万指数分析备用源总数字段无效") from exc
        if count < 1 or count > page_size * maximum_pages or (expected is not None and count != expected):
            raise ValueError("申万指数分析备用源总数为空、越界或分页间变化")
        expected = count
        part = data["results"]
        if not part or any(not isinstance(row, dict) or _code(row.get("swindexcode")) != index_code for row in part):
            raise ValueError("申万指数分析备用源空页或没有遵守目标指数身份筛选")
        rows.extend(part)
        if len(rows) >= expected:
            if len(rows) != expected:
                raise ValueError("申万指数分析备用源实际行数与总数不一致")
            return {"data": rows}, {"reported_count": expected, "received_rows": len(rows), "pages": page,
                                     "complete_pagination": True}
    raise ValueError("申万指数分析备用源超过受限分页次数")


def _fetch_index(index_code: str, dates: pd.DatetimeIndex) -> tuple[str, pd.DataFrame | None, dict[str, Any]]:
    meta: dict[str, Any] = {"source": _HISTORY_URL, "industry_code": index_code, "missing_reasons": [],
                            "fallback_used": False, "source_checks": []}
    frame: pd.DataFrame | None = None
    with _client("swsresearch_industry_history") as client:
        for source in (_HISTORY_URL, _ANALYSIS_URL):
            check: dict[str, Any] = {"source": source, "missing_reasons": []}
            try:
                if source == _HISTORY_URL:
                    payload, _ = client.qingqiu_json([source], params={"swindexcode": index_code, "period": "DAY"}, headers=_HEADERS)
                else:
                    payload, pagination = _analysis_history(client, index_code, dates)
                    check.update(pagination)
                candidate, detail = _index_returns(payload, index_code, dates, close_only=source == _ANALYSIS_URL)
                check.update(detail)
                usable = candidate["industry_ret_20"].notna().any()
                current = bool(pd.notna(candidate.iloc[-1]["industry_ret_20"]))
                check["status"] = "ok" if current else "partial" if usable else "unavailable"
                if not current:
                    check["missing_reasons"] = ["latest_industry_return_endpoints_missing"]
                if frame is None or current:
                    # 整段替换，不以备用来源补主源某个缺失交易日。
                    frame = candidate
                    meta.update(check)
                    meta["fallback_used"] = source == _ANALYSIS_URL
                meta["source_checks"].append(check)
                if current:
                    break
            except (WangluoQingqiuYichang, ValueError, OSError) as exc:
                check.update(status="unavailable", missing_reasons=[str(exc)[:240]])
                meta["source_checks"].append(check)
        if frame is None:
            meta.update(status="unavailable", missing_reasons=[reason for check in meta["source_checks"] for reason in check["missing_reasons"]])
        meta.update(fetched_at=_now(), attempted_providers=[attempt.to_dict() for attempt in client.attempts])
    return index_code, frame, meta


def huoqu_hangye_huanjing(
    codes: list[str], *, calendar: Any, start_date: Any, end_date: Any,
    market_features: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """复用有来源的行业参考，行情每次重新获取；历史归类近似与行情缺失分开。"""
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    calendar_values = getattr(calendar, "open_dates", calendar)
    all_dates = pd.DatetimeIndex(sorted({pd.Timestamp(value).normalize() for value in calendar_values if pd.Timestamp(value).normalize() <= end}))
    dates = all_dates[all_dates >= start]
    # 返回窗口前保留20个权威交易日以计算首日20日收益。
    first = int(all_dates.searchsorted(start))
    calculation_dates = all_dates[max(0, first - 20):]
    if dates.empty or end not in dates:
        raise ValueError("行业环境缺少目标日权威交易日历")
    requested = list(dict.fromkeys(str(code) for code in codes))
    meta: dict[str, Any] = {"status": "unavailable", "classification": "SW2021", "industry_level": "L2",
                            "as_of": end.strftime("%Y-%m-%d"), "fetched_at": _now(), "persistence": "industry_references_only",
                            "point_in_time_verified": False, "limitations": list(_LIMITATIONS), "by_code": {},
                            "by_index": {}, "attempted_providers": [], "missing_reasons": []}
    panels: dict[str, pd.DataFrame] = {}
    references, reference_meta = huoqu_hangye_cankao(requested)
    meta.update(reference_meta)
    indices: set[str] = set()
    for code in requested:
        panel, detail = _membership_panel(references.get(code), dates)
        panels[code] = panel
        meta["by_code"][code] = {**detail, "reference_reused": code in reference_meta["reused_codes"],
                                "point_in_time_verified": False}
        indices.update(panel.loc[panel["membership_usable"], "industry_code"].dropna().astype(str))
    index_frames: dict[str, pd.DataFrame] = {}
    if indices:
        with ThreadPoolExecutor(max_workers=min(3, len(indices))) as pool:
            for index, frame, detail in pool.map(lambda value: _fetch_index(value, calculation_dates), sorted(indices)):
                meta["by_index"][index] = detail
                if frame is not None:
                    index_frames[index] = frame.set_index("trade_date")

    market = pd.DataFrame(index=dates)
    if "trade_date" in market_features.columns:
        observed = market_features.copy()
        observed["trade_date"] = pd.to_datetime(observed["trade_date"], errors="coerce").dt.normalize()
        # 指数因子输入应为一日一行；重复日期不任选其中一行。
        observed = observed[observed["trade_date"].notna() & ~observed["trade_date"].duplicated(keep=False)].set_index("trade_date")
        for window in (5, 20):
            field = f"market_csi300_ret_{window}"
            values = pd.to_numeric(observed[field], errors="coerce") if field in observed else pd.Series(dtype=float)
            market[f"market_ret_{window}"] = values.where(np.isfinite(values)).reindex(dates)
    for window in (5, 20):
        if f"market_ret_{window}" not in market:
            market[f"market_ret_{window}"] = np.nan
    ready_codes = 0
    for code, panel in panels.items():
        panel["industry_ret_5"] = np.nan
        panel["industry_ret_20"] = np.nan
        for index in panel["industry_code"].dropna().unique():
            history = index_frames.get(str(index))
            if history is None:
                continue
            mask = panel["membership_usable"] & panel["industry_code"].eq(index)
            for window in (5, 20):
                panel.loc[mask, f"industry_ret_{window}"] = history[f"industry_ret_{window}"].reindex(pd.DatetimeIndex(panel.loc[mask, "trade_date"])).to_numpy()
        for window in (5, 20):
            panel[f"market_ret_{window}"] = market[f"market_ret_{window}"].reindex(pd.DatetimeIndex(panel["trade_date"])).to_numpy()
        detail = meta["by_code"][code]
        current = panel.iloc[-1]
        ready = bool(current["membership_usable"] and np.isfinite(current["industry_ret_20"]) and np.isfinite(current["market_ret_20"]))
        ready_codes += int(ready)
        used = sorted(panel["industry_code"].dropna().unique().tolist())
        detail.update(status="ok" if ready else "unavailable", index_codes=used,
                      industry_code=current["industry_code"], industry_name=current["industry_name"],
                      index_sources={index: meta["by_index"].get(index, {}) for index in used},
                      current_environment_ready=ready,
                      valid_environment_dates=int((panel["membership_usable"] & panel["industry_ret_20"].notna() & panel["market_ret_20"].notna()).sum()))
        if not current["membership_usable"]:
            detail["missing_reasons"].append("public_membership_reference_missing")
        if not np.isfinite(current["industry_ret_20"]):
            detail["missing_reasons"].append("current_industry_return_missing")
        if not np.isfinite(current["market_ret_20"]):
            detail["missing_reasons"].append("current_market_return_missing")
        detail["missing_reasons"] = sorted(set(detail["missing_reasons"] + meta["missing_reasons"]))
    meta.update(status="ok" if requested and ready_codes == len(requested) else "partial" if ready_codes else "unavailable",
                requested_codes=len(requested), current_environment_ready_codes=ready_codes,
                requested_industry_indices=len(indices))
    return panels, meta


__all__ = ["huoqu_hangye_huanjing", "huoqu_hangye_cankao"]
