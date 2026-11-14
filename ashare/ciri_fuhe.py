"""导出盘后已出具的预期，次日用两端同源正式日线逐股复核。"""

from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, Iterable, Mapping

import pandas as pd

from ashare.lishi_jiaoyi import jiaoyi_riqi, lishi_shuzhi
from ashare.peizhi import jiazai_lianghua_peizhi
from ashare.shichang_shuju import FenxiShujuShangxiawen, JiaoyiRili, _beijing_now, shichang_shizhong, zuixin_wanzheng_jiaoyiri
from ashare.shuju_yuan import biaozhunhua_gupiao_daima
from ashare.xuangu_bangdan import LEADERBOARD_SCOPE


FORECAST_RECORD_TYPE = "next_session_forecast_record"
_AWARE_BEIJING = "Asia/Shanghai"


def _timestamp(value: Any) -> pd.Timestamp:
    if not isinstance(value, (str, datetime)):
        raise ValueError("预测生成时间无效")
    result = pd.to_datetime(value, errors="coerce")
    if pd.isna(result):
        raise ValueError("预测生成时间无效")
    return result.tz_convert(_AWARE_BEIJING).tz_localize(None) if result.tzinfo else result


def goujian_ciri_yuce_jilu(
    *, signal_date: pd.Timestamp, clock: Mapping[str, Any],
    scope: Mapping[str, Any], method: str, qualified: Iterable[dict[str, Any]],
    displayed: Iterable[str], comparison_complete: bool, has_custom_conditions: bool,
) -> dict[str, Any] | None:
    """使用已完成排序的入围股票与双榜成员，生成调用方可选择保存的记录。"""
    if scope.get("scope_type") != LEADERBOARD_SCOPE or clock.get("session_status") not in {"post_close", "non_trading_day"}:
        return None
    members = scope.get("leaderboard_members") or []
    if not members:
        return None
    candidates = []
    for item in qualified:
        selection = item["selection"]
        baseline = selection.get("baseline_estimate") or {}
        candidates.append({
            "ts_code": item["ts_code"], "name": item.get("name"),
            "expected_next_day_return": selection["expected_next_day_return"],
            "legacy_24_expected_next_day_return": baseline.get("expected_next_day_return"),
            "return_rank": selection["comparison_rank"], "observation_rank": selection["observation_rank"],
        })
    return {
        "type": FORECAST_RECORD_TYPE,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "issued_at": _beijing_now().isoformat(timespec="seconds") + "+08:00",
        "selection_method": method, "scope_type": LEADERBOARD_SCOPE,
        "scope_status": scope.get("status"), "scope_source": scope.get("source"),
        "comparison_complete": bool(comparison_complete),
        "default_conditions": not has_custom_conditions,
        "leaderboard_members": members, "qualified_candidates": candidates, "displayed_codes": list(displayed),
    }


def _next_session(day: pd.Timestamp, calendar: JiaoyiRili) -> pd.Timestamp | None:
    return min((session for session in calendar.open_dates if session > day), default=None)


def chuju_ciri_yuce(context: FenxiShujuShangxiawen, *, count: int | None = None) -> dict[str, Any]:
    """走固定日终双榜入口，返回现时可交付给调用者的预测记录。"""
    from ashare.xuangu_fenxi import fenxi_xuangu

    now = context.reference
    calendar = context.jiaoyi_rili(end_date=(pd.Timestamp(now.date()) + pd.Timedelta(days=14)).strftime("%Y-%m-%d"))
    signal = zuixin_wanzheng_jiaoyiri(now, calendar=calendar)
    target = _next_session(signal, calendar)
    allowed = {"post_close", "non_trading_day"}
    if target is None or shichang_shizhong(now, calendar=calendar)["session_status"] not in allowed:
        raise ValueError("只能在收盘确认或非交易日出具固定双榜记录，且需要已经核验的目标交易日")
    result = fenxi_xuangu(fanwei=LEADERBOARD_SCOPE, shuliang=count, context=context)
    record = result.get("next_session_forecast")
    if not record:
        return {key: result[key] for key in ("status", "outcome", "error", "error_code", "stage", "scope") if key in result}
    if _timestamp(record["issued_at"]) >= target + pd.Timedelta(hours=9, minutes=15):
        raise ValueError("本次选股结束时目标日集合竞价已开始，不能出具该日事前记录")
    return {**record, "target_date": str(target.date()), "calendar_source_at_issue": calendar.source}


def _validate_record(record: Mapping[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, dict[str, Any]]]:
    if not isinstance(record, Mapping) or record.get("type") != FORECAST_RECORD_TYPE or record.get("scope_type") != LEADERBOARD_SCOPE:
        raise ValueError("请提供程序导出的固定双榜预测记录")
    if record.get("selection_method") not in {"environment_96", "legacy_24"}:
        raise ValueError("预测记录缺少有效的选股方法")
    if not isinstance(record.get("signal_date"), str):
        raise ValueError("预测记录缺少信号日")
    signal = jiaoyi_riqi(record["signal_date"])
    issued = _timestamp(record.get("issued_at"))
    if pd.isna(signal) or issued < signal + pd.Timedelta(hours=15, minutes=5):
        raise ValueError("预测记录早于信号日正式收盘")
    members = record.get("leaderboard_members")
    candidates = record.get("qualified_candidates")
    shown = record.get("displayed_codes")
    if not isinstance(members, list) or not 1 <= len(members) <= 200 or not isinstance(candidates, list) or not isinstance(shown, list):
        raise ValueError("预测记录缺少双榜、合格候选或已展示对象")
    identities: dict[str, dict[str, Any]] = {}
    gain_ranks, turnover_ranks = set(), set()
    for item in members:
        if not isinstance(item, dict) or not isinstance(item.get("ts_code"), str):
            raise ValueError("双榜成员身份无效")
        code = biaozhunhua_gupiao_daima(item["ts_code"])
        if code != item["ts_code"] or code in identities:
            raise ValueError("双榜成员代码格式错误或重复")
        if item.get("gain_rank") is None and item.get("turnover_rank") is None:
            raise ValueError("双榜成员缺少原始榜单名次")
        for key, ranks in (("gain_rank", gain_ranks), ("turnover_rank", turnover_ranks)):
            rank = item.get(key)
            if rank is not None:
                if type(rank) is not int or not 1 <= rank <= 100 or rank in ranks:
                    raise ValueError("双榜原始名次无效或重复")
                ranks.add(rank)
        identities[code] = {"ts_code": code, "name": item.get("name"),
                            "gain_rank": item.get("gain_rank"), "turnover_rank": item.get("turnover_rank")}
    qualified_codes = []
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("ts_code"), str) or item["ts_code"] not in identities:
            raise ValueError("预测记录中有双榜范围外的合格候选")
        code = item["ts_code"]
        primary = item.get("expected_next_day_return")
        legacy = item.get("legacy_24_expected_next_day_return")
        ranks = (item.get("return_rank"), item.get("observation_rank"))
        if (code in qualified_codes or type(primary) not in (int, float) or lishi_shuzhi(primary) is None
                or (legacy is not None and (type(legacy) not in (int, float) or lishi_shuzhi(legacy) is None
                                            or record["selection_method"] != "environment_96"))
                or any(type(rank) is not int or not 1 <= rank <= len(members) for rank in ranks)):
            raise ValueError("预测记录的未舍入估计、名次或候选代码无效")
        qualified_codes.append(code)
        identities[code].update({"return_rank": ranks[0], "observation_rank": ranks[1],
                                 "expected_next_day_return": float(primary),
                                 "legacy_24_expected_next_day_return": float(legacy) if legacy is not None else None})
    last_expected = last_rank = last_observation = None
    first_in_tie = 1
    for position, code in enumerate(qualified_codes, 1):
        item = identities[code]
        expected, rank, observation = (item[key] for key in ("expected_next_day_return", "return_rank", "observation_rank"))
        if expected != last_expected:
            first_in_tie = position
        if (last_expected is not None and expected > last_expected or rank != first_in_tie
                or observation not in (position, last_observation) or observation < rank
                or (rank != last_rank and observation != position)):
            raise ValueError("原始候选的预期涨幅、收益名次或观察顺序不一致")
        last_expected, last_rank, last_observation = expected, rank, observation
    if (shown != qualified_codes[:len(shown)] or len(shown) > 10 or bool(qualified_codes) != bool(shown)
            or any(not isinstance(code, str) for code in shown)):
        raise ValueError("当时展示对象与原始候选排序不一致")
    return signal, issued, identities


def duqu_ciri_huifang_jizhun(
    records: Iterable[Mapping[str, Any]], calendar: JiaoyiRili, *, start: pd.Timestamp, end: pd.Timestamp,
) -> dict[pd.Timestamp, dict[str, Any]]:
    """以原始预测中的当日已核验合格池作为 MACD 回放基线。"""
    baseline = {}
    for record in records:
        signal, issued, _ = _validate_record(record)
        if record.get("default_conditions") is not True:
            raise ValueError("MACD 基线需要默认条件的事前记录，避免基线预先筛入待检验的结构指标")
        if not start <= signal <= end or signal not in calendar.open_dates or signal in baseline:
            raise ValueError("MACD 基线预测须对应回放区间内互不重复的市场交易日")
        target = _next_session(signal, calendar)
        if target is None or issued > _timestamp(_beijing_now()) or issued >= target + pd.Timedelta(hours=9, minutes=15):
            raise ValueError("MACD 基线缺少有效的目标日或事前生成时点")
        if record.get("target_date") is not None and jiaoyi_riqi(record["target_date"]) != target:
            raise ValueError("MACD 基线预测目标日与本次核验日历冲突")
        baseline[signal] = {
            "codes": {item["ts_code"] for item in record["qualified_candidates"]},
            "issued_at": record["issued_at"], "selection_method": record["selection_method"],
            "comparison_complete_at_issue": record.get("comparison_complete") is True,
        }
    return baseline


def _endpoint_return(history: pd.DataFrame | None, signal: pd.Timestamp, target: pd.Timestamp) -> dict[str, Any]:
    missing = {"signal_close": None, "target_close": None, "realized_next_day_return": None}
    if history is None or history.empty or not {"trade_date", "close", "volume", "amount_yuan"}.issubset(history):
        return {**missing, "reason": "没有取得同源日线"}
    dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        return {**missing, "reason": "日线日期无效或重复"}
    values = history.assign(trade_date=dates).set_index("trade_date").reindex([signal, target])
    closes = [lishi_shuzhi(value) for value in values["close"]]
    volumes = [lishi_shuzhi(value) for value in values["volume"]]
    amounts = [lishi_shuzhi(value) for value in values["amount_yuan"]]
    valid = all(value is not None and value > 0 for value in closes + volumes + amounts)
    return {"signal_close": closes[0], "target_close": closes[1],
            "realized_next_day_return": closes[1] / closes[0] - 1 if valid else None,
            "reason": None if valid else "信号日或目标日缺少真实成交和有效收盘"}


def yanzheng_ciri_yuce(
    records: Iterable[Mapping[str, Any]], context: FenxiShujuShangxiawen,
) -> dict[str, Any]:
    """按记录逐日复核，每个信号日只评估一份原始双榜预测。"""
    records = list(records)
    if not records:
        raise ValueError("缺少原始双榜预测记录")
    parsed = [_validate_record(record) for record in records]
    if len({signal for signal, _, _ in parsed}) != len(parsed):
        raise ValueError("同一信号日只接受一份原始预测记录")
    calendar = context.jiaoyi_rili(start_date=min(item[0] for item in parsed).strftime("%Y-%m-%d"),
                                  end_date=(pd.Timestamp(context.reference.date()) + pd.Timedelta(days=14)).strftime("%Y-%m-%d"))
    completed = zuixin_wanzheng_jiaoyiri(context.reference, calendar=calendar)
    config, _ = jiazai_lianghua_peizhi()
    fallback_limit = int((config.get("wangluo") or {}).get("history_fallback_max_stocks", 16))
    dates = []
    for record, (signal, issued, identities) in sorted(zip(records, parsed), key=lambda item: item[1][0]):
        if signal not in calendar.open_dates or issued > _timestamp(context.reference):
            raise ValueError("预测记录的信号日或生成时点不能由本次开市日历核验")
        target = _next_session(signal, calendar)
        original_target = record.get("target_date")
        if original_target is not None and (target is None or jiaoyi_riqi(original_target) != target):
            raise ValueError("预测记录的原目标交易日与本次日历存在冲突")
        if target is not None and issued >= target + pd.Timedelta(hours=9, minutes=15):
            raise ValueError("存在目标交易日开始集合竞价后才生成的预测记录")
        day = {"signal_date": str(signal.date()), "target_date": str(target.date()) if target is not None else None,
               "issued_at": record["issued_at"], "selection_method": record["selection_method"],
               "scope_status_at_issue": record.get("scope_status"), "scope_source_at_issue": record.get("scope_source"),
               "comparison_complete_at_issue": record.get("comparison_complete") is True}
        if target is None or target > completed:
            dates.append({**day, "status": "calendar_unavailable" if target is None else "pending",
                          "pool_stocks": len(identities), "forecast_stocks": len(record["qualified_candidates"]),
                          "displayed_forecasts": len(record["displayed_codes"]),
                          "reason": "下一市场交易日尚未由日历确认" if target is None else "下一市场交易日尚未完成"})
            continue
        try:
            histories, metadata = context.piliang_lishi(tuple(identities), start_date=signal.strftime("%Y%m%d"),
                                                        end_date=target.strftime("%Y%m%d"), minimum_rows=1)
        except (RuntimeError, OSError, ValueError) as exc:
            histories, metadata = {}, {"status": "unavailable", "error": str(exc)}
        sources = metadata.get("source_by_code") or {}
        rows = []
        missing_ends = 0
        # 单股备用请求有界。优先核对实际展示的预测，其次核对同期其余合格候选。
        preferred = dict.fromkeys([*record["displayed_codes"], *(item["ts_code"] for item in record["qualified_candidates"]), *identities])
        for code in preferred:
            item = identities[code]
            observed = _endpoint_return(histories.get(code), signal, target)
            source = sources.get(code)
            failure = None
            if observed["realized_next_day_return"] is None and missing_ends < fallback_limit:
                missing_ends += 1
                try:
                    tail, tail_meta = context.dangu_qfq_jinri(code, start_date=signal.strftime("%Y%m%d"), end_date=target.strftime("%Y%m%d"))
                    alternative = _endpoint_return(tail, signal, target)
                    if alternative["realized_next_day_return"] is not None:
                        observed, source = alternative, tail_meta["source"]
                    else:
                        failure = alternative["reason"]
                except (RuntimeError, OSError, ValueError) as exc:
                    failure = str(exc)
            elif observed["realized_next_day_return"] is None:
                failure = "本日个股备用源已达到请求上限"
            realized = observed["realized_next_day_return"]
            primary = item.get("expected_next_day_return")
            legacy = item.get("legacy_24_expected_next_day_return")
            rows.append({**item, **observed, "displayed": code in record["displayed_codes"], "daily_source": source,
                         **({"fallback_failure": failure} if failure else {}),
                         "error": primary - realized if primary is not None and realized is not None else None,
                         "legacy_24_error": legacy - realized if legacy is not None and realized is not None else None})
        known = {row["ts_code"]: row["realized_next_day_return"] for row in rows if row["realized_next_day_return"] is not None}
        evaluated = [row for row in rows if row["error"] is not None]
        paired = [row for row in evaluated if row["legacy_24_error"] is not None]
        old_mae = mean(abs(row["legacy_24_error"]) for row in paired) if paired else None
        new_mae = mean(abs(row["error"]) for row in paired) if paired else None
        shown = record["displayed_codes"]
        top = mean(known[code] for code in shown) if shown and all(code in known for code in shown) else None
        known_mean = mean(known.values()) if known else None
        full_recorded_pool = known_mean if len(known) == len(identities) else None
        boundary = max((identities[code]["observation_rank"] for code in shown), default=None)
        boundary_codes = [item["ts_code"] for item in record["qualified_candidates"]
                          if boundary is not None and item["observation_rank"] <= boundary]
        ranked = mean(known[code] for code in boundary_codes) if boundary_codes and all(code in known for code in boundary_codes) else None
        paired_comparison = record["selection_method"] == "environment_96"
        dates.append({**day, "status": "ok" if len(evaluated) == len(record["qualified_candidates"]) and full_recorded_pool is not None
                      and (not paired_comparison or len(paired) == len(evaluated))
                      else "partial" if known else "unavailable",
                      "pool_stocks": len(identities), "realized_stocks": len(known),
                      "forecast_stocks": len(record["qualified_candidates"]), "matched_forecasts": len(evaluated),
                      "forecast_match_rate": len(evaluated) / len(record["qualified_candidates"]) if record["qualified_candidates"] else None,
                      "pool_observation_rate": len(known) / len(identities),
                      "mean_absolute_error": mean(abs(row["error"]) for row in evaluated) if evaluated else None,
                      "legacy_24_comparison": {"paired_stocks": len(paired), "environment_96_mae": new_mae, "legacy_24_mae": old_mae},
                      "displayed_forecasts": len(shown), "displayed_realized": sum(code in known for code in shown),
                      "displayed_mean_return": top, "top_ranked_codes_including_ties": boundary_codes,
                      "top_ranked_mean_return": ranked, "observable_pool_mean_return": known_mean,
                      "complete_recorded_pool_mean_return": full_recorded_pool,
                      "top_ranked_excess_vs_complete_recorded_pool": ranked - full_recorded_pool if ranked is not None and full_recorded_pool is not None else None,
                      "history_provenance": metadata, "stocks": rows})
    finished = [day for day in dates if "realized_stocks" in day]
    mae_by_method = {method: [day["mean_absolute_error"] for day in finished
                              if day["selection_method"] == method and day["mean_absolute_error"] is not None]
                     for method in ("environment_96", "legacy_24")}
    pairs = [day["legacy_24_comparison"] for day in finished if day["legacy_24_comparison"]["paired_stocks"]]
    excess = [day["top_ranked_excess_vs_complete_recorded_pool"] for day in finished if day["top_ranked_excess_vs_complete_recorded_pool"] is not None]
    forecast_stocks = sum(day["forecast_stocks"] for day in finished)
    matched_forecasts = sum(day["matched_forecasts"] for day in finished)
    return {
        "status": "pending" if all(day["status"] == "pending" for day in dates) else "ok" if all(day["status"] == "ok" for day in dates)
        else "partial" if any(day["realized_stocks"] for day in finished) else "unavailable",
        "analysis_type": "next_session_forecast_verification", "calendar_source": calendar.source,
        "price_basis": "same_source_qfq_close_to_close", "costs_applied": False, "persistence": "none",
        "summary": {"signal_dates": len(dates), "completed_dates": len(finished),
                    "forecast_stocks": forecast_stocks, "matched_forecasts": matched_forecasts,
                    "forecast_match_rate": matched_forecasts / forecast_stocks if forecast_stocks else None,
                    "date_equal_weight_mae_by_method": {key: {"dates": len(values), "mae": mean(values) if values else None}
                                                        for key, values in mae_by_method.items()},
                    "paired_comparison_dates": len(pairs),
                    "paired_environment_96_mae": mean(pair["environment_96_mae"] for pair in pairs) if pairs else None,
                    "paired_legacy_24_mae": mean(pair["legacy_24_mae"] for pair in pairs) if pairs else None,
                    "complete_recorded_pool_comparison_dates": len(excess),
                    "mean_top_ranked_excess_vs_complete_recorded_pool": mean(excess) if excess else None},
        "date_results": dates,
    }
