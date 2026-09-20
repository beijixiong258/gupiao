"""统一选股分析门面：编排候选池、因子、增强证据与最终排序。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ashare.buchong_zhenduan import goujian_buchong_zhenduan
from src.ashare.fanwei_faxian import shencha_fanwei_houxuan
from src.ashare.fenxi_weipan import WeipanJieduan, fenxi_weipan, panduan_weipan_jieduan
from src.ashare.fenxi_xingtai import fenxi_zhangting_huimaqiang
from src.ashare.dangu_fenxi import DANGU_ANALYSIS_TYPE, fenxi_dangu
from src.ashare.fenxi_yinzi import (
    goujian_fenxi_yinzi_mianban,
    huizong_houxuan_yinzi,
    jisuan_hengjiemian_jibenmian,
    zengjia_dangri_guzhi_yinzi,
)
from src.ashare.gupiao_yanjiu import zongjie_jishu
from src.ashare.peizhi import jiazai_lianghua_peizhi
from src.ashare.shichang_shuju import FenxiShujuShangxiawen
from src.ashare.shuju_yuan import biaozhunhua_gupiao_daima
from src.ashare.wangluo_kehu import WangluoQingqiuYichang
from src.ashare.xuangu_fanwei import (
    FanweiLeixing,
    FenxiFanwei,
    YijiexiFenxiFanwei,
    huoqu_houxuanchi_celue,
)
from src.ashare.xuangu_guize import (
    choushu_liudongxing_houxuan,
    goujian_houxuan_zhaiyao,
    goujian_kejiaoyixing_zhaiyao,
    goujian_kuaizhao_jilu,
    goujian_zhen_duan_shixiaoxing,
    guolv_lishi_wanzhengxing,
    hebing_jibenmian_zhengju,
    jichu_ying_guolv,
    paixu_shangzhang_houxuan,
    pinggu_shangzhang_tiaojian,
    shishi_ying_guolv,
    xuyao_shishi_kuaizhao,
    zhuan_json_zhi,
    zhuan_you_xian_shuzhi,
)
from src.ashare.xuangu_shizhi import guolv_shizhi, jiexi_shizhi_tiaojian


XUANGU_TOOL_CONTRACT_VERSION = 9


def _normalize_requested_count(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(5, count))


def xianzhi_xuangu_jieguo(result: dict[str, Any], requested_count: Any) -> dict[str, Any]:
    """按用户明确要求的数量裁剪公开候选。"""

    count = _normalize_requested_count(requested_count)
    if (
        count is None
        or result.get("status") not in {"ok", "partial"}
        or result.get("analysis_type") != "unified_stock_selection"
        or not result.get("recommendation_available")
    ):
        return result
    primary = result.get("primary")
    if not isinstance(primary, dict):
        return result
    alternatives = result.get("alternatives")
    alternatives = alternatives if isinstance(alternatives, list) else []
    selected = [primary, *[item for item in alternatives if isinstance(item, dict)][: max(0, count - 1)]]
    limited = dict(result)
    limited["alternatives"] = selected[1:]
    limited["requested_candidate_count"] = count
    limited["displayed_candidate_count"] = len(selected)
    if isinstance(result.get("candidate_counts"), dict):
        limited["candidate_counts"] = {**result["candidate_counts"], "displayed": len(selected)}
    return limited


class XuanguFenxiFuWu:
    """统一分析门面：范围策略可变，其余流程保持单一实现。"""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        context: FenxiShujuShangxiawen,
    ) -> None:
        self._config = config
        self._context = context

    def fenxi(
        self, fanwei: FenxiFanwei, *, requested_count: int | None = None,
        market_cap_condition: dict[str, Any] | None = None,
        scope_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if fanwei.leixing is FanweiLeixing.DANGU_GUPIAO:
            return fenxi_dangu(
                gupiao=str(fanwei.gupiao or ""),
                config=self._config,
                context=self._context,
            )
        settings = self._config["fenxi"]
        requested_count = _normalize_requested_count(requested_count)
        display_target = min(requested_count or 1 + int(settings["backup_limit"]),
                             1 + int(settings["backup_limit"]), int(settings["deep_analysis_limit"]))
        if fanwei.leixing is FanweiLeixing.MINGMING_FANWEI:
            discovery = self._context.faxian_fanwei(str(fanwei.mingcheng or ""))
            if scope_choice is not None:
                discovery = shencha_fanwei_houxuan(discovery, scope_choice)
            if discovery.status != "resolved" or discovery.scope is None:
                return {
                    **discovery.to_result(),
                    "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                    "analysis_type": "unified_stock_selection",
                    "recommendation_available": False,
                    "primary": None,
                    "alternatives": [],
                    "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
                }
            resolved_scope = YijiexiFenxiFanwei(
                request=fanwei,
                market_scope=discovery.scope,
            )
        else:
            resolved_scope = YijiexiFenxiFanwei(request=fanwei)
        pool = huoqu_houxuanchi_celue(resolved_scope).goujian(
            resolved_scope,
            self._context,
        )
        if pool.data.empty:
            return {
                "status": "unavailable",
                "outcome": "data_unavailable",
                "error_code": "scope_constituents_empty",
                "stage": "candidate_pool",
                "source": pool.metadata.get("constituent_source") or pool.metadata.get("source"),
                "retryable": True,
                "error": "实时范围成分中没有可分析的 A 股候选",
                "next_action": "稍后重试，程序不会使用旧本地成分数据",
                "scope": pool.metadata,
                "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                "recommendation_available": False,
                "primary": None,
                "alternatives": [],
            }
        data = pool.data.copy()
        data["ts_code"] = data["ts_code"].astype(str).map(biaozhunhua_gupiao_daima)
        data = data.drop_duplicates("ts_code", keep="first").reset_index(drop=True)
        analysis_date = self._context.zuixin_wanzheng_jiaoyiri()
        quote_is_completed = pool.metadata.get("source") == "tushare_daily_cross_section"
        rejected: list[dict[str, Any]] = []
        accepted_indices: list[int] = []
        for index, row in data.iterrows():
            reasons = jichu_ying_guolv(
                row,
                analysis_date=analysis_date,
                config=self._config,
                quote_is_completed=quote_is_completed,
            )
            if reasons:
                required_fields = ("latest_price", "volume", "amount_yuan") if quote_is_completed else ("latest_price",)
                missing_fields = [field for field in required_fields if zhuan_you_xian_shuzhi(row.get(field)) is None]
                rejected.append({"ts_code": row.get("ts_code"), "name": row.get("name"), "reasons": reasons,
                                 "missing_fields": missing_fields})
            else:
                accepted_indices.append(index)
        filtered = data.loc[accepted_indices].reset_index(drop=True)
        base_filtered_count = len(filtered)
        cap_summary: dict[str, Any] = {"status": "not_requested"}
        if market_cap_condition is not None and not filtered.empty:
            cap_metadata: dict[str, Any] = {}
            cap_error = None
            try:
                valuation = self._context.rizhong_guzhi(
                    analysis_date.strftime("%Y-%m-%d"), metadata=cap_metadata,
                )
            except Exception as exc:
                valuation, cap_error = pd.DataFrame(), str(exc)
            filtered, cap_summary = guolv_shizhi(
                filtered, market_cap_condition, valuation, analysis_date=analysis_date,
                fetched_at=cap_metadata.get("fetched_at"), source_error=cap_error,
            )
        elif market_cap_condition is not None:
            cap_summary = {"status": "not_evaluated", "condition": market_cap_condition,
                           "reason": "没有通过基础检查的候选"}
        if filtered.empty:
            incomplete = any(item["missing_fields"] for item in rejected) or bool(cap_summary.get("unavailable_count"))
            return {
                "status": "partial" if incomplete else "ok",
                "outcome": "information_partial" if incomplete else "no_recommendation",
                "selection_outcome": "evidence_unavailable" if incomplete else "no_recommendation",
                "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                "analysis_type": "unified_stock_selection",
                "scope": pool.metadata,
                "as_of": analysis_date.strftime("%Y-%m-%d"),
                "recommendation_available": False,
                "no_recommendation_reason": (
                    "候选的基础价量或市值证据存在缺失，暂时无法完整确认是否合格"
                    if incomplete else "通过基础检查的候选均未满足指定市值条件"
                    if market_cap_condition and base_filtered_count else "全部候选都触发了风险硬过滤"
                ),
                "primary": None,
                "alternatives": [],
                "displayed_candidate_count": 0,
                "market_cap_filter": cap_summary,
                "candidate_counts": {"scope_input": len(data), "after_hard_filter": base_filtered_count,
                                     "after_market_cap_filter": 0, "market_cap_unverified": cap_summary.get("unavailable_count", 0),
                                     "after_prefilter": 0,
                                     "technical_reviewed": 0, "deep_reviewed": 0, "qualified": 0, "displayed": 0},
                "filter_summary": {"input_count": len(data), "rejected_count": len(rejected), "examples": rejected[:20]},
            }
        prefiltered = choushu_liudongxing_houxuan(filtered, int(settings["prefilter_limit"]))
        start_date = (analysis_date - pd.Timedelta(days=int(settings["history_calendar_days"]))).strftime("%Y%m%d")
        end_date = analysis_date.strftime("%Y%m%d")
        histories, history_meta = self._context.piliang_lishi(
            prefiltered["ts_code"].tolist(),
            start_date=start_date,
            end_date=end_date,
            minimum_rows=int(settings["minimum_history_rows"]),
        )
        history_loaded_count = len(histories)
        histories, history_rejected = guolv_lishi_wanzhengxing(
            histories,
            analysis_date=analysis_date,
            minimum_rows=int(settings["minimum_history_rows"]),
            minimum_amount=float(settings["min_amount_yuan"]),
            minimum_listing_calendar_days=int(settings["minimum_listing_calendar_days"]),
        )
        rejected.extend(history_rejected)
        if not histories:
            all_unmet = bool(history_rejected) and history_loaded_count == len(prefiltered) and all(
                item.get("status") == "unmet" for item in history_rejected
            )
            if all_unmet:
                incomplete = bool(cap_summary.get("unavailable_count")) or any(item.get("missing_fields") for item in rejected)
                return {
                    "status": "partial" if incomplete else "ok",
                    "outcome": "information_partial" if incomplete else "no_recommendation",
                    "selection_outcome": "evidence_unavailable" if incomplete else "no_recommendation",
                    "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION, "analysis_type": "unified_stock_selection",
                    "scope": pool.metadata, "as_of": analysis_date.strftime("%Y-%m-%d"),
                    "recommendation_available": False, "primary": None, "alternatives": [],
                    "displayed_candidate_count": 0,
                    "no_recommendation_reason": (
                        "已复核日线未通过条件，另有候选的基础价量或市值尚无法核验"
                        if incomplete else "本次已取得的完整日线均未通过流动性条件，没有合格候选"
                    ),
                    "market_cap_filter": cap_summary,
                    "candidate_counts": {"scope_input": len(data), "after_hard_filter": base_filtered_count,
                                         "after_market_cap_filter": len(filtered), "market_cap_unverified": cap_summary.get("unavailable_count", 0),
                                         "after_prefilter": len(prefiltered), "history_ready": 0,
                                         "technical_reviewed": 0, "deep_reviewed": 0, "qualified": 0, "displayed": 0},
                    "data_provenance": {"history": history_meta},
                    "filter_summary": {"rejected_count": len(rejected), "rejected_examples": rejected[:20]},
                }
            return {
                "status": "unavailable",
                "outcome": "data_unavailable",
                "error_code": "candidate_history_insufficient",
                "stage": "history_data",
                "source": history_meta.get("source"),
                "retryable": True,
                "next_action": "稍后重新获取完整远端日线",
                "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                "analysis_type": "unified_stock_selection",
                "recommendation_available": False, "primary": None, "alternatives": [],
                "scope": pool.metadata,
                "as_of": analysis_date.strftime("%Y-%m-%d"),
                "market_cap_filter": cap_summary,
                "error": "硬过滤后没有具备足够完整日线的候选",
                "data_provenance": {"history": history_meta},
                "filter_summary": {"input_count": len(data), "rejected_count": len(rejected), "examples": rejected[:20]},
            }
        ready_profiles = prefiltered[prefiltered["ts_code"].isin(histories)].reset_index(drop=True)
        panel, panel_meta = goujian_fenxi_yinzi_mianban(
            histories,
            ready_profiles,
            source="auto",
            calendar=self._context.jiaoyi_rili(start_date=start_date, end_date=end_date),
        )
        if panel.empty:
            raise RuntimeError("候选日 K 因子面板为空")
        panel_dates = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
        latest_panel = panel.loc[panel_dates.eq(analysis_date.normalize())].copy()
        latest_panel = latest_panel[latest_panel["ts_code"].astype(str).isin(histories)].reset_index(drop=True)
        if latest_panel.empty:
            raise RuntimeError("候选因子面板没有与分析日一致的完整日线")
        latest_panel = zengjia_dangri_guzhi_yinzi(latest_panel, ready_profiles)
        factor_map = huizong_houxuan_yinzi(latest_panel, config=settings)
        profile_subset = ready_profiles[ready_profiles["ts_code"].isin(factor_map)].copy().reset_index(drop=True)
        fundamental_map = jisuan_hengjiemian_jibenmian(profile_subset)
        clock = self._context.shichang_shizhong()
        late_stage = panduan_weipan_jieduan(clock, self._config["weipan"])
        realtime_table = pd.DataFrame()
        realtime_meta: dict[str, Any] = {"status": "not_requested", "source": None}
        realtime_required = xuyao_shishi_kuaizhao(clock, late_stage)
        if realtime_required:
            has_scope_snapshot = bool(
                pool.metadata.get("constituent_source")
                and {"ts_code", "latest_price", "amount_yuan"}.issubset(profile_subset.columns)
            )
            if has_scope_snapshot:
                realtime_table = profile_subset.copy()
                realtime_meta = {
                    "status": "ok",
                    "source": pool.metadata.get("constituent_source"),
                    "captured_at": pool.metadata.get("constituent_fetched_at"),
                    "rows": int(len(realtime_table)),
                    "timeliness": "命名范围成分接口与候选池同次获取，未重复下载全市场快照",
                    "persistence": "none",
                }
            else:
                try:
                    realtime_table, realtime_meta = self._context.shishi_kuaizhao()
                except Exception as exc:
                    realtime_table, realtime_meta = pd.DataFrame(), {
                        "status": "unavailable", "source": "remote_realtime_snapshot", "error": str(exc),
                    }
            if not realtime_table.empty:
                realtime_table = realtime_table.set_index("ts_code", drop=False)
        completed_quote_is_current = bool(
            quote_is_completed
            and analysis_date.normalize() == pd.Timestamp(self._context.reference.date())
        )
        profiles_by_code = profile_subset.set_index("ts_code", drop=False)
        preliminary: list[dict[str, Any]] = []
        factor_limit = int(settings["factor_candidate_limit"])
        factor_order = sorted(factor_map)
        for code in factor_order:
            if code not in profiles_by_code.index or code not in histories:
                continue
            profile_row = profiles_by_code.loc[code]
            if isinstance(profile_row, pd.DataFrame):
                profile_row = profile_row.iloc[0]
            name = str(profile_row.get("name") or "")
            if not realtime_table.empty and code in realtime_table.index:
                realtime_row = realtime_table.loc[code]
                if isinstance(realtime_row, pd.DataFrame):
                    realtime_row = realtime_row.iloc[0]
                snapshot = goujian_kuaizhao_jilu(realtime_row, realtime_meta)
            elif realtime_required and not (
                late_stage is WeipanJieduan.SHOUPAN_FUHE and completed_quote_is_current
            ):
                snapshot = {
                    "status": "unavailable",
                    "source": realtime_meta.get("source"),
                    "captured_at": realtime_meta.get("captured_at") or clock.get("captured_at"),
                    "error": str(
                        realtime_meta.get("error")
                        or f"全市场实时快照未找到 {code}"
                    ),
                    "completed_daily_reference": goujian_kuaizhao_jilu(profile_row, pool.metadata),
                }
            else:
                snapshot = goujian_kuaizhao_jilu(profile_row, pool.metadata)
            quote_check = goujian_kejiaoyixing_zhaiyao(
                code=code, name=name, snapshot=snapshot, history=histories[code],
                minimum_amount=float(settings["min_amount_yuan"]),
                realtime_required=str(clock.get("session_status")) in {"opening_auction", "trading", "midday_break", "close_pending"},
                reference_time=self._context.reference,
                market_clock=clock,
            )
            realtime_blocks = shishi_ying_guolv(
                snapshot if quote_check["current_quote_verified"] or quote_check["historical_quote_verified"] else {},
                code=code,
                name=name,
                config=self._config,
            )
            if realtime_blocks:
                rejected.append(
                    {
                        "ts_code": code,
                        "name": name,
                        "reasons": realtime_blocks,
                    }
                )
                continue
            pattern = fenxi_zhangting_huimaqiang(
                histories[code],
                code=code,
                name=name,
                config=self._config["xingtai"],
                realtime_quote=snapshot if str(clock.get("session_status")) in {"trading", "midday_break"} and quote_check["current_quote_verified"] else None,
            )
            late = fenxi_weipan(
                histories[code],
                snapshot=snapshot,
                clock=clock,
                config=self._config["weipan"],
            )
            factor = factor_map[code]
            fundamental = fundamental_map.get(code, {"status": "unavailable", "evidence": []})
            item = {
                "ts_code": code,
                "name": name,
                "industry": str(profile_row.get("industry") or ""),
                "profile": {key: zhuan_json_zhi(value) for key, value in profile_row.to_dict().items()},
                "history": histories[code],
                "snapshot": snapshot,
                "factor": factor,
                "fundamental": fundamental,
                "pattern": pattern,
                "late": late,
                "technical": {},
                "tradability": quote_check,
                "data_quality": {
                    "history_source": history_meta.get("source"),
                    "history_rows": int(len(histories[code])),
                    "as_of": analysis_date.strftime("%Y-%m-%d"),
                    "realtime_source": snapshot.get("source"),
                },
            }
            preliminary.append(item)
        preliminary = paixu_shangzhang_houxuan(preliminary, config=self._config)[:factor_limit]
        # 本地技术复核可继续检查后续候选；远端财务和分钟请求仍受原上限约束。
        # 已明确未满足基础条件者不消耗补充来源请求，仅在没有合格者时保留一个观察报告。
        review_pool = [item for item in preliminary if item["selection"]["eligible"]]
        if not review_pool:
            review_pool = preliminary[:1]
        technical_reviewed: list[dict[str, Any]] = []
        deep_candidates: list[dict[str, Any]] = []
        review_stop_reason = "candidate_pool_exhausted"
        for item in review_pool:
            try:
                technical = zongjie_jishu(
                    item["history"],
                    macd_structure_config=settings.get("macd_structure"),
                )
                if technical.get("status") == "error":
                    item["technical"] = {
                        "status": "error",
                        "outcome": "program_error",
                        "error": str(
                            technical.get("reason")
                            or technical.get("error")
                            or (technical.get("macd_structure") or {}).get("reason")
                            or "技术结构研判发生程序错误"
                        ),
                    }
                else:
                    item["technical"] = technical
            except Exception as exc:
                item["technical"] = {
                    "status": "error",
                    "outcome": "program_error",
                    "error": str(exc),
                }
            item["selection"] = pinggu_shangzhang_tiaojian(item, config=self._config, deep_reviewed=True)
            technical_reviewed.append(item)
            if item["selection"]["outcome"] == "program_error":
                review_stop_reason = "program_error"
                break
            if item["selection"]["eligible"]:
                deep_candidates.append(item)
                if len(deep_candidates) >= display_target:
                    review_stop_reason = "display_target_reached"
                    break
        if not deep_candidates and technical_reviewed:
            deep_candidates = technical_reviewed[:1]
        technical_errors = [
            {"ts_code": item["ts_code"], "reason": item["technical"].get("error") or item["technical"].get("reason")}
            for item in technical_reviewed if item["selection"]["outcome"] == "program_error"
        ]
        for index, item in enumerate(deep_candidates):
            if not technical_errors:
                try:
                    fundamentals = self._context.jibenmian(
                        item["ts_code"],
                        trade_date=analysis_date.strftime("%Y-%m-%d"),
                        allow_current_snapshot=bool(
                            late_stage is WeipanJieduan.SHOUPAN_FUHE and completed_quote_is_current
                        ),
                    )
                except Exception as exc:
                    fundamentals = {"profile": item["profile"], "financials": {}, "valuation": {}, "errors": [str(exc)]}
                if not fundamentals.get("profile"):
                    fundamentals["profile"] = item["profile"]
                item["fundamental"] = hebing_jibenmian_zhengju(item["fundamental"], fundamentals)
                if late_stage is WeipanJieduan.PANZHONG_ZANDING and index < int(self._config["weipan"]["minute_candidate_limit"]):
                    try:
                        minute_data, minute_meta = self._context.fenzhong_xingqing(item["ts_code"])
                    except Exception as exc:
                        minute_data, minute_meta = pd.DataFrame(), {"status": "unavailable", "error": str(exc)}
                    item["late"] = fenxi_weipan(
                        item["history"], snapshot=item["snapshot"], clock=clock,
                        config=self._config["weipan"], minute_data=minute_data,
                    )
                    item["late"]["minute_data"] = minute_meta
            item["supplemental_diagnostics"] = goujian_buchong_zhenduan(
                item["history"],
                item["fundamental"],
                as_of_date=analysis_date.strftime("%Y-%m-%d"),
                minimum_amount_yuan=float(settings["min_amount_yuan"]),
            )
            item["tradability"] = goujian_kejiaoyixing_zhaiyao(
                code=item["ts_code"],
                name=item["name"],
                snapshot=item["snapshot"],
                history=item["history"],
                minimum_amount=float(settings["min_amount_yuan"]),
                realtime_required=str(clock.get("session_status")) in {
                    "opening_auction", "trading", "midday_break", "close_pending",
                },
                reference_time=self._context.reference,
                market_clock=clock,
            )
        deep_candidates = paixu_shangzhang_houxuan(deep_candidates, config=self._config, deep_reviewed=True)
        summaries = [
            goujian_houxuan_zhaiyao(
                item,
                rank=index,
            )
            for index, item in enumerate(deep_candidates, start=1)
        ]
        if market_cap_condition is not None:
            for summary, item in zip(summaries, deep_candidates):
                summary["market_cap_check"] = item["profile"]["market_cap_check"]
        qualified = [] if technical_errors else [item for item in summaries if item["meets_selection_conditions"]]
        primary = qualified[0] if qualified else None
        alternatives = qualified[1 : 1 + int(settings["backup_limit"])]
        recommendation_available = primary is not None
        reviewed_codes = {item["ts_code"] for item in technical_reviewed}
        condition_checks = [
            {
                "ts_code": item["ts_code"], "name": item["name"],
                "technical_reviewed": item["ts_code"] in reviewed_codes,
                "missing_conditions": item["selection"]["missing_conditions"],
                "unmet_conditions": item["selection"]["unmet_conditions"],
                "technical_review": {
                    "status": item["technical"].get("status", "not_requested"),
                    "outcome": item["technical"].get("outcome"),
                    "reason": item["technical"].get("error") or item["technical"].get("reason"),
                    "missing_indicators": item["technical"].get("missing_indicators", []),
                },
            }
            for item in preliminary
        ]
        undecided_count = sum(bool(item["missing_conditions"]) and not item["unmet_conditions"] for item in condition_checks)
        source_partial = (
            bool(cap_summary.get("unavailable_count"))
            or any(item.get("missing_fields") for item in rejected)
            or any(item.get("status") == "unavailable" for item in history_rejected)
            or history_loaded_count < len(prefiltered) or len(factor_map) < len(histories)
            or any(metadata.get("status") in {"partial", "unavailable", "degraded"}
                   for metadata in (pool.metadata, history_meta, panel_meta, panel_meta.get("market_benchmarks") or {}))
        )
        evidence_partial = bool(
            source_partial or any(item["missing_conditions"] for item in condition_checks)
            or any(item["evidence_gaps"] or item["technical_summary"].get("status") == "partial"
                   or item["late_session_analysis"].get("status") in {"partial", "unavailable"}
                   or (item["late_session_analysis"].get("minute_data") or {}).get("status") == "unavailable"
                   for item in summaries)
        )
        selection_outcome = (
            "program_error" if technical_errors else "recommendation" if recommendation_available
            else "evidence_unavailable" if undecided_count or source_partial else "no_recommendation"
        )
        result = {
            "status": "error" if technical_errors else "partial" if evidence_partial else "ok",
            "outcome": "program_error" if technical_errors else "information_partial" if evidence_partial else selection_outcome,
            "selection_outcome": selection_outcome,
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": "unified_stock_selection",
            "scope": pool.metadata,
            "market_cap_filter": cap_summary,
            "as_of": analysis_date.strftime("%Y-%m-%d"),
            "generated_at": self._context.reference.strftime("%Y-%m-%d %H:%M:%S"),
            "market_clock": clock,
            "diagnosis_validity": goujian_zhen_duan_shixiaoxing(
                as_of=analysis_date.strftime("%Y-%m-%d"),
                generated_at=self._context.reference.strftime("%Y-%m-%d %H:%M:%S"),
                clock=clock,
                realtime_status=(
                    "verified" if summaries and all(
                        (item.get("tradability") or {}).get("current_quote_verified")
                        for item in summaries
                    ) else "unavailable"
                ),
                realtime_required=str(clock.get("session_status")) in {
                    "opening_auction", "trading", "midday_break", "close_pending",
                },
            ),
            "result_confirmation": (
                "intraday_provisional"
                if str(clock.get("session_status")) in {"opening_auction", "trading", "midday_break"}
                else "close_pending"
                if str(clock.get("session_status")) == "close_pending"
                else "completed_daily_close"
            ),
            "recommendation_available": recommendation_available,
            "primary": primary,
            "alternatives": alternatives,
            "reviewed_candidates": summaries,
            "diagnostic_candidates": [
                {**summaries[0], "diagnostic_role": "observation_only", "diagnostic_label": "观察对象（未通过选股条件）"}
            ] if summaries and not recommendation_available and not technical_errors else [],
            "no_recommendation_reason": (
                None
                if recommendation_available
                else "技术复核发生程序错误，当前诊断未完整完成"
                if technical_errors
                else "部分候选的必需证据尚未取得或核验，当前无法完整确认是否合格；不能将数据不足解释为全部股票不满足条件"
                if selection_outcome == "evidence_unavailable"
                else "当前样本中没有股票同时满足趋势、动量、相对强弱、量能、波动和可交易性条件；具体缺项与反证已保留"
            ),
            "selection_limits": {
                "maximum_alternatives": int(settings["backup_limit"]),
                "maximum_full_reports": int(settings["deep_analysis_limit"]),
                "maximum_technical_reviews": factor_limit,
                "maximum_minute_requests": int(self._config["weipan"]["minute_candidate_limit"]),
                "display_target": display_target,
                "review_stop_reason": review_stop_reason,
            },
            "displayed_candidate_count": int(bool(primary)) + len(alternatives),
            "candidate_condition_checks": condition_checks,
            "candidate_counts": {
                "scope_input": int(len(data)),
                "after_hard_filter": base_filtered_count,
                "after_market_cap_filter": int(len(filtered)),
                "market_cap_unverified": cap_summary.get("unavailable_count", 0),
                "after_prefilter": int(len(prefiltered)),
                "history_ready": int(len(histories)),
                "factor_ready": int(len(factor_map)),
                "after_factor_limit": int(len(preliminary)),
                "technical_reviewed": int(len(technical_reviewed)),
                "deep_reviewed": int(len(deep_candidates)),
                "qualified": int(len(qualified)),
                "unverified": int(undecided_count),
                "displayed": int(bool(primary)) + len(alternatives),
            },
            "filter_summary": {
                "rejected_count": int(len(rejected)),
                "rejected_examples": rejected[:20],
                "rules": [
                    "ST、退市风险和不稳定新股不进入排序",
                    "无有效价格、成交量、成交额或完整分析日的股票不进入排序",
                    "一字涨停、历史不足和低于流动性底线的股票不进入排序",
                ],
            },
            "data_provenance": {
                "scope": pool.metadata,
                "history": history_meta,
                "factor_panel": panel_meta,
                "realtime_snapshot": realtime_meta,
            },
            "selection_methodology": {
                "method": "explicit_conditions_then_pareto_fronts",
                "conditions": "价格和短均线高于MA20、5/20日收益为正且MACD柱为正、20日跑赢沪深300、5/20日量比不低于1、波动未超过上限、深度复核与可交易性通过",
                "comparison_dimensions": ["20日相对沪深300超额", "5日收益", "20日收益", "20日波动（越低越好）", "完整日线真实成交额"],
                "ranking_basis": "满足条件后按多维不劣且至少一维更优的关系分层；同层按成交额和代码稳定展示，不代表上涨概率高低",
                "sampling_limit": "只比较本次抽样取得并完成复核的候选；本地技术复核顺序补位，达到展示目标或候选上限即停止，完整报告和分钟请求各有上限，不等于已经遍历全市场所有机会",
                "missing_evidence": "必需条件缺失不能视为通过，保留原因供复核",
                "llm_boundary": "解释实际指标与证据冲突，不生成分数、权重或上涨概率，不将候选排序说成经验证的概率排名",
            },
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
        }
        if technical_errors:
            result.update({
                "error_code": "candidate_technical_review_error",
                "stage": "technical_analysis",
                "error": "技术复核发生程序错误；已保留候选证据供排查，本次不发布筛选候选",
                "technical_review_errors": technical_errors,
                "next_action": "修复技术复核错误后重新获取数据并诊断",
                "retryable": False,
            })
        if requested_count is not None:
            result["requested_candidate_count"] = requested_count
        return result


def fenxi_xuangu(
    *,
    fanwei: str = "all_market",
    mingcheng: str | None = None,
    gupiao: str | None = None,
    shuliang: int | str | None = None,
    shizhi: dict[str, Any] | None = None,
    context: FenxiShujuShangxiawen | None = None,
    scope_choice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """自然语言智能体调用的统一分析入口。"""
    analysis_type = DANGU_ANALYSIS_TYPE if str(fanwei or "").strip().lower() in {
        "single_stock",
        "single",
        "stock",
        "dangu",
        "单股",
        "个股",
    } or (gupiao and str(fanwei or "all_market").strip().lower() in {"", "all_market", "all", "quan_shichang", "全市场"}) else "unified_stock_selection"
    try:
        resolved_scope = FenxiFanwei.create(fanwei, mingcheng, gupiao)
        market_cap_condition = None
        if resolved_scope.leixing is not FanweiLeixing.DANGU_GUPIAO:
            try:
                market_cap_condition = jiexi_shizhi_tiaojian(shizhi)
            except ValueError as exc:
                question = str(exc)
                return {
                    "status": "clarification_required", "outcome": "clarification_required",
                    "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                    "analysis_type": analysis_type, "stage": "request_validation",
                    "error_code": "market_cap_condition_required", "error": question,
                    "clarification": {"kind": "market_cap", "title": "确认市值条件", "question": question},
                    "requested_candidate_count": _normalize_requested_count(shuliang),
                    "recommendation_available": False, "primary": None, "alternatives": [],
                    "displayed_candidate_count": 0,
                }
        config, _ = jiazai_lianghua_peizhi()
        request_context = context or FenxiShujuShangxiawen()
        result = XuanguFenxiFuWu(config=config, context=request_context).fenxi(
            resolved_scope, requested_count=_normalize_requested_count(shuliang),
            market_cap_condition=market_cap_condition,
            scope_choice=scope_choice,
        )
        return xianzhi_xuangu_jieguo(result, shuliang)
    except WangluoQingqiuYichang as exc:
        return {
            "status": "unavailable",
            "outcome": "data_unavailable",
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": analysis_type,
            "error_code": exc.error_code,
            "stage": "data_acquisition",
            "source": "remote_market_data",
            "retryable": exc.retryable,
            "attempted_providers": [item.to_dict() for item in exc.attempts],
            "error": "远端市场数据当前不可用，无法形成可靠的量化结论",
            "next_action": "稍后重试；程序不会读取旧本地市场数据代替实时结果",
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
        }
    except ValueError as exc:
        return {
            "status": "clarification_required",
            "outcome": "clarification_required",
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": analysis_type,
            "error_code": "analysis_request_invalid",
            "stage": "request_validation",
            "source": None,
            "retryable": False,
            "error": " ".join(str(exc).split())[:240],
            "next_action": "请用日常语言补充要分析的股票范围",
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
        }
    except RuntimeError as exc:
        return {
            "status": "unavailable",
            "outcome": "data_unavailable",
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": analysis_type,
            "error_code": "analysis_data_unavailable",
            "stage": "quantitative_data_pipeline",
            "source": "remote_market_data",
            "retryable": True,
            "error": " ".join(str(exc).split())[:240],
            "next_action": "稍后重新获取远端数据；不使用本地旧数据降级",
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
        }
    except Exception as exc:
        return {
            "status": "error",
            "outcome": "error",
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": analysis_type,
            "error_code": "analysis_internal_error",
            "stage": "quantitative_analysis",
            "source": None,
            "retryable": False,
            "error": f"量化分析程序发生内部错误：{' '.join(str(exc).split())[:180]}",
            "next_action": "请记录运行编号并检查程序日志",
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
        }


__all__ = [
    "FanweiLeixing",
    "FenxiFanwei",
    "XuanguFenxiFuWu",
    "XUANGU_TOOL_CONTRACT_VERSION",
    "fenxi_xuangu",
    "xianzhi_xuangu_jieguo",
]
