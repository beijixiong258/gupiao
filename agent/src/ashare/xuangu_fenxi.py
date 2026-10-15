"""统一选股分析门面：编排候选池、因子、增强证据与最终排序。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ashare.buchong_zhenduan import goujian_buchong_zhenduan
from src.ashare.fanwei_faxian import shencha_fanwei_houxuan
from src.ashare.fenxi_weipan import WeipanJieduan, fenxi_weipan, panduan_weipan_jieduan
from src.ashare.fenxi_xingtai import fenxi_zhangting_huimaqiang
from src.ashare.fenxi_wenti import jiexi_xuangu_tiaojian, jiexi_yanjiu_wenti, pinggu_yanjiu_wenti
from src.ashare.dangu_fenxi import DANGU_ANALYSIS_TYPE, fenxi_dangu
from src.ashare.fenxi_yinzi import (
    FenxiJisuanYichang,
    goujian_fenxi_yinzi_mianban,
    huizong_houxuan_yinzi,
    jisuan_hengjiemian_jibenmian,
    zengjia_dangri_guzhi_yinzi,
)
from src.ashare.peizhi import jiazai_lianghua_peizhi
from src.ashare.shichang_shuju import FenxiShujuShangxiawen
from src.ashare.shuju_yuan import biaozhunhua_gupiao_daima
from src.ashare.tongdaxin_chicang import fenxi_kongpan_shuju
from src.ashare.wangluo_kehu import WangluoQingqiuYichang
from src.ashare.xuangu_fanwei import (
    FanweiLeixing,
    FenxiFanwei,
    YijiexiFenxiFanwei,
    huoqu_houxuanchi_celue,
)
from src.ashare.xuangu_guize import (
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
        user_conditions: list[dict[str, Any]] | None = None,
        research_question: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if fanwei.leixing is FanweiLeixing.DANGU_GUPIAO:
            return fenxi_dangu(
                gupiao=str(fanwei.gupiao or ""),
                config=self._config,
                context=self._context,
                research_question=research_question,
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
                "selection_outcome": "evidence_unavailable",
                "error_code": pool.metadata.get("error_code") or "scope_constituents_empty",
                "stage": "candidate_pool",
                "source": pool.metadata.get("scope_source") or pool.metadata.get("constituent_source") or pool.metadata.get("source"),
                "retryable": pool.metadata.get("retryable", True),
                "error": pool.metadata.get("error") or "实时范围成分中没有可分析的 A 股候选",
                "next_action": "稍后重试，程序不会使用旧本地成分数据",
                "scope": pool.metadata,
                "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                "analysis_type": "unified_stock_selection",
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
            except FenxiJisuanYichang:
                raise
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
                    if market_cap_condition and base_filtered_count else "全部候选均未通过数据或基础成交核验"
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
        # 尽早获取本次快照，避免全范围历史计算后才取盘中证据，偏离请求时点。
        clock = self._context.shichang_shizhong()
        late_stage = panduan_weipan_jieduan(clock, self._config["weipan"])
        realtime_table = pd.DataFrame()
        realtime_meta: dict[str, Any] = {"status": "not_requested", "source": None}
        realtime_required = xuyao_shishi_kuaizhao(clock, late_stage)
        if realtime_required:
            has_scope_snapshot = bool(
                pool.metadata.get("constituent_source")
                and {"ts_code", "latest_price", "amount_yuan"}.issubset(filtered.columns)
            )
            if has_scope_snapshot:
                realtime_table = filtered.copy()
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
                except FenxiJisuanYichang:
                    raise
                except Exception as exc:
                    realtime_table, realtime_meta = pd.DataFrame(), {
                        "status": "unavailable", "source": "remote_realtime_snapshot", "error": str(exc),
                    }
            if not realtime_table.empty:
                realtime_table = realtime_table.set_index("ts_code", drop=False)
        # 批大小只控制一次取数规模，不再按成交额抽样或截断范围。
        prefiltered = filtered.sort_values("ts_code", kind="stable").reset_index(drop=True)
        start_date = (analysis_date - pd.Timedelta(days=int(settings["history_calendar_days"]))).strftime("%Y%m%d")
        end_date = analysis_date.strftime("%Y%m%d")
        histories: dict[str, pd.DataFrame] = {}
        history_batches: list[dict[str, Any]] = []
        batch_size = int(settings.get("history_batch_size", settings.get("prefilter_limit", 240)))
        codes = prefiltered["ts_code"].tolist()
        for offset in range(0, len(codes), batch_size):
            batch_codes = codes[offset:offset + batch_size]
            try:
                batch, metadata = self._context.piliang_lishi(
                    batch_codes, start_date=start_date, end_date=end_date, minimum_rows=21,
                )
                histories.update({code: frame for code, frame in batch.items() if code in batch_codes})
            except FenxiJisuanYichang:
                raise
            except Exception as exc:
                metadata = {"status": "unavailable", "error": str(exc), "requested_stocks": len(batch_codes), "loaded_stocks": 0}
            history_batches.append(metadata)
        history_meta = {
            "status": "ok" if len(histories) == len(codes) else "partial" if histories else "unavailable",
            "source": "+".join(dict.fromkeys(str(batch["source"]) for batch in history_batches if batch.get("source"))),
            "requested_stocks": len(codes), "loaded_stocks": len(histories), "batch_size": batch_size,
            "batches": history_batches, "persistence": "none",
        }
        history_loaded_count = len(histories)
        histories, history_rejected = guolv_lishi_wanzhengxing(
            histories,
            analysis_date=analysis_date,
            minimum_rows=21,  # 20日已实现收益需要21条价格；与上市时长偏好无关。
            minimum_amount=0,
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
        technical_summaries: dict[str, dict[str, Any]] = {}
        panel, panel_meta = goujian_fenxi_yinzi_mianban(
            histories,
            ready_profiles,
            source="auto",
            calendar=self._context.jiaoyi_rili(start_date=start_date, end_date=end_date),
            latest_only=True,
            technical_summaries=technical_summaries,
            macd_structure_config=settings.get("macd_structure"),
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
        completed_quote_is_current = bool(
            quote_is_completed
            and analysis_date.normalize() == pd.Timestamp(self._context.reference.date())
        )
        profiles_by_code = profile_subset.set_index("ts_code", drop=False)
        preliminary: list[dict[str, Any]] = []
        factor_limit = len(factor_map)
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
                quote_check["hard_blocks"].extend(realtime_blocks)
                quote_check.update(status="blocked", basic_execution_feasible=False)
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
                "user_conditions": user_conditions or [],
                "tradability": quote_check,
                "data_quality": {
                    "history_source": next((batch.get("source_by_code", {}).get(code) or batch.get("source") for batch in history_batches if code in (batch.get("source_by_code") or {})), history_meta.get("source")),
                    "history_rows": int(len(histories[code])),
                    "as_of": analysis_date.strftime("%Y-%m-%d"),
                    "realtime_source": snapshot.get("source"),
                },
            }
            preliminary.append(item)
        preliminary = paixu_shangzhang_houxuan(preliminary, config=self._config)
        # 完整比较与技术核验先完成；展示数量只限制昂贵的财务和分钟补充报告。
        # 初筛缺失不等于条件不满足：技术原值可补齐的候选仍须完成复核。
        review_pool = [item for item in preliminary if not item["selection"]["unmet_conditions"]]
        if not review_pool:
            review_pool = preliminary[:1]
        technical_reviewed: list[dict[str, Any]] = []
        review_stop_reason = "candidate_pool_exhausted"
        for item in review_pool:
            try:
                technical = technical_summaries.pop(item["ts_code"])
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
        technical_reviewed = paixu_shangzhang_houxuan(technical_reviewed, config=self._config, deep_reviewed=True)
        qualified_reviewed = [item for item in technical_reviewed if item["selection"]["eligible"]]
        deep_candidates = qualified_reviewed[:display_target] or technical_reviewed[:1]
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
                except FenxiJisuanYichang:
                    raise
                except Exception as exc:
                    fundamentals = {"profile": item["profile"], "financials": {}, "valuation": {}, "errors": [str(exc)]}
                if not fundamentals.get("profile"):
                    fundamentals["profile"] = item["profile"]
                item["fundamental"] = hebing_jibenmian_zhengju(item["fundamental"], fundamentals)
                if late_stage is WeipanJieduan.PANZHONG_ZANDING and index < int(self._config["weipan"]["minute_candidate_limit"]):
                    try:
                        minute_data, minute_meta = self._context.fenzhong_xingqing(item["ts_code"])
                    except FenxiJisuanYichang:
                        raise
                    except Exception as exc:
                        minute_data, minute_meta = pd.DataFrame(), {"status": "unavailable", "error": str(exc)}
                    item["late"] = fenxi_weipan(
                        item["history"], snapshot=item["snapshot"], clock=clock,
                        config=self._config["weipan"], minute_data=minute_data,
                    )
                    item["late"]["minute_data"] = minute_meta
                # 控盘只补充最终报告，不参与默认条件、推荐指数或候选截取。
                ownership_source = self._context.kongpan_shuju(item["ts_code"])
                try:
                    item["ownership_analysis"] = fenxi_kongpan_shuju(
                        ownership_source,
                        item["history"],
                        analysis_date=analysis_date.strftime("%Y-%m-%d"),
                    )
                except FenxiJisuanYichang:
                    raise
                except Exception as exc:
                    raise FenxiJisuanYichang(f"{item['ts_code']} 控盘分析程序错误：{exc}") from exc
                if item["ownership_analysis"].get("status") == "error":
                    raise FenxiJisuanYichang(
                        f"{item['ts_code']} 控盘分析程序错误："
                        + str(item["ownership_analysis"].get("error") or item["ownership_analysis"].get("missing_reason") or "未返回有效计算结果")
                    )
            item["supplemental_diagnostics"] = goujian_buchong_zhenduan(
                item["history"],
                item["fundamental"],
                as_of_date=analysis_date.strftime("%Y-%m-%d"),
                minimum_amount_yuan=float(settings["min_amount_yuan"]),
            )
            item["selection"].update(pinggu_shangzhang_tiaojian(item, config=self._config, deep_reviewed=True))
            item["research_assessment"] = pinggu_yanjiu_wenti(research_question, item)
        # 对完整技术复核池重算共同分位；昂贵补充证据不能改变比较范围或排序。
        # 资格若在补充阶段意外改变，停止发布，避免用旧的详报对象冒充最终Top。
        technical_reviewed = paixu_shangzhang_houxuan(technical_reviewed, config=self._config, deep_reviewed=True)
        qualified_reviewed = [item for item in technical_reviewed if item["selection"]["eligible"]]
        final_deep_candidates = qualified_reviewed[:display_target] or technical_reviewed[:1]
        if {item["ts_code"] for item in final_deep_candidates} != {item["ts_code"] for item in deep_candidates}:
            raise FenxiJisuanYichang("完整复核后的推荐指数候选与已补充报告对象不一致，本次不能发布未完成报告的排名")
        deep_candidates = final_deep_candidates
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
        reviewed_by_code = {item["ts_code"]: item for item in technical_reviewed}
        preliminary = [reviewed_by_code.get(item["ts_code"], item) for item in preliminary]
        condition_checks = [
            {
                "ts_code": item["ts_code"], "name": item["name"],
                "technical_reviewed": item["ts_code"] in reviewed_codes,
                "missing_conditions": item["selection"]["missing_conditions"],
                "unmet_conditions": item["selection"]["unmet_conditions"],
                "recommendation_index": item["selection"].get("recommendation_index"),
                "recommendation_index_status": item["selection"].get("comparison_status"),
                "recommendation_index_missing_fields": item["selection"].get("comparison_missing_fields", []),
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
        # 范围覆盖与可选报告证据分开：指数、财务等失败不改写已核验的选股结论。
        scope_partial = (
            bool(cap_summary.get("unavailable_count"))
            or any(item.get("missing_fields") for item in rejected)
            or any(item.get("status") == "unavailable" for item in history_rejected)
            or history_loaded_count < len(prefiltered) or len(factor_map) < len(histories)
            or pool.metadata.get("identity_directory_coverage_verified") is False
            or any(metadata.get("status") in {"partial", "unavailable", "degraded"}
                   for metadata in (pool.metadata, history_meta))
        )
        optional_source_partial = any(
            metadata.get("status") in {"partial", "unavailable", "degraded"}
            for metadata in (panel_meta, panel_meta.get("market_benchmarks") or {})
        )
        evidence_partial = bool(
            scope_partial or optional_source_partial or any(item["missing_conditions"] for item in condition_checks)
            or (research_question or {}).get("unverifiable_parts")
            or any(item["evidence_gaps"] or item["technical_summary"].get("status") == "partial"
                   or item["late_session_analysis"].get("status") in {"partial", "unavailable"}
                   or (item["late_session_analysis"].get("minute_data") or {}).get("status") == "unavailable"
                   for item in summaries)
        )
        selection_outcome = (
            "program_error" if technical_errors else "recommendation" if recommendation_available
            else "evidence_unavailable" if undecided_count or scope_partial else "no_recommendation"
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
                else "当前范围中没有股票同时通过用户明确条件、比较数据和适用的成交核验；具体缺项与反证已保留"
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
            "user_conditions": user_conditions or [],
            "research_question": research_question,
            "comparison_coverage": {
                "method": "all_scope_members_in_batches", "sampled": False,
                "scope_input": len(data), "history_requested": len(codes), "factor_ready": len(factor_map),
                "technical_review_planned": len(review_pool), "technical_reviewed": len(technical_reviewed),
                "recommendation_index_ready": len(qualified_reviewed),
                "recommendation_index_unavailable": sum(
                    not item["selection"]["unmet_conditions"] and item["selection"].get("recommendation_index") is None
                    for item in technical_reviewed
                ),
                "full_report_requested": len(deep_candidates), "full_report_generated": len(summaries),
                "full_report_complete": sum(
                    not item["evidence_gaps"] and item["technical_summary"].get("status") == "ok"
                    and item["late_session_analysis"].get("status") not in {"partial", "unavailable"}
                    and (item["late_session_analysis"].get("minute_data") or {}).get("status") != "unavailable"
                    for item in summaries
                ),
                "ownership_requested": sum("ownership_analysis" in item for item in deep_candidates),
                "ownership_observed": sum(bool((item.get("ownership_analysis") or {}).get("latest")) for item in deep_candidates),
                "complete": not scope_partial and not undecided_count and not technical_errors,
                "limitation": "覆盖只指本次已核验范围；数据失败与条件缺失逐项保留，未取得者不能视为已比较。",
            },
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
                "qualified": 0 if technical_errors else int(len(qualified_reviewed)),
                "recommendation_index_ready": 0 if technical_errors else int(len(qualified_reviewed)),
                "recommendation_index_unavailable": int(sum(
                    not item["selection"]["unmet_conditions"] and item["selection"].get("recommendation_index") is None
                    for item in technical_reviewed
                )),
                "unverified": int(undecided_count),
                "displayed": int(bool(primary)) + len(alternatives),
            },
            "filter_summary": {
                "rejected_count": int(len(rejected)),
                "rejected_examples": rejected[:20],
                "rules": [
                    "默认不按ST、上市时长、波动率或成交额大小施加风险偏好",
                    "完整分析日、有效价量和20日比较窗口必须可核验；缺失不是通过",
                    "保留适用的当前成交及一字涨停核验；额外限制仅来自用户明确条件",
                ],
            },
            "data_provenance": {
                "scope": pool.metadata,
                "history": history_meta,
                "factor_panel": panel_meta,
                "realtime_snapshot": realtime_meta,
            },
            "selection_methodology": {
                "method": "user_constraints_then_current_price_volume_recommendation_index",
                "conditions": "用户明确条件、有效比较窗口、技术一致性及本次适用的成交核验；未指定时不追加风险或全部看涨条件",
                "comparison_dimensions": ["20日已实现收益", "20日带方向路径效率", "20日上涨成交额占优度"],
                "ranking_basis": "完整合格共同池N至少为2；三原值使用平均升序名次分位p=(rank-0.5)/N，P=(pR+pE)/2，V=pU，推荐指数=100sqrt(PV)。两组几何等权、价格组内部等权是公开固定设计，不训练、不按结果寻优。按未舍入指数降序，完全相同明确并列；代码只稳定展示。",
                "best_definition": "最优指本次实际可核验同一资格池中，当前价格推进与成交配合最符合上述公开指数规则；分数不是未来收益率、上涨概率、置信度或已验证收益优势，跨范围跨日期不可直接比较绝对强弱",
                "sampling_limit": "范围成分分批全量请求，完整比较并复核潜在合格者后才裁剪展示。财务与分钟补充仅服务最终候选，各守既有上限；来源失败时明确覆盖缺口。",
                "missing_evidence": "必需条件或任一推荐原值缺失、分母为零、共同比较池不足2只均不产生指数，不填中性分、不临时重分权重；证据不足单列，不解释成条件不满足",
                "llm_boundary": "只解释程序真实计算的指数、原值分位组分和反证，不改写数值权重排序，不生成上涨概率，不将状态排序冒充经验证的未来收益排名",
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
    tiaojian: dict[str, Any] | None = None,
    yanjiu_wenti: dict[str, Any] | None = None,
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
        user_conditions = []
        research_question = jiexi_yanjiu_wenti(yanjiu_wenti)
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
            try:
                user_conditions = jiexi_xuangu_tiaojian(tiaojian)
            except ValueError as exc:
                question = str(exc)
                return {
                    "status": "clarification_required", "outcome": "clarification_required",
                    "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                    "analysis_type": analysis_type, "stage": "request_validation",
                    "error_code": "selection_condition_required", "error": question,
                    "clarification": {"kind": "selection_condition", "title": "确认选股条件", "question": question},
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
            user_conditions=user_conditions,
            research_question=research_question,
        )
        return xianzhi_xuangu_jieguo(result, shuliang)
    except FenxiJisuanYichang as exc:
        return {
            **({"selection_outcome": "program_error"} if analysis_type != DANGU_ANALYSIS_TYPE else {}),
            "status": "error",
            "outcome": "program_error",
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": analysis_type,
            "error_code": "analysis_calculation_error",
            "stage": "quantitative_analysis",
            "source": None,
            "retryable": False,
            "error": " ".join(str(exc).split())[:240],
            "next_action": "请记录运行编号并检查程序日志",
            "recommendation_available": False,
            "primary": None,
            "alternatives": [],
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
        }
    except WangluoQingqiuYichang as exc:
        return {
            **({"selection_outcome": "evidence_unavailable"} if analysis_type != DANGU_ANALYSIS_TYPE else {}),
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
            **({"selection_outcome": "evidence_unavailable"} if analysis_type != DANGU_ANALYSIS_TYPE else {}),
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
            **({"selection_outcome": "program_error"} if analysis_type != DANGU_ANALYSIS_TYPE else {}),
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
