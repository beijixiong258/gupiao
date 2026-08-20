"""统一选股分析门面：编排候选池、因子、增强证据与最终排序。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ashare.fenxi_weipan import WeipanJieduan, fenxi_weipan, panduan_weipan_jieduan
from src.ashare.fenxi_xingtai import fenxi_zhangting_huimaqiang
from src.ashare.dangu_fenxi import DANGU_ANALYSIS_TYPE, fenxi_dangu
from src.ashare.fenxi_yinzi import (
    goujian_fenxi_yinzi_mianban,
    huizong_houxuan_yinzi,
    jisuan_hengjiemian_jibenmian,
    zengjia_dangri_guzhi_yinzi,
)
from src.ashare.gupiao_yanjiu import huoqu_jibenmian, zongjie_jishu
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
    guolv_lishi_wanzhengxing,
    hecheng_hengjiemian_yu_shendu_jibenmian,
    hecheng_houxuan_fenshu,
    jichu_ying_guolv,
    jisuan_fengxian_koufen,
    shibie_yizijia_zhangting,
    xuyao_shishi_kuaizhao,
    zhuan_json_zhi,
)


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
    """按用户明确要求的数量裁剪公开候选及预测交接身份。"""

    count = _normalize_requested_count(requested_count)
    if (
        count is None
        or result.get("status") != "ok"
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
    prediction_context = result.get("_prediction_context")
    if isinstance(prediction_context, dict) and isinstance(prediction_context.get("candidates"), dict):
        selected_codes = {
            str(item.get("ts_code"))
            for item in selected
            if item.get("ts_code")
        }
        limited["_prediction_context"] = {
            **prediction_context,
            "primary_code": str(primary.get("ts_code") or prediction_context.get("primary_code") or ""),
            "candidates": {
                code: candidate
                for code, candidate in prediction_context["candidates"].items()
                if str(code) in selected_codes
            },
        }
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

    def fenxi(self, fanwei: FenxiFanwei) -> dict[str, Any]:
        if fanwei.leixing is FanweiLeixing.DANGU_GUPIAO:
            return fenxi_dangu(
                gupiao=str(fanwei.gupiao or ""),
                config=self._config,
                context=self._context,
            )
        settings = self._config["fenxi"]
        if fanwei.leixing is FanweiLeixing.MINGMING_FANWEI:
            discovery = self._context.faxian_fanwei(str(fanwei.mingcheng or ""))
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
                rejected.append({"ts_code": row.get("ts_code"), "name": row.get("name"), "reasons": reasons})
            else:
                accepted_indices.append(index)
        filtered = data.loc[accepted_indices].reset_index(drop=True)
        if filtered.empty:
            return {
                "status": "ok",
                "outcome": "no_recommendation",
                "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                "scope": pool.metadata,
                "as_of": analysis_date.strftime("%Y-%m-%d"),
                "recommendation_available": False,
                "no_recommendation_reason": "全部候选都触发了风险硬过滤",
                "primary": None,
                "alternatives": [],
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
        histories, history_rejected = guolv_lishi_wanzhengxing(
            histories,
            analysis_date=analysis_date,
            minimum_rows=int(settings["minimum_history_rows"]),
            minimum_amount=float(settings["min_amount_yuan"]),
            minimum_listing_calendar_days=int(settings["minimum_listing_calendar_days"]),
        )
        rejected.extend(history_rejected)
        if not histories:
            return {
                "status": "unavailable",
                "outcome": "data_unavailable",
                "error_code": "candidate_history_insufficient",
                "stage": "history_data",
                "source": history_meta.get("source"),
                "retryable": True,
                "next_action": "稍后重新获取完整远端日线",
                "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
                "scope": pool.metadata,
                "as_of": analysis_date.strftime("%Y-%m-%d"),
                "error": "硬过滤后没有具备足够完整日线的候选",
                "data_provenance": {"history": history_meta},
                "filter_summary": {"input_count": len(data), "rejected_count": len(rejected), "examples": rejected[:20]},
            }
        ready_profiles = prefiltered[prefiltered["ts_code"].isin(histories)].reset_index(drop=True)
        panel, panel_meta = goujian_fenxi_yinzi_mianban(
            histories,
            ready_profiles,
            source="auto",
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
                realtime_table, realtime_meta = self._context.shishi_kuaizhao()
            if not realtime_table.empty:
                realtime_table = realtime_table.set_index("ts_code", drop=False)
        completed_quote_is_current = bool(
            quote_is_completed
            and analysis_date.normalize() == pd.Timestamp(self._context.reference.date())
        )
        profiles_by_code = profile_subset.set_index("ts_code", drop=False)
        preliminary: list[dict[str, Any]] = []
        factor_limit = int(settings["factor_candidate_limit"])
        factor_order = sorted(
            factor_map,
            key=lambda code: float(factor_map[code].get("score_0_100") or -1.0),
            reverse=True,
        )[:factor_limit]
        for code in factor_order:
            if code not in profiles_by_code.index or code not in histories:
                continue
            profile_row = profiles_by_code.loc[code]
            if isinstance(profile_row, pd.DataFrame):
                profile_row = profile_row.iloc[0]
            name = str(profile_row.get("name") or "")
            if (
                realtime_required
                and realtime_meta.get("status") == "ok"
                and (realtime_table.empty or code not in realtime_table.index)
            ):
                rejected.append(
                    {
                        "ts_code": code,
                        "name": name,
                        "reasons": ["全市场实时快照缺少该股票，当前可交易状态无法确认"],
                    }
                )
                continue
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
            try:
                realtime_block = shibie_yizijia_zhangting(
                    snapshot,
                    code=code,
                    name=name,
                    tolerance_yuan=float(self._config["xingtai"]["limit_up_tolerance_yuan"]),
                )
            except ValueError:
                realtime_block = "股票代码不满足程序已有的 A 股市场规则"
            if realtime_block:
                rejected.append(
                    {
                        "ts_code": code,
                        "name": name,
                        "reasons": [f"当前行情为{realtime_block}"],
                    }
                )
                continue
            pattern = fenxi_zhangting_huimaqiang(
                histories[code],
                code=code,
                name=name,
                config=self._config["xingtai"],
                realtime_quote=snapshot if str(clock.get("session_status")) in {"trading", "midday_break"} else None,
            )
            late = fenxi_weipan(
                histories[code],
                snapshot=snapshot,
                clock=clock,
                config=self._config["weipan"],
            )
            factor = factor_map[code]
            fundamental = fundamental_map.get(code, {"status": "unavailable", "score_0_100": None, "confidence": 0.0, "evidence": []})
            penalties, risks = jisuan_fengxian_koufen(
                code=code,
                name=name,
                snapshot=snapshot,
                factor=factor,
                pattern=pattern,
                late=late,
                config=self._config,
            )
            ranking = hecheng_houxuan_fenshu(
                factor=factor,
                fundamental=fundamental,
                pattern=pattern,
                late=late,
                penalties=penalties,
                config=self._config,
            )
            preliminary.append(
                {
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
                    "ranking": ranking,
                    "technical": {},
                    "tradability": {"status": "prefiltered", "basic_execution_feasible": True, "hard_blocks": [], "cautions": []},
                    "risks": risks,
                    "data_quality": {
                        "history_source": history_meta.get("source"),
                        "history_rows": int(len(histories[code])),
                        "as_of": analysis_date.strftime("%Y-%m-%d"),
                        "realtime_source": snapshot.get("source"),
                    },
                }
            )
        preliminary.sort(key=lambda item: float(item["ranking"].get("score_0_100") or -1.0), reverse=True)
        if late_stage is WeipanJieduan.PANZHONG_ZANDING:
            for item in preliminary[: int(self._config["weipan"]["minute_candidate_limit"])]:
                minute_data, minute_meta = self._context.fenzhong_xingqing(item["ts_code"])
                item["late"] = fenxi_weipan(
                    item["history"],
                    snapshot=item["snapshot"],
                    clock=clock,
                    config=self._config["weipan"],
                    minute_data=minute_data,
                )
                item["late"]["minute_data"] = minute_meta
                penalties, extra_risks = jisuan_fengxian_koufen(
                    code=item["ts_code"],
                    name=item["name"],
                    snapshot=item["snapshot"],
                    factor=item["factor"],
                    pattern=item["pattern"],
                    late=item["late"],
                    config=self._config,
                )
                item["risks"] = list(dict.fromkeys(item["risks"] + extra_risks))
                item["ranking"] = hecheng_houxuan_fenshu(
                    factor=item["factor"],
                    fundamental=item["fundamental"],
                    pattern=item["pattern"],
                    late=item["late"],
                    penalties=penalties,
                    config=self._config,
                )
            preliminary.sort(key=lambda item: float(item["ranking"].get("score_0_100") or -1.0), reverse=True)
        deep_candidates = preliminary[: int(settings["deep_analysis_limit"])]
        for item in deep_candidates:
            try:
                fundamentals = huoqu_jibenmian(
                    item["ts_code"],
                    trade_date=analysis_date.strftime("%Y-%m-%d"),
                    allow_current_snapshot=bool(
                        late_stage is WeipanJieduan.SHOUPAN_FUHE
                        and completed_quote_is_current
                    ),
                )
            except Exception as exc:
                fundamentals = {"profile": item["profile"], "financials": {}, "valuation": {}, "errors": [str(exc)]}
            if not fundamentals.get("profile"):
                fundamentals["profile"] = item["profile"]
            item["fundamental"] = hecheng_hengjiemian_yu_shendu_jibenmian(
                item["fundamental"],
                fundamentals,
                deep_weight=float(settings["fundamental_deep_weight"]),
            )
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
                            (technical.get("macd_structure") or {}).get("reason")
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
            item["tradability"] = goujian_kejiaoyixing_zhaiyao(
                code=item["ts_code"],
                name=item["name"],
                snapshot=item["snapshot"],
                history=item["history"],
                minimum_amount=float(settings["min_amount_yuan"]),
            )
            penalties, deep_risks = jisuan_fengxian_koufen(
                code=item["ts_code"],
                name=item["name"],
                snapshot=item["snapshot"],
                factor=item["factor"],
                pattern=item["pattern"],
                late=item["late"],
                config=self._config,
                technical=item["technical"],
            )
            financials = (item["fundamental"].get("financials") or {}) if isinstance(item["fundamental"], dict) else {}
            if not financials:
                deep_risks.append("深度财务指标不可用，基本面证据不完整")
            item["risks"] = list(dict.fromkeys(item["risks"] + deep_risks))
            item["ranking"] = hecheng_houxuan_fenshu(
                factor=item["factor"],
                fundamental=item["fundamental"],
                pattern=item["pattern"],
                late=item["late"],
                penalties=penalties,
                config=self._config,
            )
        deep_candidates.sort(key=lambda item: float(item["ranking"].get("score_0_100") or -1.0), reverse=True)
        minimum_score = float(settings["minimum_recommendation_score"])
        minimum_confidence = float(settings["minimum_confidence"])
        summaries = [
            goujian_houxuan_zhaiyao(
                item,
                rank=index,
                minimum_score=minimum_score,
                minimum_confidence=minimum_confidence,
            )
            for index, item in enumerate(deep_candidates, start=1)
        ]
        qualified = [item for item in summaries if item["meets_recommendation_threshold"]]
        primary = qualified[0] if qualified else None
        alternatives = qualified[1 : 1 + int(settings["backup_limit"])]
        qualified_codes = {str(item["ts_code"]) for item in qualified}
        prediction_contexts: dict[str, dict[str, Any]] = {}
        for candidate in deep_candidates:
            if candidate["ts_code"] not in qualified_codes:
                continue
            prediction_contexts[candidate["ts_code"]] = {
                "code": candidate["ts_code"],
                "name": candidate["name"],
                "industry": candidate["industry"],
                "technical": candidate["technical"],
                "fundamentals": candidate["fundamental"],
                "tradability": candidate["tradability"],
            }
        prediction_context = (
            {
                "primary_code": primary["ts_code"] if primary else None,
                "source": "auto",
                "signal_date": analysis_date.strftime("%Y-%m-%d"),
                "config": self._config,
                "candidates": prediction_contexts,
            }
            if prediction_contexts
            else None
        )
        recommendation_available = primary is not None
        result = {
            "status": "ok",
            "outcome": "recommendation" if recommendation_available else "no_recommendation",
            "tool_contract_version": XUANGU_TOOL_CONTRACT_VERSION,
            "analysis_type": "unified_stock_selection",
            "scope": pool.metadata,
            "as_of": analysis_date.strftime("%Y-%m-%d"),
            "generated_at": self._context.reference.strftime("%Y-%m-%d %H:%M:%S"),
            "market_clock": clock,
            "result_confirmation": (
                "intraday_provisional"
                if late_stage in {WeipanJieduan.CHUSHAI, WeipanJieduan.PANZHONG_ZANDING}
                else "close_pending"
                if late_stage is WeipanJieduan.SHOUPAN_DAIDING
                else "completed_daily_close"
            ),
            "recommendation_available": recommendation_available,
            "primary": primary,
            "alternatives": alternatives,
            "reviewed_candidates": summaries,
            "no_recommendation_reason": (
                None
                if recommendation_available
                else f"深度复核后的候选没有同时达到排名分 {minimum_score:.1f} 和可信度 {minimum_confidence:.2f} 门槛"
            ),
            "thresholds": {
                "minimum_recommendation_score": minimum_score,
                "minimum_confidence": minimum_confidence,
                "maximum_alternatives": int(settings["backup_limit"]),
            },
            "candidate_counts": {
                "scope_input": int(len(data)),
                "after_hard_filter": int(len(filtered)),
                "after_prefilter": int(len(prefiltered)),
                "history_ready": int(len(histories)),
                "factor_ready": int(len(preliminary)),
                "deep_reviewed": int(len(deep_candidates)),
                "qualified": int(len(qualified)),
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
            "ranking_methodology": {
                "components": list(settings["component_weights"].keys()),
                "component_weights": settings["component_weights"],
                "factor_group_weights": settings["factor_group_weights"],
                "missing_evidence": "不可用组件不按失败处理；其余有效证据重新归一，同时降低可信度",
                "llm_boundary": "LLM只能解释程序结果，不得修改数值、条件状态和排序",
            },
            "research_scope": "公开数据研究排序，不连接证券账户、不提交委托、不自动交易",
            "_prediction_context": prediction_context,
        }
        return result


def fenxi_xuangu(
    *,
    fanwei: str = "all_market",
    mingcheng: str | None = None,
    gupiao: str | None = None,
    shuliang: int | str | None = None,
    context: FenxiShujuShangxiawen | None = None,
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
        config, _ = jiazai_lianghua_peizhi()
        resolved_scope = FenxiFanwei.create(fanwei, mingcheng, gupiao)
        request_context = context or FenxiShujuShangxiawen()
        result = XuanguFenxiFuWu(config=config, context=request_context).fenxi(resolved_scope)
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
