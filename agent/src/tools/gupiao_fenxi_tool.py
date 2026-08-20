"""自然语言智能体使用的统一 A 股选股分析工具。"""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_state import analysis_session_store


def _mianxiang_zhinengti_jieguo(result: dict[str, Any]) -> dict[str, Any]:
    """去掉重复的大字段，同时保留智能体解释与审查所需的完整证据。"""
    public = {
        key: value
        for key, value in result.items()
        if key != "reviewed_candidates"
    }
    reviewed = result.get("reviewed_candidates")
    if isinstance(reviewed, list):
        requested_count = result.get("requested_candidate_count")
        if isinstance(requested_count, int) and requested_count > 0:
            public["reviewed_candidate_count"] = min(
                requested_count,
                int(bool(result.get("primary"))) + len(result.get("alternatives") or []),
            )
        else:
            public["reviewed_candidate_count"] = len(reviewed)

    provenance = public.get("data_provenance")
    if isinstance(provenance, dict):
        # 顶层 scope 已包含同一份已核验信息；不在工具消息中重复一遍。
        public["data_provenance"] = {
            key: value for key, value in provenance.items() if key != "scope"
        }
    return public


class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "Run the primary quantitative A-share analysis. Use single_stock with gupiao for a quantitative buy assessment of one named stock, "
        "all_market for the whole mainland A-share market, or named_scope for one ordinary-language named scope. "
        "For a named scope it dynamically queries and verifies the live source "
        "catalog instead of guessing whether the phrase is an industry or concept. It automatically applies hard risk filters, all eight daily-K factor "
        "groups, fundamentals and valuation, explanatory MACD structure evidence, the limit-up pullback pattern, and late-session evidence when the local "
        "market time is after 14:30. A single-stock analysis builds a bounded live comparison pool, reuses the same quantitative "
        "scoring and recommendation thresholds, and returns an explicit buy/no-buy research recommendation plus a plain-language closing synthesis without prediction eligibility. A selection returns one primary stock, "
        "up to four alternatives, or an explicit decision not to recommend. The score is a research ranking score, never an upside probability."
    )
    parameters = {
        "type": "object",
        "properties": {
            "fanwei": {
                "type": "string",
                "enum": ["all_market", "named_scope", "single_stock"],
                "default": "all_market",
                "description": "Use single_stock for one stock, all_market for the whole market, or named_scope whenever the user says an ordinary industry/board/theme phrase. Never classify a named scope yourself.",
            },
            "mingcheng": {
                "type": "string",
                "description": "The user's ordinary-language scope phrase, such as 电子板块. Omit it for all_market; do not translate it into a professional taxonomy.",
            },
            "gupiao": {
                "type": "string",
                "description": "The user's complete stock name or 6 digit security code when fanwei=single_stock. Do not use it for a range request.",
            },
            "shuliang": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Optional number of stocks the user explicitly requested for a range selection. If omitted, use the normal primary plus available alternatives; never invent a count.",
            },
        },
        "required": [],
    }
    repeatable = True
    # 只保存多轮对话所需的进程内会话状态，不保存市场时间序列。
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.xuangu_fenxi import fenxi_xuangu

        full_result = fenxi_xuangu(
            fanwei=str(kwargs.get("fanwei") or "all_market"),
            mingcheng=str(kwargs.get("mingcheng") or "").strip() or None,
            gupiao=str(kwargs.get("gupiao") or "").strip() or None,
            shuliang=kwargs.get("shuliang"),
        )
        if full_result.get("status") != "ok":
            return json.dumps(full_result, ensure_ascii=False)
        prediction_context = full_result.get("_prediction_context")
        stored_result = {
            key: value for key, value in full_result.items() if not str(key).startswith("_")
        }
        analysis_id = analysis_session_store.save(
            stored_result,
            prediction_context=prediction_context,
        )
        if stored_result.get("analysis_type") == "single_stock_analysis":
            stock = stored_result.get("stock") or stored_result.get("selected_stock")
            public_result = {
                **_mianxiang_zhinengti_jieguo(stored_result),
                "analysis_id": analysis_id,
                "selected_stock": stock,
                "analysis_stage": stored_result.get("analysis_stage") or {
                    "status": "completed",
                    "scope": "单股八组因子、综合评分和买入门槛复核已完成",
                    "prediction_status": "not_available",
                    "prediction_confirmation_required": False,
                    "confirmation_timing": "not_applicable",
                    "initial_preapproval_counts": False,
                    "affirmative_reply_defaults_to": None,
                    "prediction_data_policy": "fresh_remote_download_without_local_market_cache",
                    "next_step": "当前结果已明确是否建议买入，但不产生自动预测资格",
                },
            }
            return json.dumps(public_result, ensure_ascii=False)
        recommendation_available = bool(stored_result.get("recommendation_available"))
        public_result = {
            **_mianxiang_zhinengti_jieguo(stored_result),
            "analysis_id": analysis_id,
            "selected_stock": stored_result.get("primary"),
            "analysis_stage": {
                "status": "completed",
                "scope": "候选池、风险硬过滤、八组日K因子、基本面、形态、尾盘证据、深度复核和排序已完成",
                "prediction_status": "not_requested" if recommendation_available else "not_available",
                "prediction_confirmation_required": recommendation_available,
                "confirmation_timing": (
                    "later_user_turn_after_analysis_result"
                    if recommendation_available
                    else "not_applicable"
                ),
                "initial_preapproval_counts": False,
                "affirmative_reply_defaults_to": "primary" if recommendation_available else None,
                "prediction_data_policy": "fresh_remote_download_without_local_market_cache",
                "next_step": (
                    "先用自然语言讲清量化结论，再单独询问是否预测；即使用户最初说不用再问，也必须在分析结果后询问一次。"
                    "后续回复‘行’或‘继续’即默认预测首选，点名合格备选则预测该备选，不再重复确认；确认后才重新下载远端数据并训练 T+1/T+2/T+3 模型"
                    if recommendation_available
                    else "当前没有达到门槛的候选，不能继续预测"
                ),
            },
        }
        return json.dumps(public_result, ensure_ascii=False)


__all__ = ["GupiaoFenxiTool"]
