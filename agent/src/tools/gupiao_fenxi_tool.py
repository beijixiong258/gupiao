"""自然语言智能体使用的统一 A 股选股分析工具。"""

from __future__ import annotations

import json
from typing import Any, Literal, get_args

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_state import analysis_session_store


FenxiScope = Literal["all_market", "named_scope", "single_stock"]


def _mianxiang_zhinengti_jieguo(result: dict[str, Any]) -> dict[str, Any]:
    """去掉重复的大字段，同时保留智能体解释与审查所需的完整证据。"""
    public = {
        key: value
        for key, value in result.items()
        if key != "reviewed_candidates"
    }
    reviewed = result.get("reviewed_candidates")
    if isinstance(reviewed, list):
        public["reviewed_candidate_count"] = len(reviewed)
        public["displayed_candidate_count"] = int(bool(result.get("primary"))) + len(result.get("alternatives") or [])

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
        "Analyze mainland A-share evidence or select research candidates by explicit conditions. Use single_stock with gupiao "
        "to retrieve all currently analyzable stock evidence, including every raw daily-factor value and missing field, technical "
        "structure, financials and valuation context, pattern and late-session conditions, supplemental diagnostics, execution "
        "constraints and source provenance. Optional source failure yields a partial report, never a forced buy/no-buy label. "
        "Use all_market or named_scope for rule-based selection; named scopes are dynamically discovered and verified. "
        "Selection filters explicit upward-signal conditions and compares five raw dimensions by non-dominated layers; "
        "same-layer ordering is stable display only. Return the exposed research candidates or explain no qualification. "
        "Do not calculate scores, weights, predicted probabilities or model forecasts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "fanwei": {
                "type": "string",
                "enum": list(get_args(FenxiScope)),
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
        if full_result.get("status") not in {"ok", "partial"}:
            return json.dumps(full_result, ensure_ascii=False)
        stored_result = {
            key: value for key, value in full_result.items() if not str(key).startswith("_")
        }
        analysis_id = analysis_session_store.save(stored_result)
        if stored_result.get("analysis_type") == "single_stock_analysis":
            stock = stored_result.get("stock") or stored_result.get("selected_stock")
            public_result = {
                **_mianxiang_zhinengti_jieguo(stored_result),
                "analysis_id": analysis_id,
                "selected_stock": stock,
                "analysis_stage": stored_result.get("analysis_stage") or {
                    "status": "completed",
                    "scope": "单股可取得的原始指标、财务、形态、尾盘和来源证据已整理",
                    "next_step": "完整说明支持和反向证据、缺口、风险与重新评估条件",
                },
            }
            return json.dumps(public_result, ensure_ascii=False)
        recommendation_available = bool(stored_result.get("recommendation_available"))
        public_result = {
            **_mianxiang_zhinengti_jieguo(stored_result),
            "analysis_id": analysis_id,
            "selected_stock": stored_result.get("primary"),
            "analysis_stage": {
                "status": "partial" if stored_result.get("status") == "partial" else "completed",
                "scope": (
                    "已整理当前可取得的候选证据；部分来源或条件尚未完成核验"
                    if stored_result.get("status") == "partial"
                    else "候选范围核验、原始证据、显式选股条件与非支配比较已完成"
                ),
                "next_step": (
                    "说明候选的量化依据、证据缺口、主要风险与重新评估条件"
                    if recommendation_available
                    else "说明当前没有通过选股条件的候选，以及缺失和未满足条件"
                ),
            },
        }
        return json.dumps(public_result, ensure_ascii=False)


__all__ = ["FenxiScope", "GupiaoFenxiTool"]
