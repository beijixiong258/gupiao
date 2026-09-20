"""自然语言智能体使用的统一 A 股选股分析工具。"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal, get_args

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_state import analysis_session_store


FenxiScope = Literal["all_market", "named_scope", "single_stock"]

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "metric": {"type": "string", "description": "Exact raw metric from the returned registry, e.g. ma_gap_20, ma_trend_5_20, ret_5, ret_20, macd_hist_pct, volume_ratio_5_20, excess_vs_csi300_ret_20, volatility_20. Selection also supports is_st/is_delisting (0/1), listing_days and amount_yuan. Research additionally supports roe_pct, net_profit_yoy_pct, revenue_yoy_pct, debt_to_assets_pct and pattern_close_confirmed (only the specific limit-up pullback pattern)."},
        "operator": {"type": "string", "enum": ["gt", "gte", "lt", "lte", "eq"]},
        "value": {"type": "number", "description": "Raw units: returns/volatility use fractions (5%=0.05), volume ratio uses times, amount uses yuan, financial *_pct uses percentage points. Supply exactly one of value/reference_metric."},
        "reference_metric": {"type": "string", "description": "Another supported observable in the same units instead of a numeric threshold."},
        "label": {"type": "string", "description": "Plain Chinese description of this observable assertion, faithful to the user's condition or research question."},
    },
    "required": ["metric", "operator", "label"],
    "additionalProperties": False,
}


def _mianxiang_zhinengti_jieguo(result: dict[str, Any]) -> dict[str, Any]:
    """去掉重复的大字段，同时保留智能体解释与审查所需的完整证据。"""
    public = deepcopy({
        key: value
        for key, value in result.items()
        if key != "reviewed_candidates"
    })
    # 定义按字段共享一次；逐股样本数留在原位置，原始观测与缺失项不裁剪。
    definitions: dict[str, Any] = {}
    from src.ashare.yinzi_gongcheng import FACTOR_REGISTRY
    known = {entry["feature"] for entry in FACTOR_REGISTRY}

    def share_definitions(value: Any) -> None:
        if isinstance(value, dict):
            metadata = value.get("metric_definitions")
            if isinstance(metadata, dict):
                for feature, definition in list(metadata.items()):
                    if feature not in known or not isinstance(definition, dict):
                        continue
                    shared = {key: item for key, item in definition.items() if key != "valid_observations"}
                    if feature in definitions and definitions[feature] != shared:
                        continue
                    definitions[feature] = shared
                    metadata[feature] = {"definition_ref": feature, **(
                        {"valid_observations": definition["valid_observations"]} if "valid_observations" in definition else {}
                    )}
            for key, item in value.items():
                if key != "metric_definitions":
                    share_definitions(item)
        elif isinstance(value, list):
            for item in value:
                share_definitions(item)

    share_definitions(public)
    if definitions:
        public["indicator_definitions"] = definitions
    reviewed = result.get("reviewed_candidates")
    if isinstance(reviewed, list):
        public["reviewed_candidate_count"] = len(reviewed)
        public["displayed_candidate_count"] = int(bool(result.get("primary"))) + len(result.get("alternatives") or [])

    checks = public.get("candidate_condition_checks")
    if isinstance(checks, list):
        unresolved = [item for item in checks if item.get("missing_conditions") or item.get("unmet_conditions")]
        public["candidate_check_summary"] = {
            "checked_candidates": len(checks), "with_missing_conditions": sum(bool(item.get("missing_conditions")) for item in checks),
            "with_unmet_conditions": sum(bool(item.get("unmet_conditions")) for item in checks),
            "examples_shown": min(20, len(unresolved)), "scope": "公开汇总与最多20个异常示例；完整逐股核验保留于本次内存结果，未因此缩小比较范围",
        }
        public["candidate_condition_checks"] = unresolved[:20]

    provenance = public.get("data_provenance")
    if isinstance(provenance, dict):
        # 顶层 scope 已包含同一份已核验信息；不在工具消息中重复一遍。
        public["data_provenance"] = {
            key: value for key, value in provenance.items() if key != "scope"
        }
        for batch in (public["data_provenance"].get("history") or {}).get("batches", []):
            # 返回候选保有逐股实际来源；全范围来源数量已经在每批source_counts中记录。
            batch.pop("source_by_code", None)
    return public


class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "Analyze mainland A-share evidence or select research candidates by explicit conditions. Use single_stock with gupiao "
        "to retrieve all currently analyzable stock evidence, including every raw daily-factor value and missing field, technical "
        "structure, financials and valuation context, pattern and late-session conditions, supplemental diagnostics, execution "
        "constraints and source provenance. Optional source failure yields a partial report, never a forced buy/no-buy label. "
        "Use all_market or named_scope for rule-based selection; named scopes are dynamically discovered and verified. "
        "Always state shizhi explicitly: none only when no market-cap constraint was requested, unresolved when the user "
        "says small-cap without a confirmed basis and limit, or upper_limit with the user's exact bound. "
        "Unresolved constraints return a clarification before fetching market data. Never substitute unrestricted selection. "
        "Selection compares all obtainable scope members by 20-day/5-day realized performance Pareto fronts, then prefers "
        "20-day performance within a front. No default ST, age, volatility or amount preference. Only user-requested extra "
        "conditions belong in tiaojian; never invent risk constraints. Pass a stated stock conjecture in yanjiu_wenti with "
        "faithful observable checks and explicitly untestable parts. Do not equate an arbitrary proxy with the whole claim. "
        "Do not calculate scores, weights, predicted probabilities or model forecasts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tiaojian": {
                "type": "object",
                "description": "Only explicit selection restrictions beyond scope/count/market cap. Use none when none were requested; never ask risk tolerance unprompted. For a user-requested but undefined restriction use unresolved. E.g. user's uptrend means ma_gap_20>0 AND ma_trend_5_20>0, explicitly disclose this operational definition; do not invent a numeric volatility ceiling for 'low risk'.",
                "properties": {"mode": {"type": "string", "enum": ["none", "rules", "unresolved"]},
                               "rules": {"type": "array", "maxItems": 12, "items": _CHECK_SCHEMA}},
                "required": ["mode"], "additionalProperties": False,
            },
            "yanjiu_wenti": {
                "type": ["object", "null"],
                "description": "single_stock only. Preserve an explicit user conjecture/question. Convert only faithfully defined observable parts into checks; expose definitions and list the remainder in unverifiable_parts. A generic breakout is not the specific limit-up pullback pattern. Future probability, causality and undefined claims cannot be established by substituting indicator signs. Omit for a general stock analysis; do not force a clarification.",
                "properties": {"question": {"type": "string"},
                               "checks": {"type": "array", "maxItems": 12, "items": _CHECK_SCHEMA},
                               "unverifiable_parts": {"type": "array", "items": {"type": "string"}}},
                "required": ["question", "checks", "unverifiable_parts"], "additionalProperties": False,
            },
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
            "fanwei_xuanze": {
                "type": ["object", "null"],
                "description": "Only after scope_review_required: select one of this request's verified candidates using its exact code/kind and a concise semantic reason. Keep all original arguments unchanged. If real ambiguity remains, use clarify. Omit or null on the initial call.",
                "properties": {
                    "code": {"type": "string"},
                    "kind": {"type": "string", "enum": ["industry", "concept"]},
                    "reason": {"type": "string", "description": "Why this candidate fits the user's meaning better than the other verified candidates; identity verification alone does not prove semantic equivalence."},
                },
                "required": ["code", "kind", "reason"],
                "additionalProperties": False,
            },
            "shuliang": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Optional number of stocks the user explicitly requested for a range selection. If omitted, use the normal primary plus available alternatives; never invent a count.",
            },
            "shizhi": {
                "type": "object",
                "description": "Explicit market-cap constraint. Small-cap is a numeric condition, not an industry/theme. Do not invent a threshold. Use unresolved for missing basis/limit or an unsupported percentile request; single_stock uses none.",
                "properties": {
                    "mode": {"type": "string", "enum": ["none", "unresolved", "upper_limit"]},
                    "basis": {"type": ["string", "null"], "enum": ["total", "circulating", None], "description": "total=总市值, circulating=流通市值. Required for upper_limit; omit or null for none/unresolved."},
                    "max_yi": {"type": ["number", "null"], "exclusiveMinimum": 0, "description": "User's upper bound in 亿元. Required for upper_limit; 100 means 100亿元. Omit or null for none/unresolved."},
                    "inclusive": {"type": ["boolean", "null"], "description": "true for 不超过/以下 (<=), false for 低于/小于 (<). Omit or null for none/unresolved."},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
        "required": ["shizhi"],
    }
    repeatable = True
    # 只保存多轮对话所需的进程内会话状态，不保存市场时间序列。
    is_readonly = False

    def __init__(self) -> None:
        self.reset_request()

    def reset_request(self) -> None:
        """只在同一智能体请求的范围审查两步之间暂存目录，不跨轮复用。"""
        self._scope_review: tuple[Any, dict[str, Any], dict[str, Any]] | None = None
        self._completed_analysis: tuple[dict[str, Any], Any, str] | None = None

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.xuangu_fenxi import fenxi_xuangu
        from src.ashare.shichang_shuju import FenxiShujuShangxiawen

        request = dict(
            fanwei=str(kwargs.get("fanwei") or "all_market"),
            mingcheng=str(kwargs.get("mingcheng") or "").strip() or None,
            gupiao=str(kwargs.get("gupiao") or "").strip() or None,
            shuliang=kwargs.get("shuliang"),
            shizhi=kwargs.get("shizhi") if kwargs.get("shizhi") is not None else {"mode": "unresolved"},
            tiaojian=kwargs.get("tiaojian") if kwargs.get("tiaojian") is not None else {"mode": "none"},
            yanjiu_wenti=kwargs.get("yanjiu_wenti"),
        )
        request["shizhi"] = {key: value for key, value in request["shizhi"].items() if value is not None} if isinstance(request["shizhi"], dict) else request["shizhi"]
        choice = kwargs.get("fanwei_xuanze")
        if self._completed_analysis is not None:
            previous_request, previous_choice, previous_output = self._completed_analysis
            same_choice = choice is None or (isinstance(choice, dict) and isinstance(previous_choice, dict) and
                                            all(choice.get(key) == previous_choice.get(key) for key in ("code", "kind")))
            if request == previous_request and same_choice:
                return previous_output
        pending = self._scope_review
        self.reset_request()
        if pending is not None:
            request_context, original_request, original_result = pending
            if request != original_request or not isinstance(choice, dict):
                return json.dumps({**original_result, "error": "范围仍待确认；不能在审查时改变原始范围、数量或用户条件"}, ensure_ascii=False)
        elif choice is not None:
            return json.dumps({"status": "clarification_required", "outcome": "clarification_required", "stage": "scope_discovery",
                               "error": "没有本次请求的已核验范围候选，请重新说明要分析的范围"}, ensure_ascii=False)
        else:
            request_context = FenxiShujuShangxiawen()
        full_result = fenxi_xuangu(**request, context=request_context, scope_choice=choice)
        candidates = full_result.get("candidates") or []
        if pending is None and full_result.get("status") == "clarification_required" and full_result.get("stage") == "scope_discovery" and candidates and all(
            isinstance(candidate, dict) and (candidate.get("verification") or {}).get("verified") is True for candidate in candidates
        ):
            self._scope_review = (request_context, request, full_result)
            return json.dumps({**full_result, "status": "scope_review_required", "outcome": "scope_review_required",
                               "error": None, "error_code": None,
                               "candidates": [{**candidate, "ambiguity_resolution": "agent_review_pending"} for candidate in candidates],
                               "original_request": request,
                               "next_action": "结合完整用户语义比较这些真实候选；唯一合理解释可用 fanwei_xuanze 选择并保持原条件，仍有歧义调用 clarify 一次。不得以名称相似度、身份核验或热度代替语义判断。"}, ensure_ascii=False)
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
            output = json.dumps(public_result, ensure_ascii=False, separators=(",", ":"))
            self._completed_analysis = (request, choice, output)
            return output
        recommendation_available = bool(stored_result.get("recommendation_available"))
        public_result = {
            **_mianxiang_zhinengti_jieguo(stored_result),
            "analysis_id": analysis_id,
            "selected_stock": {key: stored_result["primary"].get(key) for key in ("ts_code", "name")} if stored_result.get("primary") else None,
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
        output = json.dumps(public_result, ensure_ascii=False, separators=(",", ":"))
        self._completed_analysis = (request, choice, output)
        return output


__all__ = ["FenxiScope", "GupiaoFenxiTool"]
