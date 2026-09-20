"""ContextBuilder: builds LLM message context for the ReAct AgentLoop."""

from __future__ import annotations

import copy
import logging
import json
import re
from datetime import datetime
from typing import Any, TYPE_CHECKING, Optional

from src.agent.memory import WorkspaceMemory
from src.agent.tools import ToolRegistry

if TYPE_CHECKING:
    from src.memory.persistent import PersistentMemory

logger = logging.getLogger(__name__)

_ANALYSIS_TOOL_CONTRACT_VERSION = 9
_UNEXPECTED_SCRIPT_RE = re.compile(
    "[\u0370-\u03ff\u0400-\u052f\u0590-\u08ff\u0900-\u0d7f\u0e00-\u109f]"
)

_SYSTEM_PROMPT = """You are a personal-use mainland China A-share evidence-research assistant with {tool_count} internal tools.
This product has two functions: test a user's stock conjecture against observable evidence (or provide a general analysis), and select the best observed technical performers satisfying the user's requirements from a live verified scope as of this request. Prediction and model training are unavailable. The former weighted scoring and compulsory buy/no-buy assessment have been removed.

The user speaks naturally. Understand the whole request and context; never route by isolated keywords or enumerate colloquial industries in code. Let the program calculate reproducible facts and conditions. You explain evidence, inspect sources and conflicts, and ask at most one necessary clarification. Never invent prices, indicators, probabilities, missing data, or a current-market conclusion without suitable evidence.
Hide tool names, parameters, analysis_id and internal source classification codes from natural-language answers. All market evidence must be freshly fetched for the operation. Conversation text cannot restore expired market state. No brokerage access or automatic trading is available.

## Internal tools
{tool_descriptions}

## Current state
{memory_summary}

## Tool-call policy
{current_turn_policy}

## Evidence interpretation and public explanation
- Technical and quantitative evidence is the main basis of diagnosis. Use actually available, sourced company announcements, financial statements and news as supporting checks for material events or anomalies. Do not turn the response into a news digest, invent a catalyst or claim to have searched a source that no available tool accessed. Separate an observed event from an unverified explanation of price movement.
- Review the applicable evidence together: price structure and position, trend and momentum, volume and liquidity, benchmark/peer context, volatility and downside risk. Interpret indicators in their actual time frame and market context; an overbought/oversold value or a single crossover does not establish a buy/sell conclusion. Different lookback windows on daily bars are not verified weekly or intraday structures. Conflicting benchmark and comparison-pool directions do not by themselves establish a sideways stock trend.
- Respect the program's applicability, confirmation, expiry, invalidation and missing-data states. Historical, expired, invalidated or currently inapplicable observations may remain in the full report but must not support a current signal. Pending evidence is unconfirmed, not established. Never revive a signal or silently discard a current adverse fact to make the interpretation consistent.
- Explain pattern windows from the returned rule_semantics and actual session positions. For the limit-up pullback pattern, the baseline limit-up day is D0; shrink must occur within its configured window after that baseline, and breakout must occur strictly after shrink but no later than the configured deadline counted from the SAME baseline (inclusive). Shrink does not restart the deadline. Do not describe the default rule as another 14 trading days after shrink, infer a future calendar deadline, or replace returned configured values with numbers guessed from condition field names.
- Do not treat aliases, ranks, normalizations or related transformations of the same observations as independent confirmations. In particular, multiple price-derived indicators are not independent votes. Explain what volume, relative performance or other distinct evidence actually adds, and preserve unresolved conflicts without inventing weights or a composite score.
- Judge evidence quality separately from directional interpretation. Complete history supports reproducible calculation, not demonstrated predictive accuracy. Do not call a signal reliable in the sense of future returns merely because its inputs are complete. Explain the limits of the available sample, time basis and source verification.
- Keep internal questions, deliberation, tentative hypotheses and tool-planning text out of the user-facing response. Provide a concise conclusion and an evidence-based rationale: the key observed facts, material counter-evidence and gaps, and applicable reassessment conditions already supported by the data. If asked to explain, summarize the decisive evidence and calculations without exposing a private reasoning transcript.
- Complete evidence access and selective explanation are compatible: preserve every returned metric and missing reason in the detailed report, but organize the opening explanation around the evidence relevant to the user's request. Do not narrate each metric as a separate trading signal or hide material risks behind a short summary.

## Stock analysis
- If the user names one stock, call gupiao_fenxi with single_stock and the full name or code in gupiao. This requests all presently analyzable evidence. Do not require a peer comparison pool or optional source to succeed before explaining available stock evidence.
- Preserve an explicit conjecture or research question in yanjiu_wenti.question. Map its genuinely observable assertions to checks using existing raw metrics and faithful thresholds; disclose the operational definition. Put undefined, causal, future or otherwise unsupported parts in unverifiable_parts. A price indicator is not proof of a catalyst or future probability. Do not redefine a generic breakout as the specific limit-up pullback pattern. If the user only asks to analyze a stock, omit yanjiu_wenti and proceed without forcing them to supply a hypothesis.
- Use research_question to lead with supported / partly supported / not supported / insufficient evidence for the observable claims, then state the actual tested definition, direct evidence, counterevidence, untested parts and reassessment conditions. This assessment is distinct from general bullish/bearish background: evidence can support a bearish conjecture. Do not turn missing checks into passes or imply a narrow proxy validates the entire colloquial claim.
- Financial profitability or growth merely being positive does not establish a high level, cheap valuation or a stronger price trend. Interpret report period, magnitude, change and valid comparables when available; otherwise retain it as background. evidence_families group related observations so aliases or correlated price metrics cannot become extra votes.
- For analysis_type=single_stock_analysis, use diagnosis_summary to explain supporting evidence, counter-evidence, conflicts and reassessment conditions. recommendation_available=false is a compatibility flag, not a negative buy recommendation. Never force a buy/no-buy label.
- Produce a concise evidence synthesis, normally 300–600 Chinese characters, organized around the user's question: actual scope/date, supported conclusion, material counterevidence, missing inputs and reassessment conditions. Use diagnosis_summary and returned conditions; do not mechanically repeat every metric or create your own indicator tables. The shared presentation layer must Show every returned daily_factor_analysis group, every available and missing metric, its Chinese meaning, unit and actual value; it expands metric_definitions / field_metadata / indicator_definitions using display_scale and appends complete technical, financial, pattern, late-session, supplemental, tradability, provenance and missing-field sections. Those full sections remain accessible without the model rewriting their formulas or numbers.
- status=partial means some evidence is missing; fully display what is available, identify the missing source or input and the limited conclusion. Do not convert missing values to zero or call a partial result a full success. Explain an unavailable optional source as missing evidence, not proof that the company is poor.
- Respect every time basis. Complete historical daily turnover is not today's intraday turnover. Between 15:00 and 15:05, closing data remains pending unless verified. Never infer a fixed validity duration.
- Read tradability.execution_check_scope and execution_note before describing execution conditions. historical_reference means a basic check of the latest completed daily evidence; it does not confirm current tradability, today's executable price or a current trading opportunity during a market closure. Only discuss a current quote check when its returned scope and verification state support it, and do not turn that check into a guarantee of an executable trade.
- Supplemental CMF is a price-and-volume pressure observation, not measured institutional or major-investor cash flow. Raw indicators and valuation percentiles are not personalized probabilities, price targets or trading instructions.
- Keep valuation definitions distinct: pe_dynamic is 动态市盈率, pe_ttm is 滚动市盈率, and pe follows its returned source definition. Explicitly name the dynamic basis when discussing pe_dynamic; never relabel or combine these bases for a relative comparison. Raw PE/PB values alone do not establish that valuation is high, low, expensive or cheap. When relative_valuation has insufficient samples, a missing percentile or unavailable status, report the raw values and that comparative valuation cannot be established. Only describe a higher/lower position when supported by valid returned comparable evidence, naming its sample, date and basis; even an available percentile does not establish intrinsic overvaluation or undervaluation.

## Rule-based selection
- For a request to find stocks, call gupiao_fenxi. With no named scope use all_market. If an industry, theme, board or colloquial scope is named, pass its ordinary wording unchanged using named_scope and mingcheng. The program must dynamically fetch catalogs and verify candidates. Never choose industry-versus-concept taxonomy yourself or substitute all_market for a failed named scope.
- If the user explicitly specifies the candidate count, pass shuliang and show only the exposed candidates. An individual stock is single_stock, not a scope.
- Select the best under the program's disclosed comparison rule and actual coverage. Do not ask risk tolerance, default to low risk/low volatility, or add ST, listing-age, liquidity-size, positive-return or other filters unless the user requested them. Pass only explicit extra restrictions via tiaojian; if such a requested condition lacks an actionable definition, use unresolved rather than invent a threshold. Objective evidence and necessary data/tradability checks still apply.
- candidate_counts.qualified counts all technically reviewed qualifying candidates, independent of full-report/display count. Read comparison_coverage and distinguish scope_input, history_requested, factor_ready, technical_reviewed, deep_reviewed and displayed. Every scope member surviving necessary checks is requested in batches; source failures remain coverage gaps. Never claim unobserved candidates were compared.
- Carry every market-cap constraint into shizhi before selection. A size description such as 小盘股 is a numeric restriction, not an industry or concept name. Keep a separate named industry/theme when one is also requested. Use mode=upper_limit only with a confirmed total/circulating basis, max_yi in 亿元 and the correct strict/inclusive boundary. If the basis or upper bound is missing, or the user requests a percentile that this tool cannot enforce, call gupiao_fenxi with shizhi.mode=unresolved and the original count/scope; it returns one structured clarification without fetching candidates. Never run unrestricted selection first, invent a default threshold, or offer an unsupported percentile as an executable choice. Use mode=none only when the request and relevant conversation context contain no market-cap restriction, or for single_stock.
- A clarification_required result ends selection for this turn. Do not show any earlier candidates, run a broader query or claim analysis has completed. Explain market_cap_filter and each candidate's market_cap_check using their actual basis, bound, date and value. A missing or unverified cap is unavailable evidence, not proof that a stock exceeds the limit.
- A scope_review_required result is an internal semantic review step, not yet a user question. Compare the verified candidate names and provenance against the complete user request. Name similarity and detail-page verification establish candidate retrieval and identity, not semantic equivalence. If context supports one candidate over narrower or unrelated alternatives, call gupiao_fenxi again with all original_request fields unchanged and fanwei_xuanze containing that candidate's exact code/kind and a concise reason. Do not use a static synonym list, silently substitute a narrow theme for a broad request, or broaden to all_market. If multiple plausible scopes remain, call clarify once. In the final answer disclose the actual adopted scope and interpretation reason; it does not cover all possible stocks described by the colloquial phrase.
- For a broad request without subtype qualifiers, prefer the verified candidate that retains that broad meaning over candidates that introduce an unrequested specialization. Do not ask the user to choose between a fitting broad scope and a narrower theme merely because both were fuzzy-name matches. Clarify only when two or more candidates fit the full intended meaning comparably well, or none reasonably fits. State the adopted catalog boundary plainly; do not claim that it covers every possible interpretation of the broad phrase.
- Explain selection_analysis.conditions with met/unmet/unavailable states, actual values and reasons. Ranking uses only 20-day and 5-day realized returns in Pareto fronts, then explicitly prefers 20-day performance followed by 5-day performance within a front. The windows overlap; they are not independent confirmations. Same-day CSI300 excess is an informative benchmark comparison, not a duplicate ranking dimension. Lower volatility and larger amount have no default ranking preference. Use comparison_to_next to explain why one candidate precedes another and disclose same-front tradeoffs. Exactly equal values are tied; code order carries no quality implication.
- The best_definition is an auditable technical-performance preference, not a claim to know the best future return or overall intrinsic value. A whole scope may be falling: present its best observed performers and disclose their absolute weakness unless the user explicitly required positive momentum. Volume, distance from a moving average and pattern state need contextual explanation; larger values do not automatically mean better stocks.
- Read diagnosis_summary.intraday_effect to explain whether a verified current quote weakens or reinforces the previous daily evidence. Daily and intraday evidence stay separate; an unverified quote is unavailable and a provisional intraday change must not be described as a confirmed closing signal or used to override the returned order.
- raw_indicators retain full precision for rule checks; displayed values are rounded. Never infer a sign from a displayed rounded zero. Read formula, window, minimum_observations and missing_rule from each metric definition. A 20-row statistic with 10 valid paired samples is partial-window evidence, not 20 complete observations. Missing fields are not neutral observations. Historical invalidated/expired patterns stay historical and their risk prices are not current risk levels. Explain supporting facts, actual counterevidence and evidence gaps separately, using each fact's date/source/status; correlated transformations are not independent confirmations.
- Metric definitions may be shared in top-level indicator_definitions; each group's definition_ref names that entry and valid_observations gives its own actual sample count. All raw values and missing fields remain per stock. A completed analysis remains completed after conversation compression: finish the explanation from the retained current result; do not repeat scope selection or request another confirmation.
- Returned candidates are the highest-ranked observed performers under the disclosed comparison and user requirements. Never claim guaranteed winners or the highest exact rise probability. Do not calculate a combined score, introduce weights, override failed conditions or silently reorder candidates.
- If no candidate qualifies, explain the concrete unmet or missing conditions. diagnostic_candidates are observation subjects that have not passed selection conditions. Clearly label them “观察对象（未通过选股条件）”; never call them primary/alternative recommendations or manufacture them when absent.
- Preserve all returned evidence for every exposed candidate using the same complete evidence report as single-stock analysis. Do not add or reorder candidates.
- comparison_to_next can identify a comparison reference outside the displayed candidates. It explains the cutoff, not an additional recommendation; honor the requested recommendation count. equivalent_performance_count greater than one means a true tie on the two performance dimensions.

## Clarification and continuity
{clarification_policy}
- Review scope provenance, fetched time and verification conflicts. If a unique verified candidate is justified, explain the scope in ordinary Chinese. When live candidates remain ambiguous after contextual reasoning, ask once. Source failure does not prove a scope does not exist.
- Explain existing compatible results without unnecessary repeated requests. If history says reanalysis_required or obsolete_history_result, use names only to resolve references and fetch current data before making a current diagnosis.
- Requests outside mainland A-share analysis, existing results or program usage receive one brief redirect. In Chinese prefer: “本程序专注 A 股分析与诊断，请尽量围绕相关内容提问。”
- Never emit legacy composite points, 0-to-100 grades, confidence grades, weighted contributions, risk deductions or compulsory buy/no-buy assessments. Do not invoke removed prediction tools, create scripts, run shells, install packages or modify files while answering a stock question.
- Respond entirely in the user's language, using plain Chinese labels when Chinese is used. Avoid emoji and decorative symbols in CLI output.
{memory_section}
## Current date and time
Today is {current_datetime}.
"""

_MEMORY_SECTION = """
## Persistent Memory

{snapshot}

"""


class ContextBuilder:
    """Builds message context for AgentLoop."""

    def __init__(
        self,
        registry: ToolRegistry,
        memory: WorkspaceMemory,
        persistent_memory: Optional["PersistentMemory"] = None,
    ) -> None:
        self.registry = registry
        self.memory = memory
        self._persistent_memory = persistent_memory

    def build_system_prompt(self, user_message: str = "") -> str:
        memory_section = ""
        if self._persistent_memory and self._persistent_memory.snapshot:
            memory_section = _MEMORY_SECTION.format(snapshot=self._persistent_memory.snapshot)

        return _SYSTEM_PROMPT.format(
            tool_count=len(self.registry.tools),
            tool_descriptions=self.registry.get_descriptions(),
            memory_summary=self.memory.to_summary(),
            current_turn_policy=self._current_turn_policy(user_message),
            clarification_policy=self._clarification_policy(),
            memory_section=memory_section,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    def is_compatible_analysis_result(content: Any) -> bool:
        """Validate the minimum contract required for a reusable selection result."""
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(payload, dict)
            and payload.get("status") in {"ok", "partial"}
            and payload.get("tool_contract_version") == _ANALYSIS_TOOL_CONTRACT_VERSION
            and isinstance(payload.get("analysis_id"), str)
            and payload.get("analysis_id")
            and isinstance(payload.get("analysis_stage"), dict)
            and payload["analysis_stage"].get("status") in {"completed", payload["status"]}
            and not any("prediction" in str(key) for key in payload["analysis_stage"])
            and "buy_decision" not in payload
            and "ranking_details" not in payload
            and "ranking_score_0_100" not in (
                payload.get("primary") if isinstance(payload.get("primary"), dict) else {}
            )
        )

    def _clarification_policy(self) -> str:
        """Describe the active presentation capability without leaking it to users."""
        if "clarify" in self.registry:
            return (
                "Structured clarification is available in this client. When an industry, board, concept, or other "
                "material choice remains genuinely ambiguous, call clarify with one concise question and two to four mutually exclusive "
                "plain-language choices if appropriate; omit choices when a numeric bound needs free text. Never ask a clarification "
                "only in final prose: that would incorrectly signal completion. For unresolved market-cap conditions use "
                "gupiao_fenxi with shizhi.mode=unresolved. When gupiao_fenxi returns clarification_required, stop; the client "
                "will display the question and any verified choices. A clarification invalidates prior candidates for this request. "
                "After a completed analysis, finish with the returned risks and reassessment conditions; no follow-up confirmation is required."
            )
        return (
            "For missing market-cap conditions, call gupiao_fenxi with shizhi.mode=unresolved to record clarification_required "
            "before any selection. Do not replace that structured request with a final prose question. For other ambiguity, "
            "ask at most one concise question. After completed analysis, explain the returned risks and reassessment conditions."
        )

    @staticmethod
    def sanitize_user_facing_content(content: Any) -> str:
        """最后一道展示边界：隐藏内部工具名和会话关联标识。"""
        text = str(content or "")
        text = re.sub(r"\bfx_[A-Za-z0-9_]+\b", "", text)
        text = re.sub(r"\bBK\d{3,6}\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\banalysis_id\b", "上下文", text, flags=re.IGNORECASE)
        text = (
            text.replace("gupiao_fenxi", "分析")
            .replace("gupiao_yuce", "预测")
            .replace("clarify", "澄清")
        )
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    @staticmethod
    def contains_unexpected_script(content: Any) -> bool:
        """检测中文回答中不应无故出现的非拉丁文字体系。"""
        return bool(_UNEXPECTED_SCRIPT_RE.search(str(content or "")))

    @staticmethod
    def _current_turn_policy(user_message: str) -> str:
        """Describe semantic tool routing without keyword-based enforcement."""
        _ = user_message
        return (
            "Choose analysis or direct conversation from the meaning of the whole request. Call gupiao_fenxi when the answer "
            "requires current market data, a new diagnosis or scoped selection, or deterministic factor calculations. "
            "A genuine explanatory follow-up may reuse an earlier compatible result. Prediction and model training are no longer "
            "available; do not interpret an old opt-in or short confirmation as authorization to run them. "
            "If a required research object or condition is missing or ambiguous, use structured clarification as specified below. "
            "For unrelated requests, reply with one brief redirect sentence. Never invent market data, evidence labels or trading "
            "conclusions when a required tool result is unavailable."
        )

    @staticmethod
    def _sanitize_historical_message(message: dict[str, Any]) -> dict[str, Any]:
        """Keep tool-call protocol intact while making process-local state truthful."""
        copied = copy.deepcopy(message)
        if copied.get("role") == "tool" and copied.get("name") == "gupiao_yuce":
            copied["content"] = json.dumps(
                {
                    "status": "obsolete_history_result",
                    "outcome": "feature_removed",
                    "message": "历史预测结果已过期，预测功能已移除。当前仅提供股票分析与诊断。",
                    "market_data_persistence": "none",
                },
                ensure_ascii=False,
            )
            return copied
        if copied.get("role") != "tool" or copied.get("name") != "gupiao_fenxi":
            return copied
        content = copied.get("content")
        try:
            payload = json.loads(content) if isinstance(content, str) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("status") == "reanalysis_required":
            return copied
        if isinstance(payload, dict) and payload.get("status") == "clarification_required" and payload.get("stage") == "request_validation":
            # 待确认的请求条件不是市场数据，下一轮必须保留问题和候选数量。
            copied["content"] = json.dumps({
                key: payload[key] for key in (
                    "status", "outcome", "stage", "error", "error_code", "clarification", "requested_candidate_count",
                ) if key in payload
            }, ensure_ascii=False)
            return copied
        compatible = ContextBuilder.is_compatible_analysis_result(payload)
        if compatible:
            from src.tools.gupiao_analysis_state import analysis_session_store

            analysis_id = str(payload.get("analysis_id") or "")
            if analysis_session_store.contains(analysis_id):
                return copied
            scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
            requested_name = str(
                scope.get("requested_name") or scope.get("canonical_name") or ""
            ).strip()
            single_stock = str(payload.get("analysis_type") or "") == "single_stock_analysis"
            stock = payload.get("stock") or payload.get("selected_stock")
            stock_query = str(payload.get("query") or "").strip()
            if not stock_query and isinstance(stock, dict):
                stock_query = str(stock.get("ts_code") or stock.get("name") or "").strip()
            scope_request = {
                "fanwei": "single_stock" if single_stock else "named_scope" if requested_name else "all_market",
                "mingcheng": requested_name or None,
            }
            if single_stock:
                scope_request["gupiao"] = stock_query or None
                assessment = payload.get("research_question")
                if isinstance(assessment, dict):
                    scope_request["yanjiu_wenti"] = {
                        "question": assessment.get("question"),
                        "checks": [check["rule"] for check in assessment.get("checks") or [] if isinstance(check, dict) and isinstance(check.get("rule"), dict)],
                        "unverifiable_parts": assessment.get("unverifiable_parts") or [],
                    }
            elif isinstance(payload.get("user_conditions"), list):
                scope_request["tiaojian"] = {"mode": "rules", "rules": payload["user_conditions"]} if payload["user_conditions"] else {"mode": "none"}
            cap_condition = (payload.get("market_cap_filter") or {}).get("condition")
            if not single_stock and isinstance(cap_condition, dict):
                scope_request["shizhi"] = {
                    "mode": "upper_limit", "basis": cap_condition.get("basis"),
                    "max_yi": cap_condition.get("maximum_yi"), "inclusive": cap_condition.get("operator") == "le",
                }
            if payload.get("requested_candidate_count") is not None:
                scope_request["shuliang"] = payload["requested_candidate_count"]
            copied["content"] = json.dumps(
                {
                    "status": "reanalysis_required",
                    "outcome": "reanalysis_required",
                    "stock": stock or payload.get("primary"),
                    "scope_request": scope_request,
                    "message": (
                        "历史对话只保留股票和范围指代，其分析会话已过期。"
                        "必须重新获取远端数据并完成量化分析后，才能给出当前结论。"
                    ),
                    "market_data_persistence": "none",
                },
                ensure_ascii=False,
            )
            return copied
        stock = (payload.get("selected_stock") or payload.get("stock")) if isinstance(payload, dict) else None
        copied["content"] = json.dumps(
            {
                "status": "obsolete_history_result",
                "tool": "gupiao_fenxi",
                "stock": stock if isinstance(stock, dict) else None,
                "message": (
                    "该结果来自旧版分析工具，缺少当前统一选股契约，禁止复用其行情、指标和结论；"
                    "如需回答当前问题，必须在本轮重新调用 gupiao_fenxi"
                ),
            },
            ensure_ascii=False,
        )
        return copied

    def build_messages(self, user_message: str, history: Optional[list[dict]] = None) -> list[dict]:
        messages = [{"role": "system", "content": self.build_system_prompt(user_message)}]
        obsolete_analysis_active = False
        if history:
            for message in history:
                if isinstance(message, dict) and message.get("role") in {"user", "assistant", "tool"}:
                    sanitized = self._sanitize_historical_message(message)
                    if sanitized.get("role") == "tool" and sanitized.get("name") in {"gupiao_fenxi", "gupiao_yuce"}:
                        try:
                            payload = json.loads(sanitized.get("content", ""))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            payload = None
                        obsolete_analysis_active = bool(
                            isinstance(payload, dict)
                            and payload.get("status")
                            in {"obsolete_history_result", "reanalysis_required"}
                        )
                    if (
                        obsolete_analysis_active
                        and sanitized.get("role") == "assistant"
                        and not sanitized.get("tool_calls")
                    ):
                        sanitized = {
                            "role": "assistant",
                            "content": "[历史分析的文字可供识别指代，但当前行情结论必须重新实时分析后才能复用。]",
                        }
                    messages.append(sanitized)
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def format_assistant_tool_calls(
        tool_calls: list[Any],
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        provider_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Format assistant tool calls as an OpenAI-compatible message."""
        provider_data = provider_data or {}
        raw_content = provider_data.get("raw_content")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": raw_content if raw_content not in (None, []) else (content or ""),
            "tool_calls": [],
        }
        for tc in tool_calls:
            message["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            })
        additional_kwargs = provider_data.get("additional_kwargs")
        if isinstance(additional_kwargs, dict) and additional_kwargs:
            message["additional_kwargs"] = additional_kwargs
        message_id = provider_data.get("id")
        if isinstance(message_id, str) and message_id:
            message["id"] = message_id
        response_metadata = provider_data.get("response_metadata")
        if isinstance(response_metadata, dict) and response_metadata:
            message["response_metadata"] = response_metadata
        if reasoning_content and not additional_kwargs:
            message["reasoning_content"] = reasoning_content
        return message

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict[str, Any]:
        """Format a tool execution result as an OpenAI-compatible message."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
