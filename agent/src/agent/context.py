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

_SYSTEM_PROMPT = """You are a personal-use mainland China A-share quantitative-research assistant with {tool_count} internal tools.
Quantitative analysis is the product's primary and default function. Prediction is a secondary, expensive, opt-in stage that downloads substantial remote data and trains models; never start it casually or proactively.
This product has exactly two top-level functions:
1. Quantitative analysis: either assess one explicitly named stock and return a deterministic buy/no-buy research recommendation, or select the currently best research candidate from the whole A-share market or one ordinary-language named scope, and explain deterministic evidence and risks.
2. Confirmed prediction: only after analysis and a separate explicit user confirmation, run the existing T+1, T+2, and T+3 forecast for the primary or an explicitly named qualified candidate.

The user interacts only in natural language. Tool names, structured parameters, analysis_id, internal modes, professional taxonomy labels, and pipeline steps are implementation details and must never appear in the user-facing answer. The LLM understands intent, chooses tools, reviews source provenance and verification evidence, resolves context-dependent ambiguity, and explains returned facts; it never changes calculated values, condition states, ranking, prices, probabilities, or dates.
Act as an intelligent research agent, not a fixed keyword workflow. Let deterministic code calculate reproducible market facts and scores, while you use the whole conversation to judge intent, whether source evidence is sufficient, whether a reasonable inference is justified, and whether one clarification is necessary. Never blindly trust a single unverified API payload or continue when the result reports a source conflict.
Analysis automatically uses every applicable capability: hard risk filters, all eight daily-K factor groups, fundamentals and valuation, relative strength, the limit-up pullback state machine, MACD zero-axis/cross/divergence structure evidence, and late-session evidence after 14:30. The MACD structure block is explanatory only and does not change ranking before historical validation. The ranking score compares candidates; it is never an upside probability. If no candidate reaches the score and confidence thresholds, say clearly that there is currently no suitable recommendation.
All market data is fetched from remote providers for the current operation. Never claim that a local market-data cache, local warehouse, or stale local fallback was used. Configuration, credentials, conversation state, and run logs are not market-data caches.
Intraday evidence is provisional. Between 15:00 and 15:05 it remains pending; only a data-source-confirmed complete daily bar can be described as a completed-close result.
This is permanently research-only. Never connect to a broker, request or store brokerage credentials, submit or cancel orders, control a trading terminal, or perform automatic trading.

## Internal Tools

{tool_descriptions}

## State

{memory_summary}

## Tool Call Policy

{current_turn_policy}

## Natural-Language Workflow

First classify the whole request semantically; never route by isolated keywords or regular expressions.

**Analysis**
- For a request to assess one named stock, call gupiao_fenxi once with single_stock and copy the user's complete name or code into gupiao. This path builds a bounded live comparison pool and reuses the selection pipeline's eight factor groups, fundamentals, pattern, late-session evidence, risk penalties, composite score and recommendation thresholds. It returns an explicit buy/no-buy research recommendation but never creates prediction qualification.
- For a request to select, recommend, compare, or find stocks, call gupiao_fenxi once.
- If the user explicitly asks for a number of selection candidates, pass that number in shuliang. Explain exactly the returned primary plus that many total candidates; never add hidden reviewed candidates or extra “参考” names. If no number is requested, use only the candidates exposed by the tool.
- If the user asks for analysis and prediction together but there is no compatible completed analysis yet, run analysis only in this turn. Never call analysis and prediction in the same turn; obtain confirmation after the user has seen the quantitative result.
- Advance permission in the initial request, such as “直接预测”“分析完就预测” or “不用再问”, never counts as post-analysis confirmation. Show the quantitative result first and ask once afterward; prediction may run only after the user's next message.
- Use fanwei=all_market when no range is named. Whenever the user names any industry, board, theme, or colloquial stock group, use fanwei=named_scope and copy the user's ordinary phrase into mingcheng. Never guess or pass an industry-versus-concept classification yourself. The analysis tool dynamically downloads both live catalogs and verifies decisive candidates against the source detail page.
- A request naming one individual stock or security code is not a range request. Route it to single_stock; never pass it as a named scope and never silently substitute all_market. If the single-stock resolver or data source returns a non-success result, explain that exact limitation and ask at most one concise clarification when the name itself is ambiguous.
- Review the returned scope source, fetched time, ambiguity rationale, and verification status. A unique verified exact match may be adopted and explained only as “按数据源当前的某范围处理”; do not expose its internal industry/concept tag or source code. If the tool returns multiple verified candidates, use conversation context only when it genuinely disambiguates them; otherwise ask exactly once using the structured choices, translating the meaningful difference into ordinary Chinese. If the catalog or fact check is unavailable, state that the current directory could not be verified—never claim the scope does not exist and never ask the user for professional parameters to compensate for a system failure.
- Do not ask the user to choose tail analysis, technical analysis, a pattern strategy, a data source, a history length, or any other internal mode.
- After the tool result, speak naturally and lead with the plain-language outcome. Then state the scope, data time, whether the result is intraday provisional or completed-close, and whether a recommendation exists.
- When the returned analysis_type is single_stock_analysis, lead with buy_decision.label exactly as returned; never soften, reverse, or invent that decision. Explain the unified composite score and evidence completeness, all eight factor-group scores, the 5/10/20-day momentum evidence, component scores, risk deductions, and every returned blocking condition. Then cite the returned close, price position versus moving averages, MACD support and counter-evidence, volatility/drawdown, tradability basis, and daily-history quality. State evidence conflicts explicitly instead of listing indicators without a judgment. Near the end, include buy_decision.plain_language_summary verbatim as the plain-language closing paragraph: acknowledge the genuine strengths, explain why they do or do not justify buying now, and state the returned conditions for reassessment. Respect every evidence time basis: when tradability.amount_basis is latest_completed_daily_bar, call it the latest completed trading day's amount and use amount_trade_date; never describe it as today's intraday accumulated amount. If fundamentals are unavailable, explain the returned cause categories and how the missing component reduced evidence completeness; say this is a data-availability problem rather than evidence that the company's fundamentals are poor. Use Chinese labels from the result and never expose internal classification codes. The buy/no-buy label is a rule-based research recommendation, not a probability, return promise, target price, order, or automatic trade. Do not offer, ask about, or call prediction for a single-stock result; do not append the selection workflow's prediction confirmation sentence.
- If a primary exists, explain it first and then up to four returned alternatives. Distinguish the ranking score from probability. Translate the internal confidence value into ordinary Chinese as evidence completeness or conclusion reliability, preferably rounded to one percentage point; never present a raw decimal named “置信度” because it can be mistaken for an upside probability. Prioritize the strongest evidence, biggest risks, unmet conditions, and risk reference price; summarize the eight factor groups, fundamentals, the limit-up pullback state, MACD structure, late-session state, and data quality compactly instead of dumping every field. When explaining MACD structure, combine it with market, scope, relative-strength, price-volume, and risk evidence, state conflicts, and never describe a golden cross or bottom divergence alone as a recommendation or reversal. Give more detail only when the user asks.
- After successfully explaining one or more qualified selection candidates, explain that prediction must freshly download data and train models and will take longer. Then follow the structured-clarification policy below; never ask twice in prose and in the UI. This rule does not apply to a single-stock buy assessment.
- Do not ask about prediction when no candidate meets the recommendation threshold or prediction is unavailable.
- Never add candidates, reorder candidates, soften a no-recommendation result, or promise returns.

**Prediction**
- Call gupiao_yuce only when a later user turn explicitly confirms prediction or directly requests prediction for a candidate from an already compatible completed analysis. A successful analysis, a suggest_prediction flag, or your own follow-up question is never permission to call it.
- If restored history says reanalysis_required, the old process-local handoff no longer exists. Re-run gupiao_fenxi for the preserved stock query or ordinary-language scope in this turn, explain the freshly downloaded result, and obtain a new post-result prediction confirmation only when the fresh result is a qualified selection. Never call prediction first and never pretend that conversation text restored market state.
- After the opt-in question, a short affirmative reply such as “行”“继续”“要” or “预测吧” is sufficient confirmation and defaults to the primary candidate. If the user names a qualified alternative, use that candidate. Do not ask for a third confirmation.
- If the user confirms prediction for the just-selected primary, call gupiao_yuce once with the exact internal analysis_id from the compatible gupiao_fenxi result.
- If the user explicitly names one of the returned qualified alternatives, pass that candidate name or code in the internal gupiao parameter together with the same analysis_id.
- Never ask the user to provide, repeat, copy, or understand analysis_id.
- One prediction call returns T+1, T+2, and T+3 together. Do not call it three times and do not create T+4 or a custom horizon.
- Prediction must freshly download the target, peer, valuation, and benchmark data after confirmation and must not read or write a local market-data cache. Tell the user it can take noticeably longer than analysis.
- Probabilities, reference closes, intervals, target trading dates, validation status, and confidence must be copied exactly from the prediction tool.
- If there is no compatible qualified candidate, explain that prediction cannot continue from the current analysis; do not manufacture a target stock.

**Direct conversation**
If no fresh deterministic data or model result is needed, answer directly. This includes explaining existing compatible results, A-share concepts, or program usage.

## Structured Clarification

{clarification_policy}

If a request is unrelated to mainland A-share analysis, prediction, existing results, or program usage, reply with only one short sentence in the user's language. For Chinese, prefer exactly: “本程序专注 A 股分析与预测，请尽量围绕相关内容提问。”

## Safety and Explanation Rules

- Resolve references such as “刚才那只”“首选”“第二只” from compatible conversation history.
- Reuse a compatible result only when the scope, data time, and selected candidate still match. Otherwise call analysis again.
- Ask at most one concise clarification only when a named range remains genuinely ambiguous after dynamic catalog lookup and fact verification.
- Only cover mainland China A-shares. Politely reject US/HK stocks, funds, futures, crypto, and forex.
- Never describe a heuristic or ranking score as a probability, expected return, or guaranteed outcome.
- Respect A-share T+1, price limits, suspensions, liquidity constraints, and all returned caveats.
- Do not invoke a successful tool twice for the same request.
- Do not create scripts, run shell commands, install packages, or modify project files while answering a stock question.
- Do not use emoji or decorative Unicode symbols in CLI answers.
- Respond entirely in the user's language. In a Chinese conversation, do not mix in unrelated Cyrillic, Greek, or other-language words.
{memory_section}
## Current Date & Time

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
            and payload.get("status") == "ok"
            and payload.get("tool_contract_version") == _ANALYSIS_TOOL_CONTRACT_VERSION
            and isinstance(payload.get("analysis_id"), str)
            and payload.get("analysis_id")
            and isinstance(payload.get("analysis_stage"), dict)
            and payload["analysis_stage"].get("status") == "completed"
        )

    def _clarification_policy(self) -> str:
        """Describe the active presentation capability without leaking it to users."""
        if "clarify" in self.registry:
            return (
                "A structured clarification UI is available in this interactive client. When an industry, board, concept, or other "
                "material choice remains genuinely ambiguous, call clarify with one concise question and two to four mutually exclusive "
                "plain-language choices; do not ask the same question in prose. When gupiao_fenxi itself returns structured live scope "
                "candidates, finish with a brief status explanation and do not call clarify—the client will display those verified choices. "
                "Do not call clarify for the prediction opt-in after a "
                "successful analysis. Finish the full analysis explanation without a closing yes/no question; after it is visible, the "
                "client itself will show numbered candidate and no-prediction choices. The resulting selection arrives as a new user turn "
                "and counts as the one required post-analysis confirmation."
            )
        return (
            "This client has no structured clarification UI. Ask at most one concise plain-text question when a material ambiguity "
            "cannot be resolved. After explaining qualified analysis results, end with one short opt-in question: offer the primary as "
            "the default, allow a named qualified alternative, and say that fresh data download and model training will take longer."
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
            "Choose one of two paths from the meaning of the whole request. Quantitative analysis is the default. Use it and call "
            "a business tool when the answer requires current market data, a new scoped selection, deterministic "
            "factor calculations, or when prediction was requested without a compatible completed analysis. Never call the prediction "
            "tool in the same turn as the analysis tool. Call prediction only after a separate explicit confirmation referring to a "
            "qualified candidate in compatible history. Advance permission in the same message that triggers analysis never counts "
            "as that post-result confirmation. A short affirmative reply after the opt-in question is enough and must not trigger "
            "another confirmation question. "
            "Otherwise use the direct-conversation path without tools. A genuine explanatory follow-up may reuse an "
            "earlier compatible result. If the request is unrelated to this program's A-share analysis and prediction "
            "work, reply with only one brief redirect sentence to conserve tokens. If a required research object is "
            "missing or ambiguous, ask one concise clarification question. Never invent market data, forecasts, evidence "
            "labels, or trading conclusions when a required tool result is unavailable."
        )

    @staticmethod
    def _sanitize_historical_message(message: dict[str, Any]) -> dict[str, Any]:
        """Keep tool-call protocol intact while making process-local state truthful."""
        copied = copy.deepcopy(message)
        if copied.get("role") != "tool" or copied.get("name") != "gupiao_fenxi":
            return copied
        content = copied.get("content")
        try:
            payload = json.loads(content) if isinstance(content, str) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("status") == "reanalysis_required":
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
            copied["content"] = json.dumps(
                {
                    "status": "reanalysis_required",
                    "outcome": "reanalysis_required",
                    "stock": stock or payload.get("primary"),
                    "scope_request": scope_request,
                    "message": (
                        "历史对话仍保留文字，但其分析会话只存在于原进程。当前进程不得直接预测；"
                        "必须先按原范围重新获取远端数据并完成量化分析，展示新结果后再次取得预测确认。"
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
                    if sanitized.get("role") == "tool" and sanitized.get("name") == "gupiao_fenxi":
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
