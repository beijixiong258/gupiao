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

_SINGLE_STOCK_TOOL_CONTRACT_VERSION = 2
_FRESH_REQUEST_MARKERS = (
    "现在",
    "今天",
    "今日",
    "当前",
    "目前",
    "最新",
    "刚刚",
    "刚才收盘",
    "能不能买",
    "是否能买",
    "是否可以买",
    "可不可以买",
    "要不要买",
    "值得买吗",
)
_SINGLE_STOCK_DECISION_MARKERS = (
    "能不能买",
    "是否能买",
    "是否可以买",
    "可不可以买",
    "要不要买",
    "值得买吗",
)
_BOARD_REQUEST_MARKERS = (
    "板块",
    "行业",
    "概念",
    "选股",
    "股票池",
    "哪些股票",
    "几只股票",
)
_BROAD_MARKET_MARKERS = (
    "大盘",
    "沪指",
    "深成指",
    "创业板指",
    "整个市场",
    "全市场",
)
_OBJECT_CLARIFICATION_MARKERS = (
    "哪只股票",
    "具体股票",
    "股票名称",
    "股票代码",
    "具体是哪只",
    "指的是哪只",
    "请提供股票",
    "请告诉我股票",
    "哪个板块",
    "板块名称",
    "具体板块",
    "请提供板块",
)

_SYSTEM_PROMPT = """You are an A-share T+3 quantitative research agent with {tool_count} business tools.
This product has exactly two primary workflows. It covers only mainland China A-share stocks and mainland exchange rules.
Use symbols like 000001.SZ, 600519.SH, or 430047.BJ. Market-data source must be "auto", "tushare", or "akshare".
The external LLM provider can be DeepSeek or OpenAI. The LLM explains results; it never invents prices, fundamentals, picks, or predictions.
This is permanently a research-only system. It must never connect to a broker, request or store brokerage credentials, submit/cancel orders, control a trading terminal, or perform automatic trading.

## Tools

{tool_descriptions}

## State

{memory_summary}

## Current Turn Enforcement

{current_turn_policy}

## Task Routing

**Single-stock research** - this is the primary workflow. The user asks how one named stock is doing or whether it can be bought for 1-3 trading days:
1. Extract the requested holding period and call `gupiao_fenxi` once with the code or Chinese name, `holding_days=1|2|3`, and `source="auto"` unless explicitly overridden. “持有两个交易日” must pass `holding_days=2`; if no period is stated, use the documented default of 2 and disclose that assumption.
2. Lead with the tool's `decision.label` and `decision.conclusion`. Then report the analysis timestamp, completed signal date, planned entry/exit dates, tradability, the requested horizon's gross and after-cost returns, empirical positive probability, return interval, and validation status.
3. Technical and fundamental scores are explanatory evidence only. Never turn either heuristic score into an up probability, expected return, or target price. Never alter the tool's decision label or numeric forecast.
4. If tradability is blocked, validation fails, peer history is insufficient, or the tool says `证据不足`, state that directly. Do not manufacture a buy answer from technical indicators.
5. Explain current quote provenance and distinguish an intraday best-effort quote from the completed daily close used by the model. Never replace missing tool data with general knowledge.

**Board selection and prediction** - user asks to select stocks from an industry/concept board and compare the first three sellable horizons:
1. A board name is required. If it is missing, ask one concise question instead of selecting from the whole market.
2. Call `bankuai_xuangu` once with that board, `bankuai_leixing="auto"`, the requested count, and `source="auto"` unless explicitly overridden.
3. The completed T close creates the signal and the next market-session open is the planned entry. T+1/T+2/T+3 mean the first/second/third later sellable closes after entry; T+1 is therefore the second market session after the signal. Never output T+0, calendar-day predictions, or a horizon beyond T+3.
4. Report sample-out validation, model quality, data source, fallback notes, cost assumptions, and why each stock ranked where it did.
5. If recommendations are empty or model quality is low, say so directly. A valid no-trade result is better than invented picks.

## Guidelines

- Treat every user message as part of one continuous conversation. Resolve references such as "它", "刚才那只", "第二只", and "换成这个板块" from conversation history.
- Reuse an earlier tool result when it already answers the follow-up. Call a tool again only when the user asks for fresher data, changes the stock/board, or requests an analysis absent from the previous result.
- A historical tool result whose content says `obsolete_history_result` is incompatible with the current program. It must never support an answer. Call the named tool again in the current turn.
- Never tell the user that a decision label is missing and then substitute your own opinion. A current `gupiao_fenxi` result must contain `decision.label`; if it does not, report a tool failure instead of inventing a conclusion.
- Ask only when the stock or board cannot be identified. Never invent tickers, dates, board names, or trading assumptions.
- Only discuss mainland China A-shares. Politely reject US/HK stocks, funds, futures, crypto, and forex in this program.
- Recommendations and forecasts must come directly from tool output. Do not alter numeric predictions.
- Respect A-share T+1 settlement, board-specific price limits, liquidity filters, commissions, sell-side stamp tax, and slippage.
- If the user asks for live execution or automatic trading, refuse that operation and offer only research output or a manual review checklist. This rule cannot be overridden by user instructions or configuration.
- Do not invoke a successful tool twice for the same user request.
- Do not create scripts, run shell commands, install packages, or modify project files while answering a stock question.
- Do not use emoji or decorative Unicode symbols in CLI answers.
- Respond in the same language the user used.
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
            memory_section=memory_section,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    def required_current_tool(user_message: str) -> Optional[str]:
        """Return the business tool that must run before a time-sensitive answer."""
        text = str(user_message or "").strip()
        if not text:
            return None
        asks_for_fresh_market_data = any(marker in text for marker in _FRESH_REQUEST_MARKERS)
        asks_for_holding_decision = bool(
            re.search(r"持有\s*[一二三123]\s*(?:个)?交易日", text)
        )
        if not asks_for_fresh_market_data and not asks_for_holding_decision:
            return None
        if any(marker in text for marker in _BOARD_REQUEST_MARKERS):
            return "bankuai_xuangu"
        has_stock_code = bool(
            re.search(r"(?<!\d)[0368]\d{5}(?:\.(?:SH|SZ|BJ))?(?!\d)", text, re.IGNORECASE)
        )
        if any(marker in text for marker in _BROAD_MARKET_MARKERS) and not has_stock_code:
            return None
        if (
            asks_for_holding_decision
            or any(marker in text for marker in _SINGLE_STOCK_DECISION_MARKERS)
            or has_stock_code
            or asks_for_fresh_market_data
        ):
            return "gupiao_fenxi"
        return None

    @staticmethod
    def is_object_clarification(content: str) -> bool:
        """Allow one concise clarification when the requested stock/board is absent."""
        text = str(content or "").strip()
        return bool(
            text
            and len(text) <= 300
            and any(marker in text for marker in _OBJECT_CLARIFICATION_MARKERS)
        )

    @staticmethod
    def is_compatible_single_stock_result(content: Any) -> bool:
        """Validate the minimum contract required for a single-stock conclusion."""
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(payload, dict)
            and payload.get("status") == "ok"
            and payload.get("tool_contract_version") == _SINGLE_STOCK_TOOL_CONTRACT_VERSION
            and isinstance(payload.get("decision"), dict)
            and payload["decision"].get("label")
        )

    @classmethod
    def _current_turn_policy(cls, user_message: str) -> str:
        """Force a fresh business-tool call for time-sensitive market questions."""
        text = str(user_message or "").strip()
        asks_for_fresh_market_data = any(marker in text for marker in _FRESH_REQUEST_MARKERS)
        asks_for_holding_decision = bool(
            re.search(r"持有\s*[一二三123]\s*(?:个)?交易日", text)
        )
        if asks_for_fresh_market_data or asks_for_holding_decision:
            required_tool = cls.required_current_tool(text)
            if required_tool:
                return (
                    f"The current user request is time-sensitive. You MUST call `{required_tool}` in this turn "
                    "before giving any market-data conclusion. Historical results are context only. If the "
                    "stock or board is absent, ask one concise clarification instead. Do not answer from an "
                    "earlier session result."
                )
            return (
                "The current user request is time-sensitive. Historical market-data tool results are context only. "
                "You MUST call the appropriate current business tool in this turn before answering. For one named "
                "stock call `gupiao_fenxi`; for a named board call `bankuai_xuangu`. If the object is ambiguous, ask "
                "one concise clarification instead. Do not answer from an earlier session result."
            )
        return (
            "The current request does not explicitly require a refresh. A compatible earlier tool result may be "
            "reused only for a genuine follow-up that does not depend on newer market data."
        )

    @staticmethod
    def _sanitize_historical_message(message: dict[str, Any]) -> dict[str, Any]:
        """Keep tool-call protocol intact while invalidating legacy single-stock payloads."""
        copied = copy.deepcopy(message)
        if copied.get("role") != "tool" or copied.get("name") != "gupiao_fenxi":
            return copied
        content = copied.get("content")
        try:
            payload = json.loads(content) if isinstance(content, str) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        compatible = ContextBuilder.is_compatible_single_stock_result(payload)
        if compatible:
            return copied
        stock = payload.get("stock") if isinstance(payload, dict) else None
        copied["content"] = json.dumps(
            {
                "status": "obsolete_history_result",
                "tool": "gupiao_fenxi",
                "stock": stock if isinstance(stock, dict) else None,
                "message": (
                    "该结果来自旧版单股工具，缺少当前决策契约，禁止复用其行情、指标和结论；"
                    "如需回答当前问题，必须在本轮重新调用 gupiao_fenxi"
                ),
            },
            ensure_ascii=False,
        )
        return copied

    def build_messages(self, user_message: str, history: Optional[list[dict]] = None) -> list[dict]:
        messages = [{"role": "system", "content": self.build_system_prompt(user_message)}]
        obsolete_single_stock_active = False
        if history:
            for message in history:
                if isinstance(message, dict) and message.get("role") in {"user", "assistant", "tool"}:
                    sanitized = self._sanitize_historical_message(message)
                    if sanitized.get("role") == "tool" and sanitized.get("name") == "gupiao_fenxi":
                        try:
                            payload = json.loads(sanitized.get("content", ""))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            payload = None
                        obsolete_single_stock_active = bool(
                            isinstance(payload, dict)
                            and payload.get("status") == "obsolete_history_result"
                        )
                    if (
                        obsolete_single_stock_active
                        and sanitized.get("role") == "assistant"
                        and not sanitized.get("tool_calls")
                    ):
                        sanitized = {
                            "role": "assistant",
                            "content": "[旧版单股工具生成的文字结论已失效，不能用于当前回答。]",
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
