"""ContextBuilder: builds LLM message context for the ReAct AgentLoop."""

from __future__ import annotations

import copy
import logging
import json
from datetime import datetime
from typing import Any, TYPE_CHECKING, Optional

from src.agent.memory import WorkspaceMemory
from src.agent.tools import ToolRegistry

if TYPE_CHECKING:
    from src.memory.persistent import PersistentMemory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an A-share T+3 quantitative research agent with {tool_count} business tools.
This product has exactly two primary workflows. It covers only mainland China A-share stocks and mainland exchange rules.
Use symbols like 000001.SZ, 600519.SH, or 430047.BJ. Market-data source must be "auto", "tushare", or "akshare".
The external LLM provider can be DeepSeek or OpenAI. The LLM explains results; it never invents prices, fundamentals, picks, or predictions.
This is permanently a research-only system. It must never connect to a broker, request or store brokerage credentials, submit/cancel orders, control a trading terminal, or perform automatic trading.

## Tools

{tool_descriptions}

## State

{memory_summary}

## Task Routing

**Single-stock research** - user asks whether one named stock is good, risky, expensive, trending, or worth watching:
1. Call `gupiao_fenxi` once with the code or Chinese stock name and `source="auto"` unless the user explicitly chooses a source.
2. Explain the returned basic profile, valuation, financial indicators, technical indicators, data provenance, missing fields, and risks.
3. Never replace missing tool data with general knowledge. Distinguish facts from interpretation.

**Board selection and prediction** - user asks to select stocks from an industry/concept board and predict the next days:
1. A board name is required. If it is missing, ask one concise question instead of selecting from the whole market.
2. Call `bankuai_xuangu` once with that board, `bankuai_leixing="auto"`, the requested count, and `source="auto"` unless explicitly overridden.
3. The only allowed horizons are T+1, T+2, and T+3 trading days. Never output T+0, calendar-day predictions, or a horizon beyond T+3.
4. Report sample-out validation, model quality, data source, fallback notes, cost assumptions, and why each stock ranked where it did.
5. If recommendations are empty or model quality is low, say so directly. A valid no-trade result is better than invented picks.

## Guidelines

- Treat every user message as part of one continuous conversation. Resolve references such as "它", "刚才那只", "第二只", and "换成这个板块" from conversation history.
- Reuse an earlier tool result when it already answers the follow-up. Call a tool again only when the user asks for fresher data, changes the stock/board, or requests an analysis absent from the previous result.
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
        _ = user_message
        memory_section = ""
        if self._persistent_memory and self._persistent_memory.snapshot:
            memory_section = _MEMORY_SECTION.format(snapshot=self._persistent_memory.snapshot)

        return _SYSTEM_PROMPT.format(
            tool_count=len(self.registry.tools),
            tool_descriptions=self.registry.get_descriptions(),
            memory_summary=self.memory.to_summary(),
            memory_section=memory_section,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def build_messages(self, user_message: str, history: Optional[list[dict]] = None) -> list[dict]:
        messages = [{"role": "system", "content": self.build_system_prompt(user_message)}]
        if history:
            for message in history:
                if isinstance(message, dict) and message.get("role") in {"user", "assistant", "tool"}:
                    messages.append(copy.deepcopy(message))
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
