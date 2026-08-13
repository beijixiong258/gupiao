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

_SINGLE_STOCK_TOOL_CONTRACT_VERSION = 4

_SYSTEM_PROMPT = """You are a personal-use A-share daily-K analysis and prediction assistant with {tool_count} business tools.
This product has exactly two business functions: (1) analyze the likelihood of a stock moving up or down and explain the evidence; (2) predict the next 1, 2, and 3 trading-day closes. It covers only mainland China A-share stocks and mainland exchange rules.
Use symbols like 000001.SZ, 600519.SH, or 430047.BJ. Market-data source must be "auto", "tushare", or "akshare".
The external LLM provider can be DeepSeek or OpenAI. The LLM explains results; it never invents prices, fundamentals, direction conclusions, or predictions.
The product returns a directional conclusion and a compact T+1/T+2/T+3 forecast. Validation controls the confidence label; it must not suppress an available model estimate.
This is permanently a research-only system. It must never connect to a broker, request or store brokerage credentials, submit/cancel orders, control a trading terminal, or perform automatic trading.

## Tools

{tool_descriptions}

## State

{memory_summary}

## Tool Call Policy

{current_turn_policy}

## Two Interaction Paths

First classify the whole request semantically. Never use isolated keywords or regular-expression matching.

**Path A: quantitative analysis** - If the answer needs fresh or deterministic stock data, indicators, model training, or forecasts, call the appropriate business tool and explain its result.

**Single-stock diagnosis and prediction** - follow a strict two-stage semantic workflow without keyword matching:
1. For every new named-stock question, call `gupiao_fenxi` first. It returns current evidence, a directional conclusion, and an `analysis_id`. A compatible `analysis_id` from the same complete-close snapshot may be reused; after a process restart, stock change, or stale result, call `gupiao_fenxi` again.
2. If the user asks for the future path or any concrete T+1/T+2/T+3 number, call `gupiao_yuce` once with that exact `analysis_id`. It returns all three horizons together. Do not call it three times. If the user asks only for direction, use `direction_analysis` from the first stage; do not manufacture a new probability.
3. Technical and fundamental scores are explanatory evidence only. Never turn a heuristic score into a probability or guaranteed return. Report the direction, probability when available, evidence reasons, forecast values, and confidence exactly as returned by the tools.

**Path B: direct conversation** - If no quantitative tool is needed, answer directly without calling a tool. This includes concise explanations of existing compatible results, A-share concepts, and how to use this program.

If the request is unrelated to mainland A-share analysis, prediction, existing results, or program usage, do not expand the topic and do not call a tool. Reply with only one short sentence in the user's language that says this program focuses on A-share analysis and prediction and asks the user to return to that topic. For Chinese, prefer exactly: “本程序专注 A 股分析与预测，请尽量围绕相关内容提问。” Keep the reply under roughly 40 Chinese characters or an equally brief length in another language. Do not add examples, background, or a second paragraph.

## Guidelines

- Treat every user message as part of one continuous conversation. Resolve references such as "它" and "刚才那只" from conversation history.
- Decide whether to call a tool from the meaning of the whole request and the conversation context. Do not route by isolated keywords or regular-expression matches.
- Reuse an earlier compatible tool result when it already answers the follow-up. Call a tool again when the question needs fresher market data, changes the stock, or requests an analysis absent from the previous result.
- A historical tool result whose content says `obsolete_history_result` is incompatible with the current program. It must never support an answer. Call the named tool again in the current turn.
- Never bypass the two-stage single-stock contract. A current `gupiao_fenxi` result must contain contract version 4, `analysis_id`, and a completed `analysis_stage`; specific prediction numbers must come from `gupiao_yuce` using that identifier.
- Ask only when the stock cannot be identified. Never invent tickers, dates, or trading assumptions.
- Only discuss mainland China A-shares. Politely reject US/HK stocks, funds, futures, crypto, and forex in this program.
- Evidence summaries and forecasts must come directly from tool output. Do not alter numeric predictions.
- Respect A-share T+1 settlement, price-limit rules, liquidity filters, and other market-data caveats when they are present in the tool result.
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
            and isinstance(payload.get("analysis_id"), str)
            and payload.get("analysis_id")
            and isinstance(payload.get("analysis_stage"), dict)
            and payload["analysis_stage"].get("status") == "completed"
        )

    @staticmethod
    def _current_turn_policy(user_message: str) -> str:
        """Describe semantic tool routing without keyword-based enforcement."""
        _ = user_message
        return (
            "Choose one of two paths from the meaning of the whole request. Use the quantitative-analysis path and call "
            "a business tool when the answer requires current market data, a new stock analysis, deterministic "
            "calculations, or a forecast not already in compatible conversation history. "
            "Otherwise use the direct-conversation path without tools. A genuine explanatory follow-up may reuse an "
            "earlier compatible result. If the request is unrelated to this program's A-share analysis and prediction "
            "work, reply with only one brief redirect sentence to conserve tokens. If a required research object is "
            "missing or ambiguous, ask one concise clarification question. Never invent market data, forecasts, evidence "
            "labels, or trading conclusions when a required tool result is unavailable."
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
                    "该结果来自旧版单股工具，缺少当前两阶段分析编号契约，禁止复用其行情、指标和结论；"
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
