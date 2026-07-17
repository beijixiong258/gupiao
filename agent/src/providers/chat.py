"""ChatLLM: raw LLM message interface with function calling support.

ChatLLM is designed specifically for the AgentLoop ReAct cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.providers.llm import _provider_name, build_llm


def _dedupe_finish_reason(raw: str) -> str:
    """Return the canonical finish_reason suffix from streamed chunks."""
    return next(
        (m for m in ("tool_calls", "function_call", "content_filter", "length", "stop")
         if raw.endswith(m)),
        raw,
    )


def _content_text(content: Any) -> str:
    """Extract text from string or Responses API content blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and block.get("type") in {"text", "output_text"}:
                parts.append(text)
    return "".join(parts)


@dataclass
class ToolCallRequest:
    """Tool call request returned by the LLM.

    Attributes:
        id: Tool call ID (used to match tool_result messages).
        name: Tool name.
        arguments: Tool argument dict.
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """LLM response.

    Attributes:
        content: Text content (final answer or thinking text).
        tool_calls: List of tool call requests.
        finish_reason: Finish reason string.
    """

    content: Optional[str] = None
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    reasoning_content: Optional[str] = None
    finish_reason: str = "stop"
    provider_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Return True if the response contains tool calls."""
        return len(self.tool_calls) > 0


class ChatLLM:
    """LLM chat client with function calling support.

    Uses build_llm() to obtain a ChatOpenAI instance and bind_tools() to attach tool definitions.

    Attributes:
        model_name: Model name.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize ChatLLM.

        Args:
            model_name: Model name; defaults to the environment variable value.
        """
        self.model_name = model_name
        self.provider = _provider_name()
        self._llm = build_llm(model_name=model_name)

    @staticmethod
    def _prepare_messages(messages: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
        """Adapt message roles required by the ChatGPT Codex backend."""
        if provider != "openai_codex":
            return messages
        prepared: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                converted = dict(message)
                converted["role"] = "developer"
                prepared.append(converted)
            else:
                prepared.append(message)
        return prepared

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, timeout: Optional[int] = None) -> LLMResponse:
        """Call the LLM synchronously.

        Args:
            messages: Message list (OpenAI format).
            tools: Tool definition list (OpenAI function calling format).
            timeout: Optional per-call timeout in seconds.

        Returns:
            LLMResponse.
        """
        llm = self._llm.bind_tools(tools) if tools else self._llm
        config = {"timeout": timeout} if timeout else {}
        ai_message = llm.invoke(self._prepare_messages(messages, self.provider), config=config)
        return self._parse_response(ai_message)

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        on_text_chunk: Optional[Any] = None,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        """Stream the LLM and optionally forward text deltas (e.g. thinking).

        Iterates AIMessageChunk; each text delta invokes ``on_text_chunk``.
        Aggregates chunks into one response; on failure falls back to ``chat()``.

        Args:
            messages: Messages in OpenAI format.
            tools: Tool definitions for function calling.
            on_text_chunk: Optional callback ``(delta: str) -> None``.
            timeout: Optional per-call timeout in seconds.

        Returns:
            Parsed ``LLMResponse``.
        """
        try:
            llm = self._llm.bind_tools(tools) if tools else self._llm
            config = {"timeout": timeout} if timeout else {}
            accumulated = None
            prepared_messages = self._prepare_messages(messages, self.provider)
            for chunk in llm.stream(prepared_messages, config=config):
                text_delta = _content_text(chunk.content)
                if text_delta and on_text_chunk:
                    on_text_chunk(text_delta)
                accumulated = chunk if accumulated is None else accumulated + chunk
            if accumulated is None:
                return LLMResponse(content="", tool_calls=[], finish_reason="stop")
            return self._parse_response(accumulated)
        except Exception:
            return self.chat(messages, tools=tools, timeout=timeout)

    async def achat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, timeout: Optional[int] = None) -> LLMResponse:
        """Async LLM invocation.

        Args:
            messages: Messages in OpenAI format.
            tools: Tool definitions (OpenAI function-calling format).
            timeout: Optional per-call timeout in seconds.

        Returns:
            ``LLMResponse``.
        """
        llm = self._llm.bind_tools(tools) if tools else self._llm
        config = {"timeout": timeout} if timeout else {}
        ai_message = await llm.ainvoke(self._prepare_messages(messages, self.provider), config=config)
        return self._parse_response(ai_message)

    @staticmethod
    def _parse_response(ai_message: Any) -> LLMResponse:
        """Convert a LangChain AIMessage (or AIMessageChunk) to ``LLMResponse``.

        Single source for reasoning: ``additional_kwargs["reasoning_content"]``,
        populated by ``ChatOpenAIWithReasoning`` on both stream and non-stream paths.
        """
        raw_content = ai_message.content
        content = _content_text(raw_content)
        if not content and raw_content and not isinstance(raw_content, (str, list)):
            content = str(raw_content)

        additional = dict(ai_message.additional_kwargs or {})
        reasoning_content = additional.get("reasoning_content")
        if not reasoning_content and isinstance(additional.get("reasoning"), dict):
            summary = additional["reasoning"].get("summary") or []
            reasoning_content = "\n".join(
                str(item.get("text") or "")
                for item in summary
                if isinstance(item, dict) and item.get("text")
            ).strip() or None

        return LLMResponse(
            content=content,
            tool_calls=[
                ToolCallRequest(id=tc["id"], name=tc["name"], arguments=tc["args"])
                for tc in ai_message.tool_calls
            ],
            reasoning_content=reasoning_content,
            finish_reason=_dedupe_finish_reason(
                ai_message.response_metadata.get("finish_reason", "stop")
            ),
            provider_data={
                "raw_content": raw_content,
                "additional_kwargs": additional,
                "id": getattr(ai_message, "id", None),
                "response_metadata": dict(ai_message.response_metadata or {}),
            },
        )
