"""Regression tests for DeepSeek and OpenAI response replay."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import convert_to_messages

from src.agent.context import ContextBuilder
from src.providers.chat import ChatLLM, ToolCallRequest, _dedupe_finish_reason
from src.providers.llm import ChatOpenAIWithReasoning


def test_deepseek_reasoning_and_tool_call_are_preserved() -> None:
    message = SimpleNamespace(
        content="",
        tool_calls=[{"id": "call_1", "name": "gupiao_fenxi", "args": {"gupiao": "600519.SH"}}],
        additional_kwargs={"reasoning_content": "先取事实数据"},
        response_metadata={"finish_reason": "tool_calls"},
        id="message_1",
    )

    response = ChatLLM._parse_response(message)

    assert response.reasoning_content == "先取事实数据"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].arguments == {"gupiao": "600519.SH"}
    assert response.provider_data["id"] == "message_1"


def test_responses_api_text_blocks_become_plain_answer() -> None:
    message = SimpleNamespace(
        content=[
            {"type": "reasoning", "summary": []},
            {"type": "output_text", "text": "研究结果"},
        ],
        tool_calls=[],
        additional_kwargs={},
        response_metadata={"finish_reason": "stop"},
        id="message_2",
    )

    response = ChatLLM._parse_response(message)

    assert response.content == "研究结果"
    assert response.provider_data["raw_content"] == message.content


def test_streaming_responses_blocks_are_forwarded_as_strings() -> None:
    message = SimpleNamespace(
        content=[{"type": "text", "text": "深科技研究结果"}],
        tool_calls=[],
        additional_kwargs={},
        response_metadata={"finish_reason": "stop"},
        id="message_stream_1",
    )

    class _FakeStreamingLLM:
        @staticmethod
        def stream(_messages, config=None):
            _ = config
            yield message

    client = object.__new__(ChatLLM)
    client.provider = "openai_codex"
    client._llm = _FakeStreamingLLM()
    chunks: list[str] = []

    response = client.stream_chat(
        [{"role": "system", "content": "A股规则"}, {"role": "user", "content": "分析深科技"}],
        on_text_chunk=chunks.append,
    )

    assert chunks == ["深科技研究结果"]
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert response.content == "深科技研究结果"


def test_deepseek_reasoning_replay_uses_canonical_field() -> None:
    replay = ContextBuilder.format_assistant_tool_calls(
        [ToolCallRequest(id="call_1", name="gupiao_fenxi", arguments={"gupiao": "600519.SH"})],
        reasoning_content="调用单股工具",
    )

    assert replay["reasoning_content"] == "调用单股工具"
    assert replay["tool_calls"][0]["function"]["name"] == "gupiao_fenxi"


def test_openai_provider_metadata_survives_langchain_conversion() -> None:
    replay = ContextBuilder.format_assistant_tool_calls(
        [ToolCallRequest(id="call_2", name="bankuai_xuangu", arguments={"bankuai": "白酒"})],
        provider_data={
            "raw_content": [{"type": "output_text", "text": ""}],
            "additional_kwargs": {"reasoning": {"encrypted_content": "encrypted"}},
            "id": "response_1",
            "response_metadata": {"model_name": "gpt-5.6"},
        },
    )
    converted = convert_to_messages([replay])[0]

    assert converted.id == "response_1"
    assert converted.additional_kwargs["reasoning"]["encrypted_content"] == "encrypted"
    assert converted.response_metadata["model_name"] == "gpt-5.6"
    assert converted.tool_calls[0]["name"] == "bankuai_xuangu"


def test_reasoning_alias_is_normalized() -> None:
    target = SimpleNamespace(additional_kwargs={})
    ChatOpenAIWithReasoning._capture({"reasoning": "兼容字段"}, target)
    assert target.additional_kwargs["reasoning_content"] == "兼容字段"


def test_duplicated_finish_reason_is_deduplicated() -> None:
    assert _dedupe_finish_reason("stopstop") == "stop"


def test_chatgpt_codex_converts_system_role_without_mutating_context() -> None:
    messages = [
        {"role": "system", "content": "A股研究规则"},
        {"role": "user", "content": "分析深科技"},
    ]

    prepared = ChatLLM._prepare_messages(messages, "openai_codex")

    assert prepared[0] == {"role": "developer", "content": "A股研究规则"}
    assert prepared[1] == messages[1]
    assert messages[0]["role"] == "system"


def test_other_providers_keep_system_role() -> None:
    messages = [{"role": "system", "content": "A股研究规则"}]
    assert ChatLLM._prepare_messages(messages, "deepseek") is messages
    assert _dedupe_finish_reason("tool_callstool_calls") == "tool_calls"
