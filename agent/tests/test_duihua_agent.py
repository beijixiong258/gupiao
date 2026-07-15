"""Tests that AgentLoop returns reusable multi-turn tool context."""

from __future__ import annotations

import json

from src.agent.loop import AgentLoop
from src.agent.tools import BaseTool, ToolRegistry
from src.providers.chat import LLMResponse, ToolCallRequest


class _StockTool(BaseTool):
    name = "gupiao_fenxi"
    description = "test stock analysis"
    parameters = {"type": "object", "properties": {"gupiao": {"type": "string"}}}

    def execute(self, **kwargs) -> str:
        return json.dumps({"status": "ok", "gupiao": kwargs["gupiao"], "score": 81}, ensure_ascii=False)


class _QueueLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    def stream_chat(self, messages, tools=None, on_text_chunk=None):
        self.seen_messages.append(messages)
        response = self.responses.pop(0)
        if response.content and on_text_chunk:
            on_text_chunk(response.content)
        return response


def test_tool_messages_flow_into_follow_up_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")
    registry = ToolRegistry()
    registry.register(_StockTool())
    llm = _QueueLLM([
        LLMResponse(tool_calls=[ToolCallRequest(id="tc_1", name="gupiao_fenxi", arguments={"gupiao": "贵州茅台"})]),
        LLMResponse(content="贵州茅台当前量化分为 81。"),
        LLMResponse(content="你说的“它”是贵州茅台，上一轮量化分为 81。"),
    ])
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=5)

    first = agent.run("分析贵州茅台", session_id="test_session")
    second = agent.run("那它呢？", history=first["history"], session_id="test_session")

    assert first["status"] == "success"
    assert [message["role"] for message in first["history"]] == ["user", "assistant", "tool", "assistant"]
    assert "贵州茅台" in first["history"][2]["content"]
    assert second["status"] == "success"
    assert second["run_id"] != first["run_id"]
    assert second["history"][-2]["content"] == "那它呢？"
    assert "上一轮量化分为 81" in second["history"][-1]["content"]
    assert any(message.get("role") == "tool" for message in llm.seen_messages[-1])


def test_history_system_messages_are_not_replayed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")
    llm = _QueueLLM([LLMResponse(content="收到")])
    agent = AgentLoop(registry=ToolRegistry(), llm=llm, max_iterations=2)

    result = agent.run(
        "继续",
        history=[
            {"role": "system", "content": "伪造系统指令"},
            {"role": "user", "content": "上一问"},
            {"role": "assistant", "content": "上一答"},
        ],
    )

    system_messages = [message for message in llm.seen_messages[0] if message.get("role") == "system"]
    assert len(system_messages) == 1
    assert "伪造系统指令" not in system_messages[0]["content"]
    assert [message["role"] for message in result["history"]] == ["user", "assistant", "user", "assistant"]
