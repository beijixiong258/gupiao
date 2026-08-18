"""Tests that AgentLoop returns reusable multi-turn tool context."""

from __future__ import annotations

import json

from src.agent.loop import AgentLoop
from src.agent.context import ContextBuilder
from src.agent.tools import BaseTool, ToolRegistry
from src.providers.chat import LLMResponse, ToolCallRequest


class _StockTool(BaseTool):
    name = "gupiao_fenxi"
    description = "test stock analysis"
    parameters = {"type": "object", "properties": {"gupiao": {"type": "string"}}}

    def execute(self, **kwargs) -> str:
        return json.dumps(
            {
                "status": "ok",
                "tool_contract_version": 7,
                "analysis_id": "fx_test",
                "analysis_stage": {"status": "completed"},
                "stock": {"name": kwargs["gupiao"]},
                "score": 81,
            },
            ensure_ascii=False,
        )


class _LargeResultTool(BaseTool):
    name = "large_result"
    description = "return a large deterministic result"
    parameters = {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> str:
        return json.dumps({"status": "ok", "payload": "量" * 15_000}, ensure_ascii=False)


class _UnavailableStockTool(BaseTool):
    name = "gupiao_fenxi"
    description = "test unavailable analysis"
    parameters = {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> str:
        return json.dumps(
            {
                "status": "unavailable",
                "outcome": "data_unavailable",
                "error_code": "scope_catalog_unavailable",
                "stage": "scope_discovery",
                "error": "当前无法取得实时板块目录",
            },
            ensure_ascii=False,
        )


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

    def chat(self, messages, tools=None):
        self.seen_messages.append(messages)
        return LLMResponse(content="保留了目标、事实和下一步。")


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


def test_system_prompt_defines_analysis_first_prediction_confirmation_and_short_redirect(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")
    llm = _QueueLLM([LLMResponse(content="本程序专注 A 股分析与预测，请尽量围绕相关内容提问。")])
    agent = AgentLoop(registry=ToolRegistry(), llm=llm, max_iterations=2)

    result = agent.run("给我讲一个太空故事")

    system_prompt = llm.seen_messages[0][0]["content"]
    assert "Quantitative analysis is the product's primary and default function" in system_prompt
    assert "Never call analysis and prediction in the same turn" in system_prompt
    assert "later user turn explicitly confirms prediction" in system_prompt
    assert "Advance permission in the initial request" in system_prompt
    assert "Do not ask for a third confirmation" in system_prompt
    assert "freshly download the target, peer, valuation, and benchmark data" in system_prompt
    assert "本程序专注 A 股分析与预测，请尽量围绕相关内容提问。" in system_prompt
    assert result["content"] == "本程序专注 A 股分析与预测，请尽量围绕相关内容提问。"


def test_large_tool_result_stays_in_memory_but_default_run_log_saves_only_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")
    registry = ToolRegistry()
    registry.register(_LargeResultTool())
    llm = _QueueLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="tc_large", name="large_result", arguments={})]),
            LLMResponse(content="已读取完整结果。"),
        ]
    )
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=3)

    result = agent.run("读取完整结果")

    tool_message = next(message for message in result["history"] if message.get("role") == "tool")
    assert tool_message["content"].count("量") == 15_000
    saved = json.loads((tmp_path / "runs" / result["run_id"] / "response.json").read_text(encoding="utf-8"))
    assert saved == {
        "status": "success",
        "iterations": 2,
        "business_outcome": None,
    }
    assert "history" not in saved


def test_auto_compact_does_not_require_a_persisted_transcript_in_metadata_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("src.agent.loop.TOKEN_THRESHOLD", 1_000)
    registry = ToolRegistry()
    registry.register(_LargeResultTool())
    llm = _QueueLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="tc_compact", name="large_result", arguments={})]),
            LLMResponse(content="压缩后仍可继续回答。"),
        ]
    )

    result = AgentLoop(registry=registry, llm=llm, max_iterations=3).run("读取并总结大结果")

    assert result["status"] == "success"
    assert result["content"] == "压缩后仍可继续回答。"
    run_dir = tmp_path / "runs" / result["run_id"]
    assert list(run_dir.glob("transcript_*.jsonl")) == []
    trace = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "transcript_not_saved" in trace
    assert "保留了目标、事实和下一步" not in trace


def test_unavailable_analysis_cannot_become_success_because_llm_wrote_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.agent.loop.RUNS_DIR", tmp_path / "runs")
    registry = ToolRegistry()
    registry.register(_UnavailableStockTool())
    llm = _QueueLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="tc_unavailable", name="gupiao_fenxi", arguments={})]),
            LLMResponse(content="当前目录不可用，请稍后重试。"),
        ]
    )

    result = AgentLoop(registry=registry, llm=llm, max_iterations=3).run("从电子板块选一只")

    assert result["status"] == "data_unavailable"
    assert result["business_outcome"] == "data_unavailable"
    state = json.loads((tmp_path / "runs" / result["run_id"] / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "data_unavailable"


def test_unexpected_foreign_script_is_blocked_but_normal_chinese_is_allowed() -> None:
    assert ContextBuilder.contains_unexpected_script("需要 уточнить 范围") is True
    assert ContextBuilder.contains_unexpected_script("需要确认一次范围") is False
