"""MCP exposure must stay limited to the single read-only A-share analysis tool."""

import asyncio
import json

import pytest

from fastmcp import Client

import mcp_server
from mcp_server import mcp
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool


async def _list_tools_through_client():
    async with Client(mcp) as client:
        return await client.list_tools()


def test_mcp_exposes_only_a_share_research_tools() -> None:
    tools = asyncio.run(_list_tools_through_client())

    assert [tool.name for tool in tools] == ["gupiao_fenxi"]
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False


def test_mcp_analysis_schema_matches_business_parameters() -> None:
    analysis = asyncio.run(_list_tools_through_client())[0]
    properties = analysis.inputSchema["properties"]
    expected = GupiaoFenxiTool.parameters["properties"]
    assert set(properties) == set(expected)
    assert properties["fanwei"]["enum"] == expected["fanwei"]["enum"]
    count_schema = next(item for item in properties["shuliang"]["anyOf"] if item.get("type") == "integer")
    assert count_schema["minimum"] == expected["shuliang"]["minimum"]
    assert count_schema["maximum"] == expected["shuliang"]["maximum"]


@pytest.mark.parametrize("arguments", [
    {"fanwei": "single_stock", "gupiao": "深科技"},
    {"fanwei": "named_scope", "mingcheng": "电子", "shuliang": 2},
])
def test_mcp_analysis_forwards_stock_and_count(monkeypatch, arguments) -> None:
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return json.dumps({"status": "ok", "received": kwargs})

    monkeypatch.setattr(mcp_server._GUPIAO_FENXI, "execute", execute)

    async def call():
        async with Client(mcp) as client:
            return await client.call_tool("gupiao_fenxi", arguments)

    result = asyncio.run(call())
    assert not result.is_error
    assert len(calls) == 1
    assert all(calls[0][key] == value for key, value in arguments.items())


def test_mcp_rejects_out_of_range_count_before_execution(monkeypatch) -> None:
    def execute(**kwargs):
        pytest.fail("无效数量不应进入分析工具")

    monkeypatch.setattr(mcp_server._GUPIAO_FENXI, "execute", execute)

    async def call():
        async with Client(mcp) as client:
            return await client.call_tool("gupiao_fenxi", {"shuliang": 6}, raise_on_error=False)

    assert asyncio.run(call()).is_error
