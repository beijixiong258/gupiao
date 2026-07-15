"""MCP exposure must stay limited to the two read-only A-share workflows."""

import asyncio

from fastmcp import Client

from mcp_server import mcp


async def _list_tools_through_client():
    async with Client(mcp) as client:
        return await client.list_tools()


def test_mcp_exposes_only_a_share_research_tools() -> None:
    tools = asyncio.run(_list_tools_through_client())

    assert [tool.name for tool in tools] == ["gupiao_fenxi", "bankuai_xuangu"]
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
