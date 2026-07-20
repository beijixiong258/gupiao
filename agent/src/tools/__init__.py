"""Tool registry for the intentionally narrow A-share research workflows."""

from __future__ import annotations

from src.agent.tools import ToolRegistry
from src.tools.bankuai_xuangu_tool import BankuaiXuanguTool
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
from src.tools.gupiao_yuce_tool import GupiaoYuceTool

_BUSINESS_TOOLS = (GupiaoFenxiTool, GupiaoYuceTool, BankuaiXuanguTool)


def build_registry(
    *,
    persistent_memory: object | None = None,
    include_shell_tools: bool = False,
) -> ToolRegistry:
    """Build the intentionally narrow business-tool registry.

    ``persistent_memory`` is accepted for compatibility with the agent loop;
    this build does not expose a memory-write tool.
    """
    _ = persistent_memory, include_shell_tools
    registry = ToolRegistry()
    for tool_class in _BUSINESS_TOOLS:
        if tool_class.check_available():
            registry.register(tool_class())
    return registry


def build_filtered_registry(tool_names: list[str], *, include_shell_tools: bool = False) -> ToolRegistry:
    full = build_registry(include_shell_tools=include_shell_tools)
    filtered = ToolRegistry()
    for name in tool_names:
        tool = full.get(name)
        if tool:
            filtered.register(tool)
    return filtered


__all__ = ["build_registry", "build_filtered_registry"]
