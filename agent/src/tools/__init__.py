"""Tool registry for the intentionally narrow A-share research workflows."""

from __future__ import annotations

from src.agent.clarification import ClarificationHandler
from src.agent.tools import ToolRegistry
from src.tools.clarify_tool import ClarifyTool
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool

_BUSINESS_TOOLS = (GupiaoFenxiTool,)


def build_registry(
    *,
    persistent_memory: object | None = None,
    include_shell_tools: bool = False,
    clarification_handler: ClarificationHandler | None = None,
) -> ToolRegistry:
    """Build business tools and structured clarification for every client.

    ``persistent_memory`` is accepted for compatibility with the agent loop;
    this build does not expose a memory-write tool.
    """
    _ = persistent_memory, include_shell_tools
    registry = ToolRegistry()
    for tool_class in _BUSINESS_TOOLS:
        if tool_class.check_available():
            registry.register(tool_class())
    registry.register(ClarifyTool(clarification_handler))
    return registry


def build_filtered_registry(
    tool_names: list[str],
    *,
    include_shell_tools: bool = False,
    clarification_handler: ClarificationHandler | None = None,
) -> ToolRegistry:
    full = build_registry(
        include_shell_tools=include_shell_tools,
        clarification_handler=clarification_handler,
    )
    filtered = ToolRegistry()
    for name in tool_names:
        tool = full.get(name)
        if tool:
            filtered.register(tool)
    return filtered


__all__ = ["build_registry", "build_filtered_registry"]
