"""Agent core module: ReAct loop, tool registry, and conversation memory."""

from typing import TYPE_CHECKING, Any

from src.agent.memory import WorkspaceMemory
from src.agent.tools import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from src.agent.loop import AgentLoop


def __getattr__(name: str) -> Any:
    """工具与 MCP 可独立导入，使用智能体循环时才加载模型链路。"""
    if name == "AgentLoop":
        from src.agent.loop import AgentLoop

        return AgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentLoop", "WorkspaceMemory", "BaseTool", "ToolRegistry"]
