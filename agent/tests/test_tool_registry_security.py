"""Security regression tests for default tool exposure."""

from src.tools import build_registry


def test_shell_tools_absent_from_default_registry() -> None:
    registry = build_registry()

    assert "bash" not in registry.tool_names
    assert "background_run" not in registry.tool_names


def test_shell_tools_cannot_be_enabled() -> None:
    registry = build_registry(include_shell_tools=True)

    assert registry.tool_names == ["gupiao_fenxi", "gupiao_yuce", "bankuai_xuangu"]
