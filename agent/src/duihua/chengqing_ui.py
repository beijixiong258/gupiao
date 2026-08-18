"""Rich terminal adapter for structured clarification requests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.agent.clarification import ClarificationAnswer, ClarificationRequest


class RichChengqingTishi:
    """Render one numbered decision without coupling the agent to Rich."""

    def __init__(self, console: Console, status_ref: dict[str, Any]) -> None:
        self._console = console
        self._status_ref = status_ref

    @contextmanager
    def _zanting_jindu(self) -> Iterator[None]:
        status = self._status_ref.get("value")
        paused = False
        if status is not None:
            try:
                status.stop()
                paused = True
            except Exception:
                paused = False
        try:
            yield
        finally:
            if status is not None and paused:
                try:
                    status.start()
                except Exception:
                    pass

    def _duqu_zidingyi(self, request: ClarificationRequest) -> ClarificationAnswer:
        try:
            value = self._console.input("请输入你的答案：").strip()
        except (KeyboardInterrupt, EOFError):
            return ClarificationAnswer(cancelled=True)
        if not value:
            self._console.print("[yellow]答案不能为空，已取消本次选择[/yellow]")
            return ClarificationAnswer(cancelled=True)
        return ClarificationAnswer(
            response=f"{request.custom_response_prefix}{value}",
            is_custom=True,
        )

    def xunwen(self, request: ClarificationRequest) -> ClarificationAnswer:
        """Show a panel and return a normalized response for the next agent turn."""
        with self._zanting_jindu():
            lines = [request.question]
            for index, choice in enumerate(request.choices, start=1):
                lines.append(f"{index}. {choice.label}")
            custom_index = len(request.choices) + 1
            if request.allow_custom and request.choices:
                lines.append(f"{custom_index}. {request.custom_label}")
            self._console.print(Panel(Text("\n".join(lines)), title=request.title, border_style="cyan"))

            if not request.choices:
                return self._duqu_zidingyi(request)

            while True:
                upper = custom_index if request.allow_custom else len(request.choices)
                try:
                    raw = self._console.input(
                        f"请输入 1-{upper}，也可以直接输入自己的答案："
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    self._console.print("\n[dim]已取消本次选择[/dim]")
                    return ClarificationAnswer(cancelled=True)
                if not raw:
                    self._console.print("[yellow]请输入一个序号或答案[/yellow]")
                    continue
                if raw.isdigit():
                    selected = int(raw)
                    if 1 <= selected <= len(request.choices):
                        choice = request.choices[selected - 1]
                        return ClarificationAnswer(
                            response=choice.response,
                            selected_index=selected,
                        )
                    if request.allow_custom and selected == custom_index:
                        return self._duqu_zidingyi(request)
                    self._console.print(f"[yellow]请输入 1-{upper} 之间的序号[/yellow]")
                    continue
                if request.allow_custom:
                    return ClarificationAnswer(
                        response=f"{request.custom_response_prefix}{raw}",
                        is_custom=True,
                    )
                self._console.print(f"[yellow]请输入 1-{upper} 之间的序号[/yellow]")


__all__ = ["RichChengqingTishi"]
