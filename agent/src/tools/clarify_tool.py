"""Interactive clarification tool backed by a platform-provided UI callback."""

from __future__ import annotations

import json
from typing import Any

from src.agent.clarification import (
    ClarificationChoice,
    ClarificationHandler,
    ClarificationRequest,
    MAX_PREDEFINED_CHOICES,
)
from src.agent.tools import BaseTool


class ClarifyTool(BaseTool):
    """Ask one structured question without embedding terminal code in the agent."""

    name = "clarify"
    description = (
        "Ask the user one necessary clarification before continuing. Use it when an industry, board, concept, "
        "or another material choice remains genuinely ambiguous. Offer two to four short plain-language choices "
        "when possible; the UI also accepts a custom answer. Do not use this tool for the post-analysis prediction "
        "opt-in because the interactive client presents that prompt after the complete analysis answer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "One concise, non-empty question in the user's language.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": MAX_PREDEFINED_CHOICES,
                "description": "Two to four mutually exclusive plain-language choices. Omit for free text.",
            },
        },
        "required": ["question"],
    }
    repeatable = True
    is_readonly = False

    def __init__(self, handler: ClarificationHandler) -> None:
        self._handler = handler

    def execute(self, **kwargs: Any) -> str:
        raw_choices = kwargs.get("choices")
        if raw_choices is None:
            choices: tuple[ClarificationChoice, ...] = ()
        elif isinstance(raw_choices, list):
            if any(not isinstance(value, str) for value in raw_choices):
                return json.dumps(
                    {"status": "error", "error": "choices 只能包含字符串"},
                    ensure_ascii=False,
                )
            choices = tuple(
                ClarificationChoice(label=value, response=value) for value in raw_choices
            )
        else:
            return json.dumps(
                {"status": "error", "error": "choices 必须是字符串数组"},
                ensure_ascii=False,
            )
        try:
            request = ClarificationRequest(
                kind="general",
                title="需要你确认",
                question=kwargs.get("question", ""),
                choices=choices,
                allow_custom=True,
            )
        except ValueError as exc:
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

        answer = self._handler(request)
        return json.dumps(
            {
                "status": "ok",
                "question": request.question,
                "choices_offered": [choice.label for choice in request.choices],
                "user_response": answer.response,
                "selected_index": answer.selected_index,
                "custom_response": answer.is_custom,
                "cancelled": answer.cancelled,
            },
            ensure_ascii=False,
        )


__all__ = ["ClarifyTool"]
