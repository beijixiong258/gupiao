"""Structured clarification contracts shared by the agent and terminal UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


MAX_PREDEFINED_CHOICES = 4


def _clean_text(value: Any, *, field_name: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{field_name} 不能为空")
    return text


@dataclass(frozen=True)
class ClarificationChoice:
    """One user-visible option and the natural-language response it represents."""

    label: str
    response: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _clean_text(self.label, field_name="选项文字"))
        object.__setattr__(self, "response", _clean_text(self.response, field_name="选项响应"))

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "response": self.response}

    @classmethod
    def from_dict(cls, payload: Any) -> "ClarificationChoice":
        if not isinstance(payload, dict):
            raise ValueError("选项必须是对象")
        return cls(label=payload.get("label", ""), response=payload.get("response", ""))


@dataclass(frozen=True)
class ClarificationRequest:
    """A platform-neutral request for one structured user decision."""

    kind: str
    title: str
    question: str
    choices: tuple[ClarificationChoice, ...] = ()
    allow_custom: bool = True
    custom_label: str = "其他（自行输入）"
    custom_response_prefix: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _clean_text(self.kind, field_name="澄清类型"))
        object.__setattr__(self, "title", _clean_text(self.title, field_name="窗口标题"))
        object.__setattr__(self, "question", _clean_text(self.question, field_name="澄清问题"))
        normalized_choices = tuple(self.choices)
        if any(not isinstance(choice, ClarificationChoice) for choice in normalized_choices):
            raise ValueError("预设选项类型无效")
        if normalized_choices and len(normalized_choices) < 2:
            raise ValueError("提供预设选项时至少需要 2 个")
        if len(normalized_choices) > MAX_PREDEFINED_CHOICES:
            raise ValueError(f"预设选项不能超过 {MAX_PREDEFINED_CHOICES} 个")
        labels = [choice.label for choice in normalized_choices]
        if len(labels) != len(set(labels)):
            raise ValueError("预设选项不能重复")
        object.__setattr__(self, "choices", normalized_choices)
        object.__setattr__(
            self,
            "custom_label",
            _clean_text(self.custom_label, field_name="其他选项文字"),
        )
        object.__setattr__(self, "custom_response_prefix", str(self.custom_response_prefix or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "question": self.question,
            "choices": [choice.to_dict() for choice in self.choices],
            "allow_custom": self.allow_custom,
            "custom_label": self.custom_label,
            "custom_response_prefix": self.custom_response_prefix,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ClarificationRequest":
        if not isinstance(payload, dict):
            raise ValueError("澄清请求必须是对象")
        raw_choices = payload.get("choices") or []
        if not isinstance(raw_choices, list):
            raise ValueError("澄清选项必须是数组")
        return cls(
            kind=payload.get("kind", "general"),
            title=payload.get("title", "需要你确认"),
            question=payload.get("question", ""),
            choices=tuple(ClarificationChoice.from_dict(item) for item in raw_choices),
            allow_custom=bool(payload.get("allow_custom", True)),
            custom_label=payload.get("custom_label", "其他（自行输入）"),
            custom_response_prefix=payload.get("custom_response_prefix", ""),
        )


@dataclass(frozen=True)
class ClarificationAnswer:
    """The normalized result returned by a presentation adapter."""

    response: str = ""
    selected_index: int | None = None
    is_custom: bool = False
    cancelled: bool = False


ClarificationHandler = Callable[[ClarificationRequest], ClarificationAnswer]


def build_scope_clarification(content: Any) -> ClarificationRequest | None:
    """Build one plain-language choice from live, fact-checked scope candidates."""
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "clarification_required":
        return None
    if payload.get("stage") not in {"scope_discovery", "request_validation"}:
        return None
    raw_candidates = payload.get("candidates") or []
    choices: list[ClarificationChoice] = []
    if isinstance(raw_candidates, list):
        for candidate in raw_candidates[:MAX_PREDEFINED_CHOICES]:
            if not isinstance(candidate, dict):
                continue
            label = " ".join(
                str(candidate.get("user_label") or candidate.get("canonical_name") or "").split()
            )
            response = " ".join(
                str(candidate.get("user_response") or "").split()
            )
            verification = candidate.get("verification")
            if not label or not response:
                continue
            if isinstance(verification, dict) and verification.get("verified") is not True:
                continue
            choices.append(ClarificationChoice(label=label, response=response))
    if len(choices) < 2:
        choices = []
    question = str(
        payload.get("error")
        or "实时数据源里有多个成分不同的范围，请选择更符合你意思的一个。"
    )
    return ClarificationRequest(
        kind="scope_clarification",
        title="确认选股范围",
        question=question,
        choices=tuple(choices),
        allow_custom=True,
        custom_label="其他说法（自行输入）",
        custom_response_prefix="我想按这个范围分析：",
    )


__all__ = [
    "MAX_PREDEFINED_CHOICES",
    "ClarificationAnswer",
    "ClarificationChoice",
    "ClarificationHandler",
    "ClarificationRequest",
    "build_scope_clarification",
]
