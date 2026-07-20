"""Small in-process cache joining diagnosis and prediction tool calls."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from threading import RLock
from typing import Any
from uuid import uuid4

_MAX_ANALYSES = 32
_LOCK = RLock()
_ANALYSES: OrderedDict[str, dict[str, Any]] = OrderedDict()


def store_analysis(result: dict[str, Any]) -> str:
    analysis_id = f"fx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    with _LOCK:
        _ANALYSES[analysis_id] = result
        _ANALYSES.move_to_end(analysis_id)
        while len(_ANALYSES) > _MAX_ANALYSES:
            _ANALYSES.popitem(last=False)
    return analysis_id


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    key = str(analysis_id).strip()
    with _LOCK:
        result = _ANALYSES.get(key)
        if result is not None:
            _ANALYSES.move_to_end(key)
        return result


def clear_analysis_cache() -> None:
    """Clear the bounded cache; intended for process cleanup and tests."""
    with _LOCK:
        _ANALYSES.clear()


__all__ = ["clear_analysis_cache", "get_analysis", "store_analysis"]
