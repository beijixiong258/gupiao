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
_PREDICTION_CONTEXTS: OrderedDict[str, dict[str, Any]] = OrderedDict()


def store_analysis(result: dict[str, Any], prediction_context: dict[str, Any] | None = None) -> str:
    """Store one analysis result and its in-process prediction context.

    The public analysis result is deliberately JSON-safe and does not carry
    pandas frames.  The optional context is kept separately so the expensive
    prediction model can be fitted only when the second-stage tool is called.
    Both stores have the same bounded lifetime and are invalidated on process
    restart.
    """
    analysis_id = f"fx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    with _LOCK:
        _ANALYSES[analysis_id] = result
        _ANALYSES.move_to_end(analysis_id)
        if prediction_context is not None:
            _PREDICTION_CONTEXTS[analysis_id] = prediction_context
            _PREDICTION_CONTEXTS.move_to_end(analysis_id)
        while len(_ANALYSES) > _MAX_ANALYSES:
            old_id, _ = _ANALYSES.popitem(last=False)
            _PREDICTION_CONTEXTS.pop(old_id, None)
        while len(_PREDICTION_CONTEXTS) > _MAX_ANALYSES:
            _PREDICTION_CONTEXTS.popitem(last=False)
    return analysis_id


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    key = str(analysis_id).strip()
    with _LOCK:
        result = _ANALYSES.get(key)
        if result is not None:
            _ANALYSES.move_to_end(key)
        return result


def get_prediction_context(analysis_id: str) -> dict[str, Any] | None:
    """Return the private context needed for an on-demand prediction fit."""
    key = str(analysis_id).strip()
    with _LOCK:
        context = _PREDICTION_CONTEXTS.get(key)
        if context is not None:
            _PREDICTION_CONTEXTS.move_to_end(key)
        return context


def clear_analysis_cache() -> None:
    """Clear the bounded cache; intended for process cleanup and tests."""
    with _LOCK:
        _ANALYSES.clear()
        _PREDICTION_CONTEXTS.clear()


__all__ = ["clear_analysis_cache", "get_analysis", "get_prediction_context", "store_analysis"]
