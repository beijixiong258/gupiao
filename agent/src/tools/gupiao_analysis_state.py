"""分析与按需预测之间的进程内会话状态，不保存任何市场时间序列。"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any
from uuid import uuid4


def _contains_tabular_market_data(value: Any, seen: set[int] | None = None) -> bool:
    """阻止 DataFrame/Series 越过分析与预测的会话边界。"""
    value_type = type(value)
    if value_type.__module__.startswith("pandas") and value_type.__name__ in {
        "DataFrame",
        "Series",
    }:
        return True
    if not isinstance(value, (dict, list, tuple, set)):
        return False
    visited = seen if seen is not None else set()
    identity = id(value)
    if identity in visited:
        return False
    visited.add(identity)
    items = value.values() if isinstance(value, dict) else value
    return any(_contains_tabular_market_data(item, visited) for item in items)


@dataclass(frozen=True)
class AnalysisSession:
    """一次量化分析的会话交接信息；进程退出后自然失效。"""

    result: dict[str, Any]
    prediction_context: dict[str, Any] | None
    created_at: datetime


class AnalysisSessionStore:
    """有界、线程安全的会话仓储，仅用于自然语言多轮衔接。"""

    def __init__(self, maximum_sessions: int = 32) -> None:
        if maximum_sessions <= 0:
            raise ValueError("maximum_sessions 必须为正整数")
        self._maximum_sessions = maximum_sessions
        self._lock = RLock()
        self._sessions: OrderedDict[str, AnalysisSession] = OrderedDict()

    def save(
        self,
        result: dict[str, Any],
        prediction_context: dict[str, Any] | None = None,
    ) -> str:
        """保存分析结果和候选身份；禁止放入行情 DataFrame。"""
        if _contains_tabular_market_data(result) or _contains_tabular_market_data(
            prediction_context
        ):
            raise ValueError("分析会话状态禁止保存 DataFrame 或 Series")
        analysis_id = f"fx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        session = AnalysisSession(
            result=copy.deepcopy(result),
            prediction_context=copy.deepcopy(prediction_context),
            created_at=datetime.now(),
        )
        with self._lock:
            self._sessions[analysis_id] = session
            self._sessions.move_to_end(analysis_id)
            while len(self._sessions) > self._maximum_sessions:
                self._sessions.popitem(last=False)
        return analysis_id

    def get(self, analysis_id: str) -> AnalysisSession | None:
        key = str(analysis_id).strip()
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            self._sessions.move_to_end(key)
            return copy.deepcopy(session)

    def contains(self, analysis_id: str) -> bool:
        """只检查当前进程是否仍持有会话，不延长其生命周期。"""
        key = str(analysis_id).strip()
        with self._lock:
            return key in self._sessions

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


analysis_session_store = AnalysisSessionStore()


__all__ = ["AnalysisSession", "AnalysisSessionStore", "analysis_session_store"]
