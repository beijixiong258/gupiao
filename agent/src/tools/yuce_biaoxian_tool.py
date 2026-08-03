"""Tool wrapper for genuine forward prediction-performance reporting."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool


class YuceBiaoxianTool(BaseTool):
    name = "yuce_biaoxian"
    description = (
        "Settle previously frozen A-share predictions against later daily warehouse prices and report "
        "genuine forward performance. Use this when the user asks whether earlier predictions were right, "
        "how recent live predictions performed, or requests 20/60/120-trading-day direction accuracy, "
        "MAE, net return, interval coverage, or probability Brier score. This tool never backfills invented "
        "predictions: tracking starts only after a prediction was actually recorded."
    )
    parameters = {
        "type": "object",
        "properties": {
            "window": {
                "type": "integer",
                "enum": [20, 60, 120],
                "description": "Optional target trading-day window. Omit it to report 20, 60, and 120 days together.",
            },
            "source_kind": {
                "type": "string",
                "enum": ["all", "single", "board"],
                "default": "all",
                "description": "Report all predictions, single-stock forecasts, or board-selection forecasts.",
            },
        },
        "required": [],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.yuce_liushui import performance_report

        raw_window = kwargs.get("window")
        windows = [int(raw_window)] if raw_window is not None else [20, 60, 120]
        result = performance_report(
            windows=windows,
            source_kind=str(kwargs.get("source_kind") or "all"),
        )
        return json.dumps(result, ensure_ascii=False)


__all__ = ["YuceBiaoxianTool"]
