"""Tool wrapper for single-stock A-share research."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool
class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "Analyze one mainland China A-share by code or Chinese name. Returns verified market-data provenance, "
        "fundamentals, valuation, technical indicators, risk evidence, and applicable A-share trading rules. "
        "Use this whenever the user asks how a specific stock is doing. Research only; it never submits orders."
    )
    parameters = {
        "type": "object",
        "properties": {
            "gupiao": {"type": "string", "description": "A-share code or Chinese name, for example 600519.SH or 贵州茅台"},
            "source": {
                "type": "string",
                "enum": ["auto", "tushare", "akshare"],
                "description": "Market-data source. auto means Tushare first, then AKShare fallback.",
            },
            "history_calendar_days": {
                "type": "integer",
                "description": "Calendar days of daily history, default 540 and minimum 180.",
            },
            "run_dir": {"type": "string", "description": "Optional run directory for saving the JSON artifact."},
        },
        "required": ["gupiao"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.gupiao_yanjiu import fenxi_gupiao

        return json.dumps(fenxi_gupiao(**kwargs), ensure_ascii=False)
