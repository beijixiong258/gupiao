"""Tool wrapper for single-stock A-share research."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool


class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "Analyze one mainland China A-share by code or Chinese name. Returns verified market-data provenance, "
        "fundamentals, valuation, technical indicators, current tradability, and T+1/T+2/T+3 peer-panel forecasts "
        "with walk-forward validation and after-cost decisions. Use this whenever the user asks how a specific "
        "stock is doing or whether it fits a 1-3 trading-day holding period. Research only; it never submits orders."
    )
    parameters = {
        "type": "object",
        "properties": {
            "gupiao": {"type": "string", "description": "A-share code or Chinese name, for example 600519.SH or 贵州茅台"},
            "source": {
                "type": "string",
                "enum": ["auto", "tushare", "akshare"],
                "description": (
                    "Stock-name resolution and daily-bar source. auto means Tushare first, then AKShare fallback. "
                    "Fundamentals still use their own Tushare-first fallback policy."
                ),
            },
            "history_calendar_days": {
                "type": "integer",
                "minimum": 540,
                "maximum": 1800,
                "default": 1080,
                "description": "Calendar days used for the peer-panel model; default 1080.",
            },
            "holding_days": {
                "type": "integer",
                "enum": [1, 2, 3],
                "default": 2,
                "description": "Requested sellable holding horizon. 两个交易日必须传2并使用T+2模型。",
            },
            "budget_yuan": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Optional research budget used for legal-lot sizing and minimum-commission costs.",
            },
            "config_path": {"type": "string", "description": "Optional path to lianghua_peizhi.json."},
            "run_dir": {"type": "string", "description": "Optional run directory for saving the JSON artifact."},
        },
        "required": ["gupiao"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.gupiao_yanjiu import fenxi_gupiao

        return json.dumps(fenxi_gupiao(**kwargs), ensure_ascii=False)
