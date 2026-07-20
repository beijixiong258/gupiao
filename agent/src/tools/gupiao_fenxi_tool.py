"""Tool wrapper for single-stock A-share research."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool
from src.tools.gupiao_analysis_cache import store_analysis


class GupiaoFenxiTool(BaseTool):
    name = "gupiao_fenxi"
    description = (
        "First-stage comprehensive diagnosis for one mainland China A-share. It checks data freshness and provenance, "
        "fundamentals, valuation, technical state, volatility, tradability, peer context, risks, and the internal model "
        "workspace. Always call this before answering a new stock diagnosis, upside, buy, sell, or 1-3 trading-day "
        "prediction request. The returned analysis_id is required by gupiao_yuce. Do not quote a profit forecast from "
        "this first-stage result; use gupiao_yuce for the user's requested number."
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
                "default": 1440,
                "description": "Calendar days used for the daily-K peer-panel model; default 1440.",
            },
            "holding_days": {
                "type": "integer",
                "enum": [1, 2, 3],
                "default": 2,
                "description": "Requested sellable holding horizon; for example, a two-session holding question maps to horizon 2.",
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

        full_result = fenxi_gupiao(**kwargs)
        if full_result.get("status") != "ok":
            return json.dumps(full_result, ensure_ascii=False)

        analysis_id = store_analysis(full_result)
        hidden = {"quantitative_analysis", "future_3_trading_days", "analysis_assessment"}
        public_result = {key: value for key, value in full_result.items() if key not in hidden}
        quantitative = full_result.get("quantitative_analysis") or {}
        public_result.update(
            {
                "tool_contract_version": 4,
                "analysis_id": analysis_id,
                "peer_analysis": {
                    "status": quantitative.get("status"),
                    "peer_universe": quantitative.get("peer_universe"),
                    "daily_factor_data": quantitative.get("daily_factor_data"),
                    "methodology": quantitative.get("methodology"),
                    "limitations": quantitative.get("limitations"),
                    "error": quantitative.get("error"),
                },
                "analysis_stage": {
                    "status": "completed",
                    "scope": "行情时点、基本面、估值、技术面、波动、可交易约束、同行和风险已完成",
                    "next_tool_for_numbers": "gupiao_yuce",
                    "instruction": "如用户询问上涨空间、能否盈利、买卖或T+1至T+3，必须继续调用gupiao_yuce",
                },
            }
        )
        return json.dumps(public_result, ensure_ascii=False)
