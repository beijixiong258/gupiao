"""Tool wrapper for sector-constrained A-share selection and T+3 prediction."""

from __future__ import annotations

import json
from typing import Any

from src.agent.tools import BaseTool
class BankuaiXuanguTool(BaseTool):
    name = "bankuai_xuangu"
    description = (
        "Select mainland China A-shares only from a user-specified industry or concept board. "
        "Runs chronological out-of-sample validation and returns cost-adjusted T+1, T+2, and T+3 sellable-horizon forecasts. "
        "Signals use the completed close, entry is the next session open, and T+1 is the first later session when the new shares can be sold. "
        "Use this for natural-language requests to pick stocks from a sector and compare those three sellable horizons. "
        "Research only; it never connects to a broker or submits orders."
    )
    parameters = {
        "type": "object",
        "properties": {
            "bankuai": {"type": "string", "description": "Chinese industry or concept board name, for example 白酒 or 人工智能"},
            "bankuai_leixing": {
                "type": "string",
                "enum": ["auto", "hangye", "gainian"],
                "description": "Board type. auto resolves industry or concept automatically.",
            },
            "top_n": {"type": "integer", "description": "Maximum recommendations to return, default 3 and max 10."},
            "source": {
                "type": "string",
                "enum": ["auto", "tushare", "akshare"],
                "description": "Daily-bar source. auto uses Tushare first and AKShare as fallback.",
            },
            "config_path": {"type": "string", "description": "Optional path to lianghua_peizhi.json."},
            "run_dir": {"type": "string", "description": "Optional run directory for saving JSON and CSV artifacts."},
        },
        "required": ["bankuai"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs: Any) -> str:
        from src.ashare.bankuai_yuce import bankuai_xuangu

        return json.dumps(bankuai_xuangu(**kwargs), ensure_ascii=False)
