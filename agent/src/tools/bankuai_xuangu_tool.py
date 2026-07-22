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
        "Uses a per-date XGBRanker for recommendation order and a separate board-neutral return model for numeric forecasts; "
        "selection_confidence and return_confidence must be explained separately. "
        "Always returns model-ranked recommendations when forecast data is available; failed validation lowers confidence "
        "but does not suppress recommendations or numeric return estimates. "
        "Signals use the completed close, entry is the next session open, and T+1 is the first later session when the new shares can be sold. "
        "Use this for natural-language requests to pick stocks from a sector and compare those three sellable horizons. "
        "One batch returns at most 8 candidates. For a later batch, reuse selection_id and pass next_offset so ranking "
        "continues without duplicates. "
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
            "top_n": {
                "type": "integer",
                "default": 8,
                "description": "Requested candidate count. The tool defaults to and hard-caps each batch at 8.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Zero-based offset for the next batch; use the previous result's next_offset.",
            },
            "selection_id": {
                "type": "string",
                "description": "Stable sequence identifier from the previous batch; required when offset is greater than 0.",
            },
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
