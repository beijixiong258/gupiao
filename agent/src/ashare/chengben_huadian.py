"""A-share transaction-cost assumptions used to adjust model forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "jiaoyi_chengben.json"


@dataclass(frozen=True)
class CostScenario:
    name: str
    buy_commission_rate: float
    sell_commission_rate: float
    stamp_tax_sell_rate: float
    transfer_fee_buy_rate: float
    transfer_fee_sell_rate: float
    buy_slippage_bps: float
    sell_slippage_bps: float
    min_commission_yuan: float


DEFAULT_SCENARIOS = (
    CostScenario("zero_cost", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    CostScenario("normal_cost", 0.00025, 0.00025, 0.0005, 0.00001, 0.00001, 2.0, 2.0, 5.0),
    CostScenario("stress_cost", 0.0003, 0.0003, 0.0005, 0.00001, 0.00001, 5.0, 5.0, 5.0),
)


def _scenario_from_dict(raw: dict[str, Any]) -> CostScenario:
    required = {
        "name",
        "buy_commission_rate",
        "sell_commission_rate",
        "stamp_tax_sell_rate",
        "transfer_fee_buy_rate",
        "transfer_fee_sell_rate",
        "buy_slippage_bps",
        "sell_slippage_bps",
        "min_commission_yuan",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"cost scenario missing fields: {missing}")
    return CostScenario(
        name=str(raw["name"]),
        buy_commission_rate=float(raw["buy_commission_rate"]),
        sell_commission_rate=float(raw["sell_commission_rate"]),
        stamp_tax_sell_rate=float(raw["stamp_tax_sell_rate"]),
        transfer_fee_buy_rate=float(raw["transfer_fee_buy_rate"]),
        transfer_fee_sell_rate=float(raw["transfer_fee_sell_rate"]),
        buy_slippage_bps=float(raw["buy_slippage_bps"]),
        sell_slippage_bps=float(raw["sell_slippage_bps"]),
        min_commission_yuan=float(raw["min_commission_yuan"]),
    )


def _load_cost_config(config_path: str | None) -> tuple[float, tuple[CostScenario, ...], str, list[str]]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"cost config not found, using built-in defaults: {path}")
        return 20000.0, DEFAULT_SCENARIOS, str(path), errors
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        notional_yuan = float(raw.get("notional_yuan", 20000.0))
        scenarios = tuple(_scenario_from_dict(item) for item in raw.get("scenarios", []))
        if not scenarios:
            raise ValueError("scenarios must not be empty")
        return notional_yuan, scenarios, str(path), errors
    except Exception as exc:
        errors.append(f"cost config invalid, using built-in defaults: {path}: {exc}")
        return 20000.0, DEFAULT_SCENARIOS, str(path), errors


def _commission_rate(rate: float, min_commission_yuan: float, notional_yuan: float) -> float:
    if notional_yuan > 0:
        return max(rate, min_commission_yuan / notional_yuan)
    return rate


__all__ = ["CostScenario", "DEFAULT_CONFIG_PATH", "DEFAULT_SCENARIOS"]
