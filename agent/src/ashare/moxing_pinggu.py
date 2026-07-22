"""Small shared helpers for comparable model baselines and signal abstention."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_baseline_metrics(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_target: np.ndarray,
) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    training_values = np.asarray(training_target, dtype=float)
    paired_length = min(len(actual_values), len(predicted_values))
    actual_values = actual_values[:paired_length]
    predicted_values = predicted_values[:paired_length]
    paired_valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[paired_valid]
    predicted_values = predicted_values[paired_valid]
    training_values = training_values[np.isfinite(training_values)]
    if not len(actual_values) or not len(training_values):
        return {
            "model_mae": None,
            "baselines": {},
            "best_naive_baseline": None,
            "skill_vs_best_naive_baseline": 0.0,
        }
    model_mae = float(np.mean(np.abs(actual_values - predicted_values)))
    baseline_values = {
        "zero_return": 0.0,
        "training_median": float(np.median(training_values)),
        "training_mean": float(np.mean(training_values)),
    }
    baselines = {
        name: {
            "constant_prediction": round(value, 6),
            "mae": round(float(np.mean(np.abs(actual_values - value))), 6),
        }
        for name, value in baseline_values.items()
    }
    best_name = min(baselines, key=lambda name: float(baselines[name]["mae"]))
    best_mae = float(baselines[best_name]["mae"])
    skill = 1.0 - model_mae / best_mae if best_mae > 0 else 0.0
    return {
        "model_mae": round(model_mae, 6),
        "baselines": baselines,
        "best_naive_baseline": best_name,
        "best_naive_baseline_mae": round(best_mae, 6),
        "skill_vs_best_naive_baseline": round(float(skill), 6),
    }


def signal_evidence_gate(
    *,
    validation_passed: bool,
    execution_feasible: bool,
    net_return: float | None,
    positive_probability: float | None,
    quality_score: float | None,
    minimum_net_return: float,
    minimum_positive_probability: float,
    minimum_quality_score: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not validation_passed:
        reasons.append("模型未通过样本外验证")
    if not execution_feasible:
        reasons.append("整手或流动性容量约束不允许按当前资金建仓")
    if net_return is None or float(net_return) < float(minimum_net_return):
        reasons.append("成本后预测收益未达到弃权门槛")
    if positive_probability is None or float(positive_probability) < float(minimum_positive_probability):
        reasons.append("校准上涨概率未达到弃权门槛")
    if quality_score is None or float(quality_score) < float(minimum_quality_score):
        reasons.append("样本外质量分未达到弃权门槛")
    return {
        "actionable_signal": not reasons,
        "decision": "research_candidate" if not reasons else "abstain",
        "reasons": reasons,
        "thresholds": {
            "minimum_net_return_after_cost": float(minimum_net_return),
            "minimum_positive_probability": float(minimum_positive_probability),
            "minimum_quality_score": float(minimum_quality_score),
        },
    }


__all__ = ["regression_baseline_metrics", "signal_evidence_gate"]
