"""Freeze model predictions and settle them against later warehouse prices."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.ashare.riping_cangku import DAILY_WAREHOUSE_PATH
from src.ashare.shuju_yuan import CACHE_DIR, _normalize_code


PREDICTION_LEDGER_PATH = CACHE_DIR / "prediction_ledger.sqlite3"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw[:10] if pattern == "%Y-%m-%d" else raw[:8], pattern).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            continue
    return None


def _db_date(value: str | None) -> str | None:
    return value.replace("-", "") if value else None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stock_code(value: Any) -> str | None:
    try:
        return _normalize_code(str(value or ""))
    except (TypeError, ValueError):
        return None


def _interval(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    low, high = _number(value[0]), _number(value[1])
    if low is None or high is None:
        return None, None
    return min(low, high), max(low, high)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if readonly:
        connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True, timeout=30.0)
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    connection.row_factory = sqlite3.Row
    return connection


def initialize_prediction_ledger(
    path: Path | str = PREDICTION_LEDGER_PATH,
) -> str:
    resolved = Path(path).expanduser().resolve()
    with _connect(resolved) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                scope_name TEXT,
                ts_code TEXT NOT NULL,
                stock_name TEXT,
                signal_date TEXT NOT NULL,
                entry_date TEXT,
                target_date TEXT,
                horizon INTEGER NOT NULL,
                return_basis TEXT NOT NULL,
                forecast_status TEXT,
                validation_passed INTEGER NOT NULL DEFAULT 0,
                quality_label TEXT,
                is_recommended INTEGER NOT NULL DEFAULT 0,
                predicted_gross_return REAL,
                predicted_net_return REAL,
                positive_probability REAL,
                interval_low REAL,
                interval_high REAL,
                interval_return_kind TEXT,
                cost_rate REAL,
                metadata_json TEXT NOT NULL,
                realization_status TEXT NOT NULL DEFAULT 'pending',
                actual_start_price REAL,
                actual_exit_price REAL,
                actual_gross_return REAL,
                actual_net_return REAL,
                direction_hit INTEGER,
                interval_hit INTEGER,
                absolute_error REAL,
                squared_error REAL,
                brier_score REAL,
                realized_at TEXT,
                realization_note TEXT,
                UNIQUE(source_kind, source_id, mode, ts_code, horizon)
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_realization
                ON predictions(realization_status, target_date);
            CREATE INDEX IF NOT EXISTS idx_predictions_signal
                ON predictions(signal_date, source_kind, horizon);
            """
        )
    return str(resolved)


def _prediction_id(
    *, source_kind: str, source_id: str, mode: str, ts_code: str, horizon: int
) -> str:
    payload = "|".join([source_kind, source_id, mode, ts_code, str(horizon)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _insert_prediction(record: dict[str, Any], *, path: Path | str) -> bool:
    resolved = Path(path).expanduser().resolve()
    initialize_prediction_ledger(resolved)
    columns = [
        "prediction_id",
        "created_at",
        "source_kind",
        "source_id",
        "mode",
        "scope_name",
        "ts_code",
        "stock_name",
        "signal_date",
        "entry_date",
        "target_date",
        "horizon",
        "return_basis",
        "forecast_status",
        "validation_passed",
        "quality_label",
        "is_recommended",
        "predicted_gross_return",
        "predicted_net_return",
        "positive_probability",
        "interval_low",
        "interval_high",
        "interval_return_kind",
        "cost_rate",
        "metadata_json",
    ]
    values = [record.get(column) for column in columns]
    with _connect(resolved) as connection:
        cursor = connection.execute(
            f"INSERT OR IGNORE INTO predictions({','.join(columns)}) "
            f"VALUES({','.join('?' for _ in columns)})",
            values,
        )
        connection.commit()
        return bool(cursor.rowcount)


def _effective_cost_rate(gross: float | None, net: float | None) -> float | None:
    if gross is None or net is None or 1.0 + gross <= 0:
        return None
    rate = 1.0 - (1.0 + net) / (1.0 + gross)
    return min(max(float(rate), 0.0), 0.25) if math.isfinite(rate) else None


def record_single_prediction(
    result: dict[str, Any],
    *,
    path: Path | str = PREDICTION_LEDGER_PATH,
) -> dict[str, Any]:
    forecast = result.get("forecast") or {}
    request = result.get("request") or {}
    stock = result.get("stock") or {}
    if result.get("status") != "ok" or not forecast:
        return {"status": "skipped", "reason": "没有可冻结的单股预测", "recorded": 0}
    mode = str(request.get("mode") or "future_close")
    horizon = int(str(request.get("horizon") or "T+1").replace("T+", ""))
    code = _stock_code(stock.get("ts_code") or stock.get("code"))
    signal_date = _date_text(result.get("analysis_as_of"))
    source_id = str(result.get("analysis_id") or "").strip()
    if not code or not signal_date or not source_id:
        return {"status": "skipped", "reason": "缺少股票代码、信号日或分析编号", "recorded": 0}

    if mode == "future_close":
        gross = _number(forecast.get("cumulative_return_from_signal_close"))
        net = _number(forecast.get("estimated_return_after_roundtrip_cost"))
        entry_date = signal_date
        target_date = _date_text(forecast.get("target_trade_date"))
        return_basis = "signal_close_to_target_close"
        preferred = forecast.get("preferred_return_interval_80")
        interval_kind = "gross"
    else:
        gross = _number(forecast.get("entry_to_exit_gross_return"))
        net = _number(forecast.get("estimated_net_return_after_cost"))
        entry_date = _date_text(forecast.get("assumed_entry_date"))
        target_date = _date_text(forecast.get("scenario_exit_date"))
        return_basis = "next_open_to_sellable_close"
        preferred = forecast.get("preferred_net_return_interval_80")
        interval_kind = "net"
        if not preferred:
            preferred = forecast.get("preferred_return_interval_80")
            interval_kind = "gross"
    if not preferred:
        preferred = (
            forecast.get("conformal_net_return_interval_80")
            if mode != "future_close"
            else forecast.get("conformal_return_interval_80")
        )
        interval_kind = "net" if mode != "future_close" else "gross"
    low, high = _interval(preferred)
    probability = _number(forecast.get("calibrated_direction_positive_probability"))
    cost_rate = _effective_cost_rate(gross, net)
    record = {
        "prediction_id": _prediction_id(
            source_kind="single", source_id=source_id, mode=mode, ts_code=code, horizon=horizon
        ),
        "created_at": str(result.get("generated_at") or _now_text()),
        "source_kind": "single",
        "source_id": source_id,
        "mode": mode,
        "scope_name": None,
        "ts_code": code,
        "stock_name": stock.get("name"),
        "signal_date": signal_date,
        "entry_date": entry_date,
        "target_date": target_date,
        "horizon": horizon,
        "return_basis": return_basis,
        "forecast_status": result.get("forecast_status"),
        "validation_passed": int(bool(forecast.get("validation_passed"))),
        "quality_label": forecast.get("model_quality"),
        "is_recommended": int((result.get("model_recommendation") or {}).get("decision") == "recommend"),
        "predicted_gross_return": gross,
        "predicted_net_return": net,
        "positive_probability": probability,
        "interval_low": low,
        "interval_high": high,
        "interval_return_kind": interval_kind if low is not None else None,
        "cost_rate": cost_rate,
        "metadata_json": _json_text(result),
    }
    inserted = _insert_prediction(record, path=path)
    return {
        "status": "recorded" if inserted else "already_recorded",
        "recorded": int(inserted),
        "prediction_id": record["prediction_id"],
        "ledger_path": str(Path(path).expanduser().resolve()),
    }


def _board_name(result: dict[str, Any]) -> str | None:
    board = result.get("board") or {}
    for key in ("matched_name", "board_name", "name", "query"):
        if board.get(key):
            return str(board[key])
    return None


def record_board_predictions(
    result: dict[str, Any],
    *,
    path: Path | str = PREDICTION_LEDGER_PATH,
) -> dict[str, Any]:
    if result.get("status") != "ok":
        return {"status": "skipped", "reason": "板块预测未成功", "recorded": 0}
    selection = result.get("selection") or {}
    source_id = str(selection.get("selection_id") or "").strip()
    signal_date = _date_text(result.get("as_of"))
    if not source_id or not signal_date:
        return {"status": "skipped", "reason": "缺少候选序列编号或信号日", "recorded": 0}
    recommended_codes = {
        code
        for item in result.get("recommended_candidates", [])
        if (code := _stock_code(item.get("ts_code"))) is not None
    }
    candidates: dict[str, dict[str, Any]] = {}
    for item in [*(result.get("model_ranking") or []), *(result.get("recommended_candidates") or [])]:
        code = _stock_code(item.get("ts_code"))
        if code:
            candidates[code] = item
    validation = (result.get("validation") or {}).get("horizons") or {}
    recorded = 0
    duplicates = 0
    for code, item in candidates.items():
        position = item.get("position_and_cost") or {}
        cost_rate = _number(position.get("estimated_roundtrip_cost_rate"))
        for horizon in (1, 2, 3):
            label = f"T+{horizon}"
            forecast = (item.get("forecast") or {}).get(label) or {}
            gross = _number(forecast.get("entry_to_exit_gross_return"))
            if gross is None:
                continue
            net = _number(forecast.get("estimated_net_return_after_cost"))
            interval_value = forecast.get("preferred_net_return_interval_80")
            interval_kind = "net"
            if not interval_value:
                interval_value = forecast.get("preferred_return_interval_80")
                interval_kind = "gross"
            low, high = _interval(interval_value)
            metrics = validation.get(label) or {}
            record = {
                "prediction_id": _prediction_id(
                    source_kind="board",
                    source_id=source_id,
                    mode="holding_return",
                    ts_code=code,
                    horizon=horizon,
                ),
                "created_at": _now_text(),
                "source_kind": "board",
                "source_id": source_id,
                "mode": "holding_return",
                "scope_name": _board_name(result),
                "ts_code": code,
                "stock_name": item.get("name"),
                "signal_date": signal_date,
                "entry_date": None,
                "target_date": None,
                "horizon": horizon,
                "return_basis": "next_open_to_sellable_close",
                "forecast_status": "validated" if metrics.get("validation_passed") else "model_estimate",
                "validation_passed": int(bool(metrics.get("validation_passed"))),
                "quality_label": forecast.get("model_quality"),
                "is_recommended": int(code in recommended_codes),
                "predicted_gross_return": gross,
                "predicted_net_return": net,
                "positive_probability": _number(
                    forecast.get("direction_model_positive_probability")
                ),
                "interval_low": low,
                "interval_high": high,
                "interval_return_kind": interval_kind if low is not None else None,
                "cost_rate": cost_rate or _effective_cost_rate(gross, net),
                "metadata_json": _json_text(
                    {
                        "board": result.get("board"),
                        "selection": selection,
                        "candidate": {
                            key: item.get(key)
                            for key in [
                                "ts_code",
                                "name",
                                "as_of",
                                "selection_score",
                                "weighted_expected_net_return",
                                "strongest_forecast_horizon",
                                "ranking_validation_mode",
                                "model_recommendation",
                                "signal_gate",
                            ]
                        },
                        "forecast": forecast,
                        "horizon_validation": {
                            key: metrics.get(key)
                            for key in [
                                "validation_passed",
                                "selection_validation_passed",
                                "return_validation_passed",
                                "quality_score",
                                "quality_label",
                                "direction_accuracy",
                                "mean_daily_rank_ic",
                                "experiment_fingerprint",
                            ]
                        },
                    }
                ),
            }
            if _insert_prediction(record, path=path):
                recorded += 1
            else:
                duplicates += 1
    return {
        "status": "recorded" if recorded else "already_recorded" if duplicates else "skipped",
        "recorded": recorded,
        "already_recorded": duplicates,
        "candidate_count": len(candidates),
        "ledger_path": str(Path(path).expanduser().resolve()),
    }


def _market_sessions_after(
    warehouse: sqlite3.Connection, signal_date: str, count: int
) -> list[str]:
    rows = warehouse.execute(
        """
        SELECT cal_date
        FROM trade_calendar
        WHERE is_open=1 AND cal_date>?
        ORDER BY cal_date
        LIMIT ?
        """,
        (_db_date(signal_date), int(count)),
    ).fetchall()
    return [datetime.strptime(str(row["cal_date"]), "%Y%m%d").strftime("%Y-%m-%d") for row in rows]


def _bar(
    warehouse: sqlite3.Connection, ts_code: str, trade_date: str
) -> sqlite3.Row | None:
    return warehouse.execute(
        "SELECT open,close FROM daily_bars WHERE ts_code=? AND trade_date=?",
        (ts_code, _db_date(trade_date)),
    ).fetchone()


def settle_predictions(
    *,
    path: Path | str = PREDICTION_LEDGER_PATH,
    warehouse_path: Path | str = DAILY_WAREHOUSE_PATH,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    warehouse_resolved = Path(warehouse_path).expanduser().resolve()
    initialize_prediction_ledger(resolved)
    if not warehouse_resolved.is_file():
        return {
            "status": "warehouse_unavailable",
            "settled": 0,
            "message": "全市场日线仓库不存在，暂时不能揭晓预测",
        }
    settled = 0
    unavailable = 0
    pending = 0
    with _connect(resolved) as ledger, _connect(warehouse_resolved, readonly=True) as warehouse:
        latest_row = warehouse.execute("SELECT MAX(trade_date) AS value FROM daily_bars").fetchone()
        latest_db_date = str(latest_row["value"] or "")
        rows = ledger.execute(
            "SELECT * FROM predictions WHERE realization_status='pending' ORDER BY created_at"
        ).fetchall()
        for row in rows:
            prediction = dict(row)
            entry_date = _date_text(prediction.get("entry_date"))
            target_date = _date_text(prediction.get("target_date"))
            if prediction["return_basis"] == "next_open_to_sellable_close" and (
                not entry_date or not target_date
            ):
                sessions = _market_sessions_after(
                    warehouse, prediction["signal_date"], int(prediction["horizon"]) + 1
                )
                if len(sessions) >= int(prediction["horizon"]) + 1:
                    entry_date = sessions[0]
                    target_date = sessions[int(prediction["horizon"])]
            if not target_date or not latest_db_date or latest_db_date < str(_db_date(target_date)):
                pending += 1
                continue
            if prediction["return_basis"] == "signal_close_to_target_close":
                start_bar = _bar(warehouse, prediction["ts_code"], prediction["signal_date"])
                exit_bar = _bar(warehouse, prediction["ts_code"], target_date)
                start_price = _number(start_bar["close"]) if start_bar else None
            else:
                start_bar = _bar(warehouse, prediction["ts_code"], entry_date) if entry_date else None
                exit_bar = _bar(warehouse, prediction["ts_code"], target_date)
                start_price = _number(start_bar["open"]) if start_bar else None
            exit_price = _number(exit_bar["close"]) if exit_bar else None
            if start_price is None or exit_price is None or start_price <= 0:
                note = "目标日期已进入仓库，但股票缺少规定入口或退出行情，可能为停牌或退市"
                ledger.execute(
                    """
                    UPDATE predictions
                    SET entry_date=?,target_date=?,realization_status='unavailable',
                        realized_at=?,realization_note=?
                    WHERE prediction_id=?
                    """,
                    (entry_date, target_date, _now_text(), note, prediction["prediction_id"]),
                )
                unavailable += 1
                continue
            actual_gross = exit_price / start_price - 1.0
            cost_rate = _number(prediction.get("cost_rate")) or 0.0
            actual_net = (1.0 + actual_gross) * (1.0 - cost_rate) - 1.0
            predicted_gross = _number(prediction.get("predicted_gross_return"))
            predicted_net = _number(prediction.get("predicted_net_return"))
            direction_hit = (
                int((predicted_gross > 0) == (actual_gross > 0))
                if predicted_gross is not None
                else None
            )
            comparison_actual = actual_net if predicted_net is not None else actual_gross
            comparison_predicted = predicted_net if predicted_net is not None else predicted_gross
            error = (
                comparison_actual - comparison_predicted
                if comparison_predicted is not None
                else None
            )
            interval_actual = (
                actual_net if prediction.get("interval_return_kind") == "net" else actual_gross
            )
            low = _number(prediction.get("interval_low"))
            high = _number(prediction.get("interval_high"))
            interval_hit = int(low <= interval_actual <= high) if low is not None and high is not None else None
            probability = _number(prediction.get("positive_probability"))
            brier = (
                (probability - float(actual_gross > 0)) ** 2 if probability is not None else None
            )
            ledger.execute(
                """
                UPDATE predictions
                SET entry_date=?,target_date=?,realization_status='realized',
                    actual_start_price=?,actual_exit_price=?,actual_gross_return=?,actual_net_return=?,
                    direction_hit=?,interval_hit=?,absolute_error=?,squared_error=?,brier_score=?,
                    realized_at=?,realization_note=NULL
                WHERE prediction_id=?
                """,
                (
                    entry_date,
                    target_date,
                    start_price,
                    exit_price,
                    actual_gross,
                    actual_net,
                    direction_hit,
                    interval_hit,
                    abs(error) if error is not None else None,
                    error * error if error is not None else None,
                    brier,
                    _now_text(),
                    prediction["prediction_id"],
                ),
            )
            settled += 1
        ledger.commit()
    return {
        "status": "ok",
        "settled": settled,
        "marked_unavailable": unavailable,
        "still_pending": pending,
        "warehouse_latest_date": (
            datetime.strptime(latest_db_date, "%Y%m%d").strftime("%Y-%m-%d")
            if latest_db_date
            else None
        ),
    }


def _mean(values: Iterable[Any]) -> float | None:
    clean = [_number(value) for value in values]
    clean = [value for value in clean if value is not None]
    return sum(clean) / len(clean) if clean else None


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "target_dates": 0, "status": "no_realized_predictions"}
    direction_values = [row["direction_hit"] for row in rows if row.get("direction_hit") is not None]
    interval_values = [row["interval_hit"] for row in rows if row.get("interval_hit") is not None]
    probabilities = [row for row in rows if row.get("brier_score") is not None]
    actual_positive_rate = _mean(
        float((actual := _number(row.get("actual_gross_return"))) is not None and actual > 0)
        for row in probabilities
    )
    baseline_brier = (
        actual_positive_rate * (1.0 - actual_positive_rate)
        if actual_positive_rate is not None
        else None
    )
    brier = _mean(row.get("brier_score") for row in probabilities)
    probability_skill = (
        1.0 - brier / baseline_brier
        if brier is not None and baseline_brier is not None and baseline_brier > 0
        else None
    )
    return {
        "status": "ok",
        "samples": len(rows),
        "signal_dates": len({row["signal_date"] for row in rows}),
        "target_dates": len({row["target_date"] for row in rows}),
        "direction_accuracy": _rounded(_mean(direction_values)),
        "mae": _rounded(_mean(row.get("absolute_error") for row in rows)),
        "rmse": _rounded(
            math.sqrt(value) if (value := _mean(row.get("squared_error") for row in rows)) is not None else None
        ),
        "mean_predicted_net_return": _rounded(
            _mean(row.get("predicted_net_return") for row in rows)
        ),
        "mean_actual_net_return": _rounded(_mean(row.get("actual_net_return") for row in rows)),
        "interval_samples": len(interval_values),
        "interval_coverage": _rounded(_mean(interval_values)),
        "probability_samples": len(probabilities),
        "actual_positive_rate": _rounded(actual_positive_rate),
        "brier_score": _rounded(brier),
        "brier_skill_vs_window_constant": _rounded(probability_skill),
    }


def _evidence_level(signal_dates: int) -> str:
    if signal_dates < 20:
        return "insufficient_live_history"
    if signal_dates < 60:
        return "early_live_tracking"
    if signal_dates < 120:
        return "provisional_live_evidence"
    return "mature_live_tracking"


def performance_report(
    *,
    windows: Iterable[int] = (20, 60, 120),
    source_kind: str = "all",
    path: Path | str = PREDICTION_LEDGER_PATH,
    settle_first: bool = True,
) -> dict[str, Any]:
    if source_kind not in {"all", "single", "board"}:
        raise ValueError("source_kind 必须是 all、single 或 board")
    resolved = Path(path).expanduser().resolve()
    initialize_prediction_ledger(resolved)
    settlement = settle_predictions(path=resolved) if settle_first else None
    with _connect(resolved) as connection:
        conditions = ["realization_status='realized'"]
        parameters: list[Any] = []
        if source_kind != "all":
            conditions.append("source_kind=?")
            parameters.append(source_kind)
        realized_rows = [
            dict(row)
            for row in connection.execute(
                f"SELECT * FROM predictions WHERE {' AND '.join(conditions)} ORDER BY target_date",
                parameters,
            ).fetchall()
        ]
        status_rows = connection.execute(
            "SELECT realization_status,COUNT(*) AS count FROM predictions GROUP BY realization_status"
        ).fetchall()
    target_dates = sorted({row["target_date"] for row in realized_rows if row.get("target_date")})
    reports: dict[str, Any] = {}
    for requested_window in sorted({max(1, int(value)) for value in windows}):
        selected_dates = target_dates[-requested_window:]
        selected = [row for row in realized_rows if row.get("target_date") in selected_dates]
        groups: dict[str, Any] = {"overall": _aggregate(selected)}
        for kind in ("single", "board"):
            kind_rows = [row for row in selected if row["source_kind"] == kind]
            groups[kind] = _aggregate(kind_rows)
        groups["recommended_only"] = _aggregate(
            [row for row in selected if bool(row.get("is_recommended"))]
        )
        groups["by_horizon"] = {
            f"T+{horizon}": _aggregate(
                [row for row in selected if int(row["horizon"]) == horizon]
            )
            for horizon in (1, 2, 3)
        }
        overall_signal_dates = int(groups["overall"].get("signal_dates", 0))
        reports[str(requested_window)] = {
            "requested_trading_day_window": requested_window,
            "actual_target_date_count": len(selected_dates),
            "date_range": [selected_dates[0], selected_dates[-1]] if selected_dates else [None, None],
            "live_evidence_level": _evidence_level(overall_signal_dates),
            "metrics": groups,
        }
    return {
        "status": "ok",
        "ledger_path": str(resolved),
        "source_kind": source_kind,
        "settlement": settlement,
        "record_counts": {str(row["realization_status"]): int(row["count"]) for row in status_rows},
        "windows": reports,
        "metric_notes": {
            "returns": "实际收益使用冻结预测规定的入口和退出日；有成本参数时同时计算扣成本收益",
            "mae_rmse": "优先比较成本后预测与成本后实际收益，否则比较毛收益",
            "interval_coverage": "按冻结时记录的毛收益或成本后收益区间计算",
            "brier_skill": "窗口内描述性比较；正值表示优于该窗口固定上涨比例，不能替代滚动校准验证",
            "live_evidence": "少于20个独立信号日只表示记录不足，不授予live_validated标签",
        },
    }


__all__ = [
    "PREDICTION_LEDGER_PATH",
    "initialize_prediction_ledger",
    "performance_report",
    "record_board_predictions",
    "record_single_prediction",
    "settle_predictions",
]
