"""Execution-timing and ranking safeguards for A-share board forecasts."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ashare import bankuai_yuce
from src.ashare.bankuai_yuce import (
    _fetch_histories,
    _best_name,
    _paginate_candidates,
    _position_for_budget,
    _prediction_rows,
    goujian_moxing_shuju,
    xunlian_yuce_moxing,
)
from src.ashare.gupiao_yanjiu import FEATURE_COLUMNS, jiazai_lianghua_peizhi


def _price_frame(dates: list[str], opens: list[float], closes: list[float]) -> pd.DataFrame:
    values = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(dates),
            "open": opens,
            "close": closes,
        }
    )
    values["high"] = values[["open", "close"]].max(axis=1) + 0.1
    values["low"] = values[["open", "close"]].min(axis=1) - 0.1
    values["volume"] = 1_000_000.0
    return values


def test_labels_use_next_market_open_and_do_not_skip_over_suspension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bankuai_yuce, "jisuan_tezheng_biao", lambda frame: frame.copy())
    histories = {
        "600001.SH": _price_frame(
            ["2025-01-02", "2025-01-06", "2025-01-07", "2025-01-08"],
            [10, 12, 13, 14],
            [10.5, 12.5, 13.5, 14.5],
        ),
        "600002.SH": _price_frame(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"],
            [10, 11, 12, 13, 14],
            [10.5, 11.5, 13, 13.5, 14.5],
        ),
    }
    panel = goujian_moxing_shuju(histories, {key: key for key in histories}, [1, 2, 3])

    suspended = panel[(panel["ts_code"] == "600001.SH") & (panel["trade_date"] == pd.Timestamp("2025-01-02"))].iloc[0]
    assert suspended["entry_date_t1"] == pd.Timestamp("2025-01-03")
    assert pd.isna(suspended["entry_open_t1"])
    assert pd.isna(suspended["target_t1"])

    normal = panel[(panel["ts_code"] == "600002.SH") & (panel["trade_date"] == pd.Timestamp("2025-01-02"))].iloc[0]
    assert normal["entry_date_t1"] == pd.Timestamp("2025-01-03")
    assert normal["target_date_t1"] == pd.Timestamp("2025-01-06")
    assert normal["target_t1"] == pytest.approx(13.0 / 11.0 - 1.0)


def test_labels_exclude_unbuyable_one_price_limit_up_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bankuai_yuce, "jisuan_tezheng_biao", lambda frame: frame.copy())
    history = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "open": [10.0, 11.0, 11.2],
            "high": [10.2, 11.0, 11.5],
            "low": [9.8, 11.0, 11.0],
            "close": [10.0, 11.0, 11.4],
            "volume": [1_000_000.0, 500_000.0, 1_000_000.0],
        }
    )

    panel = goujian_moxing_shuju(
        {"600001.SH": history},
        {"600001.SH": "样本股份"},
        [1],
    )
    signal = panel[panel["trade_date"] == pd.Timestamp("2025-01-02")].iloc[0]

    assert bool(signal["entry_blocked_limit_up_t1"]) is True
    assert pd.isna(signal["entry_open_t1"])
    assert pd.isna(signal["target_t1"])


def test_position_sizing_obeys_board_buy_units_and_budget() -> None:
    main = _position_for_budget("600001.SH", 15.0, 20_000.0)
    star = _position_for_budget("688001.SH", 51.0, 20_000.0)
    unaffordable = _position_for_budget("688002.SH", 120.0, 20_000.0)

    assert main["estimated_buy_shares"] == 1_300
    assert main["buy_share_increment"] == 100
    assert star["estimated_buy_shares"] == 392
    assert star["minimum_buy_shares"] == 200
    assert star["buy_share_increment"] == 1
    assert unaffordable["estimated_buy_shares"] == 0
    assert unaffordable["execution_feasible"] is False


def test_selection_batches_are_capped_ranked_and_do_not_repeat() -> None:
    eligible = [{"ts_code": f"600{index:03d}.SH"} for index in range(1, 19)]

    first, first_next, first_has_more = _paginate_candidates(eligible, offset=0, batch_size=8)
    second, second_next, second_has_more = _paginate_candidates(eligible, offset=first_next, batch_size=8)
    last, last_next, last_has_more = _paginate_candidates(eligible, offset=second_next, batch_size=8)

    assert [item["candidate_rank"] for item in first] == list(range(1, 9))
    assert [item["candidate_rank"] for item in second] == list(range(9, 17))
    assert [item["ts_code"] for item in first] == [item["ts_code"] for item in eligible[:8]]
    assert not ({item["ts_code"] for item in first} & {item["ts_code"] for item in second})
    assert (first_next, first_has_more) == (8, True)
    assert (second_next, second_has_more) == (16, True)
    assert (last_next, last_has_more) == (18, False)
    assert [item["candidate_rank"] for item in last] == [17, 18]


def test_selection_offset_requires_stable_sequence_id() -> None:
    result = bankuai_yuce.bankuai_xuangu(bankuai="白酒", offset=8)

    assert result["status"] == "error"
    assert result["error_code"] == "selection_id_required"


def test_ambiguous_board_name_is_not_silently_selected() -> None:
    with pytest.raises(ValueError, match="存在歧义"):
        _best_name("新能源", ["新能源车", "新能源电池"])


def test_auto_history_source_is_decided_per_stock_not_by_fetch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    frame = _price_frame(["2025-01-02", "2025-01-03"], [10, 10], [10, 10])

    def fetch(_code: str, *, source: str, **_kwargs):
        calls.append(source)
        return SimpleNamespace(
            data=frame,
            source="akshare" if len(calls) == 1 else "tushare",
            adjustment="qfq",
            warnings=("adj_factor unavailable",) if len(calls) == 1 else (),
            errors=(),
        )

    monkeypatch.setattr(bankuai_yuce, "huoqu_rili_xingqing", fetch)
    constituents = pd.DataFrame(
        [
            {"ts_code": "600001.SH", "name": "样本一"},
            {"ts_code": "600002.SH", "name": "样本二"},
        ]
    )

    histories, _, _, _ = _fetch_histories(
        constituents,
        source="auto",
        history_calendar_days=30,
        minimum_rows=1,
        max_stocks=2,
        pause_seconds=0,
    )

    assert calls == ["auto", "auto"]
    assert set(histories) == {"600001.SH", "600002.SH"}


def test_board_model_replaces_unadjusted_auto_history_for_only_that_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    frame = _price_frame(["2025-01-02", "2025-01-03"], [10, 10], [10, 10])

    def fetch(code: str, *, source: str, **_kwargs):
        calls.append((code, source))
        raw = code == "600001.SH" and source == "auto"
        return SimpleNamespace(
            data=frame,
            source="tushare" if source == "auto" else "akshare",
            adjustment="raw_unadjusted" if raw else "qfq",
            warnings=("adj_factor unavailable",) if raw else (),
            errors=(),
        )

    monkeypatch.setattr(bankuai_yuce, "huoqu_rili_xingqing", fetch)
    constituents = pd.DataFrame(
        [
            {"ts_code": "600001.SH", "name": "样本一"},
            {"ts_code": "600002.SH", "name": "样本二"},
        ]
    )

    histories, _, errors, warnings = _fetch_histories(
        constituents,
        source="auto",
        history_calendar_days=30,
        minimum_rows=1,
        max_stocks=2,
        pause_seconds=0,
    )

    assert calls == [("600001.SH", "auto"), ("600001.SH", "akshare"), ("600002.SH", "auto")]
    assert not errors
    assert histories["600001.SH"]["adjustment"].iloc[-1] == "qfq"
    assert any("单独改用 AKShare" in warning for warning in warnings)


def test_only_validated_horizons_drive_weighted_return_and_price_is_bounded() -> None:
    config, _ = jiazai_lianghua_peizhi()
    predictions = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "样本股份",
                "trade_date": pd.Timestamp("2025-01-02"),
                "close": 10.0,
                "pred_t1": 1.0,
                "pred_t2": 0.02,
                "pred_t3": 0.80,
            }
        ]
    )
    constituents = pd.DataFrame([{"ts_code": "600001.SH", "name": "样本股份"}])
    validation = {
        "horizons": {
            "T+1": {"validation_passed": False, "quality_label": "low"},
            "T+2": {"validation_passed": True, "quality_label": "medium"},
            "T+3": {"validation_passed": False, "quality_label": "low"},
        }
    }

    row = _prediction_rows(predictions, constituents, validation, config, 0.0)[0]

    assert row["ranking_horizons"] == ["T+2"]
    assert row["strongest_forecast_horizon"] == "T+2"
    assert row["strongest_horizon_trading_days"] == 2
    assert row["strongest_horizon_validation_passed"] is True
    assert row["weighted_expected_net_return"] == row["forecast"]["T+2"]["estimated_net_return_after_cost"]
    assert row["forecast"]["T+1"]["used_for_ranking"] is False
    assert row["forecast"]["T+1"]["predicted_close"] is None
    assert row["forecast"]["T+1"]["model_reference_exit_price_unconstrained"] == 20.0
    assert row["forecast"]["T+1"]["model_reference_exit_price_clipped_to_legal_range"] == 12.1
    assert row["forecast"]["T+1"]["price_limit_sessions_from_signal_close"] == 2
    assert "实际T+1开盘价尚未知" in row["forecast"]["T+1"]["predicted_close_unavailable_reason"]


def test_validated_zero_weight_horizon_does_not_silently_become_equal_weight() -> None:
    config, _ = jiazai_lianghua_peizhi()
    config = copy.deepcopy(config)
    config["moxing"]["horizon_weights"] = {"1": 0.0, "2": 1.0, "3": 0.0}
    predictions = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "样本股份",
                "trade_date": pd.Timestamp("2025-01-02"),
                "close": 10.0,
                "pred_t1": 0.05,
                "pred_t2": 0.02,
                "pred_t3": 0.01,
            }
        ]
    )
    validation = {
        "horizons": {
            "T+1": {"validation_passed": True, "quality_label": "medium"},
            "T+2": {"validation_passed": False, "quality_label": "low"},
            "T+3": {"validation_passed": False, "quality_label": "low"},
        }
    }

    row = _prediction_rows(
        predictions,
        pd.DataFrame([{"ts_code": "600001.SH", "name": "样本股份"}]),
        validation,
        config,
        0.0,
    )[0]

    assert row["ranking_horizons"] == []
    assert row["weighted_expected_net_return"] is None
    assert row["strongest_forecast_horizon"] is None
    assert row["strongest_horizon_trading_days"] is None
    assert row["strongest_horizon_validation_passed"] is False


def test_validation_reports_top_n_cost_metrics_and_refits_all_labeled_rows() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=105)
    rows: list[dict[str, object]] = []
    for code_index in range(5):
        for date_index, date in enumerate(dates):
            row: dict[str, object] = {
                "ts_code": f"6000{code_index + 1:02d}.SH",
                "name": f"样本{code_index}",
                "trade_date": date,
                "close": 10.0 + code_index,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(rng.normal())
            signal = float(row[FEATURE_COLUMNS[0]])
            for horizon in [1, 2, 3]:
                row[f"entry_date_t{horizon}"] = date + pd.offsets.BDay(1)
                row[f"entry_open_t{horizon}"] = 10.0 + code_index
                row[f"target_date_t{horizon}"] = date + pd.offsets.BDay(horizon + 1)
                row[f"target_t{horizon}"] = 0.002 * signal + float(rng.normal(0, 0.005))
            rows.append(row)
    panel = pd.DataFrame(rows)
    latest = panel.sort_values("trade_date").groupby("ts_code", as_index=False).tail(1)
    config, _ = jiazai_lianghua_peizhi()
    config = copy.deepcopy(config)
    config["moxing"].update(
        {
            "min_training_samples": 100,
            "min_validation_samples": 50,
            "max_iter": 8,
            "min_samples_leaf": 10,
        }
    )

    predictions, validation = xunlian_yuce_moxing(panel, latest, config)

    assert {"pred_t1", "pred_t2", "pred_t3"}.issubset(predictions.columns)
    for metrics in validation["horizons"].values():
        assert metrics["top_n_days"] >= 10
        assert "top_n_mean_net_return" in metrics
        assert "top_n_mean_excess_vs_universe" in metrics
        assert metrics["final_train_samples"] > metrics["train_samples"]
        assert metrics["retrained_on_all_labeled_data"] is True
