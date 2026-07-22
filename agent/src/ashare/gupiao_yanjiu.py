"""Shared A-share data, technical indicators, and single-stock research."""

from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ashare.shuju_yuan import (
    STOCK_BASIC_CACHE,
    STOCK_BASIC_CACHE_TTL,
    _load_or_fetch_stock_basic,
    _price_limit_rule,
    _tushare_pro,
)
from src.ashare.shuju_zhiliang import build_data_health, classify_failure
from src.providers.llm import _ensure_dotenv
from src.tools.path_utils import safe_run_dir

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "lianghua_peizhi.json"
AK_STOCK_NAMES_CACHE = STOCK_BASIC_CACHE.parent / "akshare_stock_names.csv"
AK_STOCK_NAMES_CACHE_TTL_SECONDS = 24 * 60 * 60
DAILY_BAR_CACHE_DIR = STOCK_BASIC_CACHE.parent / "daily_bar_cache"
DAILY_BAR_CACHE_TTL_SECONDS = 12 * 60 * 60
MARKET_DATA_STALE_WARNING_BUSINESS_DAYS = 2
MARKET_DATA_STALE_ERROR_BUSINESS_DAYS = 7
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 5
SINGLE_STOCK_TOOL_CONTRACT_VERSION = 4

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "ma_gap_60",
    "ma_trend_5_20",
    "rsi_14",
    "macd_dif_pct",
    "macd_hist_pct",
    "atr_14_pct",
    "volatility_20",
    "drawdown_20",
    "position_20",
    "volume_ratio_5_20",
    "amplitude_1",
]
FINANCIAL_CRITICAL_FIELDS = ("roe_pct", "net_profit_yoy_pct", "debt_to_assets_pct")

# Kept for compatibility with callers that imported the old module global.  A
# failed adj_factor request is now isolated to that request and never disables
# adjustment for later stocks.
_ADJ_FACTOR_DISABLED_REASON = ""
_PROXY_ENV_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@dataclass(frozen=True)
class XingqingJieguo:
    data: pd.DataFrame
    source: str
    adjustment: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def jiazai_lianghua_peizhi(config_path: str | None = None) -> tuple[dict[str, Any], str]:
    """Load and validate the external quant configuration."""
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"量化配置文件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("量化配置必须是 JSON 对象")

    def finite_number(section: dict[str, Any], key: str, default: float, label: str) -> float:
        try:
            number = float(section.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{key} 必须是数值") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}.{key} 必须是有限数值")
        return number

    data_settings = value.get("shuju", {})
    if not isinstance(data_settings, dict):
        raise ValueError("shuju 必须是 JSON 对象")
    data_history_days = int(finite_number(data_settings, "history_calendar_days", 540, "shuju"))
    max_board_stocks = int(finite_number(data_settings, "max_board_stocks", 20, "shuju"))
    warehouse_max_board_stocks = int(
        finite_number(data_settings, "warehouse_max_board_stocks", 80, "shuju")
    )
    minimum_history_rows = int(finite_number(data_settings, "minimum_history_rows", 120, "shuju"))
    data_pause = finite_number(data_settings, "request_pause_seconds", 0.15, "shuju")
    if not 180 <= data_history_days <= 3650:
        raise ValueError("shuju.history_calendar_days 必须在 180 到 3650 之间")
    if not 1 <= max_board_stocks <= 100:
        raise ValueError("shuju.max_board_stocks 必须在 1 到 100 之间")
    if not max_board_stocks <= warehouse_max_board_stocks <= 200:
        raise ValueError("shuju.warehouse_max_board_stocks 必须不小于普通上限且不大于200")
    if not 60 <= minimum_history_rows <= 2000:
        raise ValueError("shuju.minimum_history_rows 必须在 60 到 2000 之间")
    if not 0 <= data_pause <= 10:
        raise ValueError("shuju.request_pause_seconds 必须在 0 到 10 之间")
    if not isinstance(data_settings.get("akshare_bypass_proxy", True), bool):
        raise ValueError("shuju.akshare_bypass_proxy 必须是 true 或 false")
    if data_settings.get("frequency", "daily_only") != "daily_only":
        raise ValueError("shuju.frequency 必须为 daily_only；本产品永久只做日K")
    if data_settings.get("minute_bars_enabled", False) is not False:
        raise ValueError("shuju.minute_bars_enabled 必须为 false；分钟K不属于产品范围")

    filters = value.get("guolv", {})
    if not isinstance(filters, dict):
        raise ValueError("guolv 必须是 JSON 对象")
    minimum_price = finite_number(filters, "min_price", 2.0, "guolv")
    maximum_price = finite_number(filters, "max_price", 300.0, "guolv")
    minimum_amount = finite_number(filters, "min_amount_yuan", 50_000_000, "guolv")
    if minimum_price <= 0 or maximum_price <= minimum_price:
        raise ValueError("guolv 价格范围必须满足 0 < min_price < max_price")
    if minimum_amount < 0:
        raise ValueError("guolv.min_amount_yuan 不能小于 0")
    keywords = filters.get("exclude_name_keywords", [])
    if not isinstance(keywords, list) or any(not isinstance(item, str) for item in keywords):
        raise ValueError("guolv.exclude_name_keywords 必须是字符串数组")
    if not isinstance(filters.get("exclude_latest_limit_up", True), bool):
        raise ValueError("guolv.exclude_latest_limit_up 必须是 true 或 false")

    model = value.get("moxing", {})
    if not isinstance(model, dict):
        raise ValueError("moxing 必须是 JSON 对象")
    horizons = model.get("horizons")
    if horizons != [1, 2, 3]:
        raise ValueError("moxing.horizons 必须严格为 [1, 2, 3]")
    try:
        validation_ratio = float(model.get("validation_ratio"))
        clip_quantiles = [float(item) for item in model.get("prediction_clip_quantiles", [])]
        weights = {int(key): float(item) for key, item in model.get("horizon_weights", {}).items()}
        integer_defaults = {
            "min_training_samples": 500,
            "min_validation_samples": 100,
            "min_rank_ic_days": 10,
            "validation_top_n": 3,
            "min_top_n_days": 10,
            "ensemble_min_calibration_samples": 80,
            "ensemble_min_calibration_dates": 20,
            "factor_stability_slices": 3,
            "factor_min_valid_slices": 2,
            "factor_min_features": 12,
            "direction_logistic_max_iter": 500,
            "probability_calibration_min_samples": 120,
            "conformal_min_samples": 80,
            "ranking_relevance_grades": 5,
            "ranking_pair_top_k": 8,
            "ranking_n_estimators": 180,
        }
        positive_integer_fields = {
            key: int(model.get(key, default)) for key, default in integer_defaults.items()
        }
        min_direction = float(model.get("min_direction_accuracy", 0.52))
        min_rank_ic = float(model.get("min_mean_daily_rank_ic", 0.01))
        min_skill = float(model.get("min_skill_vs_baseline", 0.01))
        min_best_naive_skill = float(model.get("min_skill_vs_best_naive_baseline", 0.0))
        abstain_min_net_return = float(model.get("abstain_min_net_return", 0.003))
        abstain_min_probability = float(model.get("abstain_min_positive_probability", 0.55))
        abstain_min_quality = float(model.get("abstain_min_quality_score", 0.40))
        ridge_alpha = float(model.get("ridge_alpha", 10.0))
        ensemble_default_tree_weight = float(model.get("ensemble_default_tree_weight", 0.75))
        ensemble_calibration_ratio = float(model.get("ensemble_calibration_ratio", 0.15))
        ensemble_weight_grid = [
            float(item)
            for item in model.get("ensemble_tree_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
        ]
        feature_winsor_quantiles = [
            float(item) for item in model.get("feature_winsor_quantiles", [0.01, 0.99])
        ]
        model_feature_coverage = float(model.get("min_feature_coverage", 0.20))
        factor_min_sign_agreement = float(model.get("factor_min_sign_agreement", 0.67))
        factor_min_abs_ic = float(model.get("factor_min_abs_mean_rank_ic", 0.005))
        direction_logistic_c = float(model.get("direction_logistic_c", 0.5))
        calibration_evaluation_ratio = float(model.get("probability_calibration_evaluation_ratio", 0.30))
        calibration_min_improvement = float(model.get("probability_calibration_min_brier_improvement", 0.0005))
        conformal_coverage = float(model.get("conformal_coverage", 0.80))
        ranking_min_ndcg_improvement = float(model.get("ranking_min_ndcg_improvement", 0.0))
        return_interval_coverage_range = [
            float(item) for item in model.get("return_interval_coverage_range", [0.75, 0.85])
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"moxing 数值配置无效：{exc}") from exc
    if not 0.05 <= validation_ratio <= 0.4:
        raise ValueError("moxing.validation_ratio 必须在 0.05 到 0.4 之间")
    if len(clip_quantiles) != 2 or not 0 <= clip_quantiles[0] < clip_quantiles[1] <= 1:
        raise ValueError("moxing.prediction_clip_quantiles 必须是两个递增的 0~1 数值")
    if (
        set(weights) != {1, 2, 3}
        or any(not math.isfinite(item) or item < 0 for item in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("moxing.horizon_weights 必须为 T+1/T+2/T+3 提供非负权重且总和大于0")
    if any(item <= 0 for item in positive_integer_fields.values()):
        raise ValueError("moxing 的样本数、验证天数和 Top-N 配置必须为正整数")
    if not 2 <= positive_integer_fields["ranking_relevance_grades"] <= 31:
        raise ValueError("moxing.ranking_relevance_grades 必须在 2 到 31 之间")
    if positive_integer_fields["ranking_pair_top_k"] > 100:
        raise ValueError("moxing.ranking_pair_top_k 不能大于 100")
    if positive_integer_fields["factor_min_valid_slices"] > positive_integer_fields["factor_stability_slices"]:
        raise ValueError("moxing.factor_min_valid_slices 不能大于 factor_stability_slices")
    if not 0.5 <= min_direction <= 1:
        raise ValueError("moxing.min_direction_accuracy 必须在 0.5 到 1 之间")
    if not -1 <= min_rank_ic <= 1 or not -1 <= min_skill <= 1 or not -1 <= min_best_naive_skill <= 1:
        raise ValueError("moxing 的 Rank IC 和基线提升门槛必须在 -1 到 1 之间")
    if not 0 <= abstain_min_net_return <= 0.2:
        raise ValueError("moxing.abstain_min_net_return 必须在 0 到 0.2 之间")
    if not 0.5 <= abstain_min_probability <= 1:
        raise ValueError("moxing.abstain_min_positive_probability 必须在 0.5 到 1 之间")
    if not 0 <= abstain_min_quality <= 1:
        raise ValueError("moxing.abstain_min_quality_score 必须在 0 到 1 之间")
    if not isinstance(model.get("ensemble_enabled", True), bool):
        raise ValueError("moxing.ensemble_enabled 必须是 true 或 false")
    if not isinstance(model.get("factor_stability_enabled", True), bool):
        raise ValueError("moxing.factor_stability_enabled 必须是 true 或 false")
    if not isinstance(model.get("ranking_enabled", True), bool):
        raise ValueError("moxing.ranking_enabled 必须是 true 或 false")
    if not 0 < model_feature_coverage <= 1:
        raise ValueError("moxing.min_feature_coverage 必须在 0 到 1 之间")
    if not 0.5 <= factor_min_sign_agreement <= 1:
        raise ValueError("moxing.factor_min_sign_agreement 必须在 0.5 到 1 之间")
    if not 0 <= factor_min_abs_ic <= 1:
        raise ValueError("moxing.factor_min_abs_mean_rank_ic 必须在 0 到 1 之间")
    if not math.isfinite(direction_logistic_c) or direction_logistic_c <= 0:
        raise ValueError("moxing.direction_logistic_c 必须是正有限数")
    if not 0.1 <= calibration_evaluation_ratio <= 0.5:
        raise ValueError("moxing.probability_calibration_evaluation_ratio 必须在 0.1 到 0.5 之间")
    if not 0 <= calibration_min_improvement <= 0.2:
        raise ValueError("moxing.probability_calibration_min_brier_improvement 必须在 0 到 0.2 之间")
    if not 0.5 < conformal_coverage < 1:
        raise ValueError("moxing.conformal_coverage 必须在 0.5 到 1 之间")
    if not -1 <= ranking_min_ndcg_improvement <= 1:
        raise ValueError("moxing.ranking_min_ndcg_improvement 必须在 -1 到 1 之间")
    if (
        len(return_interval_coverage_range) != 2
        or not 0 < return_interval_coverage_range[0] < return_interval_coverage_range[1] < 1
    ):
        raise ValueError("moxing.return_interval_coverage_range 必须是两个递增的 0~1 数值")
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("moxing.ridge_alpha 必须是正有限数")
    if not 0 <= ensemble_default_tree_weight <= 1:
        raise ValueError("moxing.ensemble_default_tree_weight 必须在 0 到 1 之间")
    if not 0.05 <= ensemble_calibration_ratio <= 0.4:
        raise ValueError("moxing.ensemble_calibration_ratio 必须在 0.05 到 0.4 之间")
    if (
        not ensemble_weight_grid
        or any(not math.isfinite(item) or not 0 <= item <= 1 for item in ensemble_weight_grid)
    ):
        raise ValueError("moxing.ensemble_tree_weight_grid 必须是非空的 0 到 1 数值数组")
    if (
        len(feature_winsor_quantiles) != 2
        or not 0 <= feature_winsor_quantiles[0] < feature_winsor_quantiles[1] <= 1
    ):
        raise ValueError("moxing.feature_winsor_quantiles 必须是两个递增的 0 到 1 数值")
    learning_rate = finite_number(model, "learning_rate", 0.05, "moxing")
    l2_regularization = finite_number(model, "l2_regularization", 1.0, "moxing")
    model_integer_fields = {
        "max_iter": int(finite_number(model, "max_iter", 180, "moxing")),
        "max_leaf_nodes": int(finite_number(model, "max_leaf_nodes", 15, "moxing")),
        "max_depth": int(finite_number(model, "max_depth", 4, "moxing")),
        "min_samples_leaf": int(finite_number(model, "min_samples_leaf", 30, "moxing")),
    }
    try:
        int(model.get("random_state", 42))
    except (TypeError, ValueError) as exc:
        raise ValueError("moxing.random_state 必须是整数") from exc
    if not 0 < learning_rate <= 1:
        raise ValueError("moxing.learning_rate 必须在 0 到 1 之间")
    if l2_regularization < 0:
        raise ValueError("moxing.l2_regularization 不能小于 0")
    if any(item <= 0 for item in model_integer_fields.values()):
        raise ValueError("moxing 的迭代次数、树规模、深度和叶节点样本数必须为正整数")
    max_holding_days = int(value.get("jiaoyi", {}).get("max_holding_days", 0))
    if max_holding_days != 3:
        raise ValueError("jiaoyi.max_holding_days 必须为 3")
    trading = value.get("jiaoyi", {})
    if (
        trading.get("execution_mode") != "research_only"
        or trading.get("allow_live_trading") is not False
        or trading.get("allow_order_submission") is not False
    ):
        raise ValueError("本程序被硬限制为 research_only，禁止实盘交易和订单提交")
    if not isinstance(trading.get("dynamic_slippage_enabled", True), bool):
        raise ValueError("jiaoyi.dynamic_slippage_enabled 必须是 true 或 false")
    try:
        maximum_dynamic_slippage = float(trading.get("max_dynamic_slippage_bps_roundtrip", 40.0))
        maximum_participation = float(trading.get("max_participation_rate", 0.005))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"jiaoyi 动态成交成本配置无效：{exc}") from exc
    if not 0 <= maximum_dynamic_slippage <= 500:
        raise ValueError("jiaoyi.max_dynamic_slippage_bps_roundtrip 必须在 0 到 500 之间")
    if not 0 < maximum_participation <= 0.1:
        raise ValueError("jiaoyi.max_participation_rate 必须在 0 到 0.1 之间")
    single = value.get("dangu", {})
    if not isinstance(single, dict):
        raise ValueError("dangu 必须是 JSON 对象")
    try:
        single_history_days = int(single.get("history_calendar_days", 1440))
        maximum_peers = int(single.get("max_peer_stocks", 20))
        same_industry_peers = int(single.get("same_industry_stocks", 16))
        warehouse_maximum_peers = int(single.get("warehouse_max_peer_stocks", 60))
        warehouse_same_industry_peers = int(single.get("warehouse_same_industry_stocks", 45))
        minimum_peers = int(single.get("minimum_peer_stocks", 8))
        walk_forward_folds = int(single.get("walk_forward_folds", 3))
        minimum_passed_folds = int(single.get("min_passed_folds", 2))
        minimum_feature_coverage = float(single.get("min_feature_coverage", 0.2))
        minimum_net_return = float(single.get("assessment_min_net_return", 0.003))
        minimum_probability = float(single.get("assessment_min_positive_probability", 0.55))
        single_minimum_history_rows = int(single.get("minimum_history_rows", 180))
        minimum_listing_days = int(single.get("min_listing_calendar_days", 180))
        single_minimum_amount = float(single.get("min_amount_yuan", 30_000_000))
        single_pause = float(single.get("request_pause_seconds", 0.08))
        validation_window_days = int(single.get("validation_window_days", 45))
        minimum_training_dates = int(single.get("minimum_training_dates", 120))
        minimum_fold_training = int(single.get("min_fold_training_samples", 500))
        minimum_fold_validation = int(single.get("min_fold_validation_samples", 80))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dangu 数值配置无效：{exc}") from exc
    if not 540 <= single_history_days <= 1800:
        raise ValueError("dangu.history_calendar_days 必须在 540 到 1800 之间")
    if not 8 <= maximum_peers <= 40:
        raise ValueError("dangu.max_peer_stocks 必须在 8 到 40 之间")
    if not 4 <= same_industry_peers < maximum_peers:
        raise ValueError("dangu.same_industry_stocks 必须至少为4且小于 max_peer_stocks")
    if not maximum_peers <= warehouse_maximum_peers <= 200:
        raise ValueError("dangu.warehouse_max_peer_stocks 必须不小于普通上限且不大于200")
    if not same_industry_peers <= warehouse_same_industry_peers < warehouse_maximum_peers:
        raise ValueError(
            "dangu.warehouse_same_industry_stocks 必须不小于普通同行数且小于仓库同行上限"
        )
    if not 5 <= minimum_peers <= maximum_peers:
        raise ValueError("dangu.minimum_peer_stocks 必须在5到 max_peer_stocks 之间")
    if walk_forward_folds < 2 or not 1 <= minimum_passed_folds <= walk_forward_folds:
        raise ValueError("dangu 的滚动验证折数或最少通过折数无效")
    if not 0 < minimum_feature_coverage <= 1:
        raise ValueError("dangu.min_feature_coverage 必须在0到1之间")
    if not 0 <= minimum_net_return <= 0.2 or not 0.5 <= minimum_probability <= 1:
        raise ValueError("dangu 的证据评估收益或上涨比例门槛无效")
    if single_minimum_history_rows < 60:
        raise ValueError("dangu.minimum_history_rows 必须至少为 60")
    if minimum_listing_days < 0 or single_minimum_amount < 0:
        raise ValueError("dangu 的最少上市天数和最低成交额不能小于 0")
    if not 0 <= single_pause <= 10:
        raise ValueError("dangu.request_pause_seconds 必须在 0 到 10 之间")
    if validation_window_days < 20 or minimum_training_dates < 80:
        raise ValueError("dangu 的验证窗口至少为 20 日，训练日期至少为 80 日")
    if minimum_fold_training <= 0 or minimum_fold_validation <= 0:
        raise ValueError("dangu 的每折训练和验证样本数必须为正整数")
    if not str(trading.get("cost_scenario", "")).strip():
        raise ValueError("jiaoyi.cost_scenario 不能为空")
    return value, str(path)


def _akshare_bypass_proxy_enabled() -> bool:
    override = os.getenv("GPYJ_AKSHARE_BYPASS_PROXY", "").strip().lower()
    if override:
        return override not in {"0", "false", "no", "off"}
    try:
        config, _ = jiazai_lianghua_peizhi()
        return bool(config.get("shuju", {}).get("akshare_bypass_proxy", True))
    except Exception:
        return True


@contextmanager
def akshare_zhilian():
    """Temporarily bypass proxy variables for mainland AKShare endpoints."""
    if not _akshare_bypass_proxy_enabled():
        yield
        return
    saved = {name: os.environ[name] for name in _PROXY_ENV_NAMES if name in os.environ}
    try:
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        yield
    finally:
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(saved)


def _digits_from_symbol(value: str) -> str:
    raw = str(value).strip().upper()
    raw = re.sub(r"^(SH|SZ|BJ)", "", raw)
    raw = re.sub(r"\.(SH|SZ|BJ)$", "", raw)
    return raw


def biaozhunhua_daima(value: str) -> str:
    """Normalize a mainland A-share stock code to Tushare format."""
    raw = str(value).strip().upper()
    digits = _digits_from_symbol(raw)
    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"不是有效的 6 位 A 股代码：{value}")

    suffix = ""
    match = re.search(r"\.(SH|SZ|BJ)$", raw)
    if match:
        suffix = match.group(1)
    elif raw.startswith(("SH", "SZ", "BJ")):
        suffix = raw[:2]

    expected = ""
    if digits.startswith(("600", "601", "603", "605", "688", "689")):
        expected = "SH"
    elif digits.startswith(("000", "001", "002", "003", "300", "301")):
        expected = "SZ"
    elif digits.startswith(("43", "83", "87", "88", "920")):
        expected = "BJ"
    if not expected:
        raise ValueError(f"代码不属于本系统支持的 A 股普通股票范围：{value}")
    if suffix and suffix != expected:
        raise ValueError(f"代码与交易所后缀不一致：{value}，应为 .{expected}")
    return f"{digits}.{expected}"


def shi_a_gu(value: str) -> bool:
    try:
        biaozhunhua_daima(value)
        return True
    except ValueError:
        return False


def _stock_basic_cache() -> pd.DataFrame:
    if not STOCK_BASIC_CACHE.is_file():
        return pd.DataFrame()
    try:
        cache_age = max(0.0, time.time() - STOCK_BASIC_CACHE.stat().st_mtime)
        if cache_age > STOCK_BASIC_CACHE_TTL.total_seconds():
            return pd.DataFrame()
        return pd.read_csv(STOCK_BASIC_CACHE, dtype=str)
    except Exception:
        return pd.DataFrame()


def _akshare_name_table() -> pd.DataFrame:
    stale_cache = pd.DataFrame()
    if AK_STOCK_NAMES_CACHE.is_file():
        try:
            cached = pd.read_csv(AK_STOCK_NAMES_CACHE, dtype=str)
            if not cached.empty and {"ts_code", "name"}.issubset(cached.columns):
                stale_cache = cached
                cache_age = max(0.0, time.time() - AK_STOCK_NAMES_CACHE.stat().st_mtime)
                if cache_age <= AK_STOCK_NAMES_CACHE_TTL_SECONDS:
                    return cached
        except Exception:
            pass
    try:
        import akshare as ak

        with akshare_zhilian():
            table = ak.stock_info_a_code_name().rename(columns={"code": "ts_code", "name": "name"})
    except Exception:
        if not stale_cache.empty:
            return stale_cache
        raise
    table = table[["ts_code", "name"]].copy()
    table["ts_code"] = table["ts_code"].astype(str).str.zfill(6)
    table = table[table["ts_code"].map(shi_a_gu)].copy()
    table["ts_code"] = table["ts_code"].map(biaozhunhua_daima)
    AK_STOCK_NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(AK_STOCK_NAMES_CACHE, index=False, encoding="utf-8-sig")
    return table


def _match_stock_basic(table: pd.DataFrame, query: str) -> dict[str, Any] | None:
    if table is None or table.empty:
        return None
    frame = table.copy()
    if "ts_code" in frame.columns:
        frame["ts_code"] = frame["ts_code"].astype(str).map(
            lambda item: biaozhunhua_daima(item) if shi_a_gu(item) else item
        )
    raw = str(query).strip()
    if shi_a_gu(raw):
        code = biaozhunhua_daima(raw)
        hits = frame[frame.get("ts_code", pd.Series(dtype=str)) == code]
    else:
        if "name" not in frame.columns:
            return None
        names = frame["name"].fillna("").astype(str)
        hits = frame[names == raw]
        if hits.empty:
            hits = frame[names.str.contains(re.escape(raw), regex=True)]
            distinct_hits = hits.drop_duplicates(subset=["ts_code"] if "ts_code" in hits.columns else ["name"])
            if len(distinct_hits) > 1:
                candidates = []
                for _, candidate in distinct_hits.head(8).iterrows():
                    label = str(candidate.get("name") or "未知名称")
                    candidate_code = str(candidate.get("ts_code") or "未知代码")
                    candidates.append(f"{label}（{candidate_code}）")
                suffix = "等" if len(distinct_hits) > len(candidates) else ""
                raise ValueError(
                    f"股票名称“{raw}”匹配到多个候选：{'、'.join(candidates)}{suffix}；"
                    "请使用完整股票名称或 6 位股票代码"
                )
    if hits.empty:
        return None
    row = hits.iloc[0]
    return {str(key): _json_value(value) for key, value in row.items()}


def jiexi_gupiao(gupiao: str, *, source: str = "auto") -> tuple[str, dict[str, Any], list[str]]:
    """Resolve either a stock code or a Chinese stock name."""
    source = source.strip().lower()
    if source not in {"auto", "tushare", "akshare"}:
        raise ValueError("source 必须是 auto、tushare 或 akshare")
    warnings: list[str] = []
    raw = str(gupiao).strip()
    if shi_a_gu(raw):
        code = biaozhunhua_daima(raw)
        resolved: dict[str, Any] = {}
        if source in {"auto", "tushare"}:
            resolved = _match_stock_basic(_stock_basic_cache(), code) or {}
        if not resolved.get("name") and source in {"auto", "tushare"}:
            try:
                resolved = _match_stock_basic(_load_or_fetch_stock_basic(_tushare_pro(), {}), code) or resolved
            except Exception as exc:
                warnings.append(f"Tushare 股票名称暂不可用：{exc}")
        if not resolved.get("name") and source in {"auto", "akshare"}:
            try:
                resolved = _match_stock_basic(_akshare_name_table(), code) or resolved
                if resolved.get("name"):
                    warnings.append("股票名称来自 AKShare 本地代码表缓存")
            except Exception as exc:
                warnings.append(f"股票名称表暂不可用：{exc}")
        return code, resolved, warnings

    if source in {"auto", "tushare"}:
        cached = _match_stock_basic(_stock_basic_cache(), raw)
        if cached and cached.get("ts_code"):
            return biaozhunhua_daima(str(cached["ts_code"])), cached, warnings

    errors: list[str] = []
    if source in {"auto", "tushare"}:
        try:
            pro = _tushare_pro()
            table = _load_or_fetch_stock_basic(pro, {})
            match = _match_stock_basic(table, raw)
            if match and match.get("ts_code"):
                return biaozhunhua_daima(str(match["ts_code"])), match, warnings
            errors.append(f"Tushare 未找到股票名称：{raw}")
        except ValueError:
            raise
        except Exception as exc:
            errors.append(f"Tushare 名称解析失败：{exc}")
            if source == "tushare":
                raise RuntimeError(errors[-1]) from exc

    if source in {"auto", "akshare"}:
        try:
            table = _akshare_name_table()
            match = _match_stock_basic(table, raw)
            if match and match.get("ts_code"):
                if errors:
                    warnings.extend(errors)
                warnings.append("股票名称由 AKShare 免费接口解析")
                return biaozhunhua_daima(str(match["ts_code"])), match, warnings
        except ValueError:
            raise
        except Exception as exc:
            errors.append(f"AKShare 名称解析失败：{exc}")
    raise RuntimeError("；".join(errors) or f"无法识别股票：{gupiao}")


def _normalize_history(frame: pd.DataFrame, *, tushare: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if tushare:
        data = data.rename(columns={"vol": "volume"})
    else:
        data = data.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount_yuan",
                "涨跌幅": "pct_chg",
                "换手率": "turnover_rate",
                "date": "trade_date",
                "turnover": "turnover_rate",
            }
        )
    if "trade_date" not in data.columns:
        raise ValueError("行情缺少 trade_date/日期列")
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount", "amount_yuan", "pct_chg", "turnover_rate"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if tushare and "amount" in data.columns and "amount_yuan" not in data.columns:
        data["amount_yuan"] = data["amount"] * 1000.0
    elif not tushare and "amount" in data.columns and "amount_yuan" not in data.columns:
        data["amount_yuan"] = data["amount"]
    required = ["trade_date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"行情缺少字段：{missing}")
    return (
        data.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .reset_index(drop=True)
    )


def _latest_expected_market_date(reference: datetime | None = None) -> pd.Timestamp:
    """Return the latest weekday whose closing bar should be complete.

    This deliberately uses only a conservative weekday calendar.  Exchange
    holidays can make the returned date later than the real last trading day,
    so stale data is warned early but rejected only after a wider tolerance.
    """
    current = reference or datetime.now()
    expected = pd.Timestamp(current.date())
    before_close = (current.hour, current.minute) < (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    if expected.weekday() < 5 and before_close:
        expected -= timedelta(days=1)
    while expected.weekday() >= 5:
        expected -= timedelta(days=1)
    return expected.normalize()


def _completed_market_history(
    history: pd.DataFrame,
    *,
    reference: datetime | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Exclude bars that are not guaranteed to represent a completed session."""
    if history is None or history.empty:
        return pd.DataFrame(), []
    expected = _latest_expected_market_date(reference)
    dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    keep = dates <= expected
    dropped = int((~keep).sum())
    warnings: list[str] = []
    if dropped:
        warnings.append(f"已忽略 {dropped} 根尚未确认收盘的日线，技术分析只使用完整交易日")
    return history.loc[keep].copy().reset_index(drop=True), warnings


def _market_data_freshness(as_of: Any, *, reference: datetime | None = None) -> dict[str, Any]:
    """Describe whether the latest completed bar is recent enough for current analysis."""
    latest = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(latest):
        return {
            "expected_latest_date": _latest_expected_market_date(reference).strftime("%Y-%m-%d"),
            "business_days_old": None,
            "status": "invalid_date",
        }
    latest = pd.Timestamp(latest).normalize()
    expected = _latest_expected_market_date(reference)
    if latest >= expected:
        business_days_old = 0
    else:
        business_days_old = int(np.busday_count(latest.date(), expected.date()))
    if business_days_old > MARKET_DATA_STALE_ERROR_BUSINESS_DAYS:
        status = "too_stale"
    elif business_days_old > MARKET_DATA_STALE_WARNING_BUSINESS_DAYS:
        status = "possibly_stale"
    else:
        status = "fresh"
    return {
        "expected_latest_date": expected.strftime("%Y-%m-%d"),
        "business_days_old": business_days_old,
        "status": status,
    }


def _can_use_current_akshare_snapshot(
    as_of: Any,
    *,
    reference: datetime | None = None,
) -> bool:
    """Allow an undated realtime snapshot only when it cannot contain intraday data."""
    as_of_date = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(as_of_date):
        return False
    current = reference or datetime.now()
    before_close = (current.hour, current.minute) < (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    if current.weekday() < 5 and before_close:
        return False
    return pd.Timestamp(as_of_date).normalize() == _latest_expected_market_date(current)


def _apply_qfq(pro: Any, code: str, start_date: str, end_date: str, data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    try:
        factors = pro.adj_factor(ts_code=code, start_date=start_date, end_date=end_date)
        if factors is None or factors.empty:
            return data, "raw_unadjusted", "Tushare adj_factor 返回空值，价格未复权"
        adj = factors[["trade_date", "adj_factor"]].copy()
        adj["trade_date"] = pd.to_datetime(adj["trade_date"], errors="coerce")
        adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        merged = data.merge(adj, on="trade_date", how="left")
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
        latest_factor = float(merged["adj_factor"].iloc[-1])
        if not math.isfinite(latest_factor) or latest_factor <= 0:
            return data, "raw_unadjusted", "Tushare adj_factor 无效，价格未复权"
        ratio = merged["adj_factor"] / latest_factor
        for column in ["open", "high", "low", "close", "pre_close"]:
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce") * ratio
        return merged.drop(columns=["adj_factor"]), "qfq_by_tushare_adj_factor", ""
    except Exception as exc:
        reason = f"Tushare adj_factor 本次请求不可用，使用未复权价格：{exc}"
        return data, "raw_unadjusted", reason


def _daily_bar_cache_path(code: str, source: str) -> Path:
    return DAILY_BAR_CACHE_DIR / f"{code.replace('.', '_')}_{source}.csv"


def _load_daily_bar_cache(
    *,
    code: str,
    source: str,
    start: str,
    end: str,
) -> XingqingJieguo | None:
    path = _daily_bar_cache_path(code, source)
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        return None
    try:
        age_seconds = max(0.0, time.time() - path.stat().st_mtime)
        if age_seconds > DAILY_BAR_CACHE_TTL_SECONDS:
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        requested_start = pd.to_datetime(meta.get("requested_start"), errors="coerce")
        requested_end = pd.to_datetime(meta.get("requested_end"), errors="coerce")
        wanted_start = pd.to_datetime(start, errors="coerce")
        wanted_end = pd.to_datetime(end, errors="coerce")
        if (
            pd.isna(requested_start)
            or pd.isna(requested_end)
            or pd.isna(wanted_start)
            or pd.isna(wanted_end)
            or pd.Timestamp(requested_start) > pd.Timestamp(wanted_start)
            or pd.Timestamp(requested_end) < pd.Timestamp(wanted_end)
        ):
            return None
        data = _normalize_history(pd.read_csv(path), tushare=False)
        data = data[
            (data["trade_date"] >= pd.Timestamp(wanted_start))
            & (data["trade_date"] <= pd.Timestamp(wanted_end))
        ].reset_index(drop=True)
        if data.empty:
            return None
        expected_latest = _latest_expected_market_date()
        if (
            pd.Timestamp(wanted_end).normalize() >= expected_latest
            and pd.Timestamp(data["trade_date"].max()).normalize() < expected_latest
        ):
            return None
        provider = str(meta.get("provider") or source)
        adjustment = str(meta.get("adjustment") or "unknown")
        return XingqingJieguo(
            data=data,
            source=provider,
            adjustment=adjustment,
            warnings=(f"日K使用12小时内的本地缓存（{provider}）",),
            errors=(),
        )
    except Exception:
        return None


def _save_daily_bar_cache(
    *,
    code: str,
    source_policy: str,
    start: str,
    end: str,
    result: XingqingJieguo,
) -> None:
    if result.data.empty or result.adjustment in {"unknown", "raw_unadjusted"}:
        return
    path = _daily_bar_cache_path(code, source_policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(path, index=False, encoding="utf-8-sig")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "provider": result.source,
                "adjustment": result.adjustment,
                "requested_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "requested_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
                "rows": int(len(result.data)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def huoqu_rili_xingqing(
    code: str,
    *,
    start_date: str,
    end_date: str,
    source: str = "auto",
    use_cache: bool = False,
) -> XingqingJieguo:
    """Fetch one stock's daily bars with Tushare-first fallback semantics."""
    normalized = biaozhunhua_daima(code)
    source = source.strip().lower()
    if source not in {"auto", "tushare", "akshare"}:
        raise ValueError("source 必须是 auto、tushare 或 akshare")
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    if use_cache:
        cached = _load_daily_bar_cache(
            code=normalized,
            source=source,
            start=start,
            end=end,
        )
        if cached is not None:
            return cached
        if source in {"auto", "tushare"}:
            try:
                from src.ashare.riping_cangku import load_qfq_history_from_warehouse

                warehouse_data, warehouse_meta = load_qfq_history_from_warehouse(
                    normalized,
                    start_date=start,
                    end_date=end,
                )
                if not warehouse_data.empty and warehouse_meta.get("status") == "ok":
                    return XingqingJieguo(
                        data=_normalize_history(warehouse_data, tushare=False),
                        source=str(warehouse_meta.get("source") or "tushare_daily_warehouse"),
                        adjustment=str(
                            warehouse_meta.get("adjustment") or "qfq_by_warehouse_adj_factor"
                        ),
                        warnings=(
                            f"日K来自全市场本地仓库，区间同步覆盖率{warehouse_meta.get('sync_coverage')}",
                        ),
                        errors=(),
                    )
            except Exception:
                pass

    def finish(result: XingqingJieguo) -> XingqingJieguo:
        if use_cache:
            try:
                _save_daily_bar_cache(
                    code=normalized,
                    source_policy=source,
                    start=start,
                    end=end,
                    result=result,
                )
            except Exception:
                pass
        return result

    errors: list[str] = []
    warnings: list[str] = []
    raw_tushare_fallback: XingqingJieguo | None = None

    if source in {"auto", "tushare"}:
        try:
            pro = _tushare_pro()
            raw = pro.daily(ts_code=normalized, start_date=start, end_date=end)
            data = _normalize_history(raw, tushare=True)
            if data.empty:
                raise RuntimeError("返回空行情")
            data, adjustment, adjustment_warning = _apply_qfq(pro, normalized, start, end, data)
            if adjustment_warning:
                warnings.append(adjustment_warning)
            tushare_result = XingqingJieguo(data, "tushare", adjustment, tuple(warnings), tuple(errors))
            if adjustment == "raw_unadjusted" and source == "auto":
                raw_tushare_fallback = tushare_result
                warnings.append("自动模式要求复权口径，继续尝试 AKShare 前复权行情")
            else:
                return finish(tushare_result)
        except Exception as exc:
            errors.append(f"Tushare 日线失败：{exc}")
            if source == "tushare":
                return finish(XingqingJieguo(pd.DataFrame(), "tushare", "unknown", tuple(warnings), tuple(errors)))

    try:
        import akshare as ak

        with akshare_zhilian():
            digits, exchange = normalized.split(".")
            raw = pd.DataFrame()
            ak_errors: list[str] = []
            if exchange in {"SH", "SZ"}:
                try:
                    raw = ak.stock_zh_a_daily(
                        symbol=f"{exchange.lower()}{digits}",
                        start_date=start,
                        end_date=end,
                        adjust="qfq",
                    )
                except Exception as exc:
                    ak_errors.append(f"新浪前复权日线失败：{exc}")
            if raw is None or raw.empty:
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=digits,
                        period="daily",
                        start_date=start,
                        end_date=end,
                        adjust="qfq",
                    )
                except Exception as exc:
                    ak_errors.append(f"东方财富前复权日线失败：{exc}")
                    raise RuntimeError("；".join(ak_errors)) from exc
        data = _normalize_history(raw, tushare=False)
        if data.empty:
            raise RuntimeError("返回空行情")
        if errors:
            warnings.extend(errors)
        warnings.append(
            "行情已降级到 AKShare 免费聚合接口"
            if source == "auto"
            else "行情使用 AKShare 免费聚合接口"
        )
        return finish(XingqingJieguo(data, "akshare", "qfq", tuple(warnings), tuple(errors)))
    except Exception as exc:
        errors.append(f"AKShare 日线失败：{exc}")
        if raw_tushare_fallback is not None:
            fallback_warnings = list(raw_tushare_fallback.warnings)
            fallback_warnings.append("AKShare 前复权降级失败，只能使用 Tushare 未复权行情")
            return finish(XingqingJieguo(
                raw_tushare_fallback.data,
                raw_tushare_fallback.source,
                raw_tushare_fallback.adjustment,
                tuple(fallback_warnings),
                tuple(errors),
            ))
        return finish(XingqingJieguo(pd.DataFrame(), "akshare", "unknown", tuple(warnings), tuple(errors)))


def jisuan_tezheng_biao(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate leak-free daily technical features used by analysis and ML."""
    data = history.copy().sort_values("trade_date").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    previous_close = close.shift(1)

    for period in [1, 3, 5, 10, 20]:
        data[f"ret_{period}"] = close.pct_change(period, fill_method=None)
    for period in [5, 10, 20, 60]:
        average = close.rolling(period, min_periods=period).mean()
        data[f"ma_{period}"] = average
        data[f"ma_gap_{period}"] = close / average - 1.0
    data["ma_trend_5_20"] = data["ma_5"] / data["ma_20"] - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    both_flat = gain.eq(0) & loss.eq(0)
    only_gains = gain.gt(0) & loss.eq(0)
    only_losses = gain.eq(0) & loss.gt(0)
    data["rsi_14"] = rsi.mask(both_flat, 50.0).mask(only_gains, 100.0).mask(only_losses, 0.0)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema_12 - ema_26
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    histogram = 2.0 * (dif - dea)
    data["macd_dif"] = dif
    data["macd_dea"] = dea
    data["macd_hist"] = histogram
    data["macd_dif_pct"] = dif / close
    data["macd_hist_pct"] = histogram / close

    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_14_pct"] = data["atr_14"] / close
    daily_return = close.pct_change(fill_method=None)
    data["volatility_20"] = daily_return.rolling(20, min_periods=20).std() * math.sqrt(252)

    rolling_high = high.rolling(20, min_periods=20).max()
    rolling_low = low.rolling(20, min_periods=20).min()
    data["drawdown_20"] = close / rolling_high - 1.0
    spread = (rolling_high - rolling_low).replace(0, np.nan)
    data["position_20"] = ((close - rolling_low) / spread).fillna(0.5)
    data["support_20"] = rolling_low
    data["resistance_20"] = rolling_high

    volume_5 = volume.rolling(5, min_periods=5).mean()
    volume_20 = volume.rolling(20, min_periods=20).mean()
    data["volume_ratio_5_20"] = volume_5 / volume_20.replace(0, np.nan)
    data["amplitude_1"] = (high - low) / previous_close.replace(0, np.nan)
    return data.replace([np.inf, -np.inf], np.nan)


def _round_optional(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _round_optional(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value) if not isinstance(value, (str, int, bool)) else value


def _technical_score(latest: pd.Series) -> tuple[int, list[str]]:
    score = 50.0
    reasons: list[str] = []
    close = _round_optional(latest.get("close"))
    ma_5 = _round_optional(latest.get("ma_5"))
    ma_20 = _round_optional(latest.get("ma_20"))
    if close is not None and ma_20 is not None:
        if ma_5 is not None and close > ma_5 > ma_20:
            score += 14
            reasons.append("收盘价、MA5、MA20 呈多头顺序")
        elif close > ma_20:
            score += 7
            reasons.append("价格位于 MA20 上方")
        else:
            score -= 8
            reasons.append("价格未站上 MA20")

    ret_5 = _round_optional(latest.get("ret_5"))
    if ret_5 is not None:
        if 0.01 <= ret_5 <= 0.12:
            score += 10
            reasons.append("5 日动量为正且未进入极端区")
        elif ret_5 < -0.05:
            score -= 10
            reasons.append("5 日动量明显偏弱")
        elif ret_5 > 0.18:
            score -= 5
            reasons.append("5 日涨幅过快，短线回撤风险增大")

    rsi = _round_optional(latest.get("rsi_14"))
    if rsi is not None:
        if 45 <= rsi <= 70:
            score += 8
            reasons.append("RSI 位于相对健康区间")
        elif rsi >= 80:
            score -= 9
            reasons.append("RSI 进入高位过热区")
        elif rsi <= 30:
            score -= 5
            reasons.append("RSI 显示弱势超卖，不等于已经反转")

    macd_hist = _round_optional(latest.get("macd_hist"))
    if macd_hist is not None:
        if macd_hist > 0:
            score += 7
            reasons.append("MACD 柱为正")
        elif macd_hist < 0:
            score -= 4
            reasons.append("MACD 柱为负")
        else:
            reasons.append("MACD 柱接近零，本项不加减分")
    volatility = _round_optional(latest.get("volatility_20"))
    if volatility is not None and volatility > 0.55:
        score -= 10
        reasons.append("20 日年化波动率偏高")
    return int(round(max(0, min(100, score)))), reasons


def zongjie_jishu(history: pd.DataFrame) -> dict[str, Any]:
    features = jisuan_tezheng_biao(history)
    usable = features.dropna(subset=["ma_20", "rsi_14", "atr_14_pct", "volatility_20"])
    if usable.empty:
        raise RuntimeError("有效日线不足，至少需要约 21 个交易日")
    latest = usable.iloc[-1]
    score, reasons = _technical_score(latest)
    indicator_warnings: list[str] = []
    if _round_optional(latest.get("ma_60")) is None:
        indicator_warnings.append("历史不足 60 个交易日，MA60 暂不可用且未参与评分")
    if _round_optional(latest.get("macd_hist")) is None:
        indicator_warnings.append("历史不足以形成完整 MACD，MACD 暂不可用且未参与评分")
    return {
        "trade_date": _json_value(latest["trade_date"]),
        "close": _round_optional(latest["close"], 3),
        "returns": {f"{period}d": _round_optional(latest[f"ret_{period}"], 6) for period in [1, 3, 5, 10, 20]},
        "moving_averages": {f"ma{period}": _round_optional(latest[f"ma_{period}"], 3) for period in [5, 10, 20, 60]},
        "rsi_14": _round_optional(latest["rsi_14"], 2),
        "macd": {
            "dif": _round_optional(latest["macd_dif"], 4),
            "dea": _round_optional(latest["macd_dea"], 4),
            "histogram": _round_optional(latest["macd_hist"], 4),
        },
        "atr_14_pct": _round_optional(latest["atr_14_pct"], 6),
        "annualized_volatility_20": _round_optional(latest["volatility_20"], 6),
        "drawdown_from_20d_high": _round_optional(latest["drawdown_20"], 6),
        "position_in_20d_range": _round_optional(latest["position_20"], 6),
        "volume_ratio_5_to_20": _round_optional(latest["volume_ratio_5_20"], 4),
        "support_20": _round_optional(latest["support_20"], 3),
        "resistance_20": _round_optional(latest["resistance_20"], 3),
        "score_0_100": score,
        "score_interpretation": (
            "启发式技术状态分，只表示当前指标组合，不是上涨概率、收益预测或精确目标分"
        ),
        "evidence": reasons,
        "indicator_warnings": indicator_warnings,
    }


def _first_number(row: pd.Series | dict[str, Any], aliases: Iterable[str]) -> float | None:
    items = row.items() if hasattr(row, "items") else []
    normalized = [(str(key).lower(), value) for key, value in items]
    for alias in aliases:
        target = alias.lower()
        for key, value in normalized:
            if key == target or target in key:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(number):
                    return float(number)
    return None


def _akshare_info(code: str) -> tuple[dict[str, Any], list[str]]:
    import akshare as ak

    errors: list[str] = []
    result: dict[str, Any] = {}
    digits = code.split(".")[0]
    try:
        with akshare_zhilian():
            table = ak.stock_individual_info_em(symbol=digits)
        if table is not None and not table.empty and {"item", "value"}.issubset(table.columns):
            result.update({str(row["item"]): _json_value(row["value"]) for _, row in table.iterrows()})
    except Exception as exc:
        errors.append(f"AKShare 个股资料失败：{exc}")

    try:
        with akshare_zhilian():
            spot = ak.stock_zh_a_spot_em()
        hit = spot[spot["代码"].astype(str).str.zfill(6) == digits]
        if not hit.empty:
            row = hit.iloc[0]
            result.update(
                {
                    "名称": _json_value(row.get("名称")),
                    "动态市盈率": _json_value(row.get("市盈率-动态")),
                    "市净率": _json_value(row.get("市净率")),
                    "总市值": _json_value(row.get("总市值")),
                    "流通市值": _json_value(row.get("流通市值")),
                    "换手率": _json_value(row.get("换手率")),
                }
            )
    except Exception as exc:
        errors.append(f"AKShare 实时估值失败：{exc}")
    return result, errors


def _financial_missing_fields(financials: dict[str, Any]) -> list[str]:
    return [field for field in FINANCIAL_CRITICAL_FIELDS if _round_optional(financials.get(field)) is None]


def _akshare_financials(code: str, *, as_of: str | None = None) -> tuple[dict[str, Any], list[str]]:
    import akshare as ak

    errors: list[str] = []
    digits = code.split(".")[0]
    try:
        with akshare_zhilian():
            table = ak.stock_financial_analysis_indicator(symbol=digits, start_year=str(datetime.now().year - 4))
        if table is None or table.empty:
            raise RuntimeError("返回空表")
        date_column = next((column for column in ["日期", "报告期", "date"] if column in table.columns), None)
        if not date_column:
            raise RuntimeError("返回结果缺少报告期，无法保证分析时点一致")
        table = table.assign(_date=pd.to_datetime(table[date_column], errors="coerce")).dropna(subset=["_date"])
        if as_of is not None:
            as_of_date = pd.to_datetime(as_of, errors="coerce")
            if pd.isna(as_of_date):
                raise ValueError(f"无效的分析日期：{as_of}")
            table = table[table["_date"].dt.normalize() <= pd.Timestamp(as_of_date).normalize()]
        if table.empty:
            raise RuntimeError(f"截至 {as_of} 没有可用财务报告")
        table = table.sort_values("_date")
        row = table.iloc[-1]
        financials = {
            "report_date": pd.Timestamp(row["_date"]).strftime("%Y-%m-%d"),
            "announcement_date": None,
            "announcement_date_status": "AKShare 未提供公告日，仅在当前分析时点作为降级数据使用",
            "roe_pct": _first_number(row, ["净资产收益率", "加权净资产收益率", "roe"]),
            "gross_margin_pct": _first_number(row, ["销售毛利率", "毛利率", "grossprofit_margin"]),
            "net_margin_pct": _first_number(row, ["销售净利率", "净利率", "netprofit_margin"]),
            "debt_to_assets_pct": _first_number(row, ["资产负债率", "debt_to_assets"]),
            "revenue_yoy_pct": _first_number(row, ["主营业务收入增长率", "营业收入同比增长", "or_yoy"]),
            "net_profit_yoy_pct": _first_number(row, ["净利润增长率", "净利润同比增长", "netprofit_yoy"]),
            "eps": _first_number(row, ["基本每股收益", "摊薄每股收益", "basic_eps"]),
        }
        financials["missing_fields"] = _financial_missing_fields(financials)
        if financials["missing_fields"]:
            errors.append(f"AKShare 最新财务报告缺少关键字段：{', '.join(financials['missing_fields'])}")
        return financials, errors
    except Exception as exc:
        errors.append(f"AKShare 财务指标失败：{exc}")
        return {}, errors


def huoqu_jibenmian(code: str, *, trade_date: str) -> dict[str, Any]:
    """Fetch profile, valuation, and financial indicators with explicit provenance."""
    as_of_date = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(as_of_date):
        raise ValueError(f"无效的分析日期：{trade_date}")
    as_of_date = pd.Timestamp(as_of_date).normalize()
    as_of_text = as_of_date.strftime("%Y-%m-%d")
    profile: dict[str, Any] = {}
    valuation: dict[str, Any] = {}
    financials: dict[str, Any] = {}
    sources: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    data_quality: dict[str, Any] = {}

    try:
        pro = _tushare_pro()
        basic_quality: dict[str, Any] = {}
        basic_all = _load_or_fetch_stock_basic(pro, basic_quality)
        data_quality["stock_basic"] = basic_quality.get("stock_basic", {})
        warnings.extend(str(item) for item in basic_quality.get("warnings", []))
        basic = basic_all[basic_all["ts_code"].astype(str) == code]
        if basic is not None and not basic.empty:
            profile = {str(key): _json_value(value) for key, value in basic.iloc[0].items()}
            basic_source = str(data_quality["stock_basic"].get("source") or "tushare")
            sources["profile"] = (
                "tushare" if basic_source == "tushare" else f"tushare_{basic_source}"
            )
    except Exception as exc:
        errors.append(f"Tushare 基本资料失败：{exc}")

    try:
        pro = _tushare_pro()
        basic_daily = pro.daily_basic(
            ts_code=code,
            trade_date=trade_date.replace("-", ""),
            fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb,total_mv,circ_mv",
        )
        if basic_daily is not None and not basic_daily.empty:
            dated = basic_daily.copy()
            trade_dates = (
                dated["trade_date"]
                if "trade_date" in dated.columns
                else pd.Series(pd.NaT, index=dated.index, dtype="datetime64[ns]")
            )
            dated["_trade_date"] = pd.to_datetime(trade_dates, errors="coerce")
            dated = dated.dropna(subset=["_trade_date"])
            dated = dated[dated["_trade_date"].dt.normalize() <= as_of_date].sort_values("_trade_date")
            if not dated.empty:
                row = dated.iloc[-1]
                valuation_date = pd.Timestamp(row["_trade_date"]).strftime("%Y-%m-%d")
                if valuation_date == as_of_text:
                    valuation = {
                        "as_of": valuation_date,
                        "pe_dynamic": _round_optional(row.get("pe")),
                        "pe_ttm": _round_optional(row.get("pe_ttm")),
                        "pb": _round_optional(row.get("pb")),
                        "total_market_value_yuan": _round_optional(float(row.get("total_mv")) * 10000 if pd.notna(row.get("total_mv")) else None, 2),
                        "circulating_market_value_yuan": _round_optional(float(row.get("circ_mv")) * 10000 if pd.notna(row.get("circ_mv")) else None, 2),
                        "turnover_rate_pct": _round_optional(row.get("turnover_rate")),
                        "volume_ratio": _round_optional(row.get("volume_ratio")),
                    }
                    sources["valuation"] = "tushare"
                else:
                    errors.append(
                        f"Tushare 估值日期为 {valuation_date}，与分析日 {as_of_text} 不一致，已忽略"
                    )
    except Exception as exc:
        errors.append(f"Tushare 估值失败：{exc}")

    try:
        pro = _tushare_pro()
        indicator = pro.fina_indicator(
            ts_code=code,
            fields=(
                "ts_code,ann_date,end_date,roe,roe_dt,grossprofit_margin,netprofit_margin,"
                "debt_to_assets,or_yoy,netprofit_yoy,ocf_to_or,basic_eps"
            ),
        )
        if indicator is not None and not indicator.empty:
            known = indicator.copy()
            announcement_dates = (
                known["ann_date"]
                if "ann_date" in known.columns
                else pd.Series(pd.NaT, index=known.index, dtype="datetime64[ns]")
            )
            report_dates = (
                known["end_date"]
                if "end_date" in known.columns
                else pd.Series(pd.NaT, index=known.index, dtype="datetime64[ns]")
            )
            known["_ann_date"] = pd.to_datetime(announcement_dates, errors="coerce")
            known["_end_date"] = pd.to_datetime(report_dates, errors="coerce")
            known = known.dropna(subset=["_ann_date", "_end_date"])
            known = known[
                (known["_ann_date"].dt.normalize() <= as_of_date)
                & (known["_end_date"].dt.normalize() <= as_of_date)
            ]
            known = known.sort_values(["_end_date", "_ann_date"])
            if known.empty:
                errors.append(f"Tushare 截至 {as_of_text} 没有已公告的财务指标")
            else:
                row = known.iloc[-1]
                roe = _round_optional(row.get("roe"))
                roe_diluted = _round_optional(row.get("roe_dt"))
                if roe is None:
                    roe = roe_diluted
                financials = {
                    "known_as_of": as_of_text,
                    "report_date": pd.Timestamp(row["_end_date"]).strftime("%Y-%m-%d"),
                    "announcement_date": pd.Timestamp(row["_ann_date"]).strftime("%Y-%m-%d"),
                    "roe_pct": roe,
                    "roe_diluted_pct": roe_diluted,
                    "gross_margin_pct": _round_optional(row.get("grossprofit_margin")),
                    "net_margin_pct": _round_optional(row.get("netprofit_margin")),
                    "debt_to_assets_pct": _round_optional(row.get("debt_to_assets")),
                    "revenue_yoy_pct": _round_optional(row.get("or_yoy")),
                    "net_profit_yoy_pct": _round_optional(row.get("netprofit_yoy")),
                    "operating_cashflow_to_revenue_pct": _round_optional(row.get("ocf_to_or")),
                    "eps": _round_optional(row.get("basic_eps")),
                }
                financials["missing_fields"] = _financial_missing_fields(financials)
                if financials["missing_fields"]:
                    errors.append(
                        f"截至 {as_of_text} 的最新已公告财报缺少关键字段："
                        f"{', '.join(financials['missing_fields'])}"
                    )
                sources["financials"] = "tushare"
    except Exception as exc:
        errors.append(f"Tushare 财务指标失败：{exc}")

    need_ak_info = not profile or not valuation
    if need_ak_info:
        try:
            ak_info, ak_errors = _akshare_info(code)
            errors.extend(ak_errors)
            if not profile and ak_info:
                profile = {
                    "ts_code": code,
                    "name": ak_info.get("股票简称") or ak_info.get("名称"),
                    "industry": ak_info.get("行业"),
                    "market": ak_info.get("市场"),
                    "list_date": ak_info.get("上市时间"),
                    "total_share": ak_info.get("总股本"),
                    "circulating_share": ak_info.get("流通股"),
                }
                sources["profile"] = "akshare"
            can_use_current_snapshot = _can_use_current_akshare_snapshot(as_of_date)
            if not valuation and ak_info and can_use_current_snapshot:
                valuation = {
                    "as_of": as_of_text,
                    "as_of_note": "AKShare 快照未提供原始交易日期，仅在最近完成交易日使用",
                    "pe_dynamic": _round_optional(ak_info.get("动态市盈率")),
                    "pe_ttm": None,
                    "pb": _round_optional(ak_info.get("市净率")),
                    "total_market_value_yuan": _round_optional(ak_info.get("总市值"), 2),
                    "circulating_market_value_yuan": _round_optional(ak_info.get("流通市值"), 2),
                    "turnover_rate_pct": _round_optional(ak_info.get("换手率")),
                    "volume_ratio": None,
                }
                sources["valuation"] = "akshare"
            elif not valuation and ak_info:
                errors.append(f"AKShare 实时估值与历史分析日 {as_of_text} 不一致，已忽略该快照")
        except Exception as exc:
            errors.append(f"AKShare 基本面降级失败：{exc}")

    if not financials:
        if _can_use_current_akshare_snapshot(as_of_date):
            try:
                financials, ak_errors = _akshare_financials(code, as_of=as_of_text)
                errors.extend(ak_errors)
                if financials:
                    financials["known_as_of"] = as_of_text
                    sources["financials"] = "akshare"
            except Exception as exc:
                errors.append(f"AKShare 财务指标降级失败：{exc}")
        else:
            errors.append(f"AKShare 财务指标缺少公告日，未用于历史分析日 {as_of_text}")

    return {
        "profile": profile,
        "valuation": valuation,
        "financials": financials,
        "sources": sources,
        "data_quality": data_quality,
        "warnings": warnings,
        "errors": errors,
    }


def _fundamental_score(fundamentals: dict[str, Any]) -> tuple[int | None, list[str]]:
    financials = fundamentals.get("financials") or {}
    valuation = fundamentals.get("valuation") or {}
    profile = fundamentals.get("profile") or {}
    industry = str(profile.get("industry") or profile.get("所属行业") or "")
    financial_industry = any(
        keyword in industry for keyword in ("银行", "保险", "证券", "多元金融", "金融服务")
    )
    evidence: list[str] = []
    score = 50.0
    observed = 0

    roe = _round_optional(financials.get("roe_pct"))
    if roe is not None:
        observed += 1
        if roe >= 15:
            score += 15
            evidence.append("ROE 较强")
        elif roe >= 8:
            score += 7
            evidence.append("ROE 为正且处于中等水平")
        elif roe < 0:
            score -= 18
            evidence.append("ROE 为负")

    growth = _round_optional(financials.get("net_profit_yoy_pct"))
    if growth is not None:
        observed += 1
        if growth >= 15:
            score += 12
            evidence.append("净利润同比增长较快")
        elif growth < -15:
            score -= 15
            evidence.append("净利润同比明显下降")

    debt = _round_optional(financials.get("debt_to_assets_pct"))
    if debt is not None:
        observed += 1
        if financial_industry:
            evidence.append("金融行业资产负债率口径特殊，本项只展示、不加减分")
        elif debt > 75:
            score -= 12
            evidence.append("资产负债率偏高，需结合行业解释")
        elif debt < 45:
            score += 5
            evidence.append("资产负债率相对温和")

    pe = _round_optional(valuation.get("pe_ttm"))
    if pe is None:
        pe = _round_optional(valuation.get("pe_dynamic"))
    if pe is not None:
        observed += 1
        if pe <= 0:
            score -= 12
            evidence.append("市盈率为负，通常意味着当前口径下亏损")
        elif pe > 80:
            score -= 8
            evidence.append("市盈率较高，估值对增长兑现要求较高")
        else:
            evidence.append("估值数据可用，需与同行比较后再下结论")

    return (int(round(max(0, min(100, score)))) if observed >= 2 else None), evidence


def _a_share_rules(
    code: str,
    name: str,
    *,
    price_limit_exempt: bool | None = None,
) -> dict[str, Any]:
    upper_name = str(name).strip().upper()
    inferred_exempt = upper_name.startswith(("N", "C"))
    exempt = inferred_exempt if price_limit_exempt is None else bool(price_limit_exempt)
    price_rule = _price_limit_rule(code, name, price_limit_exempt=exempt)
    normalized = str(code).upper()
    digits = normalized.split(".")[0]
    if normalized.endswith(".BJ"):
        buy_lot = "竞价买入单笔不少于 100 股，超过 100 股的部分可按 1 股递增"
    elif digits.startswith(("688", "689")):
        buy_lot = "科创板竞价买入单笔不少于 200 股，超过 200 股的部分可按 1 股递增"
    else:
        buy_lot = "沪深主板和创业板竞价买入通常按 100 股或其整数倍申报"
    return {
        "settlement": "T+1：当日买入的股票最早下一个交易日卖出",
        "price_limit_status": price_rule.status,
        "price_limit_pct": (
            round(price_rule.limit_rate * 100, 2) if price_rule.limit_rate is not None else None
        ),
        "price_limit_rule_effective_from": price_rule.effective_from,
        "price_limit_status_basis": (
            "股票简称 N/C 标记或调用方提供的无涨跌幅状态"
            if exempt
            else "按普通交易日板块规则归类；重新上市、退市整理首日等特殊状态仍以交易所当日信息为准"
        ),
        "price_limit_note": (
            price_rule.reason
            if exempt
            else f"{price_rule.reason}；特殊无涨跌幅限制交易日以交易所当日证券状态为准"
        ),
        "buy_lot": buy_lot,
        "prediction_horizon": (
            "单股工具支持 T+1/T+2/T+3：T日收盘后生成信号，下一交易日开盘计划入场；"
            "T+1/T+2/T+3分别表示入场后第1/2/3个可卖出交易日收盘。"
            "模型输出入场到退出收益，不伪造尚未知开盘价对应的精确目标价"
        ),
    }


def fenxi_gupiao(
    *,
    gupiao: str,
    source: str = "auto",
    history_calendar_days: int | None = None,
    holding_days: int = 2,
    budget_yuan: float | None = None,
    config_path: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """Run deterministic single-stock research with horizon-aware ML gates."""
    _ensure_dotenv()
    config, resolved_config = jiazai_lianghua_peizhi(config_path)
    holding_days = int(holding_days)
    if holding_days not in {1, 2, 3}:
        raise ValueError("holding_days 必须是 1、2 或 3 个交易日")
    if budget_yuan is not None and (not math.isfinite(float(budget_yuan)) or float(budget_yuan) <= 0):
        raise ValueError("budget_yuan 必须是大于0的有限数值")
    configured_history_days = int(config.get("dangu", {}).get("history_calendar_days", 1440))
    history_calendar_days = configured_history_days if history_calendar_days is None else int(history_calendar_days)
    history_calendar_days = max(540, min(history_calendar_days, 1800))
    code, resolved, resolve_warnings = jiexi_gupiao(gupiao, source=source)
    reference = datetime.now()
    end = reference.date()
    start = end - timedelta(days=history_calendar_days)
    market = huoqu_rili_xingqing(
        code,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        source=source,
    )
    if market.data.empty:
        return {
            "status": "error",
            "error": f"无法取得 {code} 的日线行情",
            "data_errors": list(market.errors),
        }

    analysis_history, completion_warnings = _completed_market_history(market.data, reference=reference)
    market_warnings = list(resolve_warnings) + list(market.warnings) + completion_warnings
    if analysis_history.empty:
        return {
            "status": "error",
            "error": f"{code} 没有已确认收盘的日线行情",
            "data_errors": list(market.errors),
            "market_data": {"warnings": market_warnings},
        }
    try:
        technical = zongjie_jishu(analysis_history)
    except (KeyError, RuntimeError, ValueError) as exc:
        return {
            "status": "error",
            "error": f"{code} 的有效日线不足，无法完成技术分析：{exc}",
            "data_errors": list(market.errors),
            "market_data": {"rows": int(len(analysis_history)), "warnings": market_warnings},
        }
    technical_as_of = pd.Timestamp(str(technical["trade_date"])).normalize()
    raw_analysis_end = pd.to_datetime(analysis_history["trade_date"], errors="coerce").max()
    if pd.notna(raw_analysis_end) and pd.Timestamp(raw_analysis_end).normalize() > technical_as_of:
        analysis_history = analysis_history[
            pd.to_datetime(analysis_history["trade_date"], errors="coerce").dt.normalize() <= technical_as_of
        ].copy()
        market_warnings.append("末尾行情缺少形成指标所需的数据，分析时点已回退到最近可用交易日")
    freshness = _market_data_freshness(technical["trade_date"], reference=reference)
    if freshness["status"] == "too_stale":
        return {
            "status": "error",
            "error": (
                f"{code} 最新可用行情停留在 {technical['trade_date']}，"
                f"距最近应完成交易日已 {freshness['business_days_old']} 个工作日；"
                "可能处于停牌或数据源延迟状态，已停止输出当前分析"
            ),
            "as_of": technical["trade_date"],
            "market_data": {
                "source": market.source,
                "adjustment": market.adjustment,
                "freshness": freshness,
                "warnings": market_warnings,
                "errors": list(market.errors),
            },
        }
    fundamentals = huoqu_jibenmian(code, trade_date=str(technical["trade_date"]))
    if resolved and not fundamentals.get("profile"):
        fundamentals["profile"] = resolved
        fundamentals.setdefault("sources", {})["profile"] = "local_cache"
    profile = fundamentals.get("profile") or {}
    name = str(profile.get("name") or resolved.get("name") or "")
    fundamental_score, fundamental_evidence = _fundamental_score(fundamentals)
    from src.ashare.dangu_yuce import (
        huoqu_dangqian_kuaizhao,
        pinggu_kejiaoyixing,
        yanjiu_dangu_yuce,
    )

    execution_reference = datetime.now()
    current_quote = huoqu_dangqian_kuaizhao(code, reference=execution_reference)
    tradability = pinggu_kejiaoyixing(
        code=code,
        name=name,
        profile=profile,
        history=analysis_history,
        freshness=freshness,
        current_quote=current_quote,
        config=config,
        reference=execution_reference,
    )
    industry = str(profile.get("industry") or profile.get("所属行业") or resolved.get("industry") or "")
    try:
        quantitative = yanjiu_dangu_yuce(
            code=code,
            name=name,
            industry=industry,
            target_history=analysis_history,
            target_source=market.source,
            target_adjustment=market.adjustment,
            source=source,
            signal_date=str(technical["trade_date"]),
            holding_days=holding_days,
            budget_yuan=budget_yuan,
            config=config,
            technical=technical,
            fundamentals=fundamentals,
            tradability=tradability,
        )
    except Exception as exc:
        fallback_label = "证据偏负面" if not tradability.get("basic_execution_feasible") else "证据不足"
        fallback_reasons = list(tradability.get("hard_blocks", []))
        fallback_reasons.append(f"单股量化模型本次不可用：{exc}")
        quantitative = {
            "status": "unavailable",
            "requested_horizon": f"T+{holding_days}",
            "forecast": {},
            "future_3_trading_days": {
                "status": "unavailable",
                "signal_date": str(technical["trade_date"]),
                "forecast": {},
                "error": str(exc),
            },
            "validation": {"horizons": {}, "passed_horizons": 0},
            "analysis_assessment": {
                "evidence_label": fallback_label,
                "requested_horizon": f"T+{holding_days}",
                "summary": f"{fallback_label}：{fallback_reasons[0]}",
                "reasons": fallback_reasons,
                "signal_gate": {
                    "actionable_signal": False,
                    "decision": "abstain",
                    "reasons": fallback_reasons,
                },
                "responsibility_note": "这是分析证据汇总，不是交易指令；最终决定由用户自行作出。",
            },
            "error": str(exc),
            "failure_category": classify_failure(exc),
            "failure_stage": "single_stock_model",
            "limitations": ["模型失败时不使用启发式技术分替代收益预测"],
        }
    risks: list[str] = []
    if market.adjustment == "raw_unadjusted":
        risks.append("行情未复权，历史分红送转可能影响长周期技术指标")
    if technical.get("annualized_volatility_20") and float(technical["annualized_volatility_20"]) > 0.55:
        risks.append("近期波动率较高，短线技术判断的不确定性会增大")
    if any(keyword in name.upper() for keyword in ["ST", "退"]):
        risks.append("股票名称包含 ST/退市风险标记")
    if not fundamentals.get("financials"):
        risks.append("财务指标接口未返回数据，基本面结论不完整")
    elif fundamentals["financials"].get("missing_fields"):
        risks.append(
            "最新可用财报缺少关键字段："
            + "、".join(str(field) for field in fundamentals["financials"]["missing_fields"])
            + "；基本面评分只使用实际取得的字段"
        )
    if str(fundamentals.get("sources", {}).get("profile", "")).endswith("stale_cache"):
        risks.append("股票基本资料刷新失败，名称、行业或风险状态来自过期缓存")
    if freshness["status"] == "possibly_stale":
        risks.append(
            f"最新行情距最近应完成交易日约 {freshness['business_days_old']} 个工作日，"
            "可能存在停牌、长假或数据接口延迟"
        )
    risks.extend(str(value) for value in tradability.get("hard_blocks", []))
    risks.extend(str(value) for value in tradability.get("cautions", []))
    if quantitative.get("status") != "ok":
        risks.append("指定持有期量化模型本次不可用或同行样本不足，相关证据不足")

    peer_universe = quantitative.get("peer_universe", {})
    history_fetch = peer_universe.get("history_fetch", {})
    data_health = build_data_health(
        as_of=str(technical["trade_date"]),
        expected_as_of=freshness.get("expected_latest_date"),
        freshness=freshness,
        sources={
            "market_history": market.source,
            "adjustment": market.adjustment,
            "fundamentals": fundamentals.get("sources", {}),
            "peer_history": history_fetch.get("history_sources", {}),
            "daily_factors": quantitative.get("daily_factor_data", {}).get("source"),
        },
        warehouse=peer_universe.get("warehouse_range"),
        warnings=market_warnings + list(history_fetch.get("warnings", [])),
        errors=list(market.errors) + list(history_fetch.get("errors", [])),
        constituent_history={
            **(peer_universe.get("stock_master_snapshot") or {"status": "unavailable"}),
            "usage": "同行池优先使用不晚于信号日的股票资料快照；仓库建立前的行业成员变化仍无法倒推",
        },
    )

    result: dict[str, Any] = {
        "status": "ok",
        "tool_contract_version": SINGLE_STOCK_TOOL_CONTRACT_VERSION,
        "analysis_type": "single_stock",
        "analysis_request": {
            "requested_holding_trading_days": holding_days,
            "requested_horizon": f"T+{holding_days}",
            "budget_yuan": budget_yuan,
            "history_calendar_days": history_calendar_days,
        },
        "stock": {"ts_code": code, "name": name, **{key: value for key, value in profile.items() if key not in {"ts_code", "name"}}},
        "as_of": technical["trade_date"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_data": {
            "source": market.source,
            "adjustment": market.adjustment,
            "rows": int(len(analysis_history)),
            "start_date": analysis_history["trade_date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": analysis_history["trade_date"].iloc[-1].strftime("%Y-%m-%d"),
            "freshness": freshness,
            "warnings": market_warnings,
            "errors": list(market.errors),
        },
        "data_health": data_health,
        "current_quote": current_quote,
        "tradability": tradability,
        "quantitative_analysis": quantitative,
        "future_3_trading_days": quantitative.get("future_3_trading_days", {}),
        "analysis_assessment": quantitative.get("analysis_assessment", {}),
        "technical_analysis": technical,
        "fundamental_analysis": {
            **fundamentals,
            "score_0_100": fundamental_score,
            "score_interpretation": (
                "启发式检查分，只能用于同一数据完整度下的初筛；未做完整同行估值排名，"
                "不能解释为上涨概率或精确目标分"
            ),
            "evidence": fundamental_evidence,
        },
        "a_share_rules": _a_share_rules(code, name),
        "risks": risks,
        "configuration": {
            "quant_config_path": resolved_config,
            "cost_scenario": config.get("jiaoyi", {}).get("cost_scenario"),
            "dynamic_slippage_enabled": config.get("jiaoyi", {}).get("dynamic_slippage_enabled", True),
        },
        "scope_note": (
            "这是基于公开数据、同行面板和滚动样本外验证的A股研究结果，不是收益保证；"
            "数值、模型门槛和证据标签由程序生成，LLM只负责解释，不得改写"
        ),
        "execution_policy": "research_only：程序只做分析和预测，不连接券商、不读取交易账户、不提交委托，也不替用户作买卖决定。",
    }

    if run_dir:
        try:
            run_path = safe_run_dir(run_dir)
            artifact_dir = run_path / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            output = artifact_dir / f"gupiao_fenxi_{code.replace('.', '_')}.json"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["artifact"] = str(output)
        except Exception as exc:
            result["artifact_error"] = str(exc)
    return result
