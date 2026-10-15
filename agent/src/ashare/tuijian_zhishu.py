"""固定公式的当前量价推荐指数；不是未来收益、上涨概率或模型预测。

历史输入必须已由数据层按权威交易日历核验并对齐；缺失交易日以空行
保留。本模块不取行情，不根据星期猜交易日，也不持久化市场输入。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


RECOMMENDATION_RAW_COLUMNS = (
    "ret_20", "directional_efficiency_20", "up_amount_dominance_20",
)


def _numeric(history: pd.DataFrame, name: str) -> pd.Series:
    if name not in history:
        return pd.Series(np.nan, index=history.index, dtype=float)
    return pd.to_numeric(history[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def jisuan_tuijian_yuanzhi(history: pd.DataFrame) -> pd.DataFrame:
    """计算单股各日的R、带方向路径效率E及上涨成交额占优度U。

    输入顺序应为时间升序，索引原样保留。21条正收盘组成20个连续
    差分；E为净变化/绝对路径，U为sum(真实成交额*sign(差分))/sum(真实成交额)。
    U只用最后20日成交额，平盘方向为零；禁止收盘价乘成交量代理。
    零价格路径使E留空；零成交额总额、缺口及负成交额使U留空。
    """
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history必须是单股DataFrame")
    if "ts_code" in history and history["ts_code"].dropna().nunique() > 1:
        raise ValueError("推荐原值必须逐股计算，不能跨证券滚动")
    close = _numeric(history, "close").where(lambda values: values.gt(0))
    amount = _numeric(history, "amount_yuan").where(lambda values: values.ge(0))
    if "trade_date" in history:
        dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
        if not dates.dropna().is_monotonic_increasing:
            raise ValueError("推荐原值历史必须按交易日期升序输入")
        invalid_dates = dates.isna() | dates.duplicated(keep=False)
        close = close.where(~invalid_dates)
        amount = amount.where(~invalid_dates)
    full_price_window = close.rolling(21, min_periods=21).count().eq(21)
    change = close.diff()
    path = change.abs().rolling(20, min_periods=20).sum()
    amount_sum = amount.rolling(20, min_periods=20).sum()
    signed_amount = (np.sign(change) * amount).rolling(20, min_periods=20).sum()
    result = pd.DataFrame(index=history.index)
    result["ret_20"] = (close / close.shift(20) - 1).where(full_price_window)
    result["directional_efficiency_20"] = (
        (close - close.shift(20)) / path.where(path.gt(0))
    ).where(full_price_window).clip(-1, 1)
    result["up_amount_dominance_20"] = (
        signed_amount / amount_sum.where(amount_sum.gt(0))
    ).where(full_price_window).clip(-1, 1)
    return result.replace([np.inf, -np.inf], np.nan)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def jisuan_tuijian_zhishu(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """对同一分析日、同一资格池的完整三原值计算可排序推荐指数。

    共同有效池N>=2；p=(升序平均名次-0.5)/N，价格组P=(pR+pE)/2，
    成交组V=pU，指数=100*sqrt(P*V)。两组几何等权，价格组内部等权，
    是公开固定的设计取舍，没有拟合、训练或按结果寻优。结果不舍入，
    缺项不填中性值、不重分权重；排除项及共同池数量同时返回。
    """
    results: dict[str, dict[str, Any]] = {}
    valid_rows: list[dict[str, Any]] = []
    for record in records:
        code = str(record.get("ts_code") or "").strip()
        if not code:
            raise ValueError("推荐指数输入缺少ts_code")
        if code in results:
            raise ValueError(f"推荐指数同一比较池存在重复证券：{code}")
        raw = {name: _finite_number(record.get(name)) for name in RECOMMENDATION_RAW_COLUMNS}
        missing = [f"{name}:missing_or_nonfinite" for name, value in raw.items() if value is None]
        if raw["ret_20"] is not None and raw["ret_20"] <= -1:
            missing.append("ret_20:outside_positive_price_return_range")
        for name in RECOMMENDATION_RAW_COLUMNS[1:]:
            if raw[name] is not None and not -1 <= raw[name] <= 1:
                missing.append(f"{name}:outside_minus_one_to_one")
        results[code] = {
            "ts_code": code,
            "status": "unavailable",
            "raw_values": raw,
            "percentiles": {name: None for name in RECOMMENDATION_RAW_COLUMNS},
            "price_group": None,
            "volume_group": None,
            "recommendation_index": None,
            "pool_size": 0,
            "missing_reasons": missing,
        }
        if not missing:
            valid_rows.append({"ts_code": code, **raw})
    pool_size = len(valid_rows)
    for result in results.values():
        result["pool_size"] = pool_size
    if pool_size < 2:
        for result in results.values():
            result["missing_reasons"].append("common_valid_pool_requires_at_least_two")
        return results
    values = pd.DataFrame(valid_rows).set_index("ts_code")
    percentiles = (values.rank(method="average", ascending=True) - 0.5) / pool_size
    for code in values.index:
        fractions = {name: float(percentiles.at[code, name]) for name in RECOMMENDATION_RAW_COLUMNS}
        price = (fractions["ret_20"] + fractions["directional_efficiency_20"]) / 2
        volume = fractions["up_amount_dominance_20"]
        results[code].update({
            "status": "success",
            "percentiles": fractions,
            "price_group": price,
            "volume_group": volume,
            "recommendation_index": 100 * math.sqrt(price * volume),
        })
    return results


__all__ = [
    "RECOMMENDATION_RAW_COLUMNS", "jisuan_tuijian_yuanzhi", "jisuan_tuijian_zhishu",
]
