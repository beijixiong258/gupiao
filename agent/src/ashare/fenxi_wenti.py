"""用户明确条件与研究问题：只核验可观测命题，不把证据数量变成评分。"""

from __future__ import annotations

import math
import operator
from functools import lru_cache
from typing import Any


_OPERATORS = {"gt": operator.gt, "gte": operator.ge, "lt": operator.lt,
              "lte": operator.le, "eq": operator.eq}
_SYMBOLS = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "eq": "="}
_PROFILE_FIELDS = {
    "is_st": ("简称含ST标记", "0/1"), "is_delisting": ("简称含退市标记", "0/1"),
    "listing_days": ("已核验上市自然日数", "日"), "amount_yuan": ("最近完整日线成交额", "元"),
}
_FINANCIAL_FIELDS = {"roe_pct": "净资产收益率", "net_profit_yoy_pct": "净利润同比",
                     "revenue_yoy_pct": "营业收入同比", "debt_to_assets_pct": "资产负债率"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@lru_cache(maxsize=1)
def _definitions() -> dict[str, dict[str, Any]]:
    from src.ashare.yinzi_gongcheng import FACTOR_REGISTRY
    return {entry["feature"]: entry for entry in FACTOR_REGISTRY}


def _normalize_checks(value: Any, *, research: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 12:
        raise ValueError("指标条件须为不超过12项的列表")
    allowed = set(_definitions()) | set(_PROFILE_FIELDS)
    if research:
        allowed |= set(_FINANCIAL_FIELDS) | {"pattern_close_confirmed"}
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"metric", "operator", "value", "reference_metric", "label"}:
            raise ValueError("指标条件含未知字段")
        metric, operation = item.get("metric"), item.get("operator")
        if metric not in allowed or operation not in _OPERATORS:
            raise ValueError(f"暂不支持该指标条件：{metric}；请明确可核验的指标和比较关系")
        reference = item.get("reference_metric")
        threshold = _number(item.get("value"))
        if (reference is not None) == (item.get("value") is not None):
            raise ValueError("每项条件须且只能指定数值阈值或参照指标")
        if reference is not None and reference not in allowed:
            raise ValueError(f"不支持的参照指标：{reference}")
        if reference is None and threshold is None:
            raise ValueError("指标阈值须为有限数值，单位采用指标注册表的原始口径")
        label = str(item.get("label") or metric).strip()
        result.append({"metric": metric, "operator": operation, "label": label,
                       **({"reference_metric": reference} if reference is not None else {"value": threshold})})
    return result


def jiexi_xuangu_tiaojian(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, dict) or set(value) - {"mode", "rules"}:
        raise ValueError("选股条件须明确为 none、rules 或 unresolved")
    if value.get("mode") == "none" and not value.get("rules"):
        return []
    if value.get("mode") != "rules" or not value.get("rules"):
        raise ValueError("请补充您明确要求的指标条件和界限；未提出风险要求时无需设定风险条件")
    return _normalize_checks(value["rules"], research=False)


def jiexi_yanjiu_wenti(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"question", "checks", "unverifiable_parts"}:
        raise ValueError("研究问题须包含原始观点及其可核验条件")
    question = str(value.get("question") or "").strip()
    unavailable = value.get("unverifiable_parts", [])
    if not question or not isinstance(unavailable, list) or not all(isinstance(part, str) for part in unavailable):
        raise ValueError("请保留待验证观点原文，并逐项说明暂不能量化核验的部分")
    return {"question": question, "checks": _normalize_checks(value.get("checks", []), research=True),
            "unverifiable_parts": unavailable}


def _observation(metric: str, item: dict[str, Any]) -> dict[str, Any]:
    factor = item.get("factor") or {}
    date = factor.get("trade_date") or (item.get("data_quality") or {}).get("as_of")
    definition = _definitions().get(metric, {})
    result = {"metric": metric, "label": definition.get("label", metric), "value": None,
              "unit": definition.get("unit", ""), "display_scale": definition.get("display_scale", 1),
              "as_of": date, "family": definition.get("group"),
              "source_basis": definition.get("source_basis"), "formula": definition.get("formula")}
    for group in (factor.get("groups") or {}).values():
        if metric in (group.get("values") or {}):
            result["value"] = _number(group["values"][metric])
            result["reason"] = "" if result["value"] is not None else "该指标所需窗口、来源或样本不足"
            return result
    technical = item.get("technical") or {}
    raw = technical.get("raw_indicators") or {}
    if definition and metric in raw and str(technical.get("trade_date") or "")[:10] == str(date or "")[:10] and date:
        result.update(value=_number(raw[metric]), source_basis="同次日线技术分析未舍入原值；横截面该字段未返回")
        result["reason"] = "" if result["value"] is not None else "原始技术指标窗口不足"
        return result
    if metric in _PROFILE_FIELDS:
        profile = item.get("profile") or {}
        result.update(label=_PROFILE_FIELDS[metric][0], unit=_PROFILE_FIELDS[metric][1], family="profile")
        name = str(item.get("name", profile.get("name", ""))).strip()
        if name.lower() in {"", "none", "nan", "<na>", "未知"} or name == str(item.get("ts_code") or "") or name.isdigit():
            name = ""
        if metric in {"is_st", "is_delisting"} and name:
            result["value"] = float("ST" in name.upper() if metric == "is_st" else "退" in name)
            result["source_basis"] = "本次已核验股票身份的简称；不是对公告内容的推断"
        elif metric == "listing_days":
            import pandas as pd
            listed, as_of = pd.to_datetime(str(profile.get("list_date", "")), errors="coerce"), pd.to_datetime(date, errors="coerce")
            if listed is not None and as_of is not None and not pd.isna(listed) and not pd.isna(as_of) and listed <= as_of:
                result["value"] = float((as_of.normalize() - listed.normalize()).days)
            result["source_basis"] = "本次来源的明确上市日；不以已下载日线跨度代替上市日"
        elif metric == "amount_yuan":
            history = item.get("history")
            if history is not None and not history.empty:
                result["value"] = _number(history.iloc[-1].get("amount_yuan"))
            result["source_basis"] = "最近完整日线真实成交额"
    elif metric in _FINANCIAL_FIELDS:
        fundamental = item.get("fundamental") or {}
        financials = fundamental.get("financials") or {}
        result.update(label=_FINANCIAL_FIELDS[metric], value=_number(financials.get(metric)), unit="%", family="financials",
                      as_of=financials.get("end_date") or financials.get("report_date"),
                      source_basis=fundamental.get("sources") or {},
                      report_timing={key: financials.get(key) for key in ("report_date", "announcement_date", "known_as_of")})
    elif metric == "pattern_close_confirmed":
        pattern = item.get("pattern") or {}
        state = pattern.get("state")
        result.update(label="涨停回马枪形态已收盘确认", unit="0/1", family="pattern",
                      source_basis="本次形态程序的当前适用与确认状态", state=state)
        if state and pattern.get("status") not in {"error", "unavailable"}:
            result["value"] = float(bool(pattern.get("eligible")) and state == "close_confirmed")
    result["reason"] = "" if result["value"] is not None else "本次未取得该条件所需的有效证据"
    return result


def heyan_mingque_tiaojian(rules: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, rule in enumerate(rules, 1):
        observed = _observation(rule["metric"], item)
        reference = _observation(rule["reference_metric"], item) if rule.get("reference_metric") else None
        rhs = reference["value"] if reference else rule["value"]
        lhs = observed["value"]
        compatible = not reference or ((observed["unit"], observed["display_scale"]) == (reference["unit"], reference["display_scale"])
                                       and observed["as_of"] is not None and observed["as_of"] == reference["as_of"])
        available = lhs is not None and rhs is not None and compatible
        status = "met" if available and _OPERATORS[rule["operator"]](lhs, rhs) else "unmet" if available else "unavailable"
        relation = f"{observed['label']} {_SYMBOLS[rule['operator']]} " + (reference["label"] if reference else f"{rhs:g}（原始单位）")
        result.append({"key": f"user_condition_{index}", "label": rule["label"], "status": status,
                       "rule": rule, "observation": observed, "reference": reference,
                       "values": {rule["metric"]: lhs, "threshold": rhs},
                       "reason": relation if available else "两项指标的单位或观测时点未对齐，不能直接比较" if not compatible else "必需证据缺失，不能判定条件通过"})
    return result


def pinggu_yanjiu_wenti(question: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any] | None:
    if question is None:
        return None
    checks = heyan_mingque_tiaojian(question["checks"], item)
    met = [entry for entry in checks if entry["status"] == "met"]
    unmet = [entry for entry in checks if entry["status"] == "unmet"]
    missing = [entry for entry in checks if entry["status"] == "unavailable"]
    if unmet:
        verdict = "partly_supported" if met else "not_supported"
    elif not checks or missing or question["unverifiable_parts"]:
        verdict = "insufficient_evidence"
    else:
        verdict = "supported"
    labels = {"supported": "支持所列可观测命题", "partly_supported": "部分支持，完整观点尚未成立",
              "not_supported": "所列命题存在直接反证", "insufficient_evidence": "证据不足，不能确认完整观点"}
    return {"question": question["question"], "verdict": verdict, "verdict_label": labels[verdict],
            "checks": checks, "supporting_checks": [entry["key"] for entry in met],
            "counter_checks": [entry["key"] for entry in unmet], "missing_checks": [entry["key"] for entry in missing],
            "unverifiable_parts": question["unverifiable_parts"],
            "interpretation": "逐项检验用户观点的可观测部分；全部通过才支持这些命题。相关指标不独立投票，不把命题核验换成未来涨跌概率。",
            "reassessment_conditions": [f"{entry['label']}的观测关系（{entry['reason']}）发生改变时重新核验" for entry in checks]}
