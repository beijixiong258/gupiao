"""量化分析结果的安全降级展示。

正常回答仍由智能体结合上下文组织；这里只在模型输出混入异常语言字符时，根据已计算的
结构化结果生成一份保守中文答案，避免展示层把坏文本交给用户。
"""

from __future__ import annotations

from typing import Any, Iterable


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _items(values: Any, maximum: int) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
        if len(result) >= maximum:
            break
    return result


def _scope_name(payload: dict[str, Any]) -> str:
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        return "当前请求"
    return _text(scope.get("canonical_name") or scope.get("scope_name") or "当前请求")


def _candidate_line(candidate: dict[str, Any], role: str) -> list[str]:
    name = _text(candidate.get("name")) or "未命名股票"
    code = _text(candidate.get("ts_code"))
    identity = f"{name}（{code}）" if code else name
    score = candidate.get("ranking_score_0_100")
    confidence = candidate.get("confidence")
    score_text = f"，综合排名分 {float(score):.2f}/100" if isinstance(score, (int, float)) else ""
    confidence_text = (
        f"，证据完整度约 {float(confidence) * 100:.0f}%"
        if isinstance(confidence, (int, float))
        else ""
    )
    lines = [
        f"{role}：{identity}{score_text}{confidence_text}。排名分和证据完整度都不是上涨概率。"
    ]
    positives = _items(candidate.get("positive_evidence"), 4)
    risks = _items(candidate.get("risks"), 3)
    unmet = _items(candidate.get("unmet_conditions"), 3)
    if positives:
        lines.append("主要依据：" + "；".join(positives) + "。")
    if risks or unmet:
        lines.append("主要风险或未满足条件：" + "；".join([*risks, *unmet][:5]) + "。")
    risk_price = candidate.get("risk_reference_price")
    if isinstance(risk_price, (int, float)):
        lines.append(f"形态风险参考价：{float(risk_price):.2f} 元。")
    return lines


def goujian_fenxi_anquan_huitui(payload: dict[str, Any] | None) -> str:
    """根据分析业务状态生成不含内部参数的中文安全回答。"""
    if not isinstance(payload, dict):
        return "本轮回答出现了异常语言内容，已阻止展示。请重试一次。"
    status = _text(payload.get("status"))
    if status == "clarification_required":
        lines = [_text(payload.get("error")) or "实时数据源里有多个可能范围，需要你确认一次。"]
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            labels = [_text(item.get("user_label")) for item in candidates if isinstance(item, dict)]
            labels = [value for value in labels if value]
            if labels:
                lines.append("可选范围：" + "；".join(labels[:4]) + "。")
        lines.append("请选择最符合原意的一项，也可以换一种日常说法；不需要了解专业分类或内部代码。")
        return "\n\n".join(lines)
    if status == "unavailable":
        reason = _text(payload.get("error")) or "远端市场数据当前不可用"
        action = _text(payload.get("next_action")) or "请稍后重试"
        return f"这次没有完成量化分析：{reason}。\n\n{action}。程序没有使用旧本地市场数据冒充当前结果。"
    if status in {"error", "failed"}:
        return f"量化分析没有完成：{_text(payload.get('error')) or '程序发生内部错误'}。"
    scope = _scope_name(payload)
    as_of = _text(payload.get("as_of") or payload.get("generated_at"))
    if payload.get("recommendation_available") is not True:
        reason = _text(payload.get("no_recommendation_reason")) or "当前没有候选同时达到排名分和可信度门槛"
        suffix = f"，数据截至 {as_of}" if as_of else ""
        return f"按数据源当前的“{scope}”范围分析{suffix}，这次不建议勉强选股：{reason}。"
    primary = payload.get("primary")
    if not isinstance(primary, dict):
        return "量化分析标记为完成，但没有返回可展示的首选；本轮不生成推荐。"
    time_text = f"，数据截至 {as_of}" if as_of else ""
    lines = [f"按数据源当前的“{scope}”范围完成量化分析{time_text}。"]
    lines.extend(_candidate_line(primary, "首选"))
    alternatives = payload.get("alternatives")
    if isinstance(alternatives, list):
        for index, candidate in enumerate(alternatives[:4], start=1):
            if isinstance(candidate, dict):
                lines.extend(_candidate_line(candidate, f"备选 {index}")[:1])
    lines.append("预测需要重新下载远端数据并训练模型，耗时明显更长；分析结果展示后再由你决定是否继续。")
    return "\n\n".join(lines)


__all__ = ["goujian_fenxi_anquan_huitui"]
