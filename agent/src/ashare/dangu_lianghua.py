"""单股证据提取：本股指标独立计算，同行与市场证据按可用程度补充。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ashare.buchong_zhenduan import goujian_buchong_zhenduan
from src.ashare.fenxi_weipan import WeipanJieduan, fenxi_weipan, panduan_weipan_jieduan
from src.ashare.fenxi_xingtai import fenxi_zhangting_huimaqiang
from src.ashare.fenxi_yinzi import (
    goujian_fenxi_yinzi_mianban,
    huizong_houxuan_yinzi,
    jisuan_hengjiemian_jibenmian,
    zengjia_dangri_guzhi_yinzi,
)
from src.ashare.shichang_shuju import FenxiShujuShangxiawen
from src.ashare.shuju_yuan import biaozhunhua_gupiao_daima


def _gap(code: str, component: str, reason: Any) -> dict[str, str]:
    return {
        "code": code,
        "component": component,
        "reason": " ".join(str(reason).split()),
        "impact": "仅限制这部分证据的解释，已取得的其他信息继续保留",
        "reassessment_condition": f"补齐并核验{component}数据后重新分析",
    }


def _code(value: Any) -> str | None:
    try:
        return biaozhunhua_gupiao_daima(str(value))
    except (TypeError, ValueError):
        return None


def _comparison_pool(
    universe: pd.DataFrame, *, code: str, name: str, industry: str,
    target_profile: dict[str, Any], config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """有界同行参照只用于比较，不把选股过滤器套在被点名股票上。"""
    if universe is None or universe.empty or "ts_code" not in universe:
        return pd.DataFrame([target_profile]), {"status": "unavailable", "reason": "横截面不可用"}
    data = universe.copy()
    data["ts_code"] = data["ts_code"].map(_code)
    data = data.dropna(subset=["ts_code"]).drop_duplicates("ts_code", keep="first")
    for column in ("name", "industry"):
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    amount = pd.to_numeric(data.get("amount_yuan", pd.Series(index=data.index, dtype=float)), errors="coerce")
    data = data.assign(_amount_order=amount).sort_values("_amount_order", ascending=False, na_position="last")
    found = data[data["ts_code"].eq(code)]
    current = {**target_profile, **(found.iloc[0].drop(labels=["_amount_order"]).to_dict() if not found.empty else {})}
    if target_profile.get("valuation_is_complete_daily") is True:
        # 目标日终估值已经核验；比较池的空字段或未核验快照不得覆盖它。
        for field in (
            "trade_date", "valuation_trade_date", "valuation_source", "valuation_is_complete_daily",
            "turnover_rate", "turnover_rate_pct", "circulating_market_value_yuan",
            "total_market_value_yuan", "pe", "pe_definition", "pe_ttm", "pb", "volume_ratio",
        ):
            current[field] = target_profile.get(field)
    current["name"] = name if name and name != code else current.get("name") or code
    current["industry"] = industry or current.get("industry") or ""
    current["peer_role"] = "target"
    peers = data[data["ts_code"].ne(code)].copy()
    settings = config.get("dangu") or {}
    maximum = max(1, int(settings.get("max_peer_stocks", 20)))
    same_limit = max(0, min(maximum - 1, int(settings.get("same_industry_stocks", 16))))
    same = peers[peers["industry"].eq(current["industry"])].head(same_limit) if current["industry"] else peers.iloc[:0]
    other = peers[~peers["ts_code"].isin(same["ts_code"])].head(maximum - 1 - len(same))
    selected = pd.concat([
        pd.DataFrame([current]), same.assign(peer_role="same_industry"),
        other.assign(peer_role="market_reference"),
    ], ignore_index=True, sort=False).drop(columns=["_amount_order"], errors="ignore")
    return selected, {
        "status": "ok" if len(selected) > 1 else "partial",
        "selected_stocks": len(selected),
        "configured_peer_limit": maximum,
        "same_industry_stocks": len(same),
        "selection_method": "优先本次真实行业标签一致的股票，再按成交额补足有界市场参照",
        "known_bias": "本次横截面及有界参照，不是历史完整行业成分或全市场统计",
    }


def yunxing_dangu_tongyi_lianghua(
    *, code: str, name: str, industry: str, history: pd.DataFrame,
    analysis_date: pd.Timestamp, snapshot: dict[str, Any], clock: dict[str, Any],
    technical: dict[str, Any], fundamentals: dict[str, Any], tradability: dict[str, Any],
    config: dict[str, Any], context: FenxiShujuShangxiawen,
) -> dict[str, Any]:
    """保留已有函数入口，返回原始诊断证据，不计算综合分或买入结论。"""
    del technical
    gaps: list[dict[str, Any]] = []
    target_profile = {
        **(fundamentals.get("profile") or {}),
        **(fundamentals.get("valuation") or {}),
        "ts_code": code, "name": name, "industry": industry, "peer_role": "target",
    }
    valuation = fundamentals.get("valuation") or {}
    valuation_dates = pd.to_datetime(
        [valuation.get("as_of"), valuation.get("valuation_trade_date")], errors="coerce",
    )
    target_profile["valuation_is_complete_daily"] = bool(
        valuation.get("valuation_is_complete_daily") is True
        and valuation.get("valuation_source") == "tushare_daily_basic"
        and valuation_dates.notna().all()
        and (valuation_dates.normalize() == analysis_date.normalize()).all()
    )
    if target_profile["valuation_is_complete_daily"]:
        target_profile["trade_date"] = analysis_date.normalize()
        target_profile["turnover_rate"] = valuation.get("turnover_rate_pct")
    profiles = pd.DataFrame([target_profile])
    universe_meta: dict[str, Any] = {}
    pool_meta: dict[str, Any] = {"status": "unavailable", "selected_stocks": 1}
    try:
        universe, universe_meta = context.zuixin_hengjiemian()
        profiles, pool_meta = _comparison_pool(
            universe, code=code, name=name, industry=industry,
            target_profile=target_profile, config=config,
        )
        profiles.attrs.update({key: universe_meta[key] for key in ("source", "as_of", "captured_at") if key in universe_meta})
        if pool_meta["status"] != "ok":
            gaps.append(_gap("comparison_pool_unavailable", "同行与市场参照", pool_meta.get("reason") or "当前没有可用的其他参照股票"))
    except Exception as exc:
        gaps.append(_gap("comparison_pool_unavailable", "同行与市场参照", exc))
        universe_meta = {"status": "unavailable", "error": str(exc)}
    resolved = profiles[profiles["ts_code"].eq(code)].iloc[0].to_dict()
    histories = {code: history}
    peer_meta: dict[str, Any] = {"status": "not_requested"}
    peer_codes = [value for value in profiles["ts_code"].astype(str) if value != code]
    if peer_codes:
        try:
            peer_histories, peer_meta = context.piliang_lishi(
                peer_codes,
                start_date=history["trade_date"].min().strftime("%Y%m%d"),
                end_date=analysis_date.strftime("%Y%m%d"), minimum_rows=1,
            )
            for peer_code, peer_history in peer_histories.items():
                if peer_code not in peer_codes or peer_history is None or peer_history.empty or "trade_date" not in peer_history:
                    continue
                dates = pd.to_datetime(peer_history["trade_date"], errors="coerce")
                if dates.isna().any() or dates.dt.normalize().duplicated().any():
                    gaps.append(_gap("peer_history_invalid", "同行日线", f"{peer_code} 日线日期无效或重复"))
                    continue
                usable = peer_history.loc[dates.dt.normalize().le(analysis_date.normalize())].copy()
                if not usable.empty and pd.Timestamp(usable["trade_date"].max()).normalize() == analysis_date.normalize():
                    histories[peer_code] = usable
            absent = [peer for peer in peer_codes if peer not in histories]
            if absent:
                gaps.append(_gap("peer_history_incomplete", "同行日线", f"{len(absent)} 只参照股票缺少本次有效日线：{', '.join(absent)}"))
        except Exception as exc:
            peer_meta = {"status": "unavailable", "error": str(exc)}
            gaps.append(_gap("peer_history_unavailable", "同行日线", exc))
    factor: dict[str, Any] = {"status": "unavailable", "groups": {}}
    panel_meta: dict[str, Any] = {}
    try:
        calendar = context.jiaoyi_rili()
    except Exception as exc:
        calendar = None
        gaps.append(_gap("benchmark_calendar_unavailable", "指数交易日历", exc))
    try:
        ready_profiles = profiles[profiles["ts_code"].isin(histories)].copy()
        panel, panel_meta = goujian_fenxi_yinzi_mianban(histories, ready_profiles, source="auto", calendar=calendar)
        if panel.empty:
            raise RuntimeError("本股日线没有形成可用的因子行")
        panel_dates = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
        latest = panel.loc[panel_dates.eq(analysis_date.normalize())].copy()
        latest = zengjia_dangri_guzhi_yinzi(latest, ready_profiles)
        factor_map = huizong_houxuan_yinzi(latest, config=config.get("fenxi") or {})
        factor = factor_map.get(code) or factor
        cross_section = jisuan_hengjiemian_jibenmian(ready_profiles).get(code)
        if cross_section:
            fundamentals = {**fundamentals, "cross_section_analysis": cross_section}
        for warning in panel_meta.get("warnings") or []:
            gaps.append(_gap("factor_source_partial", "市场背景与因子来源", warning))
    except Exception as exc:
        factor = {"status": "unavailable", "groups": {}, "reason": str(exc)}
        gaps.append(_gap("factor_evidence_unavailable", "八组日线指标", exc))

    try:
        pattern = fenxi_zhangting_huimaqiang(
            history, code=code, name=str(resolved.get("name") or name),
            config=config.get("xingtai") or {},
            realtime_quote=snapshot if clock.get("session_status") in {"trading", "midday_break"} and tradability.get("current_quote_verified") else None,
        )
    except Exception as exc:
        pattern = {"status": "unavailable", "reason": str(exc)}
        gaps.append(_gap("pattern_unavailable", "形态证据", exc))
    try:
        late = fenxi_weipan(history, snapshot=snapshot, clock=clock, config=config.get("weipan") or {})
        if panduan_weipan_jieduan(clock, config.get("weipan") or {}) is WeipanJieduan.PANZHONG_ZANDING:
            try:
                minute_data, minute_meta = context.fenzhong_xingqing(code)
                late = fenxi_weipan(
                    history, snapshot=snapshot, clock=clock, config=config.get("weipan") or {},
                    minute_data=minute_data,
                )
                late["minute_data"] = minute_meta
                if minute_meta.get("status") == "unavailable":
                    gaps.append(_gap("minute_data_unavailable", "分钟行情", minute_meta.get("error") or "分钟数据当前不可用"))
            except Exception as exc:
                late["minute_data"] = {"status": "unavailable", "error": str(exc)}
                gaps.append(_gap("minute_data_unavailable", "分钟行情", exc))
    except Exception as exc:
        late = {"status": "unavailable", "reason": str(exc)}
        gaps.append(_gap("late_session_unavailable", "尾盘证据", exc))
    try:
        supplemental = goujian_buchong_zhenduan(
            history, fundamentals, as_of_date=analysis_date.strftime("%Y-%m-%d"),
            minimum_amount_yuan=float((config.get("fenxi") or {}).get("min_amount_yuan", 50_000_000)),
        )
    except Exception as exc:
        supplemental = {"status": "unavailable", "error": str(exc), "blocks": []}
        gaps.append(_gap("supplemental_unavailable", "补充指标", exc))
    return {
        "status": "partial" if gaps else "ok",
        "stock_identity": {"ts_code": code, "name": resolved.get("name"), "industry": resolved.get("industry")},
        "comparison_profile": resolved,
        "factor_analysis": factor,
        "fundamental_analysis": fundamentals,
        "limit_up_pullback_pattern": pattern,
        "late_session_analysis": late,
        "supplemental_diagnostics": supplemental,
        "evidence_gaps": gaps,
        "data_provenance": {
            "comparison_pool": {**universe_meta, **pool_meta, "usable_history_stocks": len(histories), "persistence": "none"},
            "comparison_history": peer_meta, "factor_panel": panel_meta,
        },
    }


__all__ = ["yunxing_dangu_tongyi_lianghua"]
