"""单股复用统一选股量化流水线的编排适配器。"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.ashare.dangu_juece import goujian_dangu_mairu_juece
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
from src.ashare.xuangu_guize import (
    choushu_liudongxing_houxuan,
    guolv_lishi_wanzhengxing,
    hecheng_hengjiemian_yu_shendu_jibenmian,
    hecheng_houxuan_fenshu,
    jichu_ying_guolv,
    jisuan_fengxian_koufen,
)


def _daima(value: Any) -> str | None:
    try:
        return biaozhunhua_gupiao_daima(str(value))
    except (TypeError, ValueError):
        return None


def _mubiao_jilu(
    *,
    code: str,
    name: str,
    industry: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ts_code": code,
        "name": name,
        "industry": industry,
        "latest_price": snapshot.get("latest_price") or snapshot.get("last_price"),
        "open": snapshot.get("open"),
        "high": snapshot.get("high"),
        "low": snapshot.get("low"),
        "previous_close": snapshot.get("previous_close"),
        "pct_chg": snapshot.get("pct_chg") or snapshot.get("pct_change"),
        "volume": snapshot.get("volume"),
        "amount_yuan": snapshot.get("amount_yuan"),
        "turnover_rate": snapshot.get("turnover_rate") or snapshot.get("turnover_rate_pct"),
        "volume_ratio": snapshot.get("volume_ratio"),
        "pe_ttm": snapshot.get("pe_ttm"),
        "pb": snapshot.get("pb"),
        "circulating_market_value_yuan": snapshot.get("circulating_market_value_yuan"),
    }


def _xuanze_bijiao_chi(
    universe: pd.DataFrame,
    *,
    code: str,
    name: str,
    industry: str,
    snapshot: dict[str, Any],
    analysis_date: pd.Timestamp,
    quote_is_completed: bool,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if universe is None or universe.empty or "ts_code" not in universe.columns:
        raise RuntimeError("当前横截面没有可用于八组因子比较的股票")
    data = universe.copy()
    data["ts_code"] = data["ts_code"].map(_daima)
    data = data.dropna(subset=["ts_code"]).drop_duplicates("ts_code", keep="first")
    for column in ("name", "industry"):
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    if "amount_yuan" not in data.columns:
        data["amount_yuan"] = pd.NA
    data["amount_yuan"] = pd.to_numeric(data["amount_yuan"], errors="coerce")

    target = data[data["ts_code"].eq(code)].head(1).copy()
    if target.empty:
        target = pd.DataFrame(
            [_mubiao_jilu(code=code, name=name, industry=industry, snapshot=snapshot)]
        )
        data = pd.concat([data, target], ignore_index=True, sort=False)
    resolved_industry = str(industry or target.iloc[0].get("industry") or "").strip()
    provided_name = str(name or "").strip()
    cross_section_name = str(target.iloc[0].get("name") or "").strip()
    resolved_name = (
        provided_name
        if provided_name and provided_name != code
        else cross_section_name or provided_name or code
    )
    data.loc[data["ts_code"].eq(code), "name"] = resolved_name
    if resolved_industry:
        data.loc[data["ts_code"].eq(code), "industry"] = resolved_industry
    target = data[data["ts_code"].eq(code)].head(1).copy()
    target_blocks = jichu_ying_guolv(
        target.iloc[0],
        analysis_date=analysis_date,
        config=config,
        quote_is_completed=quote_is_completed,
    )

    accepted: list[int] = []
    for index, row in data.loc[~data["ts_code"].eq(code)].iterrows():
        if not jichu_ying_guolv(
            row,
            analysis_date=analysis_date,
            config=config,
            quote_is_completed=quote_is_completed,
        ):
            accepted.append(index)
    eligible = data.loc[accepted].copy()
    settings = config.get("dangu") if isinstance(config.get("dangu"), dict) else {}
    maximum = max(8, int(settings.get("max_peer_stocks", 20)))
    same_industry_limit = max(
        4,
        min(maximum - 1, int(settings.get("same_industry_stocks", 16))),
    )
    selected_parts: list[pd.DataFrame] = []
    selected_codes = {code}
    if resolved_industry:
        same_industry = eligible[eligible["industry"].eq(resolved_industry)].copy()
        same_industry = same_industry.sort_values("amount_yuan", ascending=False, na_position="last")
        same_industry = same_industry.head(same_industry_limit).assign(peer_role="same_industry")
        if not same_industry.empty:
            selected_parts.append(same_industry)
            selected_codes.update(same_industry["ts_code"].astype(str))
    room = maximum - 1 - sum(len(part) for part in selected_parts)
    if room > 0:
        references = eligible[~eligible["ts_code"].isin(selected_codes)]
        references = choushu_liudongxing_houxuan(references, room).assign(
            peer_role="market_reference"
        )
        if not references.empty:
            selected_parts.append(references)
    target["peer_role"] = "target"
    selected = pd.concat([target, *selected_parts], ignore_index=True, sort=False)
    selected = selected.drop_duplicates("ts_code", keep="first").head(maximum).reset_index(drop=True)
    roles = Counter(selected["peer_role"].fillna("market_reference").astype(str))
    return selected, target_blocks, {
        "selected_stocks": int(len(selected)),
        "target_name": resolved_name,
        "target_industry": resolved_industry or None,
        "role_counts": dict(roles),
        "configured_peer_limit": maximum,
        "selection_method": "目标所属行业优先，再用统一流动性抽样补足市场参照",
        "known_bias": "行业标签来自本次远端横截面，只用于当前分析，不冒充历史时点成分",
    }


def yunxing_dangu_tongyi_lianghua(
    *,
    code: str,
    name: str,
    industry: str,
    history: pd.DataFrame,
    analysis_date: pd.Timestamp,
    snapshot: dict[str, Any],
    clock: dict[str, Any],
    technical: dict[str, Any],
    fundamentals: dict[str, Any],
    tradability: dict[str, Any],
    config: dict[str, Any],
    context: FenxiShujuShangxiawen,
) -> dict[str, Any]:
    """按统一选股方法计算单股的八组因子与综合推荐资格。"""

    universe, universe_meta = context.zuixin_hengjiemian()
    quote_is_completed = universe_meta.get("source") == "tushare_daily_cross_section"
    comparison_pool, target_blocks, pool_meta = _xuanze_bijiao_chi(
        universe,
        code=code,
        name=name,
        industry=industry,
        snapshot=snapshot,
        analysis_date=analysis_date,
        quote_is_completed=quote_is_completed,
        config=config,
    )
    target_profile = comparison_pool[comparison_pool["ts_code"].astype(str).eq(code)].iloc[0]
    resolved_name = str(target_profile.get("name") or name or code)
    resolved_industry = str(target_profile.get("industry") or industry or "")
    fenxi_settings = config.get("fenxi") if isinstance(config.get("fenxi"), dict) else {}
    start_date = (
        analysis_date - pd.Timedelta(days=int(fenxi_settings.get("history_calendar_days", 420)))
    ).strftime("%Y%m%d")
    end_date = analysis_date.strftime("%Y%m%d")
    peer_codes = [value for value in comparison_pool["ts_code"].astype(str) if value != code]
    if not peer_codes:
        raise RuntimeError("当前横截面没有可用的同行或市场参照股票")
    peer_histories, peer_history_meta = context.piliang_lishi(
        peer_codes,
        start_date=start_date,
        end_date=end_date,
        minimum_rows=int(fenxi_settings.get("minimum_history_rows", 80)),
    )
    target_history = history.copy()
    target_dates = pd.to_datetime(target_history["trade_date"], errors="coerce").dt.normalize()
    target_history = target_history[
        target_dates.ge(pd.to_datetime(start_date, format="%Y%m%d"))
    ].copy()
    peer_histories, history_rejected = guolv_lishi_wanzhengxing(
        peer_histories,
        analysis_date=analysis_date,
        minimum_rows=int(fenxi_settings.get("minimum_history_rows", 80)),
        minimum_amount=float(fenxi_settings.get("min_amount_yuan", 50_000_000)),
        minimum_listing_calendar_days=int(
            fenxi_settings.get("minimum_listing_calendar_days", 180)
        ),
    )
    histories = {code: target_history, **peer_histories}
    minimum_peers = int((config.get("dangu") or {}).get("minimum_peer_stocks", 8))
    if len(histories) < minimum_peers:
        raise RuntimeError(
            f"统一八组因子比较池只有 {len(histories)} 只有效股票，少于 {minimum_peers} 只"
        )
    ready_profiles = comparison_pool[
        comparison_pool["ts_code"].astype(str).isin(histories)
    ].copy()
    panel, panel_meta = goujian_fenxi_yinzi_mianban(
        histories,
        ready_profiles,
        source="auto",
    )
    if panel.empty:
        raise RuntimeError("统一八组日 K 因子面板为空")
    panel_dates = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    latest_panel = panel.loc[panel_dates.eq(analysis_date.normalize())].copy()
    latest_panel = zengjia_dangri_guzhi_yinzi(latest_panel, ready_profiles)
    factor_map = huizong_houxuan_yinzi(latest_panel, config=fenxi_settings)
    factor = factor_map.get(code)
    if not isinstance(factor, dict) or factor.get("status") != "ok":
        raise RuntimeError("目标股票没有形成可用的八组日 K 因子综合分")

    fundamental_map = jisuan_hengjiemian_jibenmian(ready_profiles)
    fundamental = hecheng_hengjiemian_yu_shendu_jibenmian(
        fundamental_map.get(
            code,
            {"status": "unavailable", "score_0_100": None, "confidence": 0.0, "evidence": []},
        ),
        fundamentals,
        deep_weight=float(fenxi_settings.get("fundamental_deep_weight", 0.65)),
    )
    pattern = fenxi_zhangting_huimaqiang(
        history,
        code=code,
        name=resolved_name,
        config=config.get("xingtai") or {},
        realtime_quote=(
            snapshot
            if str(clock.get("session_status")) in {"trading", "midday_break"}
            else None
        ),
    )
    late_stage = panduan_weipan_jieduan(clock, config.get("weipan") or {})
    late = fenxi_weipan(
        history,
        snapshot=snapshot,
        clock=clock,
        config=config.get("weipan") or {},
    )
    if late_stage is WeipanJieduan.PANZHONG_ZANDING:
        minute_data, minute_meta = context.fenzhong_xingqing(code)
        late = fenxi_weipan(
            history,
            snapshot=snapshot,
            clock=clock,
            config=config.get("weipan") or {},
            minute_data=minute_data,
        )
        late["minute_data"] = minute_meta

    decision_tradability = dict(tradability)
    hard_blocks = list(dict.fromkeys([*(tradability.get("hard_blocks") or []), *target_blocks]))
    decision_tradability["hard_blocks"] = hard_blocks
    if hard_blocks:
        decision_tradability["status"] = "blocked"
        decision_tradability["basic_execution_feasible"] = False
    penalties, risks = jisuan_fengxian_koufen(
        code=code,
        name=resolved_name,
        snapshot=snapshot,
        factor=factor,
        pattern=pattern,
        late=late,
        config=config,
        technical=technical,
    )
    if not (fundamental.get("financials") or {}):
        risks.append("深度财务指标不可用，基本面证据不完整")
    ranking = hecheng_houxuan_fenshu(
        factor=factor,
        fundamental=fundamental,
        pattern=pattern,
        late=late,
        penalties=penalties,
        config=config,
    )
    item = {
        "ts_code": code,
        "name": resolved_name,
        "industry": resolved_industry,
        "factor": factor,
        "fundamental": fundamental,
        "pattern": pattern,
        "late": late,
        "ranking": ranking,
        "technical": technical,
        "tradability": decision_tradability,
        "risks": list(dict.fromkeys(risks)),
        "data_quality": {
            "as_of": analysis_date.strftime("%Y-%m-%d"),
            "comparison_stocks": int(len(histories)),
            "history_source": peer_history_meta.get("source"),
        },
    }
    decision = goujian_dangu_mairu_juece(item, config=config)
    return {
        "status": "ok",
        "stock_identity": {
            "ts_code": code,
            "name": resolved_name,
            "industry": resolved_industry or None,
        },
        "buy_decision": decision,
        "factor_analysis": factor,
        "fundamental_analysis": fundamental,
        "limit_up_pullback_pattern": pattern,
        "late_session_analysis": late,
        "ranking_details": ranking,
        "tradability": decision_tradability,
        "risks": list(dict.fromkeys([*risks, *(decision.get("risks") or [])])),
        "data_provenance": {
            "comparison_pool": {
                **pool_meta,
                "source": universe_meta.get("source"),
                "as_of": universe_meta.get("as_of"),
                "warnings": universe_meta.get("warnings") or [],
                "usable_history_stocks": int(len(histories)),
                "history_rejected": history_rejected[:20],
                "persistence": "none",
            },
            "comparison_history": peer_history_meta,
            "factor_panel": panel_meta,
        },
    }


__all__ = ["yunxing_dangu_tongyi_lianghua"]
