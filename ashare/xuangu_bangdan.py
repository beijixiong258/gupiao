"""固定日终双榜候选池；榜单只决定研究范围，不参与最终评分。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LEADERBOARD_SIZE = 100
LEADERBOARD_SCOPE = "daily_leaderboards"
LEADERBOARD_NAME = "日终涨幅前100与成交额前100合并去重"


def goujian_shuangbang_chi(
    data: pd.DataFrame, metadata: dict[str, Any], as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """仅接受已核验 A 股横截面的同日正式日线；缺失保留，不补榜外股票。"""
    target = pd.Timestamp(as_of).normalize()
    meta = {
        "source": metadata.get("source"), "status": metadata.get("status"),
        "identity_directory_coverage_verified": metadata.get("identity_directory_coverage_verified"),
        "cross_section_provenance": dict(metadata),
        "scope_type": LEADERBOARD_SCOPE, "scope_name": LEADERBOARD_NAME,
        "as_of": target.strftime("%Y-%m-%d"), "cross_section_rows": len(data),
        "leaderboard_size": LEADERBOARD_SIZE, "leaderboard_rules_fixed": True,
        "leaderboard_rule": "同一完整交易日涨幅降序前100与成交额降序前100，合并去重；数据或条件不合格后不从榜外补位",
        "leaderboard_tie_policy": "榜单原值精确并列时按证券代码升序固定入池边界；不表示代码靠前者更优，不改变最终预计涨幅的并列关系",
        "historical_pool_rule": "只用本次双榜候选中通过必要数据与资格核验的股票历史；不加入榜外股票，不重建历史每日榜单，保留当前资格及事后入榜偏差",
        "ambiguity_resolution": "fixed_daily_leaderboards", "persistence": "none",
    }
    required = {"ts_code", "name", "trade_date", "pct_chg", "amount_yuan"}
    if (metadata.get("source") != "tushare_daily_cross_section"
            or pd.to_datetime(metadata.get("as_of"), errors="coerce") != target
            or not required.issubset(data.columns) or data.empty):
        return pd.DataFrame(), {**meta, "status": "unavailable", "error_code": "leaderboard_daily_data_unavailable",
            "error": "双榜需要最近完整交易日的正式A股日线横截面；实时快照、过期日期或缺少榜单字段不能替代", "retryable": True}
    frame = data.copy()
    frame["ts_code"] = frame["ts_code"].astype("string").str.strip().str.upper()
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    # A 股身份由横截面入口核验；这里继续拒绝未匹配名称、空代码及重复身份。
    identity = (frame["ts_code"].notna() & frame["ts_code"].ne("")
                & frame["name"].fillna("").astype(str).str.strip().ne("")
                & ~frame["ts_code"].duplicated(keep=False))
    common = identity & dates.eq(target)
    boards: dict[str, pd.DataFrame] = {}
    incomplete: dict[str, list[str]] = {}
    for label, metric in (("gain", "pct_chg"), ("turnover", "amount_yuan")):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        valid = common & np.isfinite(frame[metric])
        if metric == "amount_yuan":
            valid &= frame[metric].ge(0)
        incomplete[label] = frame.loc[~valid, "ts_code"].fillna("未知代码").astype(str).tolist()
        boards[label] = frame.loc[valid].sort_values([metric, "ts_code"], ascending=[False, True], kind="stable").head(LEADERBOARD_SIZE)
    gains, turnover = boards["gain"], boards["turnover"]
    gain_ranks = {code: rank for rank, code in enumerate(gains["ts_code"], 1)}
    turnover_ranks = {code: rank for rank, code in enumerate(turnover["ts_code"], 1)}
    union = set(gain_ranks) | set(turnover_ranks)
    selected = frame.loc[frame["ts_code"].isin(union)].sort_values("ts_code", kind="stable").reset_index(drop=True)
    warnings = [f"入榜前的日终横截面：{warning}" for warning in metadata.get("warnings") or []]
    if any(incomplete.values()):
        warnings.append("部分记录缺少同日身份或榜单数值；双榜仅覆盖本次可核验记录，不能声称完整市场排名")
    if len(gains) < LEADERBOARD_SIZE or len(turnover) < LEADERBOARD_SIZE:
        warnings.append("至少一张榜不足100条有效记录，保留实际数量，不使用旧数据或榜外股票补足")
    members = [{"ts_code": row["ts_code"], "name": row["name"],
                "gain_rank": gain_ranks.get(row["ts_code"]), "turnover_rank": turnover_ranks.get(row["ts_code"])}
               for row in selected.to_dict("records")]
    unverified = set(incomplete["gain"]) | set(incomplete["turnover"])
    outside = unverified - union
    return selected, {**meta,
        "status": "unavailable" if selected.empty else "partial" if warnings or metadata.get("status") != "ok" else "ok",
        "gain_leaderboard_count": len(gains), "turnover_leaderboard_count": len(turnover),
        "leaderboard_overlap_count": len(set(gain_ranks) & set(turnover_ranks)),
        "leaderboard_union_count": len(selected), "leaderboard_members": members,
        "leaderboard_unverified_codes": incomplete, "warnings": warnings,
        "scope_member_rows": len(selected), "scope_verified_rows": len(selected),
        "leaderboard_input_unverified_rows": len(unverified), "leaderboard_unranked_count": len(outside),
        "leaderboard_partial_rank_member_count": len(unverified & union),
        "leaderboard_input_note": f"入榜前横截面有{len(unverified)}只股票的榜单输入未核验，其中{len(outside)}只未进入{len(selected)}只入榜名单，另{len(unverified & union)}只凭另一张有效榜入池。此处缺口不是入榜后历史取数失败，后者见实际比较覆盖。",
    }
