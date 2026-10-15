"""统一分析范围值对象与候选池策略。

自然语言命名范围先由实时目录解析为 ``ShichangFanwei``，候选池策略不再让大模型或业务
编排猜测“行业/概念”。整个主板由来源交易板分类与本次 A 股身份交叉核验；各范围共用
后续过滤、因子和排序流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

import pandas as pd

from src.ashare.dongcai_api import DONGCAI_PUBLIC_TOKEN, dongcai_fenye_duqu
from src.ashare.fanwei_faxian import BankuaiLeixing, ShichangFanwei
from src.ashare.shichang_shuju import FenxiShujuShangxiawen
from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu


class FanweiLeixing(str, Enum):
    QUAN_SHICHANG = "all_market"
    ZHUBAN = "main_board"
    MINGMING_FANWEI = "named_scope"
    DANGU_GUPIAO = "single_stock"


@dataclass(frozen=True)
class FenxiFanwei:
    """用户请求中的范围；这里不包含任何专业分类猜测。"""

    leixing: FanweiLeixing
    mingcheng: str | None = None
    gupiao: str | None = None

    @classmethod
    def create(
        cls,
        leixing: str,
        mingcheng: str | None,
        gupiao: str | None = None,
    ) -> "FenxiFanwei":
        key = str(leixing or "all_market").strip().lower()
        all_aliases = {"all", "all_market", "quan_shichang", "全市场"}
        single_aliases = {
            "single_stock",
            "single",
            "stock",
            "dangu",
            "单股",
            "个股",
        }
        # 旧会话可能仍携带 industry/board；只视为“用户给了名称”，不据此约束目录类型。
        named_aliases = {
            "named_scope",
            "named",
            "industry",
            "hangye",
            "行业",
            "board",
            "concept",
            "bankuai",
            "板块",
            "概念",
        }
        if key in all_aliases and gupiao:
            query = " ".join(str(gupiao).split()) or None
            if not query:
                raise ValueError("按单只股票分析时，请提供股票名称或 6 位股票代码")
            return cls(leixing=FanweiLeixing.DANGU_GUPIAO, gupiao=query)
        if key in all_aliases:
            return cls(leixing=FanweiLeixing.QUAN_SHICHANG)
        if key == FanweiLeixing.ZHUBAN.value:
            if gupiao:
                raise ValueError("整个沪深主板范围不能同时指定单只股票")
            return cls(leixing=FanweiLeixing.ZHUBAN, mingcheng=" ".join(str(mingcheng or "").split()) or None)
        if key in single_aliases:
            query = " ".join(str(gupiao or mingcheng or "").split()) or None
            if not query:
                raise ValueError("按单只股票分析时，请提供股票名称或 6 位股票代码")
            return cls(leixing=FanweiLeixing.DANGU_GUPIAO, gupiao=query)
        if key not in named_aliases:
            raise ValueError("分析范围只能是全市场、整个沪深主板、单股或一个用日常语言描述的股票范围")
        name = " ".join(str(mingcheng or "").split()) or None
        if not name:
            raise ValueError("按特定范围分析时，请提供一个日常名称")
        return cls(leixing=FanweiLeixing.MINGMING_FANWEI, mingcheng=name)


@dataclass(frozen=True)
class YijiexiFenxiFanwei:
    """候选池唯一接受的范围对象。"""

    request: FenxiFanwei
    market_scope: ShichangFanwei | None = None

    def __post_init__(self) -> None:
        named = self.request.leixing is FanweiLeixing.MINGMING_FANWEI
        if named != (self.market_scope is not None):
            raise ValueError("命名范围必须先完成实时目录解析")


@dataclass(frozen=True)
class HouXuanChiJieguo:
    data: pd.DataFrame
    metadata: dict[str, Any]


class HouXuanChiCelue(Protocol):
    """候选池策略；范围来源不同，后续分析流程保持统一。"""

    def goujian(
        self,
        fanwei: YijiexiFenxiFanwei,
        context: FenxiShujuShangxiawen,
    ) -> HouXuanChiJieguo:
        ...


class QuanShichangHouXuanChi:
    def goujian(
        self,
        fanwei: YijiexiFenxiFanwei,
        context: FenxiShujuShangxiawen,
    ) -> HouXuanChiJieguo:
        del fanwei
        data, metadata = context.zuixin_hengjiemian()
        return HouXuanChiJieguo(
            data=data,
            metadata={
                **metadata,
                "scope_type": FanweiLeixing.QUAN_SHICHANG.value,
                "scope_name": "沪深京 A 股",
                "ambiguity_resolution": "not_applicable",
            },
        )


def _jiaoyiban(value: Any) -> str | None:
    """只接受来源明确的交易板标签，交易所后缀和未知文本不是分类。"""
    text = str(value).strip() if pd.notna(value) else ""
    return text if text in {"主板", "创业板", "科创板", "北交所", "CDR"} else None


def _hedui_zhuban_fenlei(
    profiles: pd.DataFrame, classifications: dict[str, tuple[str, bool]], known: dict[str, bool],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """各来源共用身份与反证核验；已经确认的主源分类不由备用覆盖。"""
    verified = dict(known)
    indexed = profiles.set_index("ts_code", drop=False)
    duplicated = set(profiles.loc[profiles["ts_code"].duplicated(keep=False), "ts_code"])
    gaps = []
    for code, (name, is_main) in classifications.items():
        if code in verified:
            continue
        reason = ""
        if code not in indexed.index:
            reason = "本次 A 股横截面未匹配来源证券，身份未核验"
        elif code in duplicated:
            reason = "本次 A 股横截面同一证券存在重复身份"
        else:
            row = indexed.loc[code]
            observed = str(row["name"]).strip() if pd.notna(row["name"]) else ""
            board = _jiaoyiban(row.get("market"))
            if not name or name != observed:
                reason = "分类来源名称与本次 A 股身份名称不一致或缺失"
            elif is_main and not code.endswith((".SH", ".SZ")):
                reason = "来源主板分类与本次已核验的交易所身份冲突"
            elif board is not None and (board == "主板") != is_main:
                reason = f"本次横截面明确{board}分类与备用来源冲突，不能覆盖"
            else:
                verified[code] = is_main
        if reason:
            gaps.append({"ts_code": code, "source_name": name, "reason": reason})
    return verified, gaps


def _clist_zhuban_fenlei() -> tuple[dict[str, tuple[str, bool]], dict[str, Any]]:
    source = "eastmoney_main_board_members"
    with GongkaiShujuHTTPKehu(source) as client:
        rows, endpoint, reported = dongcai_fenye_duqu(client, base_params={
            "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": "f12",
            "ut": DONGCAI_PUBLIC_TOKEN, "fs": "m:0 t:6,m:1 t:2", "fields": "f12,f13,f14,f124",
        })
    if len(rows) != reported:
        raise ValueError("主板名单分页数量不一致")
    classifications = {}
    for row in rows:
        market, digits, name = row.get("f13"), str(row.get("f12") or "").strip(), str(row.get("f14") or "").strip()
        if isinstance(market, bool) or str(market) not in {"0", "1"} or len(digits) != 6 or not digits.isdigit() or not name:
            raise ValueError("主板名单缺少有效沪深交易所标识、代码或名称")
        code = f"{digits}.{'SH' if str(market) == '1' else 'SZ'}"
        if code in classifications:
            raise ValueError("主板名单证券身份重复")
        classifications[code] = (name, True)
    return classifications, {"source": source, "endpoint": endpoint, "reported_rows": reported, "received_rows": len(rows),
                             "pagination_complete": True, "classification_filter": "m:0 t:6,m:1 t:2",
                             "attempted_providers": [item.to_dict() for item in client.attempts]}


class ZhubanHouXuanChi:
    """优先本轮可信目录分类，备用只补未知，候选仍需真实 A 股身份核验。"""

    def goujian(self, fanwei: YijiexiFenxiFanwei, context: FenxiShujuShangxiawen) -> HouXuanChiJieguo:
        data, cross_meta = context.zuixin_hengjiemian()
        fetched_at = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        scope_meta = {**cross_meta, "scope_type": FanweiLeixing.ZHUBAN.value, "scope_name": "整个沪深主板 A 股",
                      "requested_name": fanwei.request.mingcheng, "ambiguity_resolution": "explicit_trading_board",
                      "cross_section_rows": len(data), "cross_section_status": cross_meta.get("status"), "scope_fetched_at": fetched_at}
        if data.empty or not {"ts_code", "name"}.issubset(data.columns):
            return HouXuanChiJieguo(pd.DataFrame(), {**scope_meta, "status": "unavailable", "error_code": "main_board_identity_unavailable",
                "error": "本次 A 股横截面缺少可核验的代码和名称，不能构建主板候选池", "retryable": True})
        profiles = data.copy()
        profiles["ts_code"] = profiles["ts_code"].astype(str).str.strip().str.upper()
        known, gaps, failures, sources = {}, {}, [], []
        basic_meta = cross_meta.get("stock_basic_provenance") or {}
        trusted_basic = basic_meta.get("source") == "tushare_live" and (basic_meta.get("security_scope") or {}).get("identity_source") == "tushare_stock_basic"
        if trusted_basic and "market" in profiles:
            primary = {row["ts_code"]: (str(row["name"]).strip(), _jiaoyiban(row["market"]) == "主板")
                       for row in profiles.to_dict("records") if _jiaoyiban(row["market"]) is not None}
            known, missing = _hedui_zhuban_fenlei(profiles, primary, known)
            gaps.update((item["ts_code"], item) for item in missing)
            sources.append({"source": "tushare_stock_basic_market", "received_rows": len(profiles),
                            "classified_rows": len(known), "reported_rows": sum(known.values()),
                            "classification_basis": "本次已上市 A 股身份目录的明确 market=主板；非主板分类不由备用覆盖",
                            "pagination_complete": True})
        else:
            failures.append({"source": "tushare_stock_basic_market", "status": "unavailable",
                             "reason": "本次未取得来源已核验的 Tushare 交易板分类，交易所后缀不能替代主板分类"})
        pending = set(profiles["ts_code"]) - set(known)
        if pending:
            try:
                classifications, member_meta = _clist_zhuban_fenlei()
                known, missing = _hedui_zhuban_fenlei(profiles, classifications, known)
                gaps.update((item["ts_code"], item) for item in missing if item["ts_code"] not in known)
                sources.append(member_meta)
            except (RuntimeError, ValueError, TypeError) as exc:
                failures.append({"source": "eastmoney_main_board_members", "status": "unavailable",
                    "reason": " ".join(str(exc).split())[:240], "attempted_providers": [item.to_dict() for item in getattr(exc, "attempts", ())]})
        outside_membership = profiles[~profiles["ts_code"].isin(known)]
        if trusted_basic:
            for row in outside_membership.to_dict("records"):
                gaps.setdefault(row["ts_code"], {"ts_code": row["ts_code"], "source_name": row.get("name"),
                                                 "cross_section_market": str(row.get("market")).strip() if pd.notna(row.get("market")) else None,
                                                 "reason": "本次交易板分类或身份尚未核验，不推测属于或不属于主板"})
        unverified = [item for code, item in gaps.items() if code not in known]
        main_codes = {code for code, is_main in known.items() if is_main}
        selected = profiles[profiles["ts_code"].isin(main_codes)].reset_index(drop=True)
        warnings = list(cross_meta.get("warnings") or [])
        if unverified:
            warnings.append(f"{len(unverified)} 条证券的主板分类或身份未核验，保留为证据缺口，不视为已比较")
        if failures and sources:
            warnings.append("部分分类来源失败，已使用本次可核验来源恢复范围；失败记录保留，不把已恢复分类继续列为不可用")
        source = "+".join(item["source"] for item in sources) or "eastmoney_main_board_members"
        membership_meta = {"source": source, "fetched_at": fetched_at, "sources": sources, "source_failures": failures,
                           "reported_rows": sources[0].get("reported_rows") if len(sources) == 1 else None, "received_rows": len(main_codes),
                           "matched_rows": len(selected), "unverified_rows": len(unverified), "cross_check_complete": not unverified,
                           "complete_exchange_listing_reconciled": False, "pagination_complete": bool(sources),
                           "directory_outside_membership_rows": len(outside_membership),
                           "directory_outside_membership_definition": "目录中未获本次分类来源确认的记录；公开目录不在主板来源正名单的记录不自动计作主板缺口，也不据此确认其上市板身份或声称主板上市名单完整",
                           "coverage_basis": "本次真实 A 股身份与可信目录交易板分类，缺失时使用来源明确主板类别的完整分页成员；未与完整交易所上市名单独立对账",
                           "unverified_members": unverified, "persistence": "none"}
        return HouXuanChiJieguo(selected, {**scope_meta, "scope_source": source,
            "membership_classification_basis": [item.get("classification_basis") or item.get("classification_filter") for item in sources],
            "status": "unavailable" if selected.empty else "partial" if unverified or cross_meta.get("status") != "ok" else "ok",
            "scope_member_rows": len(main_codes), "scope_verified_rows": len(selected), "scope_unverified_rows": len(unverified),
            "membership_provenance": membership_meta, "warnings": warnings,
            **({"error_code": "main_board_members_unavailable", "error": "当前主板分类未能形成已核验候选；" + "；".join(item["reason"] for item in failures),
                "retryable": True} if selected.empty else {})})


class MingmingFanweiHouXuanChi:
    def goujian(
        self,
        fanwei: YijiexiFenxiFanwei,
        context: FenxiShujuShangxiawen,
    ) -> HouXuanChiJieguo:
        scope = fanwei.market_scope
        if scope is None:
            raise ValueError("命名范围尚未解析")
        members, board_meta = context.bankuai_chengfen(scope)
        # 板块成分接口已经包含本范围的实时行情、估值和市值。这里不再为了补一个
        # 当前股票资料字段而下载全市场横截面；未返回上市日期时保留缺失，
        # 不用已下载日线跨度冒充上市时长；深度基本面只对最终少量候选按需读取。
        data = members.copy()
        industry = data.get("industry", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
        missing_industry = industry.str.lower().isin({"", "unknown", "nan", "none", "未知", "未分类"})
        # 只有已经核验的行业范围能补充行业；概念成员不必属于同一行业。
        verified_industry = scope.kind is BankuaiLeixing.HANGYE and scope.verification.get("verified") is True
        data["industry"] = industry.mask(missing_industry, scope.canonical_name if verified_industry else "")
        missing_count = int(data["industry"].eq("").sum())
        cross_meta: dict[str, Any] = {
            "cross_section_enrichment_status": "not_requested",
            "industry_missing_rows": missing_count,
            "industry_basis": "verified_industry_scope_or_member_profile" if verified_industry else "member_profile_only",
            "warnings": [f"{missing_count} 只成分缺少真实行业，保留范围比较，不以概念名称冒充同行"] if missing_count else [],
            "cross_section_enrichment_reason": "命名范围使用实时成分横截面，避免无关的全市场重复请求",
        }
        return HouXuanChiJieguo(
            data=data.reset_index(drop=True),
            metadata={
                **cross_meta,
                **scope.to_dict(),
                "status": "ok",
                "scope_type": FanweiLeixing.MINGMING_FANWEI.value,
                "scope_name": scope.canonical_name,
                "requested_name": scope.requested_name,
                "constituent_source": board_meta.get("constituent_source"),
                "constituent_fetched_at": board_meta.get("fetched_at"),
                "constituent_rows": board_meta.get("rows_received"),
                "warnings": list(cross_meta.get("warnings", []))
                + list(board_meta.get("warnings", [])),
                "constituent_attempted_providers": list(
                    board_meta.get("attempted_providers", [])
                ),
            },
        )


def huoqu_houxuanchi_celue(fanwei: YijiexiFenxiFanwei) -> HouXuanChiCelue:
    strategies: dict[FanweiLeixing, HouXuanChiCelue] = {
        FanweiLeixing.QUAN_SHICHANG: QuanShichangHouXuanChi(),
        FanweiLeixing.ZHUBAN: ZhubanHouXuanChi(),
        FanweiLeixing.MINGMING_FANWEI: MingmingFanweiHouXuanChi(),
    }
    return strategies[fanwei.request.leixing]


__all__ = [
    "FanweiLeixing",
    "FenxiFanwei",
    "HouXuanChiCelue",
    "HouXuanChiJieguo",
    "YijiexiFenxiFanwei",
    "huoqu_houxuanchi_celue",
]
