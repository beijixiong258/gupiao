"""统一分析范围值对象与候选池策略。

自然语言范围先由实时目录解析为 ``ShichangFanwei``，候选池策略不再让大模型或业务
编排猜测“行业/概念”。全市场与已解析范围共用后续过滤、因子和排序流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import pandas as pd

from src.ashare.fanwei_faxian import ShichangFanwei
from src.ashare.shichang_shuju import FenxiShujuShangxiawen


class FanweiLeixing(str, Enum):
    QUAN_SHICHANG = "all_market"
    MINGMING_FANWEI = "named_scope"


@dataclass(frozen=True)
class FenxiFanwei:
    """用户请求中的范围；这里不包含任何专业分类猜测。"""

    leixing: FanweiLeixing
    mingcheng: str | None = None

    @classmethod
    def create(cls, leixing: str, mingcheng: str | None) -> "FenxiFanwei":
        key = str(leixing or "all_market").strip().lower()
        all_aliases = {"all", "all_market", "quan_shichang", "全市场"}
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
        if key in all_aliases:
            return cls(leixing=FanweiLeixing.QUAN_SHICHANG)
        if key not in named_aliases:
            raise ValueError("分析范围只能是全市场或一个用日常语言描述的股票范围")
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
        # 当前股票资料字段而下载全市场横截面；上市时间由随后取得的真实日线跨度
        # 复核，深度基本面只对最终少量候选按需读取。
        data = members.copy()
        if "industry" not in data.columns:
            data["industry"] = scope.canonical_name
        else:
            data["industry"] = data["industry"].fillna(scope.canonical_name)
        cross_meta: dict[str, Any] = {
            "cross_section_enrichment_status": "not_requested",
            "warnings": [],
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
