"""自然语言分析范围发现与东方财富实时目录适配。

大模型只负责抽取用户说出的范围名称；本模块实时读取行业和概念目录，确定唯一范围，
或返回一次可展示的通俗澄清。候选池只接收已经解析且带来源的范围对象。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Iterable

import pandas as pd

from src.ashare.dongcai_api import DONGCAI_PUBLIC_TOKEN, dongcai_fenye_duqu
from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu, WangluoQingqiuYichang


_BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
def _beijing_now_text() -> str:
    return datetime.now(_BEIJING_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


class BankuaiLeixing(str, Enum):
    HANGYE = "industry"
    GAINIAN = "concept"

    @property
    def yonghu_miaoshu(self) -> str:
        return "数据源行业范围" if self is BankuaiLeixing.HANGYE else "数据源概念范围，通常更广"


@dataclass(frozen=True)
class ShichangFanwei:
    """经实时目录确认后的范围值对象。"""

    requested_name: str
    canonical_name: str
    code: str
    kind: BankuaiLeixing
    source: str
    fetched_at: str
    match_score: float
    match_basis: str
    ambiguity_resolution: str
    catalog_counts: dict[str, int] = field(default_factory=dict)
    attempted_providers: tuple[dict[str, Any], ...] = ()
    verification: dict[str, Any] = field(default_factory=dict)

    @property
    def user_label(self) -> str:
        return f"{self.canonical_name}（{self.kind.yonghu_miaoshu}）"

    def to_dict(self, *, include_internal_code: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_name": self.requested_name,
            "canonical_name": self.canonical_name,
            "scope_kind": self.kind.value,
            "scope_kind_display": self.kind.yonghu_miaoshu,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "match_score": round(float(self.match_score), 4),
            "match_basis": self.match_basis,
            "ambiguity_resolution": self.ambiguity_resolution,
            "catalog_counts": dict(self.catalog_counts),
            "attempted_providers": [dict(value) for value in self.attempted_providers],
            "verification": dict(self.verification),
        }
        if include_internal_code:
            payload["canonical_code"] = self.code
        return payload


@dataclass(frozen=True)
class FanweiFaxianJieguo:
    status: str
    scope: ShichangFanwei | None = None
    candidates: tuple[ShichangFanwei, ...] = ()
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False
    attempted_providers: tuple[dict[str, Any], ...] = ()
    catalog_source: str | None = None
    fetched_at: str | None = None
    next_action: str | None = None

    def to_result(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "outcome": (
                "scope_resolved"
                if self.status == "resolved"
                else "clarification_required"
                if self.status == "clarification_required"
                else "data_unavailable"
            ),
            "stage": "scope_discovery",
            "source": self.catalog_source,
            "fetched_at": self.fetched_at,
            "retryable": self.retryable,
            "attempted_providers": [dict(value) for value in self.attempted_providers],
            "next_action": self.next_action,
        }
        if self.scope is not None:
            payload["scope"] = self.scope.to_dict()
        if self.candidates:
            payload["candidates"] = [
                {
                    **candidate.to_dict(),
                    "user_label": candidate.user_label,
                    "user_response": f"按数据源当前的“{candidate.canonical_name}”范围分析",
                }
                for candidate in self.candidates
            ]
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.message:
            payload["error"] = self.message
        return payload


class DongcaiFanweiShujuYuan:
    """东方财富行业/概念目录及成分的实时适配器。"""

    source_name = "eastmoney_live_board_catalog"

    def huoqu_mulu(self) -> tuple[list[ShichangFanwei], dict[str, Any]]:
        fetched_at = _beijing_now_text()
        scopes: list[ShichangFanwei] = []
        counts: dict[str, int] = {}
        endpoints: dict[str, str] = {}
        with GongkaiShujuHTTPKehu(self.source_name) as client:
            for kind, fs in (
                (BankuaiLeixing.HANGYE, "m:90 t:2 f:!50"),
                (BankuaiLeixing.GAINIAN, "m:90 t:3 f:!50"),
            ):
                rows, endpoint, _ = dongcai_fenye_duqu(
                    client,
                    base_params={
                        "po": 1,
                        "np": 1,
                        "fltt": 2,
                        "invt": 2,
                        "fid": "f12",
                        "ut": DONGCAI_PUBLIC_TOKEN,
                        "fs": fs,
                        "fields": "f12,f14",
                    },
                )
                counts[kind.value] = len(rows)
                endpoints[kind.value] = endpoint
                for row in rows:
                    code = str(row.get("f12") or "").strip().upper()
                    name = str(row.get("f14") or "").strip()
                    if not code.startswith("BK") or not name:
                        continue
                    scopes.append(
                        ShichangFanwei(
                            requested_name="",
                            canonical_name=name,
                            code=code,
                            kind=kind,
                            source=self.source_name,
                            fetched_at=fetched_at,
                            match_score=0.0,
                            match_basis="catalog_entry",
                            ambiguity_resolution="pending",
                        )
                    )
            attempts = tuple(item.to_dict() for item in client.attempts)
        return scopes, {
            "source": self.source_name,
            "fetched_at": fetched_at,
            "catalog_counts": counts,
            "endpoints": endpoints,
            "attempted_providers": attempts,
            "persistence": "none",
        }

    def huoqu_chengfen(
        self,
        scope: ShichangFanwei,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        fetched_at = _beijing_now_text()
        with GongkaiShujuHTTPKehu("eastmoney_live_board_constituents") as client:
            rows, endpoint, _ = dongcai_fenye_duqu(
                client,
                base_params={
                    "po": 1,
                    "np": 1,
                    "fltt": 2,
                    "invt": 2,
                    "fid": "f3",
                    "ut": "8dec03ba335b81bf4ebdf7b29ec27d15",
                    "fs": f"b:{scope.code} f:!50",
                    "fields": (
                        "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,"
                        "f15,f16,f17,f18,f20,f21,f23"
                    ),
                },
            )
            attempts = tuple(item.to_dict() for item in client.attempts)
        data = pd.DataFrame(rows)
        rename = {
            "f12": "代码",
            "f14": "名称",
            "f2": "最新价",
            "f3": "涨跌幅",
            "f4": "涨跌额",
            "f5": "成交量",
            "f6": "成交额",
            "f7": "振幅",
            "f8": "换手率",
            "f9": "市盈率-动态",
            "f10": "量比",
            "f15": "最高",
            "f16": "最低",
            "f17": "今开",
            "f18": "昨收",
            "f20": "总市值",
            "f21": "流通市值",
            "f23": "市净率",
        }
        data = data.rename(columns=rename)
        return data, {
            **scope.to_dict(),
            "status": "ok",
            "constituent_source": "eastmoney_live_board_constituents",
            "fetched_at": fetched_at,
            "endpoint_host": re.sub(r"^https?://([^/]+).*$", r"\1", endpoint),
            "rows_received": int(len(data)),
            "attempted_providers": list(attempts),
            "persistence": "none",
        }

    def hedui_fanwei(self, scope: ShichangFanwei) -> dict[str, Any]:
        """用官方板块详情页复核目录返回的名称、代码和类别。"""
        endpoint = f"https://data.eastmoney.com/bkzj/{scope.code}.html"
        checked_at = _beijing_now_text()
        with GongkaiShujuHTTPKehu("eastmoney_board_detail_verification") as client:
            text, _ = client.qingqiu_wenben(
                (endpoint,),
                headers={"Referer": "https://data.eastmoney.com/"},
            )
            attempts = [item.to_dict() for item in client.attempts]
        match = re.search(r"var\s+bkInfo\s*=\s*(\{.*?\})\s*;", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("东方财富板块详情页缺少 bkInfo")
        detail = json.loads(match.group(1))
        observed_code = str(detail.get("code") or "").strip().upper()
        observed_name = str(detail.get("name") or "").strip()
        type_code = str(detail.get("typecode") or "").strip().lower()
        observed_kind = (
            BankuaiLeixing.HANGYE
            if type_code == "hy" or str(detail.get("typename") or "") == "行业"
            else BankuaiLeixing.GAINIAN
            if type_code == "gn" or str(detail.get("typename") or "") == "概念"
            else None
        )
        checks = {
            "code_matches": observed_code == scope.code,
            "name_matches": observed_name == scope.canonical_name,
            "kind_matches": observed_kind is scope.kind,
        }
        verified = all(checks.values())
        return {
            "status": "verified" if verified else "conflict",
            "verified": verified,
            "source": "eastmoney_board_detail_page",
            "checked_at": checked_at,
            "observed_name": observed_name,
            "observed_kind": observed_kind.value if observed_kind is not None else None,
            "checks": checks,
            "attempted_providers": attempts,
            "persistence": "none",
        }


def _scope_query(value: str) -> tuple[str, str, BankuaiLeixing | None]:
    original = " ".join(str(value or "").strip().split())
    text = re.sub(r"^(?:请|帮我|麻烦)?(?:按|从|看)?(?:数据源当前的|数据源的)?", "", original)
    text = re.sub(r"(?:范围)?(?:中|里)?(?:分析|选股|选择股票)?$", "", text)
    text = re.sub(r"(?:板块|范围)$", "", text).strip(" ：:，,。")
    hint: BankuaiLeixing | None = None
    if text.endswith("概念"):
        hint = BankuaiLeixing.GAINIAN
        base = text[: -len("概念")].strip()
    elif text.endswith("行业"):
        hint = BankuaiLeixing.HANGYE
        base = text[: -len("行业")].strip()
    else:
        base = text
    return original, base or text, hint


def _candidate_base(scope: ShichangFanwei) -> str:
    name = scope.canonical_name.strip()
    if scope.kind is BankuaiLeixing.GAINIAN and name.endswith("概念"):
        return name[: -len("概念")].strip()
    return name


def _match_score(query: str, scope: ShichangFanwei) -> tuple[float, str]:
    base = _candidate_base(scope)
    if query == scope.canonical_name:
        return 1.0, "canonical_exact"
    if query == base:
        return 0.97, "semantic_exact"
    if query in base or base in query:
        return 0.87, "contains"
    return SequenceMatcher(None, query, base).ratio(), "fuzzy"


def _with_match(
    scope: ShichangFanwei,
    *,
    requested_name: str,
    score: float,
    basis: str,
    resolution: str,
    metadata: dict[str, Any],
    verification: dict[str, Any] | None = None,
) -> ShichangFanwei:
    return ShichangFanwei(
        requested_name=requested_name,
        canonical_name=scope.canonical_name,
        code=scope.code,
        kind=scope.kind,
        source=scope.source,
        fetched_at=scope.fetched_at,
        match_score=score,
        match_basis=basis,
        ambiguity_resolution=resolution,
        catalog_counts=dict(metadata.get("catalog_counts") or {}),
        attempted_providers=tuple(metadata.get("attempted_providers") or ()),
        verification=dict(verification or {}),
    )


def faxian_fenxi_fanwei(
    requested_name: str,
    *,
    provider: DongcaiFanweiShujuYuan | None = None,
) -> FanweiFaxianJieguo:
    """实时发现用户说出的范围；精确唯一时自动采用，否则只澄清一次。"""
    original, query, hint = _scope_query(requested_name)
    if not query:
        return FanweiFaxianJieguo(
            status="clarification_required",
            error_code="scope_name_missing",
            message="请告诉我想分析哪一类股票，例如电子、消费电子或半导体",
            next_action="请补充一个通俗的股票范围名称",
        )
    source = provider or DongcaiFanweiShujuYuan()
    try:
        catalog, metadata = source.huoqu_mulu()
    except WangluoQingqiuYichang as exc:
        return FanweiFaxianJieguo(
            status="unavailable",
            error_code="scope_catalog_unavailable",
            message="当前无法取得实时板块目录，不能可靠判断你说的范围",
            retryable=exc.retryable,
            attempted_providers=tuple(item.to_dict() for item in exc.attempts),
            catalog_source=source.source_name,
            fetched_at=_beijing_now_text(),
            next_action="稍后重试；程序不会用旧本地目录冒充当前结果",
        )
    except Exception as exc:
        return FanweiFaxianJieguo(
            status="unavailable",
            error_code="scope_catalog_invalid",
            message=f"实时板块目录返回异常：{' '.join(str(exc).split())[:180]}",
            retryable=True,
            catalog_source=source.source_name,
            fetched_at=_beijing_now_text(),
            next_action="稍后重试；程序不会把数据源故障说成板块不存在",
        )

    scored: list[tuple[float, str, ShichangFanwei]] = []
    for scope in catalog:
        if hint is not None and scope.kind is not hint:
            continue
        score, basis = _match_score(query, scope)
        scored.append((score, basis, scope))
    scored.sort(
        key=lambda item: (
            item[0],
            item[1] == "canonical_exact",
            item[2].kind is BankuaiLeixing.HANGYE,
        ),
        reverse=True,
    )
    attempts = tuple(metadata.get("attempted_providers") or ())
    common = {
        "catalog_source": str(metadata.get("source") or source.source_name),
        "fetched_at": str(metadata.get("fetched_at") or _beijing_now_text()),
        "attempted_providers": attempts,
    }

    def checked(
        items: Iterable[tuple[float, str, ShichangFanwei]],
    ) -> tuple[list[tuple[float, str, ShichangFanwei, dict[str, Any]]], list[str]]:
        verified: list[tuple[float, str, ShichangFanwei, dict[str, Any]]] = []
        errors: list[str] = []
        verifier = getattr(source, "hedui_fanwei", None)
        if not callable(verifier):
            return [], ["当前范围数据源没有提供详情页核验能力"]
        for score, basis, scope in items:
            try:
                verification = verifier(scope)
                if verification.get("verified") is True:
                    verified.append((score, basis, scope, verification))
                else:
                    errors.append(
                        f"{scope.canonical_name} 的目录记录与详情页存在冲突"
                    )
            except Exception as exc:
                errors.append(
                    f"{scope.canonical_name} 详情核验失败：{' '.join(str(exc).split())[:120]}"
                )
        return verified, errors

    semantic_exact = [item for item in scored if item[0] >= 0.97]
    if len(semantic_exact) == 1:
        verified_exact, _ = checked(semantic_exact)
        if not verified_exact:
            return FanweiFaxianJieguo(
                status="unavailable",
                error_code="scope_verification_unavailable",
                message="实时目录找到了候选，但详情页二次核验没有通过，不能据此继续分析",
                retryable=True,
                next_action="稍后重试关键事实核验；程序不会盲信单次目录响应",
                **common,
            )
        score, basis, scope, verification = verified_exact[0]
        resolution = "explicit_scope_hint" if hint is not None else "unique_exact_live_catalog_match"
        return FanweiFaxianJieguo(
            status="resolved",
            scope=_with_match(
                scope,
                requested_name=original,
                score=score,
                basis=basis,
                resolution=resolution,
                metadata=metadata,
                verification=verification,
            ),
            **common,
        )
    if len(semantic_exact) > 1:
        verified_exact, _ = checked(semantic_exact[:4])
        if not verified_exact:
            return FanweiFaxianJieguo(
                status="unavailable",
                error_code="scope_verification_unavailable",
                message="实时目录返回了多个候选，但详情页二次核验当前不可用",
                retryable=True,
                next_action="稍后重试；在关键事实未核验前不生成候选池",
                **common,
            )
        if len(verified_exact) == 1:
            score, basis, scope, verification = verified_exact[0]
            return FanweiFaxianJieguo(
                status="resolved",
                scope=_with_match(
                    scope,
                    requested_name=original,
                    score=score,
                    basis=basis,
                    resolution="unique_candidate_after_detail_verification",
                    metadata=metadata,
                    verification=verification,
                ),
                **common,
            )
        candidates = tuple(
            _with_match(
                scope,
                requested_name=original,
                score=score,
                basis=basis,
                resolution="user_choice_required",
                metadata=metadata,
                verification=verification,
            )
            for score, basis, scope, verification in verified_exact
        )
        return FanweiFaxianJieguo(
            status="clarification_required",
            candidates=candidates,
            error_code="scope_ambiguous",
            message="实时数据源里有多个名称相近但成分不同的范围，请选择一个",
            next_action="只需用通俗名称选择一次，不需要理解内部分类或代码",
            **common,
        )
    plausible = [item for item in scored if item[0] >= 0.72]
    if plausible:
        verified_plausible, _ = checked(plausible[:4])
        if not verified_plausible:
            return FanweiFaxianJieguo(
                status="unavailable",
                error_code="scope_verification_unavailable",
                message="可能的范围未能通过详情页二次核验，当前不适合继续推断",
                retryable=True,
                next_action="稍后重试关键事实核验",
                **common,
            )
        top_score, top_basis, top_scope, top_verification = verified_plausible[0]
        second_score = verified_plausible[1][0] if len(verified_plausible) > 1 else 0.0
        if top_score >= 0.87 and top_score - second_score >= 0.08:
            return FanweiFaxianJieguo(
                status="resolved",
                scope=_with_match(
                    top_scope,
                    requested_name=original,
                    score=top_score,
                    basis=top_basis,
                    resolution="unique_high_confidence_live_catalog_match",
                    metadata=metadata,
                    verification=top_verification,
                ),
                **common,
            )
        candidates = tuple(
            _with_match(
                scope,
                requested_name=original,
                score=score,
                basis=basis,
                resolution="user_choice_required",
                metadata=metadata,
                verification=verification,
            )
            for score, basis, scope, verification in verified_plausible
        )
        return FanweiFaxianJieguo(
            status="clarification_required",
            candidates=candidates,
            error_code="scope_ambiguous",
            message="实时数据源里有几个可能的范围，请选择最符合你意思的一个",
            next_action="只需选择一次；也可以直接输入自己的说法",
            **common,
        )
    nearest_raw = [item for item in scored[:4] if item[0] >= 0.45]
    verified_nearest, _ = checked(nearest_raw) if nearest_raw else ([], [])
    nearest = tuple(
        _with_match(
            scope,
            requested_name=original,
            score=score,
            basis=basis,
            resolution="user_choice_required",
            metadata=metadata,
            verification=verification,
        )
        for score, basis, scope, verification in verified_nearest
    )
    return FanweiFaxianJieguo(
        status="clarification_required",
        candidates=nearest,
        error_code="scope_not_found",
        message=f"已取得实时目录，但没有可靠识别“{original}”具体指哪个范围",
        next_action="请换一种日常说法，或从候选中选择；不需要提供专业分类",
        **common,
    )


def huoqu_dongcai_chengfen(
    scope: ShichangFanwei,
    *,
    provider: DongcaiFanweiShujuYuan | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return (provider or DongcaiFanweiShujuYuan()).huoqu_chengfen(scope)


__all__ = [
    "BankuaiLeixing",
    "DongcaiFanweiShujuYuan",
    "FanweiFaxianJieguo",
    "ShichangFanwei",
    "faxian_fenxi_fanwei",
    "huoqu_dongcai_chengfen",
]
