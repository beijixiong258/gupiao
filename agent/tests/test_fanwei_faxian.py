"""动态范围发现、事实复核与歧义边界测试。"""

from __future__ import annotations

from typing import Any

import pytest

from src.ashare.fanwei_faxian import (
    BankuaiLeixing,
    ShichangFanwei,
    faxian_fenxi_fanwei,
)
from src.ashare.wangluo_kehu import WangluoQingqiuYichang
from src.agent.clarification import build_scope_clarification


def _scope(name: str, code: str, kind: BankuaiLeixing) -> ShichangFanwei:
    return ShichangFanwei(
        requested_name="",
        canonical_name=name,
        code=code,
        kind=kind,
        source="fake_live_catalog",
        fetched_at="2026-08-18 20:00:00",
        match_score=0.0,
        match_basis="catalog_entry",
        ambiguity_resolution="pending",
    )


class _FakeDynamicProvider:
    source_name = "fake_live_catalog"

    def __init__(
        self,
        scopes: list[ShichangFanwei],
        *,
        conflicts: set[str] | None = None,
        failures: set[str] | None = None,
    ) -> None:
        self._scopes = scopes
        self._conflicts = conflicts or set()
        self._failures = failures or set()
        self.verification_calls: list[str] = []

    def huoqu_mulu(self) -> tuple[list[ShichangFanwei], dict[str, Any]]:
        return list(self._scopes), {
            "source": self.source_name,
            "fetched_at": "2026-08-18 20:00:00",
            "catalog_counts": {"industry": 1, "concept": 1},
            "attempted_providers": [
                {"provider": self.source_name, "host": "example.test", "attempt": 1, "outcome": "ok"}
            ],
        }

    def hedui_fanwei(self, scope: ShichangFanwei) -> dict[str, Any]:
        self.verification_calls.append(scope.code)
        if scope.code in self._failures:
            raise WangluoQingqiuYichang(
                "详情页超时",
                error_code="network_timeout",
                retryable=True,
            )
        verified = scope.code not in self._conflicts
        return {
            "status": "verified" if verified else "conflict",
            "verified": verified,
            "source": "fake_detail_page",
            "observed_name": scope.canonical_name,
            "observed_kind": scope.kind.value,
            "checks": {
                "code_matches": verified,
                "name_matches": True,
                "kind_matches": True,
            },
        }


def test_exact_ordinary_name_is_resolved_from_dynamic_catalog_after_verification() -> None:
    provider = _FakeDynamicProvider(
        [
            _scope("电子", "BK1201", BankuaiLeixing.HANGYE),
            _scope("电子烟", "BK9999", BankuaiLeixing.GAINIAN),
        ]
    )

    result = faxian_fenxi_fanwei("电子板块", provider=provider)

    assert result.status == "resolved"
    assert result.scope is not None
    assert result.scope.canonical_name == "电子"
    assert result.scope.kind is BankuaiLeixing.HANGYE
    assert result.scope.ambiguity_resolution == "unique_exact_live_catalog_match"
    assert result.scope.verification["verified"] is True
    assert result.scope.catalog_counts == {"industry": 1, "concept": 1}


def test_two_verified_live_candidates_require_only_plain_language_clarification() -> None:
    provider = _FakeDynamicProvider(
        [
            _scope("消费电子", "BK1037", BankuaiLeixing.HANGYE),
            _scope("消费电子概念", "BK1646", BankuaiLeixing.GAINIAN),
        ]
    )

    result = faxian_fenxi_fanwei("消费电子板块", provider=provider)
    public = result.to_result()

    assert result.status == "clarification_required"
    assert [item.canonical_name for item in result.candidates] == [
        "消费电子",
        "消费电子概念",
    ]
    assert all(item.verification["verified"] is True for item in result.candidates)
    assert all("BK" not in item["user_label"] for item in public["candidates"])
    assert public["next_action"].startswith("只需用通俗名称")
    clarification = build_scope_clarification(public)
    assert clarification is not None
    assert clarification.kind == "scope_clarification"
    assert len(clarification.choices) == 2
    assert all("BK" not in choice.label for choice in clarification.choices)
    assert clarification.allow_custom is True


def test_catalog_failure_is_unavailable_instead_of_claiming_scope_absent() -> None:
    class _UnavailableProvider(_FakeDynamicProvider):
        def huoqu_mulu(self):
            raise WangluoQingqiuYichang(
                "source down",
                error_code="network_connection_failed",
                retryable=True,
            )

    result = faxian_fenxi_fanwei("电子板块", provider=_UnavailableProvider([]))

    assert result.status == "unavailable"
    assert result.error_code == "scope_catalog_unavailable"
    assert "无法取得实时板块目录" in str(result.message)
    assert "不存在" not in str(result.message)


def test_exact_catalog_hit_is_rejected_when_detail_fact_check_conflicts() -> None:
    provider = _FakeDynamicProvider(
        [_scope("电子", "BK1201", BankuaiLeixing.HANGYE)],
        conflicts={"BK1201"},
    )

    result = faxian_fenxi_fanwei("电子", provider=provider)

    assert result.status == "unavailable"
    assert result.error_code == "scope_verification_unavailable"
    assert "二次核验" in str(result.message)
    assert result.candidates[0].verification["status"] == "conflict"
    assert result.to_result()["candidates"][0]["verification"]["checks"]["code_matches"] is False


@pytest.mark.parametrize("failed_check", ["timeout", "conflict"])
def test_partially_verified_exact_candidates_are_preserved_without_resolving(failed_check: str) -> None:
    provider = _FakeDynamicProvider(
        [
            _scope("消费电子", "BK1037", BankuaiLeixing.HANGYE),
            _scope("消费电子概念", "BK1646", BankuaiLeixing.GAINIAN),
        ],
        conflicts={"BK1646"} if failed_check == "conflict" else None,
        failures={"BK1646"} if failed_check == "timeout" else None,
    )

    result = faxian_fenxi_fanwei("消费电子", provider=provider)
    payload = result.to_result()

    assert result.status == "unavailable"
    assert result.scope is None
    assert payload["outcome"] == "data_unavailable"
    assert [candidate["canonical_code"] for candidate in payload["candidates"]] == ["BK1037", "BK1646"]
    assert payload["candidates"][0]["verification"]["verified"] is True
    failed = payload["candidates"][1]["verification"]
    assert failed["verified"] is False
    assert failed["status"] == ("conflict" if failed_check == "conflict" else "unavailable")
    assert failed["error_code"] == ("scope_detail_conflict" if failed_check == "conflict" else "network_timeout")
    assert failed["error"]
    assert result.retryable is True


def test_fuzzy_candidate_timeout_cannot_turn_an_ambiguous_match_into_unique_scope() -> None:
    provider = _FakeDynamicProvider(
        [
            _scope("芯片设计", "BK1001", BankuaiLeixing.GAINIAN),
            _scope("芯片材料", "BK1002", BankuaiLeixing.GAINIAN),
        ],
        failures={"BK1002"},
    )

    result = faxian_fenxi_fanwei("芯片", provider=provider)

    assert result.status == "unavailable"
    assert result.scope is None
    assert len(result.candidates) == 2
    assert result.candidates[1].verification["error"] == "详情页超时"


def test_unique_verified_high_confidence_candidate_keeps_automatic_resolution() -> None:
    provider = _FakeDynamicProvider([_scope("半导体设备", "BK1001", BankuaiLeixing.HANGYE)])

    result = faxian_fenxi_fanwei("半导体", provider=provider)

    assert result.status == "resolved"
    assert result.scope is not None
    assert result.scope.ambiguity_resolution == "unique_high_confidence_live_catalog_match"
    assert result.scope.verification["verified"] is True


def test_candidates_beyond_detail_request_limit_remain_explicitly_unverified() -> None:
    provider = _FakeDynamicProvider(
        [_scope("电子", f"BK100{index}", BankuaiLeixing.HANGYE) for index in range(5)]
    )

    result = faxian_fenxi_fanwei("电子", provider=provider)

    assert result.status == "unavailable"
    assert len(provider.verification_calls) == 4
    assert len(result.candidates) == 5
    assert result.candidates[-1].verification["status"] == "not_checked"
    assert result.candidates[-1].verification["error_code"] == "scope_verification_limit_reached"
