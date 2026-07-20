"""Tests for resumable UTF-8 terminal conversations."""

from __future__ import annotations

import json
import os

import pytest

from src.duihua.huihua import DuihuaCunchu, HuihuaCuoWu, zhengli_xiaoxi


def test_utf8_session_round_trip(tmp_path) -> None:
    store = DuihuaCunchu(tmp_path)
    session = store.xinjian()
    session.shezhi_shouci_biaoti("分析贵州茅台，然后继续追问")
    session.lunshu = 1
    session.xiaoxi = [
        {"role": "user", "content": "分析贵州茅台"},
        {"role": "assistant", "content": "基本面稳健。"},
    ]

    path = store.baocun(session)
    loaded = store.duqu(session.huihua_id)

    assert path.read_bytes().decode("utf-8").startswith("{")
    assert "贵州茅台" in path.read_text(encoding="utf-8")
    assert loaded.biaoti == "分析贵州茅台，然后继续追问"
    assert loaded.lunshu == 1
    assert loaded.xiaoxi == session.xiaoxi
    assert list(tmp_path.glob("*.tmp")) == []


def test_latest_session_uses_file_update_time(tmp_path) -> None:
    store = DuihuaCunchu(tmp_path)
    first = store.xinjian()
    second = store.xinjian()
    first_path = store.baocun(first)
    second_path = store.baocun(second)
    os.utime(first_path, (100, 100))
    os.utime(second_path, (200, 200))

    assert store.zuijin().huihua_id == second.huihua_id


def test_invalid_session_id_cannot_escape_storage_directory(tmp_path) -> None:
    store = DuihuaCunchu(tmp_path)
    with pytest.raises(HuihuaCuoWu, match="无效的会话 ID"):
        store.duqu("../outside")


def test_corrupt_session_is_rejected_and_skipped_in_list(tmp_path) -> None:
    store = DuihuaCunchu(tmp_path)
    bad_id = "20260715_120000_abcdef"
    (tmp_path / f"{bad_id}.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(HuihuaCuoWu, match="会话文件损坏"):
        store.duqu(bad_id)
    assert store.liechu() == []


def test_message_cleanup_preserves_tool_context_and_drops_system() -> None:
    raw = [
        {"role": "system", "content": "不能进入会话历史"},
        {"role": "user", "content": "第二只呢？"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "gupiao_fenxi", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_1", "name": "gupiao_fenxi", "content": {"股票": "贵州茅台"}},
        object(),
    ]

    cleaned = zhengli_xiaoxi(raw)

    assert [message["role"] for message in cleaned] == ["user", "assistant", "tool"]
    assert "贵州茅台" in cleaned[-1]["content"]
    json.dumps(cleaned, ensure_ascii=False)


def test_clear_keeps_session_identity(tmp_path) -> None:
    store = DuihuaCunchu(tmp_path)
    session = store.xinjian()
    original_id = session.huihua_id
    session.lunshu = 3
    session.biaoti = "白酒板块"
    session.xiaoxi = [{"role": "user", "content": "继续"}]

    session.qingkong()
    store.baocun(session)
    loaded = store.duqu(original_id)

    assert loaded.huihua_id == original_id
    assert loaded.biaoti == "新会话"
    assert loaded.lunshu == 0
    assert loaded.xiaoxi == []


def test_clear_all_history_deletes_only_valid_saved_sessions(tmp_path) -> None:
    store = DuihuaCunchu(tmp_path)
    first = store.xinjian()
    store.baocun(first)
    second = store.xinjian()
    store.baocun(second)
    unrelated = tmp_path / "notes.json"
    unrelated.write_text("{}", encoding="utf-8")

    deleted = store.qingkong_quanbu()

    assert deleted == 2
    assert store.liechu() == []
    assert unrelated.is_file()
