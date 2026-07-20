"""Tests for the continuous terminal chat entry point."""

from __future__ import annotations

import copy

import cli
import src.duihua.huihua as huihua_module
import src.preflight as preflight_module
from src.duihua.huihua import DuihuaCunchu


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, user_message, history=None, session_id=""):
        prior = copy.deepcopy(history or [])
        self.calls.append({"prompt": user_message, "history": prior, "session_id": session_id})
        answer = f"回答：{user_message}"
        return {
            "status": "success",
            "content": answer,
            "run_id": f"run_{len(self.calls)}",
            "history": prior + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ],
        }


def test_chat_reuses_history_and_saves_session(tmp_path, monkeypatch) -> None:
    fake_agent = _FakeAgent()
    prompts = iter(["分析贵州茅台", "那它未来三天呢？", "/exit"])
    monkeypatch.setattr(huihua_module, "DUIHUA_MULU", tmp_path)
    monkeypatch.setattr(preflight_module, "run_preflight", lambda console: [])
    monkeypatch.setattr(cli, "_build_agent", lambda max_iter, event_callback=None: fake_agent)
    monkeypatch.setattr(cli.console, "input", lambda *_args, **_kwargs: next(prompts))

    assert cli.cmd_chat(5, new_session=True) == cli.EXIT_SUCCESS

    assert len(fake_agent.calls) == 2
    assert fake_agent.calls[0]["history"] == []
    assert fake_agent.calls[1]["history"][-2:] == [
        {"role": "user", "content": "分析贵州茅台"},
        {"role": "assistant", "content": "回答：分析贵州茅台"},
    ]
    saved = DuihuaCunchu(tmp_path).zuijin()
    assert saved is not None
    assert saved.lunshu == 2
    assert saved.biaoti == "分析贵州茅台"
    assert saved.xiaoxi[-2]["content"] == "那它未来三天呢？"


def test_no_subcommand_starts_chat(monkeypatch) -> None:
    calls: list[tuple[int, str | None, bool]] = []

    def fake_chat(max_iter, *, session_id=None, new_session=False):
        calls.append((max_iter, session_id, new_session))
        return 0

    monkeypatch.setattr(cli, "cmd_chat", fake_chat)

    assert cli.main([]) == 0
    assert cli.main(["chat", "--max-iter", "7", "--new"]) == 0
    assert calls == [(50, None, False), (7, None, True)]


def test_clear_history_command_requires_confirmation_and_removes_saved_sessions(tmp_path, monkeypatch) -> None:
    store = DuihuaCunchu(tmp_path)
    session = store.xinjian()
    session.lunshu = 1
    session.xiaoxi = [{"role": "user", "content": "旧问题"}]
    store.baocun(session)
    fake_agent = _FakeAgent()
    prompts = iter(["/clear-history", "确认清除", "/exit"])
    monkeypatch.setattr(huihua_module, "DUIHUA_MULU", tmp_path)
    monkeypatch.setattr(preflight_module, "run_preflight", lambda console: [])
    monkeypatch.setattr(cli, "_build_agent", lambda max_iter, event_callback=None: fake_agent)
    monkeypatch.setattr(cli.console, "input", lambda *_args, **_kwargs: next(prompts))

    assert cli.cmd_chat(5) == cli.EXIT_SUCCESS

    assert DuihuaCunchu(tmp_path).liechu() == []
    assert fake_agent.calls == []
