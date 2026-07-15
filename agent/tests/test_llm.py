"""Tests for the supported DeepSeek and OpenAI provider mapping."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.providers import llm as llm_mod
from src.providers.llm import _extract_balanced_json, resolve_provider_settings


def _settings(env: dict[str, str]) -> dict:
    llm_mod._dotenv_loaded = True
    with patch.dict(os.environ, env, clear=True):
        return resolve_provider_settings()


def test_deepseek_settings() -> None:
    settings = _settings(
        {
            "LANGCHAIN_PROVIDER": "deepseek",
            "LANGCHAIN_MODEL_NAME": "deepseek-chat",
            "DEEPSEEK_API_KEY": "ds-test",
        }
    )
    assert settings["provider"] == "deepseek"
    assert settings["base_url"] == "https://api.deepseek.com/v1"
    assert settings["use_responses_api"] is False


def test_openai_api_settings() -> None:
    settings = _settings(
        {
            "LANGCHAIN_PROVIDER": "openai",
            "LANGCHAIN_MODEL_NAME": "gpt-5.6",
            "OPENAI_API_KEY": "sk-test",
        }
    )
    assert settings["provider"] == "openai"
    assert settings["base_url"] == "https://api.openai.com/v1"
    assert settings["use_responses_api"] is True


def test_chatgpt_pro_hint_when_api_key_is_missing() -> None:
    with pytest.raises(RuntimeError, match="ChatGPT Pro"):
        _settings({"LANGCHAIN_PROVIDER": "openai", "LANGCHAIN_MODEL_NAME": "gpt-5.6"})


def test_codex_alias_and_model_prefix(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.providers.openai_codex.get_codex_credentials",
        lambda refresh_if_needed=True: {
            "access_token": "token-test",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.setattr("src.providers.openai_codex.codex_request_headers", lambda token: {"X-Test": token})
    settings = _settings(
        {
            "LANGCHAIN_PROVIDER": "openai-codex",
            "LANGCHAIN_MODEL_NAME": "openai-codex/gpt-5.3-codex",
        }
    )
    assert settings["provider"] == "openai_codex"
    assert settings["model"] == "gpt-5.3-codex"
    assert settings["default_headers"] == {"X-Test": "token-test"}


def test_unsupported_provider_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="deepseek, openai, or openai_codex"):
        _settings(
            {
                "LANGCHAIN_PROVIDER": "groq",
                "LANGCHAIN_MODEL_NAME": "anything",
            }
        )


def test_extract_balanced_json() -> None:
    assert _extract_balanced_json('prefix {"outer": {"value": 3}} suffix') == {"outer": {"value": 3}}
