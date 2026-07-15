"""Offline tests for ChatGPT Codex OAuth credential handling."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from src.providers.openai_codex import (
    DEFAULT_CODEX_BASE_URL,
    _jwt_claims,
    _token_expiring,
    codex_auth_status,
    codex_request_headers,
    get_codex_credentials,
    logout_openai_codex,
)


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"header.{encoded}.signature"


def test_jwt_claims_and_codex_account_header() -> None:
    token = _jwt(
        {
            "exp": time.time() + 3600,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_test"},
        }
    )
    assert _jwt_claims(token)["exp"] > time.time()
    headers = codex_request_headers(token)
    assert headers["ChatGPT-Account-ID"] == "acct_test"
    assert headers["originator"] == "codex_cli_rs"


def test_expiry_detection() -> None:
    assert _token_expiring(_jwt({"exp": time.time() - 1}))
    assert not _token_expiring(_jwt({"exp": time.time() + 3600}))


def test_credentials_are_read_from_independent_auth_file(tmp_path: Path, monkeypatch) -> None:
    auth_file = tmp_path / "openai_auth.json"
    token = _jwt({"exp": time.time() + 3600})
    auth_file.write_text(
        json.dumps({"access_token": token, "refresh_token": "refresh-test"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPYJ_CODEX_AUTH_FILE", str(auth_file))

    credentials = get_codex_credentials()
    assert credentials["access_token"] == token
    assert credentials["base_url"] == DEFAULT_CODEX_BASE_URL
    assert codex_auth_status()["configured"] is True
    assert logout_openai_codex() is True
    assert not auth_file.exists()


def test_provider_metadata_contains_api_and_oauth_options() -> None:
    provider_path = Path(__file__).resolve().parents[1] / "src" / "providers" / "llm_providers.json"
    providers = {item["name"]: item for item in json.loads(provider_path.read_text(encoding="utf-8"))}
    assert set(providers) == {"deepseek", "openai", "openai_codex"}
    assert providers["openai_codex"]["auth_type"] == "oauth"
    assert providers["openai_codex"]["api_key_required"] is False
