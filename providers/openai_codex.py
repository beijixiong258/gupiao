"""ChatGPT Codex OAuth for the local A-share research CLI.

The OAuth grant is stored outside the project so it is never committed with
research code or data.  The device-code flow mirrors the public Codex CLI flow
used by Hermes Agent, but keeps this project's credentials independent.
"""

from __future__ import annotations

import base64
import json
import os
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx


CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_DEVICE_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
CODEX_DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
CODEX_LOGIN_URL = "https://auth.openai.com/codex/device"
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
TOKEN_REFRESH_SKEW_SECONDS = 120


class CodexAuthError(RuntimeError):
    """Raised when ChatGPT OAuth login or refresh fails."""


def _auth_path() -> Path:
    override = os.getenv("GPYJ_CODEX_AUTH_FILE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".gupiaoyanjiu" / "openai_codex_auth.json"


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        value = json.loads(decoded.decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _token_expiring(token: str, skew_seconds: int = TOKEN_REFRESH_SKEW_SECONDS) -> bool:
    exp = _jwt_claims(token).get("exp")
    try:
        return float(exp) <= time.time() + max(0, int(skew_seconds))
    except (TypeError, ValueError):
        return False


def codex_request_headers(access_token: str) -> dict[str, str]:
    """Build the headers expected by the ChatGPT Codex backend."""
    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (A-Share Daily Researcher)",
        "originator": "codex_cli_rs",
    }
    auth_claims = _jwt_claims(access_token).get("https://api.openai.com/auth", {})
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id.strip():
            headers["ChatGPT-Account-ID"] = account_id.strip()
    return headers


def _save_credentials(payload: dict[str, Any]) -> Path:
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def _read_credentials() -> dict[str, Any]:
    path = _auth_path()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _refresh_credentials(payload: dict[str, Any], timeout_seconds: float = 20.0) -> dict[str, Any]:
    refresh_token = str(payload.get("refresh_token") or "").strip()
    if not refresh_token:
        raise CodexAuthError("OpenAI OAuth refresh_token 缺失，请重新运行 gpyj openai-login。")

    try:
        response = httpx.post(
            CODEX_OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "gupiaoyanjiu/0.2",
            },
            timeout=max(5.0, float(timeout_seconds)),
        )
    except Exception as exc:
        raise CodexAuthError(f"OpenAI OAuth token 刷新失败：{exc}") from exc

    if response.status_code == 429:
        raise CodexAuthError("OpenAI OAuth 当前被限频，请稍后重试；无需重复登录。")
    if response.status_code != 200:
        raise CodexAuthError(
            f"OpenAI OAuth token 刷新失败（HTTP {response.status_code}），请重新运行 gpyj openai-login。"
        )

    try:
        refreshed = response.json()
    except ValueError as exc:
        raise CodexAuthError("OpenAI OAuth token 刷新返回了无效 JSON。") from exc

    access_token = str(refreshed.get("access_token") or "").strip()
    if not access_token:
        raise CodexAuthError("OpenAI OAuth token 刷新结果缺少 access_token。")

    payload = dict(payload)
    payload["access_token"] = access_token
    next_refresh = str(refreshed.get("refresh_token") or "").strip()
    if next_refresh:
        payload["refresh_token"] = next_refresh
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_credentials(payload)
    return payload


def get_codex_credentials(*, refresh_if_needed: bool = True) -> dict[str, Any]:
    """Return a usable OAuth credential set without exposing it to logs."""
    payload = _read_credentials()
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise CodexAuthError("尚未登录 ChatGPT，请先运行 gpyj openai-login。")
    if refresh_if_needed and _token_expiring(access_token):
        payload = _refresh_credentials(payload)
    payload.setdefault(
        "base_url",
        os.getenv("OPENAI_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL,
    )
    return payload


def codex_auth_status() -> dict[str, Any]:
    payload = _read_credentials()
    access_token = str(payload.get("access_token") or "").strip()
    claims = _jwt_claims(access_token) if access_token else {}
    auth_claims = claims.get("https://api.openai.com/auth", {})
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    return {
        "configured": bool(access_token),
        "expiring": _token_expiring(access_token) if access_token else False,
        "account_id": account_id if isinstance(account_id, str) else "",
        "auth_file": str(_auth_path()),
    }


def login_openai_codex(
    *,
    open_browser: bool = True,
    timeout_seconds: int = 15 * 60,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run OpenAI's device-code login and persist an independent OAuth grant."""
    response: httpx.Response | None = None
    for attempt in range(4):
        try:
            response = httpx.post(
                CODEX_DEVICE_CODE_URL,
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )
        except Exception as exc:
            raise CodexAuthError(f"无法申请 OpenAI 登录码：{exc}") from exc
        if response.status_code != 429:
            break
        if attempt < 3:
            delay = min(2 ** (attempt + 1), 30)
            print_fn(f"OpenAI 登录请求被限频，{delay} 秒后重试...")
            time.sleep(delay)

    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "unknown"
        raise CodexAuthError(f"申请 OpenAI 登录码失败（HTTP {status}）。")

    data = response.json()
    user_code = str(data.get("user_code") or "").strip()
    device_auth_id = str(data.get("device_auth_id") or "").strip()
    poll_interval = max(3, int(data.get("interval") or 5))
    if not user_code or not device_auth_id:
        raise CodexAuthError("OpenAI 登录码响应缺少必要字段。")

    print_fn(f"请在浏览器打开：{CODEX_LOGIN_URL}")
    print_fn(f"输入登录码：{user_code}")
    print_fn("正在等待登录完成，按 Ctrl+C 可取消。")
    if open_browser:
        try:
            webbrowser.open(CODEX_LOGIN_URL)
        except Exception:
            pass

    deadline = time.monotonic() + max(60, int(timeout_seconds))
    code_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        poll = httpx.post(
            CODEX_DEVICE_TOKEN_URL,
            json={"device_auth_id": device_auth_id, "user_code": user_code},
            headers={"Content-Type": "application/json"},
            timeout=15.0,
        )
        if poll.status_code == 200:
            value = poll.json()
            code_payload = value if isinstance(value, dict) else None
            break
        if poll.status_code in {403, 404}:
            continue
        raise CodexAuthError(f"等待 OpenAI 登录时失败（HTTP {poll.status_code}）。")

    if not code_payload:
        raise CodexAuthError("OpenAI 登录等待超时，请重新运行 gpyj openai-login。")

    authorization_code = str(code_payload.get("authorization_code") or "").strip()
    code_verifier = str(code_payload.get("code_verifier") or "").strip()
    if not authorization_code or not code_verifier:
        raise CodexAuthError("OpenAI 登录响应缺少授权码。")

    token_response = httpx.post(
        CODEX_OAUTH_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": "https://auth.openai.com/deviceauth/callback",
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15.0,
    )
    if token_response.status_code != 200:
        raise CodexAuthError(f"OpenAI token 交换失败（HTTP {token_response.status_code}）。")

    tokens = token_response.json()
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not access_token or not refresh_token:
        raise CodexAuthError("OpenAI token 响应不完整，请重新登录。")

    payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "base_url": os.getenv("OPENAI_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _save_credentials(payload)
    status = codex_auth_status()
    return {
        "status": "ok",
        "auth_file": str(path),
        "account_id": status.get("account_id", ""),
    }


def logout_openai_codex() -> bool:
    path = _auth_path()
    if not path.exists():
        return False
    path.unlink()
    return True
