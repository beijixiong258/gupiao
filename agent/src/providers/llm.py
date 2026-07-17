"""LLM provider factory for DeepSeek, OpenAI API, and ChatGPT Codex OAuth."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore


if ChatOpenAI is not None:

    class ChatOpenAIWithReasoning(ChatOpenAI):  # type: ignore[misc,valid-type]
        """OpenAI-compatible client that preserves DeepSeek reasoning fields."""

        @staticmethod
        def _capture(src: Any, msg: Any) -> None:
            if value := src.get("reasoning_content") or src.get("reasoning"):
                msg.additional_kwargs["reasoning_content"] = value

        def _create_chat_result(self, response, generation_info=None):  # type: ignore[override]
            result = super()._create_chat_result(response, generation_info)
            raw = response if isinstance(response, dict) else response.model_dump()
            for gen, choice in zip(result.generations, raw["choices"]):
                self._capture(choice["message"], gen.message)
            return result

        def _convert_chunk_to_generation_chunk(  # type: ignore[override]
            self,
            chunk: dict,
            default_chunk_class: type,
            base_generation_info: Optional[dict],
        ):
            gen = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
            if gen is None:
                return None
            choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices")
            if choices:
                self._capture(choices[0]["delta"], gen.message)
            return gen

        def _get_request_payload(self, input_: Any, *, stop: Optional[list[str]] = None, **kwargs: Any) -> dict:  # type: ignore[override]
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            messages = super()._convert_input(input_).to_messages()
            for i, message in enumerate(payload["messages"]):
                if message.get("role") != "assistant":
                    continue
                if message.get("content") is None:
                    message["content"] = ""
                message["reasoning_content"] = messages[i].additional_kwargs.get("reasoning_content", "")
            return payload

else:
    ChatOpenAIWithReasoning = None  # type: ignore


AGENT_DIR = Path(__file__).resolve().parents[2]
_ENV_CANDIDATES = [
    Path.home() / ".gupiaoyanjiu" / ".env",
    AGENT_DIR / ".env",
    Path.cwd() / ".env",
]

_dotenv_loaded = False


def _load_env_file(path: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=path, override=False)
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_dotenv() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    for candidate in _ENV_CANDIDATES:
        if candidate.exists():
            _load_env_file(candidate)
            break
    _dotenv_loaded = True


def _provider_name() -> str:
    _ensure_dotenv()
    provider = os.getenv("LANGCHAIN_PROVIDER", "deepseek").strip().lower().replace("-", "_")
    aliases = {"codex": "openai_codex", "openai_oauth": "openai_codex"}
    return aliases.get(provider, provider)


def resolve_provider_settings() -> Dict[str, Any]:
    """Resolve provider credentials without logging secret values."""
    provider = _provider_name()
    model = os.getenv("LANGCHAIN_MODEL_NAME", "").strip()
    if not model:
        raise RuntimeError("LANGCHAIN_MODEL_NAME is not set")

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or "https://api.deepseek.com/v1"
        )
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url.rstrip("/"),
            "default_headers": None,
            "use_responses_api": False,
        }

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; ChatGPT Pro users can use LANGCHAIN_PROVIDER=openai_codex")
        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": (os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1").rstrip("/"),
            "default_headers": None,
            "use_responses_api": True,
        }

    if provider == "openai_codex":
        from src.providers.openai_codex import codex_request_headers, get_codex_credentials

        credentials = get_codex_credentials(refresh_if_needed=True)
        api_key = str(credentials["access_token"])
        for prefix in ("openai-codex/", "openai_codex/"):
            if model.startswith(prefix):
                model = model[len(prefix) :]
        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": str(credentials.get("base_url") or "").rstrip("/"),
            "default_headers": codex_request_headers(api_key),
            "use_responses_api": True,
        }

    raise RuntimeError("LANGCHAIN_PROVIDER must be deepseek, openai, or openai_codex")


def _sync_provider_env() -> Dict[str, Any]:
    """Compatibility helper used by preflight; returns resolved settings."""
    settings = resolve_provider_settings()
    os.environ["OPENAI_API_KEY"] = str(settings["api_key"])
    os.environ["OPENAI_BASE_URL"] = str(settings["base_url"])
    os.environ["OPENAI_API_BASE"] = str(settings["base_url"])
    return settings


def build_llm(*, model_name: Optional[str] = None, callbacks: Any = None) -> Any:
    """Construct the configured DeepSeek/OpenAI chat model."""
    settings = resolve_provider_settings()
    name = model_name or settings["model"]
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed")

    effort = os.getenv("LANGCHAIN_REASONING_EFFORT", "").strip().lower()
    configured_service_tier = os.getenv("LANGCHAIN_SERVICE_TIER", "").strip().lower() or None
    # Codex exposes the user-facing tier as "fast", while the Responses wire
    # value accepted by the ChatGPT backend is "priority".
    service_tier = "priority" if configured_service_tier == "fast" else configured_service_tier
    common: Dict[str, Any] = {
        "model": name,
        "api_key": settings["api_key"],
        "base_url": settings["base_url"],
        "timeout": int(os.getenv("TIMEOUT_SECONDS", "120")),
        "max_retries": int(os.getenv("MAX_RETRIES", "2")),
        "callbacks": callbacks,
    }

    if settings["provider"] == "deepseek":
        return ChatOpenAIWithReasoning(
            **common,
            temperature=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.0")),
            extra_body={"reasoning": {"effort": effort}} if effort else None,
        )

    reasoning = {"effort": effort, "summary": "auto"} if effort else None
    return ChatOpenAI(
        **common,
        default_headers=settings.get("default_headers"),
        use_responses_api=bool(settings.get("use_responses_api")),
        output_version="v0",
        store=False,
        streaming=settings["provider"] == "openai_codex",
        include=["reasoning.encrypted_content"] if settings["provider"] == "openai_codex" else None,
        reasoning=reasoning,
        service_tier=service_tier,
    )


def _extract_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the outermost JSON object from text using bracket balancing."""
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None
