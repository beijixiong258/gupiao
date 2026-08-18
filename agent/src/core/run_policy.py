"""运行记录的隐私、保留和清理策略。"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "lianghua_peizhi.json"
_SENSITIVE_KEY = re.compile(
    r"(?:token|secret|password|passwd|authorization|cookie|api[_-]?key|credential)",
    flags=re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"
)
_CONTENT_KEYS = {
    "prompt",
    "content",
    "history",
    "react_trace",
    "reasoning_content",
    "args",
    "arguments",
    "preview",
    "response",
    "summary",
    "detail",
    "reason",
    "error",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RunLogPolicy:
    enabled: bool = True
    content_mode: str = "metadata_only"
    retention_days: int = 14
    maximum_runs: int = 100
    maximum_total_mb: int = 100

    @classmethod
    def load(cls) -> "RunLogPolicy":
        section: dict[str, Any] = {}
        try:
            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            raw = payload.get("run_logs", {}) if isinstance(payload, dict) else {}
            if isinstance(raw, dict):
                section = raw
        except (OSError, ValueError, json.JSONDecodeError):
            section = {}
        policy = cls(
            enabled=_env_bool(
                "GPYJ_RUN_LOGS_ENABLED",
                bool(section.get("enabled", True)),
            ),
            content_mode=os.getenv(
                "GPYJ_RUN_LOG_CONTENT",
                str(section.get("content_mode", "metadata_only")),
            ).strip().lower(),
            retention_days=int(section.get("retention_days", 14)),
            maximum_runs=int(section.get("maximum_runs", 100)),
            maximum_total_mb=int(section.get("maximum_total_mb", 100)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.content_mode not in {"metadata_only", "full_redacted"}:
            raise ValueError("run_logs.content_mode 必须是 metadata_only 或 full_redacted")
        if not 1 <= self.retention_days <= 3650:
            raise ValueError("run_logs.retention_days 必须在 1 到 3650 之间")
        if not 1 <= self.maximum_runs <= 10000:
            raise ValueError("run_logs.maximum_runs 必须在 1 到 10000 之间")
        if not 1 <= self.maximum_total_mb <= 102400:
            raise ValueError("run_logs.maximum_total_mb 必须在 1 到 102400 之间")


def _redact_string(value: str) -> str:
    text = _BEARER_VALUE.sub("Bearer [REDACTED]", value)
    return _KEY_VALUE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)


def redact_for_log(value: Any, *, key: str | None = None) -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): redact_for_log(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_for_log(item) for item in value]
    if isinstance(value, tuple):
        return [redact_for_log(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def trace_entry_for_policy(entry: dict[str, Any], policy: RunLogPolicy) -> dict[str, Any]:
    redacted = redact_for_log(entry)
    if policy.content_mode == "full_redacted":
        return redacted
    return {
        key: value
        for key, value in redacted.items()
        if key not in _CONTENT_KEYS
    }


def response_for_policy(response: dict[str, Any], policy: RunLogPolicy) -> dict[str, Any]:
    redacted = redact_for_log(response)
    if policy.content_mode == "full_redacted":
        return redacted
    result = {
        key: value
        for key, value in redacted.items()
        if key in {"status", "iterations", "business_outcome"}
    }
    reason = response.get("reason")
    if reason:
        result["reason_saved"] = False
        result["reason_length"] = len(str(reason))
    clarification = response.get("clarification")
    if isinstance(clarification, dict):
        choices = clarification.get("choices")
        result["clarification"] = {
            "kind": clarification.get("kind"),
            "choice_count": len(choices) if isinstance(choices, list) else 0,
            "allow_custom": bool(clarification.get("allow_custom")),
        }
    return result


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _safe_child_directories(workspace: Path) -> list[Path]:
    root = workspace.resolve()
    if not root.exists():
        return []
    result: list[Path] = []
    for item in root.iterdir():
        try:
            resolved = item.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if item.is_dir() and not item.is_symlink():
            result.append(item)
    return result


def prune_run_directories(workspace: Path, policy: RunLogPolicy) -> list[str]:
    """按日期、数量和空间依次清理，目标严格限制在 runs 子目录。"""
    if not policy.enabled:
        return []
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now() - timedelta(days=policy.retention_days)
    removed: list[str] = []

    def remove(path: Path) -> None:
        resolved = path.resolve()
        resolved.relative_to(root)
        shutil.rmtree(resolved)
        removed.append(path.name)

    directories = sorted(
        _safe_child_directories(root),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for item in list(directories):
        if datetime.fromtimestamp(item.stat().st_mtime) < cutoff:
            remove(item)
            directories.remove(item)
    for item in directories[policy.maximum_runs :]:
        remove(item)
    directories = [item for item in directories[: policy.maximum_runs] if item.exists()]
    maximum_bytes = policy.maximum_total_mb * 1024 * 1024
    sizes = {item: _directory_size(item) for item in directories}
    total = sum(sizes.values())
    for item in reversed(directories):
        if total <= maximum_bytes:
            break
        remove(item)
        total -= sizes[item]
    return removed


def clear_run_directories(workspace: Path) -> list[str]:
    """显式清理入口；调用方必须由用户命令触发。"""
    root = workspace.resolve()
    removed: list[str] = []
    for item in _safe_child_directories(root):
        resolved = item.resolve()
        resolved.relative_to(root)
        shutil.rmtree(resolved)
        removed.append(item.name)
    return removed


def create_ephemeral_run_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="gpyj-run-"))


__all__ = [
    "RunLogPolicy",
    "clear_run_directories",
    "create_ephemeral_run_dir",
    "prune_run_directories",
    "redact_for_log",
    "response_for_policy",
    "trace_entry_for_policy",
]
