"""Validate output directories used by A-share research tools."""

from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_RUN_ROOTS_ENV = "GPYJ_ALLOWED_RUN_ROOTS"


def _reject_unc(path: str) -> None:
    if path.startswith(("\\\\", "//")):
        raise ValueError(f"UNC paths are not allowed: {path!r}")


def _allowed_run_roots() -> list[Path]:
    agent_root = Path(__file__).resolve().parents[2]
    roots = [
        agent_root / "runs",
        Path.cwd().resolve() / "runs",
        Path.home().resolve() / ".gupiaoyanjiu" / "runs",
    ]
    for raw in os.getenv(_ALLOWED_RUN_ROOTS_ENV, "").split(","):
        value = raw.strip()
        if value:
            _reject_unc(value)
            roots.append(Path(value).expanduser().resolve())

    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def safe_run_dir(path: str) -> Path:
    """Resolve a research output directory and reject paths outside allowed roots."""
    _reject_unc(path)
    resolved = Path(path).expanduser().resolve()
    if any(resolved.is_relative_to(root) for root in _allowed_run_roots()):
        return resolved
    raise ValueError(
        f"run_dir {path!r} is outside allowed run roots. "
        f"Set {_ALLOWED_RUN_ROOTS_ENV} to add a run directory."
    )


__all__ = ["safe_run_dir"]
