"""Startup preflight checks for A-share data sources and the selected LLM.

Runs connectivity checks at startup and prints a status table.
Non-critical failures are warnings (degraded functionality),
LLM provider failure is critical (blocks startup).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class CheckResult:
    """Result of a single preflight check."""

    name: str
    status: str  # "ready", "error", "not_configured", "skipped"
    message: str
    impact: str  # what breaks if this fails
    critical: bool = False


def _check_llm_provider() -> CheckResult:
    """Verify LLM provider connectivity."""
    from src.providers.llm import _ensure_dotenv, resolve_provider_settings

    _ensure_dotenv()
    try:
        settings = resolve_provider_settings()
    except Exception as exc:
        return CheckResult(
            name=f"LLM ({os.getenv('LANGCHAIN_PROVIDER', 'not set')})",
            status="not_configured",
            message=str(exc),
            impact="agent cannot function",
            critical=True,
        )

    provider = str(settings["provider"])
    model = str(settings["model"])
    base_url = str(settings["base_url"])
    if provider == "openai_codex":
        return CheckResult(
            name="LLM (openai_codex)",
            status="ready",
            message=f"{model} via ChatGPT OAuth",
            impact="",
        )

    try:
        import httpx

        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings['api_key']}"},
            timeout=10.0,
        )
        if 200 <= response.status_code < 300:
            return CheckResult(
                name=f"LLM ({provider})",
                status="ready",
                message=f"{model} via {base_url}",
                impact="",
            )
        if response.status_code in {401, 403}:
            return CheckResult(
                name=f"LLM ({provider})",
                status="error",
                message=f"credential rejected (HTTP {response.status_code})",
                impact="agent cannot function",
                critical=True,
            )
        if response.status_code == 429:
            return CheckResult(
                name=f"LLM ({provider})",
                status="error",
                message="provider rate limited the connectivity check (HTTP 429)",
                impact="agent may be temporarily unavailable",
                critical=True,
            )
        if response.status_code >= 500:
            return CheckResult(
                name=f"LLM ({provider})",
                status="error",
                message=f"provider unavailable (HTTP {response.status_code})",
                impact="agent cannot function",
                critical=True,
            )
        return CheckResult(
            name=f"LLM ({provider})",
            status="error",
            message=f"unexpected provider response (HTTP {response.status_code})",
            impact="agent cannot function",
            critical=True,
        )
    except Exception as exc:
        return CheckResult(
            name=f"LLM ({provider})",
            status="error",
            message=f"{type(exc).__name__}: {exc}",
            impact="agent cannot function",
            critical=True,
        )


def _check_tushare() -> CheckResult:
    """Check Tushare token configuration."""
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token or token == "your-tushare-token":
        return CheckResult(
            name="Tushare",
            status="not_configured",
            message="TUSHARE_TOKEN not set (optional)",
            impact="A-share data unavailable",
        )

    try:
        import tushare  # noqa: F401
    except ImportError:
        return CheckResult(
            name="Tushare",
            status="skipped",
            message="package not installed",
            impact="A-share data unavailable",
        )

    return CheckResult(name="Tushare", status="ready", message="token configured", impact="")


def _check_akshare() -> CheckResult:
    """Check akshare availability."""
    try:
        import akshare  # noqa: F401
    except ImportError:
        return CheckResult(
            name="akshare",
            status="skipped",
            message="package not installed",
            impact="A-share fallback unavailable",
        )
    return CheckResult(name="akshare", status="ready", message="installed", impact="")


# -- Status icons and colors --------------------------------------------------

_STATUS_DISPLAY = {
    "ready": ("[green]OK[/green]", "green"),
    "error": ("[red]FAIL[/red]", "red"),
    "not_configured": ("[yellow]N/A[/yellow]", "yellow"),
    "skipped": ("[dim]SKIP[/dim]", "dim"),
}


def run_preflight(console: Optional[Console] = None) -> List[CheckResult]:
    """Run all preflight checks and print results.

    Args:
        console: Rich console for output. Creates one if not provided.

    Returns:
        List of check results.
    """
    if console is None:
        console = Console()

    checks = [
        _check_llm_provider,
        _check_tushare,
        _check_akshare,
    ]

    results: List[CheckResult] = []
    for check_fn in checks:
        results.append(check_fn())

    # Build display table
    table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=False)
    table.add_column(width=4)   # icon
    table.add_column(width=18)  # name
    table.add_column()          # message

    for r in results:
        icon, color = _STATUS_DISPLAY[r.status]
        detail = r.message
        if r.status in ("error", "not_configured") and r.impact:
            detail = f"{r.message} ({r.impact})"
        table.add_row(icon, f"[{color}]{r.name}[/{color}]", f"[{color}]{detail}[/{color}]")

    console.print()
    console.print("[bold]Preflight Check[/bold]")
    console.print(table)

    has_critical = any(r.critical and r.status != "ready" for r in results)
    if has_critical:
        console.print("\n[bold red]Critical check failed - agent cannot start without a working LLM provider.[/bold red]")
        console.print("[dim]  See: agent/.env.example for configuration reference[/dim]")
    else:
        ready_count = sum(1 for r in results if r.status == "ready")
        console.print(f"\n[dim]{ready_count}/{len(results)} services ready[/dim]")

    console.print()
    return results
