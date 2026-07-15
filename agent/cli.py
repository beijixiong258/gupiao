#!/usr/bin/env python3
"""CLI for the A-share T+3 quantitative researcher."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

console = Console()
AGENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = AGENT_DIR / "runs"

EXIT_SUCCESS = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE_ERROR = 2


def _console_safe(value: object) -> str:
    """Return text that the current Windows console encoding can print."""
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _build_agent(max_iter: int = 50):
    from src.agent.loop import AgentLoop
    from src.memory.persistent import PersistentMemory
    from src.providers.chat import ChatLLM
    from src.tools import build_registry

    pm = PersistentMemory()
    return AgentLoop(
        registry=build_registry(persistent_memory=pm, include_shell_tools=False),
        llm=ChatLLM(),
        max_iterations=max_iter,
        persistent_memory=pm,
    )


def _run_agent(prompt: str, max_iter: int = 50) -> dict:
    agent = _build_agent(max_iter=max_iter)
    return agent.run(user_message=prompt)


def cmd_run(prompt: str, max_iter: int, *, json_mode: bool = False) -> int:
    from src.preflight import run_preflight

    if not json_mode:
        results = run_preflight(console)
        if any(result.critical and result.status != "ready" for result in results):
            return EXIT_RUN_FAILED

    started = time.perf_counter()
    try:
        result = _run_agent(prompt, max_iter=max_iter)
    except KeyboardInterrupt:
        result = {"status": "cancelled", "reason": "Interrupted"}
    except Exception as exc:
        result = {"status": "failed", "reason": str(exc)}

    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
        return EXIT_SUCCESS if result.get("status") == "success" else EXIT_RUN_FAILED

    elapsed = time.perf_counter() - started
    status = result.get("status", "unknown")
    reason = result.get("reason", "")
    run_id = result.get("run_id", "")
    body = [f"Status: {status}", f"Elapsed: {elapsed:.1f}s"]
    if run_id:
        body.append(f"Run: {run_id}")
    if reason:
        body.append(f"Reason: {reason}")
    if content := result.get("content"):
        body.append("")
        body.append(_console_safe(content))
    console.print(Panel(_console_safe("\n".join(body)), title="A股 T+3 量化研究员"))
    return EXIT_SUCCESS if status == "success" else EXIT_RUN_FAILED


def cmd_chat(max_iter: int) -> int:
    from src.preflight import run_preflight

    results = run_preflight(console)
    if any(result.critical and result.status != "ready" for result in results):
        return EXIT_RUN_FAILED

    agent = _build_agent(max_iter=max_iter)
    history: list[dict] = []

    console.print(
        Panel(
            "进入 CLI 对话模式。可询问一只 A 股，或指定板块进行选股和 T+3 预测。\n"
            "退出：exit / quit / q / 退出\n"
            "示例：分析 600519.SH 的基本面和技术面\n"
            "示例：从白酒板块选 3 只股票并预测未来 3 个交易日",
            title="A股 T+3 量化研究员",
        )
    )

    while True:
        try:
            prompt = console.input("[bold cyan]你 > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]已退出[/dim]")
            return EXIT_SUCCESS

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q", "退出"}:
            console.print("[dim]已退出[/dim]")
            return EXIT_SUCCESS

        started = time.perf_counter()
        try:
            result = agent.run(user_message=prompt, history=history)
        except KeyboardInterrupt:
            console.print("\n[yellow]本轮已中断[/yellow]")
            continue
        except Exception as exc:
            console.print(Panel(_console_safe(exc), title="Error", style="red"))
            continue

        elapsed = time.perf_counter() - started
        status = result.get("status", "unknown")
        run_id = result.get("run_id", "")
        content = str(result.get("content") or result.get("reason") or "")

        if content:
            console.print(Panel(_console_safe(content), title=f"研究员 | {status} | {elapsed:.1f}s | {run_id}"))
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": content})
        else:
            console.print(Panel(f"Status: {status}\nRun: {run_id}", title="研究员"))


def cmd_list(limit: int = 20) -> int:
    if not RUNS_DIR.exists():
        console.print("[dim]No runs yet[/dim]")
        return EXIT_SUCCESS
    runs = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for run in runs:
        console.print(run.name)
    return EXIT_SUCCESS


def cmd_settings() -> int:
    import os
    from src.providers.llm import _ensure_dotenv, _provider_name
    from src.providers.openai_codex import codex_auth_status

    _ensure_dotenv()
    provider = _provider_name()
    oauth_status = codex_auth_status()

    console.print(
        Panel(
            "\n".join(
                [
                    f"Provider: {provider}",
                    f"Model: {os.getenv('LANGCHAIN_MODEL_NAME', os.getenv('DEEPSEEK_MODEL', '(not set)'))}",
                    f"DeepSeek key: {'set' if os.getenv('DEEPSEEK_API_KEY') else 'not set'}",
                    f"OpenAI API key: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}",
                    f"ChatGPT OAuth: {'ready' if oauth_status['configured'] else 'not logged in'}",
                    f"Tushare token: {'set' if os.getenv('TUSHARE_TOKEN') else 'not set'}",
                ]
            ),
            title="Settings",
        )
    )
    return EXIT_SUCCESS


def cmd_openai_login(no_browser: bool) -> int:
    from src.providers.openai_codex import CodexAuthError, login_openai_codex

    try:
        result = login_openai_codex(open_browser=not no_browser, print_fn=lambda value: console.print(_console_safe(value)))
    except KeyboardInterrupt:
        console.print("\n[yellow]OpenAI 登录已取消[/yellow]")
        return EXIT_RUN_FAILED
    except CodexAuthError as exc:
        console.print(Panel(_console_safe(exc), title="OpenAI Login", style="red"))
        return EXIT_RUN_FAILED
    console.print(
        Panel(
            _console_safe(f"Status: ok\nAuth file: {result['auth_file']}\n请在 agent/.env 中设置 LANGCHAIN_PROVIDER=openai_codex"),
            title="OpenAI Login",
        )
    )
    return EXIT_SUCCESS


def cmd_openai_logout() -> int:
    from src.providers.openai_codex import logout_openai_codex

    deleted = logout_openai_codex()
    console.print("OpenAI OAuth 登录已清除" if deleted else "当前没有 OpenAI OAuth 登录")
    return EXIT_SUCCESS


def cmd_gupiao(gupiao: str, source: str, history_calendar_days: int, json_mode: bool) -> int:
    from src.ashare.gupiao_yanjiu import fenxi_gupiao

    result = fenxi_gupiao(
        gupiao=gupiao,
        source=source,
        history_calendar_days=history_calendar_days,
    )
    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
    else:
        console.print_json(json.dumps(result, ensure_ascii=False))
    return EXIT_SUCCESS if result.get("status") == "ok" else EXIT_RUN_FAILED


def cmd_bankuai(
    bankuai: str,
    bankuai_leixing: str,
    top_n: int,
    source: str,
    config_path: str | None,
    json_mode: bool,
) -> int:
    from src.ashare.bankuai_yuce import bankuai_xuangu

    result = bankuai_xuangu(
        bankuai=bankuai,
        bankuai_leixing=bankuai_leixing,
        top_n=top_n,
        source=source,
        config_path=config_path,
    )
    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
    else:
        console.print_json(json.dumps(result, ensure_ascii=False))
    return EXIT_SUCCESS if result.get("status") == "ok" else EXIT_RUN_FAILED


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="A-share quant research assistant")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run one A-share research request")
    run.add_argument("prompt", nargs="?", help="Natural-language task prompt")
    run.add_argument("-p", "--prompt", dest="prompt_opt", help="Natural-language task prompt")
    run.add_argument("--max-iter", type=int, default=50)
    run.add_argument("--json", action="store_true")

    chat = sub.add_parser("chat", help="Start interactive terminal chat")
    chat.add_argument("--max-iter", type=int, default=50)

    list_cmd = sub.add_parser("list", help="List recent runs")
    list_cmd.add_argument("--limit", type=int, default=20)

    gupiao = sub.add_parser("gupiao", help="Analyze one A-share by code or Chinese name")
    gupiao.add_argument("gupiao", help="For example 600519.SH or 贵州茅台")
    gupiao.add_argument("--source", choices=["auto", "tushare", "akshare"], default="auto")
    gupiao.add_argument("--history-calendar-days", type=int, default=540)
    gupiao.add_argument("--json", action="store_true")

    bankuai = sub.add_parser("bankuai", help="Select stocks from one board and predict T+1/T+2/T+3")
    bankuai.add_argument("bankuai", help="Chinese industry or concept board name")
    bankuai.add_argument("--type", dest="bankuai_leixing", choices=["auto", "hangye", "gainian"], default="auto")
    bankuai.add_argument("--top-n", type=int, default=3)
    bankuai.add_argument("--source", choices=["auto", "tushare", "akshare"], default="auto")
    bankuai.add_argument("--config", dest="config_path", help="Path to lianghua_peizhi.json")
    bankuai.add_argument("--json", action="store_true")

    sub.add_parser("settings", help="Show runtime settings")
    openai_login = sub.add_parser("openai-login", help="Login with ChatGPT OAuth for the openai_codex provider")
    openai_login.add_argument("--no-browser", action="store_true", help="Do not open the login page automatically")
    sub.add_parser("openai-logout", help="Remove the locally stored ChatGPT OAuth grant")

    args = parser.parse_args(argv)

    if args.command == "run":
        prompt = args.prompt_opt or args.prompt
        if not prompt:
            console.print('[red]Missing prompt. Use: gpyj run -p "分析 600519.SH"[/red]')
            return EXIT_USAGE_ERROR
        return cmd_run(prompt, args.max_iter, json_mode=args.json)
    if args.command == "chat":
        return cmd_chat(args.max_iter)
    if args.command == "list":
        return cmd_list(args.limit)
    if args.command == "gupiao":
        return cmd_gupiao(args.gupiao, args.source, args.history_calendar_days, args.json)
    if args.command == "bankuai":
        return cmd_bankuai(
            args.bankuai,
            args.bankuai_leixing,
            args.top_n,
            args.source,
            args.config_path,
            args.json,
        )
    if args.command == "settings":
        return cmd_settings()
    if args.command == "openai-login":
        return cmd_openai_login(args.no_browser)
    if args.command == "openai-logout":
        return cmd_openai_logout()

    parser.print_help()
    return EXIT_USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
