#!/usr/bin/env python3
"""CLI for the A-share T+3 quantitative researcher."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

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


def _build_agent(
    max_iter: int = 50,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
):
    from src.agent.loop import AgentLoop
    from src.memory.persistent import PersistentMemory
    from src.providers.chat import ChatLLM
    from src.tools import build_registry

    pm = PersistentMemory()
    return AgentLoop(
        registry=build_registry(persistent_memory=pm, include_shell_tools=False),
        llm=ChatLLM(),
        event_callback=event_callback,
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


def _dayin_duihua_bangzhu() -> None:
    console.print(
        Panel(
            "直接输入自然语言即可连续追问。\n"
            "/new      新建空白会话\n"
            "/clear    清空当前会话\n"
            "/clear-history 清除全部历史会话\n"
            "/sessions 查看最近会话\n"
            "/resume ID 切换到指定会话\n"
            "/history  查看当前会话最近内容\n"
            "/help     查看命令\n"
            "/exit     退出并保留会话",
            title="会话命令",
        )
    )


def _dayin_huihua_liebiao(cunchu: Any, dangqian_id: str) -> None:
    sessions = cunchu.liechu(10)
    if not sessions:
        console.print("[dim]还没有已保存的会话[/dim]")
        return
    table = Table(title="最近会话")
    table.add_column("当前", width=4)
    table.add_column("会话 ID", no_wrap=True)
    table.add_column("标题")
    table.add_column("轮数", justify="right")
    table.add_column("更新时间", no_wrap=True)
    for session in sessions:
        table.add_row(
            "*" if session.huihua_id == dangqian_id else "",
            session.huihua_id,
            session.biaoti,
            str(session.lunshu),
            session.gengxin_shijian.replace("T", " ")[:19],
        )
    console.print(table)


def _dayin_dangqian_lishi(huihua: Any) -> None:
    visible: list[tuple[str, str]] = []
    for message in huihua.xiaoxi:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        visible.append(("你" if role == "user" else "研究员", content.strip()))
    if not visible:
        console.print("[dim]当前会话还没有内容[/dim]")
        return
    console.print(f"[dim]当前会话最近 {min(len(visible), 12)} 条消息：[/dim]")
    for role, content in visible[-12:]:
        preview = content if len(content) <= 500 else f"{content[:500]}..."
        console.print(f"[bold]{role} >[/bold] {_console_safe(preview)}")


def _chuangjian_jindu_huidiao(status_ref: dict[str, Any]) -> Callable[[str, dict[str, Any]], None]:
    tool_text = {
        "gupiao_fenxi": "正在核对单股时点、拉取同行历史并训练三周期模型...",
        "gupiao_yuce": "正在计算指定周期预测、交易费用和收益空间...",
        "bankuai_xuangu": "正在拉取板块成分并训练 T+3 模型...",
    }

    def callback(event_type: str, data: dict[str, Any]) -> None:
        status = status_ref.get("value")
        if status is None:
            return
        if event_type == "tool_call":
            message = tool_text.get(str(data.get("tool")), "正在调用研究工具...")
        elif event_type == "tool_result":
            message = "数据计算完成，正在整理研判..."
        elif event_type == "text_delta":
            message = "正在组织回答..."
        else:
            return
        status.update(f"[cyan]{message}[/cyan]")

    return callback


def cmd_chat(max_iter: int, *, session_id: str | None = None, new_session: bool = False) -> int:
    from src.preflight import run_preflight
    from src.duihua.huihua import DuihuaCunchu, HuihuaCuoWu, zhengli_xiaoxi

    results = run_preflight(console)
    if any(result.critical and result.status != "ready" for result in results):
        return EXIT_RUN_FAILED

    cunchu = DuihuaCunchu()
    try:
        if new_session:
            huihua = cunchu.xinjian()
        elif session_id:
            huihua = cunchu.duqu(session_id)
        else:
            huihua = cunchu.zuijin() or cunchu.xinjian()
    except HuihuaCuoWu as exc:
        console.print(Panel(_console_safe(exc), title="会话载入失败", style="red"))
        return EXIT_USAGE_ERROR

    history: list[dict[str, Any]] = zhengli_xiaoxi(huihua.xiaoxi)
    status_ref: dict[str, Any] = {"value": None}
    agent = _build_agent(max_iter=max_iter, event_callback=_chuangjian_jindu_huidiao(status_ref))
    mode = f"已续接 {huihua.lunshu} 轮" if huihua.lunshu else "新会话"

    console.print(
        Panel(
            f"{mode}\n会话：{huihua.huihua_id}\n"
            "可直接连续追问。示例：分析贵州茅台，然后追问：它目前主要有哪些风险？\n\n"
            "可用斜杠命令：\n"
            "/new            新建空白会话\n"
            "/clear          清空当前会话\n"
            "/clear-history  清除全部历史会话\n"
            "/sessions       查看最近会话\n"
            "/resume 会话ID  切换到指定会话\n"
            "/history        查看当前会话最近内容\n"
            "/help           再次显示命令说明\n"
            "/exit           保存并退出",
            title="A股 T+3 量化研究员 | 连续对话",
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
        command = prompt.lower()
        if command in {"exit", "quit", "q", "退出", "/exit", "/quit", "/q"}:
            if huihua.lunshu:
                console.print(f"[dim]会话已保存：{huihua.huihua_id}[/dim]")
            else:
                console.print("[dim]已退出；空会话未保存[/dim]")
            return EXIT_SUCCESS
        if command in {"/help", "帮助"}:
            _dayin_duihua_bangzhu()
            continue
        if command in {"/new", "新对话"}:
            huihua = cunchu.xinjian()
            history = []
            console.print(f"[green]已新建会话：{huihua.huihua_id}[/green]")
            continue
        if command in {"/clear", "清空"}:
            huihua.qingkong()
            history = []
            cunchu.baocun(huihua)
            console.print("[green]当前会话已清空[/green]")
            continue
        if command in {"/clear-history", "/clearhistory", "清除历史"}:
            console.print("[yellow]这会永久删除全部已保存的历史会话，但不会删除运行记录或行情缓存。[/yellow]")
            try:
                confirmation = console.input("请输入 [bold]确认清除[/bold] 继续，直接回车取消：").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]已取消清除历史[/dim]")
                continue
            if confirmation != "确认清除":
                console.print("[dim]已取消清除历史[/dim]")
                continue
            deleted = cunchu.qingkong_quanbu()
            huihua = cunchu.xinjian()
            history = []
            console.print(f"[green]已清除 {deleted} 个历史会话，并新建空白会话：{huihua.huihua_id}[/green]")
            continue
        if command in {"/sessions", "会话列表"}:
            _dayin_huihua_liebiao(cunchu, huihua.huihua_id)
            continue
        if command in {"/history", "历史"}:
            _dayin_dangqian_lishi(huihua)
            continue
        if command.startswith("/resume"):
            parts = prompt.split(maxsplit=1)
            if len(parts) != 2:
                console.print("[yellow]用法：/resume 会话ID[/yellow]")
                continue
            try:
                huihua = cunchu.duqu(parts[1])
            except HuihuaCuoWu as exc:
                console.print(f"[red]{_console_safe(exc)}[/red]")
                continue
            history = zhengli_xiaoxi(huihua.xiaoxi)
            console.print(f"[green]已切换：{huihua.biaoti}（{huihua.lunshu} 轮）[/green]")
            continue
        if prompt.startswith("/"):
            console.print("[yellow]未知会话命令。输入 /help 查看可用命令。[/yellow]")
            continue

        started = time.perf_counter()
        try:
            with console.status("[cyan]正在理解你的问题...[/cyan]", spinner="dots") as running:
                status_ref["value"] = running
                result = agent.run(user_message=prompt, history=history, session_id=huihua.huihua_id)
        except KeyboardInterrupt:
            console.print("\n[yellow]本轮已中断，会话保留在上一轮[/yellow]")
            continue
        except Exception as exc:
            console.print(Panel(_console_safe(exc), title="本轮失败", style="red"))
            continue
        finally:
            status_ref["value"] = None

        elapsed = time.perf_counter() - started
        status = result.get("status", "unknown")
        run_id = result.get("run_id", "")
        content = str(result.get("content") or result.get("reason") or "")

        if status == "success" and content:
            returned_history = result.get("history")
            if isinstance(returned_history, list):
                history = zhengli_xiaoxi(returned_history)
            else:
                history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                ])
            huihua.xiaoxi = history
            huihua.lunshu += 1
            huihua.shezhi_shouci_biaoti(prompt)
            try:
                cunchu.baocun(huihua)
            except OSError as exc:
                console.print(f"[yellow]会话暂未写入磁盘：{_console_safe(exc)}[/yellow]")
            console.print("\n[bold green]研究员 >[/bold green]")
            console.print(Markdown(_console_safe(content)))
            console.print(f"[dim]{elapsed:.1f}s | {run_id} | 会话已保存[/dim]")
        else:
            reason = content or "本轮没有生成有效回答"
            console.print(Panel(_console_safe(reason), title=f"本轮未完成 | {status} | {elapsed:.1f}s", style="red"))


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
                    f"Reasoning: {os.getenv('LANGCHAIN_REASONING_EFFORT', '(default)')}",
                    f"Service tier: {os.getenv('LANGCHAIN_SERVICE_TIER', 'standard')}",
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


def cmd_gupiao(
    gupiao: str,
    source: str,
    history_calendar_days: int,
    holding_days: int,
    budget_yuan: float | None,
    config_path: str | None,
    json_mode: bool,
) -> int:
    from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool

    result = json.loads(GupiaoFenxiTool().execute(
        gupiao=gupiao,
        source=source,
        history_calendar_days=history_calendar_days,
        holding_days=holding_days,
        budget_yuan=budget_yuan,
        config_path=config_path,
    ))
    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
    else:
        console.print_json(json.dumps(result, ensure_ascii=False))
    return EXIT_SUCCESS if result.get("status") == "ok" else EXIT_RUN_FAILED


def cmd_yuce(
    gupiao: str,
    source: str,
    history_calendar_days: int,
    config_path: str | None,
    json_mode: bool,
) -> int:
    """Run diagnosis once, then print gated forecasts for the next three market sessions."""
    from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
    from src.tools.gupiao_yuce_tool import GupiaoYuceTool

    diagnosis = json.loads(GupiaoFenxiTool().execute(
        gupiao=gupiao,
        source=source,
        history_calendar_days=history_calendar_days,
        holding_days=2,
        config_path=config_path,
    ))
    predictions: dict[str, Any] = {}
    analysis_id = diagnosis.get("analysis_id")
    if diagnosis.get("status") == "ok" and analysis_id:
        predictor = GupiaoYuceTool()
        for horizon in (1, 2, 3):
            predictions[f"T+{horizon}"] = json.loads(predictor.execute(
                analysis_id=analysis_id,
                horizon=horizon,
                mode="future_close",
                intent="forecast",
            ))
    compact = {
        "status": diagnosis.get("status"),
        "stock": diagnosis.get("stock"),
        "as_of": diagnosis.get("as_of"),
        "generated_at": diagnosis.get("generated_at"),
        "analysis_id": analysis_id,
        "predictions": predictions,
        "market_data": diagnosis.get("market_data"),
        "error": diagnosis.get("error"),
    }
    if json_mode:
        print(json.dumps(compact, ensure_ascii=False))
    else:
        console.print_json(json.dumps(compact, ensure_ascii=False))
    has_validated = any(item.get("forecast_status") == "validated" for item in predictions.values())
    return EXIT_SUCCESS if compact.get("status") == "ok" and has_validated else EXIT_RUN_FAILED


def cmd_bankuai(
    bankuai: str,
    bankuai_leixing: str,
    top_n: int,
    offset: int,
    selection_id: str | None,
    source: str,
    config_path: str | None,
    json_mode: bool,
) -> int:
    from src.ashare.bankuai_yuce import bankuai_xuangu

    result = bankuai_xuangu(
        bankuai=bankuai,
        bankuai_leixing=bankuai_leixing,
        top_n=top_n,
        offset=offset,
        selection_id=selection_id,
        source=source,
        config_path=config_path,
    )
    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
    else:
        console.print_json(json.dumps(result, ensure_ascii=False))
    return EXIT_SUCCESS if result.get("status") == "ok" else EXIT_RUN_FAILED


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="A 股 T+3 量化研究员")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="执行一次自然语言研究")
    run.add_argument("prompt", nargs="?", help="自然语言问题")
    run.add_argument("-p", "--prompt", dest="prompt_opt", help="自然语言问题")
    run.add_argument("--max-iter", type=int, default=50)
    run.add_argument("--json", action="store_true")

    chat = sub.add_parser("chat", help="进入可续接的终端连续对话")
    chat.add_argument("--max-iter", type=int, default=50)
    chat_session = chat.add_mutually_exclusive_group()
    chat_session.add_argument("--new", dest="new_session", action="store_true", help="新建空白会话")
    chat_session.add_argument("--session", help="按会话 ID 续接历史会话")

    list_cmd = sub.add_parser("list", help="列出最近运行记录")
    list_cmd.add_argument("--limit", type=int, default=20)

    gupiao = sub.add_parser("gupiao", help="按代码或名称直接分析一只 A 股")
    gupiao.add_argument("gupiao", help="例如 600519.SH 或贵州茅台")
    gupiao.add_argument(
        "--source",
        choices=["auto", "tushare", "akshare"],
        default="auto",
        help="股票名称解析和日线行情来源；基本面仍可能混合使用 Tushare/AKShare",
    )
    gupiao.add_argument("--history-calendar-days", type=int, default=1080)
    gupiao.add_argument("--holding-days", type=int, choices=[1, 2, 3], default=2)
    gupiao.add_argument("--budget-yuan", type=float, help="用于整手和最低佣金估算；不填写则使用成本配置默认资金")
    gupiao.add_argument("--config", dest="config_path", help="量化配置文件路径；默认使用项目根目录配置")
    gupiao.add_argument("--json", action="store_true")

    yuce = sub.add_parser("yuce", help="预测一只 A 股未来第 1、2、3 个交易日收盘")
    yuce.add_argument("gupiao", help="例如 600519.SH 或贵州茅台")
    yuce.add_argument(
        "--source",
        choices=["auto", "tushare", "akshare"],
        default="auto",
        help="股票名称解析和日线行情来源",
    )
    yuce.add_argument("--history-calendar-days", type=int, default=1080)
    yuce.add_argument("--config", dest="config_path", help="量化配置文件路径；默认使用项目根目录配置")
    yuce.add_argument("--json", action="store_true")

    bankuai = sub.add_parser("bankuai", help="从指定板块选股并预测 T+1/T+2/T+3")
    bankuai.add_argument("bankuai", help="中国大陆行业或概念板块名称")
    bankuai.add_argument("--type", dest="bankuai_leixing", choices=["auto", "hangye", "gainian"], default="auto")
    bankuai.add_argument("--top-n", type=int, default=8, help="每批候选数量，默认8且最多8")
    bankuai.add_argument("--offset", type=int, default=0, help="顺延批次的起始偏移；第一批为0")
    bankuai.add_argument("--selection-id", help="上一批返回的候选序列ID；offset大于0时必填")
    bankuai.add_argument(
        "--source",
        choices=["auto", "tushare", "akshare"],
        default="auto",
        help="个股日线行情来源；板块成分仍按独立的数据源顺序获取",
    )
    bankuai.add_argument("--config", dest="config_path", help="Path to lianghua_peizhi.json")
    bankuai.add_argument("--json", action="store_true")

    sub.add_parser("settings", help="查看当前运行配置")
    openai_login = sub.add_parser("openai-login", help="使用 ChatGPT OAuth 登录 OpenAI Provider")
    openai_login.add_argument("--no-browser", action="store_true", help="不自动打开登录网页")
    sub.add_parser("openai-logout", help="删除本机保存的 ChatGPT OAuth 登录")

    args = parser.parse_args(argv)

    if args.command == "run":
        prompt = args.prompt_opt or args.prompt
        if not prompt:
            console.print('[red]Missing prompt. Use: gpyj run -p "分析 600519.SH"[/red]')
            return EXIT_USAGE_ERROR
        return cmd_run(prompt, args.max_iter, json_mode=args.json)
    if args.command == "chat":
        return cmd_chat(args.max_iter, session_id=args.session, new_session=args.new_session)
    if args.command == "list":
        return cmd_list(args.limit)
    if args.command == "gupiao":
        return cmd_gupiao(
            args.gupiao,
            args.source,
            args.history_calendar_days,
            args.holding_days,
            args.budget_yuan,
            args.config_path,
            args.json,
        )
    if args.command == "yuce":
        return cmd_yuce(
            args.gupiao,
            args.source,
            args.history_calendar_days,
            args.config_path,
            args.json,
        )
    if args.command == "bankuai":
        return cmd_bankuai(
            args.bankuai,
            args.bankuai_leixing,
            args.top_n,
            args.offset,
            args.selection_id,
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

    return cmd_chat(50)


if __name__ == "__main__":
    sys.exit(main())
