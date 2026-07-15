#!/usr/bin/env python3
"""Minimal MCP surface for the two supported A-share research workflows."""

from __future__ import annotations

import argparse
from typing import Any, Literal

from fastmcp import FastMCP

from src.ashare.bankuai_yuce import bankuai_xuangu as _bankuai_xuangu
from src.ashare.gupiao_yanjiu import fenxi_gupiao as _fenxi_gupiao

MCP_INSTRUCTIONS = """
Only provide research on mainland China A-share stocks. The server exposes single-stock
analysis and board selection with T+1/T+2/T+3 forecasts. It never connects to brokers,
accepts trading credentials, submits orders, controls trading terminals, or performs
automatic trading. All outputs are research results for manual review.
""".strip()

mcp = FastMCP(
    name="A股 T+3 量化研究员",
    version="0.2.0",
    instructions=MCP_INSTRUCTIONS,
    mask_error_details=True,
    strict_input_validation=True,
)


@mcp.tool(
    name="gupiao_fenxi",
    description=(
        "分析一只中国大陆 A 股，返回基本资料、估值、财务指标、技术指标、A 股交易规则、"
        "数据来源和风险。只做研究，不连接券商或下单。"
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
def gupiao_fenxi(
    gupiao: str,
    source: Literal["auto", "tushare", "akshare"] = "auto",
    history_calendar_days: int = 540,
) -> dict[str, Any]:
    """研究一只 A 股；gupiao 可传代码或中文名称。"""
    return _fenxi_gupiao(
        gupiao=gupiao,
        source=source,
        history_calendar_days=history_calendar_days,
    )


@mcp.tool(
    name="bankuai_xuangu",
    description=(
        "从指定中国大陆 A 股行业或概念板块中量化筛选股票，预测 T+1、T+2、T+3 累计收益，"
        "并按样本外验证决定是否给出研究候选。只做研究，不连接券商或下单。"
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
def bankuai_xuangu(
    bankuai: str,
    bankuai_leixing: Literal["auto", "hangye", "gainian"] = "auto",
    top_n: int = 3,
    source: Literal["auto", "tushare", "akshare"] = "auto",
) -> dict[str, Any]:
    """研究一个 A 股板块并返回最多 top_n 个通过验证的候选。"""
    return _bankuai_xuangu(
        bankuai=bankuai,
        bankuai_leixing=bankuai_leixing,
        top_n=top_n,
        source=source,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A股 T+3 量化研究员 MCP 服务")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址；默认只允许本机访问")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        mcp.run(transport="http", host=args.host, port=args.port, show_banner=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
