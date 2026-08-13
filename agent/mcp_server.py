#!/usr/bin/env python3
"""Minimal MCP surface for the supported A-share research workflows."""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal

from fastmcp import FastMCP

from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
from src.tools.gupiao_yuce_tool import GupiaoYuceTool

_GUPIAO_FENXI = GupiaoFenxiTool()
_GUPIAO_YUCE = GupiaoYuceTool()

MCP_INSTRUCTIONS = """
Only provide personal-use research on mainland China A-share stocks. The server has exactly
two business tools: diagnosis and a single three-trading-day forecast. It never connects to brokers,
accepts trading credentials, submits orders, controls trading terminals, or performs
automatic trading. All outputs are research results for manual review.
""".strip()

mcp = FastMCP(
    name="A股分析与三交易日预测",
    instructions=MCP_INSTRUCTIONS,
    mask_error_details=True,
    strict_input_validation=True,
)


@mcp.tool(
    name="gupiao_fenxi",
    description=(
        "第一阶段：分析一只中国大陆 A 股的当前量化因子，返回行情时点、可交易性、基本面、估值、技术面、"
        "价量、相对强弱、同行与风险证据，不训练收益预测模型，并返回analysis_id。具体预测数值必须再调用gupiao_yuce。"
        "只做个人研究分析和三交易日预测，不替用户作交易决定，不连接券商或下单。"
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
    history_calendar_days: int = 1440,
) -> dict[str, Any]:
    """研究一只 A 股；gupiao 可传代码或中文名称。"""
    return json.loads(_GUPIAO_FENXI.execute(
        gupiao=gupiao,
        source=source,
        history_calendar_days=history_calendar_days,
    ))


@mcp.tool(
    name="gupiao_yuce",
    description=(
        "第二阶段：根据gupiao_fenxi返回的analysis_id，一次返回未来第1、2、3个交易日的方向、"
        "上涨可能性、参考收盘价、收益估计和可信度。"
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def gupiao_yuce(
    analysis_id: str,
) -> dict[str, Any]:
    """根据一次完成的分析发布未来三个交易日预测。"""
    return json.loads(_GUPIAO_YUCE.execute(
        analysis_id=analysis_id,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A股分析与三交易日预测 MCP 服务")
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
