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
two business tools: unified stock-selection analysis and a single three-trading-day forecast for a
qualified selected candidate. Natural-language assistants must hide tool parameters and analysis_id
from end users. When the user names a scope, pass the ordinary phrase unchanged as named_scope;
the analysis tool dynamically downloads industry and concept catalogs and may return one plain-language
clarification. Never guess an industry-versus-concept category. Prediction is expensive and may run only
after the completed quantitative result has been shown and the user confirms in a later message. The server never connects to brokers,
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
        "统一分析：在全市场或用户用日常语言描述的范围中，先动态发现并核验范围，再自动执行风险过滤、八组日K因子、基本面、形态、尾盘证据和排序，"
        "返回一只首选、最多四只备选或明确不推荐。内部analysis_id只供后续预测工具衔接，不应向自然语言用户展示。"
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
    fanwei: Literal["all_market", "named_scope"] = "all_market",
    mingcheng: str | None = None,
) -> dict[str, Any]:
    """按自然语言解析出的范围执行统一选股分析。"""
    return json.loads(_GUPIAO_FENXI.execute(
        fanwei=fanwei,
        mingcheng=mingcheng,
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
    gupiao: str | None = None,
) -> dict[str, Any]:
    """根据一次完成的分析预测首选，或预测明确点名的合格备选。"""
    return json.loads(_GUPIAO_YUCE.execute(
        analysis_id=analysis_id,
        gupiao=gupiao,
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
