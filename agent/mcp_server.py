#!/usr/bin/env python3
"""Minimal MCP surface for the supported A-share research workflows."""

from __future__ import annotations

import argparse
import json
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from src.tools.gupiao_fenxi_tool import FenxiScope, GupiaoFenxiTool

_GUPIAO_FENXI = GupiaoFenxiTool()
_COUNT_PARAMETER = GupiaoFenxiTool.parameters["properties"]["shuliang"]
RequestedCount = Annotated[
    int,
    Field(ge=_COUNT_PARAMETER["minimum"], le=_COUNT_PARAMETER["maximum"]),
]

MCP_INSTRUCTIONS = """
Only provide personal-use research on mainland China A-share stocks. The server has exactly
one business tool for unified stock analysis and diagnosis. Natural-language assistants must hide tool
parameters and analysis_id from end users. When the user names a scope, pass the ordinary phrase
unchanged as named_scope; the analysis tool dynamically downloads industry and concept catalogs
and may return one plain-language clarification. Never guess an industry-versus-concept category.
For a named stock use single_stock and gupiao, then explain the returned buy/no-buy assessment,
evidence gaps, result validity and reassessment conditions. Prediction and model training are
not available. The server never connects to brokers,
accepts trading credentials, submits orders, controls trading terminals, or performs
automatic trading. All outputs are research results for manual review.
""".strip()

mcp = FastMCP(
    name="A股分析与诊断",
    instructions=MCP_INSTRUCTIONS,
    mask_error_details=True,
    strict_input_validation=True,
)


@mcp.tool(
    name="gupiao_fenxi",
    description=(
        "统一分析：点名单股时复用统一量化规则给出买入建议；在全市场或用户用日常语言描述的范围中，先动态发现并核验范围，再自动执行风险过滤、八组日K因子、基本面、形态、尾盘证据和排序，"
        "返回一只首选、最多四只备选或明确不推荐。内部analysis_id只供当前进程解释已有分析，不应向自然语言用户展示。"
        "只做个人股票研究分析与诊断，不替用户作交易决定，不连接券商或下单。"
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
def gupiao_fenxi(
    fanwei: FenxiScope = "all_market",
    mingcheng: str | None = None,
    gupiao: str | None = None,
    shuliang: RequestedCount | None = None,
) -> dict[str, Any]:
    """按自然语言解析出的范围执行统一选股分析。"""
    return json.loads(_GUPIAO_FENXI.execute(
        fanwei=fanwei,
        mingcheng=mingcheng,
        gupiao=gupiao,
        shuliang=shuliang,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A股分析与诊断 MCP 服务")
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
