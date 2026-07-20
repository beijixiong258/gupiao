#!/usr/bin/env python3
"""Minimal MCP surface for the supported A-share research workflows."""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal

from fastmcp import FastMCP

from src.ashare.bankuai_yuce import bankuai_xuangu as _bankuai_xuangu
from src.tools.gupiao_fenxi_tool import GupiaoFenxiTool
from src.tools.gupiao_yuce_tool import GupiaoYuceTool

_GUPIAO_FENXI = GupiaoFenxiTool()
_GUPIAO_YUCE = GupiaoYuceTool()

MCP_INSTRUCTIONS = """
Only provide research on mainland China A-share stocks. Single-stock work uses diagnosis
first and a separate request-specific T+1/T+2/T+3 forecast second. The server also exposes
board selection. It never connects to brokers,
accepts trading credentials, submits orders, controls trading terminals, or performs
automatic trading. All outputs are research results for manual review.
""".strip()

mcp = FastMCP(
    name="A股 T+3 量化研究员",
    version="0.3.0",
    instructions=MCP_INSTRUCTIONS,
    mask_error_details=True,
    strict_input_validation=True,
)


@mcp.tool(
    name="gupiao_fenxi",
    description=(
        "第一阶段：全面分析一只中国大陆 A 股，返回行情时点、可交易性、基本面、估值、技术面、"
        "波动、同行与风险证据，并返回analysis_id。具体预测数值必须再调用gupiao_yuce。"
        "只做分析和预测，不替用户作交易决定，不连接券商或下单。"
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
    history_calendar_days: int = 1080,
    holding_days: Literal[1, 2, 3] = 2,
    budget_yuan: float | None = None,
) -> dict[str, Any]:
    """研究一只 A 股；gupiao 可传代码或中文名称。"""
    return json.loads(_GUPIAO_FENXI.execute(
        gupiao=gupiao,
        source=source,
        history_calendar_days=history_calendar_days,
        holding_days=holding_days,
        budget_yuan=budget_yuan,
    ))


@mcp.tool(
    name="gupiao_yuce",
    description=(
        "第二阶段：根据gupiao_fenxi返回的analysis_id，只计算用户指定的T+1/T+2/T+3预测、扣除广义"
        "交易成本后的上涨空间，以及可选的持仓净收益。未通过样本外验证或时点已失效时不公开原始点预测。"
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
    horizon: Literal[1, 2, 3],
    mode: Literal["future_close", "holding_return"],
    intent: Literal["forecast", "buy_upside", "sell_upside"] = "forecast",
    buy_price: float | None = None,
    shares: int | None = None,
    position_value_yuan: float | None = None,
) -> dict[str, Any]:
    """发布一个经过验证门禁的指定周期预测，可同时计算持仓扣费收益。"""
    return json.loads(_GUPIAO_YUCE.execute(
        analysis_id=analysis_id,
        horizon=horizon,
        mode=mode,
        intent=intent,
        buy_price=buy_price,
        shares=shares,
        position_value_yuan=position_value_yuan,
    ))


@mcp.tool(
    name="bankuai_xuangu",
    description=(
        "从指定中国大陆 A 股行业或概念板块中量化筛选股票，预测 T+1、T+2、T+3 累计收益，"
        "并按样本外验证决定是否给出研究候选。每批最多8只，可用selection_id和offset稳定顺延。"
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
    top_n: int = 8,
    offset: int = 0,
    selection_id: str | None = None,
    source: Literal["auto", "tushare", "akshare"] = "auto",
) -> dict[str, Any]:
    """研究一个 A 股板块并返回最多 top_n 个通过验证的候选。"""
    return _bankuai_xuangu(
        bankuai=bankuai,
        bankuai_leixing=bankuai_leixing,
        top_n=top_n,
        offset=offset,
        selection_id=selection_id,
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
