#!/usr/bin/env python3
"""Minimal MCP surface for the supported A-share research workflows."""

from __future__ import annotations

import argparse
import json
from threading import Lock
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from src.tools.gupiao_fenxi_tool import FenxiScope, GupiaoFenxiTool

_GUPIAO_FENXI = GupiaoFenxiTool()
_ANALYSIS_LOCK = Lock()
_COUNT_PARAMETER = GupiaoFenxiTool.parameters["properties"]["shuliang"]
RequestedCount = Annotated[
    int,
    Field(ge=_COUNT_PARAMETER["minimum"], le=_COUNT_PARAMETER["maximum"]),
]

MCP_INSTRUCTIONS = """
Only provide personal-use research on mainland China A-share stocks. The server has exactly
one business tool for unified stock analysis and diagnosis. Natural-language assistants must hide tool
parameters and analysis_id from end users. Default selection uses daily_leaderboards: same completed-day
gain top 100 union turnover top 100, deduplicated, with historical statistics restricted to that pool.
Keep this fixed for the five-day test; do not backfill off-list stocks or claim a full-market optimum.
When the user names an industry/theme scope, pass the ordinary phrase
unchanged as named_scope; the entire Shanghai/Shenzhen main-board market uses main_board instead,
with its original wording in mingcheng. The analysis tool dynamically downloads industry and concept catalogs
and may return one plain-language clarification. Never guess an industry-versus-concept category.
For a named stock use single_stock and gupiao; pass an explicit conjecture through yanjiu_wenti,
then explain the observable claim assessment, counterevidence, gaps and reassessment conditions.
For selection carry explicit cap restrictions in shizhi and other explicit conditions in tiaojian.
Use null/omit shuliang for vague quantities. Preserve future holding/return objectives in yanjiu_wenti,
preserving unsupported horizons and guarantees. The supported forecast is next-session close relative to the
latest complete signal-session close, without fees, using 24 stock states conditioned on CSI300 20-session
direction and SW2021 level-two industry relative direction (96 joint states), with a 120-session lookback
and 20-session recency half-life. The explicit legacy_24 mode preserves the old observation separately.
Compare returned expected_next_day_return, chronological checks and actual coverage; preserve return ties.
Observation order breaks exact return ties using industry_excess_ret_20 then stock_excess_ret_20, without
changing forecasts. Preserve complete cutoff boundary ties and first-decisive-field explanations.
Never add risk preferences or invent bounds. A verified scope_review_required can be resolved once with
fanwei_xuanze while preserving original_request, or clarified if truly ambiguous.
Fixed-rule historical estimates are available but are not guaranteed future outcomes. Model training is not available. The server never connects to brokers,
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
    description=GupiaoFenxiTool.description,
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
def gupiao_fenxi(
    fanwei: FenxiScope = "daily_leaderboards",
    mingcheng: str | None = None,
    gupiao: str | None = None,
    shuliang: RequestedCount | None = None,
    shizhi: Annotated[dict[str, Any] | None, Field(json_schema_extra=GupiaoFenxiTool.parameters["properties"]["shizhi"])] = None,
    tiaojian: Annotated[dict[str, Any] | None, Field(json_schema_extra=GupiaoFenxiTool.parameters["properties"]["tiaojian"])] = None,
    yanjiu_wenti: Annotated[dict[str, Any] | None, Field(json_schema_extra=GupiaoFenxiTool.parameters["properties"]["yanjiu_wenti"])] = None,
    fanwei_xuanze: Annotated[dict[str, Any] | None, Field(json_schema_extra=GupiaoFenxiTool.parameters["properties"]["fanwei_xuanze"])] = None,
) -> dict[str, Any]:
    """按自然语言解析出的范围执行统一选股分析。"""
    with _ANALYSIS_LOCK:
        # 新调用重新取数；只有当前范围审查的第二步复用本次目录。
        if fanwei_xuanze is None:
            _GUPIAO_FENXI.reset_request()
        result = json.loads(_GUPIAO_FENXI.execute(
            fanwei=fanwei, mingcheng=mingcheng, gupiao=gupiao, shuliang=shuliang,
            shizhi=shizhi, tiaojian=tiaojian, yanjiu_wenti=yanjiu_wenti, fanwei_xuanze=fanwei_xuanze,
        ))
        if result.get("status") != "scope_review_required":
            _GUPIAO_FENXI.reset_request()
        return result


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
