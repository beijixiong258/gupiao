"""BaoStock 公共会话，用于按需补充单只沪深股票的历史资料。"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def bao_gupiao_shuju(code: str, start: str, end: str, *, timeout: float, include_factors: bool = True) -> dict[str, Any]:
    """BaoStock SDK 使用全局 TCP 会话与标准输出，子进程限定总耗时。"""
    try:
        command = [sys.executable, "-B", "-m", "ashare.baostock_api", code, start, end]
        if not include_factors:
            command.append("--skip-factors")
        result = subprocess.run(
            command,
            capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("BaoStock 公共会话超过本次取数时限") from exc
    if result.returncode != 0:
        raise RuntimeError("BaoStock 公共会话失败：" + " ".join(result.stderr.split())[:260])
    return json.loads(result.stdout)


def _bao_public_query(code: str, start: str, end: str, *, include_factors: bool = True) -> dict[str, Any]:
    import socket
    from contextlib import redirect_stdout

    import baostock as bs

    def rows(result: Any) -> list[dict[str, Any]]:
        items = []
        while result.error_code == "0" and result.next():
            items.append(dict(zip(result.fields, result.get_row_data(), strict=True)))
        if result.error_code != "0":
            raise RuntimeError(str(result.error_msg))
        return items

    # 仅在本次子进程中约束 SDK 创建的 socket。
    socket.setdefaulttimeout(6)
    with redirect_stdout(sys.stderr):
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(str(login.error_msg))
        try:
            data: dict[str, Any] = {"errors": {}}
            queries = [
                ("daily", bs.query_history_k_data_plus, (code, "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST", start, end),
                 {"adjustflag": "3"}),
                ("profile", bs.query_stock_basic, (code,), {}),
            ]
            if include_factors:
                queries.append(("factors", bs.query_adjust_factor, (code, "1990-12-19", end), {}))
            for field, query, args, kwargs in queries:
                try:
                    data[field] = rows(query(*args, **kwargs))
                except (RuntimeError, OSError, ValueError) as exc:
                    data["errors"][field] = str(exc)
            return data
        finally:
            bs.logout()


if __name__ == "__main__":
    try:
        print(json.dumps(_bao_public_query(*sys.argv[1:4], include_factors="--skip-factors" not in sys.argv[4:]), ensure_ascii=True))
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
