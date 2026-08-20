"""UTF-8 JSON storage for resumable terminal conversations."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DUIHUA_MULU = Path.home() / ".gupiaoyanjiu" / "duihua"
HUIHUA_ID_GESHI = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")
YUNXU_JUESE = {"user", "assistant", "tool"}
ZUIDA_WENJIAN_ZIJIE = 20 * 1024 * 1024


class HuihuaCuoWu(ValueError):
    """Raised when a conversation ID or file is invalid."""


def _xianzai() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _shengcheng_id() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"


def _zhuan_json_anquan(value: Any) -> Any:
    """Return a detached JSON-safe value while preserving Chinese text."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def zhengli_xiaoxi(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Keep only valid chat roles and JSON-safe message fields."""
    cleaned: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict) or raw.get("role") not in YUNXU_JUESE:
            continue
        message = _zhuan_json_anquan(raw)
        content = message.get("content", "")
        if not isinstance(content, (str, list)):
            message["content"] = json.dumps(content, ensure_ascii=False, default=str)
        cleaned.append(message)
    return cleaned


def _chijiuhua_fenxi_biaoji(message: dict[str, Any]) -> dict[str, Any]:
    """把完整量化工具结果缩为跨进程不可复用的会话指代标记。"""
    copied = _zhuan_json_anquan(message)
    content = copied.get("content")
    try:
        payload = json.loads(content) if isinstance(content, str) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        copied["content"] = json.dumps(
            {
                "status": "reanalysis_required",
                "message": "历史量化工具结果未持久化；需要时必须重新获取远端数据。",
                "market_data_persistence": "none",
            },
            ensure_ascii=False,
        )
        return copied
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    requested_name = str(
        scope.get("requested_name") or scope.get("canonical_name") or ""
    ).strip()
    stock = payload.get("selected_stock") or payload.get("primary")
    single_stock = str(payload.get("analysis_type") or "") == "single_stock_analysis"
    stock_query = str(payload.get("query") or "").strip()
    if not stock_query and isinstance(stock, dict):
        stock_query = str(stock.get("ts_code") or stock.get("name") or "").strip()
    minimal_stock = (
        {
            key: stock.get(key)
            for key in ("ts_code", "name")
            if stock.get(key) is not None
        }
        if isinstance(stock, dict)
        else None
    )
    scope_request = {
        "fanwei": "single_stock" if single_stock else "named_scope" if requested_name else "all_market",
        "mingcheng": requested_name or None,
    }
    if single_stock:
        scope_request["gupiao"] = stock_query or None
    copied["content"] = json.dumps(
        {
            "status": "reanalysis_required",
            "outcome": "reanalysis_required",
            "previous_business_outcome": payload.get("outcome"),
            "stock_reference": minimal_stock,
            "scope_request": scope_request,
            "message": (
                "会话只保留范围和股票指代，不保存行情、板块成分、因子或预测输入；"
                "恢复会话后必须重新获取远端数据并分析。"
            ),
            "market_data_persistence": "none",
        },
        ensure_ascii=False,
    )
    return copied


def zhengli_chijiuhua_xiaoxi(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """保存会话时移除可被误作本地行情缓存的完整量化工具结果。"""
    persistent: list[dict[str, Any]] = []
    for message in zhengli_xiaoxi(messages):
        if message.get("role") == "tool" and message.get("name") == "gupiao_fenxi":
            persistent.append(_chijiuhua_fenxi_biaoji(message))
        else:
            persistent.append(message)
    return persistent


@dataclass
class DuihuaHuihua:
    """One resumable terminal conversation."""

    huihua_id: str
    biaoti: str = "新会话"
    chuangjian_shijian: str = field(default_factory=_xianzai)
    gengxin_shijian: str = field(default_factory=_xianzai)
    lunshu: int = 0
    xiaoxi: list[dict[str, Any]] = field(default_factory=list)

    def shezhi_shouci_biaoti(self, prompt: str) -> None:
        if self.biaoti != "新会话":
            return
        title = " ".join(prompt.strip().split())
        if title:
            self.biaoti = title[:36]

    def qingkong(self) -> None:
        self.biaoti = "新会话"
        self.lunshu = 0
        self.xiaoxi = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "banben": 1,
            "huihua_id": self.huihua_id,
            "biaoti": self.biaoti,
            "chuangjian_shijian": self.chuangjian_shijian,
            "gengxin_shijian": self.gengxin_shijian,
            "lunshu": self.lunshu,
            "xiaoxi": zhengli_chijiuhua_xiaoxi(self.xiaoxi),
        }


class DuihuaCunchu:
    """Create, save, load, and list terminal conversations."""

    def __init__(self, mulu: Path | None = None) -> None:
        self.mulu = (mulu or DUIHUA_MULU).expanduser().resolve()
        self.mulu.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _yanzheng_id(huihua_id: str) -> str:
        value = str(huihua_id).strip()
        if not HUIHUA_ID_GESHI.fullmatch(value):
            raise HuihuaCuoWu(f"无效的会话 ID：{value}")
        return value

    def _lujing(self, huihua_id: str) -> Path:
        return self.mulu / f"{self._yanzheng_id(huihua_id)}.json"

    def xinjian(self) -> DuihuaHuihua:
        return DuihuaHuihua(huihua_id=_shengcheng_id())

    def baocun(self, huihua: DuihuaHuihua) -> Path:
        path = self._lujing(huihua.huihua_id)
        huihua.gengxin_shijian = _xianzai()
        payload = json.dumps(huihua.to_dict(), ensure_ascii=False, indent=2)
        temp = self.mulu / f".{huihua.huihua_id}.{secrets.token_hex(3)}.tmp"
        try:
            temp.write_text(payload, encoding="utf-8")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def duqu(self, huihua_id: str) -> DuihuaHuihua:
        path = self._lujing(huihua_id)
        if not path.is_file():
            raise HuihuaCuoWu(f"找不到会话：{huihua_id}")
        if path.stat().st_size > ZUIDA_WENJIAN_ZIJIE:
            raise HuihuaCuoWu(f"会话文件过大，拒绝载入：{huihua_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HuihuaCuoWu(f"会话文件损坏：{huihua_id}") from exc
        if not isinstance(data, dict) or data.get("huihua_id") != huihua_id:
            raise HuihuaCuoWu(f"会话文件内容无效：{huihua_id}")
        try:
            lunshu = max(0, int(data.get("lunshu") or 0))
        except (TypeError, ValueError) as exc:
            raise HuihuaCuoWu(f"会话文件内容无效：{huihua_id}") from exc
        return DuihuaHuihua(
            huihua_id=huihua_id,
            biaoti=str(data.get("biaoti") or "新会话")[:80],
            chuangjian_shijian=str(data.get("chuangjian_shijian") or _xianzai()),
            gengxin_shijian=str(data.get("gengxin_shijian") or _xianzai()),
            lunshu=lunshu,
            xiaoxi=zhengli_xiaoxi(data.get("xiaoxi") or []),
        )

    def liechu(self, shuliang: int = 10) -> list[DuihuaHuihua]:
        sessions: list[DuihuaHuihua] = []
        paths = sorted(self.mulu.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in paths:
            if len(sessions) >= max(1, shuliang):
                break
            try:
                sessions.append(self.duqu(path.stem))
            except HuihuaCuoWu:
                continue
        return sessions

    def qingkong_quanbu(self) -> int:
        """Delete every saved conversation JSON file in the conversation directory."""
        deleted = 0
        for path in self.mulu.glob("*.json"):
            if not path.is_file() or not HUIHUA_ID_GESHI.fullmatch(path.stem):
                continue
            path.unlink()
            deleted += 1
        return deleted

    def zuijin(self) -> DuihuaHuihua | None:
        sessions = self.liechu(1)
        return sessions[0] if sessions else None


__all__ = [
    "DUIHUA_MULU",
    "DuihuaCunchu",
    "DuihuaHuihua",
    "HuihuaCuoWu",
    "zhengli_chijiuhua_xiaoxi",
    "zhengli_xiaoxi",
]
