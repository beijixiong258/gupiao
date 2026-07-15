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
            "xiaoxi": zhengli_xiaoxi(self.xiaoxi),
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

    def zuijin(self) -> DuihuaHuihua | None:
        sessions = self.liechu(1)
        return sessions[0] if sessions else None
