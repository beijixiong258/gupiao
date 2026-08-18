"""远端公开数据源的轻量 HTTP 访问边界。

该模块只解决当前程序已经存在的三类问题：Windows 代理策略、短暂网络失败重试、
以及把底层异常整理成可追踪的来源尝试记录。业务模块不直接依赖 ``httpx``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable
from urllib.parse import urlsplit

import httpx

from src.ashare.peizhi import jiazai_lianghua_peizhi


class LianjieMoshi(str, Enum):
    """远端连接是否继承 Windows/环境代理。"""

    ZHILIAN = "direct"
    XITONG_DAILI = "system_proxy"


@dataclass(frozen=True)
class QingqiuChangshi:
    """一次不会泄漏查询参数的远端尝试记录。"""

    provider: str
    host: str
    attempt: int
    outcome: str
    error_code: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "host": self.host,
            "attempt": self.attempt,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "detail": self.detail,
        }


class WangluoQingqiuYichang(RuntimeError):
    """所有候选端点都失败后的结构化异常。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool,
        attempts: Iterable[QingqiuChangshi] = (),
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.attempts = tuple(attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "retryable": self.retryable,
            "attempted_providers": [item.to_dict() for item in self.attempts],
        }


def _wangluo_peizhi() -> dict[str, Any]:
    config, _ = jiazai_lianghua_peizhi()
    section = config.get("wangluo", {})
    return section if isinstance(section, dict) else {}


def _lianjie_moshi(settings: dict[str, Any]) -> LianjieMoshi:
    raw = str(settings.get("domestic_connection_mode", "direct")).strip().lower()
    try:
        return LianjieMoshi(raw)
    except ValueError as exc:
        raise ValueError("wangluo.domestic_connection_mode 必须是 direct 或 system_proxy") from exc


def _cuowu_fenlei(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, httpx.TimeoutException):
        return "network_timeout", True
    if isinstance(exc, httpx.NetworkError):
        return "network_connection_failed", True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return "source_rate_limited", True
        if status in {408, 425, 500, 502, 503, 504}:
            return "source_http_temporary", True
        return "source_http_rejected", False
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "source_payload_invalid", True
    return "source_request_failed", True


def _anquan_xiangqing(exc: Exception) -> str:
    """只保留异常类别和短消息，不把 URL 查询参数写入运行记录。"""
    detail = " ".join(str(exc).split())
    if "?" in detail:
        detail = detail.split("?", 1)[0]
    return detail[:240]


class GongkaiShujuHTTPKehu:
    """遵循配置的短生命周期 HTTP 客户端，支持端点级回退。"""

    def __init__(self, provider: str, *, settings: dict[str, Any] | None = None) -> None:
        self.provider = str(provider).strip() or "unknown"
        self.settings = dict(settings or _wangluo_peizhi())
        mode = _lianjie_moshi(self.settings)
        timeout = httpx.Timeout(
            connect=float(self.settings.get("connect_timeout_seconds", 6.0)),
            read=float(self.settings.get("read_timeout_seconds", 20.0)),
            write=float(self.settings.get("read_timeout_seconds", 20.0)),
            pool=float(self.settings.get("connect_timeout_seconds", 6.0)),
        )
        self._client = httpx.Client(
            timeout=timeout,
            trust_env=mode is LianjieMoshi.XITONG_DAILI,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
            },
        )
        self.attempts: list[QingqiuChangshi] = []

    def __enter__(self) -> "GongkaiShujuHTTPKehu":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def qingqiu_json(
        self,
        endpoints: Iterable[str],
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """依次尝试候选端点，返回 JSON 对象和实际端点。"""
        candidates = tuple(dict.fromkeys(str(value).strip() for value in endpoints if str(value).strip()))
        if not candidates:
            raise ValueError("远端端点不能为空")
        maximum_attempts = int(self.settings.get("max_attempts_per_endpoint", 2))
        backoff = float(self.settings.get("retry_backoff_seconds", 0.35))
        last_error: Exception | None = None
        last_code = "source_request_failed"
        last_retryable = True
        for endpoint in candidates:
            host = urlsplit(endpoint).hostname or "unknown"
            for attempt in range(1, maximum_attempts + 1):
                try:
                    response = self._client.get(endpoint, params=params, headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("响应不是 JSON 对象")
                    self.attempts.append(
                        QingqiuChangshi(self.provider, host, attempt, "ok")
                    )
                    return payload, endpoint
                except Exception as exc:  # httpx 与 JSON 解码统一在此分类
                    last_error = exc
                    last_code, last_retryable = _cuowu_fenlei(exc)
                    self.attempts.append(
                        QingqiuChangshi(
                            self.provider,
                            host,
                            attempt,
                            "failed",
                            error_code=last_code,
                            detail=_anquan_xiangqing(exc),
                        )
                    )
                    if not last_retryable:
                        break
                    if attempt < maximum_attempts and backoff > 0:
                        time.sleep(backoff * (2 ** (attempt - 1)))
        message = "所有远端端点均不可用"
        if last_error is not None:
            message = f"{message}：{_anquan_xiangqing(last_error)}"
        raise WangluoQingqiuYichang(
            message,
            error_code=last_code,
            retryable=last_retryable,
            attempts=self.attempts,
        )

    def qingqiu_wenben(
        self,
        endpoints: Iterable[str],
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """文本版端点回退；主要供交易日历使用。"""
        candidates = tuple(dict.fromkeys(str(value).strip() for value in endpoints if str(value).strip()))
        maximum_attempts = int(self.settings.get("max_attempts_per_endpoint", 2))
        backoff = float(self.settings.get("retry_backoff_seconds", 0.35))
        last_error: Exception | None = None
        last_code = "source_request_failed"
        last_retryable = True
        for endpoint in candidates:
            host = urlsplit(endpoint).hostname or "unknown"
            for attempt in range(1, maximum_attempts + 1):
                try:
                    response = self._client.get(endpoint, params=params, headers=headers)
                    response.raise_for_status()
                    if not response.text.strip():
                        raise ValueError("响应文本为空")
                    self.attempts.append(QingqiuChangshi(self.provider, host, attempt, "ok"))
                    return response.text, endpoint
                except Exception as exc:
                    last_error = exc
                    last_code, last_retryable = _cuowu_fenlei(exc)
                    self.attempts.append(
                        QingqiuChangshi(
                            self.provider,
                            host,
                            attempt,
                            "failed",
                            error_code=last_code,
                            detail=_anquan_xiangqing(exc),
                        )
                    )
                    if not last_retryable:
                        break
                    if attempt < maximum_attempts and backoff > 0:
                        time.sleep(backoff * (2 ** (attempt - 1)))
        message = "所有远端端点均不可用"
        if last_error is not None:
            message = f"{message}：{_anquan_xiangqing(last_error)}"
        raise WangluoQingqiuYichang(
            message,
            error_code=last_code,
            retryable=last_retryable,
            attempts=self.attempts,
        )


__all__ = [
    "GongkaiShujuHTTPKehu",
    "LianjieMoshi",
    "QingqiuChangshi",
    "WangluoQingqiuYichang",
]
