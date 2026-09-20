"""轻量配置定位与环境加载；数据层无需导入模型或研究模块。"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


AGENT_DIR = Path(__file__).resolve().parents[2]
_SOURCE_CONFIG = AGENT_DIR.parent / "lianghua_peizhi.json"
_PACKAGED_CONFIG = Path(__file__).resolve().parents[1] / "lianghua_peizhi.json"
# 保留源码根目录的显式配置覆盖，默认配置随 Python 包一同发布。
DEFAULT_CONFIG_PATH = _SOURCE_CONFIG if _SOURCE_CONFIG.is_file() else _PACKAGED_CONFIG
_ENV_CANDIDATES = (
    Path.home() / ".gupiaoyanjiu" / ".env",
    AGENT_DIR / ".env",
    Path.cwd() / ".env",
)
_dotenv_loaded = False


def ensure_dotenv() -> None:
    """按既有优先级加载一次凭据，保留进程中已设置的环境变量。"""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    for candidate in _ENV_CANDIDATES:
        if not candidate.is_file():
            continue
        if load_dotenv is not None:
            load_dotenv(dotenv_path=candidate, override=False)
        else:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip():
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break
    _dotenv_loaded = True
