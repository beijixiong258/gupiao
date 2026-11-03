"""轻量配置定位与环境加载；数据层无需导入模型或研究模块。"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "lianghua_peizhi.json"
RUNS_DIR = Path.home() / ".gupiaoyanjiu" / "runs"
RECYCLE_BIN_PATH = Path.home() / "Desktop" / "智能体回收站" / "垃圾桶"
INDUSTRY_REFERENCE_PATH = Path.home() / ".gupiaoyanjiu" / "industry_references.json"
_ENV_CANDIDATES = (
    Path.home() / ".gupiaoyanjiu" / ".env",
    PROJECT_ROOT / ".env",
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
