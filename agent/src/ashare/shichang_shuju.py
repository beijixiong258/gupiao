"""统一的 A 股市场时钟、横截面、板块成分和批量行情入口。"""

from __future__ import annotations

import os
import ast
import importlib.util
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src.ashare.peizhi import jiazai_lianghua_peizhi
from src.ashare.dongcai_api import (
    DONGCAI_PUBLIC_TOKEN,
    dongcai_fenye_duqu,
    dongcai_rili_kxian_duqu,
)
from src.ashare.fanwei_faxian import (
    BankuaiLeixing,
    FanweiFaxianJieguo,
    ShichangFanwei,
    faxian_fenxi_fanwei,
    huoqu_dongcai_chengfen,
)
from src.ashare.shuju_yuan import (
    _latest_tushare_daily,
    _normalize_code,
    _tushare_pro,
    huoqu_gupiao_jichu_ziliao,
)
from src.ashare.tengxun_api import tengxun_qfq_rili_duqu
from src.ashare.wangluo_kehu import GongkaiShujuHTTPKehu, WangluoQingqiuYichang

_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_NO_PROXY_ENV_NAMES = ("NO_PROXY", "no_proxy")
_BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _beijing_now() -> datetime:
    """返回不带时区对象的北京时间，便于与现有行情时间字段比较。"""
    return datetime.now(_BEIJING_TIMEZONE).replace(tzinfo=None)


class JiaoyiJieduan(str, Enum):
    """A 股本地交易阶段；交易日身份由交易日历决定。"""

    RILI_BUKE_YONG = "calendar_unavailable"
    FEI_JIAOYIRI = "non_trading_day"
    PANQIAN = "pre_market"
    JIHE_JINGJIA = "opening_auction"
    JIAOYI = "trading"
    WUJIAN_XIUSHI = "midday_break"
    SHOUPAN_DAIDING = "close_pending"
    PANHOU = "post_close"


@dataclass(frozen=True)
class JiaoyiRili:
    """一次请求内使用的交易日历快照。"""

    open_dates: frozenset[pd.Timestamp]
    source: str
    start_date: str
    end_date: str
    warnings: tuple[str, ...] = ()
    fetched_at: str | None = None
    attempted_providers: tuple[dict[str, Any], ...] = ()

    def shi_jiaoyiri(self, value: datetime | pd.Timestamp | str) -> bool:
        day = pd.Timestamp(value).normalize()
        return day in self.open_dates

    def zuijin_jiaoyiri(self, value: datetime | pd.Timestamp | str, *, include: bool = True) -> pd.Timestamp | None:
        day = pd.Timestamp(value).normalize()
        candidates = [
            item
            for item in self.open_dates
            if item <= day and (include or item < day)
        ]
        return max(candidates) if candidates else None


@dataclass
class FenxiShujuShangxiawen:
    """一次选股请求的数据工作单元；只在当前调用内复用远端响应。"""

    reference: datetime = field(default_factory=_beijing_now)
    source: str = "auto"
    _memo: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def jiaoyi_rili(self) -> JiaoyiRili:
        if "jiaoyi_rili" not in self._memo:
            self._memo["jiaoyi_rili"] = huoqu_jiaoyi_rili(self.reference)
        return self._memo["jiaoyi_rili"]

    def shichang_shizhong(self) -> dict[str, Any]:
        if "shichang_shizhong" not in self._memo:
            self._memo["shichang_shizhong"] = shichang_shizhong(
                self.reference,
                calendar=self.jiaoyi_rili(),
            )
        return dict(self._memo["shichang_shizhong"])

    def zuixin_wanzheng_jiaoyiri(self) -> pd.Timestamp:
        memo_key = "zuixin_wanzheng_jiaoyiri"
        if memo_key not in self._memo:
            self._memo[memo_key] = zuixin_wanzheng_jiaoyiri(
                self.reference,
                calendar=self.jiaoyi_rili(),
            )
        return pd.Timestamp(self._memo[memo_key])

    def shishi_kuaizhao(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """整次分析至多读取一次全市场实时快照。"""
        if "shishi_kuaizhao" not in self._memo:
            self._memo["shishi_kuaizhao"] = huoqu_shishi_kuaizhao(self.reference)
        frame, meta = self._memo["shishi_kuaizhao"]
        return frame.copy(), dict(meta)

    def dangu_kuaizhao(self, code: str) -> dict[str, Any]:
        table, meta = self.shishi_kuaizhao()
        normalized = _normalize_code(code)
        hit = table[table["ts_code"].astype(str).eq(normalized)] if not table.empty else pd.DataFrame()
        if hit.empty:
            return {
                "status": "unavailable",
                "source": meta.get("source"),
                "captured_at": meta.get("captured_at"),
                "error": f"实时快照未找到 {normalized}",
            }
        return {"status": "ok", **_json_safe_record(hit.iloc[0].to_dict()), **meta}

    def zuixin_hengjiemian(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if "zuixin_hengjiemian" not in self._memo:
            self._memo["zuixin_hengjiemian"] = huoqu_zuixin_hengjiemian(
                self.zuixin_wanzheng_jiaoyiri(),
                realtime_loader=self.shishi_kuaizhao,
            )
        frame, meta = self._memo["zuixin_hengjiemian"]
        return frame.copy(), dict(meta)

    def faxian_fanwei(self, name: str) -> FanweiFaxianJieguo:
        """同一分析请求只读取一次行业和概念目录。"""
        key = f"fanwei_faxian:{' '.join(str(name).split())}"
        if key not in self._memo:
            self._memo[key] = faxian_fenxi_fanwei(name)
        return self._memo[key]

    def bankuai_chengfen(
        self,
        scope: ShichangFanwei | str,
        *,
        board_type: str = "auto",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        identity = scope.code if isinstance(scope, ShichangFanwei) else str(scope).strip()
        key = f"bankuai:{board_type}:{identity}"
        if key not in self._memo:
            self._memo[key] = huoqu_bankuai_chengfen(scope, bankuai_leixing=board_type)
        frame, meta = self._memo[key]
        return frame.copy(), dict(meta)

    def piliang_lishi(
        self,
        codes: Iterable[str],
        *,
        start_date: str,
        end_date: str,
        minimum_rows: int,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        normalized = tuple(dict.fromkeys(_normalize_code(code) for code in codes))
        memo_key = f"piliang_lishi:{start_date}:{end_date}:{minimum_rows}:{'|'.join(normalized)}"
        if memo_key not in self._memo:
            self._memo[memo_key] = huoqu_piliang_qfq_lishi(
                normalized,
                start_date=start_date,
                end_date=end_date,
                minimum_rows=minimum_rows,
                calendar=self.jiaoyi_rili(),
            )
        histories, meta = self._memo[memo_key]
        return {code: frame.copy() for code, frame in histories.items()}, dict(meta)

    def fenzhong_xingqing(self, code: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        normalized = _normalize_code(code)
        key = f"fenzhong:{normalized}"
        if key not in self._memo:
            self._memo[key] = huoqu_fenzhong_xingqing(normalized, reference=self.reference)
        frame, meta = self._memo[key]
        return frame.copy(), dict(meta)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            result[str(key)] = None
        elif isinstance(value, pd.Timestamp):
            result[str(key)] = value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _shi_a_gu(value: Any) -> bool:
    try:
        _normalize_code(str(value))
        return True
    except ValueError:
        return False


def _akshare_bypass_proxy_enabled() -> bool:
    override = os.getenv("GPYJ_AKSHARE_BYPASS_PROXY", "").strip().lower()
    if override:
        return override not in {"0", "false", "no", "off"}
    try:
        config, _ = jiazai_lianghua_peizhi()
        network = config.get("wangluo", {})
        return bool(
            isinstance(network, dict)
            and str(network.get("domestic_connection_mode", "direct")).strip().lower()
            == "direct"
        )
    except Exception:
        return True


@contextmanager
def akshare_zhilian():
    """访问境内 AKShare 数据时临时绕过环境和 Windows 系统代理。

    Requests 在 Windows 上即使没有 ``HTTP_PROXY`` 也会读取系统代理。设置
    ``NO_PROXY=*`` 才能让 AKShare 内部新建的 Session 一并执行直连策略。
    新增的数据源适配器优先使用 ``httpx.Client(trust_env=False)``，此上下文只服务于
    仍由 AKShare 封装的实时行情接口。
    """
    if not _akshare_bypass_proxy_enabled():
        yield
        return
    managed_names = (*_PROXY_ENV_NAMES, *_NO_PROXY_ENV_NAMES)
    saved = {name: os.environ[name] for name in managed_names if name in os.environ}
    try:
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        for name in _NO_PROXY_ENV_NAMES:
            os.environ[name] = "*"
        yield
    finally:
        for name in managed_names:
            os.environ.pop(name, None)
        os.environ.update(saved)


def _tushare_error_code(exc: Exception) -> tuple[str, bool]:
    text = str(exc).lower()
    if any(marker in text for marker in ("每分钟", "频率", "rate", "limit")):
        return "source_rate_limited", True
    if any(marker in text for marker in ("权限", "permission", "积分")):
        return "source_permission_denied", False
    if any(marker in text for marker in ("timeout", "timed out", "connection", "连接")):
        return "network_connection_failed", True
    return "source_request_failed", True


def _akshare_jiben_nengli() -> dict[str, Any]:
    """读取安装版本而不导入 AKShare 的庞大顶层模块。"""
    try:
        installed = package_version("akshare")
    except PackageNotFoundError as exc:
        raise RuntimeError("AKShare 未安装，无法使用其数据格式解码能力") from exc
    spec = importlib.util.find_spec("akshare")
    package_paths = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not package_paths:
        raise RuntimeError("无法定位 AKShare 安装目录")
    return {"version": installed, "package_path": package_paths[0]}


def _jiancha_akshare_hanshu(ak: Any, *required: str) -> dict[str, Any]:
    """在调用前检查安装版本和接口能力，避免升级后以 AttributeError 模糊失败。"""
    capability = _akshare_jiben_nengli()
    missing = [name for name in required if not callable(getattr(ak, name, None))]
    if missing:
        raise RuntimeError(
            f"AKShare {capability['version']} 缺少所需接口：{', '.join(missing)}；请升级或更新适配器"
        )
    return {
        "akshare_version": capability["version"],
        "required_interfaces": list(required),
        "capability_check": "passed",
    }


def _duqu_akshare_rili_jiemao() -> tuple[str, dict[str, Any]]:
    """从已安装 AKShare 源码读取新浪公开格式解码器，并显式校验能力。

    这里只读取第三方库的算法常量，不读取任何本地市场数据，也不执行该源码文件。
    """
    capability = _akshare_jiben_nengli()
    source_path = Path(str(capability["package_path"])) / "stock" / "cons.py"
    if not source_path.is_file():
        raise RuntimeError("当前 AKShare 缺少 stock/cons.py，新浪交易日历解码能力不可用")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "hk_js_decode" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str) or "function d" not in value:
            break
        return value, {"akshare_version": capability["version"], "decoder_symbol": "hk_js_decode"}
    raise RuntimeError("当前 AKShare 不再提供 hk_js_decode，需升级适配器")


def _huoqu_xinlang_jiaoyi_rili(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[frozenset[pd.Timestamp], dict[str, Any]]:
    """从新浪公开交易日历取数；请求和解码均不落盘。"""
    endpoint = "https://finance.sina.com.cn/realstock/company/klc_td_sh.txt"
    with GongkaiShujuHTTPKehu("sina_exchange_calendar") as client:
        text, used_endpoint = client.qingqiu_wenben((endpoint,))
        attempts = tuple(item.to_dict() for item in client.attempts)
    match = text.split("=", 1)
    if len(match) != 2:
        raise ValueError("新浪交易日历响应格式无效")
    encoded = match[1].split(";", 1)[0].strip().strip('"')
    if not encoded:
        raise ValueError("新浪交易日历响应缺少编码数据")
    decoder, capability = _duqu_akshare_rili_jiemao()
    try:
        from py_mini_racer import MiniRacer
    except ImportError as exc:
        raise RuntimeError("当前 AKShare 运行环境缺少交易日历解码组件") from exc
    runtime = MiniRacer()
    runtime.eval(decoder)
    decoded = runtime.call("d", encoded)
    values = pd.to_datetime(pd.Series(decoded), errors="coerce", utc=True).dropna()
    if values.empty:
        raise ValueError("新浪交易日历解码后为空")
    normalized = values.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    dates = frozenset(pd.Timestamp(value) for value in normalized if start <= value <= end)
    if not dates:
        raise ValueError("新浪交易日历不覆盖所需日期范围")
    return dates, {
        "source": "sina_exchange_calendar",
        "endpoint_host": "finance.sina.com.cn",
        "fetched_at": _beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
        "attempted_providers": attempts,
        "persistence": "none",
        **capability,
    }


def huoqu_jiaoyi_rili(reference: datetime | None = None) -> JiaoyiRili:
    """读取远端交易日历；Tushare 受限时降级新浪，不用星期数冒充。"""
    current = reference or _beijing_now()
    start = pd.Timestamp(current.date()) - pd.Timedelta(days=550)
    end = pd.Timestamp(current.date()) + pd.Timedelta(days=45)
    warnings: list[str] = []
    attempted: list[dict[str, Any]] = []
    config, _ = jiazai_lianghua_peizhi()
    network_settings = config.get("wangluo", {}) if isinstance(config.get("wangluo"), dict) else {}
    maximum_attempts = int(network_settings.get("tushare_max_attempts", 2))
    backoff = float(network_settings.get("retry_backoff_seconds", 0.35))
    pro = _tushare_pro()
    for attempt in range(1, maximum_attempts + 1):
        try:
            frame = pro.trade_cal(
                exchange="",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            if frame is None or frame.empty or not {"cal_date", "is_open"}.issubset(frame.columns):
                raise RuntimeError("Tushare 交易日历为空或字段不完整")
            dates = pd.to_datetime(
                frame.loc[pd.to_numeric(frame["is_open"], errors="coerce").eq(1), "cal_date"],
                errors="coerce",
            ).dropna()
            attempted.append(
                {"provider": "tushare_trade_cal", "attempt": attempt, "outcome": "ok"}
            )
            fetched_at = _beijing_now().strftime("%Y-%m-%d %H:%M:%S")
            return JiaoyiRili(
                open_dates=frozenset(pd.Timestamp(value).normalize() for value in dates),
                source="tushare_trade_cal",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                fetched_at=fetched_at,
                attempted_providers=tuple(attempted),
            )
        except Exception as exc:
            error_code, retryable = _tushare_error_code(exc)
            attempted.append(
                {
                    "provider": "tushare_trade_cal",
                    "attempt": attempt,
                    "outcome": "failed",
                    "error_code": error_code,
                    "detail": " ".join(str(exc).split())[:180],
                }
            )
            if not retryable or error_code == "source_rate_limited":
                break
            if attempt < maximum_attempts and backoff > 0:
                time.sleep(backoff * (2 ** (attempt - 1)))
    warnings.append("Tushare 交易日历当前受限，已切换远端备用日历")
    try:
        dates, metadata = _huoqu_xinlang_jiaoyi_rili(start=start, end=end)
        attempted.extend(metadata.get("attempted_providers", []))
        return JiaoyiRili(
            open_dates=dates,
            source="sina_exchange_calendar",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            warnings=tuple(warnings),
            fetched_at=str(metadata.get("fetched_at") or ""),
            attempted_providers=tuple(attempted),
        )
    except WangluoQingqiuYichang as exc:
        attempted.extend(item.to_dict() for item in exc.attempts)
        warnings.append(f"新浪备用交易日历网络失败：{exc}")
    except Exception as exc:
        warnings.append(f"新浪备用交易日历解析失败：{exc}")
    return JiaoyiRili(
        open_dates=frozenset(),
        source="unavailable",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        warnings=tuple(warnings),
        fetched_at=_beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
        attempted_providers=tuple(attempted),
    )


def shichang_shizhong(
    reference: datetime | None = None,
    *,
    calendar: JiaoyiRili | None = None,
) -> dict[str, Any]:
    """根据交易日历和上海本地时钟返回唯一市场阶段。"""
    current = reference or _beijing_now()
    resolved_calendar = calendar or huoqu_jiaoyi_rili(current)
    minute = current.hour * 60 + current.minute
    if not resolved_calendar.open_dates:
        status = JiaoyiJieduan.RILI_BUKE_YONG
        note = "交易日历不可用，不能判断当前是否为交易日"
    elif not resolved_calendar.shi_jiaoyiri(current):
        status = JiaoyiJieduan.FEI_JIAOYIRI
        note = "交易日历确认当前休市，只使用最近完整收盘日"
    elif minute < 9 * 60 + 15:
        status = JiaoyiJieduan.PANQIAN
        note = "开盘前，只使用最近完整收盘日"
    elif minute < 9 * 60 + 30:
        status = JiaoyiJieduan.JIHE_JINGJIA
        note = "集合竞价阶段，实时快照仅作可交易性参考"
    elif minute < 11 * 60 + 30:
        status = JiaoyiJieduan.JIAOYI
        note = "交易时段，完整日线与实时证据分开保存"
    elif minute < 13 * 60:
        status = JiaoyiJieduan.WUJIAN_XIUSHI
        note = "午间休市，完整日线与实时证据分开保存"
    elif minute < 15 * 60:
        status = JiaoyiJieduan.JIAOYI
        note = "交易时段，完整日线与实时证据分开保存"
    elif minute < 15 * 60 + 5:
        status = JiaoyiJieduan.SHOUPAN_DAIDING
        note = "刚收盘，等待数据源确认完整日线"
    else:
        status = JiaoyiJieduan.PANHOU
        note = "收盘后，使用数据源已确认的最新完整日线"
    return {
        "captured_at": current.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Shanghai",
        "session_status": status.value,
        "is_trading_day": (
            None
            if status is JiaoyiJieduan.RILI_BUKE_YONG
            else status is not JiaoyiJieduan.FEI_JIAOYIRI
        ),
        "calendar_source": resolved_calendar.source,
        "calendar_fetched_at": resolved_calendar.fetched_at,
        "calendar_attempted_providers": [
            dict(value) for value in resolved_calendar.attempted_providers
        ],
        "calendar_precision": (
            "远端交易日历"
            if resolved_calendar.source in {"tushare_trade_cal", "sina_exchange_calendar"}
            else "不可用；没有用系统星期数替代法定交易日历"
        ),
        "calendar_warnings": list(resolved_calendar.warnings),
        "analysis_basis": note,
    }


def zuixin_wanzheng_jiaoyiri(
    reference: datetime | None = None,
    *,
    calendar: JiaoyiRili | None = None,
) -> pd.Timestamp:
    current = reference or _beijing_now()
    resolved_calendar = calendar or huoqu_jiaoyi_rili(current)
    today = pd.Timestamp(current.date())
    today_completed = resolved_calendar.shi_jiaoyiri(today) and (
        current.hour * 60 + current.minute >= 15 * 60 + 5
    )
    latest = resolved_calendar.zuijin_jiaoyiri(today, include=today_completed)
    if latest is None:
        raise RuntimeError("交易日历中找不到最近完整交易日")
    return latest


def _normalize_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "代码": "ts_code",
        "名称": "name",
        "最新价": "latest_price",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "昨收": "previous_close",
        "涨跌幅": "pct_chg",
        "成交量": "volume",
        "成交额": "amount_yuan",
        "换手率": "turnover_rate",
        "量比": "volume_ratio",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "total_market_value_yuan",
        "流通市值": "circulating_market_value_yuan",
    }
    data = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    if "ts_code" not in data.columns:
        raise RuntimeError("全市场快照缺少股票代码")
    data["ts_code"] = data["ts_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    data = data[data["ts_code"].map(_shi_a_gu)].copy()
    data["ts_code"] = data["ts_code"].map(_normalize_code)
    if "name" not in data.columns:
        data["name"] = ""
    data["name"] = data["name"].fillna("").astype(str)
    numeric_columns = [
        "latest_price",
        "open",
        "high",
        "low",
        "previous_close",
        "pct_chg",
        "volume",
        "amount_yuan",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "total_market_value_yuan",
        "circulating_market_value_yuan",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "volume" in data.columns:
        # 东方财富现货接口以“手”为单位；统一转换为股，与日线口径一致。
        data["volume"] = data["volume"] * 100.0
    return data.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def _huoqu_dongcai_shishi_kuaizhao(
    captured: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """直接读取东方财富沪深京 A 股完整分页，避免随机子域和系统代理漂移。"""
    with GongkaiShujuHTTPKehu("eastmoney_live_a_share_snapshot") as client:
        rows, endpoint, reported_total = dongcai_fenye_duqu(
            client,
            base_params={
                "po": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fid": "f12",
                "ut": DONGCAI_PUBLIC_TOKEN,
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
                "fields": (
                    "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f14,f15,f16,f17,f18,"
                    "f20,f21,f23"
                ),
            },
        )
        attempts = tuple(client.attempts)
    raw = pd.DataFrame(rows).rename(
        columns={
            "f12": "代码",
            "f14": "名称",
            "f2": "最新价",
            "f3": "涨跌幅",
            "f5": "成交量",
            "f6": "成交额",
            "f8": "换手率",
            "f9": "市盈率-动态",
            "f10": "量比",
            "f15": "最高",
            "f16": "最低",
            "f17": "今开",
            "f18": "昨收",
            "f20": "总市值",
            "f21": "流通市值",
            "f23": "市净率",
        }
    )
    data = _normalize_snapshot(raw)
    if data.empty:
        raise RuntimeError("东方财富全市场实时行情没有合法 A 股")
    host_counts = Counter(item.host for item in attempts if item.outcome == "ok")
    failures = [
        item.to_dict()
        for item in attempts
        if item.outcome == "failed"
    ]
    return data, {
        "status": "ok",
        "source": "eastmoney_live_a_share_snapshot",
        "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(data)),
        "reported_rows": int(reported_total),
        "endpoint_host": urlsplit(endpoint).hostname,
        "successful_page_hosts": dict(host_counts),
        "failed_attempt_count": len(failures),
        "failed_attempt_examples": failures[:5],
        "provider_trade_date": None,
        "timeliness": "接口不返回逐行交易日期，盘中证据均标记为暂定",
        "persistence": "none",
    }


def huoqu_shishi_kuaizhao(reference: datetime | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    captured = reference or _beijing_now()
    direct_error: Exception | None = None
    try:
        return _huoqu_dongcai_shishi_kuaizhao(captured)
    except Exception as exc:
        direct_error = exc
    try:
        import akshare as ak

        capability = _jiancha_akshare_hanshu(ak, "stock_zh_a_spot_em")
        with akshare_zhilian():
            raw = ak.stock_zh_a_spot_em()
        if raw is None or raw.empty:
            raise RuntimeError("AKShare 全市场实时行情为空")
        data = _normalize_snapshot(raw)
        return data, {
            "status": "ok",
            "source": "akshare_eastmoney_spot",
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(len(data)),
            "provider_trade_date": None,
            "timeliness": "接口不返回逐行交易日期，盘中证据均标记为暂定",
            "warnings": [f"东方财富直连接口失败后使用 AKShare 适配器：{' '.join(str(direct_error).split())[:160]}"],
            **capability,
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "unavailable",
            "source": "eastmoney_snapshot_and_akshare_fallback",
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
            "error_code": "live_snapshot_unavailable",
            "retryable": True,
            "error": (
                "东方财富直连接口与 AKShare 备用适配器均不可用："
                f"{' '.join(str(direct_error).split())[:120]}；{' '.join(str(exc).split())[:120]}"
            ),
        }


def huoqu_dangqian_kuaizhao(code: str, reference: datetime | None = None) -> dict[str, Any]:
    """从统一全市场快照中取出一只股票，保持单股研究的字段语义。"""
    data, meta = huoqu_shishi_kuaizhao(reference)
    normalized = _normalize_code(code)
    hit = data[data["ts_code"].astype(str).eq(normalized)] if not data.empty else pd.DataFrame()
    if hit.empty:
        return {
            "status": "unavailable",
            "source": meta.get("source"),
            "captured_at": meta.get("captured_at"),
            "error": str(meta.get("error") or f"实时行情未找到 {normalized}"),
            "note": "实时快照不可用时，分析退回最近完整日线，不伪装成实时价格",
        }
    row = hit.iloc[0]
    return {
        "status": "ok",
        "source": meta.get("source"),
        "captured_at": meta.get("captured_at"),
        "provider_trade_date": meta.get("provider_trade_date"),
        "timeliness": meta.get("timeliness"),
        "name": row.get("name"),
        "last_price": _number(row.get("latest_price")),
        "open": _number(row.get("open")),
        "high": _number(row.get("high")),
        "low": _number(row.get("low")),
        "previous_close": _number(row.get("previous_close")),
        "pct_change": _number(row.get("pct_chg")),
        "volume": _number(row.get("volume")),
        "amount_yuan": _number(row.get("amount_yuan")),
        "turnover_rate_pct": _number(row.get("turnover_rate")),
        "volume_ratio": _number(row.get("volume_ratio")),
        "pe_dynamic": _number(row.get("pe_ttm")),
        "pb": _number(row.get("pb")),
        "circulating_market_value_yuan": _number(row.get("circulating_market_value_yuan")),
    }


def huoqu_zuixin_hengjiemian(
    trade_date: pd.Timestamp,
    *,
    realtime_loader: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取一个完整交易日的全市场行情、估值和股票资料横截面。"""
    warnings: list[str] = []
    date_text = pd.Timestamp(trade_date).strftime("%Y%m%d")
    try:
        pro = _tushare_pro()
        actual_date, daily = _latest_tushare_daily(pro, date_text)
        daily = daily.copy()
        daily["ts_code"] = daily["ts_code"].map(_normalize_code)
        daily["latest_price"] = pd.to_numeric(daily.get("close"), errors="coerce")
        daily["open"] = pd.to_numeric(daily.get("open"), errors="coerce")
        daily["high"] = pd.to_numeric(daily.get("high"), errors="coerce")
        daily["low"] = pd.to_numeric(daily.get("low"), errors="coerce")
        daily["previous_close"] = pd.to_numeric(daily.get("pre_close"), errors="coerce")
        daily["pct_chg"] = pd.to_numeric(daily.get("pct_chg"), errors="coerce")
        daily["volume"] = pd.to_numeric(daily.get("vol"), errors="coerce") * 100.0
        daily["amount_yuan"] = pd.to_numeric(daily.get("amount"), errors="coerce") * 1000.0
        basic_quality: dict[str, Any] = {}
        basic = huoqu_gupiao_jichu_ziliao(pro, basic_quality)
        warnings.extend(str(value) for value in basic_quality.get("warnings", []))
        basic = basic.copy()
        basic["ts_code"] = basic["ts_code"].map(_normalize_code)
        try:
            valuation = pro.daily_basic(
                trade_date=actual_date,
                fields="ts_code,trade_date,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv",
            )
            if valuation is not None and not valuation.empty:
                valuation = valuation.copy()
                valuation["ts_code"] = valuation["ts_code"].map(_normalize_code)
                for column in ("turnover_rate", "volume_ratio", "pe_ttm", "pb", "total_mv", "circ_mv"):
                    valuation[column] = pd.to_numeric(valuation.get(column), errors="coerce")
                valuation["total_market_value_yuan"] = valuation["total_mv"] * 10_000.0
                valuation["circulating_market_value_yuan"] = valuation["circ_mv"] * 10_000.0
                keep = [
                    "ts_code",
                    "turnover_rate",
                    "volume_ratio",
                    "pe_ttm",
                    "pb",
                    "total_market_value_yuan",
                    "circulating_market_value_yuan",
                ]
                daily = daily.merge(valuation[keep], on="ts_code", how="left")
        except Exception as exc:
            warnings.append(f"完整交易日估值横截面不可用：{exc}")
        keep_daily = [
            column
            for column in (
                "ts_code",
                "latest_price",
                "open",
                "high",
                "low",
                "previous_close",
                "pct_chg",
                "volume",
                "amount_yuan",
                "turnover_rate",
                "volume_ratio",
                "pe_ttm",
                "pb",
                "total_market_value_yuan",
                "circulating_market_value_yuan",
            )
            if column in daily.columns
        ]
        data = basic.merge(daily[keep_daily].drop_duplicates("ts_code"), on="ts_code", how="inner")
        return data.reset_index(drop=True), {
            "status": "ok",
            "source": "tushare_daily_cross_section",
            "as_of": pd.Timestamp(actual_date).strftime("%Y-%m-%d"),
            "rows": int(len(data)),
            "warnings": warnings,
        }
    except Exception as exc:
        warnings.append(f"Tushare 完整交易日横截面不可用：{exc}")
        if realtime_loader is None:
            return pd.DataFrame(), {"status": "unavailable", "as_of": trade_date.strftime("%Y-%m-%d"), "warnings": warnings}
        realtime, meta = realtime_loader()
        if realtime.empty:
            warnings.append(str(meta.get("error") or "实时快照也不可用"))
            return pd.DataFrame(), {"status": "unavailable", "as_of": trade_date.strftime("%Y-%m-%d"), "warnings": warnings}
        return realtime, {
            "status": "degraded",
            "source": meta.get("source"),
            "as_of": trade_date.strftime("%Y-%m-%d"),
            "captured_at": meta.get("captured_at"),
            "rows": int(len(realtime)),
            "warnings": warnings + ["候选横截面已降级为实时快照；完整日线时点仍单独校验"],
        }


def _normalise_universe_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """同行和统一选股共用的横截面代码归一化。"""
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data = data[data["ts_code"].map(_shi_a_gu)].copy()
    data["ts_code"] = data["ts_code"].map(_normalize_code)
    return data.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def huoqu_tushare_hengjiemian(
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    """单股同行选择使用的 Tushare 时点横截面。"""
    pro = _tushare_pro()
    warnings: list[str] = []
    quality: dict[str, Any] = {}
    basic = huoqu_gupiao_jichu_ziliao(pro, quality)
    warnings.extend(str(value) for value in quality.get("warnings", []))
    stock_master_meta: dict[str, Any] = {
        "status": "live_current_snapshot",
        "source": str((quality.get("stock_basic") or {}).get("source") or "tushare_live"),
        "persistence": "none",
        "known_bias": "股票资料和行业标签来自当前接口；分析历史日期时可能存在行业分类时点偏差",
    }
    daily = pd.DataFrame()
    daily_date = ""
    for offset in range(12):
        candidate = signal_date - timedelta(days=offset)
        candidate_text = candidate.strftime("%Y%m%d")
        raw = pro.daily(trade_date=candidate_text)
        if raw is not None and not raw.empty:
            daily = raw.copy()
            daily_date = candidate_text
            break
    if daily.empty:
        raise RuntimeError("Tushare 未返回信号日前的全市场日行情")
    daily["amount_yuan"] = pd.to_numeric(daily.get("amount"), errors="coerce") * 1000.0
    daily["latest_price"] = pd.to_numeric(daily.get("close"), errors="coerce")
    daily["pct_chg"] = pd.to_numeric(daily.get("pct_chg"), errors="coerce")
    daily = daily[[column for column in ("ts_code", "latest_price", "pct_chg", "amount_yuan") if column in daily.columns]]
    try:
        daily_basic = pro.daily_basic(
            trade_date=daily_date,
            fields="ts_code,trade_date,turnover_rate,pe_ttm,pb,total_mv,circ_mv",
        )
        if daily_basic is not None and not daily_basic.empty:
            for column in ("turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv"):
                daily_basic[column] = pd.to_numeric(daily_basic.get(column), errors="coerce")
            daily_basic["total_market_value_yuan"] = daily_basic["total_mv"] * 10_000.0
            daily_basic["circulating_market_value_yuan"] = daily_basic["circ_mv"] * 10_000.0
            keep = [
                "ts_code",
                "turnover_rate",
                "pe_ttm",
                "pb",
                "total_market_value_yuan",
                "circulating_market_value_yuan",
            ]
            daily = daily.merge(daily_basic[keep], on="ts_code", how="left")
    except Exception as exc:
        warnings.append(f"同行估值横截面不可用：{exc}")
    basic = _normalise_universe_codes(basic)
    daily = _normalise_universe_codes(daily)
    data = basic.merge(daily, on="ts_code", how="left")
    return data, pd.Timestamp(daily_date).strftime("%Y-%m-%d"), warnings, stock_master_meta


def huoqu_akshare_hengjiemian(
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    """单股同行选择使用的 AKShare 降级横截面。"""
    data, snapshot_meta = huoqu_shishi_kuaizhao()
    if data.empty:
        raise RuntimeError(str(snapshot_meta.get("error") or "AKShare 全市场快照为空"))
    warnings = ["同行池降级为 AKShare 当前快照，横截面日期由分析信号日近似"]
    try:
        quality: dict[str, Any] = {}
        basic = huoqu_gupiao_jichu_ziliao(_tushare_pro(), quality)
        if not basic.empty:
            basic = _normalise_universe_codes(basic)
            supplement = [column for column in ("ts_code", "name", "industry", "market", "list_date") if column in basic.columns]
            data = data.merge(basic[supplement], on="ts_code", how="left", suffixes=("", "_basic"))
            if "name_basic" in data.columns:
                data["name"] = data["name"].fillna(data["name_basic"])
                data = data.drop(columns=["name_basic"])
    except Exception as exc:
        warnings.append(f"Tushare 实时股票资料补充失败：{exc}")
    return (
        data,
        signal_date.strftime("%Y-%m-%d"),
        warnings,
        {
            "status": "live_current_snapshot",
            "source": "akshare_current_snapshot",
            "persistence": "none",
            "known_bias": "AKShare 快照没有历史行业成员时点，不能用于回填过去行业标签",
        },
    )


def _normalize_constituents(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "代码": "ts_code",
        "名称": "name",
        "最新价": "latest_price",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "昨收": "previous_close",
        "涨跌幅": "pct_chg",
        "成交量": "volume",
        "成交额": "amount_yuan",
        "换手率": "turnover_rate",
        "量比": "volume_ratio",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "total_market_value_yuan",
        "流通市值": "circulating_market_value_yuan",
        "code": "ts_code",
        "name": "name",
        "trade": "latest_price",
        "changepercent": "pct_chg",
        "volume": "volume",
        "amount": "amount_yuan",
        "turnoverratio": "turnover_rate",
        "per": "pe_ttm",
    }
    data = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    if not {"ts_code", "name"}.issubset(data.columns):
        raise ValueError("板块成分接口缺少代码或名称列")
    data["ts_code"] = data["ts_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    data = data[data["ts_code"].map(_shi_a_gu)].copy()
    data["ts_code"] = data["ts_code"].map(_normalize_code)
    data["name"] = data["name"].fillna("").astype(str)
    for column in (
        "latest_price",
        "open",
        "high",
        "low",
        "previous_close",
        "pct_chg",
        "volume",
        "amount_yuan",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "total_market_value_yuan",
        "circulating_market_value_yuan",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "volume" in data.columns:
        data["volume"] = data["volume"] * 100.0
    return data.drop_duplicates("ts_code", keep="first").reset_index(drop=True)


def huoqu_bankuai_chengfen(
    bankuai: ShichangFanwei | str,
    *,
    bankuai_leixing: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """返回已解析范围的实时成分；字符串入口仅保留给内部迁移调用。"""
    if isinstance(bankuai, ShichangFanwei):
        resolved_scope = bankuai
    else:
        query = " ".join(str(bankuai or "").split())
        if not query:
            raise ValueError("板块名称不能为空")
        aliases = {
            "industry": BankuaiLeixing.HANGYE,
            "hangye": BankuaiLeixing.HANGYE,
            "行业": BankuaiLeixing.HANGYE,
            "concept": BankuaiLeixing.GAINIAN,
            "gainian": BankuaiLeixing.GAINIAN,
            "概念": BankuaiLeixing.GAINIAN,
        }
        hinted = aliases.get(str(bankuai_leixing).strip().lower())
        discovery_name = (
            f"{query}{'行业' if hinted is BankuaiLeixing.HANGYE else '概念'}"
            if hinted is not None
            else query
        )
        discovery = faxian_fenxi_fanwei(discovery_name)
        if discovery.status != "resolved" or discovery.scope is None:
            result = discovery.to_result()
            if discovery.status == "clarification_required":
                labels = [candidate.user_label for candidate in discovery.candidates]
                detail = f"，候选：{'、'.join(labels)}" if labels else ""
                raise ValueError(f"范围需要用户确认{detail}")
            raise RuntimeError(str(result.get("error") or "实时板块目录不可用"))
        resolved_scope = discovery.scope
    raw, metadata = huoqu_dongcai_chengfen(resolved_scope)
    data = _normalize_constituents(raw)
    if data.empty:
        raise RuntimeError("实时板块成分中没有合法 A 股")
    return data, {
        **metadata,
        "resolved_name": resolved_scope.canonical_name,
        "board_type": resolved_scope.kind.value,
        "name_similarity": round(resolved_scope.match_score, 4),
        "warnings": [],
    }


_DONGCAI_KLINE_COLUMNS = (
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount_yuan",
    "amplitude_pct",
    "pct_chg",
    "price_change",
    "turnover_rate",
)


def _jiexi_dongcai_rili(rows: list[str]) -> pd.DataFrame:
    """把东方财富日 K 文本转换为统一的前复权日线口径。"""
    records: list[list[str]] = []
    for raw in rows:
        fields = raw.split(",")
        if len(fields) < len(_DONGCAI_KLINE_COLUMNS):
            raise ValueError("东方财富日 K 行字段不完整")
        records.append(fields[: len(_DONGCAI_KLINE_COLUMNS)])
    data = pd.DataFrame(records, columns=_DONGCAI_KLINE_COLUMNS)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for column in _DONGCAI_KLINE_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # 东方财富日 K 成交量以手为单位；统一为股。成交额本身即为元。
    data["volume"] = data["volume"] * 100.0
    data = (
        data.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    data["pre_close"] = data["close"].shift(1)
    return data.replace([np.inf, -np.inf], np.nan)


def _jiexi_tengxun_rili(
    rows: list[list[Any]],
    identity: dict[str, Any],
) -> pd.DataFrame:
    """转换腾讯前复权日线，并用同响应证券快照核对最新一行。"""
    data = pd.DataFrame(
        [value[:6] + [value[8]] for value in rows],
        columns=[
            "trade_date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount_10k_yuan",
        ],
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for column in ("open", "close", "high", "low", "volume", "amount_10k_yuan"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    volume_unit = str(identity.get("volume_unit") or "hands")
    if volume_unit not in {"hands", "shares"}:
        raise ValueError(f"腾讯日线成交量单位无法识别：{volume_unit}")
    volume_multiplier = 100.0 if volume_unit == "hands" else 1.0
    data["volume"] = data["volume"] * volume_multiplier
    # 新版公开接口的第 9 个字段为成交额（万元）。成交额必须使用未复权的真实
    # 成交口径，不能再用前复权价格乘成交量估算。
    data["amount_yuan"] = data["amount_10k_yuan"] * 10_000.0
    data = data.drop(columns=["amount_10k_yuan"])
    data = (
        data.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    if data.empty:
        return data
    data["pre_close"] = data["close"].shift(1)
    data["pct_chg"] = data["close"].pct_change(fill_method=None) * 100.0

    quote_timestamp = str(identity.get("quote_timestamp") or "")
    quote_date = pd.to_datetime(quote_timestamp[:8], format="%Y%m%d", errors="coerce")
    latest_index = data.index[-1]
    latest_date = pd.Timestamp(data.loc[latest_index, "trade_date"])
    if not pd.isna(quote_date) and pd.Timestamp(quote_date).normalize() == latest_date:
        latest_close = pd.to_numeric(identity.get("latest_close"), errors="coerce")
        if pd.isna(latest_close) or abs(float(latest_close) - float(data.loc[latest_index, "close"])) > 0.011:
            raise ValueError("腾讯前复权日线与同响应证券快照的最新收盘价冲突")
        latest_volume = pd.to_numeric(
            identity.get("latest_volume_source_units"),
            errors="coerce",
        )
        if not pd.isna(latest_volume):
            observed_source_volume = (
                float(data.loc[latest_index, "volume"]) / volume_multiplier
            )
            if abs(float(latest_volume) - observed_source_volume) > 0.5:
                raise ValueError("腾讯前复权日线与同响应证券快照的最新成交量冲突")
        latest_amount = pd.to_numeric(identity.get("latest_amount_yuan"), errors="coerce")
        if not pd.isna(latest_amount) and float(latest_amount) > 0:
            observed_amount = pd.to_numeric(data.loc[latest_index, "amount_yuan"], errors="coerce")
            if (
                not pd.isna(observed_amount)
                and abs(float(observed_amount) - float(latest_amount))
                / max(float(latest_amount), 1.0)
                > 0.001
            ):
                raise ValueError("腾讯前复权日线与同响应证券快照的最新成交额冲突")
            data.loc[latest_index, "amount_yuan"] = float(latest_amount)
    return data.replace([np.inf, -np.inf], np.nan)


def _bingfa_qfq_history(
    codes: tuple[str, ...],
    *,
    provider: str,
    settings: dict[str, Any],
    loader: Callable[[GongkaiShujuHTTPKehu, str], tuple[pd.DataFrame, str]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """数据源无关的并发模板；业务适配器只负责一只证券的读取和核验。"""
    histories: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, Any]] = []
    endpoint_counts: Counter[str] = Counter()
    failed_attempts = 0
    maximum_workers = min(int(settings.get("history_max_workers", 8)), len(codes))

    def fetch_one(code: str) -> tuple[str, pd.DataFrame, str, int]:
        with GongkaiShujuHTTPKehu(
            provider,
            settings=settings,
        ) as client:
            data, endpoint = loader(client, code)
            if data.empty:
                raise ValueError(f"{provider} 没有返回 {code} 的有效前复权日线")
            return (
                code,
                data,
                urlsplit(endpoint).hostname or "unknown",
                sum(item.outcome == "failed" for item in client.attempts),
            )

    with ThreadPoolExecutor(max_workers=max(1, maximum_workers), thread_name_prefix="qfq") as executor:
        future_by_code = {executor.submit(fetch_one, code): code for code in codes}
        for future in as_completed(future_by_code):
            code = future_by_code[future]
            try:
                loaded_code, data, endpoint_host, failed_count = future.result()
                histories[loaded_code] = data
                endpoint_counts[endpoint_host] += 1
                failed_attempts += failed_count
            except WangluoQingqiuYichang as exc:
                failures.append(
                    {
                        "ts_code": code,
                        "error_code": exc.error_code,
                        "retryable": exc.retryable,
                        "error": "远端连接失败",
                    }
                )
                failed_attempts += sum(item.outcome == "failed" for item in exc.attempts)
            except Exception as exc:
                failures.append(
                    {
                        "ts_code": code,
                        "error_code": "source_payload_invalid",
                        "retryable": True,
                        "error": " ".join(str(exc).split())[:160],
                    }
                )
    return histories, {
        "source": provider,
        "requested_stocks": len(codes),
        "loaded_stocks": len(histories),
        "endpoint_success_counts": dict(endpoint_counts),
        "failed_network_attempts": failed_attempts,
        "failures": failures[:20],
        "failure_count": len(failures),
        "persistence": "none",
    }


def _qfq_history_from_tencent(
    codes: tuple[str, ...],
    *,
    start_date: str,
    end_date: str,
    settings: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """腾讯日线为主源；证券快照与日线在同一响应中交叉核验。"""

    def loader(client: GongkaiShujuHTTPKehu, code: str) -> tuple[pd.DataFrame, str]:
        rows, identity, endpoint = tengxun_qfq_rili_duqu(
            client,
            code=code,
            start_date=start_date,
            end_date=end_date,
        )
        if identity.get("adjustment") != "qfq" or identity.get("identity_check") != "passed":
            raise ValueError(f"腾讯证券没有确认 {code} 的身份或前复权口径")
        data = _jiexi_tengxun_rili(rows, identity)
        lower = pd.Timestamp(start_date).normalize()
        upper = pd.Timestamp(end_date).normalize()
        data = data[
            pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize().between(lower, upper)
        ].reset_index(drop=True)
        return data, endpoint

    histories, metadata = _bingfa_qfq_history(
        codes,
        provider="tencent_qfq_daily",
        settings=settings,
        loader=loader,
    )
    metadata["amount_field_policy"] = (
        "历史成交额读取腾讯日线响应中的万元字段并换算为元；"
        "最新日同时与同响应证券快照的精确成交额交叉核验"
    )
    metadata["fact_verification"] = (
        "证券身份及最新同日收盘价、成交量、成交额由同响应快照交叉核验"
    )
    return histories, metadata


def _qfq_history_from_eastmoney(
    codes: tuple[str, ...],
    *,
    start_date: str,
    end_date: str,
    settings: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """东方财富前复权日线备用源；校验响应证券身份和复权标记。"""

    def loader(client: GongkaiShujuHTTPKehu, code: str) -> tuple[pd.DataFrame, str]:
        rows, identity, endpoint = dongcai_rili_kxian_duqu(
            client,
            code=code,
            start_date=start_date,
            end_date=end_date,
            qfq=True,
        )
        if identity.get("adjustment") != "qfq":
            raise ValueError(f"东方财富没有确认 {code} 的前复权口径")
        return _jiexi_dongcai_rili(rows), endpoint

    return _bingfa_qfq_history(
        codes,
        provider="eastmoney_qfq_daily",
        settings=settings,
        loader=loader,
    )


def _qfq_history_from_tushare(
    codes: tuple[str, ...],
    *,
    start_date: str,
    end_date: str,
    pause_seconds: float,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """仅为主源失败的少量股票逐只降级，不再按交易日发起数百次请求。"""
    if not codes:
        return {}, []
    pro = _tushare_pro()
    histories: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for code in codes:
        try:
            daily = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            factors = pro.adj_factor(ts_code=code, start_date=start_date, end_date=end_date)
            if daily is None or daily.empty:
                raise RuntimeError("日线为空")
            if factors is None or factors.empty:
                raise RuntimeError("复权因子为空")
            data = daily.copy()
            data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
            factor_table = factors[["trade_date", "adj_factor"]].copy()
            factor_table["trade_date"] = pd.to_datetime(factor_table["trade_date"], errors="coerce").dt.normalize()
            factor_table["adj_factor"] = pd.to_numeric(factor_table["adj_factor"], errors="coerce")
            data = data.merge(factor_table, on="trade_date", how="left").sort_values("trade_date").reset_index(drop=True)
            data["adj_factor"] = data["adj_factor"].ffill().bfill()
            latest_factor = pd.to_numeric(data["adj_factor"], errors="coerce").dropna()
            if latest_factor.empty or float(latest_factor.iloc[-1]) <= 0:
                raise RuntimeError("复权因子无效")
            multiplier = data["adj_factor"] / float(latest_factor.iloc[-1])
            for column in ("open", "high", "low", "close", "pre_close"):
                data[column] = pd.to_numeric(data.get(column), errors="coerce") * multiplier
            data["volume"] = pd.to_numeric(data.get("vol"), errors="coerce") * 100.0
            data["amount_yuan"] = pd.to_numeric(data.get("amount"), errors="coerce") * 1000.0
            data["pct_chg"] = pd.to_numeric(data.get("pct_chg"), errors="coerce")
            histories[code] = data[
                [
                    column
                    for column in (
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "pct_chg",
                        "volume",
                        "amount_yuan",
                    )
                    if column in data.columns
                ]
            ].replace([np.inf, -np.inf], np.nan)
        except Exception as exc:
            warnings.append(f"{code} Tushare 前复权备用源失败：{' '.join(str(exc).split())[:140]}")
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return histories, warnings


def huoqu_piliang_qfq_lishi(
    codes: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    minimum_rows: int,
    calendar: JiaoyiRili | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """从公开远端接口读取候选前复权日线；请求结束后不持久化。"""
    normalized = tuple(dict.fromkeys(_normalize_code(code) for code in codes))
    if not normalized:
        return {}, {"status": "unavailable", "error": "候选代码为空"}
    warnings: list[str] = []
    resolved_calendar = calendar or huoqu_jiaoyi_rili(pd.Timestamp(end_date).to_pydatetime())
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    config, _ = jiazai_lianghua_peizhi()
    pause = float(config.get("shuju", {}).get("request_pause_seconds", 0.15))
    network_settings = (
        dict(config.get("wangluo", {}))
        if isinstance(config.get("wangluo"), dict)
        else {}
    )
    try:
        fetched_histories, primary_meta = _qfq_history_from_tencent(
            normalized,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            settings=network_settings,
        )
    except Exception as exc:
        return {}, {
            "status": "unavailable",
            "source": "tencent_qfq_daily",
            "requested_stocks": len(normalized),
            "loaded_stocks": 0,
            "warnings": warnings,
            "error": str(exc),
        }
    missing = tuple(code for code in normalized if code not in fetched_histories)
    fallback_limit = int(network_settings.get("history_fallback_max_stocks", 16))
    secondary_codes = missing[:fallback_limit]
    secondary_histories: dict[str, pd.DataFrame] = {}
    secondary_meta: dict[str, Any] = {
        "source": "eastmoney_qfq_daily",
        "requested_stocks": 0,
        "loaded_stocks": 0,
        "persistence": "none",
    }
    if secondary_codes:
        try:
            secondary_histories, secondary_meta = _qfq_history_from_eastmoney(
                secondary_codes,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                settings=network_settings,
            )
            fetched_histories.update(secondary_histories)
        except Exception as exc:
            warnings.append(f"东方财富前复权备用源不可用：{' '.join(str(exc).split())[:180]}")
    remaining = tuple(code for code in secondary_codes if code not in fetched_histories)
    tushare_histories: dict[str, pd.DataFrame] = {}
    if remaining:
        try:
            tushare_histories, fallback_warnings = _qfq_history_from_tushare(
                remaining,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                pause_seconds=pause,
            )
            warnings.extend(fallback_warnings)
            fetched_histories.update(tushare_histories)
        except Exception as exc:
            warnings.append(f"Tushare 前复权备用源不可用：{' '.join(str(exc).split())[:180]}")
    if len(missing) > len(secondary_codes):
        warnings.append(
            f"主源失败股票共 {len(missing)} 只；备用源按上限只尝试 {len(secondary_codes)} 只，避免失控请求"
        )
    expected_dates = sorted(day for day in resolved_calendar.open_dates if start <= day <= end)
    expected_set = set(expected_dates)
    minimum_coverage = float(config.get("fenxi", {}).get("minimum_history_session_coverage", 0.9))
    coverage_by_code: dict[str, float] = {}
    incomplete: list[dict[str, Any]] = []
    ready: dict[str, pd.DataFrame] = {}
    for code, frame in fetched_histories.items():
        dates = pd.to_datetime(frame.get("trade_date"), errors="coerce").dropna().dt.normalize()
        if dates.empty:
            coverage = 0.0
        else:
            first = pd.Timestamp(dates.min())
            relevant = {day for day in expected_set if first <= day <= end}
            actual = set(pd.Timestamp(value) for value in dates)
            coverage = len(actual & relevant) / len(relevant) if relevant else 0.0
        coverage_by_code[code] = round(float(coverage), 4)
        if len(frame) < minimum_rows or coverage < minimum_coverage:
            incomplete.append(
                {
                    "ts_code": code,
                    "rows": int(len(frame)),
                    "session_coverage": round(float(coverage), 4),
                }
            )
            continue
        ready[code] = frame
    actual_min = min(
        (pd.Timestamp(frame["trade_date"].min()) for frame in ready.values()),
        default=None,
    )
    actual_max = max(
        (pd.Timestamp(frame["trade_date"].max()) for frame in ready.values()),
        default=None,
    )
    source_counts = {
        "tencent_qfq_daily": (
            len(ready)
            - sum(code in ready for code in secondary_histories)
            - sum(code in ready for code in tushare_histories)
        ),
        "eastmoney_qfq_fallback": sum(code in ready for code in secondary_histories),
        "tushare_qfq_fallback": sum(code in ready for code in tushare_histories),
    }
    used_sources = [source for source, count in source_counts.items() if count > 0]
    return ready, {
        "status": "ok" if ready else "unavailable",
        "source": "+".join(used_sources) if used_sources else "remote_qfq_daily_unavailable",
        "requested_stocks": len(normalized),
        "loaded_stocks": len(ready),
        "minimum_rows": minimum_rows,
        "requested_range": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        "actual_range": [
            actual_min.strftime("%Y-%m-%d") if actual_min is not None else None,
            actual_max.strftime("%Y-%m-%d") if actual_max is not None else None,
        ],
        "expected_open_sessions": len(expected_dates),
        "minimum_session_coverage": minimum_coverage,
        "session_coverage": {
            "minimum": min(coverage_by_code.values(), default=None),
            "median": (
                round(float(np.median(list(coverage_by_code.values()))), 4)
                if coverage_by_code
                else None
            ),
            "maximum": max(coverage_by_code.values(), default=None),
        },
        "incomplete_examples": incomplete[:20],
        "incomplete_count": len(incomplete),
        "source_counts": source_counts,
        "primary_source": primary_meta,
        "secondary_source": secondary_meta,
        "fallback_attempted_stocks": len(secondary_codes),
        "persistence": "none",
        "warnings": warnings,
    }


def huoqu_fenzhong_xingqing(
    code: str,
    *,
    reference: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """仅为少量候选读取当日 5 分钟数据。"""
    captured = reference or _beijing_now()
    normalized = _normalize_code(code)
    date_text = captured.strftime("%Y-%m-%d")
    try:
        import akshare as ak

        capability = _jiancha_akshare_hanshu(ak, "stock_zh_a_hist_min_em")
        with akshare_zhilian():
            raw = ak.stock_zh_a_hist_min_em(
                symbol=normalized.split(".")[0],
                start_date=f"{date_text} 09:30:00",
                end_date=f"{date_text} 15:00:00",
                period="5",
                adjust="",
            )
        if raw is None or raw.empty:
            raise RuntimeError("5 分钟行情为空")
        rename = {
            "时间": "trade_time",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount_yuan",
            "均价": "average_price",
        }
        data = raw.rename(columns={key: value for key, value in rename.items() if key in raw.columns}).copy()
        if "trade_time" not in data.columns:
            raise RuntimeError("5 分钟行情缺少时间列")
        data["trade_time"] = pd.to_datetime(data["trade_time"], errors="coerce")
        for column in ("open", "close", "high", "low", "volume", "amount_yuan", "average_price"):
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["trade_time"]).sort_values("trade_time").reset_index(drop=True)
        return data, {
            "status": "ok",
            "source": "akshare_eastmoney_5min",
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(len(data)),
            **capability,
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "unavailable",
            "source": "akshare_eastmoney_5min",
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(exc),
        }


__all__ = [
    "huoqu_akshare_hengjiemian",
    "huoqu_tushare_hengjiemian",
    "FenxiShujuShangxiawen",
    "JiaoyiJieduan",
    "JiaoyiRili",
    "akshare_zhilian",
    "huoqu_bankuai_chengfen",
    "huoqu_dangqian_kuaizhao",
    "huoqu_fenzhong_xingqing",
    "huoqu_jiaoyi_rili",
    "huoqu_piliang_qfq_lishi",
    "huoqu_shishi_kuaizhao",
    "huoqu_zuixin_hengjiemian",
    "shichang_shizhong",
    "zuixin_wanzheng_jiaoyiri",
]
