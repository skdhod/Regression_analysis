from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
META_DIR = DATA_DIR / "meta"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QLIB_DIR = DATA_DIR / "qlib" / "cn_data"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SINA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}

BENCHMARKS = {
    "上证指数": ["sh000001"],
    "深证成指": ["sz399001"],
    "上证100": ["sh000132"],
    "深证100": ["sz399330"],
    "中证500": ["sh000905", "sz399905"],
}


def ensure_project_dirs() -> None:
    dirs = [
        META_DIR,
        RAW_DIR / "sina_kline_raw",
        RAW_DIR / "sina_factor_hfq",
        RAW_DIR / "sina_factor_qfq",
        RAW_DIR / "sina_index_kline",
        PROCESSED_DIR / "daily_standard",
        PROCESSED_DIR / "qlib_csv",
        PROCESSED_DIR / "benchmark_indices",
        QLIB_DIR,
        LOG_DIR,
        OUTPUT_DIR / "reports",
        OUTPUT_DIR / "predictions",
        OUTPUT_DIR / "backtest",
        OUTPUT_DIR / "benchmark",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    ensure_project_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def backup_if_exists(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.rename(path.with_name(f"{path.name}.{stamp}.bak"))


def decode_response(resp: requests.Response) -> str:
    if resp.encoding:
        try:
            return resp.text
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return resp.content.decode(enc)
        except UnicodeDecodeError:
            continue
    return resp.content.decode("utf-8", errors="replace")


def http_get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    retries: int = 3,
    sleep_range: tuple[float, float] = (0.5, 2.0),
    headers: dict[str, str] | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers or SINA_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return decode_response(resp)
        except Exception as exc:  # noqa: BLE001 - preserve network failure detail in logs.
            last_error = exc
            if attempt < retries:
                time.sleep(random.uniform(*sleep_range))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} {params or ''}: {last_error}")


def sina_kline_url(symbol: str, datalen: int, scale: int = 240) -> str:
    params = urlencode({"symbol": symbol, "scale": scale, "ma": "no", "datalen": datalen})
    return f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?{params}"


def fetch_sina_kline(symbol: str, datalen: int, timeout: int = 15, retries: int = 3) -> list[dict[str, Any]]:
    text = http_get_text(sina_kline_url(symbol, datalen), timeout=timeout, retries=retries)
    data = json.loads(text)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Unexpected K-line payload for {symbol}: {text[:120]}")
    return data


def parse_sina_factor_js(text: str) -> dict[str, Any]:
    match = re.search(r"var\s+\w+\s*=\s*", text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot parse Sina factor JS: {text[:120]}")
    start = text.find("{", match.end())
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Cannot locate JSON object in Sina factor JS: {text[:120]}")
    return json.loads(text[start : end + 1])


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_date_arg(value: str) -> pd.Timestamp:
    value = str(value).strip()
    fmt = "%Y%m%d" if re.fullmatch(r"\d{8}", value) else "%Y-%m-%d"
    return pd.Timestamp(datetime.strptime(value, fmt).date())


def format_date(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def sina_to_standard_symbol(sina_symbol: str) -> str:
    code = sina_symbol[-6:]
    if sina_symbol.startswith("sh"):
        return f"{code}.SH"
    if sina_symbol.startswith("sz"):
        return f"{code}.SZ"
    if sina_symbol.startswith("bj"):
        return f"{code}.BJ"
    raise ValueError(f"Unknown Sina symbol: {sina_symbol}")


def standard_to_sina_symbol(symbol: str) -> str:
    code, exchange = symbol.split(".")
    return ("sh" if exchange.upper() == "SH" else "sz") + code


def standard_to_qlib_symbol(symbol: str) -> str:
    code, exchange = symbol.split(".")
    return f"{exchange.upper()}{code}"


def sina_to_qlib_symbol(sina_symbol: str) -> str:
    return standard_to_qlib_symbol(sina_to_standard_symbol(sina_symbol))


def classify_board(code: str) -> tuple[str | None, str | None]:
    if code.startswith(("600", "601", "603", "605")):
        return "SH", "沪市主板"
    if code.startswith(("688", "689")):
        return "SH", "科创板"
    if code.startswith(("000", "001", "002", "003")):
        return "SZ", "深市主板"
    if code.startswith(("300", "301")):
        return "SZ", "创业板"
    return None, None


def normalize_kline_df(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
    rename = {"day": "date"}
    df = df.rename(columns=rename)
    df["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]


def safe_ratio(num: float, den: float) -> float:
    if den is None or pd.isna(den) or den == 0:
        return math.nan
    return num / den


def max_drawdown(nav: pd.Series) -> float:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if nav.empty:
        return math.nan
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def annual_return(nav: pd.Series, ann_scaler: int = 252) -> float:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return math.nan
    return float((nav.iloc[-1] / nav.iloc[0]) ** (ann_scaler / (len(nav) - 1)) - 1)


def sharpe(returns: pd.Series, ann_scaler: int = 252) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return math.nan
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(ann_scaler))


def write_failed(path: Path, rows: Iterable[dict[str, Any]], step: str | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.DataFrame()
    if path.exists():
        old = pd.read_csv(path)
        if step and "step" in old.columns:
            old = old[old["step"] != step]
    df = pd.DataFrame(rows)
    if not old.empty or not df.empty:
        df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif path.exists() and step:
        path.unlink()
