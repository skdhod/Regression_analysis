from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import META_DIR, PROCESSED_DIR, ensure_project_dirs, setup_logger, standard_to_qlib_symbol


FIELDS = ["date", "symbol", "open", "high", "low", "close", "volume", "factor", "amount", "vwap"]


def to_qlib_prices(df: pd.DataFrame, qlib_symbol: str) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["factor"] = pd.to_numeric(df.get("factor", 1.0), errors="coerce").ffill().fillna(1.0)
    raw_close = pd.to_numeric(df["close"], errors="coerce")
    adjusted_close = raw_close * df["factor"]
    first_adj_close = adjusted_close.dropna().iloc[0] if adjusted_close.notna().any() else math.nan
    if not first_adj_close or pd.isna(first_adj_close):
        raise RuntimeError(f"Cannot normalize {qlib_symbol}: first adjusted close is missing")
    qlib_factor = df["factor"] / first_adj_close
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * df["factor"] / first_adj_close
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / qlib_factor
    df["factor"] = qlib_factor
    df["symbol"] = qlib_symbol
    for col in ["amount", "vwap"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[FIELDS]


def convert_stocks(limit: int | None = None) -> list[str]:
    src_dir = PROCESSED_DIR / "daily_standard"
    out_dir = PROCESSED_DIR / "qlib_csv"
    files = sorted(src_dir.glob("*.csv"))
    if limit:
        files = files[:limit]
    symbols = []
    for path in tqdm(files, desc="Convert stock CSV to Qlib CSV"):
        std_symbol = path.stem
        qlib_symbol = standard_to_qlib_symbol(std_symbol)
        df = pd.read_csv(path)
        if df.empty:
            continue
        out = to_qlib_prices(df, qlib_symbol)
        out.to_csv(out_dir / f"{qlib_symbol}.csv", index=False, encoding="utf-8-sig")
        symbols.append(qlib_symbol)
    return symbols


def convert_benchmarks() -> list[str]:
    src_dir = PROCESSED_DIR / "benchmark_indices"
    out_dir = PROCESSED_DIR / "qlib_csv"
    symbols = []
    for path in tqdm(sorted(src_dir.glob("*.csv")), desc="Convert benchmark CSV to Qlib CSV"):
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        sina_symbol = str(raw["index_symbol"].iloc[0])
        qlib_symbol = sina_symbol.upper()
        df = raw.rename(columns={"index_symbol": "symbol"})[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].copy()
        df["symbol"] = qlib_symbol
        df["factor"] = 1.0
        df["vwap"] = np.nan
        out = to_qlib_prices(df, qlib_symbol)
        out.to_csv(out_dir / f"{qlib_symbol}.csv", index=False, encoding="utf-8-sig")
        symbols.append(qlib_symbol)
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("convert_qlib_csv", "06_convert_to_qlib_format.log")
    stock_symbols = convert_stocks(args.limit)
    benchmark_symbols = convert_benchmarks()
    pd.DataFrame({"symbol": stock_symbols}).to_csv(META_DIR / "qlib_stock_symbols.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"symbol": benchmark_symbols}).to_csv(META_DIR / "qlib_benchmark_symbols.csv", index=False, encoding="utf-8-sig")
    logger.info("Converted stock=%s benchmark=%s", len(stock_symbols), len(benchmark_symbols))


if __name__ == "__main__":
    main()
