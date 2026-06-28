from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    BENCHMARKS,
    META_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_project_dirs,
    fetch_sina_kline,
    format_date,
    normalize_kline_df,
    parse_date_arg,
    read_json,
    setup_logger,
)


def selected_benchmarks() -> dict[str, str]:
    probe = read_json(META_DIR / "sina_interface_probe_result.json", {})
    selected = {}
    for name, candidates in BENCHMARKS.items():
        value = probe.get("benchmark_selection", {}).get(name, {}).get("selected")
        if value:
            selected[name] = value
        else:
            selected[name] = candidates[0]
    return selected


def safe_datalen() -> int:
    probe = read_json(META_DIR / "sina_interface_probe_result.json", {})
    return int(probe.get("stock_kline_max", {}).get("best", {}).get("datalen") or 1970)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20260628")
    args = parser.parse_args()

    ensure_project_dirs()
    logger = setup_logger("benchmarks", "04_fetch_benchmark_indices_from_sina.log")
    start = parse_date_arg(args.start)
    end = parse_date_arg(args.end)
    datalen = safe_datalen()

    raw_dir = RAW_DIR / "sina_index_kline"
    out_dir = PROCESSED_DIR / "benchmark_indices"
    meta_rows = []
    for name, sina_symbol in tqdm(selected_benchmarks().items(), desc="Fetch benchmark indices"):
        rows = fetch_sina_kline(sina_symbol, datalen)
        std_symbol = sina_symbol.upper()
        df = normalize_kline_df(rows, std_symbol)
        df = df[(pd.to_datetime(df["date"]) >= start) & (pd.to_datetime(df["date"]) <= end)].copy()
        if df.empty:
            logger.warning("No benchmark rows for %s %s", name, sina_symbol)
            meta_rows.append({"index_name": name, "index_symbol": sina_symbol, "status": "empty"})
            continue
        df.to_csv(raw_dir / f"{sina_symbol}.csv", index=False, encoding="utf-8-sig")
        df = df.rename(columns={"symbol": "index_symbol"})
        df["index_symbol"] = sina_symbol
        df["index_name"] = name
        df["amount"] = np.nan
        df["return"] = df["close"].pct_change()
        df["nav"] = df["close"] / df["close"].iloc[0]
        df = df[["date", "index_symbol", "index_name", "open", "high", "low", "close", "volume", "amount", "return", "nav"]]
        df.to_csv(out_dir / f"{sina_symbol}.csv", index=False, encoding="utf-8-sig")
        meta_rows.append({"index_name": name, "index_symbol": sina_symbol, "status": "ok", "start_date": df["date"].min(), "end_date": df["date"].max(), "rows": len(df)})

    pd.DataFrame(meta_rows).to_csv(META_DIR / "benchmark_index_status.csv", index=False, encoding="utf-8-sig")
    logger.info("Benchmark fetch done. start=%s end=%s", format_date(start), format_date(end))


if __name__ == "__main__":
    main()
