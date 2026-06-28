from __future__ import annotations

import argparse
import random
import time

import pandas as pd
from tqdm import tqdm

from utils import (
    META_DIR,
    RAW_DIR,
    ensure_project_dirs,
    fetch_sina_kline,
    format_date,
    normalize_kline_df,
    parse_date_arg,
    read_json,
    setup_logger,
    write_failed,
)


def get_safe_datalen() -> int:
    probe = read_json(META_DIR / "sina_interface_probe_result.json", {})
    datalen = probe.get("stock_kline_max", {}).get("best", {}).get("datalen")
    return int(datalen or 1970)


def strict_coverage_ok(start: pd.Timestamp) -> bool:
    probe = read_json(META_DIR / "sina_interface_probe_result.json", {})
    first = probe.get("stock_kline_max", {}).get("best", {}).get("first_date")
    if not first:
        return False
    return pd.Timestamp(first) <= start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20260628")
    parser.add_argument("--mode", choices=["strict-5y", "allow-1023"], default="strict-5y")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    logger = setup_logger("daily_kline", "02_fetch_daily_kline_from_sina.log")
    start = parse_date_arg(args.start)
    end = parse_date_arg(args.end)
    if args.mode == "strict-5y" and not strict_coverage_ok(start):
        raise RuntimeError(
            "Sina K-line probe cannot cover the requested start date in strict-5y mode. "
            "Run with --mode allow-1023 to accept limited history."
        )

    stock_path = META_DIR / "stock_list.csv"
    if not stock_path.exists():
        raise FileNotFoundError("Missing data/meta/stock_list.csv. Run 01_fetch_stock_list_from_sina.py first.")
    stocks = pd.read_csv(stock_path, dtype={"code": str})
    if args.limit:
        stocks = stocks.head(args.limit)

    datalen = get_safe_datalen()
    out_dir = RAW_DIR / "sina_kline_raw"
    failed = []
    for row in tqdm(stocks.itertuples(index=False), total=len(stocks), desc="Fetch daily K"):
        out = out_dir / f"{row.symbol}.csv"
        if out.exists() and not args.force:
            continue
        try:
            rows = fetch_sina_kline(row.sina_symbol, datalen)
            df = normalize_kline_df(rows, row.symbol)
            df = df[(pd.to_datetime(df["date"]) >= start) & (pd.to_datetime(df["date"]) <= end)]
            if df.empty:
                raise RuntimeError("No rows after date filtering")
            df.to_csv(out, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch %s", row.symbol)
            failed.append({"symbol": row.symbol, "sina_symbol": row.sina_symbol, "step": "kline", "error": str(exc)})
        time.sleep(random.uniform(0.5, 2.0))

    write_failed(META_DIR / "failed_symbols.csv", failed, step="kline")
    logger.info("Daily K fetch done. start=%s end=%s mode=%s datalen=%s failed=%s", format_date(start), format_date(end), args.mode, datalen, len(failed))


if __name__ == "__main__":
    main()
