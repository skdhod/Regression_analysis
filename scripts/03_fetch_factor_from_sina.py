from __future__ import annotations

import argparse
import random
import time

import pandas as pd
from tqdm import tqdm

from utils import META_DIR, RAW_DIR, ensure_project_dirs, http_get_text, parse_sina_factor_js, setup_logger, write_failed, write_json


def fetch_factor(sina_symbol: str, kind: str) -> dict:
    url = f"https://finance.sina.com.cn/realstock/company/{sina_symbol}/{kind}.js"
    text = http_get_text(url)
    parsed = parse_sina_factor_js(text)
    parsed["_source_url"] = url
    parsed["_kind"] = kind
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    logger = setup_logger("factor", "03_fetch_factor_from_sina.log")
    stocks = pd.read_csv(META_DIR / "stock_list.csv", dtype={"code": str})
    if args.limit:
        stocks = stocks.head(args.limit)
    failed = []
    for row in tqdm(stocks.itertuples(index=False), total=len(stocks), desc="Fetch factors"):
        got_any = False
        for kind, subdir in [("hfq", "sina_factor_hfq"), ("qfq", "sina_factor_qfq")]:
            out = RAW_DIR / subdir / f"{row.symbol}.json"
            if out.exists() and not args.force:
                got_any = True
                continue
            try:
                data = fetch_factor(row.sina_symbol, kind)
                write_json(out, data)
                got_any = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Factor %s failed for %s: %s", kind, row.symbol, exc)
        if not got_any:
            failed.append({"symbol": row.symbol, "sina_symbol": row.sina_symbol, "step": "factor", "error": "hfq and qfq failed"})
        time.sleep(random.uniform(0.3, 1.2))
    write_failed(META_DIR / "failed_symbols.csv", failed, step="factor")
    logger.info("Factor fetch done. failed=%s", len(failed))


if __name__ == "__main__":
    main()
