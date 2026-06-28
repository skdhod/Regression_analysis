from __future__ import annotations

import argparse
import json
from urllib.parse import urlencode

import pandas as pd
from tqdm import tqdm

from utils import META_DIR, classify_board, ensure_project_dirs, http_get_text, setup_logger


NODES = ["sh_a", "sz_a", "cyb", "kcb"]


def fetch_node(node: str, page_size: int = 80) -> list[dict]:
    rows: list[dict] = []
    page = 1
    while True:
        params = urlencode(
            {"page": page, "num": page_size, "sort": "symbol", "asc": 1, "node": node, "symbol": "", "_s_r_a": "page"}
        )
        url = f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?{params}"
        data = json.loads(http_get_text(url))
        if not data:
            break
        rows.extend(data)
        if len(data) < page_size:
            break
        page += 1
    return rows


def build_stock_list(raw_rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in raw_rows:
        code = str(row.get("code", "")).zfill(6)
        exchange, board = classify_board(code)
        if exchange is None:
            continue
        sina_symbol = ("sh" if exchange == "SH" else "sz") + code
        records.append(
            {
                "symbol": f"{code}.{exchange}",
                "sina_symbol": sina_symbol,
                "code": code,
                "exchange": exchange,
                "name": row.get("name", ""),
                "board": board,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "sina_symbol", "code", "exchange", "name", "board"])
    return df.drop_duplicates("symbol").sort_values(["exchange", "code"])[
        ["symbol", "sina_symbol", "code", "exchange", "name", "board"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=80)
    args = parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("stock_list", "01_fetch_stock_list_from_sina.log")

    manual = META_DIR / "stock_list_manual.csv"
    output = META_DIR / "stock_list.csv"
    if manual.exists():
        df = pd.read_csv(manual, dtype={"code": str})
        df.to_csv(output, index=False, encoding="utf-8-sig")
        logger.info("Loaded manual stock list: %s rows", len(df))
        return

    raw_rows = []
    for node in tqdm(NODES, desc="Sina stock list nodes"):
        node_rows = fetch_node(node, args.page_size)
        logger.info("Fetched %s rows from node %s", len(node_rows), node)
        raw_rows.extend(node_rows)
    df = build_stock_list(raw_rows)
    if df.empty:
        raise RuntimeError("No A-share stocks were fetched from Sina.")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    logger.info("Saved %s stocks to %s", len(df), output)


if __name__ == "__main__":
    main()
