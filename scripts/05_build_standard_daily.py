from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import META_DIR, PROCESSED_DIR, RAW_DIR, ensure_project_dirs, read_json, setup_logger


def load_factor(symbol: str) -> tuple[pd.DataFrame | None, str]:
    for kind, subdir in [("hfq", "sina_factor_hfq"), ("qfq", "sina_factor_qfq")]:
        path = RAW_DIR / subdir / f"{symbol}.json"
        data = read_json(path)
        if not data or not data.get("data"):
            continue
        df = pd.DataFrame(data["data"]).rename(columns={"d": "date", "f": "factor"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["factor"] = pd.to_numeric(df["factor"], errors="coerce")
        df = df.dropna(subset=["date", "factor"]).sort_values("date")
        if not df.empty:
            return df, kind
    return None, "fallback_1.0"


def attach_factor(df: pd.DataFrame, factor_df: pd.DataFrame | None) -> pd.Series:
    dates = pd.DataFrame({"date_dt": pd.to_datetime(df["date"])})
    if factor_df is None or factor_df.empty:
        return pd.Series(1.0, index=df.index)
    right = factor_df.rename(columns={"date": "date_dt"}).sort_values("date_dt")
    merged = pd.merge_asof(dates.sort_values("date_dt"), right, on="date_dt", direction="backward")
    merged["factor"] = merged["factor"].ffill().fillna(1.0)
    merged = merged.set_index(dates.sort_values("date_dt").index).sort_index()
    return merged["factor"].reindex(df.index).fillna(1.0)


def calc_vwap(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "amount" not in df.columns or df["amount"].isna().all() or df["volume"].isna().all():
        return pd.Series(np.nan, index=df.index), "missing_amount_or_volume"
    vwap = df["amount"] / df["volume"]
    bad = (vwap <= 0) | (vwap > df[["open", "high", "low", "close"]].max(axis=1) * 20)
    alt = df["amount"] / (df["volume"] * 100)
    vwap = vwap.mask(bad, alt)
    return vwap, "amount_div_volume"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("build_standard", "05_build_standard_daily.log")

    stock_path = META_DIR / "stock_list.csv"
    stocks = pd.read_csv(stock_path, dtype={"code": str})
    if args.limit:
        stocks = stocks.head(args.limit)

    out_dir = PROCESSED_DIR / "daily_standard"
    report_rows = []
    for row in tqdm(stocks.itertuples(index=False), total=len(stocks), desc="Build standard daily"):
        try:
            src = RAW_DIR / "sina_kline_raw" / f"{row.symbol}.csv"
            if not src.exists():
                raise FileNotFoundError(str(src))
            df = pd.read_csv(src)
            df["symbol"] = row.symbol
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")
            before = len(df)
            df = df.dropna(subset=["date", "symbol", "close"])
            df = df[df["close"] > 0]
            df = df[(df["volume"].isna()) | (df["volume"] >= 0)]
            df = df[(df["amount"].isna()) | (df["amount"] >= 0)]
            df = df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
            factor_df, factor_source = load_factor(row.symbol)
            df["factor"] = attach_factor(df, factor_df)
            df["vwap"], vwap_source = calc_vwap(df)
            has_amount = bool(df["amount"].notna().any())
            has_vwap = bool(df["vwap"].notna().any())
            has_factor = factor_source != "fallback_1.0"
            df = df[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "vwap", "factor"]]
            df.to_csv(out_dir / f"{row.symbol}.csv", index=False, encoding="utf-8-sig")
            requested = max(before, 1)
            report_rows.append(
                {
                    "symbol": row.symbol,
                    "start_date": df["date"].min() if not df.empty else "",
                    "end_date": df["date"].max() if not df.empty else "",
                    "rows": len(df),
                    "coverage_status": "ok" if not df.empty else "empty",
                    "coverage_ratio": len(df) / requested,
                    "amount_source": "sina_history_kline" if has_amount else "missing_from_sina_history_kline",
                    "vwap_source": vwap_source if has_vwap else "missing_amount_or_volume",
                    "factor_source": factor_source,
                    "has_amount": has_amount,
                    "has_vwap": has_vwap,
                    "has_factor": has_factor,
                    "error_message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to build standard data for %s", row.symbol)
            report_rows.append(
                {
                    "symbol": row.symbol,
                    "start_date": "",
                    "end_date": "",
                    "rows": 0,
                    "coverage_status": "error",
                    "coverage_ratio": math.nan,
                    "amount_source": "unknown",
                    "vwap_source": "unknown",
                    "factor_source": "unknown",
                    "has_amount": False,
                    "has_vwap": False,
                    "has_factor": False,
                    "error_message": str(exc),
                }
            )
    report = pd.DataFrame(report_rows)
    report.to_csv(META_DIR / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    ok_dates = []
    for path in out_dir.glob("*.csv"):
        part = pd.read_csv(path, usecols=["date"])
        ok_dates.extend(part["date"].dropna().tolist())
    if ok_dates:
        pd.DataFrame({"date": sorted(set(ok_dates))}).to_csv(META_DIR / "calendar.csv", index=False, encoding="utf-8-sig")
    logger.info("Standard daily build done. rows=%s", len(report))


if __name__ == "__main__":
    main()
