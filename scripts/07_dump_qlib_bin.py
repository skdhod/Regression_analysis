from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from utils import META_DIR, PROCESSED_DIR, PROJECT_ROOT, QLIB_DIR, ensure_project_dirs, setup_logger


def rewrite_stock_only_instruments() -> None:
    stock_symbols_path = META_DIR / "qlib_stock_symbols.csv"
    if not stock_symbols_path.exists():
        raise FileNotFoundError("Missing data/meta/qlib_stock_symbols.csv")
    symbols = pd.read_csv(stock_symbols_path)["symbol"].dropna().astype(str).tolist()
    rows = []
    for sym in symbols:
        csv_path = PROCESSED_DIR / "qlib_csv" / f"{sym}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["date"])
        if df.empty:
            continue
        rows.append(f"{sym}\t{df['date'].min()}\t{df['date'].max()}")
    inst_dir = QLIB_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    (inst_dir / "all.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("dump_qlib_bin", "07_dump_qlib_bin.log")

    if args.force and QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    QLIB_DIR.mkdir(parents=True, exist_ok=True)
    dump_script = PROJECT_ROOT / "qlib" / "scripts" / "dump_bin.py"
    if not dump_script.exists():
        raise FileNotFoundError(f"Cannot find local Qlib dump script: {dump_script}")
    env = os.environ.copy()
    existing_pythonpath = [
        p
        for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and Path(p).resolve() != (PROJECT_ROOT / "qlib").resolve()
    ]
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(existing_pythonpath)
    else:
        env.pop("PYTHONPATH", None)
    cmd = [
        sys.executable,
        str(dump_script),
        "dump_all",
        "--data_path",
        str(PROCESSED_DIR / "qlib_csv"),
        "--qlib_dir",
        str(QLIB_DIR),
        "--freq",
        "day",
        "--date_field_name",
        "date",
        "--symbol_field_name",
        "symbol",
        "--include_fields",
        "open,high,low,close,volume,factor,amount,vwap",
        "--max_workers",
        str(args.max_workers),
    ]
    logger.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    rewrite_stock_only_instruments()
    for rel in ["features", "calendars", "instruments"]:
        if not (QLIB_DIR / rel).exists():
            raise RuntimeError(f"Qlib dump missing {rel}")
    logger.info("Qlib bin dump complete at %s", QLIB_DIR)


if __name__ == "__main__":
    main()
