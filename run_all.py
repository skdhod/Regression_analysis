from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(script: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script)]
    if args:
        cmd.extend(args)
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sina A-share Qlib pipeline.")
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20260628")
    parser.add_argument("--mode", choices=["strict-5y", "allow-1023"], default="strict-5y")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--only-build-data", action="store_true")
    parser.add_argument("--only-run-qlib", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test stock limit.")
    args = parser.parse_args()

    date_args = ["--start", args.start, "--end", args.end]
    limit_args = ["--limit", str(args.limit)] if args.limit else []
    try:
        if args.probe_only:
            run_step("00_probe_sina_interfaces.py")
            return
        if args.only_build_data:
            run_step("05_build_standard_daily.py", limit_args)
            run_step("06_convert_to_qlib_format.py", limit_args)
            run_step("07_dump_qlib_bin.py", ["--force"])
            return
        if args.only_run_qlib:
            run_step("08_run_alpha158_lightgbm.py")
            run_step("09_analyze_results.py")
            return

        run_step("00_probe_sina_interfaces.py")
        if not args.skip_fetch:
            run_step("01_fetch_stock_list_from_sina.py")
            run_step("02_fetch_daily_kline_from_sina.py", date_args + ["--mode", args.mode] + limit_args)
            run_step("03_fetch_factor_from_sina.py", limit_args)
            run_step("04_fetch_benchmark_indices_from_sina.py", date_args)
        run_step("05_build_standard_daily.py", limit_args)
        run_step("06_convert_to_qlib_format.py", limit_args)
        run_step("07_dump_qlib_bin.py", ["--force"])
        run_step("08_run_alpha158_lightgbm.py")
        run_step("09_analyze_results.py")
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline stopped. Failed step returned code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
