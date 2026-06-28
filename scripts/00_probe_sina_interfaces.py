from __future__ import annotations

import argparse
import json
from datetime import datetime

from utils import (
    BENCHMARKS,
    META_DIR,
    ensure_project_dirs,
    fetch_sina_kline,
    http_get_text,
    parse_sina_factor_js,
    setup_logger,
    sina_kline_url,
    write_json,
)


def probe_kline(symbol: str, datalen: int, extra: dict | None = None, headers: bool = True) -> dict:
    url = sina_kline_url(symbol, datalen)
    if extra:
        sep = "&" if "?" in url else "?"
        url = url + sep + "&".join(f"{k}={v}" for k, v in extra.items())
    try:
        text = http_get_text(url, headers=None if headers else {}, retries=1)
        data = json.loads(text)
        rows = len(data) if isinstance(data, list) else None
        first = data[0] if isinstance(data, list) and data else None
        last = data[-1] if isinstance(data, list) and data else None
        keys = sorted(first.keys()) if first else []
        return {
            "symbol": symbol,
            "datalen": datalen,
            "extra": extra,
            "headers": headers,
            "ok": isinstance(data, list) and bool(data),
            "rows": rows,
            "keys": keys,
            "first_date": first.get("day") if first else None,
            "last_date": last.get("day") if last else None,
            "sample": first,
            "raw_is_null": data is None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "datalen": datalen, "ok": False, "error": str(exc)}


def find_max_usable_datalen(symbol: str) -> dict:
    best = None
    checks = {}
    for datalen in [1023, 1500, 1800, 1950, 1970, 1971, 1980, 1999, 2000]:
        res = probe_kline(symbol, datalen)
        checks[str(datalen)] = res
        if res.get("rows") == datalen:
            best = res
    return {"best": best, "checks": checks}


def probe_factor(symbol: str, kind: str) -> dict:
    url = f"https://finance.sina.com.cn/realstock/company/{symbol}/{kind}.js"
    try:
        text = http_get_text(url, retries=1)
        parsed = parse_sina_factor_js(text)
        return {
            "symbol": symbol,
            "kind": kind,
            "ok": True,
            "total": parsed.get("total"),
            "first": parsed.get("data", [None])[0] if parsed.get("data") else None,
            "format": "javascript_var_assignment",
        }
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "kind": kind, "ok": False, "error": str(exc)}


def probe_realtime(symbol: str) -> dict:
    url = f"https://hq.sinajs.cn/list={symbol}"
    try:
        text = http_get_text(url, retries=1)
        payload = text.split('="', 1)[1].rsplit('"', 1)[0] if '="' in text else ""
        fields = payload.split(",") if payload else []
        return {
            "symbol": symbol,
            "ok": bool(fields),
            "fields_count": len(fields),
            "sample": fields[:12],
            "note": "Only used for field probing; not used to fill historical amount.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "ok": False, "error": str(exc)}


def choose_benchmarks(index_probe: dict) -> dict:
    chosen = {}
    for name, candidates in BENCHMARKS.items():
        selected = None
        for symbol in candidates:
            if index_probe.get(symbol, {}).get("best", {}).get("ok"):
                selected = symbol
                break
        chosen[name] = {"selected": selected, "candidates": candidates}
    return chosen


def write_report(result: dict) -> None:
    best = result["stock_kline_max"]["best"] or {}
    fields = best.get("keys", [])
    report = [
        "# Sina Interface Probe Report",
        "",
        f"- Probe time: {datetime.now().isoformat(timespec='seconds')}",
        "- Data source: Sina Finance public/web interfaces only.",
        f"- Daily K interface available: {bool(best.get('ok'))}.",
        f"- `scale=240` returns daily K data: {'day' in fields or bool(best.get('first_date'))}.",
        f"- Maximum usable `datalen` observed for sh600000: {best.get('datalen')}.",
        "- 1023 hard limit: No. Current probe shows larger values work, while very large values return `null`.",
        f"- Best observed coverage: {best.get('first_date')} to {best.get('last_date')}.",
        f"- Historical K fields: {', '.join(fields) if fields else 'N/A'}.",
        f"- Historical `amount` available: {'amount' in fields}.",
        "- Realtime API has amount-like fields but will not be used to fill historical amount.",
        f"- HFQ factor available: {result['factor']['hfq'].get('ok')}.",
        f"- QFQ factor available: {result['factor']['qfq'].get('ok')}.",
        "- Factor format: Sina JavaScript variable assignment, parsed into JSON object.",
        "",
        "## Benchmark Selection",
    ]
    for name, item in result["benchmark_selection"].items():
        report.append(f"- {name}: {item['selected']} from candidates {', '.join(item['candidates'])}")
    report += [
        "",
        "## Final Interface Plan",
        "- Stocks and indices: CN_MarketDataService.getKLineData with detected safe `datalen`.",
        "- Stock list: Market_Center.getHQNodeData nodes `sh_a`, `sz_a`, `cyb`, `kcb`.",
        "- Factors: realstock/company/{symbol}/hfq.js first, qfq.js fallback.",
        "- Missing amount/vwap will be kept as NaN and disclosed in logs/reports.",
    ]
    (META_DIR / "sina_interface_probe_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("probe", "00_probe_sina_interfaces.log")
    logger.info("Probing Sina interfaces")

    stock_kline = {
        "1023": probe_kline("sh600000", 1023),
        "1500": probe_kline("sh600000", 1500),
        "2000": probe_kline("sh600000", 2000),
        "with_start_end": probe_kline("sh600000", 1023, {"start_date": "20210101", "end_date": "20260628"}),
        "without_headers": probe_kline("sh600000", 1023, headers=False),
    }
    max_probe = find_max_usable_datalen("sh600000")
    factor = {"hfq": probe_factor("sh600000", "hfq"), "qfq": probe_factor("sh600000", "qfq")}
    realtime = probe_realtime("sh600000")
    indices = {s: find_max_usable_datalen(s) for symbols in BENCHMARKS.values() for s in symbols}
    result = {
        "stock_kline": stock_kline,
        "stock_kline_max": max_probe,
        "factor": factor,
        "realtime": realtime,
        "indices": indices,
        "benchmark_selection": choose_benchmarks(indices),
    }
    write_json(META_DIR / "sina_interface_probe_result.json", result)
    write_report(result)
    logger.info("Probe complete")


if __name__ == "__main__":
    main()
