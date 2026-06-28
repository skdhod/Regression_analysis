from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from utils import (
    META_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    annual_return,
    ensure_project_dirs,
    max_drawdown,
    safe_ratio,
    setup_logger,
    sharpe,
)

METRIC_COLUMNS = [
    "benchmark_name",
    "benchmark_symbol",
    "start_date",
    "end_date",
    "strategy_total_return",
    "benchmark_total_return",
    "excess_total_return",
    "strategy_annual_return",
    "benchmark_annual_return",
    "excess_annual_return",
    "strategy_max_drawdown",
    "benchmark_max_drawdown",
    "strategy_sharpe",
    "benchmark_sharpe",
    "tracking_error",
    "information_ratio",
    "correlation",
    "beta",
    "alpha",
    "reason",
]


def newest_file(pattern: str) -> Path | None:
    files = list((PROJECT_ROOT / "mlruns").rglob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def corr_safe(left: pd.Series, right: pd.Series) -> float:
    df = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(df) < 2:
        return math.nan
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if den == 0:
        return math.nan
    return float((x * y).sum() / den)


def load_pickle(pattern: str):
    path = newest_file(pattern)
    if path is None:
        return None, None
    try:
        return pd.read_pickle(path), path
    except Exception:  # noqa: BLE001
        return None, path


def export_predictions() -> tuple[pd.DataFrame | None, str]:
    pred, path = load_pickle("pred.pkl")
    if pred is None:
        return None, f"pred.pkl not found or unreadable at {path}"
    df = pred.reset_index()
    df.to_csv(OUTPUT_DIR / "predictions" / "predictions.csv", index=False, encoding="utf-8-sig")
    return df, ""


def export_backtest() -> tuple[pd.DataFrame | None, str]:
    report, path = load_pickle("report_normal_1day.pkl")
    if report is None:
        return None, f"report_normal_1day.pkl not found or unreadable at {path}"
    df = report.copy()
    df.index.name = "date"
    df.reset_index().to_csv(OUTPUT_DIR / "backtest" / "backtest_report.csv", index=False, encoding="utf-8-sig")
    return df, ""


def load_ic_summary() -> dict:
    out = {"ic_mean": math.nan, "rank_ic_mean": math.nan, "reason": ""}
    ic, _ = load_pickle("sig_analysis/ic.pkl")
    ric, _ = load_pickle("sig_analysis/ric.pkl")
    try:
        if ic is not None:
            out["ic_mean"] = float(pd.Series(ic).mean())
        if ric is not None:
            out["rank_ic_mean"] = float(pd.Series(ric).mean())
        if pd.isna(out["ic_mean"]) or pd.isna(out["rank_ic_mean"]):
            pred, _ = load_pickle("pred.pkl")
            label, _ = load_pickle("label.pkl")
            if pred is not None and label is not None:
                pred_s = pred["score"] if isinstance(pred, pd.DataFrame) and "score" in pred.columns else pd.Series(pred)
                label_s = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else pd.Series(label)
                joined = pd.concat([pred_s.rename("score"), label_s.rename("label")], axis=1).dropna()
                if not joined.empty:
                    by_date = joined.groupby(level=0, group_keys=False)
                    ic_series = by_date.apply(lambda x: corr_safe(x["score"], x["label"]))
                    ric_series = by_date.apply(lambda x: corr_safe(x["score"].rank(), x["label"].rank()))
                    if pd.isna(out["ic_mean"]):
                        out["ic_mean"] = float(ic_series.mean())
                    if pd.isna(out["rank_ic_mean"]):
                        out["rank_ic_mean"] = float(ric_series.mean())
    except Exception as exc:  # noqa: BLE001
        out["reason"] = str(exc)
    return out


def strategy_nav_from_report(report: pd.DataFrame) -> pd.DataFrame:
    df = report.copy()
    ret = pd.to_numeric(df.get("return", pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(0)
    cost = pd.to_numeric(df.get("cost", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    nav = (1 + ret - cost).cumprod()
    return pd.DataFrame(
        {"date": pd.to_datetime(df.index), "strategy_return": (ret - cost).to_numpy(), "strategy_nav": nav.to_numpy()}
    ).reset_index(drop=True)


def benchmark_metrics(strategy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    daily_frames = []
    for path in sorted((PROCESSED_DIR / "benchmark_indices").glob("*.csv")):
        bench = pd.read_csv(path)
        if bench.empty:
            continue
        bench["date"] = pd.to_datetime(bench["date"])
        merged = strategy.merge(bench[["date", "index_symbol", "index_name", "return", "nav"]], on="date", how="inner")
        if merged.empty:
            rows.append({"benchmark_name": bench["index_name"].iloc[0], "benchmark_symbol": bench["index_symbol"].iloc[0], "reason": "no date overlap"})
            continue
        merged["benchmark_nav"] = merged["nav"] / merged["nav"].iloc[0]
        merged["strategy_nav"] = merged["strategy_nav"] / merged["strategy_nav"].iloc[0]
        merged["excess_return_daily"] = merged["strategy_return"] - merged["return"].fillna(0)
        bret = merged["return"].fillna(0)
        sret = merged["strategy_return"].fillna(0)
        te = (sret - bret).std(ddof=1) * math.sqrt(252) if len(merged) > 1 else math.nan
        if len(merged) > 1:
            s_arr = sret.to_numpy(dtype=float)
            b_arr = bret.to_numpy(dtype=float)
            b_var = float(((b_arr - b_arr.mean()) ** 2).mean())
            cov = float(((s_arr - s_arr.mean()) * (b_arr - b_arr.mean())).mean())
            beta = safe_ratio(cov, b_var)
        else:
            beta = math.nan
        alpha = annual_return(merged["strategy_nav"]) - beta * annual_return(merged["benchmark_nav"]) if not pd.isna(beta) else math.nan
        rows.append(
            {
                "benchmark_name": merged["index_name"].iloc[0],
                "benchmark_symbol": merged["index_symbol"].iloc[0],
                "start_date": merged["date"].min().strftime("%Y-%m-%d"),
                "end_date": merged["date"].max().strftime("%Y-%m-%d"),
                "strategy_total_return": merged["strategy_nav"].iloc[-1] - 1,
                "benchmark_total_return": merged["benchmark_nav"].iloc[-1] - 1,
                "excess_total_return": merged["strategy_nav"].iloc[-1] - merged["benchmark_nav"].iloc[-1],
                "strategy_annual_return": annual_return(merged["strategy_nav"]),
                "benchmark_annual_return": annual_return(merged["benchmark_nav"]),
                "excess_annual_return": annual_return((1 + merged["excess_return_daily"]).cumprod()),
                "strategy_max_drawdown": max_drawdown(merged["strategy_nav"]),
                "benchmark_max_drawdown": max_drawdown(merged["benchmark_nav"]),
                "strategy_sharpe": sharpe(sret),
                "benchmark_sharpe": sharpe(bret),
                "tracking_error": te,
                "information_ratio": safe_ratio((sret - bret).mean() * 252, te),
                "correlation": corr_safe(sret, bret),
                "beta": beta,
                "alpha": alpha,
                "reason": "",
            }
        )
        daily_frames.append(
            merged[
                [
                    "date",
                    "index_name",
                    "index_symbol",
                    "strategy_return",
                    "return",
                    "strategy_nav",
                    "benchmark_nav",
                    "excess_return_daily",
                ]
            ].rename(columns={"return": "benchmark_return"})
        )
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    metrics.to_csv(OUTPUT_DIR / "benchmark" / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    if daily_frames:
        daily = pd.concat(daily_frames, ignore_index=True)
        daily.to_csv(OUTPUT_DIR / "benchmark" / "strategy_vs_benchmarks_daily.csv", index=False, encoding="utf-8-sig")
        nav = daily.pivot(index="date", columns="index_name", values="benchmark_nav")
        nav.insert(0, "strategy", daily.drop_duplicates("date").set_index("date")["strategy_nav"])
        nav.to_csv(OUTPUT_DIR / "benchmark" / "benchmark_nav.csv", encoding="utf-8-sig")
        plot_nav(nav)
        plot_drawdown(nav)
    return metrics


def plot_line_image(df: pd.DataFrame, path: Path, title: str) -> None:
    img = Image.new("RGB", (1200, 650), "white")
    draw = ImageDraw.Draw(img)
    margin_l, margin_t, margin_r, margin_b = 70, 55, 35, 80
    x0, y0 = margin_l, margin_t
    x1, y1 = img.size[0] - margin_r, img.size[1] - margin_b
    draw.text((margin_l, 20), title, fill="black")
    draw.rectangle((x0, y0, x1, y1), outline="black")
    values = df.apply(pd.to_numeric, errors="coerce")
    ymin = float(np.nanmin(values.to_numpy()))
    ymax = float(np.nanmax(values.to_numpy()))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = 0.0, 1.0
    span = ymax - ymin
    colors = ["blue", "red", "green", "orange", "purple", "brown", "black"]
    n = max(len(values) - 1, 1)
    for idx, col in enumerate(values.columns):
        series = values[col].to_numpy(dtype=float)
        pts = []
        for i, val in enumerate(series):
            if not np.isfinite(val):
                continue
            x = x0 + int((x1 - x0) * i / n)
            y = y1 - int((y1 - y0) * (val - ymin) / span)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=colors[idx % len(colors)], width=2)
        label = "strategy" if idx == 0 else f"bench{idx}"
        draw.line((margin_l + idx * 145, img.size[1] - 45, margin_l + idx * 145 + 35, img.size[1] - 45), fill=colors[idx % len(colors)], width=3)
        draw.text((margin_l + idx * 145 + 42, img.size[1] - 53), label, fill="black")
    draw.text((10, y0), f"{ymax:.2f}", fill="black")
    draw.text((10, y1 - 10), f"{ymin:.2f}", fill="black")
    img.save(path)


def plot_nav(nav: pd.DataFrame) -> None:
    plot_line_image(nav, OUTPUT_DIR / "benchmark" / "nav_compare_all.png", "Strategy vs Benchmarks NAV")


def plot_drawdown(nav: pd.DataFrame) -> None:
    dd = nav / nav.cummax() - 1
    plot_line_image(dd, OUTPUT_DIR / "benchmark" / "drawdown_compare_all.png", "Strategy vs Benchmarks Drawdown")


def write_summary(pred_err: str, backtest_err: str, report: pd.DataFrame | None, metrics: pd.DataFrame, ic_summary: dict) -> None:
    quality_path = META_DIR / "data_quality_report.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    if not quality.empty:
        quality.to_csv(OUTPUT_DIR / "reports" / "data_quality_summary.csv", index=False, encoding="utf-8-sig")
    seg_path = META_DIR / "qlib_segments.csv"
    segments = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
    lines = [
        "# Summary",
        "",
        "## 数据源说明",
        "- 股票列表、日 K、复权因子、指数数据均来自新浪财经公开/网页接口。",
        "- 未使用 AKShare、Tushare、Baostock、Wind。",
        "- 新浪历史 K 线当前不返回历史 amount，因此 amount 与 vwap 不编造，保留 NaN。",
        "",
        "## 数据质量",
        f"- 股票池数量: {len(quality) if not quality.empty else 'NaN'}",
        f"- 有效样本数: {int(quality['rows'].sum()) if 'rows' in quality else 'NaN'}",
        f"- amount 可用股票数: {int(quality['has_amount'].sum()) if 'has_amount' in quality else 0}",
        f"- vwap 可用股票数: {int(quality['has_vwap'].sum()) if 'has_vwap' in quality else 0}",
        f"- factor 来源统计: {quality['factor_source'].value_counts().to_dict() if 'factor_source' in quality else {}}",
        "",
        "## Qlib Workflow",
        f"- 训练/验证/测试区间: {segments.to_dict(orient='records') if not segments.empty else 'NaN'}",
        "- LightGBM 参数: loss=mse, learning_rate=0.2, num_leaves=210, max_depth=8。",
        f"- IC: {ic_summary.get('ic_mean')}",
        f"- Rank IC: {ic_summary.get('rank_ic_mean')}",
    ]
    if report is not None:
        strat = strategy_nav_from_report(report)
        turnover = report.get("turnover", pd.Series(dtype=float))
        lines += [
            f"- 策略收益: {strat['strategy_nav'].iloc[-1] - 1 if not strat.empty else math.nan}",
            f"- 最大回撤: {max_drawdown(strat['strategy_nav']) if not strat.empty else math.nan}",
            f"- 夏普比率: {sharpe(strat['strategy_return']) if not strat.empty else math.nan}",
            f"- 换手率: {float(pd.to_numeric(turnover, errors='coerce').mean()) if len(turnover) else math.nan}",
        ]
    else:
        lines.append(f"- 回测读取失败: {backtest_err}")
    if pred_err:
        lines.append(f"- 预测读取失败: {pred_err}")
    lines += ["", "## 五个 benchmark 对比"]
    if metrics.empty:
        lines.append("- benchmark_metrics.csv 未生成有效对比。")
    else:
        for row in metrics.itertuples(index=False):
            lines.append(
                f"- {row.benchmark_name}({row.benchmark_symbol}): excess_total_return={getattr(row, 'excess_total_return', math.nan)}, information_ratio={getattr(row, 'information_ratio', math.nan)}"
            )
    lines += [
        "",
        "## 项目局限性",
        "- 新浪接口没有稳定的正式 SLA，字段和可用 datalen 可能变化。",
        "- 历史 K 线缺少 amount，导致真实 vwap 无法计算。",
        "- 新股覆盖期天然短于请求区间，质量报告中需单独说明。",
    ]
    (OUTPUT_DIR / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("analyze_results", "09_analyze_results.log")
    _, pred_err = export_predictions()
    report, backtest_err = export_backtest()
    ic_summary = load_ic_summary()
    metrics = pd.DataFrame()
    if report is not None:
        strategy = strategy_nav_from_report(report)
        metrics = benchmark_metrics(strategy)
    else:
        pd.DataFrame(columns=METRIC_COLUMNS).to_csv(OUTPUT_DIR / "benchmark" / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    write_summary(pred_err, backtest_err, report, metrics, ic_summary)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
