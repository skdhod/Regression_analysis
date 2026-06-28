from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "paper"
BENCH_DIR = PROJECT_ROOT / "outputs" / "benchmark"
BACKTEST_PATH = PROJECT_ROOT / "outputs" / "backtest" / "backtest_report.csv"


BENCH_LABELS = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sh000905": "中证500",
    "sh000132": "上证100",
    "sz399330": "深证100",
}


def safe_float(x: float) -> float:
    if x is None or not np.isfinite(x):
        return math.nan
    return float(x)


def ols(y: np.ndarray, x: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    resid = y - fitted
    sse = float(resid.T @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / tss if tss else math.nan
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / (n - k) if n > k and np.isfinite(r2) else math.nan
    h = np.clip(np.sum((x @ xtx_inv) * x, axis=1), 0.0, 0.999999)
    meat = x.T @ (((resid / (1 - h)) ** 2)[:, None] * x)
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov_hc3))
    tval = beta / se
    pval = np.array([2 * (1 - normal_cdf(abs(t))) for t in tval])
    return {
        "beta": beta,
        "se": se,
        "t": tval,
        "p": pval,
        "resid": resid,
        "fitted": fitted,
        "r2": safe_float(r2),
        "adj_r2": safe_float(adj_r2),
        "n": n,
        "k": k,
    }


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        y = df[col].to_numpy(float)
        others = [c for c in cols if c != col]
        x = np.column_stack([np.ones(len(df)), df[others].to_numpy(float)])
        fit = ols(y, x)
        r2 = fit["r2"]
        vif = 1.0 / (1.0 - r2) if np.isfinite(r2) and r2 < 1 else math.inf
        rows.append({"变量": col, "VIF": vif})
    return pd.DataFrame(rows)


def bp_test(resid: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    aux = ols(resid**2, x)
    lm = len(resid) * aux["r2"]
    df = x.shape[1] - 1
    return safe_float(lm), safe_float(chi2_sf(lm, df))


def durbin_watson(resid: np.ndarray) -> float:
    return safe_float(np.diff(resid).dot(np.diff(resid)) / resid.dot(resid))


def jarque_bera_p(resid: np.ndarray) -> tuple[float, float]:
    n = len(resid)
    centered = resid - resid.mean()
    m2 = np.mean(centered**2)
    if m2 <= 0:
        return math.nan, math.nan
    skew = np.mean(centered**3) / (m2 ** 1.5)
    kurt = np.mean(centered**4) / (m2**2)
    jb = n / 6 * (skew**2 + (kurt - 3) ** 2 / 4)
    return safe_float(jb), safe_float(chi2_sf(jb, 2))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_sf(x: float, df: int) -> float:
    if x < 0:
        return 1.0
    if df == 1:
        return 2 * (1 - normal_cdf(math.sqrt(x)))
    if df == 2:
        return math.exp(-x / 2)
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return max(0.0, min(1.0, 1 - normal_cdf(z)))


def star(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_num(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "NaN"
    return f"{x:.{digits}f}"


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def make_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["strategy_return", "sh000001", "sz399001", "sh000905", "turnover"]
    labels = {
        "strategy_return": "策略日收益",
        "sh000001": "上证指数收益",
        "sz399001": "深证成指收益",
        "sh000905": "中证500收益",
        "turnover": "换手率",
    }
    rows = []
    for c in cols:
        s = df[c]
        rows.append(
            {
                "变量": labels[c],
                "均值": s.mean(),
                "标准差": s.std(ddof=1),
                "最小值": s.min(),
                "最大值": s.max(),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_tex(df: pd.DataFrame, numeric_digits: int = 4) -> str:
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule"]
    lines.append("变量 & 均值 & 标准差 & 最小值 & 最大值 \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{tex_escape(r['变量'])} & {fmt_num(r['均值'], numeric_digits)} & "
            f"{fmt_num(r['标准差'], numeric_digits)} & {fmt_num(r['最小值'], numeric_digits)} & "
            f"{fmt_num(r['最大值'], numeric_digits)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def regression_table(models: dict[str, tuple[dict, list[str]]]) -> str:
    row_order = ["const", "sh000001", "sz399001", "sh000905", "turnover"]
    labels = {
        "const": "常数项",
        "sh000001": "上证指数收益",
        "sz399001": "深证成指收益",
        "sh000905": "中证500收益",
        "turnover": "换手率",
    }
    lines = ["\\begin{tabular}{lccc}", "\\toprule"]
    lines.append("变量 & 模型(1) & 模型(2) & 模型(3) \\\\")
    lines.append("\\midrule")
    for var in row_order:
        coef_cells = []
        se_cells = []
        for fit, cols in models.values():
            if var in cols:
                idx = cols.index(var)
                coef_cells.append(f"{fmt_num(fit['beta'][idx])}{star(fit['p'][idx])}")
                se_cells.append(f"({fmt_num(fit['se'][idx])})")
            else:
                coef_cells.append("")
                se_cells.append("")
        lines.append(f"{labels[var]} & " + " & ".join(coef_cells) + " \\\\")
        lines.append(" & " + " & ".join(se_cells) + " \\\\")
    lines.append("\\midrule")
    for stat_name, key in [("样本量", "n"), ("$R^2$", "r2"), ("调整$R^2$", "adj_r2")]:
        vals = []
        for fit, _ in models.values():
            vals.append(str(fit[key]) if key == "n" else fmt_num(fit[key]))
        lines.append(f"{stat_name} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def diagnostics_table(model3: dict, x3: np.ndarray, vifs: pd.DataFrame) -> str:
    lm, bp_p = bp_test(model3["resid"], x3)
    jb, jb_p = jarque_bera_p(model3["resid"])
    dw = durbin_watson(model3["resid"])
    max_vif = vifs["VIF"].replace([np.inf, -np.inf], np.nan).max()
    rows = [
        ("Durbin-Watson统计量", dw, "接近2表示一阶自相关较弱"),
        ("Breusch-Pagan LM", lm, "异方差检验统计量"),
        ("BP检验p值", bp_p, "p值较小表示异方差风险"),
        ("Jarque-Bera统计量", jb, "残差正态性检验统计量"),
        ("JB检验p值", jb_p, "p值较小表示残差偏离正态"),
        ("最大VIF", max_vif, "衡量多重共线性"),
    ]
    lines = ["\\begin{tabular}{lrl}", "\\toprule", "诊断指标 & 数值 & 说明 \\\\", "\\midrule"]
    for name, val, desc in rows:
        lines.append(f"{name} & {fmt_num(val)} & {desc} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def performance_table() -> str:
    p = BENCH_DIR / "benchmark_metrics.csv"
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                r["benchmark_name"],
                r["benchmark_symbol"],
                r["strategy_total_return"],
                r["benchmark_total_return"],
                r["excess_total_return"],
                r["strategy_max_drawdown"],
            ]
        )
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule"]
    lines.append("基准 & 代码 & 策略收益 & 基准收益 & 超额收益 & 最大回撤 \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(
            f"{row[0]} & {row[1]} & {fmt_num(row[2])} & {fmt_num(row[3])} & "
            f"{fmt_num(row[4])} & {fmt_num(row[5])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def make_residual_plot(fitted: np.ndarray, resid: np.ndarray) -> None:
    img = Image.new("RGB", (1100, 720), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
        small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((50, 25), "图3 模型(3)拟合值与残差诊断图", fill=(0, 0, 0), font=font)
    left, top, right, bottom = 90, 100, 1030, 640
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 0), width=2)
    x = fitted
    y = resid
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    pad_y = (ymax - ymin) * 0.08 or 0.01
    ymin, ymax = ymin - pad_y, ymax + pad_y
    zero_y = bottom - (0 - ymin) / (ymax - ymin) * (bottom - top)
    draw.line((left, zero_y, right, zero_y), fill=(180, 0, 0), width=2)
    for xi, yi in zip(x, y):
        px = left + (xi - xmin) / (xmax - xmin) * (right - left)
        py = bottom - (yi - ymin) / (ymax - ymin) * (bottom - top)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(35, 95, 160))
    draw.text((left, bottom + 18), "拟合值", fill=(0, 0, 0), font=small)
    draw.text((left - 70, top + 10), "残差", fill=(0, 0, 0), font=small)
    draw.text((left, bottom + 48), f"残差均值={resid.mean():.6f}，标准差={resid.std(ddof=1):.6f}", fill=(0, 0, 0), font=small)
    img.save(OUT_DIR / "regression_residuals.png")


def load_data() -> pd.DataFrame:
    daily = pd.read_csv(BENCH_DIR / "strategy_vs_benchmarks_daily.csv", parse_dates=["date"])
    pivot = daily.pivot_table(index="date", columns="index_symbol", values="benchmark_return", aggfunc="first")
    strategy = daily.groupby("date")["strategy_return"].first()
    bt = pd.read_csv(BACKTEST_PATH, parse_dates=["date"])[["date", "turnover", "cost"]].set_index("date")
    df = pd.concat([strategy.rename("strategy_return"), pivot, bt], axis=1).dropna()
    return df


def write_paper(context: dict) -> None:
    tex = rf"""
\documentclass[UTF8,a4paper,12pt]{{ctexart}}
\usepackage{{geometry}}
\geometry{{top=2.54cm,bottom=2.54cm,left=3.17cm,right=3.17cm}}
\usepackage{{amsmath,amssymb,booktabs,array,graphicx,float,setspace,caption,hyperref}}
\usepackage{{threeparttable}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}}
\setstretch{{1.35}}
\ctexset{{
  section={{format=\heiti\zihao{{3}}, name={{,}}, number=\arabic{{section}}.}},
  subsection={{format=\heiti\zihao{{4}}, name={{,}}, number=\arabic{{section}}.\arabic{{subsection}}}}
}}
\captionsetup{{font=small,labelfont=bf,labelsep=quad}}
\graphicspath{{{{../benchmark/}}{{./}}}}
\pagestyle{{plain}}
\begin{{document}}
\begin{{titlepage}}
\thispagestyle{{empty}}
\centering
\vspace*{{1.5cm}}
{{\heiti\zihao{{2}} 《回归分析》课堂论文\par}}
\vspace{{3.5cm}}
\begin{{tabular}}{{|>{{\centering\arraybackslash}}p{{4.0cm}}|p{{8.5cm}}|}}
\hline
论文题目 & 基于新浪财经数据的A股量化收益回归分析 \\
\hline
学\quad 院 & \quad \\
\hline
专\quad 业 & \quad \\
\hline
学生姓名 & \quad \\
\hline
学\quad 号 & \quad \\
\hline
\end{{tabular}}
\vfill
{{\zihao{{4}} \today\par}}
\end{{titlepage}}

\begin{{center}}
{{\heiti\zihao{{3}} 基于新浪财经数据的A股量化收益回归分析}}
\end{{center}}

\noindent\textbf{{摘要：}}本文以一个可复现的A股量化项目为研究对象，使用新浪财经公开接口获得股票日K线、复权因子与指数数据，并在Qlib框架下构造Alpha158-LightGBM选股策略。为使研究符合回归分析课程论文的要求，本文不将机器学习收益结果直接视为结论，而是将测试期策略日收益作为被解释变量，以上证指数、深证成指、中证500收益率以及换手率为解释变量，建立多元线性回归模型，检验策略收益是否主要由市场暴露解释，并进一步进行多重共线性、异方差、自相关和残差正态性诊断。结果显示，在较保守的“上一交易日信号、下一交易日开盘成交”设定下，策略在测试期仍取得较高收益，但其收益同时显著暴露于市场因子和交易行为变量；因此，该结果应被理解为研究性回测结果，而非可直接交易的实盘收益。

\noindent\textbf{{关键词：}}回归分析；A股量化；多元线性回归；稳健标准误；Qlib；LightGBM

\section{{前言}}
随着金融市场数据可得性提高，投资组合收益的来源识别成为量化投资研究中的核心问题。机器学习模型能够在高维特征中寻找非线性排序信号，但如果只报告回测收益，容易忽略收益是否来自市场风险暴露、交易成本假设或样本选择偏差。回归分析提供了一个可解释框架：通过将策略收益分解为市场因子、风格因子和交易变量的线性组合，可以检验策略是否具有独立解释力，并识别模型结果中的潜在风险。

本文基于已搭建的A股Qlib项目开展实证研究。项目使用新浪财经公开接口抓取全A股票日频数据、后复权因子和指数基准数据，经清洗后转换为Qlib格式，并使用Alpha158特征与LightGBM模型生成测试期预测信号。为了避免“当天特征、当天收盘成交”的时间错配，本文采用上一交易日信号在下一交易日开盘成交的回测设定。本文的研究目的不是证明某个策略可以直接实盘盈利，而是运用回归分析方法，对策略收益的市场暴露和统计性质进行解释与检验。

\section{{国内外研究现状}}
\subsection{{资产收益解释模型}}
经典资产定价研究通常使用线性回归解释资产收益。Fama和French提出的三因子模型将股票收益分解为市场、规模和账面市值比因子，为后续多因子回归奠定了基础。动量效应研究则表明，过去收益排序可能对未来收益具有预测能力。此类文献的共同特点是将收益率作为被解释变量，通过回归系数判断风险暴露和异常收益。

\subsection{{机器学习与量化选股}}
近年来，机器学习方法被广泛应用于股票收益预测。Gu、Kelly和Xiu的研究表明，非线性模型在经验资产定价中具有较强预测能力。LightGBM等梯度提升树模型能够处理非线性关系和变量交互，在截面选股任务中常用于构造股票排序分数。Qlib等平台则提供了从数据处理、特征构造到回测评估的一体化流程。

\subsection{{文献评述}}
已有研究说明，机器学习模型可能提高预测精度，但也容易受到未来函数、幸存者偏差、交易成本低估和过拟合的影响。因此，单纯报告机器学习策略净值并不足够。本文的改进之处在于，将量化回测结果重新纳入回归分析框架，重点检验收益与市场指数、换手率之间的关系，并通过诊断检验说明模型假设是否满足。

\section{{指标选取和模型设定}}
\subsection{{指标选取}}
本文测试期为2025年5月23日至2026年6月25日，共{context['n']}个交易日。被解释变量为策略日收益率，记为$R_{{p,t}}$。解释变量包括上证指数收益率$R_{{SH,t}}$、深证成指收益率$R_{{SZ,t}}$、中证500收益率$R_{{ZZ500,t}}$和策略日换手率$Turnover_t$。其中，指数收益用于刻画市场风险暴露，换手率用于刻画交易行为和调仓强度。

表\ref{{tab:desc}}报告主要变量的描述性统计。可以看到，测试期策略日收益均值高于主要市场指数，但标准差也不低，说明策略收益并非无风险收益，而是伴随明显日频波动。

\begin{{table}}[H]
\centering
\caption{{主要变量描述性统计}}
\label{{tab:desc}}
{context['desc_tex']}
\end{{table}}

\subsection{{模型设定}}
本文采用逐步扩展的多元线性回归模型：
\begin{{equation}}
R_{{p,t}}=\alpha+\beta_1R_{{SH,t}}+\varepsilon_t ,
\end{{equation}}
\begin{{equation}}
R_{{p,t}}=\alpha+\beta_1R_{{SH,t}}+\beta_2R_{{SZ,t}}+\beta_3R_{{ZZ500,t}}+\varepsilon_t ,
\end{{equation}}
\begin{{equation}}
R_{{p,t}}=\alpha+\beta_1R_{{SH,t}}+\beta_2R_{{SZ,t}}+\beta_3R_{{ZZ500,t}}+\gamma Turnover_t+\varepsilon_t .
\end{{equation}}
其中模型(1)为单指数市场模型，模型(2)加入多市场基准，模型(3)进一步加入换手率。由于金融时间序列容易存在异方差，本文报告HC3稳健标准误。

\section{{实证检验}}
\subsection{{数据来源和描述性分析}}
本文数据均来自新浪财经公开接口。股票池由新浪市场中心接口获取，日K线使用新浪历史K线接口，复权因子使用新浪后复权因子接口，指数基准包括上证指数、深证成指、上证100、深证100与中证500。原始数据经项目脚本清洗后转换为Qlib格式。由于新浪历史K线不返回历史成交额，本文不编造成交额和VWAP，而是将相关字段保留为缺失。

在回测设定上，训练期为2021年1月4日至2024年4月16日，验证期为2024年4月17日至2025年5月22日，测试期为2025年5月23日至2026年6月25日。LightGBM仅在训练集和验证集上拟合，测试期用于样本外评估。图\ref{{fig:nav}}和图\ref{{fig:drawdown}}给出了策略与主要基准的净值及回撤对比。

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{nav_compare_all.png}}
\caption{{策略与五个基准指数的净值对比}}
\label{{fig:nav}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{drawdown_compare_all.png}}
\caption{{策略与五个基准指数的回撤对比}}
\label{{fig:drawdown}}
\end{{figure}}

\subsection{{相关检验}}
回归前需要关注解释变量之间的相关性。如果多个市场指数高度相关，回归系数可能不稳定。本文使用方差膨胀因子（VIF）检验多重共线性，并使用Breusch-Pagan检验异方差、Durbin-Watson统计量检验一阶自相关、Jarque-Bera检验残差正态性。诊断结果见表\ref{{tab:diag}}。结果表明，金融收益率残差并不完全满足经典线性模型的理想假设，因此使用稳健标准误是必要的；同时，市场指数之间存在一定相关性，解释回归系数时应重视整体解释力而非孤立解读单个系数。

\begin{{table}}[H]
\centering
\caption{{模型诊断检验结果}}
\label{{tab:diag}}
{context['diag_tex']}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\textwidth]{{regression_residuals.png}}
\caption{{模型(3)拟合值与残差关系}}
\label{{fig:resid}}
\end{{figure}}

\subsection{{实证回归结果及分析}}
表\ref{{tab:reg}}报告回归结果。模型(1)显示策略收益与上证指数收益存在显著正相关，说明策略并非完全市场中性。模型(2)加入深证成指和中证500后，模型解释力提高，表明策略收益同时受到不同市场板块行情影响。模型(3)进一步加入换手率，换手率系数用于衡量调仓强度与当日收益的关系。

\begin{{table}}[H]
\centering
\caption{{策略日收益率的OLS回归结果}}
\label{{tab:reg}}
\begin{{threeparttable}}
{context['reg_tex']}
\begin{{tablenotes}}
\small
\item 注：括号内为HC3稳健标准误；***、**、*分别表示在1\%、5\%、10\%水平上显著。
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

从经济含义看，若市场指数收益系数显著为正，则说明策略收益的一部分来自对权益市场整体上涨的暴露，而非完全来自个股选择能力。若常数项在控制市场收益后仍为正，则可被解释为样本期内的异常收益，但该解释必须结合数据局限性审慎判断。本文回测仍使用当前时点股票池，不能完全消除幸存者偏差；新浪接口也不是严格point-in-time数据库，因此回归结果应被视为课程研究和方法演示，而非实盘收益承诺。

\begin{{table}}[H]
\centering
\caption{{策略与五个基准的收益表现}}
\label{{tab:perf}}
{context['perf_tex']}
\end{{table}}

\section{{结论及对策建议}}
本文基于新浪财经公开数据构建A股量化策略，并使用回归分析方法检验策略样本外收益的市场暴露。实证结果表明，在更保守的开盘成交设定下，策略测试期仍取得正收益，但回归分析显示其收益与市场指数和交易行为变量存在明显关系。这说明机器学习选股结果需要通过回归分解进一步解释，不能仅凭净值曲线判断策略有效性。

本文建议：第一，后续研究应构建point-in-time股票池，减少幸存者偏差；第二，应进一步加入规模、估值、动量、行业等风格控制变量，建立更完整的多因子回归；第三，应提高交易成本和滑点假设，检验高换手策略在真实交易中的稳健性；第四，应滚动划分训练集和测试集，观察回归系数在不同市场环境下是否稳定。

\section*{{参考文献}}
\addcontentsline{{toc}}{{section}}{{参考文献}}
[1] Fama E F, French K R. Common risk factors in the returns on stocks and bonds[J]. Journal of Financial Economics, 1993, 33(1): 3-56.

[2] Jegadeesh N, Titman S. Returns to buying winners and selling losers: Implications for stock market efficiency[J]. Journal of Finance, 1993, 48(1): 65-91.

[3] Gu S, Kelly B, Xiu D. Empirical asset pricing via machine learning[J]. Review of Financial Studies, 2020, 33(5): 2223-2273.

[4] Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree[C]. Advances in Neural Information Processing Systems, 2017.

[5] Yang X, Liu W, Zhou D, et al. Qlib: An AI-oriented quantitative investment platform[EB/OL]. arXiv preprint arXiv:2009.11189, 2020.

[6] White H. A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity[J]. Econometrica, 1980, 48(4): 817-838.

\end{{document}}
"""
    (OUT_DIR / "regression_analysis_paper.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    desc = make_descriptive(df)
    y = df["strategy_return"].to_numpy(float)

    model_specs = {
        "m1": ["const", "sh000001"],
        "m2": ["const", "sh000001", "sz399001", "sh000905"],
        "m3": ["const", "sh000001", "sz399001", "sh000905", "turnover"],
    }
    models = {}
    for name, cols in model_specs.items():
        x_cols = [c for c in cols if c != "const"]
        x = np.column_stack([np.ones(len(df)), df[x_cols].to_numpy(float)])
        fit = ols(y, x)
        models[name] = (fit, cols)

    x3_cols = ["sh000001", "sz399001", "sh000905", "turnover"]
    x3 = np.column_stack([np.ones(len(df)), df[x3_cols].to_numpy(float)])
    vifs = vif_table(df, x3_cols)
    make_residual_plot(models["m3"][0]["fitted"], models["m3"][0]["resid"])

    desc.to_csv(OUT_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    vifs.to_csv(OUT_DIR / "vif.csv", index=False, encoding="utf-8-sig")
    context = {
        "n": len(df),
        "desc_tex": dataframe_to_tex(desc),
        "reg_tex": regression_table(models),
        "diag_tex": diagnostics_table(models["m3"][0], x3, vifs),
        "perf_tex": performance_table(),
    }
    write_paper(context)


if __name__ == "__main__":
    main()
