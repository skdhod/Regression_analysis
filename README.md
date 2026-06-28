# 新浪 A 股 Qlib Alpha158 + LightGBM 项目

## 课程作业说明

本仓库是《回归分析》课程作业项目，论文题目为《基于新浪财经数据的 A 股指数增强模型收益回归分析》。仓库保留了可复现的抓取、清洗、Qlib 转换、LightGBM 训练、回测分析和论文生成脚本；大体量原始行情数据、Qlib 二进制数据、mlflow 运行缓存和逐股票中间文件不直接提交到 GitHub，可通过 `run_all.py` 在本地重新生成。

主要交付物：

- 论文 PDF：`outputs/paper/regression_analysis_paper_final.pdf`
- 论文 TeX：`outputs/paper/regression_analysis_paper.tex`
- 参考文献核验：`outputs/paper/reference_verification.md`
- 回测摘要：`outputs/reports/summary.md`
- Benchmark 指标：`outputs/benchmark/benchmark_metrics.csv`
- 核心脚本：`scripts/00_*` 至 `scripts/11_*`

本项目在当前目录内完成一条可复现的 A 股量化研究链路：直接从新浪财经公开/网页接口抓取股票池、日 K、复权因子和 benchmark 指数，处理为 Qlib 可用格式，再运行 Alpha158 + LightGBM workflow、回测并输出五个 benchmark 对比。

## 为什么不用 AKShare

本项目显式不使用 AKShare、Tushare、Baostock、Wind。所有行情和列表数据都由项目脚本直接请求新浪财经接口，便于在报告中诚实说明字段来源、接口限制和缺失字段。

## 新浪接口

- 股票/指数日 K：`https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData`
- 复权因子：`https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js` 和 `qfq.js`
- 股票列表：`https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData`
- 实时行情：`https://hq.sinajs.cn/list={symbol}`，只用于探测字段，不用于填充历史 amount

接口探测会写入 `data/meta/sina_interface_probe_report.md` 和 `data/meta/sina_interface_probe_result.json`。当前探测显示历史 K 线不是 1023 硬限制，但过大的 `datalen` 会返回 `null`；脚本会以探测结果为准。

## 数据说明

股票池覆盖沪市主板、深市主板、创业板、科创板，默认排除北交所。五个指数仅作为 benchmark，不进入训练股票池：上证指数、深证成指、上证100、深证100、中证500。

新浪历史 K 线可能不返回历史 `amount`。如果拿不到，标准数据保留 `amount=NaN`，`vwap=NaN`；项目不会用实时成交额补历史字段，也不会用 OHLC 均值冒充 vwap。复权因子优先使用 hfq，失败后使用 qfq，都失败时使用 `factor=1.0` 并在质量报告中标记。

## 安装

```bash
python -m pip install -r requirements.txt
```

如果使用当前目录内的本地 Qlib 源码，脚本会自动把 `qlib/` 加入 `PYTHONPATH`。

## 一键运行

```bash
python run_all.py --start 20210101 --end 20260628 --mode strict-5y
```

如果探测结果无法覆盖完整区间，`strict-5y` 会停止；可以显式接受有限历史：

```bash
python run_all.py --start 20210101 --end 20260628 --mode allow-1023
```

只运行接口探测：

```bash
python run_all.py --probe-only
```

跳过抓取、只重建数据和 Qlib bin：

```bash
python run_all.py --skip-fetch
python run_all.py --only-build-data
```

只重新跑 Qlib workflow 和分析：

```bash
python run_all.py --only-run-qlib
```

少量股票 smoke test：

```bash
python run_all.py --start 20210101 --end 20260628 --mode strict-5y --limit 20
```

## 输出

- 标准日频数据：`data/processed/daily_standard/`
- Qlib CSV：`data/processed/qlib_csv/`
- Qlib bin：`data/qlib/cn_data/`
- 数据质量报告：`data/meta/data_quality_report.csv`
- 预测：`outputs/predictions/predictions.csv`
- 回测：`outputs/backtest/backtest_report.csv`
- benchmark 对比：`outputs/benchmark/benchmark_metrics.csv`
- 总结报告：`outputs/reports/summary.md`

## 期末报告写法建议

报告应重点说明：数据完全来自新浪公开接口；历史 K 线字段没有 amount 时不计算真实 vwap；复权因子来自新浪 hfq/qfq JS 接口；股票池为全 A 但排除北交所；五个指数仅用于基准比较；接口可用性、最大 `datalen` 和覆盖期以 `00_probe_sina_interfaces.py` 的实际探测报告为准。任何 NaN 指标都应说明是字段缺失、Qlib 产物缺失或日期无法对齐导致，而不是用估算值替代。

## 局限性

新浪网页接口没有正式 SLA，字段、分页、最大 `datalen` 都可能变化。新股和停牌股票会导致覆盖期短或样本少。若 `amount` 长期缺失，成交额、真实 vwap 和依赖成交额的研究结论都不应在报告中过度解释。
