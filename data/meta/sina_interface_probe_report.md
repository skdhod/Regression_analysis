# Sina Interface Probe Report

- Probe time: 2026-06-28T21:07:52
- Data source: Sina Finance public/web interfaces only.
- Daily K interface available: True.
- `scale=240` returns daily K data: True.
- Maximum usable `datalen` observed for sh600000: 1970.
- 1023 hard limit: No. Current probe shows larger values work, while very large values return `null`.
- Best observed coverage: 2018-05-15 to 2026-06-26.
- Historical K fields: close, day, high, low, open, volume.
- Historical `amount` available: False.
- Realtime API has amount-like fields but will not be used to fill historical amount.
- HFQ factor available: True.
- QFQ factor available: True.
- Factor format: Sina JavaScript variable assignment, parsed into JSON object.

## Benchmark Selection
- 上证指数: sh000001 from candidates sh000001
- 深证成指: sz399001 from candidates sz399001
- 上证100: sh000132 from candidates sh000132
- 深证100: sz399330 from candidates sz399330
- 中证500: sh000905 from candidates sh000905, sz399905

## Final Interface Plan
- Stocks and indices: CN_MarketDataService.getKLineData with detected safe `datalen`.
- Stock list: Market_Center.getHQNodeData nodes `sh_a`, `sz_a`, `cyb`, `kcb`.
- Factors: realstock/company/{symbol}/hfq.js first, qfq.js fallback.
- Missing amount/vwap will be kept as NaN and disclosed in logs/reports.