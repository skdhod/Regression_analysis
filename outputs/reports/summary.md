# Summary

## 数据源说明
- 股票列表、日 K、复权因子、指数数据均来自新浪财经公开/网页接口。
- 未使用 AKShare、Tushare、Baostock、Wind。
- 新浪历史 K 线当前不返回历史 amount，因此 amount 与 vwap 不编造，保留 NaN。

## 数据质量
- 股票池数量: 5206
- 有效样本数: 6360837
- amount 可用股票数: 0
- vwap 可用股票数: 0
- factor 来源统计: {'hfq': 5205, 'fallback_1.0': 1}

## Qlib Workflow
- 训练/验证/测试区间: [{'segment': 'train', 'start_date': '2021-01-04', 'end_date': '2024-04-16'}, {'segment': 'valid', 'start_date': '2024-04-17', 'end_date': '2025-05-22'}, {'segment': 'test', 'start_date': '2025-05-23', 'end_date': '2026-06-25'}]
- LightGBM 参数: loss=mse, learning_rate=0.2, num_leaves=210, max_depth=8。
- IC: 0.03557268070440766
- Rank IC: 0.038517884350095094
- 策略收益: 0.9057725799830156
- 最大回撤: -0.1134149741816215
- 夏普比率: 3.2030758149542122
- 换手率: 0.22245968786642664

## 五个 benchmark 对比
- 上证指数(sh000001): excess_total_return=0.6752399509919715, information_ratio=2.6369548218865466
- 上证100(sh000132): excess_total_return=0.6851264131845403, information_ratio=2.2031241914271185
- 中证500(sh000905): excess_total_return=0.32467556393290153, information_ratio=0.9229714977100567
- 深证成指(sz399001): excess_total_return=0.292722992184967, information_ratio=0.795891491512758
- 深证100(sz399330): excess_total_return=0.3455527301957979, information_ratio=0.8875580702802431

## 项目局限性
- 新浪接口没有稳定的正式 SLA，字段和可用 datalen 可能变化。
- 历史 K 线缺少 amount，导致真实 vwap 无法计算。
- 新股覆盖期天然短于请求区间，质量报告中需单独说明。