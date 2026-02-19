#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：全市场分仓涨停策略回测
"""
from backtest.engine import MultiStockBacktestEngine
from config.config import DEFAULT_INIT_CAPITAL
from strategies.multi_limit_up_strategy import MultiLimitUpStrategy


def main():
    # ===================== 回测参数配置 =====================
    START_DATE = "2026-01-01"  # 回测开始日期
    END_DATE = "2026-02-05"    # 回测结束日期
    INIT_CAPITAL = DEFAULT_INIT_CAPITAL  # 初始本金10W元

    # ===================== 初始化策略与回测引擎 =====================
    strategy = MultiLimitUpStrategy()
    engine = MultiStockBacktestEngine(
        strategy=strategy,
        init_capital=INIT_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # ===================== 运行回测 =====================
    backtest_result = engine.run()

    # ===================== 结果保存（可选） =====================
    # backtest_result["net_value_df"].to_csv("分仓涨停策略净值曲线.csv", index=False)
    # backtest_result["trade_df"].to_csv("分仓涨停策略交易记录.csv", index=False)

if __name__ == "__main__":
    main()