import pandas as pd
import numpy as np
from utils.log_utils import logger
from config.config import ANNUAL_TRADE_DAYS, RISK_FREE_RATE


class BacktestMetrics:
    """回测指标计算类：输入净值曲线，输出所有核心收益风险指标"""

    def __init__(self, net_value_df: pd.DataFrame, init_capital: float, trade_df: pd.DataFrame):
        self.net_value_df = net_value_df
        self.init_capital = init_capital
        self.trade_df = trade_df
        # 计算日收益率
        self.net_value_df["daily_return"] = self.net_value_df["total_asset"].pct_change().fillna(0)

    def calc_all_metrics(self) -> dict:
        """计算所有核心回测指标"""
        total_days = len(self.net_value_df)
        final_asset = self.net_value_df["total_asset"].iloc[-1]

        # 1. 收益类指标
        total_return = (final_asset - self.init_capital) / self.init_capital * 100
        annual_return = (final_asset / self.init_capital) ** (ANNUAL_TRADE_DAYS / total_days) - 1
        annual_return = annual_return * 100

        # 2. 风险类指标
        self.net_value_df["cum_max"] = self.net_value_df["total_asset"].cummax()
        self.net_value_df["drawdown"] = (self.net_value_df["total_asset"] - self.net_value_df["cum_max"]) / \
                                        self.net_value_df["cum_max"] * 100
        max_drawdown = self.net_value_df["drawdown"].min()

        # 3. 风险收益比
        excess_return = self.net_value_df["daily_return"] - (RISK_FREE_RATE / ANNUAL_TRADE_DAYS)
        sharpe_ratio = np.sqrt(
            ANNUAL_TRADE_DAYS) * excess_return.mean() / excess_return.std() if excess_return.std() != 0 else 0

        # 4. 交易类指标
        total_trade_times = len(self.trade_df) // 2
        win_rate = 0
        if total_trade_times > 0:
            # 计算盈利交易次数
            sell_trades = self.trade_df[self.trade_df["direction"] == "卖出"]
            buy_trades = self.trade_df[self.trade_df["direction"] == "买入"].reset_index(drop=True)
            win_count = 0
            for idx, sell in sell_trades.reset_index(drop=True).iterrows():
                if idx >= len(buy_trades):
                    break
                buy_cost = buy_trades.iloc[idx]["total_cost"]
                sell_income = sell["total_income"]
                if sell_income > buy_cost:
                    win_count += 1
            win_rate = win_count / total_trade_times * 100

        # 汇总结果
        metrics = {
            "初始本金(元)": round(self.init_capital, 2),
            "最终资产(元)": round(final_asset, 2),
            "总收益率(%)": round(total_return, 2),
            "年化收益率(%)": round(annual_return, 2),
            "最大回撤(%)": round(max_drawdown, 2),
            "夏普比率": round(sharpe_ratio, 2),
            "交易胜率(%)": round(win_rate, 2),
            "总交易次数": total_trade_times,
            "回测交易日数": total_days
        }

        logger.info("回测指标计算完成")
        return metrics