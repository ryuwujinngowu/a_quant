# learnEngine/label.py
import pandas as pd
import numpy as np
from utils.log_utils import logger


def generate_stock_label(stock_daily_df: pd.DataFrame, trade_date: str, next_trade_date: str) -> int:
    """
    生成单只个股的标签（和你的策略逻辑完全对齐）
    :param stock_daily_df: 个股T日和T+1日的日线数据
    :param trade_date: T日（买入日）
    :param next_trade_date: T+1日（卖出日）
    :return: 1=T+1日上涨赚钱，0=T+1日下跌亏钱
    """
    try:
        # T日收盘价（买入价）
        buy_price = stock_daily_df[stock_daily_df["trade_date"] == trade_date]["close"].iloc[0]
        # T+1日收盘价（卖出价）
        sell_price = stock_daily_df[stock_daily_df["trade_date"] == next_trade_date]["close"].iloc[0]

        # 计算收益，扣除0.1%的手续费（贴合实盘）
        profit_rate = (sell_price - buy_price) / buy_price - 0.001
        # 标签：收益>0则为1（正样本），否则为0（负样本）
        return 1 if profit_rate > 0 else 0
    except Exception as e:
        logger.error(f"生成标签失败：{e}")
        return 0  # 异常情况标记为负样本