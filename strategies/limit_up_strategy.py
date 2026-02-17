import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger


class LimitUpHoldStrategy(BaseStrategy):
    """
    连板持有策略：无固定持仓天数，断板卖出
    规则：
    1. 买入：前一日涨停，当日开盘买入
    2. 卖出：当日没有涨停（断板），次日开盘卖出；触发止损立刻卖出
    3. 无固定持有天数，只要每天都涨停就一直持有
    """
    def __init__(self, stop_loss_rate: float = 0.08):
        self.stop_loss_rate = stop_loss_rate
        # 持仓状态
        self.is_holding = False
        self.buy_price = 0.0
        self.initialize()

    def initialize(self):
        self.is_holding = False
        self.buy_price = 0.0
        logger.info("连板持有策略初始化完成")

    def generate_signal(self, row: pd.Series) -> int:
        # 空仓状态：买入
        if not self.is_holding:
            if row.get("next_day_buy_signal", 0) == 1:
                self.is_holding = True
                self.buy_price = row["open"]
                logger.debug(f"[{row['trade_date']}] 生成买入信号，买入价：{self.buy_price}")
                return 1
            return 0

        # 持仓状态：判断是否卖出
        else:
            # 卖出条件1：触发止损
            if row["low"] <= self.buy_price * (1 - self.stop_loss_rate):
                self.initialize()
                logger.debug(f"[{row['trade_date']}] 触发止损，生成卖出信号")
                return -1
            # 卖出条件2：当日没有涨停（断板），次日卖出
            if row["is_limit_up"] == 0:
                self.initialize()
                logger.debug(f"[{row['trade_date']}] 断板，生成卖出信号")
                return -1
            # 继续涨停，继续持有
            return 0