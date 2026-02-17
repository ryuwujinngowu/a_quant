import pandas as pd
from abc import ABC, abstractmethod
from utils.log_utils import logger

class BaseStrategy(ABC):
    """
    策略基类：所有策略必须继承该类，实现2个核心方法
    统一接口规范，回测引擎仅依赖该基类，不耦合具体策略实现
    """

    @abstractmethod
    def initialize(self):
        """
        策略初始化方法
        用途：重置持仓状态、初始化参数，每次回测前必须调用
        """
        pass

    @abstractmethod
    def generate_signal(self, row: pd.Series) -> int:
        """
        逐行生成买卖信号（策略核心逻辑）
        :param row: 单日K线+特征数据
        :return: 信号：1=买入，-1=卖出，0=持仓/空仓无操作
        """
        pass