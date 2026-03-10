# features/base_feature.py
import pandas as pd
from abc import ABC, abstractmethod
from  utils.log_utils import logger

class BaseFeature(ABC):

    def __init__(self):

        # 因子名称
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, stock_list, trade_date, market_data):

        """
        计算因子

        Parameters
        ----------
        stock_list : list
            候选股票池

        trade_date : str
            交易日期

        market_data : dict
            已加载的数据

        Returns
        -------
        DataFrame
        """
        pass