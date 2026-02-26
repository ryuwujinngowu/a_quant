# features/market_stats.py
import pandas as pd
from .base_feature import BaseFeature

class DailyLimitCountFeature(BaseFeature):
    """
    市场每日统计特征：
    - limit_up_count: 当日涨停股票数量
    - limit_down_count: 当日跌停股票数量
    完全喂给机器学习的标准格式
    """

    def __init__(self, data_api=None):
        super().__init__(data_api)


    def _get_daily_limit_up_down(self, trade_date: str) -> dict:
        """
        你自己实现的内部方法
        入参规定死：
            trade_date: 日期字符串，格式必须为 'YYYY-MM-DD'
        返回格式规定死：
            {
                "limit_up_count": int,   # 涨停家数
                "limit_down_count": int  # 跌停家数
            }
        如果当天无数据，返回 {0, 0}
        """
        # ========================
        # 👇👇👇 这里面你自己写逻辑
        # 从你的数据库/Tushare/接口获取
        # ========================
        raise NotImplementedError("请你实现 _get_daily_limit_up_down 方法")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基类强制要求实现的方法
        口径统一、无未来函数、机器学习友好
        """
        # 按交易日去重，避免重复请求
        unique_dates = df["trade_date"].unique()

        # 构建每日涨跌停数量映射
        date_limit_map = {}
        for date in unique_dates:
            date_limit_map[date] = self._get_daily_limit_up_down(date)

        # 把当日涨停数、跌停数加到原DF
        df["limit_up_count"] = df["trade_date"].map(lambda x: date_limit_map[x]["limit_up_count"])
        df["limit_down_count"] = df["trade_date"].map(lambda x: date_limit_map[x]["limit_down_count"])

        return df