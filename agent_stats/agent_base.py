"""
智能体策略基类
所有新增智能体必须继承该类，实现get_signal_stock_pool方法
保证引擎统一调用，完全解耦
"""
from typing import List, Dict
import pandas as pd


class BaseAgent:
    # 必须由子类实现的属性
    agent_id: str = ""  # 智能体唯一标识，和数据库agent_id对应
    agent_name: str = ""  # 智能体中文名称

    def get_signal_stock_pool(self, trade_date: str, daily_data: pd.DataFrame, context: Dict) -> List[Dict]:
        """
        核心选股方法：必须由子类实现
        :param trade_date: 选股交易日（已经收盘的完整交易日，T日）
        :param daily_data: T日全市场标准化日线数据（含均线、涨跌停、连板数等）
        :param context: 上下文数据，包含：
            - st_stock_list: T日ST股票列表
            - trade_dates: 历史交易日列表
            - pre_close_data: T-1日收盘数据
            - 其他可扩展的上下文数据
        :return: 信号列表，每个元素必须包含以下字段：
            {
                "ts_code": "000001.SZ",
                "stock_name": "平安银行",
                "buy_price": 10.25  # 策略自定义买入价
            }
        """
        raise NotImplementedError("子类必须实现get_signal_stock_pool方法")