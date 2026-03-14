"""
MA5 低吸选手
=============
策略逻辑：股价从上方回踩 5 日均线附近（±0.5%）时视为低吸机会，买入成本取 MA5 价格。

TODO: 完整选股逻辑待后续迭代填充，当前为骨架占位。
"""
from typing import List, Dict
import pandas as pd
from agent_stats.agent_base import BaseAgent


class MA5LowBuyAgent(BaseAgent):
    agent_id   = "ma5_low_buy"
    agent_name = "MA5低吸选手"

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        """
        TODO: 完整逻辑待后续迭代，当前骨架返回空列表，不影响引擎运行。
        规划：
          1. 过滤 ST（context["st_stock_list"]）/ 跌停 / 停牌
          2. 查近 5 日收盘价计算 MA5
          3. 当日收盘价在 MA5 ± 0.5% 范围内 → 命中
          4. buy_price = MA5
        """
        return []
