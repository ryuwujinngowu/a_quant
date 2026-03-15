"""
昨日次强板块今日高开买入（SectorRank2HighOpenAgent）
=======================================================
策略逻辑
--------
取前一交易日（T-1）连板热度排名第二的板块，
当日（T 日）该板块内所有高开股票（open > pre_close）以开盘价买入。

与 sector_top_high_open（rank=1）的区别
---------------------------------------
rank=1 是绝对主线，往往已被市场充分定价，次日溢价空间可能受限；
rank=2 的板块往往是"副线"或"接力板块"，资金轮动时可能有更高的相对收益。
本策略与 rank=1/rank=3 同步跟踪，通过数据对比验证板块排名与次日收益的关系。

命中条件：
  1. T-1 日 limit_cpt_list 中 rank = 2 的板块
  2. T 日该板块所有高开（open > pre_close）、非ST/非北交所股票
  3. 买入价 = 当日开盘价
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from utils.common_tools import get_limit_cpt_list, get_stocks_in_sector
from utils.log_utils import logger

from agent_stats.agents._sector_high_open_helper import get_sector_high_open_signals


class SectorRank2HighOpenAgent(BaseAgent):
    agent_id   = "sector_rank2_high_open"
    agent_name = "昨日次强板块高开选手"
    agent_desc = (
        "昨日次强板块今日高开买入策略：取T-1日连板热度排名第2的板块，"
        "筛选该板块内T日高开（open > pre_close）的非ST/非北交所股票，以开盘价买入。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        return get_sector_high_open_signals(
            agent_id=self.agent_id,
            sector_rank=2,
            trade_date=trade_date,
            daily_data=daily_data,
            context=context,
        )
