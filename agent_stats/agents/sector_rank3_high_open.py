"""
昨日第三强板块今日高开买入（SectorRank3HighOpenAgent）
=========================================================
策略逻辑
--------
取前一交易日（T-1）连板热度排名第三的板块，
当日（T 日）该板块内所有高开股票（open > pre_close）以开盘价买入。

与 rank=1/rank=2 的对比意义
-----------------------------
三个板块排名梯度同步跟踪，帮助量化"板块热度排名"与次日收益之间的衰减规律：
  rank=1（主线）→ rank=2（副线）→ rank=3（跟风线）
数据积累后可分析：随板块热度排名下降，次日平均收益是否也系统性下降。

命中条件：
  1. T-1 日 limit_cpt_list 中 rank = 3 的板块
  2. T 日该板块所有高开（open > pre_close）、非ST/非北交所股票
  3. 买入价 = 当日开盘价
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from agent_stats.agents._sector_high_open_helper import get_sector_high_open_signals


class SectorRank3HighOpenAgent(BaseAgent):
    agent_id   = "sector_rank3_high_open"
    agent_name = "昨日第三板块高开选手"
    agent_desc = (
        "昨日第三强板块今日高开买入策略：取T-1日连板热度排名第3的板块，"
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
            sector_rank=3,
            trade_date=trade_date,
            daily_data=daily_data,
            context=context,
        )
