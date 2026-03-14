"""
跌停板战法（LimitDownBuyAgent）
=================================
策略逻辑
--------
模拟当日所有非 ST、非北交所 首次跌停股票的买入，买入价为跌停价。

"跌停板战法" 核心逻辑
---------------------
短线情绪激烈时，当日首次跌停的股票往往大量套牢盘被强制卖出，
形成短期超卖→次日出现反弹的概率较高。
本策略跟踪所有跌停标的的次日表现，用统计数据验证这一规律在不同市场环境下的有效性。

命中条件：
  1. 当日进入跌停池（limit_list_ths.limit_type = '跌停池'）
  2. 非 ST / *ST 股票（ST 本身就是强制卖出，情绪特征不同）
  3. 排除北交所（流动性差）

买入价：跌停价（由前收盘价根据板块规则计算）

注意事项
--------
- 跌停板战法通常适用于情绪高涨期的个股性下跌，不适合系统性下跌（大盘连跌）。
- 当前策略不区分主动性跌停（利空爆雷）和情绪性跌停，后续可通过增加过滤条件细化。
- open_times = 0 表示当日未开板过，但此处不做此过滤（全量跟踪，用数据说话）。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from utils.common_tools import get_limit_list_ths, calc_limit_down_price
from utils.log_utils import logger


class LimitDownBuyAgent(BaseAgent):
    agent_id   = "limit_down_buy"
    agent_name = "跌停板战法选手"

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        """
        返回当日跌停池全量标的（非ST/非北交所），买入价为跌停价。
        """
        st_set = set(context.get("st_stock_list", []))

        # ── 获取当日跌停池 ───────────────────────────────────────────────
        limit_df = get_limit_list_ths(trade_date, limit_type="跌停池")
        if limit_df is None or limit_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] 跌停池为空，无信号")
            return []

        # ── 构建前收价映射 ────────────────────────────────────────────────
        pre_close_map: Dict[str, float] = {}
        pre_data = context.get("pre_close_data", pd.DataFrame())
        if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
            pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
        if "pre_close" in daily_data.columns:
            for _, row in daily_data.iterrows():
                if row["ts_code"] not in pre_close_map:
                    pre_close_map[row["ts_code"]] = row["pre_close"]

        # ── 组装结果 ─────────────────────────────────────────────────────
        result = []
        for _, row in limit_df.iterrows():
            ts_code = row["ts_code"]

            # 过滤 ST
            if ts_code in st_set:
                continue
            # 过滤北交所
            code_prefix = ts_code.split(".")[0]
            if code_prefix.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
                continue

            pre_close = pre_close_map.get(ts_code, 0.0)
            if pre_close <= 0:
                continue

            buy_price = calc_limit_down_price(ts_code, pre_close)
            if buy_price <= 0:
                continue

            result.append({
                "ts_code":    ts_code,
                "stock_name": str(row.get("name", "")),
                "buy_price":  buy_price,
            })

        logger.info(f"[{self.agent_id}][{trade_date}] 跌停板命中 {len(result)} 只")
        return result
