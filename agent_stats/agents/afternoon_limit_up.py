"""
午盘打板选手（AfternoonLimitUpAgent）
========================================
策略逻辑
--------
模拟午盘阶段（13:00 后）首次触板的涨停标的买入（无仓位限制）。

命中条件：
  1. 当日进入涨停池
  2. 首次触板时间（first_time）>= 13:00:00（午盘）
  3. 非 ST / *ST 股票
  4. 排除北交所

买入价：涨停价

与早盘打板的区别
----------------
early = "开盘强势、情绪好" 标的  ↔  afternoon = "尾盘资金发动" 标的
两者统计对比能揭示市场情绪的强弱节奏（早盘/尾盘哪个更容易赚钱）。

降级处理（同早盘策略）：
  若 first_time 字段缺失，降级为全量涨停标的（与早盘策略数据相同）。
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from utils.common_tools import get_limit_list_ths, calc_limit_up_price
from utils.log_utils import logger

AFTERNOON_START = "13:00:00"


class AfternoonLimitUpAgent(BaseAgent):
    agent_id   = "afternoon_limit_up"
    agent_name = "午盘打板选手"

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        st_set = set(context.get("st_stock_list", []))

        limit_df = get_limit_list_ths(trade_date, limit_type="涨停池")
        if limit_df is None or limit_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] 涨停池为空，无信号")
            return []

        has_first_time = "first_time" in limit_df.columns
        if has_first_time:
            limit_df["first_time_str"] = (
                limit_df["first_time"]
                .astype(str)
                .str.extract(r"(\d{2}:\d{2}:\d{2})", expand=False)
                .fillna("")
            )
            limit_df = limit_df[
                limit_df["first_time_str"].notna()
                & (limit_df["first_time_str"] >= AFTERNOON_START)
                & (limit_df["first_time_str"] != "")
            ]
        else:
            logger.warning(
                f"[{self.agent_id}][{trade_date}] 缺少 first_time，降级：纳入全部涨停标的"
            )

        if limit_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] 午盘涨停池为空，无信号")
            return []

        # 前收价映射
        pre_close_map: Dict[str, float] = {}
        pre_data = context.get("pre_close_data", pd.DataFrame())
        if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
            pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
        if "pre_close" in daily_data.columns:
            for _, row in daily_data.iterrows():
                if row["ts_code"] not in pre_close_map:
                    pre_close_map[row["ts_code"]] = row["pre_close"]

        result = []
        for _, row in limit_df.iterrows():
            ts_code = row["ts_code"]
            if ts_code in st_set:
                continue
            code_prefix = ts_code.split(".")[0]
            if code_prefix.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
                continue
            pre_close = pre_close_map.get(ts_code, 0.0)
            if pre_close <= 0:
                continue
            buy_price = calc_limit_up_price(ts_code, pre_close)
            if buy_price <= 0:
                continue
            result.append({
                "ts_code":    ts_code,
                "stock_name": str(row.get("name", "")),
                "buy_price":  buy_price,
            })

        logger.info(
            f"[{self.agent_id}][{trade_date}] 午盘涨停命中 {len(result)} 只"
            + (f"（含 first_time 过滤）" if has_first_time else "（已降级）")
        )
        return result
