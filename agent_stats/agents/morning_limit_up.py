"""
早盘打板选手（MorningLimitUpAgent）
=====================================
策略逻辑
--------
模拟早盘阶段对所有涨停板的买入行为（无仓位限制，平铺全部命中标的）。

命中条件：
  1. 当日进入涨停池（limit_list_ths.limit_type = '涨停池'）
  2. 首次触板时间（first_time）< 11:30:00（早盘截止时间，含集合竞价一字板）
  3. 非 ST / *ST 股票
  4. 排除北交所（流动性差，不具代表性）

买入价：涨停价（由前收盘价根据板块规则计算）

关于 first_time 字段
---------------------
数据来自 limit_list_ths（底层接口 limit_list_d），Tushare 提供 first_time 字段。
若字段不存在（历史数据批次差异），降级为"当日收盘为涨停价的股票全量纳入"
（即不区分早盘/午盘），由 reserve_str_1 可观测到降级情况。

参考策略：strategies/multi_limit_up_strategy.py（打板逻辑）
"""
from typing import List, Dict

import pandas as pd

from agent_stats.agent_base import BaseAgent
from utils.common_tools import get_limit_list_ths, calc_limit_up_price
from utils.log_utils import logger

# 早盘截止时间（字符串比较，HH:MM:SS 格式对齐）
MORNING_CUTOFF = "11:30:00"


class MorningLimitUpAgent(BaseAgent):
    agent_id   = "morning_limit_up"
    agent_name = "早盘打板选手"

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        """
        返回当日早盘（first_time < 11:30）涨停标的，买入价为涨停价。
        :param trade_date: T日（已收盘的完整交易日，YYYY-MM-DD）
        :param daily_data: T日全市场日线（含 ts_code / pre_close / close 等字段）
        :param context:    {st_stock_list, pre_close_data, trade_dates}
        :return: [{ts_code, stock_name, buy_price}, ...]
        """
        st_set = set(context.get("st_stock_list", []))

        # ── 获取当日涨停池 ───────────────────────────────────────────────
        limit_df = get_limit_list_ths(trade_date, limit_type="涨停池")
        if limit_df is None or limit_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] 涨停池为空，无信号")
            return []

        # ── 早盘过滤（first_time 字段）───────────────────────────────────
        has_first_time = "first_time" in limit_df.columns
        if has_first_time:
            # first_time 可能是 "09:25:00" 等字符串或 timedelta，统一转字符串
            limit_df["first_time_str"] = (
                limit_df["first_time"]
                .astype(str)
                .str.extract(r"(\d{2}:\d{2}:\d{2})", expand=False)
                .fillna("")
            )
            limit_df = limit_df[
                limit_df["first_time_str"].notna()
                & (limit_df["first_time_str"] < MORNING_CUTOFF)
                & (limit_df["first_time_str"] != "")
            ]
        else:
            logger.warning(
                f"[{self.agent_id}][{trade_date}] limit_list_ths 缺少 first_time 字段，"
                f"降级：纳入全部涨停标的（无法区分早盘/午盘）"
            )

        if limit_df.empty:
            logger.info(f"[{self.agent_id}][{trade_date}] 早盘涨停池为空，无信号")
            return []

        # ── 构建前收价映射（用于计算涨停价）────────────────────────────
        pre_close_map: Dict[str, float] = {}
        pre_data = context.get("pre_close_data", pd.DataFrame())
        if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
            pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
        # fallback：直接从当日日线取 pre_close（如有）
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
            # 过滤北交所（代码以 83/87/88 开头 或交易所为 BJ）
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
            f"[{self.agent_id}][{trade_date}] 早盘涨停命中 {len(result)} 只"
            + (f"（含 first_time 过滤）" if has_first_time else "（已降级，无 first_time）")
        )
        return result
