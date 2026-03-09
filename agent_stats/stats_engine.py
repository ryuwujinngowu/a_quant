"""
核心统计引擎（类似回测引擎）
严格遵循时序逻辑：T+1日凌晨运行，先结账T-1日的隔日表现，再生成T日的选股信号
生产级稳定：异常隔离、幂等性、断点续跑、完整日志
"""
import importlib
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agent_stats.config import (
    START_DATE, AGENT_REGISTRY, MAX_RETRY_TIMES
)
from agent_stats.agent_base import BaseAgent
from agent_stats.db_operator import AgentStatsDBOperator
from utils.log_utils import logger
from utils.common_tools import get_trade_dates, get_daily_kline_data, filter_st_stocks
from data.data_cleaner import data_cleaner


class AgentStatsEngine:
    def __init__(self):
        self.db_operator = AgentStatsDBOperator()
        self.agents: List[BaseAgent] = self._load_agents()
        self.all_trade_dates = get_trade_dates(START_DATE, datetime.now().strftime("%Y-%m-%d"))
        logger.info(f"引擎初始化完成，加载智能体数量：{len(self.agents)}，交易日范围：{START_DATE} ~ 最新")

    def _load_agents(self) -> List[BaseAgent]:
        """动态加载注册的智能体，新增智能体无需修改引擎代码"""
        agents = []
        for agent_path in AGENT_REGISTRY:
            try:
                # 动态导入类
                module_path, class_name = agent_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                agent_instance = agent_class()

                # 校验必填属性
                if not agent_instance.agent_id or not agent_instance.agent_name:
                    logger.error(f"智能体 {agent_path} 未配置agent_id/agent_name，跳过")
                    continue
                agents.append(agent_instance)
                logger.info(f"智能体加载成功：{agent_instance.agent_name}({agent_instance.agent_id})")
            except Exception as e:
                logger.error(f"智能体 {agent_path} 加载失败：{e}", exc_info=True)
                continue
        return agents

    def _get_trade_date_context(self, trade_date: str) -> Dict:
        """构建选股所需的上下文数据，统一收口，避免重复请求"""
        context = {}
        # 1. T日ST股票列表
        context["st_stock_list"] = filter_st_stocks([], trade_date)
        # 2. 历史交易日列表
        context["trade_dates"] = self.all_trade_dates
        # 3. T-1日收盘数据
        try:
            date_idx = self.all_trade_dates.index(trade_date)
            if date_idx > 0:
                pre_date = self.all_trade_dates[date_idx - 1]
                context["pre_close_data"] = get_daily_kline_data(pre_date)
            else:
                context["pre_close_data"] = pd.DataFrame()
        except Exception as e:
            logger.warning(f"[{trade_date}] 上下文数据获取异常：{e}")
            context["pre_close_data"] = pd.DataFrame()
        return context

    def _calc_intraday_stats(self, stock_list: List[Dict], trade_date: str) -> Tuple[float, List[Dict]]:
        """计算T日选股的日内收益统计"""
        if not stock_list:
            return 0.0, []

        stock_detail_list = []
        return_list = []
        ts_code_list = [item["ts_code"] for item in stock_list]
        buy_price_map = {item["ts_code"]: item["buy_price"] for item in stock_list}
        name_map = {item["ts_code"]: item["stock_name"] for item in stock_list}

        # 批量获取T日日线数据
        daily_df = get_daily_kline_data(trade_date, ts_code_list=ts_code_list)
        if daily_df.empty:
            logger.warning(f"[{trade_date}] 日内日线数据为空")
            return 0.0, stock_list

        # 逐只计算日内收益
        for _, row in daily_df.iterrows():
            ts_code = row["ts_code"]
            buy_price = buy_price_map.get(ts_code, 0)
            if buy_price <= 0:
                continue

            # 日内收益率（按买入价计算）
            intraday_return = (row["close"] - buy_price) / buy_price * 100
            return_list.append(intraday_return)

            # 补充明细
            stock_detail_list.append({
                "ts_code": ts_code,
                "stock_name": name_map.get(ts_code, ""),
                "buy_price": buy_price,
                "intraday_close_price": row["close"],
                "intraday_return": round(intraday_return, 4)
            })

        # 平均日内收益
        avg_return = round(np.mean(return_list) if return_list else 0.0, 4)
        return avg_return, stock_detail_list

    def _calc_next_day_stats(self, stock_list: List[Dict], trade_date: str, next_trade_date: str) -> Tuple[Dict, List[Dict]]:
        """计算T日选股的T+1日隔日表现统计"""
        if not stock_list:
            return {}, []

        ts_code_list = [item["ts_code"] for item in stock_list]
        buy_price_map = {item["ts_code"]: item["buy_price"] for item in stock_list}
        t_close_map = {item["ts_code"]: item["intraday_close_price"] for item in stock_list}
        name_map = {item["ts_code"]: item["stock_name"] for item in stock_list}

        # 1. 获取T+1日日线数据
        next_daily_df = get_daily_kline_data(next_trade_date, ts_code_list=ts_code_list)
        if next_daily_df.empty:
            logger.warning(f"[{trade_date}] T+1日日线数据为空")
            return {}, []

        # 2. 获取T+1日分钟线数据（计算红盘/浮盈时间）
        minute_data_map = {}
        for ts_code in ts_code_list:
            try:
                minute_df = data_cleaner.get_kline_min_by_stock_date(ts_code, next_trade_date)
                minute_data_map[ts_code] = minute_df
            except Exception as e:
                logger.warning(f"[{ts_code}][{next_trade_date}] 分钟线获取失败：{e}")
                minute_data_map[ts_code] = pd.DataFrame()

        # 3. 逐只计算隔日表现
        next_stock_detail = []
        open_return_list = []
        close_return_list = []
        max_premium_list = []
        max_drawdown_list = []
        red_minute_list = []
        profit_minute_list = []
        intraday_profit_list = []
        open_direction_list = []

        for _, row in next_daily_df.iterrows():
            ts_code = row["ts_code"]
            buy_price = buy_price_map.get(ts_code, 0)
            t_close_price = t_close_map.get(ts_code, 0)
            if buy_price <= 0:
                continue

            # 基础收益率计算
            open_return = (row["open"] - buy_price) / buy_price * 100
            close_return = (row["close"] - buy_price) / buy_price * 100
            max_premium = (row["high"] - buy_price) / buy_price * 100
            max_drawdown = (buy_price - row["low"]) / buy_price * 100

            # 开盘方向
            if row["open"] > t_close_price:
                open_direction = "high_open"
            elif row["open"] < t_close_price:
                open_direction = "low_open"
            else:
                open_direction = "flat_open"

            # 分钟线指标计算
            minute_df = minute_data_map.get(ts_code, pd.DataFrame())
            red_minute = 0
            profit_minute = 0
            avg_price = row["amount"] / row["volume"] if row["volume"] > 0 else row["close"]
            intraday_avg_profit = (avg_price - buy_price) / buy_price * 100
            red_time_range = []
            profit_time_range = []

            if not minute_df.empty:
                # 红盘时间：股价高于T日收盘价
                red_mask = minute_df["close"] >= t_close_price
                red_minute = red_mask.sum()
                # 浮盈时间：股价高于买入价
                profit_mask = minute_df["close"] >= buy_price
                profit_minute = profit_mask.sum()

            # 汇总数据
            open_return_list.append(open_return)
            close_return_list.append(close_return)
            max_premium_list.append(max_premium)
            max_drawdown_list.append(max_drawdown)
            red_minute_list.append(red_minute)
            profit_minute_list.append(profit_minute)
            intraday_profit_list.append(intraday_avg_profit)
            open_direction_list.append(open_direction)

            # 个股明细
            next_stock_detail.append({
                "ts_code": ts_code,
                "stock_name": name_map.get(ts_code, ""),
                "buy_price": buy_price,
                "next_open_price": row["open"],
                "next_close_price": row["close"],
                "next_day_avg_price": round(avg_price, 4),
                "open_return": round(open_return, 4),
                "close_return": round(close_return, 4),
                "intraday_avg_profit": round(intraday_avg_profit, 4),
                "intraday_max_return": round(max_premium, 4),
                "intraday_max_drawdown": round(max_drawdown, 4),
                "red_minute_total": red_minute,
                "profit_minute_total": profit_minute,
                "open_direction": open_direction
            })

        # 汇总统计
        def safe_mean(arr):
            return round(np.mean(arr) if arr else 0.0, 4)

        stats_result = {
            "next_day_avg_open_premium": safe_mean(open_return_list),
            "next_day_avg_close_return": safe_mean(close_return_list),
            "next_day_avg_red_minute": int(np.mean(red_minute_list) if red_minute_list else 0),
            "next_day_avg_profit_minute": int(np.mean(profit_minute_list) if profit_minute_list else 0),
            "next_day_avg_intraday_profit": safe_mean(intraday_profit_list),
            "next_day_avg_max_premium": safe_mean(max_premium_list),
            "next_day_avg_max_drawdown": safe_mean(max_drawdown_list),
            "next_day_stock_detail": {"stock_list": next_stock_detail}
        }

        return stats_result, next_stock_detail

    def process_close_task(self, latest_trade_date: str):
        """
        结账任务：处理T-1日选股的隔日表现
        latest_trade_date: 最新的完整交易日（T日），需要结账的是T-1日的选股记录
        """
        logger.info(f"===== 开始执行结账任务，最新交易日：{latest_trade_date} =====")
        try:
            # 获取T-1日的交易日
            date_idx = self.all_trade_dates.index(latest_trade_date)
            if date_idx <= 0:
                logger.warning("无历史交易日，跳过结账任务")
                return
            close_trade_date = self.all_trade_dates[date_idx - 1]

            # 获取待结账的记录
            unclosed_records = self.db_operator.get_unclosed_records(close_trade_date)
            if not unclosed_records:
                logger.info(f"[{close_trade_date}] 无待结账记录，跳过")
                return

            logger.info(f"[{close_trade_date}] 待结账记录数：{len(unclosed_records)}")
            # 逐只处理结账
            for record in unclosed_records:
                agent_id = record["agent_id"]
                trade_date = record["trade_date"].strftime("%Y-%m-%d")
                stock_list = record["signal_stock_detail"].get("stock_list", [])

                try:
                    logger.info(f"[{agent_id}][{trade_date}] 开始结账处理，股票数：{len(stock_list)}")
                    # 计算隔日表现
                    stats_result, _ = self._calc_next_day_stats(stock_list, trade_date, latest_trade_date)
                    if not stats_result:
                        logger.warning(f"[{agent_id}][{trade_date}] 隔日统计结果为空，跳过")
                        continue

                    # 更新数据库
                    update_success = self.db_operator.update_next_day_stats(agent_id, trade_date, stats_result)
                    if update_success:
                        logger.info(f"[{agent_id}][{trade_date}] 结账完成")
                    else:
                        logger.error(f"[{agent_id}][{trade_date}] 结账更新失败")

                except Exception as e:
                    logger.error(f"[{agent_id}][{trade_date}] 结账处理失败：{e}", exc_info=True)
                    continue

            logger.info(f"===== 结账任务执行完成 =====")
        except Exception as e:
            logger.error(f"结账任务执行失败：{e}", exc_info=True)

    def process_signal_task(self, trade_date: str):
        """
        选股任务：生成指定交易日的选股信号
        trade_date: 选股交易日（T日，已收盘的完整交易日）
        """
        logger.info(f"===== 开始执行选股任务，选股日期：{trade_date} =====")
        # 前置校验：数据是否已入库
        if not self.db_operator.check_date_data_exists(trade_date):
            logger.error(f"[{trade_date}] 日线数据未入库，跳过选股任务")
            return False

        # 构建上下文数据
        context = self._get_trade_date_context(trade_date)
        # 获取T日全市场日线数据
        daily_data = get_daily_kline_data(trade_date)
        if daily_data.empty:
            logger.error(f"[{trade_date}] 全市场日线数据为空，跳过选股任务")
            return False

        # 逐个智能体生成信号
        success_count = 0
        for agent in self.agents:
            try:
                logger.info(f"[{agent.agent_id}][{trade_date}] 开始生成信号")
                # 调用智能体选股方法
                signal_list = agent.get_signal_stock_pool(trade_date, daily_data, context)
                logger.info(f"[{agent.agent_id}][{trade_date}] 信号生成完成，命中股票数：{len(signal_list)}")

                # 计算日内收益
                avg_intraday_return, stock_detail_list = self._calc_intraday_stats(signal_list, trade_date)

                # 构建入库记录
                record = {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "trade_date": trade_date,
                    "intraday_avg_return": avg_intraday_return,
                    "signal_stock_detail": {"stock_list": stock_detail_list}
                }

                # 入库
                insert_success = self.db_operator.insert_signal_record(record)
                if insert_success:
                    success_count += 1
                    logger.info(f"[{agent.agent_id}][{trade_date}] 选股记录入库成功")
                else:
                    logger.error(f"[{agent.agent_id}][{trade_date}] 选股记录入库失败")

            except Exception as e:
                logger.error(f"[{agent.agent_id}][{trade_date}] 信号生成失败：{e}", exc_info=True)
                continue

        logger.info(f"===== 选股任务执行完成，成功数量：{success_count}/{len(self.agents)} =====")
        return success_count > 0

    def run_full_flow(self):
        """
        完整运行流程（服务器每日凌晨执行的核心入口）
        严格时序：先结账，再选股
        """
        logger.info("===== 智能体统计引擎完整流程启动 =====")
        # 1. 获取最新的完整交易日
        now = datetime.now()
        latest_trade_date = get_trade_dates(
            (now - timedelta(days=10)).strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d")
        )[-1]
        logger.info(f"最新完整交易日：{latest_trade_date}")

        # 2. 先执行结账任务：处理T-1日的隔日表现
        self.process_close_task(latest_trade_date)

        # 3. 再执行选股任务：生成最新交易日的选股信号
        self.process_signal_task(latest_trade_date)

        # 4. 断点续跑：补全历史缺失的选股记录
        for agent in self.agents:
            last_processed_date = self.db_operator.get_last_processed_date(agent.agent_id)
            start_date = last_processed_date if last_processed_date else START_DATE
            # 获取需要补全的交易日
            try:
                start_idx = self.all_trade_dates.index(start_date)
                missing_dates = self.all_trade_dates[start_idx + 1: self.all_trade_dates.index(latest_trade_date)]
            except Exception as e:
                logger.warning(f"[{agent.agent_id}] 补全日期计算失败：{e}")
                continue

            if missing_dates:
                logger.info(f"[{agent.agent_id}] 发现缺失交易日：{len(missing_dates)}个，开始补全")
                for trade_date in missing_dates:
                    self.process_signal_task(trade_date)

        logger.info("===== 智能体统计引擎完整流程执行完成 =====")
        return True