"""
核心统计引擎（类似回测引擎）
严格遵循时序逻辑：T+1日凌晨运行，先结账T-1日的隔日表现，再生成T日的选股信号
生产级稳定：异常隔离、幂等性、断点续跑、完整日志
"""
import importlib
import inspect
import pkgutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agent_stats.config import (
    START_DATE, MAX_RETRY_TIMES
)
from agent_stats.agent_base import BaseAgent
from agent_stats.agent_db_operator import AgentStatsDBOperator
from utils.log_utils import logger
from utils.common_tools import get_trade_dates, get_daily_kline_data, get_st_stock_codes
from data.data_cleaner import data_cleaner


class AgentStatsEngine:
    def __init__(self, start_date: str = None):
        self.start_date = start_date or START_DATE
        self.db_operator = AgentStatsDBOperator()
        self.agents: List[BaseAgent] = self._auto_load_agents()
        self.all_trade_dates = get_trade_dates(self.start_date, datetime.now().strftime("%Y-%m-%d"))
        logger.info(
            f"引擎初始化完成 | 智能体数量：{len(self.agents)} | "
            f"交易日范围：{self.start_date} ~ {self.all_trade_dates[-1] if self.all_trade_dates else '无'}"
        )

    def _auto_load_agents(self) -> List[BaseAgent]:
        """
        自动扫描 agent_stats/agents/ 目录，加载所有继承 BaseAgent 的类。
        新增/删除 agent 只需在 agents/ 目录增减 .py 文件，无需修改任何配置。
        """
        import agent_stats.agents as agents_pkg
        agents = []
        for finder, module_name, _ in pkgutil.iter_modules(agents_pkg.__path__):
            full_module = f"agent_stats.agents.{module_name}"
            try:
                module = importlib.import_module(full_module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseAgent)
                        and obj is not BaseAgent
                        and obj.__module__ == full_module  # 排除从别处 import 的基类
                    ):
                        instance = obj()
                        if not instance.agent_id or not instance.agent_name:
                            logger.error(f"[{full_module}.{name}] 未配置 agent_id/agent_name，跳过")
                            continue
                        agents.append(instance)
                        logger.info(f"智能体加载成功：{instance.agent_name}（{instance.agent_id}）")
            except Exception as e:
                logger.error(f"[{full_module}] 智能体加载失败：{e}", exc_info=True)
        if not agents:
            logger.warning("未发现任何有效智能体，请检查 agent_stats/agents/ 目录")
        return agents

    def _get_trade_date_context(self, trade_date: str) -> Dict:
        """构建选股所需的上下文数据，统一收口，避免重复请求"""
        context = {}
        # 1. T日ST股票代码列表
        context["st_stock_list"] = get_st_stock_codes(trade_date)
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

        daily_df = get_daily_kline_data(trade_date, ts_code_list=ts_code_list)
        if daily_df.empty:
            logger.warning(f"[{trade_date}] 日内日线数据为空")
            return 0.0, stock_list

        for _, row in daily_df.iterrows():
            ts_code = row["ts_code"]
            buy_price = buy_price_map.get(ts_code, 0)
            if buy_price <= 0:
                continue
            intraday_return = (row["close"] - buy_price) / buy_price * 100
            return_list.append(intraday_return)
            stock_detail_list.append({
                "ts_code": ts_code,
                "stock_name": name_map.get(ts_code, ""),
                "buy_price": buy_price,
                "intraday_close_price": row["close"],
                "intraday_return": round(intraday_return, 4)
            })

        avg_return = round(np.mean(return_list) if return_list else 0.0, 4)
        return avg_return, stock_detail_list

    def _calc_next_day_stats(
        self,
        stock_list: List[Dict],
        trade_date: str,
        next_trade_date: str
    ) -> Tuple[Dict, List[Dict]]:
        """计算T日选股的T+1日隔日表现统计（分钟线并发拉取）"""
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

        # 2. 并发拉取T+1日分钟线（I/O密集，线程池加速）
        minute_data_map: Dict[str, pd.DataFrame] = {}

        def _fetch_min(ts_code: str) -> Tuple[str, pd.DataFrame]:
            try:
                return ts_code, data_cleaner.get_kline_min_by_stock_date(ts_code, next_trade_date)
            except Exception as e:
                logger.warning(f"[{ts_code}][{next_trade_date}] 分钟线获取失败：{e}")
                return ts_code, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=min(len(ts_code_list), 8)) as pool:
            for ts_code, df in pool.map(_fetch_min, ts_code_list):
                minute_data_map[ts_code] = df

        # 3. 逐只计算隔日表现
        next_stock_detail = []
        open_return_list, close_return_list = [], []
        max_premium_list, max_drawdown_list = [], []
        red_minute_list, profit_minute_list, intraday_profit_list = [], [], []

        for _, row in next_daily_df.iterrows():
            ts_code = row["ts_code"]
            buy_price = buy_price_map.get(ts_code, 0)
            t_close_price = t_close_map.get(ts_code, 0)
            if buy_price <= 0:
                continue

            open_return   = (row["open"]  - buy_price) / buy_price * 100
            close_return  = (row["close"] - buy_price) / buy_price * 100
            max_premium   = (row["high"]  - buy_price) / buy_price * 100
            max_drawdown  = (buy_price - row["low"])   / buy_price * 100
            avg_price     = row["amount"] / row["volume"] if row["volume"] > 0 else row["close"]
            intraday_avg_profit = (avg_price - buy_price) / buy_price * 100

            if row["open"] > t_close_price:
                open_direction = "high_open"
            elif row["open"] < t_close_price:
                open_direction = "low_open"
            else:
                open_direction = "flat_open"

            minute_df = minute_data_map.get(ts_code, pd.DataFrame())
            red_minute, profit_minute = 0, 0
            if not minute_df.empty:
                red_minute    = int((minute_df["close"] >= t_close_price).sum())
                profit_minute = int((minute_df["close"] >= buy_price).sum())

            open_return_list.append(open_return)
            close_return_list.append(close_return)
            max_premium_list.append(max_premium)
            max_drawdown_list.append(max_drawdown)
            red_minute_list.append(red_minute)
            profit_minute_list.append(profit_minute)
            intraday_profit_list.append(intraday_avg_profit)

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
                "open_direction": open_direction,
            })

        def safe_mean(arr):
            return round(float(np.mean(arr)) if arr else 0.0, 4)

        stats_result = {
            "next_day_avg_open_premium":  safe_mean(open_return_list),
            "next_day_avg_close_return":  safe_mean(close_return_list),
            "next_day_avg_red_minute":    int(np.mean(red_minute_list)    if red_minute_list    else 0),
            "next_day_avg_profit_minute": int(np.mean(profit_minute_list) if profit_minute_list else 0),
            "next_day_avg_intraday_profit": safe_mean(intraday_profit_list),
            "next_day_avg_max_premium":   safe_mean(max_premium_list),
            "next_day_avg_max_drawdown":  safe_mean(max_drawdown_list),
            "next_day_stock_detail":      {"stock_list": next_stock_detail},
        }
        return stats_result, next_stock_detail

    def process_close_task(self, latest_trade_date: str):
        """
        结账任务：处理T-1日选股的隔日表现
        latest_trade_date: 最新的完整交易日（T日），需要结账的是T-1日的选股记录
        """
        logger.info(f"===== 开始执行结账任务，最新交易日：{latest_trade_date} =====")
        try:
            date_idx = self.all_trade_dates.index(latest_trade_date)
            if date_idx <= 0:
                logger.warning("无历史交易日，跳过结账任务")
                return
            close_trade_date = self.all_trade_dates[date_idx - 1]

            unclosed_records = self.db_operator.get_unclosed_records(close_trade_date)
            if not unclosed_records:
                logger.info(f"[{close_trade_date}] 无待结账记录，跳过")
                return

            logger.info(f"[{close_trade_date}] 待结账记录数：{len(unclosed_records)}")
            for record in unclosed_records:
                agent_id   = record["agent_id"]
                trade_date = record["trade_date"].strftime("%Y-%m-%d")
                stock_list = record["signal_stock_detail"].get("stock_list", [])
                try:
                    logger.info(f"[{agent_id}][{trade_date}] 开始结账，股票数：{len(stock_list)}")
                    stats_result, _ = self._calc_next_day_stats(stock_list, trade_date, latest_trade_date)
                    if not stats_result:
                        logger.warning(f"[{agent_id}][{trade_date}] 隔日统计结果为空，跳过")
                        continue
                    if self.db_operator.update_next_day_stats(agent_id, trade_date, stats_result):
                        logger.info(f"[{agent_id}][{trade_date}] 结账完成")
                    else:
                        logger.error(f"[{agent_id}][{trade_date}] 结账更新失败")
                except Exception as e:
                    logger.error(f"[{agent_id}][{trade_date}] 结账处理失败：{e}", exc_info=True)

            logger.info("===== 结账任务执行完成 =====")
        except Exception as e:
            logger.error(f"结账任务执行失败：{e}", exc_info=True)

    def process_signal_task(self, trade_date: str) -> bool:
        """
        选股任务：并发调用所有智能体生成信号，主线程共享数据只拉一次。
        trade_date: 选股交易日（T日，已收盘的完整交易日）
        """
        logger.info(f"===== 开始执行选股任务，选股日期：{trade_date} =====")
        if not self.db_operator.check_date_data_exists(trade_date):
            logger.error(f"[{trade_date}] 日线数据未入库，跳过选股任务")
            return False

        # 共享数据一次性拉取，传只读副本给各 agent，避免重复 IO
        context    = self._get_trade_date_context(trade_date)
        daily_data = get_daily_kline_data(trade_date)
        if daily_data.empty:
            logger.error(f"[{trade_date}] 全市场日线数据为空，跳过选股任务")
            return False

        def _run_agent(agent: BaseAgent):
            """单个 agent 的完整执行流程，线程隔离"""
            try:
                signal_list = agent.get_signal_stock_pool(trade_date, daily_data, context)
                logger.info(f"[{agent.agent_id}][{trade_date}] 信号生成完成，命中：{len(signal_list)} 只")
                avg_intraday_return, stock_detail_list = self._calc_intraday_stats(signal_list, trade_date)
                record = {
                    "agent_id":              agent.agent_id,
                    "agent_name":            agent.agent_name,
                    "trade_date":            trade_date,
                    "intraday_avg_return":   avg_intraday_return,
                    "signal_stock_detail":   {"stock_list": stock_detail_list},
                }
                success = self.db_operator.insert_signal_record(record)
                if success:
                    logger.info(f"[{agent.agent_id}][{trade_date}] 选股记录入库成功")
                else:
                    logger.error(f"[{agent.agent_id}][{trade_date}] 选股记录入库失败")
                return success
            except Exception as e:
                logger.error(f"[{agent.agent_id}][{trade_date}] 信号生成失败：{e}", exc_info=True)
                return False

        # 并发执行所有 agent（I/O 密集，线程池）
        success_count = 0
        with ThreadPoolExecutor(max_workers=min(len(self.agents), 4)) as pool:
            futures = {pool.submit(_run_agent, agent): agent for agent in self.agents}
            for future in as_completed(futures):
                if future.result():
                    success_count += 1

        logger.info(f"===== 选股任务执行完成，成功：{success_count}/{len(self.agents)} =====")
        return success_count > 0

    def run_full_flow(self):
        """
        完整运行流程（服务器每日凌晨执行的核心入口）
        严格时序：先结账，再选股，最后补全历史缺失
        """
        logger.info("===== 智能体统计引擎完整流程启动 =====")
        # 获取最新的完整交易日
        now = datetime.now()
        latest_trade_date = get_trade_dates(
            (now - timedelta(days=10)).strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d")
        )[-1]
        logger.info(f"最新完整交易日：{latest_trade_date}")

        # 先结账，再选股
        self.process_close_task(latest_trade_date)
        self.process_signal_task(latest_trade_date)

        # 断点续跑：补全历史缺失（按 agent 最后处理日期往前补）
        for agent in self.agents:
            last_processed = self.db_operator.get_last_processed_date(agent.agent_id)
            start = last_processed if last_processed else self.start_date
            try:
                start_idx  = self.all_trade_dates.index(start)
                latest_idx = self.all_trade_dates.index(latest_trade_date)
                missing = self.all_trade_dates[start_idx + 1: latest_idx]
            except Exception as e:
                logger.warning(f"[{agent.agent_id}] 补全日期计算失败：{e}")
                continue
            if missing:
                logger.info(f"[{agent.agent_id}] 发现缺失交易日 {len(missing)} 个，开始补全")
                for td in missing:
                    self.process_signal_task(td)

        logger.info("===== 智能体统计引擎完整流程执行完成 =====")
        return True
