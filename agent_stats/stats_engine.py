"""
核心统计引擎
============
职责：按时间序列，对所有智能体跟踪 T 日选股信号 + T+1 日隔日表现，写入 DB。

执行模式
--------
1. 历史补全（首次/断点续跑/新增 agent）
   - 新 agent（DB 无记录）：从 config.START_DATE 开始逐日处理至最新交易日
   - 已有 agent：从 DB 最后一条记录的次日开始续跑
   - 已处理但缺 D+1 数据的记录：发现后自动补全（结账）
   - 手动重跑：调用方传入 reset_agents 字典（见 run_full_flow 参数说明）

2. 日常运行（每日凌晨 3 点 cron 触发）
   - 与历史补全流程相同，只是缺失日期仅有"今日"（最新交易日）一天

每日处理流程（per agent, per date）
-------------------------------------
  Step A  生成 T 日选股信号 → 立即 INSERT（防崩溃丢数）
  Step B  若 T+1 日数据已存在（历史日期）→ 立即计算并 UPDATE 隔日表现
  Step C  若 T+1 为未来 → 跳过，次日运行时 Step B 会补全

异常处理
--------
- 单日单 agent 异常：跳过该 agent 当天，插入 error 占位记录（reserve_str_1 存错误信息）
- 数据加载异常：同日所有 agent 跳过，统一记录
- 断点续跑：重启后从 DB 末尾继续，已处理的日期不重复处理（ON DUPLICATE KEY 幂等）

关键设计
--------
- 每个交易日的全市场日线只加载一次，所有 agent 共享（减少 IO）
- 分钟线并发拉取（ThreadPoolExecutor，I/O 密集）
- D+1 统计写入时，D 日信号记录已存在，失败只影响 D+1 字段，不丢 D 日数据
"""

import inspect
import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agent_stats.config import START_DATE, MAX_RETRY_TIMES
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
        # 加载从 start_date 到今日的所有交易日（升序）
        today = datetime.now().strftime("%Y-%m-%d")
        self.all_trade_dates: List[str] = get_trade_dates(self.start_date, today)
        logger.info(
            f"引擎初始化完成 | 智能体：{len(self.agents)} 个 | "
            f"交易日范围：{self.start_date} ~ {self.all_trade_dates[-1] if self.all_trade_dates else '无'}"
        )

    # ------------------------------------------------------------------ #
    # Agent 自动发现
    # ------------------------------------------------------------------ #

    def _auto_load_agents(self) -> List[BaseAgent]:
        """
        扫描 agent_stats/agents/ 目录，自动实例化所有继承 BaseAgent 的类。
        新增/删除 agent 只需在 agents/ 放置/删除 .py 文件，无需改任何配置。
        """
        import agent_stats.agents as agents_pkg
        agents = []
        for _, module_name, _ in pkgutil.iter_modules(agents_pkg.__path__):
            full_module = f"agent_stats.agents.{module_name}"
            try:
                module = importlib.import_module(full_module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseAgent)
                        and obj is not BaseAgent
                        and obj.__module__ == full_module
                    ):
                        instance = obj()
                        if not instance.agent_id or not instance.agent_name:
                            logger.error(f"[{full_module}.{name}] 缺少 agent_id/agent_name，跳过")
                            continue
                        agents.append(instance)
                        logger.info(f"加载智能体：{instance.agent_name}（{instance.agent_id}）")
            except Exception as e:
                logger.error(f"[{full_module}] 加载失败：{e}", exc_info=True)
        if not agents:
            logger.warning("⚠ 未发现有效智能体，请检查 agent_stats/agents/ 目录")
        return agents

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _get_next_trade_date(self, trade_date: str) -> Optional[str]:
        """返回 trade_date 的下一个交易日，不存在时返回 None"""
        try:
            idx = self.all_trade_dates.index(trade_date)
            if idx + 1 < len(self.all_trade_dates):
                return self.all_trade_dates[idx + 1]
        except ValueError:
            pass
        return None

    def _get_trade_date_context(self, trade_date: str) -> Dict:
        """构建当日选股上下文（ST 列表 / 历史交易日 / T-1 收盘）"""
        context = {
            "st_stock_list": get_st_stock_codes(trade_date),
            "trade_dates":   self.all_trade_dates,
        }
        try:
            idx = self.all_trade_dates.index(trade_date)
            pre_date = self.all_trade_dates[idx - 1] if idx > 0 else None
            context["pre_close_data"] = get_daily_kline_data(pre_date) if pre_date else pd.DataFrame()
        except Exception as e:
            logger.warning(f"[{trade_date}] pre_close_data 获取失败：{e}")
            context["pre_close_data"] = pd.DataFrame()
        return context

    # ------------------------------------------------------------------ #
    # 日内统计
    # ------------------------------------------------------------------ #

    def _calc_intraday_stats(
        self, stock_list: List[Dict], trade_date: str
    ) -> Tuple[float, List[Dict]]:
        """计算 T 日收盘后相对买入价的日内收益"""
        if not stock_list:
            return 0.0, []

        ts_code_list  = [s["ts_code"]   for s in stock_list]
        buy_price_map = {s["ts_code"]: s["buy_price"]    for s in stock_list}
        name_map      = {s["ts_code"]: s["stock_name"]   for s in stock_list}

        daily_df = get_daily_kline_data(trade_date, ts_code_list=ts_code_list)
        if daily_df.empty:
            logger.warning(f"[{trade_date}] 日线数据空，日内收益无法计算")
            return 0.0, stock_list

        detail, returns = [], []
        for _, row in daily_df.iterrows():
            bp = buy_price_map.get(row["ts_code"], 0)
            if bp <= 0:
                continue
            ret = (row["close"] - bp) / bp * 100
            returns.append(ret)
            detail.append({
                "ts_code":              row["ts_code"],
                "stock_name":           name_map.get(row["ts_code"], ""),
                "buy_price":            bp,
                "intraday_close_price": row["close"],
                "intraday_return":      round(ret, 4),
            })
        return round(float(np.mean(returns)) if returns else 0.0, 4), detail

    # ------------------------------------------------------------------ #
    # 隔日统计（T+1）
    # ------------------------------------------------------------------ #

    def _calc_next_day_stats(
        self,
        stock_list:     List[Dict],
        trade_date:     str,
        next_trade_date: str,
    ) -> Tuple[Dict, List[Dict]]:
        """
        计算 T+1 日隔日表现。
        注意：调用前必须确认 next_trade_date 已经是过去的完整交易日，
              否则日线/分钟线数据不存在，会返回 ({}, [])。
        """
        if not stock_list:
            return {}, []

        ts_code_list  = [s["ts_code"]          for s in stock_list]
        buy_price_map = {s["ts_code"]: s["buy_price"]          for s in stock_list}
        t_close_map   = {s["ts_code"]: s.get("intraday_close_price", s["buy_price"]) for s in stock_list}
        name_map      = {s["ts_code"]: s.get("stock_name", "")  for s in stock_list}

        # T+1 日线
        next_df = get_daily_kline_data(next_trade_date, ts_code_list=ts_code_list)
        if next_df.empty:
            logger.warning(f"[{trade_date}→{next_trade_date}] T+1 日线空")
            return {}, []

        # 并发拉取 T+1 分钟线（I/O 密集）
        def _fetch_min(ts_code: str) -> Tuple[str, pd.DataFrame]:
            try:
                return ts_code, data_cleaner.get_kline_min_by_stock_date(ts_code, next_trade_date)
            except Exception as e:
                logger.warning(f"[{ts_code}][{next_trade_date}] 分钟线拉取失败：{e}")
                return ts_code, pd.DataFrame()

        minute_map: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=min(len(ts_code_list), 8)) as pool:
            for code, df in pool.map(_fetch_min, ts_code_list):
                minute_map[code] = df

        # 逐只计算
        detail = []
        open_ret_list, close_ret_list = [], []
        max_prem_list, max_dd_list   = [], []
        red_min_list, profit_min_list, avg_prof_list = [], [], []

        for _, row in next_df.iterrows():
            ts   = row["ts_code"]
            bp   = buy_price_map.get(ts, 0)
            tc   = t_close_map.get(ts, 0)
            if bp <= 0:
                continue

            open_ret  = (row["open"]  - bp) / bp * 100
            close_ret = (row["close"] - bp) / bp * 100
            max_prem  = (row["high"]  - bp) / bp * 100
            max_dd    = (bp - row["low"])   / bp * 100
            avg_price = row["amount"] / row["volume"] if row["volume"] > 0 else row["close"]
            avg_prof  = (avg_price - bp) / bp * 100

            open_dir = (
                "high_open" if row["open"] > tc
                else ("low_open" if row["open"] < tc else "flat_open")
            )

            min_df = minute_map.get(ts, pd.DataFrame())
            red_min, profit_min = 0, 0
            if not min_df.empty:
                red_min    = int((min_df["close"] >= tc).sum())
                profit_min = int((min_df["close"] >= bp).sum())

            open_ret_list.append(open_ret);   close_ret_list.append(close_ret)
            max_prem_list.append(max_prem);   max_dd_list.append(max_dd)
            red_min_list.append(red_min);     profit_min_list.append(profit_min)
            avg_prof_list.append(avg_prof)

            detail.append({
                "ts_code":              ts,
                "stock_name":           name_map.get(ts, ""),
                "buy_price":            bp,
                "next_open_price":      row["open"],
                "next_close_price":     row["close"],
                "next_day_avg_price":   round(avg_price, 4),
                "open_return":          round(open_ret,  4),
                "close_return":         round(close_ret, 4),
                "intraday_avg_profit":  round(avg_prof,  4),
                "intraday_max_return":  round(max_prem,  4),
                "intraday_max_drawdown":round(max_dd,    4),
                "red_minute_total":     red_min,
                "profit_minute_total":  profit_min,
                "open_direction":       open_dir,
            })

        def sm(arr):
            return round(float(np.mean(arr)) if arr else 0.0, 4)

        stats = {
            "next_day_avg_open_premium":   sm(open_ret_list),
            "next_day_avg_close_return":   sm(close_ret_list),
            "next_day_avg_red_minute":     int(np.mean(red_min_list)    if red_min_list    else 0),
            "next_day_avg_profit_minute":  int(np.mean(profit_min_list) if profit_min_list else 0),
            "next_day_avg_intraday_profit":sm(avg_prof_list),
            "next_day_avg_max_premium":    sm(max_prem_list),
            "next_day_avg_max_drawdown":   sm(max_dd_list),
            "next_day_stock_detail":       {"stock_list": detail},
        }
        return stats, detail

    # ------------------------------------------------------------------ #
    # 单 Agent × 单日 处理（原子单元：任何异常只影响该 agent 当天）
    # ------------------------------------------------------------------ #

    def _process_one(
        self,
        agent:       BaseAgent,
        trade_date:  str,
        daily_data:  pd.DataFrame,
        context:     Dict,
        today:       str,
        skip_signal: bool = False,   # True = 信号已存在，只补 D+1
    ) -> bool:
        """
        处理 agent 在 trade_date 的完整任务：
          1. 若 skip_signal=False → 生成信号并 INSERT
          2. 若 T+1 日已是历史日期 → 计算并 UPDATE 隔日表现
        出现异常时插入 error 占位记录，返回 False。
        """
        next_date = self._get_next_trade_date(trade_date)

        # ── Step A: 生成信号 ──────────────────────────────────────────────
        stock_detail: List[Dict] = []
        if not skip_signal:
            try:
                signal_list = agent.get_signal_stock_pool(trade_date, daily_data, context)
                avg_ret, stock_detail = self._calc_intraday_stats(signal_list, trade_date)
                self.db_operator.insert_signal_record({
                    "agent_id":            agent.agent_id,
                    "agent_name":          agent.agent_name,
                    "agent_desc":          agent.agent_desc,
                    "trade_date":          trade_date,
                    "intraday_avg_return": avg_ret,
                    "signal_stock_detail": {"stock_list": stock_detail},
                })
                logger.info(
                    f"[{agent.agent_id}][{trade_date}] 信号入库 | 命中 {len(signal_list)} 只 "
                    f"| 日内均收益 {avg_ret:.2f}%"
                )
            except Exception as e:
                logger.error(f"[{agent.agent_id}][{trade_date}] 信号生成失败：{e}", exc_info=True)
                self.db_operator.insert_error_record(agent.agent_id, agent.agent_name, trade_date, str(e))
                return False
        else:
            # skip_signal=True 说明 D 日信号已存在，从 DB 取 stock_detail 供 D+1 计算
            try:
                stock_detail = self.db_operator.get_signal_detail(agent.agent_id, trade_date)
            except Exception as e:
                logger.warning(f"[{agent.agent_id}][{trade_date}] 读取已有信号失败：{e}")
                return False

        # ── Step B: 隔日表现（仅当 T+1 是已完成的历史交易日）────────────
        if next_date and next_date <= today:
            try:
                stats, _ = self._calc_next_day_stats(stock_detail, trade_date, next_date)
                if stats:
                    self.db_operator.update_next_day_stats(agent.agent_id, trade_date, stats)
                    logger.info(
                        f"[{agent.agent_id}][{trade_date}→{next_date}] "
                        f"隔日表现更新 | 均收益 {stats['next_day_avg_close_return']:.2f}%"
                    )
            except Exception as e:
                logger.warning(f"[{agent.agent_id}][{trade_date}] D+1 统计失败：{e}", exc_info=True)
                # 标记 D+1 计算异常（不影响已入库的信号数据）
                self.db_operator.mark_error(
                    agent.agent_id, trade_date, f"[next_day_err]{str(e)[:200]}"
                )
        return True

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    def run_full_flow(self, reset_agents: Optional[Dict[str, str]] = None) -> bool:
        """
        完整运行入口（cron 每日凌晨调用 / 手动补全均走此方法）。

        参数
        ----
        reset_agents : {agent_id: from_date} 字典，手动指定需要重跑的 agent。
            例：{"morning_limit_up": "2024-10-01"} 表示将 morning_limit_up
            从 2024-10-01 起的所有记录删除并重跑。
            ⚠ 仅通过 run.py --reset-agent / --reset-from 触发，不会自动调用。

        流程说明
        --------
        1. 若 reset_agents 非空，先删除对应记录（不影响其他 agent）。
        2. 查询每个 agent 在 DB 中的最后处理日期：
           - 无记录 → 从 self.start_date 补全（新增 agent 场景）
           - 有记录 → 从 last_date 的次日续跑（断点续跑）
        3. 对所有 agent 需要处理的日期去重后，按日期升序遍历：
           - 每个日期只加载一次全市场日线（降低 IO）
           - 对该日需要处理的 agent 并发执行（降低耗时）
           - 每日每 agent 完成后立即入库（防止崩溃丢数）
        4. 同时处理「信号已存在但缺 D+1 数据」的记录（结账补全）。
        """
        today = self.all_trade_dates[-1] if self.all_trade_dates else datetime.now().strftime("%Y-%m-%d")
        logger.info(f"===== AgentStatsEngine 启动 | 最新交易日：{today} =====")

        # ── 1. 手动重置 ────────────────────────────────────────────────
        if reset_agents:
            for agent_id, from_date in reset_agents.items():
                logger.info(f"[重置] {agent_id} 从 {from_date} 起删除并重跑")
                self.db_operator.delete_records_from(agent_id, from_date)

        # ── 2. 查询各 agent 在 DB 中的状态 ──────────────────────────────
        # {agent_id: last_processed_date or None}
        last_dates = self.db_operator.get_all_agents_last_dates()

        # {agent_id: set of dates that already have next_day stats filled}
        closed_dates = self.db_operator.get_agents_closed_dates()

        # ── 3. 计算每个 agent 需要处理的日期 ────────────────────────────
        # dates_missing[agent_id]   = 需要生成信号的日期列表
        # dates_unclosed[agent_id]  = 信号已存在但 D+1 缺失的日期列表
        dates_missing:  Dict[str, List[str]] = {}
        dates_unclosed: Dict[str, List[str]] = {}

        for agent in self.agents:
            aid = agent.agent_id
            last = last_dates.get(aid)

            if last is None:
                # 新 agent，从头开始
                missing = list(self.all_trade_dates)
            else:
                try:
                    start_idx = self.all_trade_dates.index(last)
                    missing   = self.all_trade_dates[start_idx + 1:]
                except ValueError:
                    missing   = self.all_trade_dates

            dates_missing[aid] = missing

            # 找出信号存在但 D+1 缺失的日期（excluding today, D+1 不存在）
            agent_closed = closed_dates.get(aid, set())
            # all dates up to (not including today) that are recorded but unclosed
            all_recorded = self.db_operator.get_agent_recorded_dates(aid)
            unclosed = [
                d for d in all_recorded
                if d not in agent_closed
                and d < today                                      # today 的 D+1 还没到
                and self._get_next_trade_date(d) is not None       # D+1 存在于交易日历
                and self._get_next_trade_date(d) <= today          # D+1 已是历史日
            ]
            dates_unclosed[aid] = unclosed

        # ── 4. 汇总需要处理的日期集合，升序遍历 ─────────────────────────
        all_dates_to_process = sorted(
            set(d for dl in dates_missing.values()  for d in dl)
            | set(d for dl in dates_unclosed.values() for d in dl)
        )

        if not all_dates_to_process:
            logger.info("所有 agent 均已是最新，无需处理")
            return True

        logger.info(f"待处理交易日数：{len(all_dates_to_process)}，"
                    f"范围 {all_dates_to_process[0]} ~ {all_dates_to_process[-1]}")

        for trade_date in all_dates_to_process:
            # 找出本日需要执行的 agent（missing 或 unclosed）
            agents_need_signal  = [a for a in self.agents if trade_date in dates_missing.get(a.agent_id, [])]
            agents_need_d1_only = [a for a in self.agents if trade_date in dates_unclosed.get(a.agent_id, [])
                                   and a not in agents_need_signal]

            if not agents_need_signal and not agents_need_d1_only:
                continue

            # 每个交易日只加载一次共享数据
            daily_data = get_daily_kline_data(trade_date)
            if daily_data.empty:
                logger.warning(f"[{trade_date}] 全市场日线为空，跳过所有 agent")
                for agent in agents_need_signal:
                    self.db_operator.insert_error_record(
                        agent.agent_id, agent.agent_name, trade_date,
                        f"daily_data_empty: kline_day 无数据"
                    )
                continue

            context = self._get_trade_date_context(trade_date)

            # 并发处理本日所有 agent（异常隔离，互不影响）
            all_agents_today = [
                (a, False) for a in agents_need_signal
            ] + [
                (a, True)  for a in agents_need_d1_only
            ]

            def _run(task):
                agent, skip_signal = task
                return self._process_one(agent, trade_date, daily_data, context, today, skip_signal)

            with ThreadPoolExecutor(max_workers=min(len(all_agents_today), 4)) as pool:
                list(pool.map(_run, all_agents_today))

        logger.info("===== AgentStatsEngine 运行完成 =====")
        return True
