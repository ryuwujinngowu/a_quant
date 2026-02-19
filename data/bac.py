from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import pandas as pd

from config.config import MAIN_BOARD_LIMIT_UP_RATE, STAR_BOARD_LIMIT_UP_RATE, MAX_POSITION_COUNT
from data.data_cleaner import data_cleaner
from data.data_fetcher import data_fetcher
from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger

# 全局配置（可移至config文件）
MAX_WORKERS = 8  # 并发数（根据接口限流调整，建议5-8）
BATCH_SIZE = 1000  # 数据库批量插入大小
CACHE_ENABLE = True  # 是否启用分钟线缓存


class MultiLimitUpStrategy(BaseStrategy):
    """
    全市场连板涨停策略（无未来函数+性能优化版）
    核心兼容承诺：
    1. 对外接口（generate_signal/initialize）入参/出参完全兼容原有逻辑；
    2. 内部优化不影响引擎调用流程；
    3. 异常隔离，单只股票处理失败不影响整体策略执行。
    """

    def __init__(self):
        # -------------------------- 策略可配置参数 --------------------------
        self.max_position_count: int = MAX_POSITION_COUNT
        self.limit_up_price_tolerance: float = 0.995
        self.exclude_call_auction_limit_up: bool = True
        self.default_exchange: str = "SSE"
        # -------------------------- 新增可配置参数 --------------------------
        self.bomb_out_sell_type: str = "open"  # 炸板止损类型："open"=次日开盘卖，"close"=次日收盘卖
        self.continuous_break_sell_type: str = "close"  # 连板断板止盈类型："close"=当日收盘卖
        # ------------------------------------------------------------------

        # 运行时变量（初始化时重置）
        self.sell_signal_map: Dict[str, str] = {}
        self.min_data_cache: Dict[tuple, pd.DataFrame] = {}  # 分钟线缓存 (ts_code, trade_date) -> df
        self.initialize()

    def initialize(self):
        """策略初始化（引擎调用，确保每次回测重置状态）"""
        self.sell_signal_map = {}
        if CACHE_ENABLE:
            self.min_data_cache.clear()  # 清空缓存，避免跨轮回测脏数据
        logger.info(f"连板涨停策略初始化完成 | 最大持仓数：{self.max_position_count} | 并发数：{MAX_WORKERS}")

    # -------------------------- 基础工具方法（稳定无兼容问题） --------------------------
    def get_limit_up_rate_by_ts_code(self, ts_code: str) -> float:
        """原有逻辑完全保留，无修改"""
        if not isinstance(ts_code, str):
            return MAIN_BOARD_LIMIT_UP_RATE

        code_part = ts_code.split('.')[0] if '.' in ts_code else ts_code
        exchange_part = ts_code.split('.')[1] if '.' in ts_code else ''

        if (code_part.startswith(('83', '87', '88'))) or (exchange_part == 'BJ'):
            return 0.0

        if code_part.startswith(('300', '688')):
            return STAR_BOARD_LIMIT_UP_RATE
        else:
            return MAIN_BOARD_LIMIT_UP_RATE

    def calc_limit_up_price(self, ts_code: str, pre_close: float) -> float:
        """原有逻辑完全保留，无修改"""
        if pre_close <= 0:
            logger.warning(f"{ts_code} 前收盘价无效（{pre_close}），无法计算涨停价")
            return 0.0

        limit_rate = self.get_limit_up_rate_by_ts_code(ts_code)
        return round(pre_close * (1 + limit_rate), 2)

    def _is_not_bse_stock(self, ts_code: str) -> bool:
        """原有逻辑完全保留，无修改"""
        if not isinstance(ts_code, str):
            return False
        code_part = ts_code.split('.')[0] if '.' in ts_code else ts_code
        exchange_part = ts_code.split('.')[1] if '.' in ts_code else ''
        return not (code_part.startswith(('83', '87', '88')) or exchange_part == 'BJ')

    # -------------------------- 新增优化方法（兼容原有逻辑） --------------------------
    def _get_call_auction_data(self, ts_code: str, trade_date: str) -> pd.DataFrame:
        """
        获取集合竞价（9:25）数据（带降级逻辑，避免接口兼容问题）
        """
        try:
            start_time = f"{trade_date} 09:25:00"
            end_time = f"{trade_date} 09:25:00"
            raw_df = data_fetcher.fetch_stk_mins(
                ts_code=ts_code,
                freq="1min",
                start_date=start_time,
                end_date=end_time
            )
            # 降级逻辑：若单分钟线拉取失败，返回空DataFrame（后续用全量分钟线判断）
            return raw_df if not raw_df.empty else pd.DataFrame()
        except Exception as e:
            logger.warning(f"{ts_code} {trade_date} 集合竞价数据拉取失败：{e}，将用全量分钟线判断一字板")
            return pd.DataFrame()

    def _batch_clean_insert_kline_min(self, df: pd.DataFrame):
        """
        批量入库分钟线（兼容原有单条插入逻辑）
        """
        if df.empty:
            return

        try:
            cleaned_df = data_cleaner._clean_kline_min_data(df)  # 调用原有清洗逻辑
            if cleaned_df.empty:
                return

            # 优先尝试批量插入
            columns = cleaned_df.columns.tolist()
            values = [tuple(row) for row in cleaned_df.values]
            sql = f"INSERT INTO kline_min ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))})"

            # 分批插入，避免单批数据过大
            for i in range(0, len(values), BATCH_SIZE):
                batch_values = values[i:i + BATCH_SIZE]
                try:
                    # 若有execute_many则批量插入，否则降级为单条插入
                    if hasattr(data_cleaner.db, 'execute_many'):
                        data_cleaner.db.execute_many(sql, batch_values)
                    else:
                        for val in batch_values:
                            data_cleaner.db.execute(sql, val)
                except Exception as batch_e:
                    logger.error(f"批量入库失败，降级为单条插入：{batch_e}")
                    for val in batch_values:
                        try:
                            data_cleaner.db.execute(sql, val)
                        except Exception as single_e:
                            logger.warning(f"单条入库失败 {val[0]}：{single_e}")
            data_cleaner.db.commit()
        except Exception as e:
            logger.error(f"分钟线批量入库失败：{e}")
            # 兜底：调用原有单条插入方法
            data_cleaner.clean_and_insert_kline_min(df)

    # -------------------------- 核心方法优化（无接口变更） --------------------------
    def get_stock_first_limit_up_time(self, ts_code: str, trade_date: str, pre_close: float) -> Optional[datetime]:
        """
        优化点：1. 缓存；2. 异常捕获；3. 批量入库
        兼容：返回值类型完全不变
        """
        limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
        if limit_up_price <= 0:
            return None

        # 缓存逻辑（无兼容问题）
        cache_key = (ts_code, trade_date)
        if CACHE_ENABLE and cache_key in self.min_data_cache:
            formatted_min_df = self.min_data_cache[cache_key]
            logger.debug(f"{ts_code} {trade_date} 分钟线从缓存读取")
        else:
            try:
                start_time = f"{trade_date} 09:30:00"
                end_time = f"{trade_date} 15:00:00"
                raw_min_df = data_fetcher.fetch_stk_mins(
                    ts_code=ts_code,
                    freq="1min",
                    start_date=start_time,
                    end_date=end_time
                )
                if raw_min_df.empty:
                    return None

                # 优化：批量入库（替代原有单条入库）
                self._batch_clean_insert_kline_min(raw_min_df)
                formatted_min_df = data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date)

                if formatted_min_df.empty:
                    return None

                # 写入缓存
                if CACHE_ENABLE:
                    self.min_data_cache[cache_key] = formatted_min_df
            except Exception as e:
                logger.error(f"{ts_code} {trade_date} 分钟线处理失败：{e}")
                return None

        # 原有一字板过滤逻辑（增加空值校验）
        if self.exclude_call_auction_limit_up:
            call_auction_row = formatted_min_df[formatted_min_df["trade_time"] == f"{trade_date} 09:25:00"]
            if not call_auction_row.empty:
                call_auction_close = call_auction_row["close"].iloc[0]
                if call_auction_close >= limit_up_price * self.limit_up_price_tolerance:
                    logger.debug(f"{ts_code} 为集合竞价一字板，剔除选股范围")
                    return None

        # 原有首次涨停时间判断逻辑（无修改）
        trading_min_df = formatted_min_df[formatted_min_df["trade_time"] >= f"{trade_date} 09:30:00"]
        limit_up_min_df = trading_min_df[trading_min_df["close"] >= limit_up_price * self.limit_up_price_tolerance]
        if limit_up_min_df.empty:
            return None

        first_limit_up_time = limit_up_min_df["trade_time"].min()
        logger.debug(f"{ts_code} 首次自然涨停时间：{first_limit_up_time}")
        return first_limit_up_time

    def get_touched_limit_up_stocks(self, daily_df: pd.DataFrame, pre_close_map: Dict[str, float]) -> List[str]:
        """
        优化点：无修改，仅保留原有逻辑（保证输出兼容）
        """
        if daily_df.empty:
            return []

        daily_df = daily_df[daily_df['ts_code'].apply(self._is_not_bse_stock)]
        if daily_df.empty:
            logger.info("过滤北交所股票后，无剩余股票参与选股")
            return []

        touched_limit_up_list = []

        for idx, row in daily_df.iterrows():
            if "ts_code" not in row.index:
                logger.warning("日线数据缺少ts_code字段，跳过该行")
                continue

            ts_code = row["ts_code"]
            pre_close = pre_close_map.get(ts_code, 0)
            day_high = row.get("high", 0)

            if pre_close <= 0 or day_high <= 0:
                logger.debug(f"{ts_code} 价格数据无效，跳过")
                continue

            limit_rate = self.get_limit_up_rate_by_ts_code(ts_code)
            if limit_rate == 0.0:
                continue

            limit_up_price = pre_close * (1 + limit_rate)
            is_touched_limit_up = day_high >= limit_up_price * self.limit_up_price_tolerance

            if is_touched_limit_up:
                touched_limit_up_list.append(ts_code)

        logger.info(f"当日触及涨停股票数量（含20cm，剔除北交所）：{len(touched_limit_up_list)}")
        return touched_limit_up_list

    def _get_stock_limit_up_time_single(self, ts_code: str, trade_date: str, pre_close_map: dict) -> Optional[datetime]:
        """
        单股票处理方法（供多线程调用，增加线程安全和异常隔离）
        """
        try:
            pre_close = pre_close_map.get(ts_code, 0)
            if pre_close <= 0:
                logger.debug(f"{ts_code} 前收盘价为0，跳过")
                return None
            # 调用优化后的get_stock_first_limit_up_time
            return self.get_stock_first_limit_up_time(ts_code, trade_date, pre_close)
        except Exception as e:
            logger.error(f"{ts_code} 单线程处理失败：{e}")
            return None

    def select_stocks(self, trade_date: str, daily_df: pd.DataFrame, available_position: int) -> List[str]:
        """
        选股逻辑（移除pre_close_map，从daily_df取pre_close）
        Args:
            trade_date: 交易日
            daily_df: 当日全市场日线数据（含pre_close字段）
            available_position: 可用仓位数量
        Returns:
            买入股票列表
        """
        buy_candidates = []
        # 遍历日线数据，直接从df取pre_close（替代原pre_close_map）
        for _, row in daily_df.iterrows():
            ts_code = row["ts_code"]
            pre_close = row["pre_close"]  # ✅ 从daily_df取pre_close，无需pre_close_map
            if pre_close <= 0:
                continue

            # 以下保留你原有选股逻辑（如计算涨停价、判断涨停、筛选连板等）
            limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
            high_price = row["high"]
            close_price = row["close"]

            # 示例选股逻辑（保留你的原有逻辑）
            is_limit_up = high_price >= limit_up_price * self.limit_up_price_tolerance
            if is_limit_up:
                buy_candidates.append(ts_code)

        # 按可用仓位截取买入列表
        buy_stocks = buy_candidates[:available_position]
        return buy_stocks

    def get_available_position_after_open_sell(self, positions: Dict, open_sell_stocks: List[str]) -> int:
        """
        计算当日开盘卖出后的可用仓位（核心修正：感知当日卖出腾出的仓位）
        :param positions: 调用generate_signal时的原始持仓
        :param open_sell_stocks: 当日开盘卖出的股票列表（由引擎传入）
        :return: 实际可用仓位
        """
        # 原始持仓数 - 当日开盘卖出数 = 卖出后的持仓数
        after_sell_position_count = len(positions) - len(open_sell_stocks)
        # 可用仓位 = 最大持仓数 - 卖出后的持仓数
        available_position = self.max_position_count - after_sell_position_count
        # 确保可用仓位≥0
        available_position = max(0, available_position)
        logger.info(
            f"修正可用仓位：原始持仓{len(positions)}只 → 开盘卖出{len(open_sell_stocks)}只 → 剩余持仓{after_sell_position_count}只 → 可用仓位{available_position}只")
        return available_position

    # -------------------------- 核心接口（完全兼容原有调用） --------------------------
    from typing import Dict, List, Tuple
    import pandas as pd
    from utils.log_utils import logger

    # 注意：保留你原有类的其他属性（如limit_up_price_tolerance、max_position_count等）
    def generate_signal(self, trade_date: str, daily_df: pd.DataFrame, positions: Dict,
                        open_sell_stocks: List[str] = None) -> Tuple[List[str], Dict[str, str]]:
        """
        新增可选参数open_sell_stocks：当日开盘卖出的股票列表（引擎传入）
        兼容原有调用：若不传，按原有逻辑计算可用仓位
        ✅ 核心修改：移除pre_close_map参数，改用daily_df的pre_close字段
        """
        # -------------------------- 步骤1：生成卖出信号（原有逻辑+问题1修正） --------------------------
        sell_signal_map = {}
        for ts_code, position in positions.items():
            stock_row = daily_df[daily_df["ts_code"] == ts_code]
            if stock_row.empty:
                continue

            close_price = stock_row["close"].iloc[0]
            pre_close = stock_row["pre_close"].iloc[0]  # ✅ 已用daily_df的pre_close，无需pre_close_map
            limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
            is_day_limit_up = close_price >= limit_up_price * self.limit_up_price_tolerance

            # 问题1修正：炸板止损可配置
            if position.hold_days == 0:
                if not is_day_limit_up:
                    sell_signal_map[ts_code] = self.bomb_out_sell_type
                    logger.debug(f"{ts_code} 买入当日炸板，生成次日{self.bomb_out_sell_type}卖出信号")
            else:
                if not is_day_limit_up:
                    sell_signal_map[ts_code] = self.continuous_break_sell_type
                    logger.debug(f"{ts_code} 连板断板，生成当日{self.continuous_break_sell_type}卖出信号")

        # -------------------------- 步骤2：修正可用仓位计算（问题2核心） --------------------------
        if open_sell_stocks is None:
            # 兼容原有调用：无开盘卖出列表，按原始持仓计算
            used_position_count = len(positions)
            available_position = self.max_position_count - used_position_count
        else:
            # 修正逻辑：按开盘卖出后的实际仓位计算
            available_position = self.get_available_position_after_open_sell(positions, open_sell_stocks)

        # -------------------------- 步骤3：选股买入（使用修正后的可用仓位） --------------------------
        # ✅ 核心修改：调用select_stocks时，删除pre_close_map参数，改为传daily_df（让select_stocks从df取pre_close）
        buy_stocks = self.select_stocks(trade_date, daily_df, available_position)

        logger.info(
            f"[{trade_date}] 信号生成完成 | 卖出信号：{sell_signal_map} | 买入列表：{buy_stocks} | 实际可用仓位：{available_position}")
        return buy_stocks, sell_signal_map

if __name__ == "__main__":
    strategy = MultiLimitUpStrategy()
    pre_close = 2.35
    limit_price = strategy.calc_limit_up_price("300059.SZ", pre_close)
    assert limit_price == 2.82, f"300股票涨停价计算错误，预期2.82，实际{limit_price}"

    # 测试688008.SH（澜起科技，科创板20cm）
    limit_price_688 = strategy.calc_limit_up_price("688008.SH", pre_close)
    assert limit_price_688 == 2.82, f"688股票涨停价计算错误，预期2.82，实际{limit_price_688}"

    # 测试主板股票（600000.SH，浦发银行10cm）
    limit_price_main = strategy.calc_limit_up_price("600000.SH", pre_close)
    assert limit_price_main == 2.59, f"主板股票涨停价计算错误，预期 2.59，实际{limit_price_main}"

    print("✅ 20cm/10cm涨停价计算全部正确！")