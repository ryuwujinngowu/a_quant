import pandas as pd
import numpy as np

from backtest.account import Account
from backtest.metrics import BacktestMetrics
from config.config import MAX_POSITION_COUNT
from data.data_cleaner import data_cleaner
from data.data_fetcher import data_fetcher
from strategies.multi_limit_up_strategy import MultiLimitUpStrategy
from utils.db_utils import db  # 正确导入数据库工具
from utils.log_utils import logger


class MultiStockBacktestEngine:
    """全市场多标的回测引擎（性能优化版：缓存+预加载+索引优化）"""

    def __init__(
            self,
            strategy: MultiLimitUpStrategy,
            init_capital: float,
            start_date: str,
            end_date: str
    ):
        self.strategy = strategy
        self.init_capital = init_capital
        self.start_date = start_date
        self.end_date = end_date
        # 初始化账户
        self.account = Account(init_capital=init_capital, max_position_count=MAX_POSITION_COUNT)
        # 回测交易日列表
        self.trade_dates = self.init_trade_cal()
        # 回测结果
        self.result = {}

    def init_trade_cal(self) -> list:
        """初始化交易日历：回测前拉取、入库，返回准确的交易日列表"""
        logger.info(f"===== 初始化回测交易日历：{self.start_date} 至 {self.end_date} =====")
        data_cleaner.truncate_trade_cal_table()
        # 1. 拉取回测时间段的完整交易日历
        raw_cal_df = data_fetcher.fetch_trade_cal(
            start_date=self.start_date,
            end_date=self.end_date,
            exchange="SSE"
        )
        if raw_cal_df.empty:
            logger.critical("交易日历获取失败，终止回测")
            raise RuntimeError("交易日历获取失败，请检查接口权限或网络")

        # 2. 清洗入库
        data_cleaner.clean_and_insert_trade_cal(raw_cal_df)

        # 3. 获取交易日列表（仅is_open=1）
        trade_dates = data_cleaner.get_trade_dates(self.start_date, self.end_date)
        if not trade_dates:
            logger.critical("回测时间段内无有效交易日，终止回测")
            raise RuntimeError("回测时间段内无有效交易日")

        logger.info(f"交易日历初始化完成，有效交易日数量：{len(trade_dates)}")
        return trade_dates

    def get_daily_kline_data(self, trade_date: str) -> pd.DataFrame:
        """获取指定日期全市场日线数据（仅交易日执行，杜绝无效请求）"""
        logger.debug(f"开始获取日线数据: {trade_date}")
        trade_date_format = trade_date.replace("-", "")
        # 优先从数据库读取
        sql = """
              SELECT ts_code, trade_date, open, high, low, close, pre_close, volume, amount
              FROM kline_day
              WHERE trade_date = %s \
              """
        df = db.query(sql, params=(trade_date_format,), return_df=True)
        if df is not None and not df.empty:
            logger.debug(f"{trade_date} 日线数据从数据库读取完成，行数：{len(df)}")
            return df
        else:
            logger.error(f"{trade_date} 日线数据拉取失败，跳过当日")
            return pd.DataFrame()


    # def get_pre_close_map(self, trade_date: str) -> dict:
    #     """获取前一日收盘价映射（优化：用交易日历的pretrade_date，100%准确）"""
    #     # 从交易日历获取上一个交易日，无需循环判断
    #     pre_trade_date = data_cleaner.get_pre_trade_date(trade_date)
    #     if not pre_trade_date:
    #         logger.warning(f"{trade_date} 无有效上一个交易日")
    #         return {}
    #
    #     pre_date_format = pre_trade_date.replace("-", "")
    #     sql = """
    #           SELECT ts_code, close as pre_close
    #           FROM kline_day
    #           WHERE trade_date = %s \
    #           """
    #     df = db.query(sql, params=(pre_date_format,), return_df=True)
    #     if df is None or df.empty:
    #         logger.warning(f"{pre_trade_date} 无前收盘价数据")
    #         return {}
    #     return df.set_index("ts_code")["pre_close"].to_dict()

    def run(self) -> dict:
        """执行回测核心流程"""
        logger.info(
            f"===== 开始回测，初始本金：{self.init_capital}元，回测时间段：{self.start_date} 至 {self.end_date} =====")
        # 初始化策略
        self.strategy.initialize()
        # 回测前清空临时表
        data_cleaner.truncate_kline_min_table()

        # 按交易日循环执行（仅交易日，无休市日）
        for idx, trade_date in enumerate(self.trade_dates):
            logger.info(f"===== 处理交易日：{trade_date}（第{idx + 1}/{len(self.trade_dates)}天） =====")
            # 1. 获取当日全市场日线数据
            daily_df = self.get_daily_kline_data(trade_date)
            if daily_df.empty:
                logger.warning(f"{trade_date} 无有效日线数据，跳过当日")
                continue
            # # 2. 获取前一日收盘价映射
            # pre_close_map = self.get_pre_close_map(trade_date)
            # if not pre_close_map:
            #     logger.warning(f"{trade_date} 无前收盘价数据，跳过当日")
            #     continue

            # 3. 执行上一日的开盘卖出信号
            for ts_code, sell_type in list(self.strategy.sell_signal_map.items()):
                if sell_type == "open":
                    stock_df = daily_df[daily_df["ts_code"] == ts_code]
                    if not stock_df.empty:
                        open_price = stock_df["open"].iloc[0]
                        self.account.sell(trade_date=trade_date, ts_code=ts_code, price=open_price)
            self.strategy.sell_signal_map = {}

            # 4. 生成当日买卖信号
            buy_stocks, sell_signal_map = self.strategy.generate_signal(
                trade_date=trade_date,
                daily_df=daily_df,
                positions=self.account.positions
            )
            self.strategy.sell_signal_map = sell_signal_map

            # 5. 执行买入操作（按可用仓位买入）
            available_count = self.account.get_available_position_count()
            if available_count > 0 and buy_stocks:
                logger.info(f"{trade_date} 开始执行买入操作 | 可用仓位：{available_count} | 买入列表：{buy_stocks}")
                for ts_code in buy_stocks[:available_count]:
                    # 1. 筛选当前股票的当日日线数据
                    stock_df = daily_df[daily_df["ts_code"] == ts_code]
                    if stock_df.empty:
                        logger.warning(f"{trade_date} 买入操作：{ts_code} 无当日日线数据，跳过")
                        continue
                    # 2. 从日线数据中取pre_close（上一日收盘价）
                    pre_close = stock_df["pre_close"].iloc[0]
                    # 3. 保留原有空值校验
                    if pre_close <= 0:
                        logger.warning(f"{trade_date} 买入操作：{ts_code} pre_close={pre_close}（无效值），跳过")
                        continue
                    # 4. 计算涨停价（原有逻辑）
                    limit_up_price = self.strategy.calc_limit_up_price(ts_code, pre_close)
                    if limit_up_price <= 0:
                        logger.warning(f"{trade_date} 买入操作：{ts_code} 涨停价计算无效（{limit_up_price}），跳过")
                        continue
                    # ========== 核心补全：执行买入操作（之前漏掉了这行！） ==========
                    self.account.buy(trade_date=trade_date, ts_code=ts_code, price=limit_up_price)
                    logger.info(f"{trade_date} 买入成功 | 股票：{ts_code} | 涨停价：{limit_up_price}元")
            else:
                logger.info(f"{trade_date} 无可用仓位或无买入信号，跳过买入操作")

            # 6. 执行当日收盘卖出信号
            for ts_code, sell_type in sell_signal_map.items():
                if sell_type == "close":
                    stock_df = daily_df[daily_df["ts_code"] == ts_code]
                    if not stock_df.empty:
                        close_price = stock_df["close"].iloc[0]
                        self.account.sell(trade_date=trade_date, ts_code=ts_code, price=close_price)

            # 7. 每日收盘更新账户资产
            self.account.update_daily_asset(trade_date=trade_date, daily_price_df=daily_df)

        # 回测结束，强制清仓剩余持仓
        logger.info("===== 回测结束，强制清仓剩余持仓 =====")
        last_date = self.trade_dates[-1]
        last_daily_df = self.get_daily_kline_data(last_date)
        # ========== 核心修正：将positions.keys()转为列表（静态迭代，避免字典大小变化） ==========
        hold_stocks = list(self.account.positions.keys())  # 先转成列表，固定迭代对象
        for ts_code in hold_stocks:
            # 补充校验：防止迭代过程中该股票已被卖出（避免重复操作）
            if ts_code not in self.account.positions:
                continue
            stock_df = last_daily_df[last_daily_df["ts_code"] == ts_code]
            if stock_df.empty:
                logger.warning(f"{last_date} 强制清仓：{ts_code} 无当日日线数据，跳过清仓")
                continue
            close_price = stock_df["close"].iloc[0]
            self.account.sell(trade_date=last_date, ts_code=ts_code, price=close_price)
            logger.info(f"{last_date} 强制清仓：{ts_code} 卖出成功，价格：{close_price}")
        self.account.update_daily_asset(trade_date=last_date, daily_price_df=last_daily_df)

        # 计算回测指标
        net_value_df = self.account.get_net_value_df()
        trade_df = self.account.get_trade_df()
        metrics = BacktestMetrics(net_value_df=net_value_df, init_capital=self.init_capital, trade_df=trade_df)
        self.result = metrics.calc_all_metrics()

        # 输出结果
        logger.info("=" * 60)
        logger.info("回测完成，核心指标汇总：")
        for k, v in self.result.items():
            logger.info(f"{k}：{v}")
        logger.info("=" * 60)

        # 回测结束清空临时表
        data_cleaner.truncate_kline_min_table()
        data_cleaner.truncate_trade_cal_table()

        # 附加详细数据
        self.result["net_value_df"] = net_value_df
        self.result["trade_df"] = trade_df
        return self.result