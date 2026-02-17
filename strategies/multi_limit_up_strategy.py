import pandas as pd
from datetime import datetime
from strategies.base_strategy import BaseStrategy
from config.config import DAILY_LIMIT_UP_RATE
from config.config import MAIN_BOARD_LIMIT_UP_RATE, STAR_BOARD_LIMIT_UP_RATE
from utils.log_utils import logger
from data.data_fetcher import data_fetcher
from data.data_cleaner import data_cleaner


class MultiLimitUpStrategy(BaseStrategy):
    """
    全市场分仓涨停策略
    核心规则：
    1. 选股：每日全市场涨停股票，剔除集合竞价一字板，按首次涨停时间从早到晚排序，选前5
    2. 买入：每只股票占用1/5仓位，涨停价买入
    3. 卖出：
       - 规则1：买入当日收盘未涨停，次日开盘价卖出
       - 规则2：买入次日收盘未涨停，当日收盘价卖出
    """

    def __init__(self):
        self.sell_signal_map = {}  # 卖出信号映射：key=ts_code，value=open/close
        self.initialize()

    def initialize(self):
        """策略初始化，每次回测前自动调用"""
        self.sell_signal_map = {}
        logger.info("全市场分仓涨停策略初始化完成")

    def get_limit_up_rate_by_ts_code(self, ts_code: str) -> float:
        """
        根据股票代码自动判断涨停阈值（新增：识别北交所股票）
        - 主板(60xxxx/00xxxx)：10%
        - 创业板(300xxxx)、科创板(688xxxx)：20%
        - 北交所(83/87/88开头)：返回0（标记为非涨停判断范围）
        """
        if not isinstance(ts_code, str):
            return MAIN_BOARD_LIMIT_UP_RATE

        # 提取纯数字代码 + 交易所后缀
        code_part = ts_code.split('.')[0] if '.' in ts_code else ts_code
        exchange_part = ts_code.split('.')[1] if '.' in ts_code else ''

        # 第一步：剔除北交所股票（83/87/88开头 + .BJ后缀）
        if (code_part.startswith(('83', '87', '88'))) or (exchange_part == 'BJ'):
            return 0.0  # 返回0，后续涨停判断会自动剔除

        # 第二步：判断创业板/科创板（20cm）
        if code_part.startswith(('300', '688')):
            return STAR_BOARD_LIMIT_UP_RATE
        # 第三步：主板（10cm）
        else:
            return MAIN_BOARD_LIMIT_UP_RATE

    def calc_limit_up_price(self, ts_code: str, pre_close: float) -> float:
        """
        【支持20cm涨停】按板块计算涨停价
        :param ts_code: 股票代码（如600000.SH/300001.SZ）
        :param pre_close: 前收盘价
        :return: 涨停价（保留2位小数）
        """
        # 先校验参数有效性
        if pre_close <= 0:
            logger.warning(f"{ts_code} 前收盘价无效（{pre_close}），无法计算涨停价")
            return 0.0

        # 此时ts_code是函数参数，已定义，无红线
        limit_rate = self.get_limit_up_rate_by_ts_code(ts_code)
        return round(pre_close * (1 + limit_rate), 2)

    def get_daily_limit_up_stocks(self, daily_df: pd.DataFrame) -> list:
        """【支持20cm+剔除北交所】从当日全市场日线数据中筛选涨停股票"""
        if daily_df.empty:
            return []

        # ========== 新增：第一步先过滤北交所股票 ==========
        # 过滤规则：代码以83/87/88开头 或 交易所后缀为BJ
        def is_not_bse_stock(ts_code):
            if not isinstance(ts_code, str):
                return False
            code_part = ts_code.split('.')[0] if '.' in ts_code else ts_code
            exchange_part = ts_code.split('.')[1] if '.' in ts_code else ''
            return not (code_part.startswith(('83', '87', '88')) or exchange_part == 'BJ')

        # 过滤掉北交所股票
        daily_df = daily_df[daily_df['ts_code'].apply(is_not_bse_stock)]
        if daily_df.empty:
            logger.info("过滤北交所股票后，无剩余股票参与涨停判断")
            return []

        limit_up_stocks = []

        # 遍历剩余股票，进行涨停判断（原有逻辑不变）
        for idx, row in daily_df.iterrows():
            if "ts_code" not in row.index:
                logger.warning(f"日线数据缺少ts_code字段，跳过该行")
                continue

            ts_code = row["ts_code"]
            pre_close = row.get("pre_close", 0)
            close = row.get("close", 0)
            high = row.get("high", 0)

            if pre_close <= 0 or close <= 0 or high <= 0:
                logger.debug(f"{ts_code} 价格数据无效，跳过")
                continue

            # 若为北交所股票，limit_rate=0，自动跳过涨停判断
            limit_rate = self.get_limit_up_rate_by_ts_code(ts_code)
            if limit_rate == 0.0:
                continue  # 兜底：再次过滤北交所股票

            limit_up_price = pre_close * (1 + limit_rate)
            # 涨停条件：收盘价 & 最高价 达到涨停价（允许0.5%误差）
            is_limit_up = (close >= limit_up_price * 0.995) and (high >= limit_up_price * 0.995)

            if is_limit_up:
                limit_up_stocks.append(ts_code)

        logger.info(f"当日涨停股票数量（含20cm，剔除北交所）：{len(limit_up_stocks)}")
        return limit_up_stocks

    def get_stock_limit_up_time(self, ts_code: str, trade_date: str, pre_close: float) -> datetime:
        """
        获取单只股票首次自然涨停时间
        :return: 首次涨停时间，一字板/无数据返回None
        """
        limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
        # 拉取当日集合竞价到收盘的分钟线
        start_time = f"{trade_date} 09:25:00"
        end_time = f"{trade_date} 15:00:00"
        raw_df = data_fetcher.fetch_stk_mins(
            ts_code=ts_code,
            freq="1min",
            start_date=start_time,
            end_date=end_time
        )
        if raw_df.empty:
            return None

        # 清洗入库
        data_cleaner.clean_and_insert_kline_min(raw_df)
        # 查询格式化后的分钟线
        min_df = data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date)
        if min_df.empty:
            return None

        # 剔除一字板：9:25集合竞价已涨停
        call_auction_df = min_df[min_df["trade_time"] == f"{trade_date} 09:25:00"]
        if not call_auction_df.empty:
            if call_auction_df["close"].iloc[0] >= limit_up_price * 0.995:
                logger.debug(f"{ts_code} 为一字板，剔除")
                return None

        # 筛选9:30之后首次涨停的分钟线
        trading_min_df = min_df[min_df["trade_time"] >= f"{trade_date} 09:30:00"]
        limit_up_min_df = trading_min_df[trading_min_df["close"] >= limit_up_price * 0.995]
        if limit_up_min_df.empty:
            return None

        # 返回首次涨停时间
        first_limit_up_time = limit_up_min_df["trade_time"].min()
        logger.debug(f"{ts_code} 首次涨停时间：{first_limit_up_time}")
        return first_limit_up_time

    def select_stocks(self, trade_date: str, daily_df: pd.DataFrame, pre_close_map: dict) -> list:
        """选股核心逻辑：按涨停时间排序，选前5"""
        # 1. 获取当日涨停股票
        limit_up_stocks = self.get_daily_limit_up_stocks(daily_df)
        if not limit_up_stocks:
            logger.info("当日无涨停股票，无选股")
            return []

        # 2. 遍历获取每只股票的涨停时间
        stock_time_list = []
        for ts_code in limit_up_stocks:
            pre_close = pre_close_map.get(ts_code, 0)
            if pre_close <= 0:
                continue
            limit_up_time = self.get_stock_limit_up_time(ts_code, trade_date, pre_close)
            if limit_up_time is not None:
                stock_time_list.append({"ts_code": ts_code, "limit_up_time": limit_up_time})

        # 3. 按涨停时间从早到晚排序，选前5
        stock_time_list.sort(key=lambda x: x["limit_up_time"])
        selected_stocks = [item["ts_code"] for item in stock_time_list[:5]]
        logger.info(f"当日选中股票：{selected_stocks}")
        return selected_stocks

    def generate_signal(self, trade_date: str, daily_df: pd.DataFrame, pre_close_map: dict, positions: dict) -> tuple:
        """
        每日生成买卖信号（回测引擎调用）
        :return: (买入股票列表, 卖出信号字典)
        """
        # 1. 生成卖出信号
        sell_signal_map = {}
        for ts_code, position in positions.items():
            stock_df = daily_df[daily_df["ts_code"] == ts_code]
            if stock_df.empty:
                continue
            # 获取当日行情数据
            close_price = stock_df["close"].iloc[0]
            pre_close = stock_df["pre_close"].iloc[0]
            # 判断当日是否涨停
            is_limit_up = close_price >= self.calc_limit_up_price(pre_close) * 0.995

            # 卖出规则1：买入当日（hold_days=0）收盘未涨停，次日开盘卖
            if position.hold_days == 0:
                if not is_limit_up:
                    sell_signal_map[ts_code] = "open"
            # 卖出规则2：买入次日（hold_days=1）收盘未涨停，当日收盘卖
            elif position.hold_days == 1:
                if not is_limit_up:
                    sell_signal_map[ts_code] = "close"

        # 2. 选股生成买入列表
        buy_stocks = self.select_stocks(trade_date, daily_df, pre_close_map)
        logger.info(f"[{trade_date}] 卖出信号：{sell_signal_map}，买入列表：{buy_stocks}")
        return buy_stocks, sell_signal_map