import pandas as pd
from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger


class LimitUpPullback_Strategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.strategy_name = "涨停回马枪策略"
        # 策略参数
        self.limit_up_lookback_days = 10  # 回溯近10天的涨停
        self.pullback_rate = -0.08  # 回调幅度8%以上
        self.support_ma = 10  # 支撑均线10日线
        self.take_profit_rate = 0.1  # 止盈10%
        self.stop_loss_rate = -0.06  # 止损6%
        self.max_position_count = 6  # 最大持仓数量
        # 缓存：每只股票的涨停历史、价格历史
        self.limit_up_history = {}
        self.price_cache = {}
        self.buy_price_map = {}

    def initialize(self):
        self.limit_up_history = {}
        self.price_cache = {}
        self.buy_price_map = {}
        logger.info(f"【{self.strategy_name}】初始化完成，回溯近{self.limit_up_lookback_days}天涨停，回调幅度{self.pullback_rate*100}%")

    def generate_signal(self, trade_date: str, daily_df: pd.DataFrame, positions: dict):
        buy_stocks = []
        sell_signal_map = {}
        hold_ts_codes = list(positions.keys())

        # 1. 持仓止盈止损
        for ts_code in hold_ts_codes:
            stock_daily = daily_df[daily_df["ts_code"] == ts_code]
            if stock_daily.empty:
                continue
            current_price = stock_daily["close"].iloc[0]
            buy_price = self.buy_price_map.get(ts_code, current_price)
            pct_change = (current_price / buy_price) - 1

            if pct_change >= self.take_profit_rate:
                sell_signal_map[ts_code] = "close"
                logger.info(f"[{trade_date}] {ts_code} 止盈，生成收盘卖出信号")
                del self.buy_price_map[ts_code]
            elif pct_change <= self.stop_loss_rate:
                sell_signal_map[ts_code] = "open"
                logger.info(f"[{trade_date}] {ts_code} 止损，生成开盘卖出信号")
                del self.buy_price_map[ts_code]

        # 2. 当日涨停股记录
        valid_df = daily_df[daily_df["volume"] > 0].copy()
        valid_df["pct_change"] = (valid_df["close"] / valid_df["pre_close"]) - 1

        for idx, row in valid_df.iterrows():
            ts_code = row["ts_code"]
            pre_close = row["pre_close"]
            close_price = row["close"]
            # 计算涨停价，判断是否涨停
            limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
            is_limit_up = close_price >= limit_up_price - 0.001

            # 初始化缓存
            if ts_code not in self.limit_up_history:
                self.limit_up_history[ts_code] = []
            if ts_code not in self.price_cache:
                self.price_cache[ts_code] = []
            # 补充数据
            self.price_cache[ts_code].append(close_price)
            if is_limit_up:
                self.limit_up_history[ts_code].append({
                    "trade_date": trade_date,
                    "limit_up_price": limit_up_price,
                    "close_price": close_price
                })

        # 3. 筛选符合回马枪条件的股票
        for ts_code in self.limit_up_history.keys():
            # 已有持仓跳过
            if ts_code in hold_ts_codes:
                continue
            # 无涨停历史跳过
            if not self.limit_up_history[ts_code]:
                continue
            # 无价格数据跳过
            if ts_code not in self.price_cache or len(self.price_cache[ts_code]) < self.support_ma:
                continue
            # 当日无数据跳过
            stock_daily = daily_df[daily_df["ts_code"] == ts_code]
            if stock_daily.empty:
                continue
            current_price = stock_daily["close"].iloc[0]

            # 条件1：近N天有涨停
            recent_limit_up = [x for x in self.limit_up_history[ts_code]]
            if not recent_limit_up:
                continue
            # 取最近一次涨停的价格
            last_limit_up_price = recent_limit_up[-1]["close_price"]

            # 条件2：从涨停价回调幅度达到阈值
            pullback_pct = (current_price / last_limit_up_price) - 1
            if pullback_pct > self.pullback_rate:
                continue

            # 条件3：回调到支撑均线附近，且均线向上
            ma_price = pd.Series(self.price_cache[ts_code]).rolling(self.support_ma).mean().iloc[-1]
            is_near_support = abs(current_price / ma_price - 1) < 0.02  # 收盘价在均线2%以内
            ma_is_up = ma_price > pd.Series(self.price_cache[ts_code]).rolling(self.support_ma).mean().iloc[-2]
            if not is_near_support or not ma_is_up:
                continue

            # 符合条件，加入买入列表
            buy_stocks.append(ts_code)
            self.buy_price_map[ts_code] = current_price
            # 达到最大持仓数停止
            if len(buy_stocks) >= self.max_position_count - len(hold_ts_codes):
                break

        logger.info(f"[{trade_date}] 生成买入信号{len(buy_stocks)}只，卖出信号{len(sell_signal_map)}只")
        return buy_stocks, sell_signal_map

    def on_buy_success(self, ts_code: str, buy_price: float):
        self.buy_price_map[ts_code] = buy_price