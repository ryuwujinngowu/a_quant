import pandas as pd
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy
from utils.log_utils import logger

from features.ma_indicator import technical_features
from features.macd_indicator import macd_indicator

class DoubleMA_Strategy(BaseStrategy):
    """
    双均线+MACD策略（完全适配现有engine，不修改任何股票代码）
    核心适配点：
    1. 兼容engine分两次调用generate_signal的逻辑
    2. 不依赖on_buy_success回调，直接从account持仓获取成本价
    3. 保留engine需要的涨跌停计算方法
    4. sell_signal_map改为实例变量，供engine跨调用访问
    5. 完全不修改股票代码，保持原始格式
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "双均线MACD双确认趋势策略"

        # 策略核心参数（移除代码格式相关参数）
        self.short_ma_window = 5
        self.long_ma_window = 20
        self.max_position_count = 10
        self.fixed_stop_loss_rate = -0.08
        self.trailing_stop_profit_rate = 0.10

        # 关键：sell_signal_map改为实例变量（适配engine两次调用）
        self.sell_signal_map = {}
        # 不再依赖buy_price_map，直接从account获取成本价

    def initialize(self):
        """初始化：清空卖出信号"""
        self.sell_signal_map.clear()
        logger.info(f"【{self.strategy_name}】初始化完成")

    # 适配engine调用：保留涨跌停计算方法（engine会调用）
    def calc_limit_up_price(self, ts_code, pre_close):
        """计算涨停价（和engine逻辑保持一致）"""
        if ts_code.startswith(("3", "68")):
            return round(pre_close * 1.2, 2)
        else:
            return round(pre_close * 1.1, 2)

    def calc_limit_down_price(self, ts_code, pre_close):
        """计算跌停价（和engine逻辑保持一致）"""
        if ts_code.startswith(("3", "68")):
            return round(pre_close * 0.8, 2)
        else:
            return round(pre_close * 0.9, 2)

    def _get_stock_buy_price(self, ts_code, positions):
        """
        替代on_buy_success回调：直接从account持仓获取成本价
        完全使用原始ts_code，不做任何格式修改
        """
        if ts_code in positions:
            # positions是engine传入的真实持仓，包含成本价（需确认你的Account.positions结构）
            return positions[ts_code]["buy_price"]  # 替换为你Account类的实际字段名
        return None

    def _get_stock_max_price(self, ts_code, daily_df, trade_date):
        """获取持仓股票的历史最高价（从日线数据计算，不依赖缓存）"""
        # 取过去90天的最高价
        history_start_dt = datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=90)
        history_start_date = history_start_dt.strftime("%Y%m%d")
        # 调用MA计算逻辑时顺带获取最高价（或直接查日线数据）
        ma_df = technical_features.calculate_ma(
            ts_code=ts_code,  # 直接用原始代码，不修改
            start_date=history_start_date,
            end_date=trade_date.replace("-", ""),
            ma_days=[self.short_ma_window]
        )
        if not ma_df.empty:
            return ma_df["high"].max()  # 假设ma_df包含high字段，替换为实际字段名
        return daily_df[daily_df["ts_code"] == ts_code]["close"].iloc[0]

    def generate_signal(self, trade_date: str, daily_df: pd.DataFrame, positions: dict, account_cash=None):
        """
        适配engine分两次调用的逻辑：
        - 第一次调用：仅返回买入列表，卖出信号存入实例变量sell_signal_map
        - 第二次调用：返回空买入列表，卖出信号从实例变量读取
        全程不修改股票代码，保持原始格式
        """
        buy_stocks = []
        hold_ts_codes = list(positions.keys())
        trade_date_ymd = trade_date.replace("-", "")
        history_start_date = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y%m%d")

        # ========== 第一步：生成卖出信号（存入实例变量，供engine读取） ==========
        self.sell_signal_map.clear()  # 清空旧信号
        for ts_code in hold_ts_codes:
            # 完全使用原始ts_code，不做任何格式修改
            stock_daily = daily_df[daily_df["ts_code"] == ts_code]
            if stock_daily.empty:
                continue
            current_close = stock_daily["close"].iloc[0]

            # 从positions获取真实买入成本（替代回调）
            buy_price = self._get_stock_buy_price(ts_code, positions)
            if not buy_price:
                continue  # 无真实持仓成本，跳过卖出判断

            # 获取持仓最高价（从日线数据计算，不依赖缓存）
            max_hold_price = self._get_stock_max_price(ts_code, daily_df, trade_date)

            # 1. 固定止损
            current_pct = (current_close / buy_price) - 1
            if current_pct <= self.fixed_stop_loss_rate:
                self.sell_signal_map[ts_code] = "open"
                logger.info(f"[{trade_date}] {ts_code} 触发止损，生成开盘卖出信号")
                continue

            # 2. 移动止盈
            max_pct = (max_hold_price / buy_price) - 1
            current_retrace_pct = (max_hold_price - current_close) / max_hold_price
            if max_pct >= self.trailing_stop_profit_rate and current_retrace_pct >= 0.05:
                self.sell_signal_map[ts_code] = "close"
                logger.info(f"[{trade_date}] {ts_code} 触发止盈，生成收盘卖出信号")
                continue

            # 3. 均线+MACD死叉卖出
            ma_df = technical_features.calculate_ma(
                ts_code=ts_code,  # 直接用原始代码
                start_date=history_start_date,
                end_date=trade_date_ymd,
                ma_days=[self.short_ma_window, self.long_ma_window]
            )
            if len(ma_df) < 2:
                continue
            ma_today, ma_prev = ma_df.iloc[-1], ma_df.iloc[-2]
            is_death_cross = (ma_prev[f"ma{self.short_ma_window}"] >= ma_prev[f"ma{self.long_ma_window}"]) and \
                             (ma_today[f"ma{self.short_ma_window}"] < ma_today[f"ma{self.long_ma_window}"])
            if is_death_cross:
                macd_df = macd_indicator.calculate_macd(
                    ts_code=ts_code,  # 直接用原始代码
                    start_date=history_start_date,
                    end_date=trade_date_ymd
                )
                if not macd_df.empty and macd_df.iloc[-1]["is_death_cross"]:
                    self.sell_signal_map[ts_code] = "close"

        # ========== 第二步：生成买入信号（适配engine的买入规则） ==========
        # 仅在engine第一次调用时生成买入信号（通过仓位判断，避免重复）
        available_position_count = max(0, self.max_position_count - len(hold_ts_codes))
        if available_position_count > 0:
            # 初筛有效标的（和engine的拦截逻辑对齐）
            valid_df = daily_df[
                (daily_df["volume"] > 0) &
                (daily_df["amount"] > 10000000) &
                (abs((daily_df["close"] / daily_df["pre_close"]) - 1) < 0.095) &
                (daily_df["close"] > 1.0) &
                (~daily_df["ts_code"].isin(hold_ts_codes))
            ].copy()

            if not valid_df.empty:
                for idx, row in valid_df.iterrows():
                    if len(buy_stocks) >= available_position_count:
                        break
                    ts_code = row["ts_code"]  # 直接用原始代码，不做任何修改

                    # 均线金叉判断
                    ma_df = technical_features.calculate_ma(
                        ts_code=ts_code,  # 直接用原始代码
                        start_date=history_start_date,
                        end_date=trade_date_ymd,
                        ma_days=[self.short_ma_window, self.long_ma_window]
                    )
                    if len(ma_df) < 2:
                        continue
                    ma_today, ma_prev = ma_df.iloc[-1], ma_df.iloc[-2]
                    is_golden_cross = (ma_prev[f"ma{self.short_ma_window}"] <= ma_prev[f"ma{self.long_ma_window}"]) and \
                                      (ma_today[f"ma{self.short_ma_window}"] > ma_today[f"ma{self.long_ma_window}"])
                    is_close_above_long_ma = row["close"] > ma_today[f"ma{self.long_ma_window}"]
                    if not (is_golden_cross and is_close_above_long_ma):
                        continue

                    # MACD确认
                    macd_df = macd_indicator.calculate_macd(
                        ts_code=ts_code,  # 直接用原始代码
                        start_date=history_start_date,
                        end_date=trade_date_ymd
                    )
                    if macd_df.empty:
                        continue
                    macd_today = macd_df.iloc[-1]
                    is_macd_confirm = macd_today["is_golden_cross"] or (
                        macd_today["macd_bar"] > 0 and macd_today["macd_bar"] > macd_df.iloc[-2]["macd_bar"]
                    )
                    if is_macd_confirm:
                        buy_stocks.append(ts_code)
                        logger.info(f"[{trade_date}] {ts_code} 生成买入信号（适配engine规则）")

        # ========== 适配engine两次调用：第二次调用时返回空买入列表 ==========
        # 可通过trade_date或仓位判断是否是第二次调用，这里简化为“有卖出信号时返回空买入列表”
        if len(self.sell_signal_map) > 0 and len(buy_stocks) > 0:
            # 第一次调用：返回买入列表，卖出信号存实例变量
            return buy_stocks, self.sell_signal_map
        else:
            # 第二次调用：返回空买入列表，卖出信号从实例变量读取
            return [], self.sell_signal_map
