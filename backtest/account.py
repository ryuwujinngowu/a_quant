import pandas as pd

from config.config import (
    COMMISSION_RATE, STAMP_DUTY_RATE, SLIPPAGE_RATE,
    T_PLUS_1, MIN_TRADE_VOLUME, MAX_POSITION_COUNT
)
from utils.log_utils import logger


class Position:
    """单个持仓标的类，管理单只股票的持仓信息"""

    def __init__(self, ts_code: str, buy_price: float, buy_volume: int, buy_date: str):
        self.ts_code = ts_code
        self.buy_price = buy_price  # 买入成本价
        self.buy_volume = buy_volume  # 持仓数量
        self.buy_date = buy_date  # 买入日期
        self.hold_days = 0  # 已持有天数
        self.can_sell = False  # T+1规则，买入次日可卖

    def update_hold_days(self):
        """每日收盘后更新持有天数和可卖状态"""
        self.hold_days += 1
        if T_PLUS_1 and self.hold_days >= 1:
            self.can_sell = True


class Account:
    """多标的分仓账户管理类"""

    def __init__(self, init_capital: float, max_position_count: int = MAX_POSITION_COUNT):
        # 账户核心资产
        self.init_capital = init_capital
        self.available_cash = init_capital  # 可用资金
        self.total_asset = init_capital  # 总资产=可用资金+持仓总市值
        # 分仓配置
        self.max_position_count = max_position_count
        self.per_position_cash = init_capital / max_position_count  # 单份仓位固定资金
        # 持仓管理：key=股票代码，value=Position对象
        self.positions = {}
        # 交易记录与净值曲线
        self.trade_history = []
        self.daily_net_value = []

    def update_daily_asset(self, trade_date: str, daily_price_df: pd.DataFrame):
        """每日收盘后必须调用：更新账户资产、持仓状态"""
        # 1. 更新持仓持有天数和可卖状态
        for position in self.positions.values():
            position.update_hold_days()

        # 2. 计算持仓总市值
        total_position_value = 0.0
        for ts_code, position in self.positions.items():
            # 获取当日收盘价，无数据则用成本价
            stock_df = daily_price_df[daily_price_df["ts_code"] == ts_code]
            close_price = stock_df["close"].iloc[0] if not stock_df.empty else position.buy_price
            position_value = position.buy_volume * close_price
            total_position_value += position_value

        # 3. 更新总资产
        self.total_asset = self.available_cash + total_position_value

        # 4. 记录每日净值
        self.daily_net_value.append({
            "trade_date": trade_date,
            "total_asset": round(self.total_asset, 2),
            "available_cash": round(self.available_cash, 2),
            "position_count": len(self.positions),
            "total_position_value": round(total_position_value, 2)
        })
        logger.debug(f"[{trade_date}] 账户更新完成，总资产：{round(self.total_asset, 2)}元，持仓数：{len(self.positions)}")

    def get_available_position_count(self) -> int:
        """获取剩余可开仓的仓位数量"""
        return self.max_position_count - len(self.positions)

    def buy(self, trade_date: str, ts_code: str, price: float) -> bool:
        """执行买入操作，单只股票占用1份仓位"""
        # 买入合法性校验
        if self.get_available_position_count() <= 0:
            logger.warning(f"[{trade_date}] {ts_code} 买入失败：无可用仓位")
            return False
        if ts_code in self.positions:
            logger.warning(f"[{trade_date}] {ts_code} 买入失败：已持仓该股票")
            return False

        # 滑点处理：买入价上浮
        actual_price = price * (1 + SLIPPAGE_RATE)
        # 计算可买数量（1手的整数倍）
        max_can_buy = int(self.per_position_cash / (actual_price * MIN_TRADE_VOLUME)) * MIN_TRADE_VOLUME
        if max_can_buy < MIN_TRADE_VOLUME:
            logger.warning(f"[{trade_date}] {ts_code} 买入失败：资金不足1手")
            return False

        # 计算手续费（最低5元）
        commission = max(max_can_buy * actual_price * COMMISSION_RATE, 5)
        total_cost = max_can_buy * actual_price + commission

        # 校验可用资金
        if total_cost > self.available_cash:
            logger.warning(f"[{trade_date}] {ts_code} 买入失败：可用资金不足")
            return False

        # 更新账户与持仓
        self.available_cash -= total_cost
        self.positions[ts_code] = Position(
            ts_code=ts_code,
            buy_price=actual_price,
            buy_volume=max_can_buy,
            buy_date=trade_date
        )

        # 记录交易
        self.trade_history.append({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "direction": "买入",
            "price": round(actual_price, 4),
            "volume": max_can_buy,
            "commission": round(commission, 2),
            "stamp_duty": 0,
            "total_cost": round(total_cost, 2)
        })
        logger.info(f"[{trade_date}] {ts_code} 买入成功，价格：{round(actual_price, 4)}，数量：{max_can_buy}")
        return True

    def sell(self, trade_date: str, ts_code: str, price: float) -> bool:
        """执行卖出操作"""
        # 卖出合法性校验
        if ts_code not in self.positions:
            logger.warning(f"[{trade_date}] {ts_code} 卖出失败：无该持仓")
            return False
        position = self.positions[ts_code]
        if not position.can_sell:
            logger.warning(f"[{trade_date}] {ts_code} 卖出失败：T+1规则，当日不可卖")
            return False

        # 滑点处理：卖出价下浮
        actual_price = price * (1 - SLIPPAGE_RATE)
        volume = position.buy_volume

        # 计算手续费+印花税
        commission = max(volume * actual_price * COMMISSION_RATE, 5)
        stamp_duty = volume * actual_price * STAMP_DUTY_RATE
        total_income = volume * actual_price - commission - stamp_duty

        # 更新账户
        self.available_cash += total_income
        del self.positions[ts_code]

        # 记录交易
        self.trade_history.append({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "direction": "卖出",
            "price": round(actual_price, 4),
            "volume": volume,
            "commission": round(commission, 2),
            "stamp_duty": round(stamp_duty, 2),
            "total_income": round(total_income, 2)
        })
        logger.info(f"[{trade_date}] {ts_code} 卖出成功，价格：{round(actual_price, 4)}，数量：{volume}")
        return True

    def get_net_value_df(self) -> pd.DataFrame:
        """获取净值曲线DataFrame"""
        return pd.DataFrame(self.daily_net_value)

    def get_trade_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame(self.trade_history)