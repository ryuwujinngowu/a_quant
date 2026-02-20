import pandas as pd
from datetime import datetime

from config.config import (
    COMMISSION_RATE, STAMP_DUTY_RATE, SLIPPAGE_RATE,
    T_PLUS_1, MIN_TRADE_VOLUME, MAX_POSITION_COUNT
)
from utils.log_utils import logger


class Position:
    """单个持仓标的类，管理单只股票的持仓信息"""

    def __init__(self, ts_code: str, buy_price: float, buy_volume: int, buy_date: str, buy_total_cost: float):
        self.ts_code = ts_code
        self.buy_price = buy_price  # 买入成本价（含滑点）
        self.buy_volume = buy_volume  # 持仓数量
        self.buy_date = buy_date  # 买入日期
        self.hold_days = 0  # 已持有天数
        # ========== 优化1：合并buy_date赋值（原代码重复赋值，无错误但冗余） ==========
        self.buy_date = self._unify_date_format(buy_date)  # 直接统一格式，避免重复赋值
        self.can_sell = False  # T+1规则，买入次日可卖
        # ========== 新增：买入总成本（含手续费，用于精准计算卖出盈亏） ==========
        self.buy_total_cost = buy_total_cost

    def _unify_date_format(self, date_str: str) -> str:
        """
        统一日期格式为YYYYMMDD（无横线），兼容YYYY-MM-DD/YYYYMMDD等格式
        :param date_str: 任意格式的日期字符串
        :return: 统一格式的日期字符串（YYYYMMDD）
        """
        try:
            # 先尝试解析为datetime，再转成YYYYMMDD
            return datetime.strptime(date_str.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"日期格式转换失败：{date_str}，错误：{e}")
            return date_str  # 保留原始值，避免程序崩溃

    # ========== 核心修复1：新增单独的可卖状态更新方法（开盘前手动调用） ==========
    def update_can_sell(self, current_trade_date: str):
        """
        手动更新可卖状态（开盘卖出前调用，不依赖收盘后的hold_days更新）
        :param current_trade_date: 当前交易日（任意格式）
        """
        current_date = self._unify_date_format(current_trade_date)
        buy_date = self.buy_date
        if T_PLUS_1 and buy_date and current_date:
            self.can_sell = buy_date < current_date
        else:
            self.can_sell = True
        logger.debug(
            f"[{self.ts_code}] 手动更新可卖状态：买入日期={buy_date}，当前交易日={current_date} → can_sell={self.can_sell}")

    def update_hold_days(self, current_trade_date: str):
        """
        每日收盘后更新持有天数和可卖状态
        :param current_trade_date: 当前交易日（格式YYYY-MM-DD）
        """
        current_date = self._unify_date_format(current_trade_date)
        buy_date = self.buy_date  # 已统一为YYYYMMDD

        # 1. 策略信号用的hold_days更新（仅用于区分炸板票/历史持仓）
        # 规则：D日买入的股票，D日不+1，D+1日收盘才+1，确保D+1日盘中仍能识别为炸板票
        if buy_date < current_date:
            self.hold_days += 1

        # 2. T+1合规可卖状态判断（与hold_days完全解耦，仅看买入日期）
        # A股T+1规则：当日买入、当日不可卖，次日起全天可卖，与持有天数无关
        if T_PLUS_1:
            self.can_sell = buy_date < current_date
        else:
            self.can_sell = True  # 兼容T+0场景


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

        # ========== 新增1：回测核心信息（用于CSV字段填充） ==========
        self.strategy_name = "未命名策略"  # 策略名称，回测引擎传参赋值
        self.backtest_start_date = ""  # 回测开始日期，回测引擎传参赋值
        self.backtest_end_date = ""  # 回测结束日期，回测引擎传参赋值
        # ========== 新增2：盈亏计算辅助属性 ==========
        self.prev_total_asset = init_capital  # 前一日总资产，用于计算当日盈亏
        self.daily_sold_pnl = {}  # 每日卖出标的盈亏：{trade_date: {ts_code: 盈亏金额}}

    # ========== 新增3：回测信息设置方法（回测引擎必须调用，解决CSV字段为空问题） ==========
    def set_backtest_info(self, strategy_name: str, start_date: str, end_date: str):
        """
        设置回测核心信息（回测引擎初始化后必须调用）
        :param strategy_name: 策略名称
        :param start_date: 回测开始日期（YYYY-MM-DD/YYYYMMDD均可）
        :param end_date: 回测结束日期（YYYY-MM-DD/YYYYMMDD均可）
        """
        self.strategy_name = strategy_name
        self.backtest_start_date = self._unify_date_format(start_date)
        self.backtest_end_date = self._unify_date_format(end_date)
        logger.info(f"回测信息已设置：策略名称={strategy_name}，回测时间段={start_date} 至 {end_date}")

    # ========== 核心修改1：每日资产更新+结构化盈亏日志打印 ==========
    def update_daily_asset(self, trade_date: str, daily_price_df: pd.DataFrame):
        """每日收盘后必须调用：更新账户资产、持仓状态、打印结构化盈亏日志"""
        # 1. 更新持仓持有天数和可卖状态（原有逻辑完全不变）
        for position in self.positions.values():
            position.update_hold_days(trade_date)

        # 2. 计算持仓总市值+单只持仓标的盈亏明细
        total_position_value = 0.0
        position_pnl_detail = {}  # 持仓标的盈亏明细：{ts_code: 盈亏信息}
        for ts_code, position in self.positions.items():
            # 获取当日收盘价，无数据则用成本价
            stock_df = daily_price_df[daily_price_df["ts_code"] == ts_code]
            close_price = stock_df["close"].iloc[0] if not stock_df.empty else position.buy_price
            position_value = position.buy_volume * close_price
            total_position_value += position_value

            # 计算单只标的持仓盈亏
            hold_pnl = (close_price - position.buy_price) * position.buy_volume  # 持仓浮动盈亏
            hold_pnl_rate = (close_price - position.buy_price) / position.buy_price * 100  # 持仓收益率
            position_pnl_detail[ts_code] = {
                "持仓数量": position.buy_volume,
                "买入成本价": round(position.buy_price, 4),
                "当日收盘价": round(close_price, 4),
                "当日浮动盈亏": round(hold_pnl, 2),
                "持仓累计收益率(%)": round(hold_pnl_rate, 2)
            }

        # 3. 更新总资产
        self.total_asset = self.available_cash + total_position_value

        # 4. 计算当日整体盈亏
        daily_pnl = self.total_asset - self.prev_total_asset
        daily_pnl_rate = daily_pnl / self.prev_total_asset * 100 if self.prev_total_asset > 0 else 0
        total_pnl = self.total_asset - self.init_capital
        total_pnl_rate = total_pnl / self.init_capital * 100 if self.init_capital > 0 else 0

        # 5. 记录每日净值（新增CSV所需关键字段，解决字段为空问题）
        self.daily_net_value.append({
            # 新增回测核心字段
            "策略名称": self.strategy_name,
            "回测开始日期": self.backtest_start_date,
            "回测结束日期": self.backtest_end_date,
            "初始资金": round(self.init_capital, 2),
            # 原有字段：恢复total_asset英文字段（兼容metrics）+ 保留中文字段（CSV导出）
            "trade_date": trade_date,  # 恢复原英文字段
            "total_asset": round(self.total_asset, 2),  # 核心修复：恢复total_asset字段
            "available_cash": round(self.available_cash, 2),  # 恢复原英文字段
            "position_count": len(self.positions),  # 恢复原英文字段
            "total_position_value": round(total_position_value, 2),  # 恢复原英文字段
            # 中文字段（保留，满足CSV/日志需求）
            "交易日": trade_date,
            "总资产": round(self.total_asset, 2),
            "可用资金": round(self.available_cash, 2),
            "持仓总市值": round(total_position_value, 2),
            "持仓数量": len(self.positions),
            # 新增盈亏字段
            "当日盈亏": round(daily_pnl, 2),
            "当日收益率(%)": round(daily_pnl_rate, 2),
            "累计盈亏": round(total_pnl, 2),
            "累计收益率(%)": round(total_pnl_rate, 2)
        })

        # ========== 核心：结构化打印当日盈亏日志 ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"【{trade_date} 每日结算盈亏报告】")
        logger.info(f"{'='*60}")
        # 整体盈亏
        logger.info(f"📊 账户整体情况：")
        logger.info(f"  当日盈亏：{round(daily_pnl, 2)} 元 | 当日收益率：{round(daily_pnl_rate, 2)} %")
        logger.info(f"  累计盈亏：{round(total_pnl, 2)} 元 | 累计收益率：{round(total_pnl_rate, 2)} %")
        logger.info(f"  账户总资产：{round(self.total_asset, 2)} 元 | 可用资金：{round(self.available_cash, 2)} 元")
        logger.info(f"  持仓总市值：{round(total_position_value, 2)} 元 | 持仓标的数量：{len(self.positions)} 只")

        # 持仓标的盈亏明细
        logger.info(f"\n📈 持仓标的盈亏明细：")
        if position_pnl_detail:
            for ts_code, pnl_info in position_pnl_detail.items():
                logger.info(
                    f"  {ts_code}：持仓{pnl_info['持仓数量']}股 | 成本价{pnl_info['买入成本价']} | 收盘价{pnl_info['当日收盘价']} | 当日盈亏{pnl_info['当日浮动盈亏']}元 | 累计收益率{pnl_info['持仓累计收益率(%)']}%"
                )
        else:
            logger.info(f"  当日无持仓")

        # 当日卖出标的盈亏明细
        logger.info(f"\n💸 当日卖出标的盈亏明细：")
        current_sold_pnl = self.daily_sold_pnl.get(trade_date, {})
        if current_sold_pnl:
            total_sold_pnl = sum(current_sold_pnl.values())
            for ts_code, pnl in current_sold_pnl.items():
                logger.info(f"  {ts_code}：卖出净盈亏 {round(pnl, 2)} 元")
            logger.info(f"  👉 当日卖出总盈亏：{round(total_sold_pnl, 2)} 元")
        else:
            logger.info(f"  当日无卖出操作")
        logger.info(f"{'='*60}\n")

        # 6. 更新前一日总资产（用于下一日盈亏计算）
        self.prev_total_asset = self.total_asset
        # 7. 清空当日卖出盈亏（避免跨日残留）
        self.daily_sold_pnl.pop(trade_date, None)

    def get_available_position_count(self) -> int:
        """获取剩余可开仓的仓位数量"""
        return self.max_position_count - len(self.positions)

    # ========== 修改：买入时传入买入总成本，用于后续盈亏计算 ==========
    def buy(self, trade_date: str, ts_code: str, price: float) -> bool:
        """执行买入操作，单只股票占用1份仓位"""
        # 买入合法性校验（原有逻辑完全不变）
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

        # 更新账户与持仓（新增传入buy_total_cost）
        self.available_cash -= total_cost
        self.positions[ts_code] = Position(
            ts_code=ts_code,
            buy_price=actual_price,
            buy_volume=max_can_buy,
            buy_date=trade_date,
            buy_total_cost=total_cost  # 新增：传入买入总成本
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

    # ========== 修改：卖出时计算盈亏并记录，用于日志打印 ==========
    def sell(self, trade_date: str, ts_code: str, price: float) -> bool:
        """执行卖出操作"""
        # 卖出合法性校验（原有逻辑完全不变）
        if ts_code not in self.positions:
            logger.warning(f"[{trade_date}] {ts_code} 卖出失败：无该持仓")
            return False
        position = self.positions[ts_code]

        # 关键：卖出前手动更新一次可卖状态（解决收盘后才更新的时机问题）
        position.update_can_sell(trade_date)

        logger.debug(
            f"[{trade_date}] {ts_code} 卖出校验：买入日期={position.buy_date}，当前交易日={trade_date}，can_sell={position.can_sell}，hold_days={position.hold_days}")
        if not position.can_sell:
            logger.warning(
                f"[{trade_date}] {ts_code} 卖出失败：T+1规则，当日不可卖（买入日期={position.buy_date}，当前交易日={self._unify_date_format(trade_date)}）")
            return False

        # 滑点处理：卖出价下浮
        actual_price = price * (1 - SLIPPAGE_RATE)
        volume = position.buy_volume

        # 计算手续费+印花税
        commission = max(volume * actual_price * COMMISSION_RATE, 5)
        stamp_duty = volume * actual_price * STAMP_DUTY_RATE
        total_income = volume * actual_price - commission - stamp_duty

        # ========== 新增：计算卖出净盈亏，记录到当日卖出明细 ==========
        sell_pnl = total_income - position.buy_total_cost  # 净盈亏=卖出净收入-买入总成本
        if trade_date not in self.daily_sold_pnl:
            self.daily_sold_pnl[trade_date] = {}
        self.daily_sold_pnl[trade_date][ts_code] = sell_pnl

        # 更新账户
        self.available_cash += total_income
        del self.positions[ts_code]

        # 记录交易（新增卖出盈亏字段）
        self.trade_history.append({
            "trade_date": trade_date,
            "ts_code": ts_code,
            "direction": "卖出",
            "price": round(actual_price, 4),
            "volume": volume,
            "commission": round(commission, 2),
            "stamp_duty": round(stamp_duty, 2),
            "total_income": round(total_income, 2),
            "卖出净盈亏": round(sell_pnl, 2)  # 新增：记录卖出盈亏
        })
        logger.info(f"[{trade_date}] {ts_code} 卖出成功，价格：{round(actual_price, 4)}，数量：{volume}，净盈亏：{round(sell_pnl, 2)}元")
        return True

    # ========== 新增：回测结果导出方法（解决CSV字段缺失问题） ==========
    def export_backtest_result(self, file_path: str = "回测结果.csv"):
        """
        导出回测结果（CSV/Excel格式，包含所有关键字段）
        :param file_path: 导出文件路径，支持.csv/.xlsx后缀
        """
        # 1. 生成净值DataFrame（已包含策略名称、回测时间段等字段）
        net_value_df = self.get_net_value_df()
        # 2. 生成交易记录DataFrame
        trade_df = self.get_trade_df()
        # 3. 生成回测汇总表
        summary_df = pd.DataFrame({
            "策略名称": [self.strategy_name],
            "回测开始日期": [self.backtest_start_date],
            "回测结束日期": [self.backtest_end_date],
            "初始资金": [round(self.init_capital, 2)],
            "最终总资产": [round(self.total_asset, 2)],
            "总盈亏": [round(self.total_asset - self.init_capital, 2)],
            "总收益率(%)": [round((self.total_asset - self.init_capital)/self.init_capital*100, 2)],
            "总交易次数": [len(self.trade_history)],
            "最大持仓数": [self.max_position_count]
        })

        # 导出文件
        if file_path.endswith(".xlsx"):
            # Excel格式：分sheet存储
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="回测汇总", index=False)
                net_value_df.to_excel(writer, sheet_name="每日净值", index=False)
                trade_df.to_excel(writer, sheet_name="交易记录", index=False)
            logger.info(f"回测结果已导出Excel：{file_path}")
        else:
            # CSV格式：导出每日净值（包含所有关键字段）
            net_value_df.to_csv(file_path, index=False, encoding="utf-8-sig")
            # 同步导出交易记录和汇总
            trade_df.to_csv(file_path.replace(".csv", "_交易记录.csv"), index=False, encoding="utf-8-sig")
            summary_df.to_csv(file_path.replace(".csv", "_回测汇总.csv"), index=False, encoding="utf-8-sig")
            logger.info(f"回测结果已导出CSV：{file_path}")

    def get_net_value_df(self) -> pd.DataFrame:
        """获取净值曲线DataFrame（已包含策略名称、回测时间段等关键字段）"""
        return pd.DataFrame(self.daily_net_value)

    def get_trade_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame(self.trade_history)

    # ========== 辅助方法：复用日期格式化逻辑 ==========
    def _unify_date_format(self, date_str: str) -> str:
        """复用Position的日期格式化方法（避免重复代码）"""
        try:
            return datetime.strptime(date_str.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"日期格式转换失败：{date_str}，错误：{e}")
            return date_str