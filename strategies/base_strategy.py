# strategies/base_strategy.py
"""
所有策略的抽象基类（Base Strategy）
核心作用：
1. 定义引擎与策略之间的统一接口，实现引擎复用、多策略解耦
2. 强制所有自定义策略实现核心方法，避免运行时缺方法报错
3. 统一管理策略通用属性（如信号映射、策略名称）
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from config.config import (
    MAIN_BOARD_LIMIT_UP_RATE,
    STAR_BOARD_LIMIT_UP_RATE,
    BJ_BOARD_LIMIT_UP_RATE,
)
import pandas as pd


class BaseStrategy(ABC):
    """
    策略抽象基类：所有自定义策略必须继承此类并实现所有抽象方法
    约定引擎调用的核心接口，确保不同策略与引擎的兼容性
    """

    def __init__(self):
        # ========== 策略通用属性（所有子类共享，无需重复定义） ==========
        # 卖出信号映射：{ts_code: sell_type}，sell_type = "open"/"close"
        self.sell_signal_map: Dict[str, str] = {}
        # 策略名称（子类必须重写，用于回测指标/日志标识）
        # self.strategy_name: str = "UnnamedStrategy"
        # 策略参数（可选，子类可扩展，用于回测结果记录参数）
        self.strategy_params: Dict[str, any] = {}

    # ========== 核心抽象方法（子类必须实现） ==========
    @abstractmethod
    def initialize(self) -> None:
        """
        策略初始化方法（引擎回测启动前调用）
        用途：
        1. 清空信号缓存（如sell_signal_map）
        2. 重置策略内部状态（如持仓天数、指标缓存）
        3. 初始化策略参数
        """
        pass

    @abstractmethod
    def generate_signal(
            self,
            trade_date: str,
            daily_df: pd.DataFrame,
            positions: Dict[str, any]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        生成买卖信号（引擎每日循环调用的核心方法）
        :param trade_date: 当前交易日（格式：YYYY-MM-DD）
        :param daily_df: 当日全市场日线数据（columns包含ts_code/open/high/low/close等）
        :param positions: 当前账户持仓字典（{ts_code: 持仓对象/持仓信息}）
        :return: 
            - buy_stocks: 当日买入股票列表（[ts_code1, ts_code2, ...]）
            - sell_signal_map: 当日卖出信号字典（{ts_code: sell_type}）
        """
        pass

    def calc_limit_up_price(self, ts_code: str, pre_close: float) -> float:
        """
        计算股票涨停价（适配不同板块涨跌幅限制）
        :param ts_code: 股票代码（如600000.SH/300001.SZ/831010.BJ）
        :param pre_close: 前一日收盘价
        :return: 涨停价格（保留2位小数，符合A股价格精度）
        """
        # 空值保护：前收盘价无效时返回0，避免计算异常
        if not pre_close or pre_close <= 0:
            return 0.0

        # 1. 判断板块类型，匹配对应涨跌幅
        if ts_code.endswith(".BJ"):  # 北交所
            limit_rate = BJ_BOARD_LIMIT_UP_RATE
        elif ts_code.startswith(("300", "301", "302")) or (ts_code.startswith("3") and ts_code.endswith(".SZ")):  # 创业板
            limit_rate = STAR_BOARD_LIMIT_UP_RATE
        elif ts_code.startswith("688"):  # 科创板（补充：你配置里有FILTER_688_BOARD，这里一并适配）
            limit_rate = STAR_BOARD_LIMIT_UP_RATE  # 科创板和创业板涨跌幅一致（20%）
        else:  # 主板（60/00开头）
            limit_rate = MAIN_BOARD_LIMIT_UP_RATE

        # 2. 计算涨停价（四舍五入保留2位小数，符合A股价格规则）
        limit_up_price = pre_close * (1 + limit_rate / 100)
        return round(limit_up_price, 2)

    # ========== 可选扩展方法（子类按需重写） ==========
    def get_strategy_info(self) -> Dict[str, any]:
        """
        获取策略完整信息（用于回测结果记录）
        :return: 包含名称、参数的字典
        """
        return {
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params
        }

    def clear_signal(self) -> None:
        """清空卖出信号（通用方法，子类可直接调用）"""
        self.sell_signal_map.clear()