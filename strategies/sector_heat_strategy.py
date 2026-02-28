# strategies/sector_heat_strategy.py
"""
板块热度选股策略（适配XGBoost模型）
核心逻辑：每日筛选前3活跃板块→板块内候选股→XGBoost预测胜率→按胜率选股→D+1卖出
完全匹配BaseStrategy基类接口，正确生成买卖信号
"""
from typing import List, Dict, Tuple
import pandas as pd

from strategies.base_strategy import BaseStrategy
from features.market_stats import SectorHeatFeature
from learnEngine.dataset import DatasetBuilder
from learnEngine.model import SectorHeatXGBModel
from utils.log_utils import logger
from config.config  import (
    MAIN_BOARD_LIMIT_UP_RATE,
    STAR_BOARD_LIMIT_UP_RATE,
    BJ_BOARD_LIMIT_UP_RATE,
    MAX_POSITION_COUNT,
    FILTER_BSE_STOCK,
    FILTER_STAR_BOARD,
    FILTER_MAIN_BOARD
)


class SectorHeatStrategy(BaseStrategy):
    """板块热度选股策略，完全继承并实现基类所有抽象方法"""

    def __init__(self):
        super().__init__()
        # 策略基础配置（必须重写，基类约定）
        self.strategy_name = "SectorHeatStrategy"
        self.strategy_params = {
            "top_sector_num": 3,  # 前3活跃板块
            "stock_per_sector": 20,  # 每个板块选20只候选股
            "buy_top_k": 20,  # 按胜率选前20只买入
            "model_path": "models/sector_heat_xgb.pkl"
        }
        # 核心组件初始化
        self.sector_feature = SectorHeatFeature()
        self.dataset_builder = DatasetBuilder()
        self.xgb_model = SectorHeatXGBModel()


    def initialize(self) -> None:
        """
        策略初始化（基类强制实现）
        完全符合基类要求：清空信号+重置内部状态
        """
        # 1. 调用基类通用方法清空卖出信号（匹配基类设计）
        self.clear_signal()
        # 2. 重置策略内部持仓状态
        self.hold_stock_dict.clear()
        # 3. 加载XGBoost模型（策略专属初始化逻辑）
        load_success = self.xgb_model.load_model(self.strategy_params["model_path"])
        if not load_success:
            logger.warning("模型加载失败，策略将仅返回空信号")

    def generate_signal(
            self,
            trade_date: str,
            daily_df: pd.DataFrame,
            positions: Dict[str, any]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        核心：生成买卖信号（完全匹配基类接口要求）
        返回值严格对应基类定义：buy_stocks（买入列表） + sell_signal_map（卖出字典）
        """
        # ========== 步骤1：生成卖出信号（严格匹配基类的sell_signal_map格式） ==========
        # 初始化空的卖出信号字典（键：股票代码，值：卖出类型"close"）
        sell_signal_map = {}











        # ========== 步骤2：生成买入信号（严格匹配基类的buy_stocks列表格式） ==========
        buy_stocks = []
        # 防护：模型未加载/无数据时，返回空买入信号
        if self.xgb_model.model is None or daily_df.empty:
            return buy_stocks, sell_signal_map

        # 2.1 计算板块因子，筛选候选板块和候选股票（后续填充业务逻辑）
        sector_factor_df, top_sector_ids, stock_sector_map = self.sector_feature.calculate(daily_df, trade_date)
        candidate_stocks = self._select_candidate_stocks(daily_df, top_sector_ids, stock_sector_map)
        if not candidate_stocks:
            return buy_stocks, sell_signal_map

        # 2.2 构建推理数据集（匹配模型输入格式）
        X_infer, infer_df = self.dataset_builder.build_inference_dataset(daily_df, trade_date, candidate_stocks)
        if X_infer.size == 0:
            return buy_stocks, sell_signal_map

        # 2.3 模型预测胜率（核心逻辑）
        win_rate_list = self.xgb_model.predict_win_rate(X_infer)
        infer_df["win_rate"] = win_rate_list

        # 2.4 按胜率排序，生成最终买入列表（严格匹配基类的List[str]格式）
        buy_top_k = self.strategy_params["buy_top_k"]
        buy_stocks = infer_df.sort_values(by="win_rate", ascending=False).head(buy_top_k)["ts_code"].tolist()

        # 2.5 记录买入日期（用于后续D+1生成卖出信号）
        for ts_code in buy_stocks:
            self.hold_stock_dict[ts_code] = trade_date

        # 日志记录信号生成结果（便于调试）
        logger.info(
            f"【{self.strategy_name} - {trade_date}】生成买入信号{len(buy_stocks)}只，卖出信号{len(sell_signal_map)}只")

        # ========== 严格返回基类要求的Tuple[List[str], Dict[str, str]] ==========
        return buy_stocks, sell_signal_map

    def calc_limit_up_price(self, ts_code: str, pre_close: float) -> float:
        """
        计算买入价格（基类强制实现）
        非打板策略直接返回前收盘价，完全符合基类接口要求
        """
        return pre_close

    # ========== 预留扩展方法（后续填充业务逻辑，不影响基类接口） ==========
    def _select_candidate_stocks(
            self,
            daily_df: pd.DataFrame,
            top_sector_ids: List[int],
            stock_sector_map: Dict[str, int]
    ) -> List[str]:
        """筛选板块内候选股票（后续填充：如板块内量价筛选逻辑）"""
        # 示例骨架：后续可改为按板块内换手率/涨幅筛选前N只
        candidate_stocks = []
        for sector_id in top_sector_ids:
            # 筛选该板块的股票
            sector_stocks = [ts for ts, sid in stock_sector_map.items() if sid == sector_id]
            # 取前stock_per_sector只（后续替换为真实筛选逻辑）
            candidate_stocks.extend(sector_stocks[:self.strategy_params["stock_per_sector"]])
        return candidate_stocks


    def _is_next_trade_day(self, buy_date: str, current_date: str) -> bool:
        """判断是否为下一个交易日（后续填充：对接交易日历工具）"""
        # 示例骨架：后续替换为真实的交易日历判断（如通过utils工具类）
        # 临时返回True用于测试，实际需根据交易日历计算
        return True