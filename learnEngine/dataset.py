# learnEngine/dataset.py
"""
训练/推理数据集构建器
核心逻辑：个股只绑定所属板块的因子，避免模型混淆板块
"""
from typing import Tuple, List
import pandas as pd
import numpy as np

from features.market_stats import SectorHeatFeature
from learnEngine.label import LabelGenerator
from utils.log_utils import logger


class DatasetBuilder:
    """数据集构建核心类"""

    def __init__(self):
        self.sector_heat_feature = SectorHeatFeature()
        self.label_generator = LabelGenerator()
        # 固定特征列：30个板块因子 + 1个所属板块ID + 个股因子（预留扩展）
        self.base_feature_cols = ["sector_id"] + self.sector_heat_feature.get_factor_columns()

    def build_train_dataset(
            self,
            daily_df: pd.DataFrame,
            start_date: str,
            end_date: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        构建训练数据集（特征矩阵+标签）
        :param daily_df: 全市场日线数据
        :param start_date: 训练集开始日期
        :param end_date: 训练集结束日期
        :return:
            - X: 特征矩阵，shape=(样本数, 特征数)
            - y: 标签向量，shape=(样本数,)
            - feature_df: 带特征的完整DataFrame（用于复盘）
        """
        # ==================== 后续填充具体逻辑 ====================
        # 1. 生成标签数据
        # 2. 按交易日循环，每日计算板块因子、个股所属板块映射
        # 3. 给每只个股绑定所属板块的因子，其他板块因子填0
        # 4. 合并个股因子、标签，生成最终训练集
        # ==========================================================

        # 预设空返回值（格式固定）
        X = np.array([])
        y = np.array([])
        feature_df = pd.DataFrame(columns=["ts_code", "trade_date"] + self.base_feature_cols + ["label"])
        logger.info(f"训练数据集构建完成，时间范围：{start_date} ~ {end_date}，样本数：{len(feature_df)}，特征数：{X.shape[1] if len(X.shape)>1 else 0}")
        return X, y, feature_df

    def build_inference_dataset(
            self,
            daily_df: pd.DataFrame,
            trade_date: str,
            candidate_stocks: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        构建推理数据集（策略运行时调用，预测胜率用）
        :param daily_df: 当日全市场日线数据
        :param trade_date: 当前交易日（D日）
        :param candidate_stocks: 当日候选股票列表
        :return:
            - X_infer: 推理特征矩阵，shape=(候选股数, 特征数)
            - infer_df: 带特征的完整DataFrame（用于匹配股票代码）
        """
        # ==================== 后续填充具体逻辑 ====================
        # 1. 计算当日板块因子、个股-所属板块映射
        # 2. 给候选股绑定所属板块的因子，其他板块填0
        # 3. 生成和训练集格式完全一致的推理特征
        # ==========================================================

        # 预设空返回值（格式固定）
        X_infer = np.array([])
        infer_df = pd.DataFrame(columns=["ts_code", "trade_date"] + self.base_feature_cols)
        logger.info(f"推理数据集构建完成，日期：{trade_date}，候选股数：{len(candidate_stocks)}")
        return X_infer, infer_df