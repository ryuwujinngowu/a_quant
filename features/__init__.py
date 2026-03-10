"""
特征层统一导出入口
"""
# 基类和注册中心
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry, FeatureRegistry
from features.data_bundle import FeatureDataBundle

# 因子类自动注册（导入即注册）
from features.emotion.sei_feature import SEIFeature
from features.sector.sector_heat_feature import SectorHeatFeature
from features.sector.sector_stock_feature import SectorStockFeature

# 兼容原有入口
from features.market_stats import SectorHeatFeature as LegacySectorHeatFeature

# 重构后的特征引擎
from typing import List, Dict
import pandas as pd
from utils.log_utils import logger


class FeatureEngine:
    """【重构后】特征引擎，支持灵活选择因子，统一调度计算"""
    def __init__(self, feature_name_list: List[str] = None):
        """
        初始化特征引擎
        :param feature_name_list: 要计算的因子名称列表，不传则计算所有已注册的因子
        可用因子：sei_emotion、sector_heat、sector_stock
        """
        if feature_name_list is None:
            self.features = feature_registry.get_all_features()
        else:
            self.features = feature_registry.get_features(feature_name_list)
        self.logger = logger

    def run_single_date(self, data_bundle: FeatureDataBundle) -> pd.DataFrame:
        """
        单日特征计算入口，返回当日全量特征DataFrame
        :param data_bundle: 预加载的数据容器
        :return: 合并后的个股-交易日级特征DataFrame
        """
        date_feature_dfs = []
        trade_date = data_bundle.trade_date

        for feature in self.features:
            try:
                feature_df, _ = feature.calculate(data_bundle)
                if not feature_df.empty:
                    date_feature_dfs.append(feature_df)
                    self.logger.debug(f"[FeatureEngine] {feature.feature_name} 计算完成，{len(feature_df)}行")
            except Exception as e:
                self.logger.error(f"[FeatureEngine] {feature.feature_name} 计算失败：{str(e)}", exc_info=True)
                return pd.DataFrame()

        # 合并所有特征，按stock_code+trade_date对齐
        if not date_feature_dfs:
            self.logger.warning(f"[FeatureEngine] {trade_date} 无有效特征数据")
            return pd.DataFrame()

        # 主键合并，内连接保证所有特征都有值
        full_df = date_feature_dfs[0]
        for df in date_feature_dfs[1:]:
            merge_keys = ["stock_code", "trade_date"]
            # 处理全局因子（无stock_code的情况）
            if "stock_code" not in df.columns:
                full_df = pd.merge(full_df, df, on=["trade_date"], how="left")
            else:
                full_df = pd.merge(full_df, df, on=merge_keys, how="inner")

        # 去重列
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        self.logger.info(f"[FeatureEngine] {trade_date} 特征合并完成，最终列数：{len(full_df.columns)}，行数：{len(full_df)}")
        return full_df