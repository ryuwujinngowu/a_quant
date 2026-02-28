# features/market_stats.py
"""
板块热度因子计算（核心：30个情绪因子输出）
因子命名规则：sector{板块ID}_d{时间跨度}_{指标名}
- 板块ID：1/2/3（每日前3活跃板块）
- 时间跨度：d4=4天前、d3=3天前、d2=2天前、d1=1天前、d0=D日
- 指标名：profit=涨幅>7%个股数（赚钱效应）、loss=跌幅>7%个股数（亏钱效应）
"""
from typing import List, Dict, Tuple
import pandas as pd

from features.base_feature import BaseFeature


class SectorHeatFeature(BaseFeature):
    """板块热度特征类，继承基类，对齐现有特征层架构"""

    def __init__(self):
        super().__init__()
        # 固定配置：前3活跃板块、回溯5天、2个核心指标
        self.top_sector_num = 3
        self.lookback_days = 5
        self.factor_columns = self._gen_factor_columns()  # 预生成30个因子列名

    def _gen_factor_columns(self) -> List[str]:
        """预生成30个因子列名（固定格式，避免命名混乱）"""
        factor_cols = []
        for sector_id in range(1, self.top_sector_num + 1):
            for day_offset in range(self.lookback_days):
                # d4=4天前，d0=D日
                day_tag = f"d{self.lookback_days - 1 - day_offset}"
                factor_cols.append(f"sector{sector_id}_{day_tag}_profit")
                factor_cols.append(f"sector{sector_id}_{day_tag}_loss")
        return factor_cols

    def calculate(self, daily_df: pd.DataFrame, trade_date: str) -> Tuple[pd.DataFrame, List[int], Dict[str, int]]:
        """
        核心计算方法（对齐基类接口）
        :param daily_df: 全市场日线数据（含个股所属板块、涨跌幅、成交量等）
        :param trade_date: 当前计算日期（D日，格式：YYYY-MM-DD）
        :return:
            - sector_factor_df: 板块因子DataFrame，index=板块ID，columns=30个因子列
            - top_sector_ids: 当日前3活跃板块ID列表 [1,2,3]
            - stock_sector_map: 个股-所属板块映射 {ts_code: 所属板块ID}

        """
        # ==================== 后续填充具体计算逻辑 ====================
        # 1. 计算每日板块热度，筛选前3活跃板块，分配板块ID 1/2/3
        # 2. 回溯5天，计算每个板块每日的赚钱/亏钱效应
        # 3. 生成个股-所属板块的映射关系
        # ==============================================================

        # 预设空返回值（格式固定，后续填充真实数据）
        sector_factor_df = pd.DataFrame(columns=self.factor_columns, index=[1,2,3])
        top_sector_ids = [1,2,3]
        stock_sector_map = {}

        return sector_factor_df, top_sector_ids, stock_sector_map

    def get_factor_columns(self) -> List[str]:
        """获取30个因子列名，供数据集构建时调用"""
        return self.factor_columns

if __name__ == "__main__":
        callable= SectorHeatFeature()
        tag = callable._gen_factor_columns()
        print(tag)

