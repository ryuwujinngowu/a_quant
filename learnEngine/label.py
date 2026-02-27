# learnEngine/label.py
"""
交易标签生成器
标签规则：D日收盘买入，D+1日收盘卖出，扣除手续费后收益>0 → 标签=1，否则=0
"""
from typing import Optional
import pandas as pd
from utils.log_utils import logger


class LabelGenerator:
    """标签生成核心类"""

    def __init__(self, fee_rate: float = 0.002):
        self.fee_rate = fee_rate  # 手续费+滑点，默认千2

    def generate_label(
            self,
            daily_df: pd.DataFrame,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        生成全市场个股的交易标签
        :param daily_df: 全市场日线数据（必须包含ts_code、trade_date、close、is_trading字段）
        :param start_date: 标签生成开始日期（YYYY-MM-DD）
        :param end_date: 标签生成结束日期（YYYY-MM-DD）
        :return: 带label列的DataFrame，columns=[ts_code, trade_date, label]
        """
        # ==================== 后续填充具体计算逻辑 ====================
        # 1. 按个股分组，shift(-1)获取D+1日收盘价、交易状态
        # 2. 计算D日买入、D+1日卖出的净收益
        # 3. 按规则打标：收益>0→1，否则→0，剔除停牌/涨跌停无效样本
        # ==============================================================

        # 预设空返回值（格式固定）
        label_df = pd.DataFrame(columns=["ts_code", "trade_date", "label"])
        logger.info(f"标签生成完成，时间范围：{start_date} ~ {end_date}，样本数：{len(label_df)}")
        return label_df