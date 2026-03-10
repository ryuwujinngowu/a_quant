"""
特征数据容器 (FeatureDataBundle)
==================================
设计原则：
    1. 由外部（dataset.py）在特征计算前统一构建，所有因子类共享同一份数据
    2. 日线 / 分钟线各只发起一次 IO，因子内部禁止再自行拉数据
    3. load_minute=False 可跳过分钟线加载，适用于纯日线因子调试场景
"""
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from utils.common_tools import get_trade_dates, get_daily_kline_data
from data.data_cleaner import data_cleaner
from utils.log_utils import logger


class FeatureDataBundle:
    """
    特征计算统一数据容器

    构造参数：
        trade_date          : D 日，格式 yyyy-mm-dd
        target_ts_codes     : 所有候选股代码列表（三个板块合并去重后的完整列表）
        sector_candidate_map: {板块名: 候选股 DataFrame}（含 ts_code 等列）
        top3_sectors        : 当日 Top3 板块名称列表，顺序即板块 ID（1/2/3）
        adapt_score         : 板块轮动分（0-100），由 dataset.py 调用板块热度后传入，
                              避免 FeatureEngine 内重复调用 select_top3_hot_sectors
        load_minute         : 是否加载分钟线（默认 True），不需要 SEI 时可设 False 提速

    预加载属性（构造后即可使用）：
        lookback_dates_5d   : 含 D 日在内的最近 5 个交易日列表
        lookback_dates_20d  : 含 D 日在内的最近 20 个交易日列表
        daily_grouped       : dict，key=(ts_code, trade_date)，value=该行日线数据 dict
                              O(1) 查找，是所有因子计算的核心加速手段
        minute_cache        : dict，key=(ts_code, trade_date)，value=分钟线 DataFrame
    """

    def __init__(
            self,
            trade_date: str,
            target_ts_codes: List[str],
            sector_candidate_map: Dict[str, pd.DataFrame],
            top3_sectors: List[str],
            adapt_score: float = 0.0,
            load_minute: bool = True,
    ):
        self.trade_date = trade_date
        self.target_ts_codes = target_ts_codes
        self.sector_candidate_map = sector_candidate_map
        self.top3_sectors = top3_sectors
        self.adapt_score = adapt_score      # 透传给 SectorHeatFeature.calculate()

        self.lookback_dates_5d: List[str] = []
        self.lookback_dates_20d: List[str] = []
        self.daily_grouped: Dict[tuple, dict] = {}
        self.minute_cache: Dict[tuple, pd.DataFrame] = {}

        self._load_trade_dates()
        self._load_daily_data()
        if load_minute:
            self._load_minute_data()

    def _load_trade_dates(self):
        try:
            d_date = datetime.strptime(self.trade_date, "%Y-%m-%d")
            start_5d = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            self.lookback_dates_5d = get_trade_dates(start_5d, self.trade_date)[-5:]
            start_20d = (d_date - timedelta(days=40)).strftime("%Y-%m-%d")
            self.lookback_dates_20d = get_trade_dates(start_20d, self.trade_date)[-20:]
            logger.info(f"[DataBundle] {self.trade_date} 交易日加载完成 | 5日: {self.lookback_dates_5d}")
        except Exception as e:
            logger.error(f"[DataBundle] 交易日加载失败：{e}")
            raise

    def _load_daily_data(self):
        """批量加载日线（仅查候选股，不拉全市场）"""
        try:
            all_dates = list(set(self.lookback_dates_5d + self.lookback_dates_20d))
            frames = []
            for date in all_dates:
                df = get_daily_kline_data(trade_date=date, ts_code_list=self.target_ts_codes)
                if not df.empty:
                    df["trade_date"] = df["trade_date"].astype(str)
                    frames.append(df)
            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                self.daily_grouped = (
                    all_df.groupby(["ts_code", "trade_date"]).first().to_dict(orient="index")
                )
            logger.info(f"[DataBundle] 日线加载完成 | 日期数:{len(all_dates)} | 记录数:{len(self.daily_grouped)}")
        except Exception as e:
            logger.error(f"[DataBundle] 日线数据加载失败：{e}")
            raise

    def _load_minute_data(self):
        """加载候选股近 5 日分钟线（HDI/SEI 因子必需）"""
        try:
            for ts_code in self.target_ts_codes:
                for date in self.lookback_dates_5d:
                    key = (ts_code, date)
                    self.minute_cache[key] = data_cleaner.get_kline_min_by_stock_date(ts_code, date)
            logger.info(f"[DataBundle] 分钟线加载完成 | 记录数:{len(self.minute_cache)}")
        except Exception as e:
            logger.warning(f"[DataBundle] 分钟线加载异常（非致命）：{str(e)[:120]}")