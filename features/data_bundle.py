"""
特征数据容器：统一预加载所有所需数据，避免重复IO，提升性能
"""
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from utils.common_tools import get_trade_dates, get_daily_kline_data
from data.data_cleaner import data_cleaner
from utils.log_utils import logger


class FeatureDataBundle:
    """
    特征数据容器，一次性加载所有所需数据，供所有特征类共享使用
    核心优化：所有数据仅请求一次，避免每个因子重复拉取数据
    """
    def __init__(
            self,
            trade_date: str,
            target_ts_codes: List[str] = None,
            sector_candidate_map: Dict[str, pd.DataFrame] = None,
            top3_sectors: List[str] = None
    ):
        self.trade_date = trade_date
        self.target_ts_codes = target_ts_codes or []
        self.sector_candidate_map = sector_candidate_map or {}
        self.top3_sectors = top3_sectors or []

        # 预加载的数据属性
        self.lookback_dates_5d: List[str] = []  # d0-d4 5个交易日
        self.lookback_dates_20d: List[str] = []  # 20日涨跌幅所需交易日
        self.daily_grouped: Dict[tuple, dict] = {}  # 日线数据，key=(ts_code, trade_date)
        self.minute_cache: Dict[tuple, pd.DataFrame] = {}  # 分钟线数据，key=(ts_code, trade_date)

        # 初始化时自动加载数据
        self._load_trade_dates()
        self._load_daily_data()
        self._load_minute_data()

    def _load_trade_dates(self):
        """加载所需交易日历，仅执行一次"""
        try:
            d_date = datetime.strptime(self.trade_date, "%Y-%m-%d")
            # 5天回溯期交易日
            lookback_start_5d = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            all_trade_dates_5d = get_trade_dates(lookback_start_5d, self.trade_date)
            self.lookback_dates_5d = all_trade_dates_5d[-5:]
            # 20日涨跌幅交易日
            lookback_start_20d = (d_date - timedelta(days=40)).strftime("%Y-%m-%d")
            all_trade_dates_20d = get_trade_dates(lookback_start_20d, self.trade_date)
            self.lookback_dates_20d = all_trade_dates_20d[-20:]
            logger.info(f"[数据容器] 交易日加载完成，5天回溯期：{self.lookback_dates_5d}")
        except Exception as e:
            logger.error(f"[数据容器] 交易日加载失败：{str(e)}")
            raise

    def _load_daily_data(self):
        """批量加载全量所需日线数据，仅执行一次"""
        try:
            # 合并所有需要的日期，一次性批量拉取
            all_need_dates = list(set(self.lookback_dates_5d + self.lookback_dates_20d))
            all_daily_df = pd.DataFrame()

            for date in all_need_dates:
                daily_df = get_daily_kline_data(trade_date=date, ts_code_list=self.target_ts_codes)
                if not daily_df.empty:
                    daily_df['trade_date'] = daily_df['trade_date'].astype(str)
                    all_daily_df = pd.concat([all_daily_df, daily_df], ignore_index=True)

            # 转为(ts_code, trade_date)为key的字典，O(1)查找
            self.daily_grouped = all_daily_df.groupby(["ts_code", "trade_date"]).first().to_dict(orient="index")
            logger.info(f"[数据容器] 日线数据加载完成，共{len(self.daily_grouped)}条记录")
        except Exception as e:
            logger.error(f"[数据容器] 日线数据加载失败：{str(e)}")
            raise

    def _load_minute_data(self):
        """批量加载全量所需分钟线数据，仅执行一次"""
        try:
            for ts_code in self.target_ts_codes:
                for date in self.lookback_dates_5d:
                    key = (ts_code, date)
                    self.minute_cache[key] = data_cleaner.get_kline_min_by_stock_date(ts_code, date)
            logger.info(f"[数据容器] 分钟线数据加载完成，共{len(self.minute_cache)}条记录")
        except Exception as e:
            logger.warning(f"[数据容器] 分钟线数据加载异常：{str(e)[:50]}")