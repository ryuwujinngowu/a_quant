import datetime
import logging
from typing import Optional, List, Union

import pandas as pd

from data.data_cleaner import data_cleaner  # 导入数据清洗入库实例
from utils.db_utils import db
from utils.log_utils import logger

COMMON_MA_DAYS = [5, 10, 20, 60, 120, 250]  # 5日(周)、10日(双周)、20日(月)、60日(季)、120日(半年)、250日(年)


class TechnicalFeatures:
    """技术指标特征计算类（量化行业标准口径）"""

    def __init__(self):
        self.logger = logger or logging.getLogger(__name__)

    def _get_qfq_kline_data(
            self,
            ts_code: str,
            start_date: str,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        内部方法：获取并入库前复权K线数据（封装数据获取逻辑）
        :param ts_code: 股票代码（如600000.SH）
        :param start_date: 开始日期（YYYYMMDD）
        :param end_date: 结束日期（YYYYMMDD），不传默认到当日
        :return: 前复权K线DataFrame（空则返回空DF）
        """
        # 1. 先入库前复权数据（确保数据存在）
        self.logger.info(f"开始获取{ts_code}前复权K线数据（{start_date}~{end_date}）")
        affected_rows = data_cleaner.clean_and_insert_kline_day_qfq(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            table_name="kline_day_qfq"
        )
        sql = """
              SELECT ts_code, trade_date, open, high, low, close, volume, amount
              FROM kline_day_qfq
              WHERE ts_code = %s \
                AND trade_date BETWEEN %s \
                AND %s
              ORDER BY trade_date ASC \
              """
        # 处理默认结束日期
        if not end_date:
            end_date = pd.Timestamp.now().strftime("%Y%m%d")
        # 日期格式转换（数据库中trade_date是DATE类型，需匹配）
        start_date_format = pd.to_datetime(start_date, format="%Y%m%d").strftime("%Y-%m-%d")
        end_date_format = pd.to_datetime(end_date, format="%Y%m%d").strftime("%Y-%m-%d")
        # 从数据库查询数据
        kline_df = db.query(
            sql=sql,
            params=(ts_code, start_date_format, end_date_format),
            return_df=True
        )
        if kline_df.empty:
            self.logger.warning(f"{ts_code}在{start_date}~{end_date}范围内无前复权K线数据")
            return pd.DataFrame()

        # 确保收盘价为数值类型
        kline_df["close"] = pd.to_numeric(kline_df["close"], errors="coerce").fillna(0)
        return kline_df

    def calculate_ma(
            self,
            ts_code: str,
            start_date: str,
            end_date: Optional[str] = None,
            ma_days: Union[int, List[int]] = COMMON_MA_DAYS
    ) -> pd.DataFrame:
        """
        计算简单移动平均线（MA）- 量化行业核心口径（基于前复权收盘价）
        :param ts_code: 股票代码
        :param start_date: 开始日期（YYYYMMDD）
        :param end_date: 结束日期（YYYYMMDD），不传默认到当日
        :param ma_days: 均线天数，支持单个（如5）或列表（如[5,10,20]），默认行业通用口径
        :return: 包含均线的DataFrame（trade_date/ts_code/close/ma5/ma10...）
        """
        start_date_dt = datetime.datetime.strptime(start_date, "%Y%m%d").date()

        # 2. 处理ma_days，确保为列表并取最大值
        if isinstance(ma_days, int):
            ma_days = [ma_days]  # 转为列表统一处理
        max_ma_day = max(ma_days)  # 提取最大均线天数
        new_start_date_dt = start_date_dt - datetime.timedelta(days=max_ma_day*2)
        new_start_date = new_start_date_dt.strftime("%Y%m%d")
        # 校验均线天数合理性
        invalid_days = [d for d in ma_days if d < 1]
        if invalid_days:
            self.logger.error(f"无效均线天数：{invalid_days}，天数需≥1")
            return pd.DataFrame()

        # 2. 获取前复权K线数据
        kline_df = self._get_qfq_kline_data(ts_code, new_start_date, end_date)
        if kline_df.empty:
            return pd.DataFrame()

        # 3. 计算指定天数的均线（基于收盘价，行业标准）
        result_df = kline_df.copy()
        for day in ma_days:
            col_name = f"ma{day}"
            # 简单移动平均：rolling(window=day).mean()，不足天数填充NaN（行业惯例）
            result_df[col_name] = result_df["close"].rolling(window=day, min_periods=1).mean()
            # 保留4位小数（与行情软件精度一致）
            result_df[col_name] = result_df[col_name].round(4)

        # 4. 整理返回字段
        return_cols = ["ts_code", "trade_date", "close"] + [f"ma{day}" for day in ma_days]
        result_df = result_df[return_cols].sort_values("trade_date", ascending=True)
        # ========== 4. 筛选：仅保留用户原始日期范围的数据 ==========
        # 先把trade_date转成日期对象，方便筛选
        result_df['trade_date_dt'] = pd.to_datetime(result_df['trade_date'], format="%Y%m%d").dt.date
        # 筛选条件：trade_date在原始start_date和end_date之间（包含边界）
        if not end_date:
            end_date =  datetime.datetime.strptime(pd.Timestamp.now().strftime("%Y%m%d"), "%Y%m%d").date()
        else:
            end_date =  datetime.datetime.strptime(end_date, "%Y%m%d").date()
        filter_mask = (result_df['trade_date_dt'] >= start_date_dt) & \
                      (result_df['trade_date_dt'] <= end_date)
        result_df = result_df[filter_mask].drop(columns=['trade_date_dt'])  # 删除临时列

        # ========== 5. 整理返回格式 ==========
        return_cols = ["ts_code", "trade_date", "close"] + [f"ma{day}" for day in ma_days]
        result_df = result_df[return_cols].sort_values("trade_date", ascending=True).reset_index(drop=True)
        self.logger.info(f"{ts_code}均线计算完成，覆盖天数：{ma_days}，数据行数：{len(result_df)}")
        return result_df



# 全局实例（供策略模块调用）
technical_features = TechnicalFeatures()

# 测试代码（极简风格，与项目测试逻辑对齐）
if __name__ == "__main__":
    """均线计算测试（行业通用口径）"""
    # 1. 测试简单移动平均（MA）
    ma_result = technical_features.calculate_ma(
        ts_code="300308.SZ",
        start_date="20230105",
        ma_days=[5, 10, 20, 60]  # 测试5/10/20日均线
    )
    if not ma_result.empty:
        logger.info("=====前复权均线数据（MA5/MA10/MA20） =====")
        logger.info(ma_result)  # 打印最后5行数据

    # 2. 测试指数移动平均（EMA）
    # ema_result = technical_features.calculate_ema(
    #     ts_code="600000.SH",
    #     start_date="20250101",
    #     end_date="20250131",
    #     ema_days=5
    # )
    # if not ema_result.empty:
    #     logger.info("===== 600000.SH 前复权EMA5数据 =====")
    #     logger.info(ema_result.tail(5))