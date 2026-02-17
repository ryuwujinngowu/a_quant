import pandas as pd
import numpy as np
# 复用你项目的日志
from utils.log_utils import logger
from config.config import DAILY_LIMIT_UP_RATE


# ========== 原有技术指标代码保留，以下为新增内容 ==========
def calc_limit_up_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    涨停判断特征（涨停策略核心依赖）
    输入：必须包含 close、pre_close 字段的日线DataFrame
    输出：新增 is_limit_up（当日是否涨停）、next_day_buy_signal（次日买入信号）字段
    """

    df = df.copy()

    if not all(col in df.columns for col in ["close", "pre_close"]):
        logger.error("计算涨停特征失败：缺少close/pre_close字段")
        return df

    # 计算当日是否涨停
    df["is_limit_up"] = np.where(
        df["close"] >= df["pre_close"] * (1 + DAILY_LIMIT_UP_RATE),
        1, 0
    )
    # 次日买入信号（避免未来函数，前一日涨停，当日才能买）
    df["next_day_buy_signal"] = df["is_limit_up"].shift(1)

    logger.debug("涨停特征计算完成")
    return df


def calc_ma_feature(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    均线特征计算（后续策略扩展用）
    输入：必须包含 close 字段的日线DataFrame
    输出：新增 ma_{window} 字段
    """
    df = df.copy()
    if "close" not in df.columns:
        logger.error("计算均线特征失败：缺少close字段")
        return df

    df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
    logger.debug(f"{window}日均线特征计算完成")
    return df


def calc_feature_batch(df: pd.DataFrame, feature_funcs: list) -> pd.DataFrame:
    """
    批量特征计算调度（统一入口，回测时一键计算所有需要的特征）
    :param df: 原始日线数据
    :param feature_funcs: 特征函数列表，如 [calc_limit_up_feature, calc_ma_feature]
    :return: 新增所有特征后的DataFrame
    """
    result_df = df.copy()
    for func in feature_funcs:
        result_df = func(result_df)
    # 剔除特征计算产生的空值
    result_df = result_df.dropna()
    logger.info(f"批量特征计算完成，最终数据行数：{len(result_df)}")
    return result_df