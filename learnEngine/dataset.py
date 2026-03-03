# learnEngine/dataset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple

from utils.log_utils import logger


def load_and_process_dataset(data_path: str = "mock_data_generator.py") -> Tuple[
    pd.DataFrame, pd.Series, list]:
    """
    加载并预处理训练数据集
    :param data_path: 训练数据路径
    :return: (特征X, 标签y, 特征列名列表)
    """
    # 1. 加载数据
    try:
        train_df = pd.read_csv(data_path)
        logger.info(f"数据集加载成功，共{len(train_df)}行数据")
    except Exception as e:
        logger.error(f"数据集加载失败：{e}")
        raise

    # 2. 拆分特征和标签
    # 标签列是label，其他非特征列要去掉
    drop_cols = ["trade_date", "ts_code", "label"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X = train_df[feature_cols]  # 特征：所有因子
    y = train_df["label"]  # 标签：模型要学习的目标

    logger.info(f"特征列总数：{len(feature_cols)}")
    logger.info(f"正样本占比：{y.mean():.2%}（赚钱的样本比例）")
    return X, y, feature_cols


def split_time_series_dataset(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2) -> tuple:
    """
    【新手必看】时间序列数据集划分，绝对不能随机打乱！
    用前80%的历史数据训练，后20%的最新数据验证，防止未来数据泄露
    """
    split_idx = int(len(X) * (1 - test_ratio))
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    logger.info(f"训练集大小：{len(X_train)}行，验证集大小：{len(X_val)}行")
    return X_train, X_val, y_train, y_val