# learnEngine/dataset.py
import pandas as pd


class Dataset:
    """
    负责：
    1. 把特征 + 标签拼在一起
    2. 按时间划分训练集/测试集
    """

    def __init__(self, test_start_date: str):
        self.test_start_date = test_start_date  # 例："2025-01-01"

    def split_train_test(self, df: pd.DataFrame):
        df = df.sort_values("trade_date")

        train = df[df["trade_date"] < self.test_start_date]
        test = df[df["trade_date"] >= self.test_start_date]

        return train, test