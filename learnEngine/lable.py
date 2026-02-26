# learnEngine/label.py
import pandas as pd


class LabelGenerator:
    """
    标签：打板后次日开盘收益是否为正
    完全对接你的打板策略
    """

    def __init__(self, hold_days=1):
        self.hold_days = hold_days

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入：df 必须有 ts_code, trade_date, close, next_open
        输出：增加 label 列
        """
        df = df.sort_values(["ts_code", "trade_date"]).copy()

        # 计算收益
        df["next_return"] = df["next_open"] / df["close"] - 1

        # 标签：1=赚钱，0=不赚
        df["label"] = (df["next_return"] > 0).astype(int)

        return df