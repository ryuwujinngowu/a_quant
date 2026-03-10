import numpy as np
import pandas as pd
from features.base_feature import BaseFeature


class TrendFeature(BaseFeature):

    """
    趋势稳定性

    使用 R² 衡量
    """

    def compute(self, stock_list, trade_date, market_data):

        minute = market_data["minute"]

        result = []

        for stock in stock_list:

            df = minute[minute["stock_code"] == stock]

            if df.empty:
                continue

            close = df["close"].values

            x = np.arange(len(close))

            coef = np.polyfit(x, close, 1)

            trend = coef[0] * x + coef[1]

            ss_res = np.sum((close - trend) ** 2)
            ss_tot = np.sum((close - close.mean()) ** 2)

            r2 = 1 - ss_res / (ss_tot + 1e-6)

            result.append({

                "stock_code": stock,
                "trade_date": trade_date,

                "trend_r2": r2
            })

        return pd.DataFrame(result)