import numpy as np
import pandas as pd
from features.base_feature import BaseFeature


class VWAPFeature(BaseFeature):

    """
    VWAP相关因子

    输出：

        vwap_deviation
        vwap_cross_count
    """

    def compute(self, stock_list, trade_date, market_data):

        minute = market_data["minute"]

        result = []

        for stock in stock_list:

            df = minute[minute["stock_code"] == stock]

            if df.empty:
                continue

            close = df["close"].values
            volume = df["volume"].values

            cum_vol = np.cumsum(volume)
            cum_amt = np.cumsum(close * volume)

            vwap = cum_amt / (cum_vol + 1e-6)

            deviation = np.mean(np.abs(close - vwap) / (vwap + 1e-6))

            cross = np.sum(np.diff(np.sign(close - vwap)) != 0)

            result.append({

                "stock_code": stock,
                "trade_date": trade_date,

                "vwap_deviation": deviation,
                "vwap_cross_count": cross
            })

        return pd.DataFrame(result)