import pandas as pd
import math
from features.base_feature import BaseFeature


class SEIFeature(BaseFeature):

    """
    市场情绪指数

    综合：

    - 涨跌
    - gap
    - vwap
    - trend
    - candle
    """

    def compute(self, stock_list, trade_date, market_data):

        daily = market_data["daily"]

        result = []

        for _, row in daily.iterrows():

            ret = row["pct_chg"]

            base_score = 50 + 50 * math.tanh(10 * ret)

            result.append({

                "stock_code": row["stock_code"],
                "trade_date": trade_date,

                "sei": base_score
            })

        return pd.DataFrame(result)