import pandas as pd
from features.base_feature import BaseFeature


class CandleFeature(BaseFeature):

    """
    K线结构因子

    输出：
        candle_type
        true_candle_type

    定义：

    阳线：
        close > open

    阴线：
        close < open

    真阳：
        close > open 且 close > pre_close

    假阳：
        close > open 且 close <= pre_close

    真阴：
        close < open 且 close < pre_close

    假阴：
        close < open 且 close >= pre_close
    """

    def compute(self, stock_list, trade_date, market_data):

        daily = market_data["daily"]

        df = daily[daily["stock_code"].isin(stock_list)]

        result = []

        for _, row in df.iterrows():

            open_p = row["open"]
            close_p = row["close"]
            pre_close = row["pre_close"]

            if close_p > open_p:
                candle = 1
            else:
                candle = -1

            if close_p > open_p and close_p > pre_close:
                true_candle = 2

            elif close_p > open_p:
                true_candle = 1

            elif close_p < open_p and close_p < pre_close:
                true_candle = -2

            else:
                true_candle = -1

            result.append({

                "stock_code": row["stock_code"],
                "trade_date": trade_date,

                "candle_type": candle,
                "true_candle_type": true_candle
            })

        return pd.DataFrame(result)