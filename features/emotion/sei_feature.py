"""
SEI/HDI情绪因子计算
因子说明：
- HDI：持股难度指数，衡量个股日内交易的波动、回撤、承接难度
- SEI：情绪强度指数，基于HDI叠加量价、涨跌、涨停结构等合成的个股情绪分，取值0-100
"""
import math
import numpy as np
import pandas as pd
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from utils.common_tools import calc_limit_up_price, calc_limit_down_price


@feature_registry.register("sei_emotion")
class SEIFeature(BaseFeature):
    """SEI/HDI情绪因子类"""
    feature_name = "sei_emotion"
    # 因子列名
    factor_columns = [
        "hdi_score", "sei_score", "ret_intra", "cpr", "gap_return",
        "trend_r2", "vwap_deviation", "candle_type", "break_times",
        "seal_times", "lift_times"
    ]

    def _calculate_minute_hdi(self, minute_df: pd.DataFrame, pre_close: float, up_limit: float, down_limit: float):
        """
        分时HDI（持股难度指数）计算
        :param minute_df: 个股当日分钟线数据，必须包含time/open/high/low/close/volume
        :param pre_close: 前一日收盘价
        :param up_limit: 当日涨停价
        :param down_limit: 当日跌停价
        :return: hdi_score(0-100), 原子因子字典
        """
        if minute_df.empty:
            return 50, {}

        # 按时间排序，向量化计算
        minute_df = minute_df.sort_values("time").reset_index(drop=True)
        close_arr = minute_df["close"].values
        open_arr = minute_df["open"].values
        high_arr = minute_df["high"].values
        low_arr = minute_df["low"].values
        volume_arr = minute_df["volume"].values

        # 基础价格指标
        day_open = open_arr[0]
        day_close = close_arr[-1]
        day_high = high_arr.max()
        day_low = low_arr.min()
        day_high = day_high + 1e-6 if day_high == day_low else day_high

        # ========== 原子因子计算 ==========
        # 日内振幅
        amp = (day_high - day_low) / pre_close
        # 日内最大回撤
        cummax_close = np.maximum.accumulate(close_arr)
        drawdown_arr = (cummax_close - close_arr) / (cummax_close + 1e-6)
        max_dd_intra = drawdown_arr.max()
        # 收盘相对高点回撤
        pullback_abs = abs((day_close - day_high) / (day_high + 1e-6))
        # 当日涨跌幅
        ret = (day_close / pre_close) - 1
        ret_abs = abs(ret)
        # VWAP及偏离度
        cum_volume = np.cumsum(volume_arr)
        cum_amount = np.cumsum(close_arr * volume_arr)
        vwap_arr = cum_amount / (cum_volume + 1e-6)
        vwap_deviation = np.mean(np.abs(close_arr - vwap_arr) / (vwap_arr + 1e-6))
        # VWAP穿越次数
        cross_sign = np.sign(close_arr - vwap_arr)
        cross_times = np.sum(np.abs(np.diff(cross_sign))) / 2
        cross_times_norm = min(cross_times / 20, 1)
        # 量价背离率
        ret_minute = np.diff(close_arr) / (close_arr[:-1] + 1e-6)
        vol_change = np.diff(volume_arr) / (volume_arr[:-1] + 1e-6)
        divergence_count = np.sum(
            ((ret_minute > 0) & (vol_change < 0)) | ((ret_minute < 0) & (vol_change > 0))
        )
        divergence_ratio = divergence_count / len(ret_minute) if len(ret_minute) > 0 else 0
        # 开盘缺口
        gap_return = (day_open / pre_close) - 1
        # 趋势拟合R²
        x = np.arange(len(close_arr))
        coef = np.polyfit(x, close_arr, 1)
        trend = coef[0] * x + coef[1]
        ss_res = np.sum((close_arr - trend) ** 2)
        ss_tot = np.sum((close_arr - np.mean(close_arr)) ** 2)
        trend_r2 = 1 - ss_res / (ss_tot + 1e-6)
        # CPR收盘价位置比
        cpr = (day_close - day_low) / (day_high - day_low)
        # K线类型：2=真阳 1=假阳 -1=假阴 -2=真阴
        if day_close > day_open and day_close > pre_close:
            candle_type = 2
        elif day_close > day_open:
            candle_type = 1
        elif day_close < day_open and day_close < pre_close:
            candle_type = -2
        else:
            candle_type = -1
        # 涨跌停触碰/开板次数
        touch_up_limit = high_arr >= up_limit - 0.01
        break_limit = np.diff(touch_up_limit.astype(int)) == -1
        break_times = np.sum(break_limit)
        seal_limit = np.diff(touch_up_limit.astype(int)) == 1
        seal_times = np.sum(seal_limit)
        touch_down_limit = low_arr <= down_limit + 0.01
        lift_limit = np.diff(touch_down_limit.astype(int)) == -1
        lift_times = np.sum(lift_limit)

        # ========== HDI合成 ==========
        hdi_raw = (
                0.25 * amp
                + 0.30 * max_dd_intra
                + 0.20 * pullback_abs
                + 0.10 * ret_abs
                + 0.10 * cross_times_norm
                + 0.05 * divergence_ratio
        )
        hdi_score = max(min(hdi_raw * 100, 100), 0)

        # 原子因子返回
        factors = {
            "ret_intra": ret,
            "cpr": cpr,
            "gap_return": gap_return,
            "trend_r2": trend_r2,
            "vwap_deviation": vwap_deviation,
            "candle_type": candle_type,
            "break_times": break_times,
            "seal_times": seal_times,
            "lift_times": lift_times,
            "day_close": day_close
        }
        return round(hdi_score, 2), factors

    def _calculate_minute_sei(self, minute_df: pd.DataFrame, pre_close: float, up_limit: float, down_limit: float):
        """
        分时SEI（情绪强度指数）计算，基于HDI合成
        :param minute_df: 个股当日分钟线数据
        :param pre_close: 前一日收盘价
        :param up_limit: 当日涨停价
        :param down_limit: 当日跌停价
        :return: sei_score 0-100
        """
        hdi_score, factors = self._calculate_minute_hdi(minute_df, pre_close, up_limit, down_limit)
        if not factors:
            return 50.0

        # 基础分：基于涨跌幅的tanh归一化
        ret = factors["ret_intra"]
        base_score = 50 + 50 * math.tanh(10 * ret)

        # 分项调整
        gap_adjust = factors["gap_return"] * 20
        trend_adjust = (factors["trend_r2"] - 0.5) * 15
        vwap_adjust = factors["vwap_deviation"] * 20
        candle_adjust = {2: 6, 1: 2, -1: -2, -2: -6}[factors["candle_type"]]

        # 涨跌停调整
        limit_adjust = 0
        if factors["break_times"] > 0:
            if abs(factors["day_close"] - up_limit) < 0.01:
                limit_adjust += min(factors["break_times"] * 2, 6)
            else:
                limit_adjust -= min(factors["break_times"] * 3, 10)

        # 最终SEI合成，限制0-100
        sei_score = base_score + gap_adjust + trend_adjust + vwap_adjust + candle_adjust + limit_adjust
        sei_score = max(min(sei_score, 100), 0)
        return round(sei_score, 2)

    def calculate(self, data_bundle: "FeatureDataBundle") -> tuple[pd.DataFrame, dict]:
        """
        统一SEI因子计算入口
        :param data_bundle: 预加载数据容器
        :return: 个股SEI特征DataFrame，因子字典
        """
        result_rows = []
        factor_dict = {}
        trade_date = data_bundle.trade_date
        target_codes = data_bundle.target_ts_codes
        daily_grouped = data_bundle.daily_grouped
        minute_cache = data_bundle.minute_cache

        # 遍历个股计算SEI因子
        for ts_code in target_codes:
            daily_key = (ts_code, trade_date)
            if daily_key not in daily_grouped:
                continue
            daily_row = daily_grouped[daily_key]
            pre_close = daily_row["pre_close"]

            # 从预加载缓存获取分钟线，无需重复请求
            minute_df = minute_cache.get(daily_key, pd.DataFrame())
            up_limit = calc_limit_up_price(ts_code, pre_close)
            down_limit = calc_limit_down_price(ts_code, pre_close)

            # 计算HDI和SEI
            hdi_score, factors = self._calculate_minute_hdi(minute_df, pre_close, up_limit, down_limit)
            sei_score = self._calculate_minute_sei(minute_df, pre_close, up_limit, down_limit)

            # 组装行数据
            row_data = {
                "stock_code": ts_code,
                "trade_date": trade_date,
                "hdi_score": hdi_score,
                "sei_score": sei_score,
                **factors
            }
            result_rows.append(row_data)

        feature_df = pd.DataFrame(result_rows)
        self.logger.info(f"[SEI因子] {trade_date} 计算完成，有效样本数：{len(feature_df)}")
        return feature_df, factor_dict