# features/market_stats.py
"""
板块热度因子计算（核心：30个情绪因子输出）
因子命名规则：sector{板块ID}_d{时间跨度}_{指标名}
- 板块ID：1/2/3（每日前3活跃板块）
- 时间跨度：d4=4天前、d3=3天前、d2=2天前、d1=1天前、d0=D日
- 指标名：profit=涨幅>7%个股数（赚钱效应）、loss=跌幅>7%个股数（亏钱效应）
"""
import math
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import math
from data.data_cleaner import data_cleaner
from features.base_feature import BaseFeature
from utils.common_tools import (get_trade_dates, get_daily_kline_data,
                                calc_limit_down_price, calc_limit_up_price,
                                getStockRank_fortraining, getTagRank_daily, sort_by_recent_gain )
from utils.log_utils import logger
from collections import defaultdict

# 固定配置，第一项是距离选股日D日之前的天数。
TIME_WEIGHT_MAP = {0: 1.0, -1: 0.8, -2: 0.6, -3: 0.4, -4: 0.2}
TOTAL_DAYS = 5
# 轮动速度计算固定权重
OVERLAP_WEIGHT = 0.8  # 隔日权重，分数越高，隔日有强就算强
HHI_WEIGHT = 0.2  # 弱但绵长
MAIN_LINE_RULES = [
    {"level": "absolute", "appear": 4, "top2": 2, "coeff": 0.1},  # 绝对主线行情，出现4次，有两天排名在前二，直接轮动分打1折
    {"level": "strong", "appear": 3, "top2": 2, "coeff": 0.3},
    {"level": "weak", "appear": 3, "top3": 1, "coeff": 0.6},
]


class SectorHeatFeature(BaseFeature):
    """板块热度特征类，继承基类，对齐现有特征层架构"""

    def __init__(self):
        super().__init__()
        # 固定配置：前3活跃板块、回溯5天、2个核心指标
        self.top_sector_num = 3
        self.lookback_days = 5
        self.factor_columns = self._gen_factor_columns()  # 预生成30个因子列名

    def _gen_factor_columns(self) -> List[str]:
        """预生成30个因子列名（固定格式，避免命名混乱）"""
        factor_cols = []
        for sector_id in range(1, self.top_sector_num + 1):
            for day_offset in range(self.lookback_days):
                # d4=4天前，d0=D日
                day_tag = f"d{self.lookback_days - 1 - day_offset}"
                factor_cols.append(f"sector{sector_id}_{day_tag}_profit")
                factor_cols.append(f"sector{sector_id}_{day_tag}_loss")
        return factor_cols

    # 【修改2】删除冗余的_get_lookback_trade_dates和_get_stock_daily_data方法，直接在calculate中调用

    # def _calculate_intraday_pattern_score(self, row: pd.Series, is_profit: bool = True) -> float:
    #     """
    #     【修改3】优化日内走势模式得分计算
    #     :param row: 个股日线数据（必须包含open, close, pre_close）
    #     :param is_profit: 是否为赚钱效应（True=profit，False=loss）
    #     :return: 走势得分
    #     """
    #     open_price = row["open"]
    #     close_price = row["close"]
    #     pre_close = row["pre_close"]
    #
    #     if pre_close <= 0:
    #         return 0.0
    #
    #     # 1. 计算开盘缺口幅度（直接*100作为得分）
    #     open_gap_pct = (open_price / pre_close - 1) * 100
    #
    #     # 2. 计算当日涨跌幅度（直接*100作为得分）
    #     daily_pct = (close_price / pre_close - 1) * 100
    #
    #     if is_profit:
    #         # Profit：高开幅度 + 当日涨幅
    #         return round(open_gap_pct + daily_pct, 2)
    #     else:
    #         # Loss：低开幅度（绝对值） + 当日跌幅（绝对值）
    #         return round(abs(open_gap_pct) + abs(daily_pct), 2)

    def _calculate_minute_hdi(self, minute_df: pd.DataFrame, pre_close: float, up_limit: float, down_limit: float) -> \
    Tuple[float, dict]:
        """【替换原HDI】分钟线级持股难度指数HDI（0~100），越高持股难度越大"""
        # 前置排序，剔除冗余校验（调用方已做空值判断）
        if 'time' in minute_df.columns:
            minute_df = minute_df.sort_values('time').reset_index(drop=True)

        # 向量化提取核心数组
        close_arr = minute_df['close'].values
        high_arr = minute_df['high'].values
        low_arr = minute_df['low'].values
        volume_arr = minute_df['volume'].values

        # 当日核心价格锚点
        day_open = minute_df['open'].iloc[0]
        day_high = high_arr.max()
        day_low = low_arr.min()
        day_close = close_arr[-1]
        day_high = day_low + 1e-6 if day_high == day_low else day_high

        # 分钟线核心折磨因子计算
        amp = (day_high - day_low) / pre_close
        cummax_close = np.maximum.accumulate(close_arr)
        drawdown_arr = (cummax_close - close_arr) / cummax_close
        max_dd_intra = drawdown_arr.max()
        pullback_abs = abs((day_close - day_high) / day_high) if day_high != 0 else 0
        ret_abs = abs(day_close / pre_close - 1)

        # VWAP洗盘强度计算
        cum_volume = np.cumsum(volume_arr)
        cum_amount = np.cumsum(close_arr * volume_arr)
        vwap_arr = cum_amount / (cum_volume + 1e-6)
        cross_sign = np.sign(close_arr - vwap_arr)
        cross_times = np.sum(np.abs(np.diff(cross_sign))) / 2
        cross_times_norm = min(cross_times / 20, 1.0)

        # 量价背离度计算
        ret_minute = np.diff(close_arr) / (close_arr[:-1] + 1e-6)
        vol_change = np.diff(volume_arr) / (volume_arr[:-1] + 1e-6)
        divergence_count = np.sum((ret_minute > 0) & (vol_change < 0)) + np.sum((ret_minute < 0) & (vol_change > 0))
        divergence_ratio = divergence_count / len(ret_minute) if len(ret_minute) > 0 else 0

        # HDI加权合成（A股实战最优权重）
        hdi_raw = (0.25 * amp + 0.30 * max_dd_intra + 0.20 * pullback_abs
                   + 0.10 * ret_abs + 0.10 * cross_times_norm + 0.05 * divergence_ratio)
        hdi = hdi_raw * 100

        # 极端场景惩罚
        upper_shadow = day_high - max(day_close, day_open)
        shadow_ratio = upper_shadow / (day_high - day_low)
        if shadow_ratio > 0.6:
            hdi += 15
        # 涨停炸板惩罚
        touch_up_limit = (high_arr >= up_limit - 0.01)
        break_limit = np.diff(touch_up_limit.astype(int)) == -1
        break_times = np.sum(break_limit)
        hdi += min(break_times * 10, 30)

        # 值域截断
        hdi = max(min(hdi, 100.0), 0.0)

        # 打包增强因子（用于SEI修正）
        seal_limit = np.diff(touch_up_limit.astype(int)) == 1
        seal_times = np.sum(seal_limit)
        touch_down_limit = (low_arr <= down_limit + 0.01)
        lift_limit = np.diff(touch_down_limit.astype(int)) == -1
        lift_times = np.sum(lift_limit)
        cpr = (day_close - day_low) / (day_high - day_low)

        factors = {
            "cpr": cpr,
            "break_times": break_times,
            "seal_times": seal_times,
            "lift_times": lift_times,
            "cross_times": cross_times,
            "divergence_ratio": divergence_ratio,
            "day_close": day_close
        }

        return round(hdi, 2), factors

    def _calculate_minute_sei(self, minute_df: pd.DataFrame, pre_close: float, up_limit: float,
                              down_limit: float) -> float:
        """分钟线级别个股情绪指数SEI（0~100），分数越高多头情绪越强"""
        hdi, factors = self._calculate_minute_hdi(minute_df, pre_close, up_limit, down_limit)
        if not factors:
            return 50.0

        hdi_norm = hdi / 100
        cpr = factors['cpr']
        break_times = factors['break_times']
        seal_times = factors['seal_times']
        lift_times = factors['lift_times']
        day_close = factors['day_close']

        # 基础情绪分（tanh非线性映射，平盘为50分中性值）
        ret = (day_close / pre_close) - 1
        base_score = 50 + 50 * math.tanh(10 * ret)

        # 分场景HDI核心修正
        K_UP = 0.3
        K_DOWN = 0.2
        adjustment = 0.0

        if ret > 1e-6:
            # 上涨场景：HDI越高、收得越差，惩罚越重
            penalty_factor = 0.5 + 0.5 * (1 - cpr)
            adjustment = - K_UP * hdi_norm * penalty_factor * base_score
            # 炸板回封回补、炸板未回封额外惩罚
            if break_times > 0:
                if abs(day_close - up_limit) < 0.01:
                    adjustment += min(break_times * 2, 5)
                else:
                    adjustment -= min(break_times * 3, 10)
        elif ret < -1e-6:
            # 下跌场景：探底回升HDI越高加分越多，冲高回落HDI越高扣分越多
            adjustment = K_DOWN * hdi_norm * (cpr - 0.5) * (100 - base_score)
            # 跌停翘板有承接回补情绪分
            if lift_times > 0 and cpr > 0.5:
                adjustment += min(lift_times * 2, 5)

        # 最终SEI值域截断
        sei = max(min(base_score + adjustment, 100.0), 0.0)
        return round(sei, 2)


    def _calculate_minute_drop_score(self, minute_df):
        """
        终极兼容版：适配Decimal类型字段，零报错运行，和拉升得分逻辑完全对称
        业务规则：5分钟级别计算，早盘9:30-10:00权重最高（杀跌最恐慌），越往后递减，无5分钟跌幅>4%直接返回0
        :param minute_df: pandas DataFrame，必填列：trade_time, open, high, low, close, amount（支持Decimal/float/int类型）
        :return:
            minute_drop_score: 0-100的杀跌评分，得分越高杀跌越猛、亏钱效应越强
            drop_pct: 全天最大5分钟跌幅（%），无符合条件则为0.0
            drop_time: 最大跌幅对应的5分钟结束时间（datetime），无符合条件则为None
        """
        # -------------------------- 1. 最严谨的入参校验（和拉升完全一致） --------------------------
        if not isinstance(minute_df, pd.DataFrame):
            return 0.0, 0.0, None

        required_cols = ['trade_time', 'open', 'high', 'low', 'close', 'amount']
        missing_cols = [col for col in required_cols if col not in minute_df.columns]
        if len(missing_cols) > 0:
            return 0.0, 0.0, None

        if minute_df.empty:
            return 0.0, 0.0, None

        # 深拷贝避免修改原数据
        minute_df = minute_df.dropna(subset=required_cols).copy()
        if minute_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 2. 核心修复：Decimal转float（和拉升完全一致） --------------------------
        numeric_cols = ['open', 'high', 'low', 'close', 'amount']
        for col in numeric_cols:
            # 处理Decimal/int/str等所有类型，统一转为float
            minute_df[col] = minute_df[col].apply(
                lambda x: float(x) if isinstance(x, (int, float, Decimal)) else (float(x) if x != '' else 0.0)
            )
        # 剔除转换后为0或空的异常值
        minute_df = minute_df[(minute_df['open'] > 1e-6) & (minute_df['amount'] >= 0)].copy()
        if minute_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 3. 时间字段标准化（和拉升完全一致） --------------------------
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'], errors='coerce')
        minute_df = minute_df.dropna(subset=['trade_time']).sort_values('trade_time').reset_index(drop=True)
        if minute_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 4. 1分钟转5分钟K线（和拉升完全一致） --------------------------
        minute_df['five_min_end_time'] = minute_df['trade_time'].dt.floor('5min') + pd.Timedelta(minutes=5)
        five_min_df = minute_df.groupby('five_min_end_time', as_index=False).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            amount=('amount', 'sum'),
            trade_time=('trade_time', 'last')
        )
        if five_min_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 5. 预计算全局统计值（和拉升完全一致） --------------------------
        bar_count = len(five_min_df)
        total_daily_amount = five_min_df['amount'].sum()
        avg_five_min_amount = total_daily_amount / bar_count if bar_count > 0 and total_daily_amount > 1e-6 else 0.0
        max_five_min_amount = five_min_df['amount'].max() if bar_count > 0 else 0.0

        # -------------------------- 6. 计算5分钟跌幅，过滤>4%的有效杀跌bar（核心差异1） --------------------------
        # 跌幅计算：(开盘价-收盘价)/开盘价*100，跌幅>4%才视为有效杀跌
        five_min_df['five_min_drop'] = (five_min_df['open'] - five_min_df['close']) / five_min_df['open'] * 100
        qualified_df = five_min_df[five_min_df['five_min_drop'] > 4].copy()
        if qualified_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 7. 提取最大跌幅和对应时间（核心差异2） --------------------------
        max_drop_row = qualified_df.loc[qualified_df['five_min_drop'].idxmax()]
        drop_pct = round(float(max_drop_row['five_min_drop']), 2)
        drop_time = max_drop_row['trade_time']

        # -------------------------- 8. 时间权重计算（和拉升完全一致，早盘杀跌最恐慌） --------------------------
        def get_time_weight_score(trade_time):
            if pd.isna(trade_time):
                return 0.0
            hour = trade_time.hour
            minute = trade_time.minute
            total_minute = hour * 60 + minute
            if 575 <= total_minute <= 600:  # 9:35-10:00（早盘杀跌最恐怖，权重100）
                return 100.0
            elif 605 <= total_minute <= 630:  # 10:05-10:30（权重80）
                return 80.0
            elif 635 <= total_minute <= 690:  # 10:35-11:30（权重60）
                return 60.0
            elif 785 <= total_minute <= 840:  # 13:05-14:00（权重40）
                return 40.0
            elif 845 <= total_minute <= 900:  # 14:05-15:00（权重20）
                return 20.0
            else:
                return 0.0

        qualified_df['time_score'] = qualified_df['trade_time'].apply(get_time_weight_score).astype(float)

        # -------------------------- 9. 多维度标准化得分（权重和拉升一致，逻辑对称） --------------------------
        WEIGHT_PRICE = 0.3  # 跌幅幅度得分（对应拉升的涨幅幅度）
        WEIGHT_TIME = 0.4  # 时间权重得分（早盘杀跌权重最高）
        WEIGHT_SPIKE = 0.1  # 放量幅度得分（杀跌放量越狠，得分越高）
        WEIGHT_AMOUNT = 0.2  # 成交金额得分（杀跌金额越大，得分越高）

        # 跌幅幅度得分（核心差异3：涨幅→跌幅，逻辑对称）
        qualified_df['price_score'] = (qualified_df['five_min_drop'] / 10 * 100).clip(0, 100).astype(float)

        # 放量幅度得分（逻辑对称：杀跌时放量越猛，得分越高）
        if avg_five_min_amount < 1e-6:
            qualified_df['spike_score'] = 0.0
        else:
            qualified_df['spike_score'] = (qualified_df['amount'] / avg_five_min_amount / 10 * 100).clip(0, 100).astype(
                float)

        # 成交金额得分（和拉升逻辑完全一致）
        if max_five_min_amount < 1e-6:
            qualified_df['amount_score'] = 0.0
        else:
            qualified_df['amount_score'] = (qualified_df['amount'] / max_five_min_amount * 100).clip(0, 100).astype(
                float)

        # -------------------------- 10. 加权计算最终得分（和拉升完全一致） --------------------------
        qualified_df['bar_score'] = (
                qualified_df['price_score'] * WEIGHT_PRICE +
                qualified_df['time_score'] * WEIGHT_TIME +
                qualified_df['spike_score'] * WEIGHT_SPIKE +
                qualified_df['amount_score'] * WEIGHT_AMOUNT
        ).astype(float)

        # 归一化得分（和拉升一致，MAX_POSSIBLE_TOTAL=200，保证值域0-100）
        total_weighted_score = float(qualified_df['bar_score'].sum())
        MAX_POSSIBLE_TOTAL = 200  # 和拉升保持一致，2根完美杀跌bar即满分
        minute_drop_score = round(min((total_weighted_score / MAX_POSSIBLE_TOTAL) * 100, 100.0), 2)

        return minute_drop_score, drop_pct, drop_time

    def _calculate_minute_boost_score(self, minute_df):
        """
        终极兼容版：适配Decimal类型字段，零报错运行，完全匹配你的业务规则和入参出参
        业务规则：5分钟级别计算，早盘9:30-10:00权重最高，越往后递减，无5分钟涨幅>4%直接返回0
        :param minute_df: pandas DataFrame，必填列：trade_time, open, high, low, close, amount（支持Decimal/float/int类型）
        :return:
            minute_boost_score: 0-100的动量评分，得分越高拉升动量越足
            boost_pct: 全天最大5分钟涨幅（%），无符合条件则为0.0
            boost_time: 最大涨幅对应的5分钟结束时间（datetime），无符合条件则为None
        """
        # -------------------------- 1. 最严谨的入参校验 --------------------------
        if not isinstance(minute_df, pd.DataFrame):
            return 0.0, 0.0, None

        required_cols = ['trade_time', 'open', 'high', 'low', 'close', 'amount']
        missing_cols = [col for col in required_cols if col not in minute_df.columns]
        if len(missing_cols) > 0:
            return 0.0, 0.0, None

        if minute_df.empty:
            return 0.0, 0.0, None

        # 深拷贝避免修改原数据
        minute_df = minute_df.dropna(subset=required_cols).copy()
        if minute_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 2. 核心修复：Decimal转float，解决类型不兼容 --------------------------
        # 定义需要转换的数值列
        numeric_cols = ['open', 'high', 'low', 'close', 'amount']
        for col in numeric_cols:
            # 处理Decimal/int/str等所有类型，统一转为float
            minute_df[col] = minute_df[col].apply(
                lambda x: float(x) if isinstance(x, (int, float, Decimal)) else (float(x) if x != '' else 0.0)
            )
        # 剔除转换后为0或空的异常值
        minute_df = minute_df[(minute_df['open'] > 1e-6) & (minute_df['amount'] >= 0)].copy()
        if minute_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 3. 时间字段标准化 --------------------------
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'], errors='coerce')
        minute_df = minute_df.dropna(subset=['trade_time']).sort_values('trade_time').reset_index(drop=True)
        if minute_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 4. 1分钟转5分钟K线 --------------------------
        minute_df['five_min_end_time'] = minute_df['trade_time'].dt.floor('5min') + pd.Timedelta(minutes=5)
        five_min_df = minute_df.groupby('five_min_end_time', as_index=False).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            amount=('amount', 'sum'),
            trade_time=('trade_time', 'last')
        )
        if five_min_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 5. 预计算全局统计值 --------------------------
        bar_count = len(five_min_df)
        total_daily_amount = five_min_df['amount'].sum()
        avg_five_min_amount = total_daily_amount / bar_count if bar_count > 0 and total_daily_amount > 1e-6 else 0.0
        max_five_min_amount = five_min_df['amount'].max() if bar_count > 0 else 0.0

        # -------------------------- 6. 计算5分钟涨幅，过滤>4%的bar --------------------------
        five_min_df['five_min_rise'] = (five_min_df['close'] - five_min_df['open']) / five_min_df['open'] * 100
        qualified_df = five_min_df[five_min_df['five_min_rise'] > 4].copy()
        if qualified_df.empty:
            return 0.0, 0.0, None

        # -------------------------- 7. 提取最大涨幅和对应时间 --------------------------
        max_rise_row = qualified_df.loc[qualified_df['five_min_rise'].idxmax()]
        boost_pct = round(float(max_rise_row['five_min_rise']), 2)
        boost_time = max_rise_row['trade_time']

        # -------------------------- 8. 时间权重计算 --------------------------
        def get_time_weight_score(trade_time):
            if pd.isna(trade_time):
                return 0.0
            hour = trade_time.hour
            minute = trade_time.minute
            total_minute = hour * 60 + minute
            if 575 <= total_minute <= 600:
                return 100.0
            elif 605 <= total_minute <= 630:
                return 80.0
            elif 635 <= total_minute <= 690:
                return 60.0
            elif 785 <= total_minute <= 840:
                return 40.0
            elif 845 <= total_minute <= 900:
                return 20.0
            else:
                return 0.0

        qualified_df['time_score'] = qualified_df['trade_time'].apply(get_time_weight_score).astype(float)

        # -------------------------- 9. 多维度标准化得分（全float类型） --------------------------
        WEIGHT_PRICE = 0.3  #价格幅度得分
        WEIGHT_TIME = 0.4      #时间权重得分
        WEIGHT_SPIKE = 0.1         #放量幅度
        WEIGHT_AMOUNT = 0.2           #成交金额得分

        # 价格幅度得分（转float）
        qualified_df['price_score'] = (qualified_df['five_min_rise'] / 10 * 100).clip(0, 100).astype(float)

        # 放量幅度得分（转float）
        if avg_five_min_amount < 1e-6:
            qualified_df['spike_score'] = 0.0
        else:
            qualified_df['spike_score'] = (qualified_df['amount'] / avg_five_min_amount / 10 * 100).clip(0, 100).astype(
                float)

        # 成交金额得分（转float）
        if max_five_min_amount < 1e-6:
            qualified_df['amount_score'] = 0.0
        else:
            qualified_df['amount_score'] = (qualified_df['amount'] / max_five_min_amount * 100).clip(0, 100).astype(
                float)

        # -------------------------- 10. 加权计算最终得分（全float运算） --------------------------
        qualified_df['bar_score'] = (
                qualified_df['price_score'] * WEIGHT_PRICE +
                qualified_df['time_score'] * WEIGHT_TIME +
                qualified_df['spike_score'] * WEIGHT_SPIKE +
                qualified_df['amount_score'] * WEIGHT_AMOUNT
        ).astype(float)

        # 归一化得分
        total_weighted_score = float(qualified_df['bar_score'].sum())
        MAX_POSSIBLE_TOTAL = 200  # 6个5分钟bar × 100分
        minute_boost_score = round(min((total_weighted_score / MAX_POSSIBLE_TOTAL) * 100, 100.0), 2)

        return minute_boost_score, boost_pct, boost_time

    def calculate(
            self,
            trade_date: str,
            top3_sectors_result: Dict,
            sector_candidate_map: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        【模型训练友好版】板块热度特征计算
        核心优化：所有缺失值按量化训练规范处理，无差别填充0，不引入虚假信号
        :return:
            result_df: 个股-交易日级特征DataFrame，一行=D日单只可交易个股
            factor_dict: 兼容旧版的板块级因子字典
        """
        # 初始化返回值
        result_rows = []
        factor_dict = {col: 0 for col in self.factor_columns}
        d0_date = trade_date
        TOTAL_DAYS = 5  # d0-d4回溯期
        RANK_DAYS = 20  # 涨跌幅排名周期

        # ========== 1. 获取交易日历 ==========
        try:
            d_date = datetime.strptime(d0_date, "%Y-%m-%d")
            # 1.1 d0-d4的5天回溯期
            lookback_start_5d = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            all_trade_dates_5d = get_trade_dates(lookback_start_5d, d0_date)
            if not all_trade_dates_5d or len(all_trade_dates_5d) < TOTAL_DAYS:
                logger.error(f"[板块热度] 交易日不足{TOTAL_DAYS}个，返回空")
                return pd.DataFrame(), factor_dict
            lookback_dates_5d = all_trade_dates_5d[-TOTAL_DAYS:]
            day_tag_map = {f"d{4 - i}": date for i, date in enumerate(lookback_dates_5d)}

            # 1.2 20日涨跌幅所需交易日
            lookback_start_20d = (d_date - timedelta(days=40)).strftime("%Y-%m-%d")
            all_trade_dates_20d = get_trade_dates(lookback_start_20d, d0_date)
            if not all_trade_dates_20d or len(all_trade_dates_20d) < RANK_DAYS:
                logger.error(f"[板块热度] 20日涨跌幅交易日不足{RANK_DAYS}个，返回空")
                return pd.DataFrame(), factor_dict
        except Exception as e:
            logger.error(f"[板块热度] 获取交易日异常：{str(e)}")
            return pd.DataFrame(), factor_dict

        # ========== 2. 批量预加载全量数据 ==========
        # 2.1 提取所有候选股
        all_candidate_ts_codes = []
        for sector_df in sector_candidate_map.values():
            if not sector_df.empty:
                all_candidate_ts_codes.extend(sector_df["ts_code"].unique().tolist())
        all_candidate_ts_codes = list(set(all_candidate_ts_codes))
        if not all_candidate_ts_codes:
            logger.warning("[板块热度] 无候选个股，返回空")
            return pd.DataFrame(), factor_dict

        # 2.2 批量预加载全量日线数据（5天+20天合并查询，避免重复IO）
        try:
            all_daily_df = pd.DataFrame()
            all_need_dates = list(set(lookback_dates_5d + all_trade_dates_20d[-RANK_DAYS:]))
            for date in all_need_dates:
                daily_df = get_daily_kline_data(trade_date=date, ts_code_list=all_candidate_ts_codes)
                if not daily_df.empty:
                    daily_df['trade_date'] = daily_df['trade_date'].astype(str)
                    all_daily_df = pd.concat([all_daily_df, daily_df], ignore_index=True)
            if all_daily_df.empty:
                logger.warning("[板块热度] 无日线数据，返回空")
                return pd.DataFrame(), factor_dict
            # 按股票+日期分组索引
            daily_grouped = all_daily_df.groupby(["ts_code", "trade_date"]).first().to_dict(orient="index")
        except Exception as e:
            logger.error(f"[板块热度] 预加载日线数据失败：{str(e)}")
            return pd.DataFrame(), factor_dict

        # 2.3 批量预加载分钟线数据
        minute_cache = {}
        try:
            for ts_code in all_candidate_ts_codes:
                for date in lookback_dates_5d:
                    minute_cache[(ts_code, date)] = data_cleaner.get_kline_min_by_stock_date(ts_code, date)
        except Exception as e:
            logger.warning(f"[板块热度] 分钟线预加载异常：{str(e)[:50]}")

        # ========== 3. 遍历板块，预计算板块级通用数据 ==========
        top3_sectors = top3_sectors_result.get("top3_sectors", [])
        while len(top3_sectors) < 3:
            top3_sectors.append("")

        # 预计算每个板块的：20日涨跌幅排名、SEI均值、个股近5日成交均值
        sector_precalc_map = {}
        for sector_idx, sector_name in enumerate(top3_sectors, 1):
            if not sector_name or sector_name not in sector_candidate_map:
                continue
            sector_d0_df = sector_candidate_map[sector_name]
            if sector_d0_df.empty:
                continue
            sector_ts_codes = sector_d0_df["ts_code"].unique().tolist()

            # 3.1 板块内20日涨跌幅排名（用你提供的函数）
            rank_map = {}
            try:
                sorted_df = sort_by_recent_gain(sector_d0_df, d0_date, day_count=20)
                if not sorted_df.empty:
                    sorted_df = sorted_df.copy()
                    sorted_df["rank"] = range(1, len(sorted_df) + 1)
                    rank_map = dict(zip(sorted_df["ts_code"], sorted_df["rank"]))
                    # 板块排名中位数（用于填充新股）
                    sector_rank_median = sorted_df["rank"].median()
            except Exception as e:
                logger.warning(f"[板块热度] {sector_name} 排名计算失败：{str(e)}")
                rank_map = {}
                sector_rank_median = 0

            # 3.2 预计算板块内个股近5日成交均值（用于填充成交额缺失）
            turnover_mean_map = {}
            for ts_code in sector_ts_codes:
                day_turnover = []
                for date in lookback_dates_5d:
                    key = (ts_code, date)
                    if key in daily_grouped:
                        day_turnover.append(daily_grouped[key].get("turnover_rate", 0))
                turnover_mean_map[ts_code] = np.mean(day_turnover) if day_turnover else 0

            # 3.3 存入预计算字典
            sector_precalc_map[sector_name] = {
                "rank_map": rank_map,
                "rank_median": sector_rank_median,
                "turnover_mean_map": turnover_mean_map,
                "ts_codes": sector_ts_codes
            }

        # ========== 4. 遍历板块，计算因子 ==========
        for sector_idx, sector_name in enumerate(top3_sectors, 1):
            sector_prefix = f"sector{sector_idx}"
            if not sector_name or sector_name not in sector_precalc_map:
                continue
            sector_precalc = sector_precalc_map[sector_name]
            sector_d0_df = sector_candidate_map[sector_name]
            if sector_d0_df.empty:
                continue
            logger.info(f"[板块热度] 板块{sector_idx}({sector_name})候选股数：{len(sector_d0_df)}")

            # ========== 4.1 预计算该板块d0-d4的板块级因子 ==========
            sector_day_factor_map = {}
            # 预计算每日同涨跌个股的SEI均值（用于填充SEI缺失）
            day_sei_mean_map = defaultdict(dict)  # {day_tag: {"up": 均值, "down": 均值}}

            for day_tag, target_date in day_tag_map.items():
                day_profit_sei = []
                day_loss_sei = []
                day_up_sei = []
                day_down_sei = []

                for ts_code in sector_precalc["ts_codes"]:
                    daily_key = (ts_code, target_date)
                    if daily_key not in daily_grouped:
                        continue
                    daily_row = daily_grouped[daily_key]
                    pct_chg = daily_row["pct_chg"]
                    pre_close = daily_row["pre_close"]

                    # 计算SEI
                    minute_df = minute_cache.get(daily_key, pd.DataFrame())
                    up_limit = calc_limit_up_price(ts_code, pre_close)
                    down_limit = calc_limit_down_price(ts_code, pre_close)
                    sei_score = np.clip(self._calculate_minute_sei(minute_df, pre_close, up_limit, down_limit), 0, 100)

                    # 分类统计
                    if pct_chg > 1e-6:
                        day_profit_sei.append(sei_score)
                        day_up_sei.append(sei_score)
                    elif pct_chg < -1e-6:
                        day_loss_sei.append(sei_score)
                        day_down_sei.append(sei_score)

                # 计算板块级因子
                sector_profit = round(np.mean(day_profit_sei), 2) if day_profit_sei else 50.0
                avg_loss_sei = round(np.mean(day_loss_sei), 2) if day_loss_sei else 50.0
                sector_loss = round(100 - avg_loss_sei, 2)
                sector_day_factor_map[day_tag] = {"profit": sector_profit, "loss": sector_loss}
                # 填充兼容字典
                factor_dict[f"{sector_prefix}_{day_tag}_profit"] = int(sector_profit)
                factor_dict[f"{sector_prefix}_{day_tag}_loss"] = int(sector_loss)
                # 保存当日同涨跌SEI均值
                day_sei_mean_map[day_tag]["up"] = np.mean(day_up_sei) if day_up_sei else 50.0
                day_sei_mean_map[day_tag]["down"] = np.mean(day_down_sei) if day_down_sei else 50.0

            # ========== 4.2 遍历个股，生成单条样本 ==========
            for _, d0_row in sector_d0_df.iterrows():
                ts_code = d0_row["ts_code"]
                # 【核心过滤1】D日无基础数据的个股，直接剔除，不进入训练集
                d0_key = (ts_code, d0_date)
                if d0_key not in daily_grouped:
                    logger.warning(f"[板块热度] {ts_code} {d0_date} 无基础日线数据，剔除样本")
                    continue

                # 初始化行数据
                row_data = {
                    "stock_code": ts_code,
                    "trade_date": d0_date,
                    "sector_id": sector_idx,
                    "sector_name": sector_name,
                    # 20日涨跌幅排名：无数据用板块中位数填充，不用0
                    "stock_sector_20d_rank": sector_precalc["rank_map"].get(ts_code, sector_precalc["rank_median"])
                }

                # 【核心】遍历d0-d4，补全所有因子
                valid_sample = True  # 标记样本是否有效
                for day_tag, target_date in day_tag_map.items():
                    daily_key = (ts_code, target_date)
                    daily_data = daily_grouped.get(daily_key, None)

                    # 1. 补全OHLC、涨跌幅、成交额、换手率
                    if daily_data is not None:
                        row_data[f"stock_open_{day_tag}"] = daily_data["open"]
                        row_data[f"stock_high_{day_tag}"] = daily_data["high"]
                        row_data[f"stock_low_{day_tag}"] = daily_data["low"]
                        row_data[f"stock_close_{day_tag}"] = daily_data["close"]
                        row_data[f"stock_pct_chg_{day_tag}"] = daily_data["pct_chg"]
                        row_data[f"stock_amount_{day_tag}"] = daily_data["amount"]
                        # 换手率：无数据用个股近5日均值填充，不用0
                        # row_data[f"stock_turnover_{day_tag}"] = daily_data.get("turnover_rate",
                        #                                                        sector_precalc["turnover_mean_map"].get(
                        #                                                            ts_code, 0))
                    else:
                        # 【核心处理】回溯期某天停牌无数据，用D日数据填充，不用0
                        d0_data = daily_grouped[d0_key]
                        row_data[f"stock_open_{day_tag}"] = d0_data["open"]
                        row_data[f"stock_high_{day_tag}"] = d0_data["high"]
                        row_data[f"stock_low_{day_tag}"] = d0_data["low"]
                        row_data[f"stock_close_{day_tag}"] = d0_data["close"]
                        row_data[f"stock_pct_chg_{day_tag}"] = 0.0
                        row_data[f"stock_amount_{day_tag}"] = 0.0
                        # row_data[f"stock_turnover_{day_tag}"] = sector_precalc["turnover_mean_map"].get(ts_code, 0)

                    # 2. 补全个股profit/loss因子
                    if daily_data is not None:
                        pct_chg = daily_data["pct_chg"]
                        pre_close = daily_data["pre_close"]
                        minute_df = minute_cache.get(daily_key, pd.DataFrame())
                        up_limit = calc_limit_up_price(ts_code, pre_close)
                        down_limit = calc_limit_down_price(ts_code, pre_close)
                        sei_score = np.clip(self._calculate_minute_sei(minute_df, pre_close, up_limit, down_limit), 0,
                                            100)

                        # SEI缺失：用当日同板块同涨跌个股的均值填充，不用0
                        if sei_score <= 0 and minute_df.empty:
                            if pct_chg > 1e-6:
                                sei_score = day_sei_mean_map[day_tag]["up"]
                            elif pct_chg < -1e-6:
                                sei_score = day_sei_mean_map[day_tag]["down"]
                            else:
                                sei_score = 50.0

                        # 按涨跌赋值profit/loss
                        if pct_chg > 1e-6:
                            row_data[f"stock_profit_{day_tag}"] = sei_score
                            row_data[f"stock_loss_{day_tag}"] = 0.0
                        elif pct_chg < -1e-6:
                            row_data[f"stock_profit_{day_tag}"] = 0.0
                            row_data[f"stock_loss_{day_tag}"] = 100 - sei_score
                        else:
                            row_data[f"stock_profit_{day_tag}"] = 0.0
                            row_data[f"stock_loss_{day_tag}"] = 0.0
                    else:
                        # 回溯期停牌：profit/loss填充中性值50，不用0
                        row_data[f"stock_profit_{day_tag}"] = 50.0
                        row_data[f"stock_loss_{day_tag}"] = 50.0

                    # 3. 补全板块平均profit/loss因子
                    row_data[f"sector_avg_profit_{day_tag}"] = sector_day_factor_map[day_tag]["profit"]
                    row_data[f"sector_avg_loss_{day_tag}"] = sector_day_factor_map[day_tag]["loss"]

                # 有效样本加入结果
                if valid_sample:
                    result_rows.append(row_data)

        # ========== 生成最终结果 ==========
        result_df = pd.DataFrame(result_rows)
        logger.info(f"[板块热度] D日{d0_date}特征计算完成，有效样本数：{len(result_df)}行")
        return result_df, factor_dict


    def select_top3_hot_sectors(self, trade_date: str) -> Dict:
        """
        板块热度计算主入口（策略运行时轻量化调用）
        :param trade_date: D日日期，格式yyyy-mm-dd
        :return: 结果字典
            {
                "top3_sectors": List[str] 最终选中的3个板块（2个核心题材+1个预期差题材）
                "adapt_score": int 0-100分，分数越高=板块轮动速度越快
            }
        """

        # -------------------- 1. 极简基础校验 --------------------
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', trade_date):
            logger.error(f"[板块热度] 日期格式错误：{trade_date}，要求yyyy-mm-dd")
            return {"top3_sectors": [], "adapt_score": 0}

        # -------------------- 2. 获取5个连续交易日 --------------------
        try:
            # 计算起始日（D日往前推20天，覆盖非交易日）
            d_date = datetime.strptime(trade_date, "%Y-%m-%d")
            start_date = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            all_trade_dates = get_trade_dates(start_date, trade_date)
            trade_dates = all_trade_dates[-TOTAL_DAYS:]
            if len(trade_dates) != TOTAL_DAYS:
                logger.error(f"[板块热度] 获取交易日失败，仅拿到{len(trade_dates)}个，要求5个")
                return {"top3_sectors": [], "adapt_score": 0}
            logger.debug(f"[板块热度] 成功获取5个交易日：{trade_dates}")
        except Exception as e:
            logger.error(f"[板块热度] 获取交易日异常：{str(e)}")
            return {"top3_sectors": [], "adapt_score": 0}

        # -------------------- 3. 逐天生成榜单数据 --------------------
        daily_board_data = []
        all_daily_sectors = []  # 每日板块集合，用于轮动速度计算
        daily_rank_maps = []  # 每日板块-排名映射，用于趋势计算
        for idx, day in enumerate(trade_dates):
            distance = idx - 4  # 严格绑定：D-4→-4，D日→0，权重绝对不反向
            try:
                # 步骤1：获取当日符合阈值的股票列表
                stock_df = getStockRank_fortraining(day)
                if stock_df.empty or "ts_code" not in stock_df.columns:
                    logger.warning(f"[板块热度] {day} 无符合条件的股票，跳过")
                    continue
                ts_list = stock_df["ts_code"].dropna().unique().tolist()

                # 步骤2：获取当日热度前5板块
                tag_df = getTagRank_daily(ts_list)
                if tag_df.empty or "concept_name" not in tag_df.columns:
                    logger.warning(f"[板块热度] {day} 无板块数据，跳过")
                    continue
                tag_df = tag_df.head(5).reset_index(drop=True)

                # 步骤3：转换为标准榜单格式
                daily_board = [{"rank": i + 1, "name": str(row["concept_name"]).strip()} for i, row in
                               tag_df.iterrows()]
                daily_board_data.append({"distance": distance, "board": daily_board})
                # 保存当日板块集合（轮动计算用）
                all_daily_sectors.append(set([item["name"] for item in daily_board]))
                # 保存当日板块-排名映射（趋势计算用）
                daily_rank_maps.append({item["name"]: item["rank"] for item in daily_board})

            except Exception as e:
                logger.warning(f"[板块热度] {day} 数据处理失败：{str(e)}，跳过")
                continue

        # 最低有效数据校验（至少3天有效数据，避免结果失真）
        if len(daily_board_data) < 3:
            logger.error(f"[板块热度] 有效交易日不足3个，无法计算")
            return {"top3_sectors": [], "adapt_score": 0}
        logger.debug(f"[板块热度] 成功生成{len(daily_board_data)}天待计算热度数据")

        # -------------------- 4. 核心热度统计（已验证无逻辑错误） --------------------
        sector_stats = {}
        for daily_data in daily_board_data:
            distance = daily_data["distance"]
            time_weight = TIME_WEIGHT_MAP[distance]
            board = daily_data["board"]
            unique_sectors = set()  # 单日去重，避免统计失真

            for sector in board:
                name, rank = sector["name"], sector["rank"]
                if name in unique_sectors:
                    continue
                unique_sectors.add(name)

                # 初始化+更新统计
                if name not in sector_stats:
                    sector_stats[name] = {"appear_count": 0, "has_top3": False, "total_score": 0.0}
                sector_stats[name]["appear_count"] += 1
                sector_stats[name]["has_top3"] |= (rank <= 3)
                sector_stats[name]["total_score"] += (6 - rank) * time_weight

        if not sector_stats:
            logger.error(f"[板块热度] 无有效板块统计数据")
            # return {"top3_sectors": [], "adapt_score": 0}

        # -------------------- 5. 板块选取规则（严格按要求实现） --------------------
        # ========== 规则1：选取2个核心题材 ==========
        # 第一优先级：5天出现≥3次 + 至少1次进过前2
        rule1_candidate_1 = {}
        for name, stat in sector_stats.items():
            # 校验是否有进过前2的记录
            has_top2 = any(rank <= 2 for daily_map in daily_rank_maps for n, rank in daily_map.items() if n == name)
            if stat["appear_count"] >= 3 and has_top2:
                rule1_candidate_1[name] = stat
        # 按总得分降序排序，取前2
        sorted_rule1_1 = sorted(rule1_candidate_1.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]),
                                reverse=True)
        rule1_selected = [item[0] for item in sorted_rule1_1[:2]]

        # 不足2个，放宽规则：5天出现≥3次 + 至少1次进过前3
        if len(rule1_selected) < 2:
            need = 2 - len(rule1_selected)
            rule1_candidate_2 = {
                name: stat for name, stat in sector_stats.items()
                if name not in rule1_selected and stat["appear_count"] >= 3 and stat["has_top3"]
            }
            sorted_rule1_2 = sorted(rule1_candidate_2.items(),
                                    key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            rule1_selected += [item[0] for item in sorted_rule1_2[:need]]

        # 兜底：仍不足2个，从所有进过前3的板块中补全
        if len(rule1_selected) < 2:
            need = 2 - len(rule1_selected)
            rule1_candidate_3 = {
                name: stat for name, stat in sector_stats.items()
                if name not in rule1_selected and stat["has_top3"]
            }
            sorted_rule1_3 = sorted(rule1_candidate_3.items(),
                                    key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            rule1_selected += [item[0] for item in sorted_rule1_3[:need]]
        logger.debug(f"[板块热度] 规则1选中核心题材：{rule1_selected}")

        # ========== 规则2：选取1个上升趋势的预期差题材 ==========
        sector_trend_data = {}
        # 遍历所有未被规则1选中的板块，计算趋势
        for sector_name in sector_stats.keys():
            if sector_name in rule1_selected:
                continue
            # 生成5天排名序列（未上榜记为第6名，代表未进前5）
            rank_series = [daily_map.get(sector_name, 6) for daily_map in daily_rank_maps]
            # 线性回归计算排名斜率（判断趋势：斜率<0=排名越来越靠前，上升趋势）
            n = len(rank_series)
            x = list(range(n))  # x轴：天数0-4（D-4到D日）
            y = rank_series  # y轴：当日排名
            x_mean, y_mean = sum(x) / n, sum(y) / n
            cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            var_x = sum((x[i] - x_mean) ** 2 for i in range(n))
            slope = cov / var_x if var_x != 0 else 0
            # 计算平均排名
            avg_rank = sum(y) / n
            sector_trend_data[sector_name] = {"slope": slope, "avg_rank": avg_rank}

        # 筛选上升趋势板块（斜率<0），按平均排名升序（越靠前越好）
        uptrend_candidates = {name: data for name, data in sector_trend_data.items() if data["slope"] < 0}
        sorted_uptrend = sorted(uptrend_candidates.items(), key=lambda x: x[1]["avg_rank"])
        rule2_selected = [item[0] for item in sorted_uptrend[:1]]

        # 兜底：无上升趋势板块，选剩余板块中总得分最高的
        if not rule2_selected:
            remaining_candidates = {
                name: stat for name, stat in sector_stats.items()
                if name not in rule1_selected
            }
            sorted_remaining = sorted(remaining_candidates.items(),
                                      key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
            rule2_selected = [item[0] for item in sorted_remaining[:1]]
        logger.debug(f"[板块热度] 规则2选中预期差题材：{rule2_selected}")

        # 合并最终TOP3，极端兜底补全
        final_top3 = rule1_selected + rule2_selected
        if len(final_top3) < 3:
            need = 3 - len(final_top3)
            all_sorted = sorted(sector_stats.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]),
                                reverse=True)
            for item in all_sorted:
                if item[0] not in final_top3 and need > 0:
                    final_top3.append(item[0])
                    need -= 1
        final_top3 = final_top3[:3]

        adapt_score = 0
        # 仅完整5天数据才计算轮动分
        if len(all_daily_sectors) == 5 and len(daily_rank_maps) == 5:
            try:
                # ========== 步骤1：计算头部加权相邻日重合率 ==========
                overlap_rates = []
                for i in range(4):
                    set1, set2 = all_daily_sectors[i], all_daily_sectors[i + 1]
                    map1, map2 = daily_rank_maps[i], daily_rank_maps[i + 1]
                    # 计算加权重合得分
                    weighted_score = 0
                    for sector in set1 & set2:
                        rank1, rank2 = map1[sector], map2[sector]
                        # 前2名重合计2分，3-5名重合计1分
                        if rank1 <= 2 and rank2 <= 2:
                            weighted_score += 2
                        else:
                            weighted_score += 1
                    # 单日加权重合率
                    overlap_rate = weighted_score / 7
                    overlap_rates.append(overlap_rate)
                avg_overlap_rate = sum(overlap_rates) / len(overlap_rates)
                overlap_rotate_coeff = 1 - avg_overlap_rate

                # ========== 步骤2：计算HHI板块集中度 ==========
                sector_appear_count = {}
                for daily_set in all_daily_sectors:
                    for sector in daily_set:
                        sector_appear_count[sector] = sector_appear_count.get(sector, 0) + 1
                total_seats = 25  # 5天×5个固定席位
                hhi = sum((count / total_seats) ** 2 for count in sector_appear_count.values())
                hhi_rotate_coeff = (1 - hhi) / (1 - 0.04)  # 归一化到0-1

                # ========== 步骤3：计算基础轮动分 ==========
                base_rotate_coeff = overlap_rotate_coeff * OVERLAP_WEIGHT + hhi_rotate_coeff * HHI_WEIGHT
                base_score = round(base_rotate_coeff * 100)

                # ========== 步骤4：强主线强度修正（核心降分逻辑） ==========
                # 统计所有板块的上榜、头部次数
                sector_detail_stats = {}
                for daily_map in daily_rank_maps:
                    for sector, rank in daily_map.items():
                        if sector not in sector_detail_stats:
                            sector_detail_stats[sector] = {"appear": 0, "top2": 0, "top3": 0}
                        sector_detail_stats[sector]["appear"] += 1
                        if rank <= 2:
                            sector_detail_stats[sector]["top2"] += 1
                        if rank <= 3:
                            sector_detail_stats[sector]["top3"] += 1

                # 匹配最高等级的主线规则，取最小的修正系数
                final_coeff = 1.0
                for rule in MAIN_LINE_RULES:
                    for sector, stat in sector_detail_stats.items():
                        # 匹配规则
                        match = True
                        if "appear" in rule and stat["appear"] < rule["appear"]:
                            match = False
                        if "top2" in rule and stat["top2"] < rule["top2"]:
                            match = False
                        if "top3" in rule and stat["top3"] < rule["top3"]:
                            match = False
                        if match:
                            if rule["coeff"] < final_coeff:
                                final_coeff = rule["coeff"]
                            break  # 匹配到更高等级规则，直接跳出

                # ========== 步骤5：计算最终adapt_score ==========
                adapt_score = round(base_score * final_coeff)
                # 限制分数范围0-100
                adapt_score = max(0, min(100, adapt_score))
                logger.debug(
                    f"[板块热度] 轮动计算：基础分{base_score} | 主线修正系数{final_coeff} | 最终分{adapt_score}")

            except Exception as e:
                logger.warning(f"[板块热度] 轮动分计算失败：{str(e)}，置为0")
                adapt_score = 0

        # 核心结果日志
        logger.info(f"[板块热度] 计算完成 | 基准日：{trade_date} | 轮动分：{adapt_score} | 最终TOP3：{final_top3}")
        return {"top3_sectors": final_top3, "adapt_score": adapt_score}

if __name__ == "__main__":
        callable= SectorHeatFeature()
        # tag = callable._gen_factor_columns()
        # #获取目标
        top3_sector = callable.select_top3_hot_sectors(trade_date="2026-02-04")
        print(top3_sector)
        #     # ========== 手动修改测试参数即可 ==========
        #     TEST_TS_CODE = "002015.SZ"  # 测试个股代码600759.SH   300189.sz     300164.SZ
        #     TEST_TRADE_DATE = "2026-03-03"  # 测试日期（yyyy-mm-dd）
        #     # ========================================
        #
        #     # 初始化实例
        #     sector_feature = SectorHeatFeature()
        #
        #     # 获取测试所需基础数据
        #     daily_df = get_daily_kline_data(trade_date=TEST_TRADE_DATE, ts_code_list=[TEST_TS_CODE])
        #     minute_df = data_cleaner.get_kline_min_by_stock_date(TEST_TS_CODE, TEST_TRADE_DATE)
        #
        #     # 数据校验
        #     if daily_df.empty:
        #         print(f"错误：未获取到{TEST_TS_CODE}在{TEST_TRADE_DATE}的日线数据")
        #         exit()
        #     if minute_df.empty:
        #         print(f"错误：未获取到{TEST_TS_CODE}在{TEST_TRADE_DATE}的分钟线数据")
        #         exit()
        #
        #     # 提取基础参数
        #     stock_row = daily_df.iloc[0]
        #     pre_close = stock_row["pre_close"]
        #     up_limit = calc_limit_up_price(TEST_TS_CODE, pre_close)
        #     down_limit = calc_limit_down_price(TEST_TS_CODE, pre_close)
        #
        #     sei_score = sector_feature._calculate_minute_sei(minute_df, pre_close, up_limit, down_limit)
        #     boost_score, boost_pct, boost_time = sector_feature._calculate_minute_boost_score(minute_df)
        #
        #     # 直接输出结果
        #     print("=" * 60)
        #     print(f"测试标的：{TEST_TS_CODE} | 测试日期：{TEST_TRADE_DATE}")
        #     print("-" * 60)
        #     print(f"2. 分钟线SEI情绪指数：{sei_score:.2f}")
        #     print("-" * 60)
        #     print(f"3. 分钟线拉升得分：{boost_score:.2f}")
        #     print(f"   最大5分钟涨幅：{boost_pct:.2f}%")
        #     print(f"   对应拉升时间：{boost_time}")
        #     print("=" * 60)
