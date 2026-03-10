"""
板块内个股特征计算
==================
输出因子全览（每个交易日 d0~d4 各输出一组）：

【原始行情】
  stock_open/high/low/close/pct_chg/amount_{d}

【情绪合成分（保留，与原子因子并存）】
  stock_profit_{d}      : 上涨日 SEI；下跌/平盘日 = 0
  stock_loss_{d}        : 下跌日 100-SEI；上涨/平盘日 = 0
  stock_hdi_{d}         : HDI 持股难度指数（0-100）

【方向/价格原子因子 ★ 新增独立列】
  stock_gap_return_{d}  : 开盘缺口率 = (open-pre_close)/pre_close
                          正=高开（主力溢价拉升），负=低开（出货压力）
  stock_candle_{d}      : K 线结构 {2=真阳,1=假阳,-1=假阴,-2=真阴}
                          假阴(低开高走)是次日大涨强预测子
  stock_cpr_{d}         : 收盘位置比 = (close-low)/(high-low)
                          0=收于最低，1=收于最高（封板股≈1）

【波动/持仓结构原子因子 ★ 新增独立列】
  stock_max_dd_{d}      : 日内最大回撤（涨停板次日走势的强预测子）
  stock_upper_shadow_{d}: 上影线比率，反映上方抛压
  stock_lower_shadow_{d}: 下影线比率，反映下方支撑
  stock_trend_r2_{d}    : 分钟线趋势 R²（0=震荡，1=单边趋势）
  stock_vwap_dev_{d}    : VWAP 偏离度（筹码散乱程度）

【涨跌停行为原子因子 ★ 新增独立列】
  stock_seal_times_{d}  : 封板次数
  stock_break_times_{d} : 开板次数（开板越多情绪越弱）
  stock_lift_times_{d}  : 跌停翘板次数

【量能原子因子 ★ 全新维度】
  stock_vol_ratio_{d}   : 量比 = 当日量 / 近5日均量
                          放量>1，缩量<1，停牌=0

【板块平均因子】
  sector_avg_profit_{d}, sector_avg_loss_{d}

【个股排名】
  stock_sector_20d_rank

设计说明：
  sei_cache 存储全量 factors dict，避免双重计算
  无分钟线时从日线 OHLC 回退（calc_daily_atomic），保证缓存完整
  停牌 vs 无分钟线 使用不同中性值，语义区分明确
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from utils.log_utils import logger
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from features.emotion.sei_feature import SEIFeature
from utils.common_tools import (
    sort_by_recent_gain,
    calc_limit_up_price,
    calc_limit_down_price,
)


# ============================================================
# 无分钟线时的回退中性值（语义：数据不完整，不是停牌）
# ============================================================
ATOMIC_NEUTRAL = {
    "gap_return":    0.0,    # 无法判断，填平开
    "candle_type":   0,      # 0=无结构信息
    "cpr":           0.5,    # 中性位置
    "max_dd_intra":  0.10,   # 保守中位值
    "upper_shadow":  0.0,
    "lower_shadow":  0.0,
    "trend_r2":      0.5,    # 中性
    "vwap_deviation":0.01,   # 保守小值
    "seal_times":    0,
    "break_times":   0,
    "lift_times":    0,
    "vol_ratio":     1.0,    # 正常量
    "hdi":           50.0,
}

# 停牌时的填充值（语义：该股无行情，成交量 = 0）
SUSPENDED_FILL = dict(ATOMIC_NEUTRAL)
SUSPENDED_FILL["vol_ratio"] = 0.0    # 停牌量为 0，量比=0 有明确语义


@feature_registry.register("sector_stock")
class SectorStockFeature(BaseFeature):
    """板块内个股特征类（全量原子因子版）"""

    feature_name = "sector_stock"
    _day_tags    = [f"d{i}" for i in range(5)]

    factor_columns = [
        "sector_id", "sector_name", "stock_sector_20d_rank",
        *[f"stock_open_{t}"          for t in _day_tags],
        *[f"stock_high_{t}"          for t in _day_tags],
        *[f"stock_low_{t}"           for t in _day_tags],
        *[f"stock_close_{t}"         for t in _day_tags],
        *[f"stock_pct_chg_{t}"       for t in _day_tags],
        *[f"stock_amount_{t}"        for t in _day_tags],
        *[f"stock_profit_{t}"        for t in _day_tags],
        *[f"stock_loss_{t}"          for t in _day_tags],
        *[f"stock_hdi_{t}"           for t in _day_tags],
        *[f"stock_gap_return_{t}"    for t in _day_tags],
        *[f"stock_candle_{t}"        for t in _day_tags],
        *[f"stock_cpr_{t}"           for t in _day_tags],
        *[f"stock_max_dd_{t}"        for t in _day_tags],
        *[f"stock_upper_shadow_{t}"  for t in _day_tags],
        *[f"stock_lower_shadow_{t}"  for t in _day_tags],
        *[f"stock_trend_r2_{t}"      for t in _day_tags],
        *[f"stock_vwap_dev_{t}"      for t in _day_tags],
        *[f"stock_seal_times_{t}"    for t in _day_tags],
        *[f"stock_break_times_{t}"   for t in _day_tags],
        *[f"stock_lift_times_{t}"    for t in _day_tags],
        *[f"stock_vol_ratio_{t}"     for t in _day_tags],
        *[f"sector_avg_profit_{t}"   for t in _day_tags],
        *[f"sector_avg_loss_{t}"     for t in _day_tags],
    ]

    def __init__(self):
        super().__init__()
        self.sei_calculator = SEIFeature()

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_vol_ratio(ts_code: str, target_date: str,
                        all_dates_5d: List[str], daily_grouped: dict) -> float:
        """量比 = 当日成交量 / 近5日（含当日）均量"""
        today_row = daily_grouped.get((ts_code, target_date))
        if not today_row:
            return 0.0
        today_vol = float(today_row.get("vol", 0) or 0)
        if today_vol == 0:
            return 0.0
        vols = [
            float(daily_grouped[(ts_code, d)].get("vol", 0) or 0)
            for d in all_dates_5d
            if (ts_code, d) in daily_grouped
        ]
        avg_vol = np.mean(vols) if vols else 0.0
        return round(today_vol / (avg_vol + 1e-6), 3)

    # ------------------------------------------------------------------ #
    # 主计算入口
    # ------------------------------------------------------------------ #

    def calculate(self, data_bundle) -> tuple:
        result_rows   = []
        factor_dict   = {}
        trade_date    = data_bundle.trade_date
        top3_sectors  = data_bundle.top3_sectors
        sector_map    = data_bundle.sector_candidate_map
        daily_grouped = data_bundle.daily_grouped
        minute_cache  = data_bundle.minute_cache
        all_dates_5d  = data_bundle.lookback_dates_5d   # 升序，[-1]=trade_date

        # d0=trade_date，d4=4天前
        day_tag_map: Dict[str, str] = {
            f"d{4 - i}": date for i, date in enumerate(all_dates_5d)
        }

        # ================================================================
        # 阶段 1：板块级预计算
        # ================================================================
        for sector_idx, sector_name in enumerate(top3_sectors, 1):
            if not sector_name or sector_name not in sector_map:
                continue
            sector_d0_df = sector_map[sector_name]
            if sector_d0_df.empty:
                continue

            sector_ts_codes: List[str] = sector_d0_df["ts_code"].unique().tolist()

            # ---- 20 日涨幅排名 ----
            rank_map, rank_median = {}, 0
            try:
                sorted_df = sort_by_recent_gain(sector_d0_df, trade_date, day_count=20)
                if not sorted_df.empty:
                    sorted_df = sorted_df.copy()
                    sorted_df["rank"] = range(1, len(sorted_df) + 1)
                    rank_map    = dict(zip(sorted_df["ts_code"], sorted_df["rank"]))
                    rank_median = sorted_df["rank"].median()
            except Exception as e:
                logger.warning(f"[板块个股] {sector_name} 排名失败: {e}")

            # ================================================================
            # SEI/HDI/全量原子因子统一缓存
            # value = {"sei": float, "hdi": float, "factors": dict}
            # 无分钟线时用 calc_daily_atomic 从日线回退，保证缓存始终有值
            # ================================================================
            sei_cache: Dict[tuple, dict] = {}

            for ts_code in sector_ts_codes:
                for _, target_date in day_tag_map.items():
                    daily_key = (ts_code, target_date)
                    if daily_key in sei_cache or daily_key not in daily_grouped:
                        continue

                    daily_row = daily_grouped[daily_key]
                    pre_close = daily_row.get("pre_close", 0)
                    if not pre_close or pre_close <= 0:
                        continue

                    up_limit   = calc_limit_up_price(ts_code, pre_close)
                    down_limit = calc_limit_down_price(ts_code, pre_close)
                    minute_df  = minute_cache.get(daily_key, pd.DataFrame())

                    hdi, factors = self.sei_calculator._calculate_minute_hdi(
                        minute_df, pre_close, up_limit, down_limit
                    )

                    if factors:
                        sei = self.sei_calculator._factors_to_sei(factors, up_limit)
                    else:
                        # 无分钟线：从日线 OHLC 回退
                        sei = hdi = 50.0
                        factors = SEIFeature.calc_daily_atomic(
                            open_price  = daily_row.get("open",  pre_close),
                            high_price  = daily_row.get("high",  pre_close),
                            low_price   = daily_row.get("low",   pre_close),
                            close_price = daily_row.get("close", pre_close),
                            pre_close   = pre_close,
                        )

                    sei_cache[daily_key] = {
                        "sei":     float(sei),
                        "hdi":     float(hdi),
                        "factors": factors,
                    }

            # ---- 板块每日赚钱/亏钱效应 ----
            sector_day_factors: Dict[str, dict] = {}
            day_sei_mean = defaultdict(lambda: {"up": [], "down": []})

            for day_tag, target_date in day_tag_map.items():
                p_list, l_list = [], []
                for ts_code in sector_ts_codes:
                    dk = (ts_code, target_date)
                    if dk not in daily_grouped or dk not in sei_cache:
                        continue
                    pct = daily_grouped[dk].get("pct_chg", 0)
                    sei = sei_cache[dk]["sei"]
                    if pct > 1e-6:
                        p_list.append(sei);  day_sei_mean[day_tag]["up"].append(sei)
                    elif pct < -1e-6:
                        l_list.append(sei);  day_sei_mean[day_tag]["down"].append(sei)

                sector_day_factors[day_tag] = {
                    "profit": round(np.mean(p_list), 2) if p_list else 50.0,
                    "loss":   round(100 - np.mean(l_list), 2) if l_list else 50.0,
                }
                day_sei_mean[day_tag]["up"]   = float(np.mean(day_sei_mean[day_tag]["up"]))   if day_sei_mean[day_tag]["up"]   else 50.0
                day_sei_mean[day_tag]["down"] = float(np.mean(day_sei_mean[day_tag]["down"])) if day_sei_mean[day_tag]["down"] else 50.0

            # ================================================================
            # 阶段 2：个股级特征组装
            # ================================================================
            for _, d0_row in sector_d0_df.iterrows():
                ts_code = d0_row["ts_code"]
                d0_key  = (ts_code, trade_date)
                if d0_key not in daily_grouped:
                    logger.warning(f"[板块个股] {ts_code} {trade_date} 无日线，跳过")
                    continue

                d0_data = daily_grouped[d0_key]
                row = {
                    "stock_code":            ts_code,
                    "trade_date":            trade_date,
                    "sector_id":             sector_idx,
                    "sector_name":           sector_name,
                    "stock_sector_20d_rank": rank_map.get(ts_code, rank_median),
                }

                for day_tag, target_date in day_tag_map.items():
                    daily_key  = (ts_code, target_date)
                    daily_data = daily_grouped.get(daily_key)
                    cache      = sei_cache.get(daily_key)

                    # ---- 原始行情 ----
                    if daily_data:
                        row[f"stock_open_{day_tag}"]    = daily_data.get("open",    d0_data.get("open",    0))
                        row[f"stock_high_{day_tag}"]    = daily_data.get("high",    d0_data.get("high",    0))
                        row[f"stock_low_{day_tag}"]     = daily_data.get("low",     d0_data.get("low",     0))
                        row[f"stock_close_{day_tag}"]   = daily_data.get("close",   d0_data.get("close",   0))
                        row[f"stock_pct_chg_{day_tag}"] = daily_data.get("pct_chg", 0.0)
                        row[f"stock_amount_{day_tag}"]  = daily_data.get("amount",  0.0)
                    else:
                        row[f"stock_open_{day_tag}"]    = d0_data.get("open",  0)
                        row[f"stock_high_{day_tag}"]    = d0_data.get("high",  0)
                        row[f"stock_low_{day_tag}"]     = d0_data.get("low",   0)
                        row[f"stock_close_{day_tag}"]   = d0_data.get("close", 0)
                        row[f"stock_pct_chg_{day_tag}"] = 0.0
                        row[f"stock_amount_{day_tag}"]  = 0.0

                    # ---- 情绪合成分 + 全量原子因子 ----
                    if daily_data and cache:
                        pct_chg = daily_data.get("pct_chg", 0)
                        sei     = cache["sei"]
                        hdi     = cache["hdi"]
                        f       = cache["factors"]

                        # 分钟线缺失时 SEI 用板块均值替代（candle_type 已由日线回退）
                        if minute_cache.get(daily_key, pd.DataFrame()).empty:
                            sei = day_sei_mean[day_tag]["up"]   if pct_chg > 1e-6 else \
                                  day_sei_mean[day_tag]["down"] if pct_chg < -1e-6 else sei

                        if pct_chg > 1e-6:
                            row[f"stock_profit_{day_tag}"] = round(sei, 2)
                            row[f"stock_loss_{day_tag}"]   = 0.0
                        elif pct_chg < -1e-6:
                            row[f"stock_profit_{day_tag}"] = 0.0
                            row[f"stock_loss_{day_tag}"]   = round(100 - sei, 2)
                        else:
                            row[f"stock_profit_{day_tag}"] = 0.0
                            row[f"stock_loss_{day_tag}"]   = 0.0

                        row[f"stock_hdi_{day_tag}"]          = hdi
                        row[f"stock_gap_return_{day_tag}"]   = f.get("gap_return",     ATOMIC_NEUTRAL["gap_return"])
                        row[f"stock_candle_{day_tag}"]        = f.get("candle_type",    ATOMIC_NEUTRAL["candle_type"])
                        row[f"stock_cpr_{day_tag}"]           = f.get("cpr",            ATOMIC_NEUTRAL["cpr"])
                        row[f"stock_max_dd_{day_tag}"]        = f.get("max_dd_intra",   ATOMIC_NEUTRAL["max_dd_intra"])
                        row[f"stock_upper_shadow_{day_tag}"]  = f.get("upper_shadow",   ATOMIC_NEUTRAL["upper_shadow"])
                        row[f"stock_lower_shadow_{day_tag}"]  = f.get("lower_shadow",   ATOMIC_NEUTRAL["lower_shadow"])
                        row[f"stock_trend_r2_{day_tag}"]      = f.get("trend_r2",       ATOMIC_NEUTRAL["trend_r2"])
                        row[f"stock_vwap_dev_{day_tag}"]      = f.get("vwap_deviation", ATOMIC_NEUTRAL["vwap_deviation"])
                        row[f"stock_seal_times_{day_tag}"]    = f.get("seal_times",     0)
                        row[f"stock_break_times_{day_tag}"]   = f.get("break_times",    0)
                        row[f"stock_lift_times_{day_tag}"]    = f.get("lift_times",     0)
                        row[f"stock_vol_ratio_{day_tag}"]     = self._calc_vol_ratio(
                            ts_code, target_date, all_dates_5d, daily_grouped
                        )

                    else:
                        # 停牌或完全无数据
                        row[f"stock_profit_{day_tag}"]       = 0.0
                        row[f"stock_loss_{day_tag}"]         = 0.0
                        row[f"stock_hdi_{day_tag}"]          = SUSPENDED_FILL["hdi"]
                        row[f"stock_gap_return_{day_tag}"]   = SUSPENDED_FILL["gap_return"]
                        row[f"stock_candle_{day_tag}"]       = SUSPENDED_FILL["candle_type"]
                        row[f"stock_cpr_{day_tag}"]          = SUSPENDED_FILL["cpr"]
                        row[f"stock_max_dd_{day_tag}"]       = SUSPENDED_FILL["max_dd_intra"]
                        row[f"stock_upper_shadow_{day_tag}"] = SUSPENDED_FILL["upper_shadow"]
                        row[f"stock_lower_shadow_{day_tag}"] = SUSPENDED_FILL["lower_shadow"]
                        row[f"stock_trend_r2_{day_tag}"]     = SUSPENDED_FILL["trend_r2"]
                        row[f"stock_vwap_dev_{day_tag}"]     = SUSPENDED_FILL["vwap_deviation"]
                        row[f"stock_seal_times_{day_tag}"]   = 0
                        row[f"stock_break_times_{day_tag}"]  = 0
                        row[f"stock_lift_times_{day_tag}"]   = 0
                        row[f"stock_vol_ratio_{day_tag}"]    = SUSPENDED_FILL["vol_ratio"]

                    # ---- 板块平均（无条件填充）----
                    row[f"sector_avg_profit_{day_tag}"] = sector_day_factors[day_tag]["profit"]
                    row[f"sector_avg_loss_{day_tag}"]   = sector_day_factors[day_tag]["loss"]

                result_rows.append(row)

        feature_df = pd.DataFrame(result_rows)
        logger.info(
            f"[板块个股特征] {trade_date} 完成"
            f" | 样本: {len(feature_df)} | 列数: {len(feature_df.columns) if not feature_df.empty else 0}"
        )
        return feature_df, factor_dict