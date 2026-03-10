"""
板块内个股特征计算
因子命名规则：
- 个股基础指标：stock_{指标名}_d{时间跨度} （open/high/low/close/pct_chg/amount等）
- 个股情绪指标：stock_profit_d{时间跨度}/stock_loss_d{时间跨度}
- 板块平均指标：sector_avg_profit_d{时间跨度}/sector_avg_loss_d{时间跨度}
- 个股排名指标：stock_sector_20d_rank
"""
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from features.base_feature import BaseFeature
from features.feature_registry import feature_registry
from features.emotion.sei_feature import SEIFeature
from utils.common_tools import get_trade_dates, sort_by_recent_gain, calc_limit_up_price, calc_limit_down_price


@feature_registry.register("sector_stock")
class SectorStockFeature(BaseFeature):
    """板块内个股特征类，负责个股维度的板块相关特征计算"""
    feature_name = "sector_stock"
    # 回溯5天的因子列名预生成
    _day_tags = [f"d{i}" for i in range(5)]
    factor_columns = [
        "sector_id", "sector_name", "stock_sector_20d_rank",
        *[f"stock_open_{tag}" for tag in _day_tags],
        *[f"stock_high_{tag}" for tag in _day_tags],
        *[f"stock_low_{tag}" for tag in _day_tags],
        *[f"stock_close_{tag}" for tag in _day_tags],
        *[f"stock_pct_chg_{tag}" for tag in _day_tags],
        *[f"stock_amount_{tag}" for tag in _day_tags],
        *[f"stock_profit_{tag}" for tag in _day_tags],
        *[f"stock_loss_{tag}" for tag in _day_tags],
        *[f"sector_avg_profit_{tag}" for tag in _day_tags],
        *[f"sector_avg_loss_{tag}" for tag in _day_tags],
    ]

    def __init__(self):
        super().__init__()
        self.sei_calculator = SEIFeature()
        self.lookback_days = 5
        self.rank_days = 20

    def calculate(self, data_bundle: "FeatureDataBundle") -> tuple[pd.DataFrame, dict]:
        """
        统一板块个股特征计算入口
        :param data_bundle: 预加载数据容器，必须包含top3_sectors、sector_candidate_map
        :return: 个股-交易日级特征DataFrame，因子字典
        """
        result_rows = []
        factor_dict = {}
        trade_date = data_bundle.trade_date
        d0_date = trade_date

        # 从数据容器获取预加载数据，无需重复请求
        top3_sectors = data_bundle.top3_sectors
        sector_candidate_map = data_bundle.sector_candidate_map
        all_trade_dates_5d = data_bundle.lookback_dates_5d
        all_trade_dates_20d = data_bundle.lookback_dates_20d
        daily_grouped = data_bundle.daily_grouped
        minute_cache = data_bundle.minute_cache

        # 日期映射：d0=当日，d4=4天前
        day_tag_map = {f"d{4 - i}": date for i, date in enumerate(all_trade_dates_5d)}
        factor_dict.update({f"sector{idx+1}_name": name for idx, name in enumerate(top3_sectors)})

        # 预计算每个板块的通用数据
        sector_precalc_map = {}
        for sector_idx, sector_name in enumerate(top3_sectors, 1):
            if not sector_name or sector_name not in sector_candidate_map:
                continue
            sector_d0_df = sector_candidate_map[sector_name]
            if sector_d0_df.empty:
                continue
            sector_ts_codes = sector_d0_df["ts_code"].unique().tolist()

            # 板块内20日涨跌幅排名
            rank_map = {}
            sector_rank_median = 0
            try:
                sorted_df = sort_by_recent_gain(sector_d0_df, d0_date, day_count=20)
                if not sorted_df.empty:
                    sorted_df = sorted_df.copy()
                    sorted_df["rank"] = range(1, len(sorted_df) + 1)
                    rank_map = dict(zip(sorted_df["ts_code"], sorted_df["rank"]))
                    sector_rank_median = sorted_df["rank"].median()
            except Exception as e:
                self.logger.warning(f"[板块个股特征] {sector_name} 排名计算失败：{str(e)}")

            # 个股近5日成交均值
            turnover_mean_map = {}
            for ts_code in sector_ts_codes:
                day_turnover = []
                for date in all_trade_dates_5d:
                    key = (ts_code, date)
                    if key in daily_grouped:
                        day_turnover.append(daily_grouped[key].get("turnover_rate", 0))
                turnover_mean_map[ts_code] = np.mean(day_turnover) if day_turnover else 0

            sector_precalc_map[sector_name] = {
                "rank_map": rank_map,
                "rank_median": sector_rank_median,
                "turnover_mean_map": turnover_mean_map,
                "ts_codes": sector_ts_codes,
                "sector_idx": sector_idx
            }

        # 遍历板块，计算每日板块级因子和个股因子
        for sector_name, sector_precalc in sector_precalc_map.items():
            sector_idx = sector_precalc["sector_idx"]
            sector_prefix = f"sector{sector_idx}"
            sector_d0_df = sector_candidate_map[sector_name]
            if sector_d0_df.empty:
                continue

            # 预计算板块d0-d4的每日因子
            sector_day_factor_map = {}
            day_sei_mean_map = defaultdict(dict)  # 同涨跌SEI均值，用于缺失值填充

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

                    # 从预加载缓存获取分钟线，计算SEI
                    minute_df = minute_cache.get(daily_key, pd.DataFrame())
                    up_limit = calc_limit_up_price(ts_code, pre_close)
                    down_limit = calc_limit_down_price(ts_code, pre_close)
                    sei_score = np.clip(self.sei_calculator._calculate_minute_sei(minute_df, pre_close, up_limit, down_limit), 0, 100)

                    # 分类统计
                    if pct_chg > 1e-6:
                        day_profit_sei.append(sei_score)
                        day_up_sei.append(sei_score)
                    elif pct_chg < -1e-6:
                        day_loss_sei.append(sei_score)
                        day_down_sei.append(sei_score)

                # 板块级因子计算
                sector_profit = round(np.mean(day_profit_sei), 2) if day_profit_sei else 50.0
                avg_loss_sei = round(np.mean(day_loss_sei), 2) if day_loss_sei else 50.0
                sector_loss = round(100 - avg_loss_sei, 2)
                sector_day_factor_map[day_tag] = {"profit": sector_profit, "loss": sector_loss}

                # 填充因子字典
                factor_dict[f"{sector_prefix}_{day_tag}_profit"] = int(sector_profit)
                factor_dict[f"{sector_prefix}_{day_tag}_loss"] = int(sector_loss)

                # 保存同涨跌SEI均值
                day_sei_mean_map[day_tag]["up"] = np.mean(day_up_sei) if day_up_sei else 50.0
                day_sei_mean_map[day_tag]["down"] = np.mean(day_down_sei) if day_down_sei else 50.0

            # 遍历个股，生成单条样本
            for _, d0_row in sector_d0_df.iterrows():
                ts_code = d0_row["ts_code"]
                d0_key = (ts_code, d0_date)
                if d0_key not in daily_grouped:
                    self.logger.warning(f"[板块个股特征] {ts_code} {d0_date} 无基础日线数据，剔除样本")
                    continue

                # 初始化行数据
                row_data = {
                    "stock_code": ts_code,
                    "trade_date": d0_date,
                    "sector_id": sector_idx,
                    "sector_name": sector_name,
                    "stock_sector_20d_rank": sector_precalc["rank_map"].get(ts_code, sector_precalc["rank_median"])
                }

                # 遍历d0-d4，补全所有因子
                d0_data = daily_grouped[d0_key]
                for day_tag, target_date in day_tag_map.items():
                    daily_key = (ts_code, target_date)
                    daily_data = daily_grouped.get(daily_key, None)

                    # 补全OHLC等基础指标，停牌用D日数据填充
                    if daily_data is not None:
                        row_data[f"stock_open_{day_tag}"] = daily_data["open"]
                        row_data[f"stock_high_{day_tag}"] = daily_data["high"]
                        row_data[f"stock_low_{day_tag}"] = daily_data["low"]
                        row_data[f"stock_close_{day_tag}"] = daily_data["close"]
                        row_data[f"stock_pct_chg_{day_tag}"] = daily_data["pct_chg"]
                        row_data[f"stock_amount_{day_tag}"] = daily_data["amount"]
                    else:
                        row_data[f"stock_open_{day_tag}"] = d0_data["open"]
                        row_data[f"stock_high_{day_tag}"] = d0_data["high"]
                        row_data[f"stock_low_{day_tag}"] = d0_data["low"]
                        row_data[f"stock_close_{day_tag}"] = d0_data["close"]
                        row_data[f"stock_pct_chg_{day_tag}"] = 0.0
                        row_data[f"stock_amount_{day_tag}"] = 0.0

                    # 补全个股profit/loss因子
                    if daily_data is not None:
                        pct_chg = daily_data["pct_chg"]
                        pre_close = daily_data["pre_close"]
                        minute_df = minute_cache.get(daily_key, pd.DataFrame())
                        up_limit = calc_limit_up_price(ts_code, pre_close)
                        down_limit = calc_limit_down_price(ts_code, pre_close)
                        sei_score = np.clip(self.sei_calculator._calculate_minute_sei(minute_df, pre_close, up_limit, down_limit), 0, 100)

                        # SEI缺失值填充
                        if sei_score <= 0 and minute_df.empty:
                            if pct_chg > 1e-6:
                                sei_score = day_sei_mean_map[day_tag]["up"]
                            elif pct_chg < -1e-6:
                                sei_score = day_sei_mean_map[day_tag]["down"]
                            else:
                                sei_score = 50.0

                        # 按涨跌赋值
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
                        # 停牌填充中性值
                        row_data[f"stock_profit_{day_tag}"] = 50.0
                        row_data[f"stock_loss_{day_tag}"] = 50.0

                    # 补全板块平均因子
                    row_data[f"sector_avg_profit_{day_tag}"] = sector_day_factor_map[day_tag]["profit"]
                    row_data[f"sector_avg_loss_{day_tag}"] = sector_day_factor_map[day_tag]["loss"]

                result_rows.append(row_data)

        feature_df = pd.DataFrame(result_rows)
        self.logger.info(f"[板块个股特征] {d0_date} 计算完成，有效样本数：{len(feature_df)}")
        return feature_df, factor_dict