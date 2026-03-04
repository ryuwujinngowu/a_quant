# features/market_stats.py
"""
板块热度因子计算（核心：30个情绪因子输出）
因子命名规则：sector{板块ID}_d{时间跨度}_{指标名}
- 板块ID：1/2/3（每日前3活跃板块）
- 时间跨度：d4=4天前、d3=3天前、d2=2天前、d1=1天前、d0=D日
- 指标名：profit=涨幅>7%个股数（赚钱效应）、loss=跌幅>7%个股数（亏钱效应）
"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
# 【修改1】直接复用项目已有工具，删除冗余封装
from utils.common_tools import get_trade_dates, get_daily_kline_data
from data.data_cleaner import data_cleaner
import re
from utils.log_utils import logger
from utils.db_utils import db
from datetime import datetime, timedelta
from features.base_feature import BaseFeature

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

    def _calculate_intraday_pattern_score(self, row: pd.Series, is_profit: bool = True) -> float:
        """
        【修改3】优化日内走势模式得分计算
        :param row: 个股日线数据（必须包含open, close, pre_close）
        :param is_profit: 是否为赚钱效应（True=profit，False=loss）
        :return: 走势得分
        """
        open_price = row["open"]
        close_price = row["close"]
        pre_close = row["pre_close"]

        if pre_close <= 0:
            return 0.0

        # 1. 计算开盘缺口幅度（直接*100作为得分）
        open_gap_pct = (open_price / pre_close - 1) * 100

        # 2. 计算当日涨跌幅度（直接*100作为得分）
        daily_pct = (close_price / pre_close - 1) * 100

        if is_profit:
            # Profit：高开幅度 + 当日涨幅
            return round(open_gap_pct + daily_pct, 2)
        else:
            # Loss：低开幅度（绝对值） + 当日跌幅（绝对值）
            return round(abs(open_gap_pct) + abs(daily_pct), 2)

    def _calculate_hdi(self, row: pd.Series) -> float:
        """
        计算持股难易度HDI（Holding Difficulty Index）
        忽略换手率和日内均线，只用日线数据计算
        :param row: 个股日线数据（必须包含open, close, pre_close, high, low）
        :return: HDI得分，0-100，越高越容易持有
        """
        pre_close = row["pre_close"]
        if pre_close <= 0:
            return 50.0  # 异常值默认50分

        # 1. 日内振幅（振幅越小，越容易持有）
        amplitude = (row["high"] - row["low"]) / pre_close * 100
        amplitude_score = max(0, 100 - amplitude * 2)  # 振幅每增加1%，扣2分，封顶100分

        # 2. 盘中最大回撤（从最高价到收盘价的回撤，越小越容易持有）
        max_drawdown = (row["high"] - row["close"]) / pre_close * 100
        drawdown_score = max(0, 100 - max_drawdown * 3)  # 回撤每增加1%，扣3分

        # 3. 冲高回落绝对值（最高价-收盘价，越小越容易持有）
        pullback = row["high"] - row["close"]
        pullback_score = max(0, 100 - pullback * 10)  # 每回落0.1元，扣1分

        # 4. 涨跌幅度绝对值（涨跌越温和，越容易持有）
        pct_chg_abs = abs((row["close"] / pre_close) - 1) * 100
        pct_chg_score = max(0, 100 - pct_chg_abs * 2)  # 涨跌每增加1%，扣2分

        # 综合HDI得分（等权平均）
        hdi = (amplitude_score + drawdown_score + pullback_score + pct_chg_score) / 4
        return round(hdi, 2)

    def _calculate_minute_boost_score(self, minute_df: pd.DataFrame, is_profit: bool = True) -> Tuple[
        float, float, str]:
        """
        【修改4】完善分钟线拉升/下杀得分计算
        :param minute_df: 分钟线DataFrame（来自data_cleaner.get_kline_min_by_stock_date）
        :param is_profit: 是否为赚钱效应（True=profit找拉升，False=loss找下杀）
        :return: (得分, 幅度, 时间)
        """
        if minute_df.empty:
            return 50.0, 0.0, ""

        try:
            # 确保分钟线按时间排序
            if "time" not in minute_df.columns:
                return 50.0, 0.0, ""
            minute_df = minute_df.sort_values("time").reset_index(drop=True)

            # 以2分钟为维度，遍历120根分钟线（9:30-11:30 + 13:00-15:00）
            max_boost_pct = 0.0
            first_boost_time = ""
            first_boost_pct = 0.0

            for i in range(0, len(minute_df) - 1, 2):
                # 取当前分钟和2分钟后的close
                current_close = minute_df.iloc[i]["close"]
                next_close = minute_df.iloc[i + 1]["close"]
                current_time = minute_df.iloc[i]["time"]

                if current_close <= 0:
                    continue

                # 计算2分钟涨跌幅
                boost_pct = (next_close / current_close - 1) * 100

                if is_profit:
                    # Profit：找第一根上涨幅度大于4%的分钟线
                    if boost_pct > 4.0 and first_boost_time == "":
                        first_boost_time = current_time
                        first_boost_pct = boost_pct
                        break
                    if boost_pct > max_boost_pct:
                        max_boost_pct = boost_pct
                else:
                    # Loss：找第一根下跌幅度大于4%的分钟线
                    if boost_pct < -4.0 and first_boost_time == "":
                        first_boost_time = current_time
                        first_boost_pct = abs(boost_pct)
                        break
                    if abs(boost_pct) > max_boost_pct:
                        max_boost_pct = abs(boost_pct)

            # 如果没找到第一根>4%的，用最大幅度
            if first_boost_time == "":
                return 50.0, max_boost_pct, ""

            # 根据时间计算得分：9:45前得分最高，每往后15分钟递减
            # 时间格式假设为 "HH:MM:SS"
            try:
                boost_hour = int(first_boost_time.split(":")[0])
                boost_minute = int(first_boost_time.split(":")[1])
                total_minutes = (boost_hour - 9) * 60 + boost_minute

                # 9:45前（45分钟内）：100分
                if total_minutes <= 45:
                    score = 100.0
                # 每往后15分钟，扣10分
                else:
                    minutes_past = total_minutes - 45
                    score = max(0, 100 - (minutes_past // 15) * 10)
            except:
                score = 50.0

            return round(score, 2), round(first_boost_pct, 2), first_boost_time

        except Exception as e:
            logger.debug(f"分钟线得分计算失败：{e}")
            return 50.0, 0.0, ""

    def calculate(
            self,
            trade_date: str,
            top3_sectors_result: Dict,
            sector_candidate_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, int]:
        """
        【核心方法】计算30个板块强度因子
        :param trade_date: D日日期，格式yyyy-mm-dd
        :param top3_sectors_result: select_top3_hot_sectors返回的结果，格式：{"top3_sectors": [...], "adapt_score": ...}
        :param sector_candidate_map: 策略层传入的板块候选池，格式：{板块名: 候选个股DataFrame}
        :return: 30个因子的字典，格式：{"sector1_d4_profit": 5, "sector1_d4_loss": 2, ...}
        """
        # 初始化因子字典，所有因子默认0
        factor_dict = {col: 0 for col in self.factor_columns}

        # 【修改5】直接调用utils.common_tools.get_trade_dates，删除冗余封装
        try:
            d_date = datetime.strptime(trade_date, "%Y-%m-%d")
            start_date = (d_date - timedelta(days=20)).strftime("%Y-%m-%d")
            all_trade_dates = get_trade_dates(start_date, trade_date)
            if not all_trade_dates or len(all_trade_dates) < TOTAL_DAYS:
                logger.error(f"[板块热度] 获取交易日失败，要求{TOTAL_DAYS}个")
                return factor_dict
            lookback_dates = all_trade_dates[-TOTAL_DAYS:]
            day_tag_map = {f"d{4 - i}": date for i, date in enumerate(lookback_dates)}
        except Exception as e:
            logger.error(f"[板块热度] 获取交易日异常：{str(e)}")
            return factor_dict

        # 2. 获取所有候选股票的ts_code列表
        all_candidate_ts_codes = []
        for sector_df in sector_candidate_map.values():
            if not sector_df.empty:
                all_candidate_ts_codes.extend(sector_df["ts_code"].unique().tolist())
        all_candidate_ts_codes = list(set(all_candidate_ts_codes))
        if not all_candidate_ts_codes:
            return factor_dict

        # 【修改6】直接调用utils.common_tools.get_daily_kline_data，删除冗余封装
        try:
            all_dfs = []
            for date in lookback_dates:
                daily_df = get_daily_kline_data(trade_date=date, ts_code_list=all_candidate_ts_codes)
                if not daily_df.empty:
                    all_dfs.append(daily_df)
            if not all_dfs:
                return factor_dict
            all_daily_df = pd.concat(all_dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"[板块热度] 获取候选股票日线数据失败：{str(e)}")
            return factor_dict

        # 4. 遍历每个板块，计算30个因子
        top3_sectors = top3_sectors_result.get("top3_sectors", [])
        for sector_idx, sector_name in enumerate(top3_sectors, 1):
            sector_prefix = f"sector{sector_idx}"
            # 获取该板块的候选股票
            sector_df = sector_candidate_map.get(sector_name, pd.DataFrame())
            if sector_df.empty:
                continue
            sector_ts_codes = sector_df["ts_code"].unique().tolist()
            # 筛选该板块候选股票的日线数据
            sector_daily_df = all_daily_df[all_daily_df["ts_code"].isin(sector_ts_codes)]
            if sector_daily_df.empty:
                continue

            # 遍历每个时间跨度（d4到d0）
            for day_tag, target_date in day_tag_map.items():
                # 筛选该交易日的数据
                target_df = sector_daily_df[sector_daily_df["trade_date"] == target_date]
                if target_df.empty:
                    continue

                # ==================== 分维度计算profit ====================
                profit_df = target_df[target_df["pct_chg"] > 0].copy()
                if not profit_df.empty:
                    profit_dimensions = []
                    for _, row in profit_df.iterrows():
                        ts_code = row["ts_code"]

                        # 【修改7】直接调用data_cleaner.get_kline_min_by_stock_date获取分钟线
                        minute_df = data_cleaner.get_kline_min_by_stock_date(ts_code, target_date)

                        # 维度1：优化后的日内走势得分
                        pattern_score = self._calculate_intraday_pattern_score(row, is_profit=True)

                        # 维度2：持股难易度HDI
                        hdi_score = self._calculate_hdi(row)

                        # 维度3：分钟线拉升得分
                        minute_boost_score, boost_pct, boost_time = self._calculate_minute_boost_score(minute_df,
                                                                                                       is_profit=True)

                        profit_dimensions.append({
                            "ts_code": ts_code,
                            "pattern_score": pattern_score,
                            "hdi_score": hdi_score,
                            "minute_boost_score": minute_boost_score,
                            "boost_pct": boost_pct,
                            "boost_time": boost_time,
                            "pct_chg": row["pct_chg"]
                        })

                    # 【预留】profit总分合并位置，现在先不计算
                    # profit_total_score = ...

                    # 暂时先用原有的profit计算方式（涨幅>7%的个股数）
                    profit_count = len(profit_df[profit_df["pct_chg"] > 7.0])
                    factor_dict[f"{sector_prefix}_{day_tag}_profit"] = profit_count

                # ==================== 分维度计算loss ====================
                loss_df = target_df[target_df["pct_chg"] < 0].copy()
                if not loss_df.empty:
                    loss_dimensions = []
                    for _, row in loss_df.iterrows():
                        ts_code = row["ts_code"]

                        # 【修改7】直接调用data_cleaner.get_kline_min_by_stock_date获取分钟线
                        minute_df = data_cleaner.get_kline_min_by_stock_date(ts_code, target_date)

                        # 维度1：优化后的日内走势得分（loss模式）
                        pattern_score = self._calculate_intraday_pattern_score(row, is_profit=False)

                        # 维度2：持股难易度HDI（loss模式反转）############################                SEI情绪指数
                        hdi_score = self._calculate_hdi(row)
                        loss_hdi_score = 100 - hdi_score

                        # 维度3：分钟线下杀得分
                        minute_kill_score, kill_pct, kill_time = self._calculate_minute_boost_score(minute_df,
                                                                                                    is_profit=False)

                        loss_dimensions.append({
                            "ts_code": ts_code,
                            "pattern_score": pattern_score,
                            "loss_hdi_score": loss_hdi_score,
                            "minute_kill_score": minute_kill_score,
                            "kill_pct": kill_pct,
                            "kill_time": kill_time,
                            "pct_chg": row["pct_chg"]
                        })

                    # 【预留】loss总分合并位置，现在先不计算
                    # loss_total_score = ...

                    # 暂时先用原有的loss计算方式（跌幅>7%的个股数）
                    loss_count = len(loss_df[loss_df["pct_chg"] < -7.0])
                    factor_dict[f"{sector_prefix}_{day_tag}_loss"] = loss_count

        # ==================== breakpoint()暂停，方便调试 ====================
        logger.info(f"[板块热度] 各维度数据计算完成，暂停调试 | 基准日：{trade_date}")
        breakpoint()

        logger.info(f"[板块热度] 30个因子计算完成 | 基准日：{trade_date}")
        return factor_dict


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

            sql = """
                  SELECT cal_date
                  FROM trade_cal
                  WHERE cal_date BETWEEN %s AND %s \
                    AND is_open = 1
                  ORDER BY cal_date ASC \
                  """
            df = db.query(sql, params=(start_date, trade_date), return_df=True)
            if df.empty:
                logger.error(f"[板块热度] 未查询到{start_date}至{trade_date}的交易日数据")
                return {"top3_sectors": [], "adapt_score": 0}

            # 提取交易日列表，取最后5个（顺序：D-4, D-3, D-2, D-1, D日）
            all_trade_dates = df["cal_date"].astype(str).tolist()
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
        #获取目标
        top3_sector = callable.select_top3_hot_sectors(trade_date="2026-03-04")
        print(top3_sector)

