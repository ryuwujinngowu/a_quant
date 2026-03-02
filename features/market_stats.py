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
from utils.common_tools import getStockRank_fortraining, getTagRank_daily
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
HHI_WEIGHT = 0.2      # 弱但绵长
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

    def calculate(self, daily_df: pd.DataFrame, trade_date: str) -> Tuple[pd.DataFrame, List[int], Dict[str, int]]:
        """
        核心计算方法（对齐基类接口）
        :param daily_df: 全市场日线数据（含个股所属板块、涨跌幅、成交量等）
        :param trade_date: 当前计算日期（D日，格式：YYYY-MM-DD）
        :return:
            - sector_factor_df: 板块因子DataFrame，index=板块ID，columns=30个因子列
            - top_sector_ids: 当日前3活跃板块ID列表 [1,2,3]
            - stock_sector_map: 个股-所属板块映射 {ts_code: 所属板块ID}

        """
        # ==================== 后续填充具体计算逻辑 ====================
        # 1. 计算每日板块热度，筛选前3活跃板块，分配板块ID 1/2/3
        # 2. 回溯5天，计算每个板块每日的赚钱/亏钱效应
        # 3. 生成个股-所属板块的映射关系
        # ==============================================================

        # 预设空返回值（格式固定，后续填充真实数据）
        sector_factor_df = pd.DataFrame(columns=self.factor_columns, index=[1,2,3])
        top_sector_ids = [1,2,3]
        stock_sector_map = {}

        return sector_factor_df, top_sector_ids, stock_sector_map

    def get_factor_columns(self) -> List[str]:
        """获取30个因子列名，供数据集构建时调用"""
        return self.factor_columns





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
            return {"top3_sectors": [], "adapt_score": 0}

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
        tag = callable._gen_factor_columns()
        #获取目标
        top3_sector = callable.select_top3_hot_sectors(trade_date="2026-03-02")
        print(top3_sector)

