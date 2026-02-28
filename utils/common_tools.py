import json
import os
import re
import threading
from datetime import datetime, timedelta
import pandas as pd
import time
from pathlib import Path
import functools
from typing import List, Dict, Optional
from typing import Tuple
from utils.db_utils import db
from utils.log_utils import logger

# # Token 统计文件配置（默认统计到 data 目录，可根据需要调整）
# TOKEN_USAGE_FILE = None
# TOKEN_LOCK = threading.Lock()
#

# 增量更新配置（可统一维护）
UPDATE_RECORD_FILE = Path(__file__).parent.parent / "data" / "update_record.json"
DEFAULT_FIRST_UPDATE_DATE = "2024-01-01"

default_exclude = [
    "融资融券", "转融券标的", "标普道琼斯A股", '核准制次新股', '腾讯概念', '阿里巴巴概念', '抖音概念','ST板块','军工',
    "MSCI概念", "深股通", "沪股通", '一带一路', '新股与次新股', '节能环保','稀缺资源','电子商务','俄乌冲突概念',
    "同花顺漂亮100", "富时罗素概念", "富时罗素概念股", '比亚迪概念', '5G', '小金属概念','参股银行','锂电池','spacex',
    "央企国资改革", "地方国资改革", "证金持股", '新能源汽车', '次新股', '宁德时代概念','人民币贬值受益','中俄贸易概念',
    "汇金持股", "养老金持股", "QFII重仓", "专精特新", 'MSCI中国', '半年报预增', '华为概念', '光伏概念', '储能','一季报预增'
]
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


def getStockRank_fortraining(trade_date: str) -> Optional[pd.DataFrame]:
    """
    数据库读取指定日期trade_date
    的全市场历史K线数据（仅用于机器学习训练），筛选符合涨幅阈值的股票并返回


    筛选规则：
    1. 忽略北交所股票（代码含.BJ）；
    2. 创业板（3*.SZ）、科创板（688*.SH）：当日涨跌幅>13%（20%涨停×65%）；
    3. 主板股票（非上述板块）：当日涨跌幅>6.5%（10%涨停×65%）；

    :param trade_date: 待查询日期，格式必须为yyyy-mm-dd（如2025-04-28）
    :return: DataFrame（列：ts_code、pct_chg），按pct_chg正序排列；查询失败/无数据返回None
    """
    # ==================== 2. 构建筛选SQL（精准区分板块涨幅阈值） ====================
    logger.info('*'*60)
    logger.info('！！！！！！历史数据，非实时数据，仅供模拟训练！！！！！！！')
    logger.info('*'*60)
    sql = f"""
        SELECT ts_code, pct_chg
        FROM kline_day
        WHERE trade_date = '{trade_date}'  -- 直接拼接日期字符串
        AND ts_code NOT LIKE '%%.BJ'
        AND pct_chg > CASE
            WHEN ts_code LIKE '3%%.SZ' THEN 13.0
            WHEN ts_code LIKE '688%%.SH' THEN 13.0
            ELSE 6.5
        END;
    """

    # 调用时不传params参数
    result_df = db.query(sql=sql, return_df=True)

    # ==================== 4. 结果处理 ====================
    if result_df is None:
        logger.error(f"查询{trade_date}符合条件的股票失败（数据库异常）")
        return None
    if result_df.empty:
        logger.warning(f"{trade_date}无符合涨幅阈值的股票数据,检查该日是否交易")
        return pd.DataFrame()
    # 确保列名正确（数据库返回的字段名可能大小写/别名问题，强制对齐）
    result_df = result_df.rename(columns=str.lower).loc[:, ['ts_code', 'pct_chg']]
    # 再次确认按pct_chg正序排列（防止SQL排序失效）
    result_df = result_df.sort_values(by='pct_chg', ascending=True).reset_index(drop=True)
    logger.info(f"{trade_date}共查询到{len(result_df)}  只符合条件的股票")

    return result_df


def getTagRank_daily(
        ts_code_list: List[str],
        exclude_concepts: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    接受股票ts_code列表，本地拆分逗号分隔的题材字段，统计题材覆盖情况
    支持传入黑名单数组，过滤掉不具备分析性的题材

    :param ts_code_list: 待分析的股票代码列表（带.SZ/.SH后缀）
    :param exclude_concepts: 需要忽略的题材黑名单数组（可选，不传则使用默认黑名单）
    :return: DataFrame（列：concept_name、cover_stock_count、cover_rate），
             按cover_stock_count降序排列；查询失败/无数据返回None
    """
    # ==================== 0. 初始化默认黑名单 ====================
    # 默认过滤掉常见的无分析性题材，用户可通过参数覆盖
    if exclude_concepts is None:
        exclude_concepts = default_exclude
    else:
        # 如果用户传了黑名单，合并默认黑名单（避免遗漏），也可以直接覆盖
        # exclude_concepts = list(set(exclude_concepts + default_exclude)) # 合并模式
        pass  # 覆盖模式：直接使用用户传入的黑名单

    # ==================== 1. 输入校验 ====================
    if not ts_code_list:
        logger.warning("输入的ts_code_list为空，无法进行题材统计")
        return None

    # 去重并统计输入股票数量
    ts_code_list = list(set(ts_code_list))
    input_stock_count = len(ts_code_list)
    logger.info(f"开始统计{input_stock_count}只股票的题材覆盖情况，黑名单题材数：{len(exclude_concepts)}")

    # ==================== 2. 极简SQL查询原始数据 ====================
    ts_code_str = "','".join(ts_code_list)
    ts_code_str = f"'{ts_code_str}'"

    sql = f"""
        SELECT ts_code, concept_tags 
        FROM stock_basic 
        WHERE ts_code IN ({ts_code_str})
        AND concept_tags IS NOT NULL
        AND TRIM(concept_tags) != '';
    """

    try:
        raw_df = db.query(sql=sql, return_df=True)
    except Exception as e:
        logger.error(f"题材原始数据查询失败：{str(e)}", exc_info=True)
        return None

    if raw_df is None:
        logger.error("题材原始数据查询返回None（数据库异常）")
        return None
    if raw_df.empty:
        logger.warning("未查询到符合条件的股票题材数据")
        return raw_df

    # ==================== 3. 本地Pandas向量化处理（含黑名单过滤） ====================
    raw_df = raw_df.rename(columns=str.lower)

    # 1. 统一分隔符
    raw_df["concept_tags"] = raw_df["concept_tags"].str.replace("；|，|;", ",", regex=True)
    # 2. 拆分+展开
    exploded_df = raw_df.assign(
        concept_name=raw_df["concept_tags"].str.split(",")
    ).explode("concept_name", ignore_index=True)
    # 3. 去空格、过滤空值
    exploded_df["concept_name"] = exploded_df["concept_name"].str.strip()
    exploded_df = exploded_df[exploded_df["concept_name"] != ""].reset_index(drop=True)

    # ==================== 【新增】4. 黑名单过滤 ====================
    before_filter_count = exploded_df["concept_name"].nunique()
    # 核心过滤逻辑：使用 isin() 匹配黑名单，然后取反 ~
    exploded_df = exploded_df[~exploded_df["concept_name"].isin(exclude_concepts)].reset_index(drop=True)
    after_filter_count = exploded_df["concept_name"].nunique()

    logger.info(
        f"黑名单过滤完成：过滤前题材数 {before_filter_count}，过滤后 {after_filter_count}，共过滤 {before_filter_count - after_filter_count} 个题材")

    # ==================== 5. 分组统计 ====================
    if exploded_df.empty:
        logger.warning("经过黑名单过滤后，无剩余题材数据")
        return pd.DataFrame(columns=["concept_name", "cover_stock_count", "cover_rate"])

    result_df = exploded_df.groupby("concept_name", as_index=False).agg(
        cover_stock_count=("ts_code", "nunique"),
    )
    result_df["cover_rate"] = round(
        result_df["cover_stock_count"] / input_stock_count * 100,
        2
    )
    result_df = result_df.sort_values(
        by=["cover_stock_count", "cover_rate"],
        ascending=[False, False]
    ).head(5).reset_index(drop=True)

    # ==================== 6. 结果输出 ====================
    logger.info(f"题材统计完成，前5名题材：\n{result_df.to_string(index=False)}")
    return result_df

def select_top3_hot_sectors(trade_date: str) -> Dict:
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
    daily_rank_maps = []    # 每日板块-排名映射，用于趋势计算
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
            daily_board = [{"rank": i+1, "name": str(row["concept_name"]).strip()} for i, row in tag_df.iterrows()]
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
    sorted_rule1_1 = sorted(rule1_candidate_1.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
    rule1_selected = [item[0] for item in sorted_rule1_1[:2]]

    # 不足2个，放宽规则：5天出现≥3次 + 至少1次进过前3
    if len(rule1_selected) < 2:
        need = 2 - len(rule1_selected)
        rule1_candidate_2 = {
            name: stat for name, stat in sector_stats.items()
            if name not in rule1_selected and stat["appear_count"] >= 3 and stat["has_top3"]
        }
        sorted_rule1_2 = sorted(rule1_candidate_2.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
        rule1_selected += [item[0] for item in sorted_rule1_2[:need]]

    # 兜底：仍不足2个，从所有进过前3的板块中补全
    if len(rule1_selected) < 2:
        need = 2 - len(rule1_selected)
        rule1_candidate_3 = {
            name: stat for name, stat in sector_stats.items()
            if name not in rule1_selected and stat["has_top3"]
        }
        sorted_rule1_3 = sorted(rule1_candidate_3.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
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
        y = rank_series     # y轴：当日排名
        x_mean, y_mean = sum(x)/n, sum(y)/n
        cov = sum((x[i]-x_mean)*(y[i]-y_mean) for i in range(n))
        var_x = sum((x[i]-x_mean)**2 for i in range(n))
        slope = cov / var_x if var_x != 0 else 0
        # 计算平均排名
        avg_rank = sum(y)/n
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
        sorted_remaining = sorted(remaining_candidates.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
        rule2_selected = [item[0] for item in sorted_remaining[:1]]
    logger.debug(f"[板块热度] 规则2选中预期差题材：{rule2_selected}")

    # 合并最终TOP3，极端兜底补全
    final_top3 = rule1_selected + rule2_selected
    if len(final_top3) < 3:
        need = 3 - len(final_top3)
        all_sorted = sorted(sector_stats.items(), key=lambda x: (x[1]["total_score"], x[1]["appear_count"]), reverse=True)
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
            logger.debug(f"[板块热度] 轮动计算：基础分{base_score} | 主线修正系数{final_coeff} | 最终分{adapt_score}")

        except Exception as e:
            logger.warning(f"[板块热度] 轮动分计算失败：{str(e)}，置为0")
            adapt_score = 0

    # 核心结果日志
    logger.info(f"[板块热度] 计算完成 | 基准日：{trade_date} | 轮动分：{adapt_score} | 最终TOP3：{final_top3}")
    return {"top3_sectors": final_top3, "adapt_score": adapt_score}

# # 每日开盘前执行一次，缓存全量题材数据
# def cache_all_concept_data():
#     sql = "SELECT ts_code, concept_tags FROM stockbasic WHERE concept_tags IS NOT NULL"
#     all_concept_df = db.query(sql=sql, return_df=True)
#     all_concept_df.to_parquet("data/all_stock_concepts.parquet", index=False)
#     logger.info("全量题材数据缓存完成")
#     return all_concept_df
#
# # 盘中直接读取缓存，不用查数据库
# def getTagRank_daily_cached(ts_code_list: List[str], cached_df: pd.DataFrame) -> Optional[pd.DataFrame]:
#     # 直接从缓存中过滤需要的股票，不用查数据库
#     raw_df = cached_df[cached_df["ts_code"].isin(ts_code_list)].copy()
#     # 后续处理逻辑和上面一致，省略...

def init_update_record():
    """初始化更新记录文件"""
    init_data = {
        "last_update_date": DEFAULT_FIRST_UPDATE_DATE,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_tables": []
    }
    with open(UPDATE_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(init_data, f, ensure_ascii=False, indent=4)
    return init_data


def read_last_update_record():
    """读取上次更新记录"""
    if not os.path.exists(UPDATE_RECORD_FILE):
        return init_update_record()
    try:
        with open(UPDATE_RECORD_FILE, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["last_update_date"] = record.get("last_update_date", DEFAULT_FIRST_UPDATE_DATE)
        return record
    except Exception as e:
        logger.error(f"读取更新记录失败，初始化新记录：{e}")
        return init_update_record()


def write_update_record(update_date: str, updated_tables: list):
    """写入本次更新记录"""
    record_data = {
        "last_update_date": update_date,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_tables": updated_tables
    }
    try:
        with open(UPDATE_RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=4)
        logger.info(f"更新记录已保存：本次更新至 {update_date}")
    except Exception as e:
        logger.error(f"写入更新记录失败：{e}")


def calc_incremental_date_range(last_update_date: str, end_date: str = None) -> list:
    """计算增量更新日期列表（YYYYMMDD）"""
    last_dt = datetime.strptime(last_update_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    if last_dt >= end_dt:
        logger.info("无增量日期，无需更新")
        return []

    date_list = []
    current_dt = last_dt + timedelta(days=1)
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%Y%m%d"))
        current_dt += timedelta(days=1)

    logger.info(f"增量更新日期：共{len(date_list)}天 → {date_list}")
    return date_list


def calc_15_years_date_range() -> Tuple[str, str]:
    """
    获取倒推15年内的K线数据
    返回：(start_date, end_date) 均为YYYYMMDD格式字符串
    """
    # 结束日期：当前日期（YYYYMMDD）
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=15*365)).strftime("%Y%m%d")
    logger.info(f"日K数据获取时间范围：{start_date} 至 {end_date}")
    return start_date, end_date

def escape_mysql_reserved_words(field_name: str) -> str:
    """
    转义MySQL保留字字段（给保留字加反引号）
    用于避免字段名与MySQL关键字冲突
    """
    # 常见MySQL保留字（金融数据场景高频）
    reserved_words = {"change", "desc", "order", "select", "insert", "update", "volume"}
    if field_name.lower() in reserved_words:
        return f"`{field_name}`"
    return field_name




def auto_add_missing_table_columns(
    table_name: str,
    missing_columns: List[str],
    col_type_mapping: Dict[str, str] = None
) -> bool:
    """
    通用方法：自动为数据库表新增缺失字段（带默认值，避免插入NULL/NaN报错）
    迁移说明：从DataCleaner类迁移为通用函数，核心逻辑完全不变
    """
    # 默认字段类型映射（金融数据标准化规则）
    default_col_type_mapping = {
        # 日期类字段
        "list_date": "DATE NOT NULL DEFAULT '1970-01-01'",
        "delist_date": "DATE DEFAULT NULL",
        # 数值类字段
        "total_share": "BIGINT DEFAULT 0",
        "float_share": "BIGINT DEFAULT 0",
        "free_share": "BIGINT DEFAULT 0",
        "total_mv": "DECIMAL(20,2) DEFAULT 0.00",
        "circ_mv": "DECIMAL(20,2) DEFAULT 0.00",
        # 核心字符串字段
        "exchange": "VARCHAR(8) NOT NULL DEFAULT 'UNKNOWN'",
        "ts_code": "VARCHAR(9) NOT NULL DEFAULT 'UNKNOWN'",
        "symbol": "VARCHAR(6) NOT NULL DEFAULT 'UNKNOWN'",
        "name": "VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN'",
        # 兜底类型
        "default": "VARCHAR(255) NOT NULL DEFAULT ''"
    }

    # 合并默认映射和自定义映射（自定义优先级更高）
    final_col_map = default_col_type_mapping.copy()
    if col_type_mapping:
        final_col_map.update(col_type_mapping)

    success = True
    for col in missing_columns:
        try:
            col_type = final_col_map.get(col, final_col_map["default"])
            db.add_table_column(table_name, col, col_type)
            logger.info(f"表{table_name}新增字段{col}成功（类型：{col_type}）")
        except Exception as e:
            success = False
            logger.error(f"表{table_name}新增字段{col}失败：{e}")

    return success




# def init_token_file(file_path: str, force: bool = False):
#     """
#     初始化Token统计文件
#     :param file_path: Token统计文件的绝对/相对路径
#     :param force: 是否强制覆盖现有文件（默认False）
#     """
#     global TOKEN_USAGE_FILE
#     TOKEN_USAGE_FILE = os.path.abspath(file_path)
#
#     if force or not os.path.exists(TOKEN_USAGE_FILE):
#         try:
#             init_data = {
#                 "total_prompt_tokens": 0,
#                 "total_completion_tokens": 0,
#                 "total_tokens": 0,
#                 "call_count": 0
#             }
#             # 确保文件所在目录存在
#             os.makedirs(os.path.dirname(TOKEN_USAGE_FILE), exist_ok=True)
#
#             with open(TOKEN_USAGE_FILE, "w", encoding="utf-8") as f:
#                 json.dump(init_data, f, ensure_ascii=False, indent=2)
#             logger.info(f"初始化Token统计文件成功，路径：{TOKEN_USAGE_FILE}，初始数据：{init_data}")
#         except Exception as e:
#             logger.error(f"初始化Token统计文件失败：{str(e)}，路径：{TOKEN_USAGE_FILE}")
#             raise e


# def update_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int, token_stat: bool = True):
#     """
#     累加Token消耗到文件
#     :param prompt_tokens: 本次输入Token数
#     :param completion_tokens: 本次输出Token数
#     :param total_tokens: 本次总Token数
#     :param token_stat: 是否开启Token统计（开关控制）
#     """
#     if not token_stat:
#         logger.warning("Token统计功能已关闭，跳过累加")
#         return
#     if not TOKEN_USAGE_FILE:
#         raise ValueError("Token统计文件未初始化，请先调用init_token_file设置文件路径")
#     with TOKEN_LOCK:
#         try:
#             with open(TOKEN_USAGE_FILE, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             old_prompt = data["total_prompt_tokens"]
#             old_completion = data["total_completion_tokens"]
#             old_total = data["total_tokens"]
#             old_count = data["call_count"]
#
#             new_prompt = old_prompt + prompt_tokens
#             new_completion = old_completion + completion_tokens
#             new_total = old_total + total_tokens
#             new_count = old_count + 1
#
#             # 写入新数据
#             data.update({
#                 "total_prompt_tokens": new_prompt,
#                 "total_completion_tokens": new_completion,
#                 "total_tokens": new_total,
#                 "call_count": new_count
#             })
#             with open(TOKEN_USAGE_FILE, "w", encoding="utf-8") as f:
#                 json.dump(data, f, ensure_ascii=False, indent=2)
#
#
#
#         except json.JSONDecodeError as e:
#             logger.error(f"Token统计文件格式错误，重新初始化（原文件已备份）：{str(e)}")
#             # 备份错误文件
#             if os.path.exists(TOKEN_USAGE_FILE):
#                 bak_path = f"{TOKEN_USAGE_FILE}.bak"
#                 os.rename(TOKEN_USAGE_FILE, bak_path)
#                 logger.info(f"错误文件已备份至：{bak_path}")
#             # 重新初始化并累加本次数据
#             init_token_file(TOKEN_USAGE_FILE, force=True)
#             update_token_usage(prompt_tokens, completion_tokens, total_tokens, token_stat)
#         except Exception as e:
#             logger.error(f"Token累加失败：{str(e)}")
#             raise e


def get_token_usage(file_path: str) -> dict:
    """
    获取Token统计文件的累计数据
    :param file_path: Token统计文件路径
    :return: 累计统计数据
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        logger.warning(f"Token统计文件不存在：{file_path}，返回初始数据")
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"读取Token统计文件失败：{str(e)}")
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0
        }

# -------------------------- 重试装饰器实现 --------------------------
def retry_decorator(max_retries:  int = 3, retry_interval: float = 1.0):
    """
    ：支持异常重试 + 空DataFrame重试
    :param max_retries: 最大重试次数
    :param retry_interval: 每次重试的间隔（秒）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 执行原方法
                    result = func(self, *args, **kwargs)

                    # 检查返回值是否为空DataFrame
                    if isinstance(result, pd.DataFrame) and result.empty:
                        retry_count += 1
                        # 达到最大重试次数，记录警告并返回空DataFrame
                        if retry_count >= max_retries:
                            logger.warning(
                                f"【{func.__name__}】空数据重试次数已达上限（{max_retries}次），最终返回空数据 "
                                f"参数：args={args}, kwargs={kwargs}"
                            )
                            return result
                        # 未达最大次数，记录警告并等待后重试
                        logger.warning(
                            f"【{func.__name__}】返回空DataFrame，将进行第{retry_count}次重试（剩余{max_retries - retry_count}次） "
                            f"间隔{retry_interval}秒"
                        )
                        time.sleep(retry_interval)
                        continue  # 触发下一次重试
                    # 数据非空，直接返回
                    return result

                except Exception as e:
                    retry_count += 1
                    # 达到最大重试次数，记录错误并返回空DataFrame
                    if retry_count >= max_retries:
                        logger.error(
                            f"【{func.__name__}】异常重试次数已达上限（{max_retries}次），最终执行失败 "
                            f"参数：args={args}, kwargs={kwargs} 错误：{str(e)}"
                        )
                        return pd.DataFrame()
                    # 未达最大次数，记录警告并等待后重试
                    logger.warning(
                        f"【{func.__name__}】执行异常，将进行第{retry_count}次重试（剩余{max_retries - retry_count}次） "
                        f"间隔{retry_interval}秒 错误：{str(e)}"
                    )
                    time.sleep(retry_interval)
            # 兜底返回空DataFrame
            return pd.DataFrame()
        return wrapper
    return decorator

if __name__ == "__main__":
    result = select_top3_hot_sectors(trade_date="2024-02-20")
    print(result)

