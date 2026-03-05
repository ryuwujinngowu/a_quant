import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import time
from pathlib import Path
import functools
from typing import List, Dict, Optional
from typing import Tuple
from config.config import MAIN_BOARD_LIMIT_UP_RATE, STAR_BOARD_LIMIT_UP_RATE, BJ_BOARD_LIMIT_UP_RATE
from utils.db_utils import db
from utils.log_utils import logger
from typing import List, Dict



# 增量更新配置（可统一维护）
UPDATE_RECORD_FILE = Path(__file__).parent.parent / "data" / "update_record.json"


default_exclude = [
    "融资融券", "转融券标的", "标普道琼斯A股", '核准制次新股', '腾讯概念', '阿里巴巴概念', '抖音概念','ST板块',
    "MSCI概念", "深股通", "沪股通", '一带一路', '新股与次新股', '节能环保','稀缺资源','电子商务','俄乌冲突概念',
    "同花顺漂亮100", "富时罗素概念", "富时罗素概念股", '比亚迪概念', '5G', '小金属概念','参股银行','锂电池','spacex',
    "央企国资改革", "地方国资改革", "证金持股", '新能源汽车', '次新股', '宁德时代概念','人民币贬值受益','中俄贸易概念',
    "汇金持股", "养老金持股", "QFII重仓", "专精特新", 'MSCI中国', '半年报预增', '华为概念', '光伏概念', '储能','一季报预增'
]


def calc_limit_up_price(ts_code: str, pre_close: float) -> float:
    """
    计算股票涨停价（适配不同板块涨跌幅限制，融合调试日志+强类型+完整校验）
    :param ts_code: 股票代码（如600000.SH/300001.SZ/831010.BJ）
    :param pre_close: 前一日收盘价
    :return: 涨停价格（保留2位小数，无效值返回0.0）
    """
    if not pre_close or pre_close <= 0:
        logger.debug(f"[{ts_code}] 前收盘价无效（pre_close={pre_close}），涨停价返回0.0")
        return 0.0
    # 1. 判断板块类型，匹配对应涨跌幅
    if ts_code.endswith(".BJ"):  # 北交所
        limit_rate = BJ_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith(("300", "301", "302")) or (ts_code.startswith("3") and ts_code.endswith(".SZ")):  # 创业板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith("688"):  # 科创板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE  # 科创板和创业板涨跌幅一致（20%）
    else:  # 主板（60/00开头）
        limit_rate = MAIN_BOARD_LIMIT_UP_RATE
    limit_up_price = pre_close * (1 + limit_rate)
    limit_up_price = round(limit_up_price, 2)
    logger.debug(f"[{ts_code}] 前收盘价={pre_close}，涨停幅度={limit_rate}，涨停价={limit_up_price}")
    return round(limit_up_price, 2)


def calc_limit_down_price(ts_code: str, pre_close: float) -> float:
    """
    计算股票跌停价（和涨停价逻辑完全对齐，适配不同板块涨跌幅限制）
    :param ts_code: 股票代码
    :param pre_close: 前一日收盘价
    :return: 跌停价格（保留2位小数，无效值返回0）
    """
    if not pre_close or pre_close <= 0:
        logger.debug(f"[{ts_code}] 前收盘价无效（pre_close={pre_close}），涨停价返回0.0")
        return 0.0
        # 1. 判断板块类型，匹配对应涨跌幅
    if ts_code.endswith(".BJ"):  # 北交所
        limit_rate = BJ_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith(("300", "301", "302")) or (ts_code.startswith("3") and ts_code.endswith(".SZ")):  # 创业板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE
    elif ts_code.startswith("688"):  # 科创板
        limit_rate = STAR_BOARD_LIMIT_UP_RATE  # 科创板和创业板涨跌幅一致（20%）
    else:  # 主板（60/00开头）
        limit_rate = MAIN_BOARD_LIMIT_UP_RATE
    # 跌停价公式：前收盘价 × (1 - 涨跌幅系数)，四舍五入保留2位小数
    limit_down_price = pre_close * (1 - limit_rate)
    logger.debug(f"[{ts_code}] 前收盘价={pre_close}，跌停幅度={limit_rate}，跌停价={limit_down_price}")
    return  round(limit_down_price, 2)


# def check_stock_has_limit_up(ts_code_list: List[str], end_date: str, day_count: int = 10) -> Dict[str, bool]:
#     """
#     【修复后】批量判断股票近N个交易日是否有涨停（无未来函数+日期格式正确+性能优化）
#     :param ts_code_list: 待判断股票代码列表
#     :param end_date: 当前交易日（兼容YYYYMMDD/YYYY-MM-DD）
#     :param day_count: 回溯交易日数，默认10
#     :return: {ts_code: True=近10日有涨停, False=无涨停}
#     """
#     # 1. 入参校验
#     if not ts_code_list or day_count <= 0 or not end_date:
#         logger.warning("check_stock_has_limit_up 入参无效，返回全True（保守保留所有股票）")
#         return {ts_code: True for ts_code in ts_code_list}
#
#     # 2. 统一日期格式：最终转为YYYY-MM-DD，和kline_day表完全对齐
#     try:
#         if len(end_date) == 8 and end_date.isdigit():
#             end_date_dt = datetime.strptime(end_date, "%Y%m%d")
#             end_date_format = end_date_dt.strftime("%Y-%m-%d")
#         else:
#             end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
#             end_date_format = end_date
#     except ValueError as e:
#         logger.error(f"日期格式解析失败：{end_date}，错误：{e}，返回全True")
#         return {ts_code: True for ts_code in ts_code_list}
#
#     try:
#         # 截止到当前交易日的前一天，彻底避免未来函数
#         pre_end_date = (end_date_dt - timedelta(days=1)).strftime("%Y-%m-%d")
#         # 直接查询最近N个交易日，无需全量查询
#         sql_trade_date = """
#                          SELECT cal_date \
#                          FROM trade_cal
#                          WHERE cal_date <= %s \
#                            AND is_open = 1
#                          ORDER BY cal_date DESC
#                              LIMIT %s \
#                          """
#         trade_date_result = db.query(sql_trade_date, (pre_end_date, day_count))
#         if not trade_date_result or len(trade_date_result) < day_count:
#             logger.warning(f"可回溯交易日不足{day_count}个，返回全True")
#             return {ts_code: True for ts_code in ts_code_list}
#
#         # 提取交易日列表，格式和kline_day表一致（YYYY-MM-DD）
#         target_dates = [row["cal_date"].strftime("%Y-%m-%d") for row in trade_date_result]
#         logger.debug(f"近{day_count}个回溯交易日：{target_dates}")
#
#     except Exception as e:
#         logger.error(f"获取交易日历失败：{e}，返回全True")
#         return {ts_code: True for ts_code in ts_code_list}
#
#     # 4. 批量查询回溯期内所有候选股的日线数据
#     try:
#         sql = """
#               SELECT ts_code, trade_date, pre_close, close
#               FROM kline_day
#               WHERE ts_code IN %s \
#                 AND trade_date IN %s \
#               """
#         result = db.query(sql, (tuple(ts_code_list), tuple(target_dates)))
#         if not result:
#             logger.warning("回溯期内无有效日线数据，返回全True")
#             return {ts_code: True for ts_code in ts_code_list}
#
#     except Exception as e:
#         logger.error(f"查询日线数据失败：{e}，返回全True")
#         return {ts_code: True for ts_code in ts_code_list}
#
#     # 5. 分组判断每只股票是否有涨停
#     stock_daily_map = defaultdict(list)
#     for row in result:
#         stock_daily_map[row["ts_code"]].append(row)
#
#     # 初始化结果：默认无涨停
#     result_dict = {ts_code: False for ts_code in ts_code_list}
#     for ts_code, daily_list in stock_daily_map.items():
#         for daily in daily_list:
#             pre_close = daily["pre_close"]
#             close = daily["close"]
#
#             if pre_close <= 0 or close <= 0:
#                 continue
#             # 用全局统一的涨停价计算函数
#             limit_up_price = calc_limit_up_price_common(ts_code, pre_close)
#             logger.debug(
#                 f"【涨停判断明细】{ts_code} 日期：{daily['trade_date']} | 前收：{pre_close} | 收盘价：{close} | 涨停价：{limit_up_price} | 是否满足：{close >= limit_up_price - 0.01}")
#             # =================================================
#             if limit_up_price <= 0:
#                 continue
#             # 涨停判断标准：收盘价≥涨停价-0.01（兼容价格精度）
#             if close >= limit_up_price - 0.01:
#                 result_dict[ts_code] = True
#                 break  # 有1次涨停就终止
#
#     logger.info(f"近{day_count}日涨停判断完成：有涨停基因个股{sum(result_dict.values())}只，总候选{len(ts_code_list)}只")
#     return result_dict
#



def filter_st_stocks(ts_code_list: List[str], trade_date: str) -> List[str]:
    """
    【唯一ST过滤方法】批量过滤指定交易日的ST/*ST股票（先动态入库当日ST数据，再1次查询比对）
    :param ts_code_list: 待过滤的股票代码列表（如['000001.SZ', '600000.SH']）
    :param trade_date: 交易日（兼容YYYYMMDD/YYYY-MM-DD格式）
    :return: 过滤后的正常股票代码列表（已剔除所有ST股）
    """
    # 3. 1次数据库查询：获取当日最新ST股票代码（入库后查询，确保数据最新）
    try:
        sql = """
              SELECT DISTINCT ts_code
              FROM stock_risk_warning
              WHERE trade_date = %s \
              """
        st_result = db.query(sql, trade_date)
        st_code_set = set([row["ts_code"] for row in st_result]) if st_result else set()
        # 4. 批量比对：保留非ST股票
        normal_codes = [ts_code for ts_code in ts_code_list if ts_code not in st_code_set]
        # 5. 关键日志：明确过滤效果
        filter_count = len(ts_code_list) - len(normal_codes)
        logger.info(
            f"[filter_st_stocks] 交易日{trade_date} ST过滤完成 | "
            f"原始股票数：{len(ts_code_list)} | 剔除ST股数：{filter_count} | 剩余正常股票数：{len(normal_codes)}"
        )
        # 调试日志：打印被剔除的ST股（可选）
        if filter_count > 0:
            st_removed = [ts_code for ts_code in ts_code_list if ts_code in st_code_set]
            logger.debug(f"[filter_st_stocks] 当日被剔除的ST股：{st_removed}")
        return normal_codes
    except Exception as e:
        logger.error(f"[filter_st_stocks] 批量过滤ST股票失败 | 交易日：{trade_date} | 错误：{e}", exc_info=True)
        return []

def get_trade_dates(start_date: str, end_date: str) -> List[str]:
    """
    通用交易日历查询方法：从已入库的trade_cal表中查询指定时间段的有效交易日
    :param start_date: 开始日期，格式yyyy-mm-dd
    :param end_date: 结束日期，格式yyyy-mm-dd
    :return: 按时间升序排列的交易日字符串列表，如["2025-01-01", "2025-01-02"]
    :raise RuntimeError: 查询失败/无有效交易日时抛出异常
    """
    # 1. 基础参数校验
    if not (isinstance(start_date, str) and isinstance(end_date, str)):
        logger.error(f"交易日查询失败：日期格式错误，start_date={start_date}, end_date={end_date}")
        raise RuntimeError("交易日查询失败：日期必须为字符串格式（yyyy-mm-dd）")

    # 2. 执行SQL查询
    sql = """
          SELECT cal_date
          FROM trade_cal
          WHERE cal_date BETWEEN %s AND %s 
            AND is_open = 1
          ORDER BY cal_date ASC
          """
    try:
        df = db.query(sql, params=(start_date, end_date), return_df=True)
    except Exception as e:
        logger.error(f"交易日查询数据库异常：{str(e)}")
        raise RuntimeError(f"交易日查询失败：{str(e)}")

    # 3. 空数据校验
    if df.empty:
        logger.error(f"[{start_date} 至 {end_date}] 时间段内无有效交易日")
        raise RuntimeError(f"[{start_date} 至 {end_date}] 时间段内无有效交易日")

    # 4. 转换为字符串列表返回
    trade_dates = df["cal_date"].astype(str).tolist()
    logger.debug(f"交易日查询成功：[{start_date} 至 {end_date}] 共{len(trade_dates)}个有效交易日")
    return trade_dates


def get_daily_kline_data(trade_date: str, ts_code_list: List[str] = None) -> pd.DataFrame:
    """
    获取指定日期的日线数据（向后兼容优化版）
    【新增功能】支持仅查询指定股票的日线数据，大幅提升性能
    【向后兼容】不传ts_code_list时，保持原有全市场查询逻辑，不影响现有调用

    :param trade_date: 交易日（兼容YYYY-MM-DD/YYYYMMDD格式）
    :param ts_code_list: 【可选】指定股票代码列表，仅查询这些股票的日线数据
    :return: 日线数据DataFrame
    """
    # 1. 日期格式化（兼容两种格式）
    logger.debug(
        f"开始获取日线数据: {trade_date}" + (f"，指定股票数：{len(ts_code_list)}" if ts_code_list else "，全市场"))
    trade_date_format = trade_date.replace("-", "")

    # 2. 构建SQL和参数（根据是否指定股票动态调整）
    if ts_code_list:
        # 【新增】仅查询指定股票
        if not isinstance(ts_code_list, (list, tuple, set)):
            logger.error(f"ts_code_list格式错误，必须是列表/元组/集合，当前类型：{type(ts_code_list)}")
            return pd.DataFrame()
        if not ts_code_list:
            logger.warning("ts_code_list为空，返回空DataFrame")
            return pd.DataFrame()

        # 构建带IN条件的SQL
        sql = """
              SELECT *
              FROM kline_day
              WHERE trade_date = %s 
                AND ts_code IN %s
              """
        params = (trade_date_format, tuple(ts_code_list))
    else:
        # 【原有逻辑】查询全市场（保持向后兼容）
        sql = """
              SELECT *
              FROM kline_day
              WHERE trade_date = %s 
              """
        params = (trade_date_format,)

    # 3. 执行查询（保持原有逻辑不变）
    try:
        df = db.query(sql, params=params, return_df=True)
    except Exception as e:
        logger.error(f"{trade_date} 日线数据查询失败：{str(e)}")
        return pd.DataFrame()
    # 4. 返回结果（保持原有逻辑不变）
    if df is not None and not df.empty:
        logger.debug(f"{trade_date} 日线数据从数据库读取完成，行数：{len(df)}")
        return df
    else:
        logger.error(f"{trade_date} 日线数据拉取失败，跳过当日")
        return pd.DataFrame()


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
    # logger.info('*'*60)
    # logger.info('！！！！！！历史数据，非实时数据，仅供模拟训练！！！！！！！')
    # logger.info('*'*60)
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
    logger.debug(f"{trade_date}共查询到{len(result_df)}  只符合条件的股票")

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
    logger.debug(f"开始统计{input_stock_count}只股票的题材覆盖情况，黑名单题材数：{len(exclude_concepts)}")

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

    logger.debug(
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
        "last_update_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        record["last_update_date"] = record.get("last_update_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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



def get_stocks_in_sector(sector_name: str) -> List[str]:
    """
    【通用工具】从stock_basic表查询指定板块/概念对应的所有股票代码
    精准匹配逗号分隔的concept_text字段，避免模糊匹配错误
    :param sector_name: 板块/概念完整名称（如"人工智能"、"算力"）
    :return: 该板块下的所有股票ts_code列表，查询失败/无结果返回空列表
    """
    sector_name_clean = str(sector_name).strip()
    # 核心SQL：用MySQL原生FIND_IN_SET精准匹配逗号分隔的概念
    # 加入LOWER兼容大小写不一致的场景，DISTINCT去重避免重复数据
    sql = """
        SELECT DISTINCT ts_code 
        FROM stock_basic 
        WHERE FIND_IN_SET(%s, concept_tags) > 0
          """
    # 执行参数化查询（防SQL注入，完全适配项目db工具的调用方式）
    result_df = db.query(sql, params=(sector_name_clean,))
    return result_df



def get_sector_stock_daily_data(sector_name: str, trade_date: str) -> pd.DataFrame:
    """
    【核心工具】查询指定板块在指定交易日的所有股票日线数据
    完全匹配需求：先查板块对应股票→再查这些股票当日日线
    :param sector_name: 板块/概念完整名称（如"人工智能"、"算力"）
    :param trade_date: 交易日，格式严格为YYYYMMDD（如"20260101"，与项目全局格式对齐）
    :return: 该板块指定交易日的全量日线数据DataFrame，查询失败/无结果返回空DataFrame
    """
    # 入参合法性校验
    if not sector_name or not str(sector_name).strip():
        logger.warning("[get_sector_stock_daily_data] 板块名称不能为空")
        return pd.DataFrame()

    trade_date_clean = str(trade_date).strip()
    if len(trade_date_clean) != 8 or not trade_date_clean.isdigit():
        logger.warning(f"[get_sector_stock_daily_data] 交易日期格式错误，要求YYYYMMDD，传入：{trade_date}")
        return pd.DataFrame()

    sector_name_clean = str(sector_name).strip()

    try:
        # 步骤1：获取板块对应的全量股票代码
        ts_code_list = get_stocks_in_sector(sector_name_clean)
        if not ts_code_list:
            logger.warning(f"[get_sector_stock_daily_data] 板块[{sector_name_clean}]无对应股票，返回空数据")
            return pd.DataFrame()

        # 步骤2：批量查询这些股票在指定交易日的日线数据
        # 表名kline_day与项目全局命名对齐，如需修改请调整表名即可
        sql = """
              SELECT *
              FROM kline_day
              WHERE ts_code IN %s
                AND trade_date = %s \
              """
        # IN查询必须传元组，适配Python MySQL参数化规范
        result_df = db.query(sql, params=(tuple(ts_code_list), trade_date_clean))

        if result_df.empty:
            logger.warning(f"[get_sector_stock_daily_data] 板块[{sector_name_clean}]在{trade_date_clean}无有效日线数据")
            return pd.DataFrame()

        logger.info(
            f"[get_sector_stock_daily_data] 板块[{sector_name_clean}]在{trade_date_clean}查询到{len(result_df)}条日线数据")
        return result_df

    except Exception as e:
        logger.error(
            f"[get_sector_stock_daily_data] 查询板块日线失败，板块：{sector_name_clean}，日期：{trade_date_clean}，错误：{str(e)}",
            exc_info=True
        )
        return pd.DataFrame()

if __name__ == "__main__":
    # result = select_top3_hot_sectors(trade_date="2024-02-20")
    # print(result)
    print(get_stocks_in_sector('军工'))
    # pass

