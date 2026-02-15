#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化增量更新脚本
核心：更新stock_basic全量数据 + 日线增量数据
"""
import logging
import datetime
from utils.log_utils import logger
from utils.db_utils import db
from utils.common_tools import (
    read_last_update_record,
    write_update_record,
    calc_incremental_date_range
)
from data.data_fetcher import data_fetcher
from data.data_cleaner import DataCleaner

# 初始化核心组件
cleaner = DataCleaner()


# -------------------------- 核心更新函数 --------------------------
def update_stock_basic():
    """更新全市场股票基础信息表"""
    logger.info("===== 全量更新stock_basic表 =====")
    try:
        affected = cleaner.clean_and_insert_stockbase(table_name="stock_basic")
        logger.info(f"stock_basic更新完成，影响行数：{affected}")
        return True
    except Exception as e:
        logger.error(f"stock_basic更新失败：{e}", exc_info=True)
        return False


def update_kline_day_incremental(date_list: list):
    """增量更新日线数据（按日期列表）"""
    if not date_list:
        return False

    logger.info(f"===== 增量更新日线数据（{len(date_list)}天） =====")
    stock_codes = db.get_all_a_stock_codes()
    if not stock_codes:
        logger.error("无有效股票代码，终止日线更新")
        return False

    total_affected = 0
    for trade_date in date_list:
        daily_affected = 0
        for ts_code in stock_codes:
            try:
                # 拉取单股票单日数据
                raw_df = data_fetcher.fetch_kline_day(ts_code, trade_date, trade_date)
                if raw_df.empty:
                    continue
                # 清洗+入库
                clean_df = cleaner._clean_kline_day_data(raw_df)
                if clean_df.empty:
                    continue
                affected = db.batch_insert_df(clean_df, "kline_day", ignore_duplicate=True)
                daily_affected += affected
            except Exception as e:
                logger.error(f"{ts_code} {trade_date}更新失败：{e}")

        logger.info(f"{trade_date}日线更新完成，影响行数：{daily_affected}")
        total_affected += daily_affected

    logger.info(f"日线增量更新完成，累计影响行数：{total_affected}")
    return True


# -------------------------- 主执行流程 --------------------------
def startUpdating():
    logger.info("===== 启动增量更新脚本 =====")

    # 1. 读取上次更新记录
    last_record = read_last_update_record()
    last_date = last_record["last_update_date"]
    logger.info(f"===== 上次更新时间：{last_date} =====")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 2. 计算增量日期
    inc_dates = calc_incremental_date_range(last_date, current_date)

    # 3. 执行更新
    updated_tables = []
    if update_stock_basic():
        updated_tables.append("stock_basic")
    if update_kline_day_incremental(inc_dates):
        updated_tables.append("kline_day")

    # 4. 记录本次更新
    write_update_record(current_date, updated_tables)
    logger.info(f"===== 本次更新时间：{current_date} =====")
    logger.info(f"===== 更新完成 | 本次更新表：{updated_tables} =====")



if __name__ == "__main__":
    # 日志兜底配置（若未复用log_utils）
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    startUpdating()