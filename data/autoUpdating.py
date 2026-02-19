#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化增量更新脚本
核心：更新stock_basic全量数据 + 日线增量数据
"""
import datetime
import logging
import time

from data.data_cleaner import DataCleaner
from data.data_fetcher import data_fetcher
from utils.common_tools import (
    read_last_update_record,
    write_update_record,
    calc_incremental_date_range
)
from utils.db_utils import db
from utils.log_utils import logger

# 初始化核心组件
cleaner = DataCleaner()


# -------------------------- 核心更新函数 --------------------------
def update_stock_basic():
    """更新全市场股票基础信息表"""
    logger.info("===== 全量更新stock_basic表 =====")
    try:
        affected = cleaner.clean_and_insert_stockbase(table_name="stock_basic")
        return True
    except Exception as e:
        logger.error(f"stock_basic更新失败：{e}", exc_info=True)
        return False


def update_kline_day_incremental(date_list: list):
    """增量更新日线数据（按日期列表）- 批量请求优化版"""
    if not date_list:
        return False

    logger.info(f"===== 增量更新日线数据（{len(date_list)}天） =====")
    stock_codes = db.get_all_a_stock_codes()
    if not stock_codes:
        logger.error("无有效股票代码，终止日线更新")
        return False

    # 配置：每批请求的股票数量（根据接口限制调整，比如200个/批）
    BATCH_SIZE = 600
    total_affected = 0

    for trade_date in date_list:
        daily_affected = 0
        # 将股票代码按批次拆分，避免参数过长
        code_batches = [
            stock_codes[i:i + BATCH_SIZE]
            for i in range(0, len(stock_codes), BATCH_SIZE)
        ]

        for batch_idx, code_batch in enumerate(code_batches):
            try:
                # 拼接多股票代码（格式：000001.sz,000002.sz,...）
                ts_code_str = ",".join(code_batch)
                # 批量拉取该日期下这批股票的日线数据
                raw_df = data_fetcher.fetch_kline_day(
                    ts_code=ts_code_str,  # 多股票拼接字符串
                    start_date=trade_date,
                    end_date=trade_date
                )

                if raw_df.empty:
                    logger.debug(f"{trade_date} 第{batch_idx+1}批（{len(code_batch)}只股票）无数据")
                    continue

                # 清洗数据（保持原有逻辑）
                clean_df = cleaner._clean_kline_day_data(raw_df)
                if clean_df.empty:
                    continue

                # 批量入库（保持原有逻辑）
                affected = db.batch_insert_df(clean_df, "kline_day", ignore_duplicate=True)
                daily_affected += affected
                logger.debug(f"{trade_date} 第{batch_idx+1}批更新完成，影响行数：{affected}")

            except Exception as e:
                batch_info = f"{trade_date} 第{batch_idx+1}批（{len(code_batch)}只股票）"
                logger.error(f"{batch_info} 更新失败：{e}")
                # 可选：批次失败后，尝试单个股票重试（兜底机制）
                for ts_code in code_batch:
                    try:
                        raw_df_single = data_fetcher.fetch_kline_day(ts_code, trade_date, trade_date)
                        if not raw_df_single.empty:
                            clean_df_single = cleaner._clean_kline_day_data(raw_df_single)
                            if not clean_df_single.empty:
                                affected_single = db.batch_insert_df(clean_df_single, "kline_day", ignore_duplicate=True)
                                daily_affected += affected_single
                    except Exception as e_single:
                        logger.error(f"{ts_code} {trade_date} 单股重试更新失败：{e_single}")

        logger.info(f"{trade_date}日线更新完成，影响行数：{daily_affected}")
        total_affected += daily_affected

    logger.info(f"日线增量更新完成，累计影响行数：{total_affected}")
    return True

def update_index_daily(last_date):
    """更新指数信息表"""
    logger.info("===== 更新index_daily表 =====")
    try:
        for i in '000001.SH','399001.SZ','399006.SZ',"399107.SZ":

            cleaner.clean_and_insert_index_daily(
            ts_code = i,
            start_date = last_date.replace('-',''),
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            )
            logger.info(f"已更新{i}指数信息到今日，index_daily表")
        return True
    except Exception as e:
        logger.error(f"更新index_daily表失败：{e}", exc_info=True)
        return False

# -------------------------- 主执行流程 --------------------------
def startUpdating():
    logger.info("===== 启动增量更新脚本 =====")

    # 1. 读取上次更新记录
    last_record = read_last_update_record()
    last_date = last_record["last_update_date"]
    logger.info(f"===== 上次更新时间：{last_date} =====")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # 2. 计算增量日期
    inc_dates = calc_incremental_date_range(last_date, current_date)


    # 3. 执行更新
    updated_tables = []
    if update_stock_basic():
        updated_tables.append("stock_basic")
    if update_kline_day_incremental(inc_dates):
        updated_tables.append("kline_day")
    if update_index_daily(last_date):
        updated_tables.append("index_daily")

    # 4. 记录本次更新
    write_update_record(current_date, updated_tables)
    logger.info(f"===== 本次更新时间：{current_date} =====")
    logger.info(f"===== 更新完成 | 本次更新表：{updated_tables} =====")



if __name__ == "__main__":
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    startUpdating()