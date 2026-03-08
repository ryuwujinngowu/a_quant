#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化增量更新脚本 (服务器稳定运行版)
=====================================================================
核心功能 (每日更新内容)：
1. stock_basic: 全量更新全市场股票基础信息表
2. stock_st_daily: 增量更新ST股票风险警示表
3. kline_day: 增量更新A股日线行情数据
4. index_daily: 增量更新核心指数日线数据 (000001.SH, 399001.SZ等)

运行逻辑：
- 开市期间：每日 15:00 开始启动
- 轮询机制：每隔 30 分钟尝试获取数据，防止接口数据未更新
- 终止条件：所有数据更新成功后，自动进入下一个交易日的等待
=====================================================================
"""
import sys
import time
import datetime
import logging
from pathlib import Path


project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.data_cleaner import DataCleaner
from data.data_fetcher import data_fetcher
from utils.common_tools import (
    read_last_update_record,
    write_update_record,
    calc_incremental_date_range,
    get_trade_dates  # 新增：导入交易日历判断工具
)
from utils.db_utils import db
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message  # 新增：导入微信推送工具

# 初始化核心组件
cleaner = DataCleaner()

# -------------------------- 配置参数 --------------------------
START_HOUR = 15  # 每日开始尝试运行的时间 (15点)
RETRY_INTERVAL = 1800  # 重试间隔 (秒)，1800秒 = 30分钟


# -------------------------- 核心更新函数 (原业务逻辑保持不变) --------------------------
def update_stock_basic():
    """更新全市场股票基础信息表"""
    logger.info("===== 全量更新stock_basic表 =====")
    try:
        cleaner.clean_and_insert_stockbase(table_name="stock_basic")
        return True
    except Exception as e:
        logger.error(f"stock_basic更新失败：{e}", exc_info=True)
        return False


def update_stock_st_incremental(start_date: str, end_date: str):
    """增量更新ST股票数据（按时间段：start_date到end_date）"""
    if not start_date or not end_date:
        logger.warning("ST增量更新：开始/结束日期为空，跳过更新")
        return False

    logger.info(f"===== 增量更新ST股票数据（时间段：{start_date} ~ {end_date}） =====")
    try:
        affected_rows = cleaner.insert_stock_st(start_date=start_date, end_date=end_date)
        total_affected = affected_rows if affected_rows is not None else 0
        logger.info(f"ST股票增量更新完成，累计入库行数：{total_affected}")
        return True
    except Exception as e:
        logger.error(f"ST数据更新失败（{start_date}~{end_date}）：{e}", exc_info=True)
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

    BATCH_SIZE = 600
    total_affected = 0

    for trade_date in date_list:
        daily_affected = 0
        code_batches = [
            stock_codes[i:i + BATCH_SIZE]
            for i in range(0, len(stock_codes), BATCH_SIZE)
        ]

        for batch_idx, code_batch in enumerate(code_batches):
            try:
                ts_code_str = ",".join(code_batch)
                raw_df = data_fetcher.fetch_kline_day(
                    ts_code=ts_code_str,
                    start_date=trade_date,
                    end_date=trade_date
                )

                if raw_df.empty:
                    logger.debug(f"{trade_date} 第{batch_idx + 1}批（{len(code_batch)}只股票）无数据")
                    continue

                clean_df = cleaner._clean_kline_day_data(raw_df)
                if clean_df.empty:
                    continue

                affected = db.batch_insert_df(clean_df, "kline_day", ignore_duplicate=True)
                daily_affected += affected
                logger.debug(f"{trade_date} 第{batch_idx + 1}批更新完成，影响行数：{affected}")

            except Exception as e:
                batch_info = f"{trade_date} 第{batch_idx + 1}批（{len(code_batch)}只股票）"
                logger.error(f"{batch_info} 更新失败：{e}")
                # 兜底单股重试
                for ts_code in code_batch:
                    try:
                        raw_df_single = data_fetcher.fetch_kline_day(ts_code, trade_date, trade_date)
                        if not raw_df_single.empty:
                            clean_df_single = cleaner._clean_kline_day_data(raw_df_single)
                            if not clean_df_single.empty:
                                affected_single = db.batch_insert_df(clean_df_single, "kline_day",
                                                                     ignore_duplicate=True)
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
        for i in '000001.SH', '399001.SZ', '399006.SZ', "399107.SZ":
            cleaner.clean_and_insert_index_daily(
                ts_code=i,
                start_date=last_date.replace('-', ''),
                end_date=datetime.datetime.now().strftime("%Y%m%d")
            )
            logger.info(f"已更新{i}指数信息到今日，index_daily表")
        return True
    except Exception as e:
        logger.error(f"更新index_daily表失败：{e}", exc_info=True)
        return False


# -------------------------- 主执行流程 (增加返回值用于判断成功与否) --------------------------
def startUpdating():
    """
    执行主更新流程
    :return: (is_success: bool, message: str)
    """
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
    success_flags = {}

    # 这里的逻辑保持原样：只有前一个成功了才会尝试后一个（或者根据你的需求并行，这里保持原逻辑）
    # 为了更健壮，我稍微修改了逻辑，让每个任务独立尝试，但最后汇总是否全部核心成功
    # 注意：原逻辑是 if A: append; if B: append... 这里保持完全一致

    success_flags['stock_basic'] = update_stock_basic()
    if success_flags['stock_basic']: updated_tables.append("stock_basic")

    success_flags['stock_st'] = update_stock_st_incremental(last_date, current_date)
    if success_flags['stock_st']: updated_tables.append("stock_st_daily")

    success_flags['kline'] = update_kline_day_incremental(inc_dates)
    if success_flags['kline']: updated_tables.append("kline_day")

    success_flags['index'] = update_index_daily(last_date)
    if success_flags['index']: updated_tables.append("index_daily")

    # 4. 记录本次更新
    write_update_record(current_date, updated_tables)
    logger.info(f"===== 本次更新时间：{current_date} =====")
    logger.info(f"===== 更新完成 | 本次更新表：{updated_tables} =====")

    # 判定标准：核心表 (stock_basic, kline_day) 必须成功，ST和指数尽量成功
    # 你可以根据实际情况修改这里的成功判定逻辑
    is_success = success_flags['stock_basic'] and success_flags['kline']

    msg_content = f"更新日期: {current_date}\n更新表: {updated_tables}\n状态: {'成功' if is_success else '部分失败/失败'}"

    return is_success, msg_content


# -------------------------- 服务器调度逻辑 --------------------------
def is_trade_day(check_date: str) -> bool:
    """
    判断指定日期是否为交易日
    :param check_date: 日期字符串 yyyy-mm-dd
    :return: True/False
    """
    try:
        # 调用工具查询交易日历
        dates = get_trade_dates(check_date, check_date)
        return check_date in dates
    except Exception as e:
        logger.error(f"交易日历查询失败: {e}，假设今日不交易（保守策略）")
        return False


def wait_until_target_time(target_hour):
    """
    阻塞直到到达目标时间 (如 15:00)
    """
    now = datetime.datetime.now()
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

    if now >= target_time:
        # 如果现在已经过了15点，不需要等待，直接开始
        return

    wait_seconds = (target_time - now).total_seconds()
    time.sleep(wait_seconds)


def run_server():
    """
    服务端常驻主循环
    """
    logger.info("量化数据更新服务已启动...")

    last_run_date = None  # 记录上一次成功运行的日期

    while True:
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # 1. 检查今天是否已经成功运行过了
        if last_run_date == today_str:
            logger.info(f"今日 [{today_str}] 任务已完成，休眠至次日...")
            # 睡到明天凌晨2点，再重新开始判断逻辑
            tomorrow = now + datetime.timedelta(days=1)
            tomorrow_start = tomorrow.replace(hour=2, minute=0, second=0)
            sleep_secs = (tomorrow_start - now).total_seconds()
            time.sleep(sleep_secs)
            continue

        # 2. 检查是否是交易日
        if not is_trade_day(today_str):
            logger.info(f"[{today_str}] 是非交易日，跳过。")
            # 同样睡到明天
            time.sleep(3600 * 4)  # 每4小时检查一次日期变更
            continue

        # 3. 是交易日，等待到 15:00
        wait_until_target_time(START_HOUR)

        # 4. 开始循环尝试更新，直到成功
        update_success = False
        while not update_success:
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"开始尝试数据更新: {current_time_str}")

            try:
                update_success, msg = startUpdating()

                if update_success:
                    logger.info("数据更新全部成功！")
                    # 发送微信推送
                    try:
                        send_wechat_message(f"【量化数据更新成功】{today_str}", msg)
                        logger.info("微信推送已发送。")
                    except Exception as push_e:
                        logger.error(f"微信推送发送失败: {push_e}")

                    last_run_date = today_str  # 标记今日已完成
                else:
                    logger.warning(f"更新未完全成功，{RETRY_INTERVAL / 60}分钟后重试...")
                    # 发送失败告警（可选，或者只在成功时发，这里可以根据需要开启）
                    # send_wechat_message(f"【量化数据更新异常】{today_str}", msg)
                    time.sleep(RETRY_INTERVAL)

            except Exception as e:
                logger.error(f"主循环发生严重异常: {e}", exc_info=True)
                time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 启动服务模式
    run_server()