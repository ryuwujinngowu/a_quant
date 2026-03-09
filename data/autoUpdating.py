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
- 开市期间：每日 15:30 开始启动（原15点调整）
- 轮询机制：每隔 30 分钟尝试获取数据，防止接口数据未更新
- 终止条件：所有数据更新成功且数据库有实际行数影响，自动进入下一个交易日的等待
  （新增：数据库影响行数为0时，判定为更新失败，当日继续重试）
=====================================================================
"""
import time
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_cleaner import DataCleaner
from data_fetcher import data_fetcher
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
START_HOUR = 15  # 每日开始尝试运行的时间 (小时)
START_MINUTE = 30  # 每日开始尝试运行的时间 (分钟) 【新增】
RETRY_INTERVAL = 1800  # 重试间隔 (秒)，1800秒 = 30分钟


# -------------------------- 核心更新函数 (新增影响行数返回值) --------------------------
def update_stock_basic():
    """更新全市场股票基础信息表"""
    logger.info("===== 全量更新stock_basic表 =====")
    try:
        # 执行更新并获取影响行数（需确保clean_and_insert_stockbase返回影响行数）
        affected_rows = cleaner.clean_and_insert_stockbase(table_name="stock_basic")
        total_affected = affected_rows if affected_rows is not None else 0
        logger.info(f"stock_basic全量更新完成，影响行数：{total_affected}")
        return True, total_affected  # 【修改】返回(执行成功标识, 影响行数)
    except Exception as e:
        logger.error(f"stock_basic更新失败：{e}", exc_info=True)
        return False, 0


def update_stock_st_incremental(start_date: str, end_date: str):
    """增量更新ST股票数据（按时间段：start_date到end_date）"""
    if not start_date or not end_date:
        logger.warning("ST增量更新：开始/结束日期为空，跳过更新")
        return False, 0  # 【修改】返回(执行成功标识, 影响行数)

    logger.info(f"===== 增量更新ST股票数据（时间段：{start_date} ~ {end_date}） =====")
    try:
        affected_rows = cleaner.insert_stock_st(start_date=start_date, end_date=end_date)
        total_affected = affected_rows if affected_rows is not None else 0
        logger.info(f"ST股票增量更新完成，累计入库行数：{total_affected}")
        return True, total_affected  # 【修改】返回(执行成功标识, 影响行数)
    except Exception as e:
        logger.error(f"ST数据更新失败（{start_date}~{end_date}）：{e}", exc_info=True)
        return False, 0


def update_kline_day_incremental(date_list: list):
    """增量更新日线数据（按日期列表）- 批量请求优化版"""
    if not date_list:
        return False, 0  # 【修改】返回(执行成功标识, 影响行数)

    logger.info(f"===== 增量更新日线数据（{len(date_list)}天） =====")
    stock_codes = db.get_all_a_stock_codes()
    if not stock_codes:
        logger.error("无有效股票代码，终止日线更新")
        return False, 0

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
    return True, total_affected  # 【修改】返回(执行成功标识, 影响行数)


def update_index_daily(last_date):
    """更新指数信息表（兼容clean_and_insert_index_daily返回None的情况）"""
    logger.info("===== 更新index_daily表 =====")
    try:
        total_affected = 0
        for i in '000001.SH', '399001.SZ', '399006.SZ', "399107.SZ":
            # 【关键修正】处理返回None的情况，转为0
            affected_rows = cleaner.clean_and_insert_index_daily(
                ts_code=i,
                start_date=last_date.replace('-', ''),
                end_date=datetime.datetime.now().strftime("%Y%m%d")
            )
            # 新增：将None转为0，避免sum时出现TypeError
            affected_rows = affected_rows if affected_rows is not None else 0
            total_affected += affected_rows
            logger.info(f"已更新{i}指数信息到今日，index_daily表，影响行数：{affected_rows}")
        logger.info(f"index_daily更新完成，累计影响行数：{total_affected}")
        return True, total_affected
    except Exception as e:
        logger.error(f"更新index_daily表失败：{e}", exc_info=True)
        return False, 0


# -------------------------- 主执行流程 (新增影响行数校验) --------------------------
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
    affected_rows_dict = {}  # 【新增】记录各表影响行数

    # 执行各更新任务并记录结果和影响行数
    success_flags['stock_basic'], affected_rows_dict['stock_basic'] = update_stock_basic()
    if success_flags['stock_basic']: updated_tables.append("stock_basic")

    success_flags['stock_st'], affected_rows_dict['stock_st'] = update_stock_st_incremental(last_date, current_date)
    if success_flags['stock_st']: updated_tables.append("stock_st_daily")

    success_flags['kline'], affected_rows_dict['kline'] = update_kline_day_incremental(inc_dates)
    if success_flags['kline']: updated_tables.append("kline_day")

    success_flags['index'], affected_rows_dict['index'] = update_index_daily(last_date)
    if success_flags['index']: updated_tables.append("index_daily")

    # 4. 【核心修改】计算总影响行数，判断是否真的更新成功
    total_affected_rows = sum(affected_rows_dict.values())
    logger.info(f"本次更新各表影响行数：{affected_rows_dict}，总影响行数：{total_affected_rows}")

    # 判定标准：
    # - 核心表 (stock_basic, kline_day) 执行成功
    # - 总影响行数 > 0（排除接口数据未更新导致的0行影响）
    core_success = success_flags['stock_basic'] and success_flags['kline']
    has_actual_update = total_affected_rows > 0
    is_success = core_success and has_actual_update

    # 5. 记录本次更新
    write_update_record(current_date, updated_tables)
    logger.info(f"===== 本次更新时间：{current_date} =====")
    logger.info(f"===== 更新完成 | 本次更新表：{updated_tables} | 总影响行数：{total_affected_rows} =====")

    # 构造返回消息（新增影响行数信息）
    msg_content = (
        f"更新日期: {current_date}\n"
        f"更新表: {updated_tables}\n"
        f"各表影响行数: {affected_rows_dict}\n"
        f"总影响行数: {total_affected_rows}\n"
        f"状态: {'成功' if is_success else '部分失败/无有效数据更新'}"
    )

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
        logger.warning(f"交易日历查询失败: {e}，假设今日不交易（保守策略）")
        return False


def wait_until_target_time(target_hour, target_minute):
    """
    【修改】阻塞直到到达目标时间 (如 15:30)，支持分钟参数
    """
    now = datetime.datetime.now()
    target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

    if now >= target_time:
        # 如果现在已经过了目标时间，不需要等待，直接开始
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

        # 3. 是交易日，等待到 15:30 【修改】传入小时+分钟参数
        wait_until_target_time(START_HOUR, START_MINUTE)

        # 4. 开始循环尝试更新，直到成功
        update_success = False
        while not update_success:
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"开始尝试数据更新: {current_time_str}")

            try:
                update_success, msg = startUpdating()

                if update_success:
                    logger.info("数据更新全部成功！（有实际行数影响）")
                    # 发送微信推送
                    try:
                        send_wechat_message(f"【量化数据更新成功】{today_str}", msg)
                        logger.info("微信推送已发送。")
                    except Exception as push_e:
                        logger.error(f"微信推送发送失败: {push_e}")

                    last_run_date = today_str  # 标记今日已完成
                else:
                    logger.warning(f"更新未完全成功/无有效数据更新，{RETRY_INTERVAL / 60}分钟后重试...")
                    # 发送失败告警（可选，根据需要开启）
                    # send_wechat_message(f"【量化数据更新异常】{today_str}", msg)
                    time.sleep(RETRY_INTERVAL)

            except Exception as e:
                logger.error(f"主循环发生严重异常: {e}", exc_info=True)
                time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    # 启动服务模式
    run_server()