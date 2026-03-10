#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化增量更新脚本 (服务器稳定运行版)
=====================================================================
核心功能 (每日更新内容)：
1. stock_basic   : 全量更新全市场股票基础信息表
2. stock_st_daily: 增量更新ST股票风险警示表
3. kline_day     : 增量更新A股日线行情数据
4. index_daily   : 增量更新核心指数日线数据 (000001.SH, 399001.SZ等)

运行逻辑：
- 每日 15:30 开始首次尝试
- 每隔 30 分钟重试一次，最多重试 MAX_RETRY_TIMES 次
- 成功判定：kline_day 入库行数 >= 单日预期最低行数 * 待补天数
- 失败推送：仅第 1 次失败 + 最终放弃时各推送一次，避免消息轰炸
- 记录写入：仅更新成功后才写 last_update_record，保证幂等断点续跑
=====================================================================
"""

import datetime
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_cleaner import DataCleaner
from data_fetcher import data_fetcher
from utils.common_tools import (
    calc_incremental_date_range,
    get_trade_dates,
    read_last_update_record,
    write_update_record,
)
from utils.db_utils import db
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message

# 初始化核心组件
cleaner = DataCleaner()

# ======================== 配置参数 ========================
START_HOUR   = 15      # 每日开始尝试时间（小时）
START_MINUTE = 30      # 每日开始尝试时间（分钟）
RETRY_INTERVAL  = 1800 # 重试间隔（秒），30 分钟
MAX_RETRY_TIMES = 10   # 单日最大重试次数

# kline_day 成功判定阈值：单个交易日入库行数低于此值视为数据不完整
# A 股全市场约 5000 只，保守取 4000 以兼容停牌/新股等情况
MIN_KLINE_ROWS_PER_DAY = 4000
# ==========================================================


# ======================== 各表更新函数 ========================

def update_stock_basic() -> tuple:
    """全量更新 stock_basic 表"""
    logger.info("===== 全量更新 stock_basic 表 =====")
    try:
        affected = cleaner.clean_and_insert_stockbase(table_name="stock_basic")
        affected = affected if affected is not None else 0
        logger.info(f"stock_basic 更新完成，影响行数：{affected}")
        return True, affected
    except Exception as e:
        logger.error(f"stock_basic 更新失败：{e}", exc_info=True)
        return False, 0


def update_stock_st_incremental(start_date: str, end_date: str) -> tuple:
    """增量更新 ST 股票数据"""
    if not start_date or not end_date:
        logger.warning("ST 增量更新：日期参数为空，跳过")
        return False, 0
    logger.info(f"===== 增量更新 ST 数据（{start_date} ~ {end_date}） =====")
    try:
        affected = cleaner.insert_stock_st(start_date=start_date, end_date=end_date)
        affected = affected if affected is not None else 0
        logger.info(f"ST 数据更新完成，入库行数：{affected}")
        return True, affected
    except Exception as e:
        logger.error(f"ST 数据更新失败（{start_date}~{end_date}）：{e}", exc_info=True)
        return False, 0


def update_kline_day_incremental(date_list: list) -> tuple:
    """
    增量更新日线数据（批量拉取 + 单股兜底重试）
    :return: (is_success, total_affected_rows)
             is_success 由调用方根据 MIN_KLINE_ROWS_PER_DAY 判定，此处只负责执行
    """
    if not date_list:
        logger.warning("kline_day 更新：日期列表为空，跳过")
        return False, 0

    logger.info(f"===== 增量更新日线数据（共 {len(date_list)} 个交易日） =====")
    stock_codes = db.get_all_a_stock_codes()
    if not stock_codes:
        logger.error("无有效股票代码，终止日线更新")
        return False, 0

    BATCH_SIZE    = 600
    total_affected = 0

    for trade_date in date_list:
        daily_affected = 0
        batches = [stock_codes[i:i + BATCH_SIZE] for i in range(0, len(stock_codes), BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            try:
                raw_df = data_fetcher.fetch_kline_day(
                    ts_code=",".join(batch),
                    start_date=trade_date,
                    end_date=trade_date,
                )
                if raw_df.empty:
                    logger.debug(f"{trade_date} 第 {batch_idx+1} 批无数据")
                    continue

                clean_df = cleaner._clean_kline_day_data(raw_df)
                if clean_df.empty:
                    continue

                affected = db.batch_insert_df(clean_df, "kline_day", ignore_duplicate=True)
                daily_affected += affected
                logger.debug(f"{trade_date} 第 {batch_idx+1} 批入库 {affected} 行")

            except Exception as e:
                logger.error(f"{trade_date} 第 {batch_idx+1} 批失败：{e}，启动单股兜底重试")
                for ts_code in batch:
                    try:
                        df_s = data_fetcher.fetch_kline_day(ts_code, trade_date, trade_date)
                        if not df_s.empty:
                            clean_s = cleaner._clean_kline_day_data(df_s)
                            if not clean_s.empty:
                                daily_affected += db.batch_insert_df(
                                    clean_s, "kline_day", ignore_duplicate=True
                                )
                    except Exception as e2:
                        logger.error(f"{ts_code} {trade_date} 单股重试失败：{e2}")

        logger.info(f"{trade_date} 日线更新完成，入库 {daily_affected} 行")
        total_affected += daily_affected

    logger.info(f"日线增量更新完成，累计入库 {total_affected} 行")
    return True, total_affected


def update_index_daily(last_date: str) -> tuple:
    """增量更新核心指数日线"""
    logger.info("===== 更新 index_daily 表 =====")
    # 统一转为 YYYYMMDD，兼容 last_date 含短横线或不含的情况
    start_date_fmt = last_date.replace("-", "")
    end_date_fmt   = datetime.datetime.now().strftime("%Y%m%d")
    index_list     = ["000001.SH", "399001.SZ", "399006.SZ", "399107.SZ"]
    try:
        total_affected = 0
        for code in index_list:
            affected = cleaner.clean_and_insert_index_daily(
                ts_code=code,
                start_date=start_date_fmt,
                end_date=end_date_fmt,
            )
            affected = affected if affected is not None else 0
            total_affected += affected
            logger.info(f"  {code} 入库 {affected} 行")
        logger.info(f"index_daily 更新完成，累计入库 {total_affected} 行")
        return True, total_affected
    except Exception as e:
        logger.error(f"index_daily 更新失败：{e}", exc_info=True)
        return False, 0


# ======================== 推送格式化 ========================

def _build_push_msg(
    current_date: str,
    last_date: str,
    is_success: bool,
    affected: dict,
    total: int,
    retry_count: int,
    inc_dates: list,
) -> str:
    """
    构造格式化的微信推送消息
    对齐输出，方便在手机上阅读
    """
    status   = "✅ 成功" if is_success else "❌ 异常/数据不完整"
    rec_line = f"已更新为 {current_date}" if is_success else f"保留原记录 {last_date}"

    lines = [
        f"📅 更新日期：{current_date}",
        f"🔁 本次重试：第 {retry_count} 次",
        f"📆 补录天数：{len(inc_dates)} 天 {inc_dates}",
        f"",
        f"📊 各表入库行数：",
        f"  stock_basic  : {affected.get('stock_basic', 0):>6,} 行",
        f"  stock_st     : {affected.get('stock_st',    0):>6,} 行",
        f"  kline_day    : {affected.get('kline',       0):>6,} 行",
        f"  index_daily  : {affected.get('index',       0):>6,} 行",
        f"  ─────────────────────",
        f"  合计         : {total:>6,} 行",
        f"",
        f"📌 状态：{status}",
        f"🕐 记录：{rec_line}",
    ]
    return "\n".join(lines)


# ======================== 主更新流程 ========================

def startUpdating(retry_count: int = 1) -> tuple:
    """
    执行一次完整的数据更新流程

    成功判定逻辑：
      - kline_day 执行无异常（success_flags['kline'] is True）
      - kline_day 入库行数 >= MIN_KLINE_ROWS_PER_DAY * 待补天数
        （动态阈值，补 1 天要求 ≥4000 行，补 3 天要求 ≥12000 行）
      - stock_basic 失败不影响整体成功判定（辅助表）

    :param retry_count: 当前重试次数（仅用于推送消息显示）
    :return: (is_success: bool, push_msg: str)
    """
    logger.info("===== 启动增量更新流程 =====")

    last_record  = read_last_update_record()
    last_date    = last_record["last_update_date"]
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    logger.info(f"上次记录时间：{last_date}  本次执行时间：{current_date}")

    inc_dates = calc_incremental_date_range(last_date, current_date)
    logger.info(f"增量日期列表（共 {len(inc_dates)} 天）：{inc_dates}")

    # ---------- 执行各表更新 ----------
    success_flags = {}
    affected      = {}

    success_flags["stock_basic"], affected["stock_basic"] = update_stock_basic()
    success_flags["stock_st"],    affected["stock_st"]    = update_stock_st_incremental(last_date, current_date)
    success_flags["kline"],       affected["kline"]       = update_kline_day_incremental(inc_dates)
    success_flags["index"],       affected["index"]       = update_index_daily(last_date)

    total = sum(affected.values())
    logger.info(f"各表入库行数：{affected}  合计：{total}")

    # ---------- 成功判定 ----------
    # 核心门槛：kline_day 必须执行成功 + 行数达到动态阈值
    expected_min  = MIN_KLINE_ROWS_PER_DAY * max(len(inc_dates), 1)
    kline_ok      = success_flags["kline"] and affected["kline"] >= expected_min
    is_success    = kline_ok

    if not kline_ok:
        logger.warning(
            f"kline_day 入库行数 {affected['kline']} < 预期最低 {expected_min}"
            f"（{MIN_KLINE_ROWS_PER_DAY} 行/天 × {len(inc_dates)} 天），判定更新不完整"
        )

    # ---------- 写入记录（仅成功时）----------
    updated_tables = [k for k, v in success_flags.items() if v]
    if is_success:
        write_update_record(current_date, updated_tables)
        logger.info(f"更新记录写入成功，时间更新为：{current_date}")
    else:
        logger.warning(f"更新不完整，记录保留原值：{last_date}")

    push_msg = _build_push_msg(
        current_date, last_date, is_success, affected, total, retry_count, inc_dates
    )
    logger.info(
        f"===== 本次执行完成 | kline {affected['kline']} 行 | 预期 ≥{expected_min} 行 | 成功：{is_success} ====="
    )
    return is_success, push_msg


# ======================== 服务器调度逻辑 ========================

def is_trade_day(check_date: str) -> bool:
    """判断指定日期是否为交易日（查询失败时保守返回 False）"""
    try:
        dates = get_trade_dates(check_date, check_date)
        return check_date in dates
    except Exception as e:
        logger.warning(f"交易日历查询失败：{e}，保守策略视为非交易日")
        return False


def wait_until_target_time(target_hour: int, target_minute: int):
    """阻塞等待直到目标时间，已过则立即返回"""
    now         = datetime.datetime.now()
    target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if now >= target_time:
        logger.info(f"当前 {now.strftime('%H:%M:%S')} 已过目标时间 {target_time.strftime('%H:%M:%S')}，直接执行")
        return
    wait_sec = (target_time - now).total_seconds()
    logger.info(f"等待至 {target_time.strftime('%H:%M:%S')}，还需 {wait_sec/60:.1f} 分钟")
    time.sleep(wait_sec)


def _sleep_to_tomorrow():
    """休眠至次日凌晨 2 点"""
    now          = datetime.datetime.now()
    tomorrow_2am = (now + datetime.timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
    secs         = (tomorrow_2am - now).total_seconds()
    logger.info(f"休眠 {secs/3600:.1f} 小时至次日凌晨 2 点")
    time.sleep(secs)


def run_server():
    """服务端常驻主循环"""
    logger.info("量化数据更新服务已启动")
    last_success_date = None   # 记录最后一次成功更新的日期

    while True:
        now       = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # ---------- 今日已完成 → 休眠到明天 ----------
        if last_success_date == today_str:
            logger.info(f"[{today_str}] 今日任务已完成，休眠至次日")
            _sleep_to_tomorrow()
            continue

        # ---------- 非交易日 → 每 4 小时检查一次日期 ----------
        if not is_trade_day(today_str):
            logger.info(f"[{today_str}] 非交易日，跳过")
            time.sleep(3600 * 4)
            continue

        # ---------- 交易日：等到 15:30 ----------
        wait_until_target_time(START_HOUR, START_MINUTE)

        # ---------- 重试循环 ----------
        update_success = False
        retry_count    = 0       # 局部变量，每日任务独立计数

        while not update_success and retry_count < MAX_RETRY_TIMES:
            retry_count += 1
            logger.info(
                f"===== [{today_str}] 第 {retry_count}/{MAX_RETRY_TIMES} 次尝试 "
                f"| {datetime.datetime.now().strftime('%H:%M:%S')} ====="
            )

            try:
                update_success, push_msg = startUpdating(retry_count=retry_count)

                if update_success:
                    # 成功：推送一次，标记完成
                    logger.info("数据更新成功！")
                    try:
                        send_wechat_message(f"【量化数据更新成功】{today_str}", push_msg)
                    except Exception as e:
                        logger.error(f"成功推送失败：{e}", exc_info=True)
                    last_success_date = today_str

                else:
                    logger.warning(
                        f"第 {retry_count} 次更新不完整，"
                        f"{RETRY_INTERVAL // 60} 分钟后重试..."
                    )
                    # 失败推送：仅第 1 次失败时推送，避免消息轰炸
                    if retry_count == 1:
                        try:
                            send_wechat_message(
                                f"【量化数据更新异常】{today_str}",
                                push_msg + f"\n\n⚠️ 将每 {RETRY_INTERVAL//60} 分钟自动重试，最多 {MAX_RETRY_TIMES} 次",
                            )
                        except Exception as e:
                            logger.error(f"首次失败推送失败：{e}", exc_info=True)
                    time.sleep(RETRY_INTERVAL)

            except Exception as e:
                logger.error(f"第 {retry_count} 次重试发生严重异常：{e}", exc_info=True)
                time.sleep(RETRY_INTERVAL)

        # ---------- 达到最大重试次数仍未成功 ----------
        if not update_success:
            logger.error(f"[{today_str}] 已重试 {MAX_RETRY_TIMES} 次，停止，请手动处理")
            try:
                send_wechat_message(
                    f"【量化数据更新失败】{today_str}",
                    f"❌ 今日已重试 {MAX_RETRY_TIMES} 次仍未达到数据完整性要求\n请手动检查数据库或接口状态",
                )
            except Exception as e:
                logger.error(f"最终失败推送失败：{e}", exc_info=True)
            # 标记今日已处理，防止无限循环卡死
            last_success_date = today_str


if __name__ == "__main__":
    run_server()