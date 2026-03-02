"""
板块热度选股尾盘买入策略
核心逻辑：每日筛选前3活跃板块→板块内候选股→XGBoost预测胜率→按胜率选股→D+1卖出
完全匹配BaseStrategy基类接口，正确生成买卖信号，对齐回测引擎全流程
"""
from collections import defaultdict
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
from utils.common_tools import (
    get_stocks_in_sector,
    filter_st_stocks,
    get_trade_dates,
    get_daily_kline_data,
)
# 统一导入风格：核心配置 → 工具类 → 基类 → 业务组件
from config.config import (
    FILTER_BSE_STOCK,
    FILTER_STAR_BOARD,
    FILTER_688_BOARD
)
from utils.log_utils import logger
from strategies.base_strategy import BaseStrategy
from features.market_stats import SectorHeatFeature
from data.data_cleaner import data_cleaner
from utils.db_utils import db


class SectorHeatStrategy(BaseStrategy):
    """板块热度选股策略，完全继承并实现基类所有抽象方法"""

    def __init__(self):
        super().__init__()
        # 策略基础配置（基类约定必填）
        self.strategy_name = "板块热度XGBoost选股尾盘买入策略"
        self.strategy_params = {
            "stock_per_sector": 5,  # 每个板块候选股数量
            "buy_top_k": 6,  # 按优先级选前K只买入
            "model_path": "models/sector_heat_xgb.pkl",
            "sell_type": "open"  # D+1卖出类型：open(次日开盘)/close(次日收盘)
        }
        # 核心组件初始化
        self.sector_feature = SectorHeatFeature()
        # 持仓管理：{股票代码: 买入日期}，严格控制D+1卖出
        self.hold_stock_dict: Dict[str, str] = {}
        # 初始化策略状态
        self.initialize()

    def initialize(self) -> None:
        """
        策略初始化（基类强制实现）
        功能：清空信号+重置内部状态+加载模型，回测启动/重置时自动调用
        """
        self.clear_signal()
        self.hold_stock_dict.clear()

    def _filter_ts_code_by_board(self, ts_code_list: List[str]) -> List[str]:
        """
        私有辅助方法：对股票代码列表直接做板块过滤（无需DataFrame）
        适配前置过滤需求，减少后续日线数据筛选量
        :param ts_code_list: 原始股票代码列表
        :return: 过滤后的代码列表
        """
        filtered_list = []
        for ts_code in ts_code_list:
            if not ts_code:
                continue
            # 过滤北交所（BSE）：83/87/88开头 或 .BJ后缀（双重保障）
            if FILTER_BSE_STOCK and (ts_code.endswith(".BJ") or ts_code.startswith(("83", "87", "88"))):
                continue
            # 过滤科创板（688开头）
            if FILTER_688_BOARD and ts_code.startswith("688"):
                continue
            # 【修复】创业板过滤逻辑去重，仅保留精准判断
            if FILTER_STAR_BOARD and ts_code.startswith(("300", "301", "302")) and ts_code.endswith(".SZ"):
                continue
            filtered_list.append(ts_code)
        return filtered_list

    def _filter_limit_up_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        私有辅助方法：过滤当日已涨停的股票（尾盘无法买入）
        【修复】过滤逻辑顺序+明细日志，彻底解决误过滤问题
        :param df: 待过滤股票DataFrame，必须包含ts_code/pre_close/close字段
        :return: 过滤后的DataFrame
        """
        # 入参防御
        if df.empty:
            logger.warning("涨停过滤入参异常：DataFrame为空")
            return pd.DataFrame()
        required_cols = ["ts_code", "pre_close", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"涨停过滤入参异常：缺少关键字段 {missing_cols}")
            return df

        filtered_df = df.copy()
        # 复用基类涨停价计算方法（和全局标准统一）
        filtered_df["limit_up_price"] = filtered_df.apply(
            lambda x: self.calc_limit_up_price(x["ts_code"], x["pre_close"]),
            axis=1
        )

        # 【修复】先过滤无效涨停价，再过滤涨停股票，顺序不能乱
        # 1. 先剔除涨停价计算无效的股票
        valid_df = filtered_df[filtered_df["limit_up_price"] > 0].copy()
        invalid_count = len(filtered_df) - len(valid_df)
        if invalid_count > 0:
            logger.warning(f"涨停过滤：剔除{invalid_count}只涨停价计算无效的股票")

        # 2. 再过滤当日已涨停的股票（收盘价≥涨停价-0.01，即涨停/准涨停）
        # 保留：收盘价 < 涨停价-0.01 的未涨停股票
        final_df = valid_df[valid_df["close"] < (valid_df["limit_up_price"] - 0.01)].copy()

        # 【修复】打印被过滤股票的明细，彻底排查误过滤问题
        original_codes = set(df["ts_code"].tolist())
        remaining_codes = set(final_df["ts_code"].tolist())
        filtered_codes = original_codes - remaining_codes

        if filtered_codes:
            # 提取被过滤股票的价格明细
            filtered_detail = filtered_df[filtered_df["ts_code"].isin(filtered_codes)][
                ["ts_code", "pre_close", "close", "limit_up_price"]
            ]
            logger.info(f"当日涨停被过滤的股票明细：\n{filtered_detail}")
            logger.info(f"本次涨停过滤共剔除 {len(filtered_codes)} 只股票")
        else:
            logger.info("本次涨停过滤未剔除任何股票（无当日涨停股票）")

        # 清理临时列
        final_df.drop(columns=["limit_up_price"], inplace=True)
        return final_df

    def _check_stock_has_limit_up(self, ts_code_list: List[str], end_date: str, day_count: int = 10) -> Dict[str, bool]:
        """
        【最终优化版】批量判断股票近N个交易日是否有涨停
        核心修改：
        1. 复用get_trade_dates获取交易日
        2. 复用get_daily_kline_data获取日线数据（彻底统一数据获取逻辑）
        """
        # 1. 入参校验
        if not ts_code_list or day_count <= 0 or not end_date:
            logger.warning("_check_stock_has_limit_up 入参无效，返回全True（保守保留所有股票）")
            return {ts_code: True for ts_code in ts_code_list}

        # 2. 统一日期格式：转为YYYY-MM-DD
        try:
            if len(end_date) == 8 and end_date.isdigit():
                end_date_dt = datetime.strptime(end_date, "%Y%m%d")
                end_date_format = end_date_dt.strftime("%Y-%m-%d")
            else:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                end_date_format = end_date
        except ValueError as e:
            logger.error(f"日期格式解析失败：{end_date}，错误：{e}，返回全True")
            return {ts_code: True for ts_code in ts_code_list}

        # 3. 复用get_trade_dates获取回溯期交易日
        try:
            pre_end_date = (end_date_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date_dt = end_date_dt - timedelta(days=60)
            start_date_format = start_date_dt.strftime("%Y-%m-%d")

            all_trade_dates = get_trade_dates(start_date=start_date_format, end_date=pre_end_date)
            target_dates = all_trade_dates[-day_count:]
            if len(target_dates) < day_count:
                logger.warning(f"可回溯交易日不足{day_count}个（仅{len(target_dates)}个），返回全True")
                return {ts_code: True for ts_code in ts_code_list}

            logger.debug(f"近{day_count}个回溯交易日：{target_dates}")

        except RuntimeError as e:
            logger.error(f"调用get_trade_dates获取交易日失败：{e}，返回全True")
            return {ts_code: True for ts_code in ts_code_list}
        except Exception as e:
            logger.error(f"获取交易日历失败：{e}，返回全True")
            return {ts_code: True for ts_code in ts_code_list}

        # 4. 【核心修改】复用get_daily_kline_data获取日线数据
        try:
            # 初始化空DataFrame，用于合并多日数据
            all_daily_df = pd.DataFrame()
            # 遍历每个回溯交易日，逐个获取日线数据
            for trade_date in target_dates:
                # 调用通用方法获取单日全市场日线数据
                daily_df = get_daily_kline_data(trade_date=trade_date)
                if daily_df.empty:
                    logger.warning(f"{trade_date} 日线数据为空，跳过该交易日")
                    continue
                # 筛选出目标股票的日线数据（仅保留ts_code_list中的股票）
                filtered_daily_df = daily_df[daily_df["ts_code"].isin(ts_code_list)].copy()
                all_daily_df = pd.concat([all_daily_df, filtered_daily_df], ignore_index=True)

            # 校验合并后的数据是否为空
            if all_daily_df.empty:
                logger.warning("回溯期内无有效日线数据，返回全True")
                return {ts_code: True for ts_code in ts_code_list}

            # 【关键】将DataFrame转为原有逻辑的字典列表格式（保持后续逻辑兼容）
            result = all_daily_df.to_dict("records")
            logger.debug(f"回溯期内获取到{len(result)}条有效日线数据")
        except Exception as e:
            logger.error(f"调用get_daily_kline_data获取日线数据失败：{e}，返回全True")
            return {ts_code: True for ts_code in ts_code_list}
        # 5. 分组判断每只股票是否有涨停（逻辑完全不变）
        stock_daily_map = defaultdict(list)
        for row in result:
            stock_daily_map[row["ts_code"]].append(row)

        result_dict = {ts_code: False for ts_code in ts_code_list}
        for ts_code, daily_list in stock_daily_map.items():
            for daily in daily_list:
                pre_close = daily["pre_close"]
                close = daily["close"]
                if pre_close <= 0 or close <= 0:
                    continue
                # 调用基类的涨停价计算方法
                limit_up_price = self.calc_limit_up_price(ts_code, pre_close)
                logger.debug(
                    f"【涨停判断明细】{ts_code} 日期：{daily['trade_date']} | 前收：{pre_close} | 收盘价：{close} | 涨停价：{limit_up_price}")

                if limit_up_price <= 0:
                    continue
                # 严格判断涨停
                if abs(close - limit_up_price) <= 0.001 or close >= limit_up_price:
                    result_dict[ts_code] = True
                    logger.debug(
                        f"【涨停确认】{ts_code} 在{daily['trade_date']}涨停：收盘价{close} ≥ 涨停价{limit_up_price}")
                    break

        logger.info(
            f"近{day_count}日涨停判断完成：有涨停基因个股{sum(result_dict.values())}只，总候选{len(ts_code_list)}只")
        return result_dict

    def generate_signal(
            self,
            trade_date: str,
            daily_df: pd.DataFrame,
            positions: Dict[str, any]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        核心信号生成方法（完全匹配基类接口与回测引擎调用流程）
        引擎调用时序：
        1. 当日开盘前：执行上一日生成的open类型卖出信号
        2. 第一次调用：获取当日买入列表，执行尾盘买入
        3. 当日收盘前：第二次调用，获取次日卖出信号，执行当日close类型卖出
        :param trade_date: 交易日（格式：YYYYMMDD/YYYY-MM-DD）
        :param daily_df: 当日全市场日线数据，必须包含ts_code/pre_close/open/high/low/close/amount等字段
        :param positions: 当前账户持仓字典 {股票代码: 持仓信息}
        :return: Tuple[买入股票列表, 卖出信号字典{ts_code: sell_type}]
        """
        # ===================== 步骤1：同步持仓状态，保证数据一致性 =====================
        # 移除已卖出的股票
        for ts_code in list(self.hold_stock_dict.keys()):
            if ts_code not in positions:
                del self.hold_stock_dict[ts_code]
        # 新增当日买入的股票
        for ts_code in positions.keys():
            if ts_code not in self.hold_stock_dict:
                self.hold_stock_dict[ts_code] = trade_date
                logger.info(f"[{self.strategy_name}] {trade_date} 记录持仓：{ts_code}，买入日期：{trade_date}")

        # ===================== 步骤2：生成卖出信号（严格执行D+1卖出规则） =====================
        sell_signal_map = {}
        sell_type = self.strategy_params["sell_type"]
        for ts_code, buy_date in self.hold_stock_dict.items():
            if buy_date < trade_date:
                sell_signal_map[ts_code] = sell_type
                logger.debug(
                    f"[{self.strategy_name}] {trade_date} 生成卖出信号：{ts_code}，卖出类型：{sell_type}，买入日期：{buy_date}")

        # ===================== 步骤3：生成买入信号（按板块热度选股逻辑） =====================
        # 3.1 获取当日前3热点板块+板块轮动适配分
        sectors_status = self.sector_feature.select_top3_hot_sectors(trade_date)
        top3_sectors = sectors_status['top3_sectors']
        buy_stocks = []

        # 轮动分大于40，板块轮动过快，今日空仓
        if sectors_status['adapt_score'] >= 40:
            logger.warning(f"[{self.strategy_name}] {trade_date} 板块轮动分{sectors_status['adapt_score']}≥40，轮动过快，今日空仓")
            return buy_stocks, sell_signal_map

        logger.info(
            f"[{self.strategy_name}] {trade_date} 板块轮动分{sectors_status['adapt_score']}<40，开始执行选股，前3热点板块：{top3_sectors}")

        # 【修复】ST数据入库移到循环外，只执行1次，避免重复入库
        try:
            # 统一转换为入库需要的YYYYMMDD格式
            st_trade_date = trade_date.replace('-', '')
            data_cleaner.insert_stock_st(trade_date=st_trade_date)
            logger.info(f"{trade_date} ST数据入库完成")
        except Exception as e:
            logger.error(f"{trade_date} ST数据入库失败：{e}", exc_info=True)

        # 初始化板块候选池字典
        sector_candidate_map = {}
        # 3.2 遍历每个板块，逐层筛选选股池
        for sector in top3_sectors:
            logger.info(f"[{self.strategy_name}] {trade_date} 开始处理板块：{sector}")
            try:
                # 1. 获取板块中所有个股代码
                sector_stock_raw = get_stocks_in_sector(sector)
                if not sector_stock_raw:
                    logger.warning(f"板块[{sector}]未查询到对应股票，跳过")
                    sector_candidate_map[sector] = []
                    continue

                # 转成纯ts_code列表
                sector_ts_codes = [item["ts_code"] for item in sector_stock_raw]
                logger.info(f"板块[{sector}]原始股票数量：{len(sector_ts_codes)}")

                # ========== 前置过滤（代码列表级别） ==========
                # 1. 板块过滤（创业板、科创板、北交所）
                filtered_ts_codes = self._filter_ts_code_by_board(sector_ts_codes)
                if not filtered_ts_codes:
                    logger.warning(f"板块[{sector}]板块过滤后无剩余股票，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]板块过滤后剩余：{len(filtered_ts_codes)}只")

                # 2. ST股票过滤
                filtered_ts_codes = filter_st_stocks(filtered_ts_codes, trade_date)
                if not filtered_ts_codes:
                    logger.warning(f"板块[{sector}]ST过滤后无剩余股票，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]ST过滤后剩余：{len(filtered_ts_codes)}只")

                # 3. 筛选该板块的日线数据
                sector_daily_df = daily_df[daily_df["ts_code"].isin(filtered_ts_codes)].copy()
                if sector_daily_df.empty:
                    logger.warning(f"板块[{sector}]当日无有效日线数据，跳过")
                    sector_candidate_map[sector] = []
                    continue

                # 4. 过滤当日已涨停股票（尾盘无法买入）
                filtered_df = self._filter_limit_up_stock(sector_daily_df)
                if filtered_df.empty:
                    logger.warning(f"板块[{sector}]涨停过滤后无剩余股票，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]涨停过滤后剩余：{len(filtered_df)}只")

                # 5. 近10个交易日有涨停筛选
                candidate_ts_codes = filtered_df["ts_code"].unique().tolist()
                has_limit_up_map = self._check_stock_has_limit_up(
                    ts_code_list=candidate_ts_codes,
                    end_date=trade_date,
                    day_count=10
                )
                # 保留近10日有涨停的股票
                keep_ts_codes = [ts_code for ts_code, has_limit_up in has_limit_up_map.items() if has_limit_up]
                filtered_df = filtered_df[filtered_df["ts_code"].isin(keep_ts_codes)]
                print(filtered_df)
                breakpoint()
                if filtered_df.empty:
                    logger.warning(f"板块[{sector}]近10个交易日无符合要求的涨停个股，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]近10日有涨停筛选后剩余：{len(filtered_df)}只")

                # 6. 保存最终候选池
                final_candidate_codes = filtered_df["ts_code"].tolist()
                # 每个板块最多保留stock_per_sector只
                sector_candidate_map[sector] = final_candidate_codes[:self.strategy_params["stock_per_sector"]]
                logger.info(f"板块[{sector}]最终候选池数量：{len(sector_candidate_map[sector])}只")

            except Exception as e:
                logger.error(f"[{self.strategy_name}] {trade_date} 处理板块[{sector}]失败：{str(e)}", exc_info=True)
                sector_candidate_map[sector] = []
                continue

        # ===================== 生成最终买入列表 =====================
        all_candidates = []
        for codes in sector_candidate_map.values():
            all_candidates.extend(codes)
        # 去重+取前buy_top_k只
        all_candidates = list(dict.fromkeys(all_candidates))
        buy_stocks = all_candidates[:self.strategy_params["buy_top_k"]]
        logger.info(f"[{self.strategy_name}] {trade_date} 最终买入列表：{buy_stocks}")

        return buy_stocks, sell_signal_map