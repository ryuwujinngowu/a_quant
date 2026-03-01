"""
板块热度选股尾盘买入策略
核心逻辑：每日筛选前3活跃板块→板块内候选股→XGBoost预测胜率→按胜率选股→D+1卖出
完全匹配BaseStrategy基类接口，正确生成买卖信号，对齐回测引擎全流程
"""
from datetime import datetime
from utils.common_tools import  get_stocks_in_sector, get_sector_stock_daily_data, is_st_stock
from typing import List, Dict, Tuple
import os
import pandas as pd
from utils.common_tools import (
    get_stocks_in_sector,
    get_sector_stock_daily_data,
    is_st_stock,
    check_stock_has_limit_up  # 新增导入
)
# 统一导入风格：核心配置 → 工具类 → 基类 → 业务组件
from config.config import (
    MAIN_BOARD_LIMIT_UP_RATE,
    STAR_BOARD_LIMIT_UP_RATE,
    BJ_BOARD_LIMIT_UP_RATE,
    MAX_POSITION_COUNT,
    FILTER_BSE_STOCK,
    FILTER_STAR_BOARD,
    FILTER_688_BOARD,
    FILTER_MAIN_BOARD
)
from utils.log_utils import logger
from strategies.base_strategy import BaseStrategy
from features.market_stats import SectorHeatFeature


# from learnEngine.dataset import DatasetBuilder
# from learnEngine.model import SectorHeatXGBModel


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
        # 模型预留接口，后期启用
        # self.xgb_model = SectorHeatXGBModel()

        # 持仓管理：{股票代码: 买入日期}，严格控制D+1卖出
        self.hold_stock_dict: Dict[str, str] = {}
        # 初始化策略状态
        self.initialize()

    def initialize(self) -> None:
        """
        策略初始化（基类强制实现）
        功能：清空信号+重置内部状态+加载模型，回测启动/重置时自动调用
        """
        # 调用基类通用方法清空买卖信号
        self.clear_signal()
        # 重置策略内部持仓状态
        self.hold_stock_dict.clear()

        # 模型加载预留接口，后期启用
        # load_success = self.xgb_model.load_model(self.strategy_params["model_path"])
        # if load_success:
        #     logger.info(f"[{self.strategy_name}] XGBoost模型加载成功（路径：{self.strategy_params['model_path']}）")
        # else:
        #     logger.warning(f"[{self.strategy_name}] XGBoost模型加载失败，策略将仅返回空信号")

    def _filter_ts_code_by_board(self, ts_code_list: List[str]) -> List[str]:
        """
        私有辅助方法：对股票代码列表直接做板块过滤（无需DataFrame）
        适配前置过滤需求，减少后续日线数据筛选量
        :param ts_code_list: 原始股票代码列表
        :return: 过滤后的代码列表
        """
        filtered_list = []
        for ts_code in ts_code_list:
            # 跳过空值
            if not ts_code:
                continue
            # 过滤北交所（BSE）：83/87/88开头 或 .BJ后缀（双重保障，避免漏网）
            if FILTER_BSE_STOCK and (ts_code.startswith(("83", "87", "88")) or ts_code.endswith(".BJ")):
                logger.debug(f"过滤北交所股票：{ts_code}")  # 新增：调试日志，确认过滤生效
                continue
            # 过滤科创板（688开头）
            if FILTER_688_BOARD and ts_code.startswith("688"):
                logger.debug(f"过滤科创板股票：{ts_code}")
                continue
            # 过滤创业板（300/301/302开头 + .SZ后缀）
            if FILTER_STAR_BOARD and (ts_code.startswith(("300", "301", "302")) or (ts_code.startswith("3") and ts_code.endswith(".SZ"))):
                logger.debug(f"过滤创业板股票：{ts_code}")
                continue
            # 过滤主板（60/00开头）
            # if FILTER_MAIN_BOARD and ts_code.startswith(("60", "00")):
            #     continue
            filtered_list.append(ts_code)
        return filtered_list

    def _filter_limit_up_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        私有辅助方法：过滤当日已涨停的股票（尾盘无法买入）
        :param df: 待过滤股票DataFrame，必须包含ts_code/pre_close/close字段
        :return: 过滤后的DataFrame
        """
        # 【新增】入参防御：空DataFrame直接返回，避免内部报错
        if df.empty or not all(col in df.columns for col in ["ts_code", "pre_close", "close"]):
            logger.warning("涨停过滤入参异常：DataFrame为空或缺少关键字段")
            return pd.DataFrame()

        filtered_df = df.copy()
        # 复用基类涨停价计算方法，保证全项目逻辑统一
        filtered_df["limit_up_price"] = filtered_df.apply(
            lambda x: self.calc_limit_up_price(x["ts_code"], x["pre_close"]),
            axis=1
        )
        # 保留未涨停的股票（增加limit_up_price>0校验）
        filtered_df = filtered_df[(filtered_df["close"] < filtered_df["limit_up_price"]) & (filtered_df["limit_up_price"] > 0)]
        # 清理临时列
        filtered_df.drop(columns=["limit_up_price"], inplace=True)
        return filtered_df

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
        :param trade_date: 交易日（格式：YYYYMMDD）
        :param daily_df: 当日全市场日线数据，必须包含ts_code/pre_close/open/high/low/close/amount等字段
        :param positions: 当前账户持仓字典 {股票代码: 持仓信息}
        :return: Tuple[买入股票列表, 卖出信号字典{ts_code: sell_type}]
        """
        # ===================== 步骤1：同步持仓状态，保证数据一致性 =====================
        # 移除已卖出的股票（持仓中不存在的）
        for ts_code in list(self.hold_stock_dict.keys()):
            if ts_code not in positions:
                del self.hold_stock_dict[ts_code]
        # 新增当日买入的股票（持仓中存在但未记录的，买入日期记为当日）
        for ts_code in positions.keys():
            if ts_code not in self.hold_stock_dict:
                self.hold_stock_dict[ts_code] = trade_date
                logger.info(f"[{self.strategy_name}] {trade_date} 记录持仓：{ts_code}，买入日期：{trade_date}")

        # ===================== 步骤2：生成卖出信号（严格执行D+1卖出规则） =====================
        sell_signal_map = {}
        sell_type = self.strategy_params["sell_type"]
        # 遍历持仓，满足D+1条件（买入日期 < 当前交易日）的生成卖出信号
        for ts_code, buy_date in self.hold_stock_dict.items():
            if buy_date < trade_date:
                sell_signal_map[ts_code] = sell_type
                logger.debug(
                    f"[{self.strategy_name}] {trade_date} 生成卖出信号：{ts_code}，卖出类型：{sell_type}，买入日期：{buy_date}")

        # ===================== 步骤3：生成买入信号（按板块热度选股逻辑） =====================
        # 3.1 获取当日前3热点板块+板块轮动适配分
        sectors_status = self.sector_feature.select_top3_hot_sectors(trade_date)
        top3_sectors = sectors_status['top3_sectors']
        # 初始化买入列表，避免未定义报错
        buy_stocks = []
        # 轮动分大于40，板块轮动过快，今日空仓
        if sectors_status['adapt_score'] >= 40:
            logger.warning(f"[{self.strategy_name}] {trade_date} 板块轮动分{sectors_status['adapt_score']}≥40，轮动过快，今日空仓")
            return buy_stocks, sell_signal_map

        # 轮动分正常，执行选股逻辑
        logger.info(
            f"[{self.strategy_name}] {trade_date} 板块轮动分{sectors_status['adapt_score']}<40，开始执行选股，前3热点板块：{top3_sectors}")

        # 初始化板块候选池字典：key=板块名称，value=该板块筛选后的候选股列表
        sector_candidate_map = {}
        # 3.2 遍历每个板块，逐层筛选选股池
        for sector in top3_sectors:
            logger.info(f"[{self.strategy_name}] {trade_date} 开始处理板块：{sector}")
            # 【恢复】try-except，避免单板块异常导致整个策略崩溃
            try:
                # 1. 获取板块中所有个股代码（适配返回格式：[{'ts_code': '920006.BJ'}, ...]）
                sector_stock_raw = get_stocks_in_sector(sector)
                if not sector_stock_raw:
                    logger.warning(f"板块[{sector}]未查询到对应股票，跳过")
                    sector_candidate_map[sector] = []
                    continue

                # 转成纯ts_code列表
                sector_ts_codes = [item["ts_code"] for item in sector_stock_raw]
                logger.info(f"板块[{sector}]原始股票数量：{len(sector_ts_codes)}")

                # ========== 前置过滤（代码列表级别） ==========
                # 1. 先过滤板块（创业板、科创板、北交所）
                filtered_ts_codes = self._filter_ts_code_by_board(sector_ts_codes)
                if not filtered_ts_codes:
                    logger.warning(f"板块[{sector}]板块过滤后无剩余股票，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]板块过滤后剩余：{len(filtered_ts_codes)}只")

                # 2. 过滤ST/*ST股票
                filtered_ts_codes = [
                    ts_code for ts_code in filtered_ts_codes
                    if not is_st_stock(ts_code, trade_date)
                ]
                if not filtered_ts_codes:
                    logger.warning(f"板块[{sector}]ST过滤后无剩余股票，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]ST过滤后剩余：{len(filtered_ts_codes)}只")

                # 3. 筛选该板块的日线数据（用过滤后的代码列表）
                sector_daily_df = daily_df[daily_df["ts_code"].isin(filtered_ts_codes)].copy()
                if sector_daily_df.empty:
                    logger.warning(f"板块[{sector}]当日无有效日线数据，跳过")
                    sector_candidate_map[sector] = []
                    continue

                # 【核心修正1】先初始化filtered_df，再调用涨停过滤方法
                filtered_df = sector_daily_df.copy()

                # 4. 过滤当日已涨停股票（尾盘无法买入）
                filtered_df = self._filter_limit_up_stock(filtered_df)
                if filtered_df.empty:
                    logger.warning(f"板块[{sector}]涨停过滤后无剩余股票，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]涨停过滤后剩余：{len(filtered_df)}只")

                # 【核心新增】近10个交易日有涨停筛选
                # 1. 拿到当前板块的候选股代码
                candidate_ts_codes = filtered_df["ts_code"].unique().tolist()
                # 2. 批量判断近10个交易日是否有涨停
                has_limit_up_map = check_stock_has_limit_up(
                    ts_code_list=candidate_ts_codes,
                    end_date=trade_date,
                    day_count=10
                )
                # 3. 筛选出有涨停的股票
                keep_ts_codes = [ts_code for ts_code, has_limit_up in has_limit_up_map.items() if has_limit_up]
                filtered_df = filtered_df[filtered_df["ts_code"].isin(keep_ts_codes)]
                if filtered_df.empty:
                    logger.warning(f"板块[{sector}]近10个交易日无符合要求的涨停个股，跳过")
                    sector_candidate_map[sector] = []
                    continue
                logger.info(f"板块[{sector}]近10日有涨停筛选后剩余：{len(filtered_df)}只")

                # 5. 保存最终候选池（按活跃度排序，后续补充）
                final_candidate_codes = filtered_df["ts_code"].tolist()
                sector_candidate_map[sector] = final_candidate_codes
                logger.info(f"板块[{sector}]最终候选池数量：{len(final_candidate_codes)}只")

            except Exception as e:
                # 【恢复】异常捕获，避免单板块报错导致策略终止
                logger.error(f"[{self.strategy_name}] {trade_date} 处理板块[{sector}]失败：{str(e)}", exc_info=True)
                sector_candidate_map[sector] = []
                continue

        # ===================== 生成最终买入列表（可选补充） =====================
        # 汇总所有板块候选股，按数量筛选（匹配buy_top_k参数）
        all_candidates = []
        for codes in sector_candidate_map.values():
            all_candidates.extend(codes)
        # 去重+取前buy_top_k只
        all_candidates = list(dict.fromkeys(all_candidates))  # 保持顺序去重
        buy_stocks = all_candidates[:self.strategy_params["buy_top_k"]]
        logger.info(f"[{self.strategy_name}] {trade_date} 最终买入列表：{buy_stocks}")

        return buy_stocks, sell_signal_map