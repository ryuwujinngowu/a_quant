from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
from utils.common_tools import (
    get_stocks_in_sector,
    filter_st_stocks,
    sort_by_recent_gain,
    get_trade_dates,
    get_daily_kline_data,
    calc_limit_up_price
)
from config.config import (
    FILTER_BSE_STOCK,
    FILTER_STAR_BOARD,
    FILTER_688_BOARD
)
from utils.log_utils import logger
from features.base_feature import BaseFeature
from features.market_stats import SectorHeatFeature
from data.data_cleaner import data_cleaner


# ==================== 【新增】已处理日期管理类（原子性读写） ====================
# ==================== 【修复】已处理日期管理类（使用标准json模块） ====================
class ProcessedDatesManager:
    """专门管理已处理日期的读写，确保原子性"""
    def __init__(self, file_path: str, factor_version: str):
        self.file_path = file_path
        self.factor_version = factor_version
        self.processed_dates = self._load()

    def _load(self) -> List[str]:
        """加载已处理日期，版本不一致则清空"""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                # 【修复1】使用标准json.load，不是pd.io.json.load
                data = json.load(f)
                if data.get("factor_version") != self.factor_version:
                    logger.warning(f"因子版本变更，清空已处理记录")
                    return []
                return sorted(data.get("processed_dates", []))
        except Exception as e:
            logger.error(f"加载已处理日期失败：{e}")
            return []

    def is_processed(self, date: str) -> bool:
        """判断日期是否已处理"""
        return date in self.processed_dates

    def add(self, date: str):
        """添加已处理日期，原子性保存"""
        if date not in self.processed_dates:
            self.processed_dates.append(date)
            self._save()

    def _save(self):
        """保存已处理日期"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            # 【修复2】使用标准json.dump，不是pd.io.json.dump
            json.dump({
                "factor_version": self.factor_version,
                "processed_dates": sorted(self.processed_dates)
            }, f, ensure_ascii=False, indent=2)

# ==================== 特征引擎（保留，便于扩展） ====================
class FeatureEngine:
    """【特征引擎】统一调度所有特征类，新增因子仅需注册到feature_classes"""
    def __init__(self):
        # ==================== 新增因子仅需在这里注册 ====================
        self.feature_classes: List[BaseFeature] = [
            SectorHeatFeature()
        ]
        # =================================================================

    def run_single_date(self, trade_date: str, top3_result: Dict, sector_candidate_map: Dict) -> pd.DataFrame:
        """单日特征计算入口，返回当日全量特征DataFrame"""
        date_feature_dfs = []
        for feature in self.feature_classes:
            try:
                feature_df, _ = feature.run(
                    trade_date=trade_date,
                    top3_sectors_result=top3_result,
                    sector_candidate_map=sector_candidate_map
                )
                if not feature_df.empty:
                    date_feature_dfs.append(feature_df)
            except Exception as e:
                logger.error(f"[FeatureEngine] {feature.feature_name} 计算失败：{str(e)}")
                return pd.DataFrame()

        # 合并当日所有特征（按stock_code+trade_date对齐）
        if not date_feature_dfs:
            return pd.DataFrame()
        full_df = date_feature_dfs[0]
        for df in date_feature_dfs[1:]:
            full_df = pd.merge(full_df, df, on=["stock_code", "trade_date"], how="inner")
        # 去重列
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        return full_df


# ==================== 标签引擎（保留，便于扩展） ====================
class LabelEngine:
    """【标签引擎】统一生成标签，新增标签仅需扩展generate_single_date方法"""
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        # 标签需要D+1/D+2数据，结束日期向后偏移5天
        self.label_end_date = (pd.to_datetime(end_date) + timedelta(days=5)).strftime("%Y-%m-%d")
        self.all_trade_dates = get_trade_dates(start_date, self.label_end_date)
        self.date_idx_map = {date: i for i, date in enumerate(self.all_trade_dates)}

    def generate_single_date(self, trade_date: str, stock_list: List[str]) -> pd.DataFrame:
        """单日标签生成，返回当日个股标签DataFrame"""
        if trade_date not in self.date_idx_map:
            return pd.DataFrame()

        current_idx = self.date_idx_map[trade_date]
        # 校验D+1/D+2日期存在
        if current_idx + 2 >= len(self.all_trade_dates):
            return pd.DataFrame()
        d1_date = self.all_trade_dates[current_idx + 1]
        d2_date = self.all_trade_dates[current_idx + 2]

        # 批量获取所需日线数据
        all_daily_df = pd.DataFrame()
        for date in [trade_date, d1_date, d2_date]:
            daily_df = get_daily_kline_data(trade_date=date, ts_code_list=stock_list)
            if not daily_df.empty:
                daily_df['trade_date'] = daily_df['trade_date'].astype(str)
                all_daily_df = pd.concat([all_daily_df, daily_df], ignore_index=True)
        if all_daily_df.empty:
            return pd.DataFrame()

        # 生成标签
        label_rows = []
        for ts_code in stock_list:
            stock_df = all_daily_df[all_daily_df["ts_code"] == ts_code]
            d0_row = stock_df[stock_df["trade_date"] == trade_date]
            d1_row = stock_df[stock_df["trade_date"] == d1_date]
            d2_row = stock_df[stock_df["trade_date"] == d2_date]
            if d0_row.empty or d1_row.empty or d2_row.empty:
                continue

            # label1: D+1收盘涨跌幅≥5%=1
            d1_pct = d1_row["pct_chg"].iloc[0]
            label1 = 1 if d1_pct >= 5.0 else 0

            # label2: D+2开盘高开=1
            d1_close = d1_row["close"].iloc[0]
            d2_open = d2_row["open"].iloc[0]
            label2 = 1 if d2_open > d1_close else 0

            label_rows.append({
                "stock_code": ts_code,
                "trade_date": trade_date,
                "label1": label1,
                "label2": label2
            })
        return pd.DataFrame(label_rows)


# ==================== 数据集组装器（简化，仅保留单日清洗） ====================
class DataSetAssembler:
    """【数据集组装器】单日线数据清洗、校验"""
    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """单日数据校验&清洗，返回有效数据"""
        if df.empty:
            return pd.DataFrame()

        # 1. 核心列非空校验
        required_cols = ["stock_code", "trade_date", "label1", "label2"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"[DataSetAssembler] 缺失核心列：{col}")
                return pd.DataFrame()

        # 2. 去重、标签空值过滤
        df = df.drop_duplicates(subset=["stock_code", "trade_date"])
        df = df.dropna(subset=["label1", "label2"])
        # 3. 因子空值填充0
        df = df.fillna(0)

        return df.reset_index(drop=True)


# ==================== 私有辅助方法（完全保留你的代码，不动） ====================
def _filter_ts_code_by_board(ts_code_list: List[str]) -> List[str]:
    """板块过滤，仅保留主板股票"""
    filtered_list = []
    for ts_code in ts_code_list:
        if not ts_code:
            continue
        # 北交所过滤
        if FILTER_BSE_STOCK and (ts_code.endswith(".BJ") or ts_code.startswith(("83", "87", "88"))):
            continue
        # 科创板过滤
        if FILTER_688_BOARD and ts_code.startswith("688"):
            continue
        # 创业板过滤
        if FILTER_STAR_BOARD and ts_code.startswith(("300", "301", "302")) and ts_code.endswith(".SZ"):
            continue
        filtered_list.append(ts_code)
    return filtered_list


def _check_stock_has_limit_up(ts_code_list: List[str], end_date: str, day_count: int = 10) -> Dict[str, bool]:
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
            limit_up_price = calc_limit_up_price(ts_code, pre_close)
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
        f"近{day_count}日涨停判断完成：总候选{len(ts_code_list)}只,   有涨停基因个股{sum(result_dict.values())}只")
    return result_dict


def validate_train_dataset(csv_path: str):
    """最终训练集全量校验"""
    if not os.path.exists(csv_path):
        raise ValueError("训练集文件不存在")
    df = pd.read_csv(csv_path)
    logger.info(f"【最终校验】训练集总行数：{len(df)}")

    # 核心列校验
    required_cols = ["stock_code", "trade_date", "label1", "label2", "adapt_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失核心列：{missing_cols}")

    # 重复行校验
    duplicate_count = df.duplicated(subset=["stock_code", "trade_date"]).sum()
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["stock_code", "trade_date"])
        logger.warning(f"移除{duplicate_count}行重复数据")

    # 标签空值校验
    label_null_count = df[["label1", "label2"]].isnull().sum().sum()
    if label_null_count > 0:
        df = df.dropna(subset=["label1", "label2"])
        logger.warning(f"移除{label_null_count}行标签空值数据")

    # 保存校验后文件
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"【最终校验】完成，有效行数：{len(df)}")
    return df


# ==================== 【手动触发入口】严格遵循你的要求 ====================
if __name__ == "__main__":
    import warnings
    import json

    warnings.filterwarnings("ignore")

    # ==================== 可配置参数 ====================
    START_DATE = "2026-02-02"
    END_DATE = "2026-02-10"
    OUTPUT_CSV_PATH = os.path.join(os.getcwd(), "train_dataset.csv")
    PROCESSED_DATES_FILE = "processed_dates.json"
    FACTOR_VERSION = "v1.0_sei_base"  # 改因子逻辑必须更新版本号
    # ======================================================

    # ========== 1. 初始化核心组件 ==========
    feature_engine = FeatureEngine()
    label_engine = LabelEngine(START_DATE, END_DATE)
    sector_heat_feature = SectorHeatFeature()
    dates_manager = ProcessedDatesManager(PROCESSED_DATES_FILE, FACTOR_VERSION)

    # ========== 2. 确定待处理日期（仅看processed_dates.json） ==========
    all_trade_dates = get_trade_dates(START_DATE, END_DATE)
    to_process_dates = [d for d in all_trade_dates if not dates_manager.is_processed(d)]
    if not to_process_dates:
        logger.info("✅ 所有日期已处理完成！")
        validate_train_dataset(OUTPUT_CSV_PATH)
        exit(0)
    logger.info(f"待处理日期：{to_process_dates}")

    # ========== 3. 确定CSV写入模式 ==========
    first_write = not os.path.exists(OUTPUT_CSV_PATH)
    fixed_columns = None
    if not first_write:
        fixed_columns = pd.read_csv(OUTPUT_CSV_PATH, nrows=0).columns.tolist()
        logger.info(f"断点续跑，固定列数：{len(fixed_columns)}")

    # ========== 4. 【核心】逐天原子性处理（完全照搬策略里的逻辑，仅最小改动） ==========
    for date in to_process_dates:
        logger.info(f"\n========== 处理日期：{date} ==========")
        try:
            # ==================== 【完全照搬策略里的逻辑，开始】 ====================
            # 1. 调用板块筛选
            top3_result = sector_heat_feature.select_top3_hot_sectors(trade_date=date)
            top3_sectors = top3_result['top3_sectors']
            adapt_score = top3_result['adapt_score']

            top3_sectors_result_map = {date: top3_result}  # 仅当日用
            sector_candidate_map_map = {}  # 仅当日用

            # 2. ST数据入库
            try:
                st_trade_date = date.replace('-', '')
                data_cleaner.insert_stock_st(trade_date=st_trade_date)
                logger.info(f"{date} ST数据入库完成")
            except Exception as e:
                logger.error(f"{date} ST数据入库失败：{e}", exc_info=True)

            # 【关键修复1】循环外提前获取当日全市场日线数据（和策略一致）
            daily_df = get_daily_kline_data(date)

            # 3. 构建当日板块候选池（完全照搬策略里的逻辑）
            sector_candidate_map = {}
            for sector in top3_sectors:
                logger.info(f"处理板块：{sector}")
                try:
                    # 1. 获取板块中所有个股代码
                    sector_stock_raw = get_stocks_in_sector(sector)
                    if not sector_stock_raw:
                        logger.warning(f"板块[{sector}]未查询到对应股票，跳过")
                        sector_candidate_map[sector] = pd.DataFrame()  # 【关键修复2】即使空也赋值
                        continue

                    # 转成纯ts_code列表
                    sector_ts_codes = [item["ts_code"] for item in sector_stock_raw]
                    logger.info(f"板块[{sector}]原始股票数量：{len(sector_ts_codes)}")

                    # ========== 前置过滤（代码列表级别） ==========
                    # 1. 板块过滤（创业板、科创板、北交所）
                    filtered_ts_codes = _filter_ts_code_by_board(sector_ts_codes)
                    if not filtered_ts_codes:
                        logger.warning(f"板块[{sector}]板块过滤后无剩余股票，跳过")
                        sector_candidate_map[sector] = pd.DataFrame()  # 【关键修复2】即使空也赋值
                        continue
                    logger.info(f"板块[{sector}]板块过滤后剩余：{len(filtered_ts_codes)}只")

                    # 2. ST股票过滤
                    filtered_ts_codes = filter_st_stocks(filtered_ts_codes, date)
                    if not filtered_ts_codes:
                        logger.warning(f"板块[{sector}]ST过滤后无剩余股票，跳过")
                        sector_candidate_map[sector] = pd.DataFrame()  # 【关键修复2】即使空也赋值
                        continue
                    logger.info(f"板块[{sector}]ST过滤后剩余：{len(filtered_ts_codes)}只")

                    # 3. 筛选该板块的日线数据（用循环外提前获取的daily_df）
                    sector_daily_df = daily_df[daily_df["ts_code"].isin(filtered_ts_codes)].copy()
                    if sector_daily_df.empty:
                        logger.warning(f"板块[{sector}]当日无有效日线数据，跳过")
                        sector_candidate_map[sector] = pd.DataFrame()  # 【关键修复2】即使空也赋值
                        continue

                    # 4. 近10个交易日有涨停筛选（和策略完全一致）
                    candidate_ts_codes = sector_daily_df["ts_code"].unique().tolist()
                    has_limit_up_map = _check_stock_has_limit_up(
                        ts_code_list=candidate_ts_codes,
                        end_date=date,
                        day_count=10
                    )
                    # 保留近10日有涨停的股票
                    keep_ts_codes = [ts_code for ts_code, has_limit_up in has_limit_up_map.items() if has_limit_up]
                    filtered_df = sector_daily_df[sector_daily_df["ts_code"].isin(keep_ts_codes)]
                    sector_candidate_map[sector] = filtered_df
                    logger.info(f"板块[{sector}]最终候选股数量：{len(filtered_df)}")
                except Exception as e:
                    logger.error(f"板块[{sector}]处理失败：{e}", exc_info=True)
                    sector_candidate_map[sector] = pd.DataFrame()  # 【关键修复2】即使失败也赋值
                    continue

                # ==================== 【完全照搬策略里的逻辑，结束】 ====================

            # 4. 存当日板块候选池
            sector_candidate_map_map[date] = sector_candidate_map



            # ========== 5. 【补全】特征计算（保留你的原有流程） ==========
            if not sector_candidate_map:
                logger.warning(f"{date} 无有效候选池，跳过")
                continue
            feature_df = feature_engine.run_single_date(date, top3_result, sector_candidate_map)
            if feature_df.empty:
                logger.warning(f"{date} 特征计算失败，跳过")
                continue
            # 添加adapt_score因子
            feature_df["adapt_score"] = adapt_score

            # ========== 6. 【补全】标签生成（保留你的原有流程） ==========
            label_df = label_engine.generate_single_date(date, feature_df["stock_code"].unique().tolist())
            if label_df.empty:
                logger.warning(f"{date} 标签生成失败，跳过")
                continue

            # ========== 7. 【补全】数据合并&清洗（保留你的原有流程） ==========
            merged_df = pd.merge(feature_df, label_df, on=["stock_code", "trade_date"], how="left")
            clean_df = DataSetAssembler.validate_and_clean(merged_df)
            if clean_df.empty:
                logger.warning(f"{date} 无有效数据，跳过")
                continue

            # ========== 8. 【补全】列对齐（保留你的原有流程） ==========
            if first_write:
                fixed_columns = clean_df.columns.tolist()
            else:
                clean_df = clean_df.reindex(columns=fixed_columns, fill_value=0)

            # ========== 9. 【原子性写入】一次性写入当日完整数据（保留你的原有流程） ==========
            clean_df.to_csv(
                OUTPUT_CSV_PATH,
                mode="a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False

            # ========== 10. 【写入成功才标记】更新已处理记录（保留你的原有流程） ==========
            dates_manager.add(date)
            logger.info(f"✅ {date} 处理完成，写入{len(clean_df)}行")

        except Exception as e:
            logger.error(f"{date} 处理失败：{str(e)}", exc_info=True)
            continue

    # ========== 11. 最终全量校验 ==========
    logger.info(f"\n========== 全量处理完成 ==========")
    if os.path.exists(OUTPUT_CSV_PATH):
        validate_train_dataset(OUTPUT_CSV_PATH)
    else:
        logger.error("❌ 训练集生成失败！")