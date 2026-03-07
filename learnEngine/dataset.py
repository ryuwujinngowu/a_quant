# learnEngine/dataset.py
from collections import defaultdict
import os
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
from utils.common_tools import (
    get_stocks_in_sector,
    filter_st_stocks,
    get_trade_dates,
    get_daily_kline_data,
    sort_by_recent_gain,
    calc_limit_up_price
)
from config.config import (
    FILTER_BSE_STOCK,
    FILTER_STAR_BOARD,
    FILTER_688_BOARD
)
from utils.log_utils import logger
from strategies.base_strategy import BaseStrategy
from features.market_stats import SectorHeatFeature
from data.data_cleaner import data_cleaner

class FeatureEngine:
    """【特征引擎】统一调度所有特征计算类"""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self.trade_dates = get_trade_dates(start_date, end_date)
        # 注册所有特征类（后续加新特征，只需在这里append）
        self.feature_classes = [
            SectorHeatFeature()
        ]

    def run_all_features(self, top3_sectors_result_map: Dict[str, Dict],
                         sector_candidate_map_map: Dict[str, Dict]) -> pd.DataFrame:
        """
        运行所有特征，拼接成一个大DataFrame
        :param top3_sectors_result_map: {trade_date: top3_sectors_result}
        :param sector_candidate_map_map: {trade_date: sector_candidate_map}
        :return: 全特征DataFrame
        """
        all_feature_dfs = []

        for trade_date in self.trade_dates:
            logger.info(f"[FeatureEngine] 处理日期：{trade_date}")
            top3_result = top3_sectors_result_map.get(trade_date, {})
            sector_candidate_map = sector_candidate_map_map.get(trade_date, {})

            for feature in self.feature_classes:
                try:
                    feature_df, _ = feature.run(
                        trade_date=trade_date,
                        top3_sectors_result=top3_result,
                        sector_candidate_map=sector_candidate_map
                    )
                    if not feature_df.empty:
                        all_feature_dfs.append(feature_df)
                except Exception as e:
                    logger.error(f"[FeatureEngine] 特征{feature.feature_name}计算失败：{str(e)}")

        # 拼接所有特征（按stock_code+trade_date对齐）
        if not all_feature_dfs:
            return pd.DataFrame()

        # 简单拼接：假设所有特征的行都是一一对应的（个股-交易日）
        full_feature_df = pd.concat(all_feature_dfs, axis=1)
        # 去重列（避免重复的stock_code、trade_date）
        full_feature_df = full_feature_df.loc[:, ~full_feature_df.columns.duplicated()]
        return full_feature_df


class LabelEngine:
    """【标签引擎】统一生成所有标签"""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        # 标签需要用到D+1、D+2的数据，所以结束日期往后推2天
        self.label_end_date = (pd.to_datetime(end_date) + timedelta(days=5)).strftime("%Y-%m-%d")
        self.all_trade_dates = get_trade_dates(start_date, self.label_end_date)

    def generate_labels(self, stock_list: List[str]) -> pd.DataFrame:
        """
        生成标签DataFrame
        :param stock_list: 需要生成标签的股票列表
        :return: DataFrame，列：stock_code, trade_date, label1, label2
        """
        label_rows = []
        # 批量获取所有股票的D+1、D+2日线数据
        all_daily_df = pd.DataFrame()
        try:
            for date in self.all_trade_dates:
                daily_df = get_daily_kline_data(trade_date=date, ts_code_list=stock_list)
                if not daily_df.empty:
                    daily_df['trade_date'] = daily_df['trade_date'].astype(str)
                    all_daily_df = pd.concat([all_daily_df, daily_df], ignore_index=True)
        except Exception as e:
            logger.error(f"[LabelEngine] 获取标签数据失败：{str(e)}")
            return pd.DataFrame()

        if all_daily_df.empty:
            return pd.DataFrame()

        # 构建日期索引映射：trade_date -> 下一个交易日、下下个交易日
        date_idx_map = {date: i for i, date in enumerate(self.all_trade_dates)}

        for ts_code in stock_list:
            stock_df = all_daily_df[all_daily_df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
            if stock_df.empty:
                continue

            for i, row in stock_df.iterrows():
                trade_date = row["trade_date"]
                if trade_date not in date_idx_map or trade_date > self.end_date:
                    continue

                # 找D+1、D+2的交易日
                current_idx = date_idx_map[trade_date]
                d1_date = self.all_trade_dates[current_idx + 1] if (current_idx + 1) < len(
                    self.all_trade_dates) else None
                d2_date = self.all_trade_dates[current_idx + 2] if (current_idx + 2) < len(
                    self.all_trade_dates) else None

                if not d1_date or not d2_date:
                    continue

                # 计算label1：D+1收盘涨跌幅≥5%=1
                d1_row = stock_df[stock_df["trade_date"] == d1_date]
                label1 = 0
                if not d1_row.empty:
                    d1_pct = d1_row["pct_chg"].iloc[0]
                    label1 = 1 if d1_pct >= 5.0 else 0

                # 计算label2：D+2开盘高开=1
                d2_row = stock_df[stock_df["trade_date"] == d2_date]
                label2 = 0
                if not d2_row.empty:
                    d1_close = d1_row["close"].iloc[0] if not d1_row.empty else row["close"]
                    d2_open = d2_row["open"].iloc[0]
                    label2 = 1 if d2_open > d1_close else 0

                label_rows.append({
                    "stock_code": ts_code,
                    "trade_date": trade_date,
                    "label1": label1,
                    "label2": label2
                })

        label_df = pd.DataFrame(label_rows)
        logger.info(f"[LabelEngine] 标签生成完成，共{len(label_df)}行")
        return label_df


class DataSetAssembler:
    """【数据集组装器】统一组装特征+标签、清洗、输出CSV"""

    def __init__(self, full_feature_df: pd.DataFrame, label_df: pd.DataFrame):
        self.full_feature_df = full_feature_df
        self.label_df = label_df

    def assemble(self) -> pd.DataFrame:
        """组装最终训练集"""
        if self.full_feature_df.empty or self.label_df.empty:
            logger.error("[DataSetAssembler] 特征或标签为空，无法组装")
            return pd.DataFrame()

        # 1. 拼接特征和标签（按stock_code+trade_date对齐）
        logger.info("[DataSetAssembler] 拼接特征和标签")
        merged_df = pd.merge(
            self.full_feature_df,
            self.label_df,
            on=["stock_code", "trade_date"],
            how="inner"
        )

        # 2. 数据清洗
        logger.info("[DataSetAssembler] 数据清洗")
        # 去重
        merged_df = merged_df.drop_duplicates(subset=["stock_code", "trade_date"])
        # 去空值（标签列不能为空）
        merged_df = merged_df.dropna(subset=["label1", "label2"])
        # 填充因子空值为0
        merged_df = merged_df.fillna(0)

        # 3. 排序（按日期排序，保证时间序列）
        merged_df = merged_df.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)

        logger.info(f"[DataSetAssembler] 最终训练集组装完成，共{len(merged_df)}行")
        return merged_df

    def save_to_csv(self, output_path: str = "train_dataset.csv"):
        """保存为CSV"""
        final_df = self.assemble()
        if not final_df.empty:
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"[DataSetAssembler] 训练集已保存至：{output_path}")


# ==================== 示例调用入口 ====================
def build_train_dataset(
        start_date: str,
        end_date: str,
        top3_sectors_result_map: Dict[str, Dict],
        sector_candidate_map_map: Dict[str, Dict]
) -> pd.DataFrame:
    """
    【对外统一入口】构建训练集
    :param start_date: 训练集开始日期
    :param end_date: 训练集结束日期
    :param top3_sectors_result_map: {trade_date: select_top3_hot_sectors返回的结果}
    :param sector_candidate_map_map: {trade_date: sector_candidate_map}
    :return: 最终训练集DataFrame
    """
    # 1. 运行所有特征
    feature_engine = FeatureEngine(start_date, end_date)
    full_feature_df = feature_engine.run_all_features(top3_sectors_result_map, sector_candidate_map_map)
    if full_feature_df.empty:
        return pd.DataFrame()

    # 2. 生成标签
    stock_list = full_feature_df["stock_code"].unique().tolist()
    label_engine = LabelEngine(start_date, end_date)
    label_df = label_engine.generate_labels(stock_list)
    if label_df.empty:
        return pd.DataFrame()

    # 3. 组装并保存
    assembler = DataSetAssembler(full_feature_df, label_df)
    assembler.save_to_csv()
    return assembler.assemble()

def _filter_ts_code_by_board(ts_code_list: List[str]) -> List[str]:
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
    """训练集最终校验函数（必执行）"""
    if not os.path.exists(csv_path):
        raise ValueError("CSV文件不存在")
    df = pd.read_csv(csv_path)
    logger.info(f"开始校验训练集：{csv_path}")

    # 校验1：核心列存在
    required_cols = ["stock_code", "trade_date", "label1", "label2", "adapt_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失核心列：{missing_cols}")

    # 校验2：无重复行（股票+日期唯一）
    duplicate_rows = df.duplicated(subset=["stock_code", "trade_date"]).sum()
    if duplicate_rows > 0:
        logger.warning(f"发现{duplicate_rows}行重复数据，已自动去重")
        df = df.drop_duplicates(subset=["stock_code", "trade_date"])

    # 校验3：无空值（标签列）
    label_null = df[["label1", "label2"]].isnull().sum().sum()
    if label_null > 0:
        logger.warning(f"发现{label_null}行标签空值，已删除")
        df = df.dropna(subset=["label1", "label2"])

    # 校验4：日期格式正确
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    invalid_dates = df["trade_date"].isnull().sum()
    if invalid_dates > 0:
        raise ValueError(f"发现{invalid_dates}行无效日期")

    # 保存校验后的文件
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"校验完成！最终有效行数：{len(df)}")
    return df

# 调用校验（放在程序最后）
validate_train_dataset(OUTPUT_CSV_PATH)

if __name__ == "__main__":
    import warnings
    import os
    import json
    from decimal import Decimal
    warnings.filterwarnings("ignore")

    # ========== 【断点续跑核心配置】 ==========
    START_DATE = "2026-03-01"
    END_DATE = "2026-03-05"
    OUTPUT_CSV_PATH = os.path.join(os.getcwd(), "train_dataset.csv")
    # 记录已处理日期的文件（关键！）
    PROCESSED_DATES_FILE = "processed_dates.json"
    # 因子计算逻辑版本号（核心！防止口径不一致）
    FACTOR_VERSION = "v1.0_sei_base"

    # ========== 1. 加载已处理日期（断点续跑） ==========
    processed_dates = []
    if os.path.exists(PROCESSED_DATES_FILE):
        try:
            with open(PROCESSED_DATES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 校验因子版本：版本不一致，强制从头跑
                if data.get("factor_version") != FACTOR_VERSION:
                    logger.warning(f"因子版本变更（旧：{data.get('factor_version')} 新：{FACTOR_VERSION}），强制从头生成")
                    processed_dates = []
                else:
                    processed_dates = data.get("processed_dates", [])
        except Exception as e:
            logger.error(f"读取已处理日期失败，强制从头跑：{e}")
            processed_dates = []
    logger.info(f"已处理日期数：{len(processed_dates)} | 待处理日期：{START_DATE} ~ {END_DATE}")

    # ========== 2. 筛选待处理日期（跳过已处理） ==========
    all_trade_dates = get_trade_dates(START_DATE, END_DATE)
    to_process_dates = [d for d in all_trade_dates if d not in processed_dates]
    if not to_process_dates:
        logger.info("✅ 所有日期已处理完成，无需继续！")
        # 校验最终CSV
        if os.path.exists(OUTPUT_CSV_PATH):
            final_df = pd.read_csv(OUTPUT_CSV_PATH)
            logger.info(f"最终训练集：{len(final_df)}行 | 日期范围：{final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
        exit(0)
    logger.info(f"待处理日期数：{len(to_process_dates)} | 列表：{to_process_dates}")

    # ========== 3. 初始化其他变量（不变） ==========
    top3_sectors_result_map = {}
    sector_candidate_map_map = {}
    total_written_rows = 0
    sector_heat_feature = SectorHeatFeature()
    label_engine = LabelEngine(START_DATE, (pd.to_datetime(END_DATE) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"))

    # ========== 4. 处理CSV写入模式（断点续跑时不删除旧文件） ==========
    first_write = not os.path.exists(OUTPUT_CSV_PATH)  # 有旧文件则不写表头
    if first_write:
        logger.info("首次运行，创建新CSV文件")
    else:
        # 读取旧文件，统计已写入行数
        old_df = pd.read_csv(OUTPUT_CSV_PATH)
        total_written_rows = len(old_df)
        logger.info(f"断点续跑，旧CSV已有 {total_written_rows} 行数据")

    # ========== 5. 遍历待处理日期（仅处理未完成的） ==========
    for date in to_process_dates:
        logger.info(f"\n========== 处理日期：{date} ==========")
        try:
            # ---------- 原有逻辑（选股+特征+标签+写入）保持不变 ----------
            top3_result = sector_heat_feature.select_top3_hot_sectors(trade_date=date)
            if not top3_result or "top3_sectors" not in top3_result:
                logger.warning(f"{date} 板块筛选结果为空，跳过")
                continue
            top3_sectors = top3_result['top3_sectors']
            adapt_score = top3_result.get('adapt_score', 0)
            top3_sectors_result_map[date] = top3_result

            # ST入库（可选）
            try:
                st_trade_date = date.replace('-', '')
                data_cleaner.insert_stock_st(trade_date=st_trade_date)
            except Exception as e:
                logger.warning(f"{date} ST入库失败（不影响）：{str(e)[:50]}")

            # 构建板块候选池
            sector_candidate_map = {}
            for sector in top3_sectors:
                logger.info(f"处理板块：{sector}")
                sector_stock_raw = get_stocks_in_sector(sector)
                if not sector_stock_raw:
                    logger.warning(f"板块[{sector}]无原始股票，跳过")
                    continue
                sector_ts_codes = [item["ts_code"] for item in sector_stock_raw]
                filtered_ts_codes = _filter_ts_code_by_board(sector_ts_codes)
                filtered_ts_codes = filter_st_stocks(filtered_ts_codes, date)
                if not filtered_ts_codes:
                    logger.warning(f"板块[{sector}]过滤后无股票，跳过")
                    continue

                daily_df = get_daily_kline_data(date)
                if daily_df.empty:
                    logger.warning(f"{date} 无日线数据，跳过板块[{sector}]")
                    continue
                sector_daily_df = daily_df[daily_df["ts_code"].isin(filtered_ts_codes)].copy()
                if sector_daily_df.empty:
                    logger.warning(f"板块[{sector}]无匹配日线数据，跳过")
                    continue

                candidate_ts_codes = sector_daily_df["ts_code"].unique().tolist()
                has_limit_up_map = _check_stock_has_limit_up(candidate_ts_codes, date, 10)
                keep_ts_codes = [ts for ts, flag in has_limit_up_map.items() if flag]
                if not keep_ts_codes:
                    logger.warning(f"板块[{sector}]近10日无涨停股票，跳过")
                    continue
                filtered_df = sector_daily_df[sector_daily_df["ts_code"].isin(keep_ts_codes)]
                sector_candidate_map[sector] = filtered_df
                logger.info(f"板块[{sector}]最终候选股数：{len(filtered_df)}")

            sector_candidate_map_map[date] = sector_candidate_map

            # 计算特征
            if not sector_candidate_map:
                logger.warning(f"{date} 无有效板块候选池，跳过特征计算")
                continue
            result_df, factor_dict = sector_heat_feature.calculate(
                trade_date=date,
                top3_sectors_result=top3_result,
                sector_candidate_map=sector_candidate_map
            )
            if result_df.empty:
                logger.warning(f"{date} 特征计算结果为空，跳过")
                continue
            result_df['adapt_score'] = adapt_score

            # 生成标签
            stock_list_day = result_df["stock_code"].unique().tolist()
            label_df_day = label_engine.generate_labels(stock_list_day)
            if label_df_day.empty:
                logger.warning(f"{date} 标签生成为空，跳过写入")
                continue
            label_df_day = label_df_day[label_df_day["trade_date"] == date].copy()
            if label_df_day.empty:
                logger.warning(f"{date} 无当日标签，跳过写入")
                continue

            # 组装+清洗
            merged_df = pd.merge(result_df, label_df_day, on=["stock_code", "trade_date"], how="inner")
            merged_df = merged_df.dropna(subset=["label1", "label2"]).fillna(0)
            if merged_df.empty:
                logger.warning(f"{date} 组装后无有效数据，跳过写入")
                continue

            # 写入CSV（追加模式）
            merged_df.to_csv(
                OUTPUT_CSV_PATH,
                mode="a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False  # 第一次写入后，后续不再写表头

            # ---------- 断点续跑核心：记录已处理日期 ----------
            day_written = len(merged_df)
            total_written_rows += day_written
            processed_dates.append(date)  # 加入已处理列表
            # 即时保存已处理日期（防止再次中断）
            with open(PROCESSED_DATES_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "factor_version": FACTOR_VERSION,
                    "processed_dates": processed_dates,
                    "total_rows": total_written_rows
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ {date} 处理完成！当日写入{day_written}行 | 累计{total_written_rows}行 | 已记录到断点文件")

        except Exception as e:
            logger.error(f"{date} 处理失败：{str(e)[:100]}", exc_info=True)
            continue

    # ========== 6. 最终校验（关键！防止训练集异常） ==========
    logger.info(f"\n========== 所有待处理日期完成 ==========")
    # 校验1：CSV文件存在且有数据
    if os.path.exists(OUTPUT_CSV_PATH):
        final_df = pd.read_csv(OUTPUT_CSV_PATH)
        # 校验2：去重（防止重复写入）
        final_df = final_df.drop_duplicates(subset=["stock_code", "trade_date"])
        # 校验3：日期连续性（可选，量化训练集建议时序连续）
        final_dates = sorted(final_df["trade_date"].unique())
        expected_dates = get_trade_dates(START_DATE, END_DATE)
        missing_dates = [d for d in expected_dates if d not in final_dates]
        if missing_dates:
            logger.warning(f"⚠️ 训练集缺失日期：{missing_dates}")
        # 重新保存校验后的CSV
        final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 训练集最终校验完成！")
        logger.info(f"📁 文件路径：{OUTPUT_CSV_PATH}")
        logger.info(f"📊 最终行数：{len(final_df)} | 日期范围：{final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
        logger.info(f"⚠️ 缺失日期数：{len(missing_dates)}")
    else:
        logger.error(f"❌ 最终CSV文件未生成！")