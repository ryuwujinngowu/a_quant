import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from utils.log_utils import logger
from utils.db_utils import db
from utils.tools import calc_15_years_date_range, escape_mysql_reserved_words
from data.data_fetcher import data_fetcher


class DataCleaner:
    """数据清洗+入库核心类（终极版：100%保障exchange非空）"""

    def _clean_special_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """通用格式清洗（终极版：强制保障exchange字段存在且非空）"""
        df_cleaned = df.copy()

        # ====================== 核心强化：exchange字段100%非空 ======================
        # 1. 如果接口未返回exchange字段，手动添加
        if "exchange" not in df_cleaned.columns:
            logger.warning("接口未返回exchange字段，手动添加并赋值为UNKNOWN")
            df_cleaned["exchange"] = "UNKNOWN"
        else:
            # 2. 映射交易所名称（上证→SSE，深证→SZSE，北交→BSE）
            exchange_map = {"上证": "SSE", "深证": "SZSE", "北交": "BSE"}
            df_cleaned["exchange"] = df_cleaned["exchange"].replace(exchange_map)
            # 3. 强制填充空值（包括NaN/空字符串/None）
            df_cleaned["exchange"] = df_cleaned["exchange"].replace(["", np.nan, None], "UNKNOWN")

        # ====================== 其他字段清洗 ======================
        # 日期字段格式统一
        date_fields = ["list_date", "delist_date", "end_date", "start_date"]
        for field in date_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = pd.to_datetime(
                    df_cleaned[field], format="%Y%m%d", errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                # 日期字段非空保障
                df_cleaned[field] = df_cleaned[field].fillna("1970-01-01")

        # 股票代码格式校验+非空保障
        if "ts_code" in df_cleaned.columns:
            df_cleaned = df_cleaned[df_cleaned["ts_code"].str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False)]
            df_cleaned["ts_code"] = df_cleaned["ts_code"].fillna("UNKNOWN")
        else:
            logger.warning("接口未返回ts_code字段，手动添加并赋值为UNKNOWN")
            df_cleaned["ts_code"] = "UNKNOWN"

        # 核心NOT NULL字段非空兜底
        not_null_fields = ["ts_code", "symbol", "name", "exchange", "list_date"]
        for field in not_null_fields:
            if field in df_cleaned.columns:
                if df_cleaned[field].dtype == "object":
                    df_cleaned[field] = df_cleaned[field].fillna("UNKNOWN")
                elif "date" in field:
                    df_cleaned[field] = df_cleaned[field].fillna("1970-01-01")
                elif "int" in str(df_cleaned[field].dtype) or "float" in str(df_cleaned[field].dtype):
                    df_cleaned[field] = df_cleaned[field].fillna(0)

        # 其他字段空值填充
        for col in df_cleaned.columns:
            if col not in not_null_fields:
                if df_cleaned[col].dtype == "object":
                    df_cleaned[col] = df_cleaned[col].fillna("")
                elif "int" in str(df_cleaned[col].dtype) or "float" in str(df_cleaned[col].dtype):
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                elif "datetime" in str(df_cleaned[col].dtype):
                    df_cleaned[col] = df_cleaned[col].fillna(None)

        # 最终验证exchange字段是否非空
        exchange_null_count = df_cleaned["exchange"].isnull().sum()
        if exchange_null_count > 0:
            logger.error(f"exchange字段仍有{exchange_null_count}个空值，强制填充为UNKNOWN")
            df_cleaned["exchange"] = df_cleaned["exchange"].fillna("UNKNOWN")

        logger.info(f"格式清洗完成，exchange字段非空验证通过，行数：{len(df_cleaned)}")
        return df_cleaned

    def clean_and_insert_stockcompany(self, table_name: str = "stock_company") -> Optional[int]:
        """
        上市公司工商信息（stock_company）全字段自动化入库
        适配接口文档：https://tushare.pro/document/2?doc_id=112
        """
        logger.info("===== 开始上市公司基本信息（stock_company）全字段自动化入库 =====")

        # 1. 调用DataFetcher获取stock_company原始数据
        raw_df = data_fetcher.get_stock_company()
        if raw_df is None or raw_df.empty:
            logger.warning("stock_company接口返回原始数据为空，跳过入库")
            return 0

        # 2. 专属格式清洗（适配stock_company字段）
        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            logger.warning("清洗后数据为空，跳过入库")
            return 0

        # 3. 验证接口核心字段是否存在
        api_field_list = [
            "ts_code", "com_name", "com_id", "exchange", "chairman",
            "manager", "secretary", "reg_capital", "setup_date",
            "province", "city", "website", "email", "employees"
        ]
        missing_api_fields = [f for f in api_field_list if f not in cleaned_df.columns]
        if missing_api_fields:
            logger.warning(f"接口缺失核心字段：{missing_api_fields}（请检查DataFetcher是否获取全字段）")
        else:
            logger.info("✅ 所有接口核心字段均已获取")

        # 4. 数据库字段自动新增（精准适配stock_company字段类型）
        db_columns = db.get_table_columns(table_name)
        exclude_columns = ["id", "created_at", "updated_at"]  # 排除自动维护字段
        db_columns = [col for col in db_columns if col not in exclude_columns]
        api_columns = cleaned_df.columns.tolist()
        missing_columns = [col for col in api_columns if col not in db_columns]

        if missing_columns:
            logger.info(f"数据库表{table_name}缺失字段：{missing_columns}，开始自动新增")
            # 精准映射stock_company字段类型（完全匹配建表语句）
            col_type_mapping = {
                "ts_code": "CHAR(9)",
                "com_name": "VARCHAR(128)",
                "com_id": "CHAR(18)",
                "exchange": "VARCHAR(8)",
                "chairman": "VARCHAR(32)",
                "manager": "VARCHAR(32)",
                "secretary": "VARCHAR(32)",
                "reg_capital": "DECIMAL(20,2)",
                "setup_date": "DATE",
                "province": "VARCHAR(16)",
                "city": "VARCHAR(16)",
                "introduction": "TEXT",
                "website": "VARCHAR(128)",
                "email": "VARCHAR(64)",
                "office": "VARCHAR(128)",
                "employees": "INT",
                "main_business": "TEXT",
                "business_scope": "TEXT"
            }
            for col in missing_columns:
                col_type = col_type_mapping.get(col, "VARCHAR(255)")
                col_comment = f"{col}（stock_company接口字段）"
                db.add_table_column(table_name, col, col_type, comment=col_comment)

        # 5. 过滤出接口和数据库共有的字段（避免字段不匹配报错）
        final_db_columns = db.get_table_columns(table_name)
        final_db_columns = [col for col in final_db_columns if col not in exclude_columns]
        common_columns = [col for col in api_columns if col in final_db_columns]
        final_df = cleaned_df[common_columns].copy()

        # 6. 批量入库（基于ts_code主键去重，重复则更新）
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True  # 重复数据自动更新
            )
            if affected_rows is None:
                logger.error(f"表{table_name}全字段入库失败")
                return 0
            logger.info(
                f"✅ 表{table_name}（stock_company）入库完成，影响行数：{affected_rows}，入库字段数：{len(common_columns)}")

            # 打印核心字段示例，验证入库数据正确性
            if not final_df.empty:
                sample_fields = ["ts_code", "com_name", "chairman", "reg_capital", "province"]
                sample_fields = [f for f in sample_fields if f in final_df.columns]
                logger.info(f"核心字段入库示例：\n{final_df[sample_fields].head(5)}")

            return affected_rows
        except Exception as e:
            logger.error(f"表{table_name}（stock_company）入库异常：{str(e)}", exc_info=True)
            return 0

    def clean_and_insert_stockbase(self, table_name: str = "stock_basic") -> Optional[int]:
        """股票基础数据全字段自动化入库（终极版）"""
        logger.info("===== 开始股票基础数据全字段自动化入库（终极版） =====")
        logger.info(f"目标表名：{table_name}")

        # 1. 获取接口全字段原始数据
        raw_df = data_fetcher.get_stockbase(list_status="L")
        if raw_df is None or raw_df.empty:
            logger.warning("接口返回原始数据为空，跳过入库")
            return 0

        # 2. 终极格式清洗（100%保障exchange非空）
        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            return 0

        # 3. 获取数据库表现有字段
        db_columns = db.get_table_columns(table_name)
        exclude_columns = ["id", "created_at", "updated_at"]
        db_columns = [col for col in db_columns if col not in exclude_columns]

        # 4. 对比接口字段和数据库字段，找出缺失字段
        api_columns = cleaned_df.columns.tolist()
        missing_columns = [col for col in api_columns if col not in db_columns]

        # 5. 自动新增缺失字段到数据库（新增字段时强制加默认值）
        if missing_columns:
            logger.info(f"数据库表{table_name}缺失字段：{missing_columns}，开始自动新增")
            col_type_mapping = {
                "list_date": "DATE NOT NULL DEFAULT '1970-01-01'",
                "delist_date": "DATE DEFAULT NULL",
                "total_share": "BIGINT DEFAULT 0",
                "float_share": "BIGINT DEFAULT 0",
                "free_share": "BIGINT DEFAULT 0",
                "total_mv": "DECIMAL(20,2) DEFAULT 0.00",
                "circ_mv": "DECIMAL(20,2) DEFAULT 0.00",
                "exchange": "VARCHAR(8) NOT NULL DEFAULT 'UNKNOWN'",
                "ts_code": "VARCHAR(9) NOT NULL DEFAULT 'UNKNOWN'",
                "symbol": "VARCHAR(6) NOT NULL DEFAULT 'UNKNOWN'",
                "name": "VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN'"
            }
            for col in missing_columns:
                col_type = col_type_mapping.get(col, "VARCHAR(255) NOT NULL DEFAULT ''")
                db.add_table_column(table_name, col, col_type)

        # 6. 重新获取数据库字段（包含新增字段）
        final_db_columns = db.get_table_columns(table_name)
        final_db_columns = [col for col in final_db_columns if col not in exclude_columns]

        # 7. 过滤出接口和数据库共有的字段
        common_columns = [col for col in api_columns if col in final_db_columns]
        final_df = cleaned_df[common_columns].copy()

        # 8. 批量入库（重复数据自动更新）
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            if affected_rows is None:
                logger.error(f"表{table_name}全字段入库失败")
                return 0
            logger.info(f"表{table_name}全字段入库完成，影响行数：{affected_rows}，入库字段数：{len(common_columns)}")
            return affected_rows
        except Exception as e:
            logger.error(f"表{table_name}全字段入库异常：{str(e)}")
            return 0

    def _clean_kline_day_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """日K数据专属清洗（完全适配当前数据库表字段设计）"""
        if raw_df.empty:
            logger.warning("原始日K数据为空，跳过清洗")
            return pd.DataFrame()

        df_cleaned = raw_df.copy()

        # 1. 接口字段 → 数据库字段映射
        field_mapping = {
            "vol": "volume",  # 接口vol → 数据库volume
            "change": "change1"  # 接口change → 数据库change1
        }
        df_cleaned = df_cleaned.rename(columns=field_mapping)

        # 2. 核心字段列表（严格匹配数据库表字段）
        core_fields = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change1", "pct_chg", "volume", "amount",
            "turnover_rate", "swing", "limit_up", "limit_down", "update_time", "reserved"
        ]

        # 3. 填充数据库新增字段的默认值（接口未返回的字段）
        default_values = {
            "turnover_rate": 0.0,
            "swing": 0.0,
            "limit_up": 0.0,
            "limit_down": 0.0,
            "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reserved": ""
        }
        for field, default_val in default_values.items():
            if field not in df_cleaned.columns:
                df_cleaned[field] = default_val

        # 4. 格式清洗（复用已有逻辑风格）
        # 日期格式转换：YYYYMMDD → YYYY-MM-DD（适配数据库DATE类型）
        df_cleaned["trade_date"] = pd.to_datetime(
            df_cleaned["trade_date"], format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        df_cleaned["trade_date"] = df_cleaned["trade_date"].fillna("1970-01-01")

        # 股票代码格式校验
        df_cleaned = df_cleaned[df_cleaned["ts_code"].str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False)]
        df_cleaned["ts_code"] = df_cleaned["ts_code"].fillna("UNKNOWN")

        # 数值字段类型转换+空值填充（严格匹配数据库类型）
        numeric_fields = {
            "open": float, "high": float, "low": float, "close": float,
            "pre_close": float, "change1": float, "pct_chg": float,
            "volume": "int64",  # 数据库是bigint
            "amount": float,  # 数据库是decimal(18,2)，先转float再入库
            "turnover_rate": float, "swing": float, "limit_up": float, "limit_down": float
        }
        for field, dtype in numeric_fields.items():
            df_cleaned[field] = pd.to_numeric(df_cleaned[field], errors="coerce").fillna(0)
            if dtype == "int64":
                df_cleaned[field] = df_cleaned[field].astype("int64")
            else:
                df_cleaned[field] = df_cleaned[field].astype(dtype)

        # 字符串字段填充
        df_cleaned["reserved"] = df_cleaned["reserved"].fillna("").astype(str)

        # 5. 去重（按股票代码+交易日期去重，避免重复数据）
        df_cleaned = df_cleaned.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

        # 6. 只保留核心字段，确保和数据库表完全对齐
        df_cleaned = df_cleaned[core_fields]

        logger.info(f"日K数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_cleaned)}行")
        return df_cleaned

    def clean_and_insert_kline_day(self, table_name: str = "kline_day") -> Optional[int]:
        """
        全市场A股近15年日K数据自动化清洗+入库
        核心逻辑：循环调用fetch_kline_day → 清洗 → 自动适配数据库字段 → 批量入库
        """
        logger.info("===== 开始全市场A股近15年日K数据清洗入库 =====")

        # 1. 前置准备：获取股票代码（调用db_utils）和日期范围（调用tools）
        stock_codes = db.get_all_a_stock_codes()
        if not stock_codes:
            logger.error("无有效股票代码，终止日K数据入库")
            return 0
        start_date, end_date = calc_15_years_date_range()

        # 2. 初始化统计指标
        total_success = 0
        total_failed = 0
        total_ingest_rows = 0
        failed_codes = []

        # 3. 循环处理每只股票的日K数据
        for idx, ts_code in enumerate(stock_codes):
            logger.info(f"处理进度：{idx + 1}/{len(stock_codes)} | 股票代码：{ts_code}")
            try:
                # 3.1 调用data_fetcher获取原始日K数据
                raw_df = data_fetcher.fetch_kline_day(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                if raw_df.empty:
                    logger.warning(f"{ts_code} 无日K数据，跳过")
                    total_failed += 1
                    failed_codes.append(ts_code)
                    continue

                # 3.2 日K数据专属清洗
                cleaned_df = self._clean_kline_day_data(raw_df)
                if cleaned_df.empty:
                    logger.warning(f"{ts_code} 清洗后无有效数据，跳过")
                    total_failed += 1
                    failed_codes.append(ts_code)
                    continue

                # 3.3 自动适配数据库字段（新增缺失字段）
                db_columns = db.get_table_columns(table_name)
                exclude_columns = ["id", "created_at", "updated_at"]
                db_columns = [col for col in db_columns if col not in exclude_columns]
                api_columns = cleaned_df.columns.tolist()
                missing_columns = [col for col in api_columns if col not in db_columns]

                if missing_columns:
                    logger.info(f"表{table_name}缺失字段：{missing_columns}，开始自动新增")
                    # 精准映射kline_day字段类型（匹配金融数据存储规范）
                    col_type_mapping = {
                        "ts_code": "VARCHAR(9) NOT NULL DEFAULT 'UNKNOWN'",
                        "trade_date": "DATE NOT NULL DEFAULT '1970-01-01'",
                        "open": "FLOAT DEFAULT 0",
                        "high": "FLOAT DEFAULT 0",
                        "low": "FLOAT DEFAULT 0",
                        "close": "FLOAT DEFAULT 0",
                        "pre_close": "FLOAT DEFAULT 0",
                        "change1": "FLOAT DEFAULT 0",
                        "pct_chg": "FLOAT DEFAULT 0",
                        "volume": "BIGINT DEFAULT 0",
                        "amount": "DECIMAL(18,2) DEFAULT 0.00",
                        "turnover_rate": "FLOAT DEFAULT 0",
                        "swing": "FLOAT DEFAULT 0",
                        "limit_up": "FLOAT DEFAULT 0",
                        "limit_down": "FLOAT DEFAULT 0",
                        "update_time": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "reserved": "VARCHAR(128) DEFAULT ''"
                    }
                    for col in missing_columns:
                        col_type = col_type_mapping.get(col, "VARCHAR(255) NOT NULL DEFAULT ''")
                        db.add_table_column(table_name, col, col_type)

                # 3.4 过滤数据库共有字段
                final_db_columns = db.get_table_columns(table_name)
                final_db_columns = [col for col in final_db_columns if col not in exclude_columns]
                common_columns = [col for col in api_columns if col in final_db_columns]
                final_df = cleaned_df[common_columns].copy()

                # 3.5 批量入库（重复数据自动更新）
                affected_rows = db.batch_insert_df(
                    df=final_df,
                    table_name=table_name,
                    ignore_duplicate=True
                )
                if affected_rows:
                    total_ingest_rows += affected_rows
                    total_success += 1
                    logger.info(f"{ts_code} 日K数据入库完成，影响行数：{affected_rows}")
                else:
                    logger.warning(f"{ts_code} 入库无影响行数")
                    total_failed += 1
                    failed_codes.append(ts_code)

            except Exception as e:
                logger.error(f"{ts_code} 处理失败：{str(e)}", exc_info=True)
                total_failed += 1
                failed_codes.append(ts_code)
                continue

        # 4. 输出汇总结果
        logger.info("===== 全市场日K数据清洗入库汇总 =====")
        logger.info(f"总处理股票数：{len(stock_codes)}")
        logger.info(f"成功处理数：{total_success} | 失败处理数：{total_failed}")
        logger.info(f"累计入库/更新行数：{total_ingest_rows}")
        if failed_codes:
            logger.warning(f"失败股票代码列表（前20个）：{failed_codes[:20]}")

        return total_ingest_rows


# 全局实例
data_cleaner = DataCleaner()

if __name__ == "__main__":
    """日K数据全市场入库（核心测试）"""
    # 方式1：全量测试
    kline_affected_rows = data_cleaner.clean_and_insert_kline_day(table_name="kline_day")
    logger.info(f"全市场日K数据入库完成，累计入库/更新行数：{kline_affected_rows}")

    # 方式2：单股票测试（调试用，注释掉方式1后打开）
    # def test_single_kline_code():
    #     """单股票日K数据测试"""
    #     test_code = "000002.SZ"
    #     raw_df = data_fetcher.fetch_kline_day(
    #         ts_code=test_code,
    #         start_date="20090215",
    #         end_date="20260215"
    #     )
    #     cleaned_df = data_cleaner._clean_kline_day_data(raw_df)
    #     affected_rows = db.batch_insert_df(
    #         df=cleaned_df,
    #         table_name="kline_day",
    #         ignore_duplicate=True
    #     )
    #     logger.info(f"{test_code} 日K测试入库完成，影响行数：{affected_rows}")
    # test_single_kline_code()