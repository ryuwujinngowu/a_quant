import time
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from data.data_fetcher import data_fetcher
from utils.common_tools import calc_15_years_date_range
from utils.common_tools import auto_add_missing_table_columns
from utils.db_utils import db
from utils.log_utils import logger


class DataCleaner:
    """数据清洗+入库核心类（优化版：精简冗余、提升效率、保留核心契约）"""

    # ===================== 通用工具方法（提取重复逻辑） =====================
    def _get_db_columns(self, table_name: str, exclude_columns: List[str] = None) -> List[str]:
        """通用方法：获取数据库表字段（过滤排除字段）"""
        exclude_columns = exclude_columns or ["id", "created_at", "updated_at"]
        db_cols = db.get_table_columns(table_name)
        return [col for col in db_cols if col not in exclude_columns]

    def _align_df_with_db(self, df: pd.DataFrame, table_name: str, exclude_columns: List[str] = None) -> pd.DataFrame:
        """通用方法：过滤DataFrame字段为数据库表共有字段"""
        db_cols = self._get_db_columns(table_name, exclude_columns)
        common_cols = [col for col in df.columns if col in db_cols]
        return df[common_cols].copy()

    # ===================== 核心清洗方法 =====================
    def _clean_special_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """通用格式清洗（保留核心逻辑，删除冗余注释/校验）"""
        df_cleaned = df.copy()
        # 批量重命名关键字段（可扩展）
        reserved_field_mapping = {"change": "change1", "vol": "volume"}
        df_cleaned.rename(columns=reserved_field_mapping, inplace=True)

        # 日期字段格式统一（YYYYMMDD → YYYY-MM-DD）
        date_fields = ["list_date", "delist_date", "end_date", "start_date"]
        for field in date_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = pd.to_datetime(
                    df_cleaned[field], format="%Y%m%d", errors="coerce"
                ).dt.strftime("%Y-%m-%d")

        # 核心NOT NULL字段非空兜底（上游已做基础校验，仅保留兜底）
        not_null_fields = ["ts_code", "symbol", "name", "exchange", "list_date"]
        for field in not_null_fields:
            if field not in df_cleaned.columns:
                continue
            dtype = df_cleaned[field].dtype
            if pd.api.types.is_object_dtype(dtype):
                df_cleaned[field] = df_cleaned[field].fillna("UNKNOWN")
            elif "date" in field:
                df_cleaned[field] = df_cleaned[field].fillna("1970-01-01")
            elif pd.api.types.is_numeric_dtype(dtype):
                df_cleaned[field] = df_cleaned[field].fillna(0)

        # 其他字段空值填充（统一逻辑，提升效率）
        other_cols = [col for col in df_cleaned.columns if col not in not_null_fields]
        for col in other_cols:
            dtype = df_cleaned[col].dtype
            if pd.api.types.is_object_dtype(dtype):
                df_cleaned[col] = df_cleaned[col].fillna("")
            elif pd.api.types.is_numeric_dtype(dtype):
                df_cleaned[col] = df_cleaned[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                df_cleaned[col] = df_cleaned[col].fillna(None)

        return df_cleaned

    # ===================== 业务清洗入库方法 =====================
    def clean_and_insert_stockcompany(self, table_name: str = "stock_company") -> Optional[int]:
        """上市公司工商信息全字段自动化入库（适配tushare stock_company接口）"""
        logger.info("===== 开始上市公司工商信息清洗入库 =====")

        # 1. 获取原始数据（上游已做空值校验，仅判断empty）
        raw_df = data_fetcher.get_stock_company()
        if raw_df.empty:
            logger.warning("stock_company原始数据为空，跳过入库")
            return 0

        # 2. 通用清洗
        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            logger.warning("stock_company清洗后数据为空，跳过入库")
            return 0

        # 3. 自动新增缺失字段（复用自身通用方法）
        db_cols = self._get_db_columns(table_name)
        missing_cols = [col for col in cleaned_df.columns if col not in db_cols]
        if missing_cols:
            logger.info(f"表{table_name}缺失字段：{missing_cols}，开始自动新增")
            # stock_company专属字段类型映射
            col_type_map = {
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
            for col in missing_cols:
                col_type = col_type_map.get(col, "VARCHAR(255)")
                db.add_table_column(table_name, col, col_type, comment=f"{col}（stock_company接口字段）")

        # 4. 对齐数据库字段并入库
        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            if affected_rows is None:
                logger.error(f"表{table_name}入库失败")
                return 0

            logger.info(f"✅ 表{table_name}入库完成，影响行数：{affected_rows}，字段数：{len(final_df.columns)}")
            # 精简日志示例（保留核心字段）
            if not final_df.empty:
                sample_cols = ["ts_code", "com_name", "province"]
                sample_cols = [col for col in sample_cols if col in final_df.columns]
                logger.info(f"入库示例：\n{final_df[sample_cols].head(3)}")

            return affected_rows
        except Exception as e:
            logger.error(f"表{table_name}入库异常：{str(e)}", exc_info=True)
            return 0

    def clean_and_insert_stockbase(self, table_name: str = "stock_basic") -> Optional[int]:
        """股票基础数据全字段清洗入库（适配tushare stock_basic接口）"""
        logger.info(f"===== 开始股票基础数据清洗入库（目标表：{table_name}） =====")

        # 1. 获取原始数据
        raw_df = data_fetcher.get_stockbase(list_status="L")
        if raw_df.empty:
            logger.warning("stock_basic原始数据为空，跳过入库")
            return 0

        # 2. 通用清洗
        cleaned_df = self._clean_special_fields(raw_df)
        if cleaned_df.empty:
            logger.warning("stock_basic清洗后数据为空，跳过入库")
            return 0

        # 3. 自动新增缺失字段（复用自身通用方法，删除冗余逻辑）
        db_cols = self._get_db_columns(table_name)
        missing_cols = [col for col in cleaned_df.columns if col not in db_cols]
        if missing_cols:
            logger.info(f"表{table_name}缺失字段：{missing_cols}，开始自动新增")
            add_success = auto_add_missing_table_columns(
                table_name=table_name,
                missing_columns=missing_cols
            )
            if not add_success:
                logger.warning(f"表{table_name}部分字段新增失败，继续执行")

        # 4. 对齐数据库字段并入库
        final_df = self._align_df_with_db(cleaned_df, table_name)
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            if affected_rows is None:
                logger.error(f"表{table_name}入库失败")
                return 0

            logger.info(f"✅ 表{table_name}入库完成，影响行数：{affected_rows}，字段数：{len(final_df.columns)}")
            return affected_rows
        except Exception as e:
            logger.error(f"表{table_name}入库异常：{str(e)}", exc_info=True)
            return 0

    def _clean_kline_day_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """日K数据专属清洗（保留核心逻辑，删除上游已做的校验）"""
        if raw_df.empty:
            logger.warning("原始日K数据为空，跳过清洗")
            return pd.DataFrame()

        df_cleaned = raw_df.copy()

        # 1. 字段映射（接口→数据库）
        field_mapping = {"vol": "volume", "change": "change1"}
        df_cleaned.rename(columns=field_mapping, inplace=True)

        # 2. 核心字段列表（匹配数据库表）
        core_fields = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change1", "pct_chg", "volume", "amount",
            "turnover_rate", "swing", "limit_up", "limit_down", "update_time", "reserved"
        ]

        # 3. 填充数据库新增字段默认值
        default_vals = {
            "turnover_rate": "",
            "swing": "",
            "limit_up": "",
            "limit_down": "",
            "update_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reserved": ""
        }
        for field, val in default_vals.items():
            if field not in df_cleaned.columns:
                df_cleaned[field] = val

        # 4. 日期格式转换（上游已做ts_code校验，删除重复校验）
        df_cleaned["trade_date"] = pd.to_datetime(
            df_cleaned["trade_date"], format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        df_cleaned["trade_date"] = df_cleaned["trade_date"].fillna("1970-01-01")

        # 5. 数值字段类型转换+空值填充（批量处理，提升效率）
        numeric_fields = {
            "open": float, "high": float, "low": float, "close": float,
            "pre_close": float, "change1": float, "pct_chg": float,
            "volume": "int64", "amount": float,
            "turnover_rate": float, "swing": float, "limit_up": float, "limit_down": float
        }
        for field, dtype in numeric_fields.items():
            if field in df_cleaned.columns:
                df_cleaned[field] = pd.to_numeric(df_cleaned[field], errors="coerce").fillna(0)
                df_cleaned[field] = df_cleaned[field].astype(dtype)

        # 6. 字符串字段填充+去重
        df_cleaned["reserved"] = df_cleaned["reserved"].fillna("").astype(str)
        df_cleaned = df_cleaned.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

        # 7. 保留核心字段
        df_cleaned = df_cleaned[[col for col in core_fields if col in df_cleaned.columns]]

        logger.info(f"日K数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_cleaned)}行")
        return df_cleaned

    def clean_and_insert_kline_day(self, table_name: str = "kline_day") -> Optional[int]:
        """全市场A股近15年日K数据清洗入库（优化：减少数据库查询次数）"""
        logger.info("===== 开始全市场A股日K数据清洗入库 =====")

        # 1. 前置准备（一次性获取，避免循环内重复查询）
        stock_codes = db.get_all_a_stock_codes()
        if not stock_codes:
            logger.error("无有效股票代码，终止入库")
            return 0

        start_date, end_date = calc_15_years_date_range()
        # 提前获取数据库字段（循环内复用）
        db_cols = self._get_db_columns(table_name)
        # 提前定义kline_day字段类型映射
        kline_col_map = {
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

        # 2. 初始化统计指标
        total_success = 0
        total_failed = 0
        total_ingest_rows = 0
        failed_codes = []

        # 3. 循环处理股票（优化：减少重复逻辑）
        for idx, ts_code in enumerate(stock_codes):
            logger.info(f"处理进度：{idx + 1}/{len(stock_codes)} | 股票代码：{ts_code}")
            try:
                # 获取原始数据
                raw_df = data_fetcher.fetch_kline_day(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if raw_df.empty:
                    logger.warning(f"{ts_code} 无日K数据，跳过")
                    total_failed += 1
                    failed_codes.append(ts_code)
                    continue

                # 清洗数据
                cleaned_df = self._clean_kline_day_data(raw_df)
                if cleaned_df.empty:
                    logger.warning(f"{ts_code} 清洗后无有效数据，跳过")
                    total_failed += 1
                    failed_codes.append(ts_code)
                    continue

                # 自动新增缺失字段（复用通用方法）
                missing_cols = [col for col in cleaned_df.columns if col not in db_cols]
                if missing_cols:
                    logger.info(f"表{table_name}缺失字段：{missing_cols}，开始自动新增")
                    auto_add_missing_table_columns(
                        table_name=table_name,
                        missing_columns=missing_cols,
                        col_type_mapping=kline_col_map
                    )
                    # 刷新数据库字段（仅当有新增时）
                    db_cols = self._get_db_columns(table_name)

                # 对齐数据库字段
                final_df = self._align_df_with_db(cleaned_df, table_name)

                # 批量入库
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

        # 4. 汇总结果（精简日志）
        logger.info("===== 全市场日K数据入库汇总 =====")
        logger.info(f"总处理股票数：{len(stock_codes)} | 成功：{total_success} | 失败：{total_failed}")
        logger.info(f"累计入库/更新行数：{total_ingest_rows}")
        if failed_codes:
            logger.warning(f"失败股票代码（前20）：{failed_codes}")

        return total_ingest_rows

    def clean_and_insert_index_daily(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            table_name: str = "index_daily"
    ) -> Optional[int]:
        """指数日线数据清洗入库（精简冗余逻辑，保留核心）"""
        logger.info(f"===== 开始指数日线数据清洗入库（目标表：{table_name}） =====")

        # 1. 获取原始数据（上游已做参数校验）
        raw_df = data_fetcher.fetch_index_daily(
            ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            logger.debug("指数日线原始数据为空，跳过入库")
            return 0
        # 2. 通用清洗
        clean_df = self._clean_special_fields(raw_df)
        if clean_df.empty:
            logger.warning("指数日线数据清洗后为空，跳过入库")
            return 0

        # 3. 对齐数据库字段并入库
        final_df = self._align_df_with_db(clean_df, table_name)
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            logger.info(f"✅ 指数日线数据入库完成，影响行数：{affected_rows}，字段数：{len(final_df.columns)}")
            return affected_rows
        except Exception as e:
            logger.error(f"指数日线数据入库失败：{str(e)}", exc_info=True)
            return None

    def _clean_kline_min_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """分钟线数据清洗（精简逻辑，提升效率）"""
        if raw_df.empty:
            return pd.DataFrame()

        df_clean = raw_df.copy()
        # 字段映射（对齐日线命名）
        if "vol" in df_clean.columns:
            df_clean.rename(columns={"vol": "volume"}, inplace=True)

        # 批量替换异常值
        df_clean = df_clean.replace([np.nan, np.inf, -np.inf], 0)

        # 字段类型转换（批量处理）
        type_mapping = {"open": float, "close": float, "high": float, "low": float, "volume": int, "amount": float}
        for col, dtype in type_mapping.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(dtype, errors="ignore")

        # 时间字段格式化
        if "trade_time" in df_clean.columns:
            df_clean["trade_time"] = pd.to_datetime(df_clean["trade_time"], errors="coerce")
            df_clean.dropna(subset=["trade_time"], inplace=True)
            df_clean["trade_date"] = df_clean["trade_time"].dt.date

        # 去重
        df_clean.drop_duplicates(subset=["ts_code", "trade_time"], keep="last", inplace=True)

        logger.debug(f"分钟线数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_clean)}行")
        return df_clean

    def clean_and_insert_kline_min(self, raw_df: pd.DataFrame, table_name: str = "kline_min") -> Optional[int]:
        """分钟线数据清洗入库（精简冗余逻辑）"""
        logger.debug(f"分钟线数据入库")
        if raw_df.empty:
            return 0

        clean_df = self._clean_kline_min_data(raw_df)
        if clean_df.empty:
            return 0

        # 对齐数据库字段
        final_df = self._align_df_with_db(clean_df, table_name)

        # 批量入库
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            logger.debug(f"分钟线数据入库完成，影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"分钟线数据入库失败：{str(e)}", exc_info=True)
            return None

    def get_kline_min_by_stock_date(self, ts_code: str, trade_date: str, table_name: str = "kline_min") -> pd.DataFrame:
        """
        获取单只股票单日分钟线数据（核心修复：查库→拉接口→入库→再查库）
        :param ts_code: 股票代码（如000001.SZ）
        :param trade_date: 交易日（格式：YYYY-MM-DD）
        :param table_name: 数据库表名
        :return: 分钟线DataFrame（空则返回空DF）
        """
        # 第一步：参数校验
        if not ts_code or not trade_date:
            logger.error("股票代码/交易日为空，返回空数据")
            return pd.DataFrame()

        # 第二步：查询数据库（核心修复：trade_date用字符串匹配）
        sql = """
            SELECT ts_code, trade_time, trade_date, open, close, high, low, volume, amount
            FROM {table}
            WHERE ts_code = %s AND trade_date = %s
            ORDER BY trade_time ASC
        """.format(table=table_name)

        try:
            df = db.query(sql, params=(ts_code, trade_date), return_df=True)
            if not df.empty:
                df["trade_time"] = pd.to_datetime(df["trade_time"])
                logger.debug(f"[{ts_code}-{trade_date}] 从数据库获取分钟线，行数：{len(df)}")
                return df
        except Exception as e:
            logger.error(f"[{ts_code}-{trade_date}] 查库失败：{str(e)}")

        # 第三步：数据库无数据，调用接口拉取
        logger.debug(f"[{ts_code}-{trade_date}] 数据库无分钟线，调用接口拉取")
        time.sleep(0.5)
        raw_df = data_fetcher.fetch_stk_mins(
            ts_code=ts_code,
            freq="1min",
            start_date=f"{trade_date} 09:25:00",
            end_date=f"{trade_date} 15:00:00"
        )
        if raw_df.empty:
            logger.warning(f"[{ts_code}-{trade_date}] 接口拉取失败，返回空数据")
            return pd.DataFrame()

        # 第四步：清洗+入库
        self.clean_and_insert_kline_min(raw_df, table_name)

        # 第五步：再次查询数据库（确保返回最新入库数据）
        try:
            df = db.query(sql, params=(ts_code, trade_date), return_df=True)
            if not df.empty:
                df["trade_time"] = pd.to_datetime(df["trade_time"])
            logger.debug(f"[{ts_code}-{trade_date}] 入库后查询分钟线，行数：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"[{ts_code}-{trade_date}] 入库后查库失败：{str(e)}")
        return pd.DataFrame()


    def truncate_kline_min_table(self, table_name: str = "kline_min"):
        """清空分钟线表（精简逻辑）"""
        try:
            db.execute(f"TRUNCATE TABLE {table_name}")
            logger.info(f"分钟线表{table_name}清空完成")
        except Exception as e:
            logger.error(f"分钟线表清空失败：{str(e)}", exc_info=True)

    def _clean_trade_cal_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """交易日历数据清洗（精简逻辑）"""
        if raw_df.empty:
            return pd.DataFrame()

        df_clean = raw_df.copy()
        # 日期格式化
        date_fields = ["cal_date", "pretrade_date"]
        for field in date_fields:
            if field in df_clean.columns:
                df_clean[field] = pd.to_datetime(df_clean[field], errors="coerce").dt.date
                df_clean.dropna(subset=[field], inplace=True)

        # is_open类型转换
        if "is_open" in df_clean.columns:
            df_clean["is_open"] = pd.to_numeric(df_clean["is_open"], errors="coerce").fillna(0).astype(int)

        # 去重
        df_clean.drop_duplicates(subset=["exchange", "cal_date"], keep="last", inplace=True)

        logger.debug(f"交易日历数据清洗完成：原始{len(raw_df)}行 → 清洗后{len(df_clean)}行")
        return df_clean

    def clean_and_insert_trade_cal(self, raw_df: pd.DataFrame, table_name: str = "trade_cal") -> Optional[int]:
        """交易日历数据清洗入库（精简冗余逻辑）"""
        if raw_df.empty:
            return 0

        clean_df = self._clean_trade_cal_data(raw_df)
        if clean_df.empty:
            return 0

        # 对齐数据库字段
        final_df = self._align_df_with_db(clean_df, table_name, exclude_columns=["created_at"])

        # 批量入库
        try:
            affected_rows = db.batch_insert_df(
                df=final_df,
                table_name=table_name,
                ignore_duplicate=True
            )
            logger.debug(f"交易日历数据入库完成，影响行数：{affected_rows}")
            return affected_rows
        except Exception as e:
            logger.error(f"交易日历数据入库失败：{str(e)}", exc_info=True)
            return None

    def get_trade_dates(self, start_date: str, end_date: str, table_name: str = "trade_cal") -> List[str]:
        """获取指定时间段内的交易日列表（精简逻辑）"""
        sql = """
            SELECT cal_date
            FROM {table}
            WHERE cal_date BETWEEN %s AND %s AND is_open = 1
            ORDER BY cal_date ASC
        """.format(table=table_name)

        df = db.query(sql, params=(start_date, end_date), return_df=True)
        if df.empty:
            logger.warning(f"未查询到{start_date}至{end_date}的交易日数据")
            return []

        return df["cal_date"].astype(str).tolist()

    def get_pre_trade_date(self, current_date: str, table_name: str = "trade_cal") -> Optional[str]:
        """获取指定日期的上一个交易日（精简逻辑）"""
        sql = """
            SELECT pretrade_date
            FROM {table}
            WHERE cal_date = %s
        """.format(table=table_name)

        df = db.query(sql, params=(current_date,), return_df=True)
        if df.empty or pd.isna(df["pretrade_date"].iloc[0]):
            return None

        return df["pretrade_date"].iloc[0].strftime("%Y-%m-%d")

    def truncate_trade_cal_table(self, table_name: str = "trade_cal"):
        """清空交易日历表（精简逻辑）"""
        try:
            db.execute(f"TRUNCATE TABLE {table_name}")
            logger.info(f"交易日历表{table_name}清空完成")
        except Exception as e:
            logger.error(f"交易日历表清空失败：{str(e)}", exc_info=True)


# 全局实例（保持不变，确保下游调用）
data_cleaner = DataCleaner()

if __name__ == "__main__":

    # """基础信息表入库/更新"""
    # affected_rows = data_cleaner.clean_and_insert_stockbase(table_name="stock_basic")
    """stock_company表入库"""
    # affected_rows = data_cleaner.clean_and_insert_stockcompany(table_name="stock_company")
    # logger.info(f"stock_company入库测试完成：入库/更新行数 {affected_rows}")

    # """日K数据全市场入库（核心测试）"""
    # # # 方式1：全量测试
    # # kline_affected_rows = data_cleaner.clean_and_insert_kline_day(table_name="kline_day")
    # # logger.info(f"全市场日K数据入库完成，累计入库/更新行数：{kline_affected_rows}")

    """测试指数日线数据清洗入库"""
    # try:
    # #     # 测试1：单指数单日数据入库
    # #     affected = data_cleaner.clean_and_insert_index_daily(
    # #         ts_code="000001.SH",
    # #         trade_date="20241231"
    # #     )
    # #     logger.info(f"单指数单日入库影响行数：{affected}")
    #
    #     # 测试2：多指数日期范围数据入库
    #     affected_multi = data_cleaner.clean_and_insert_index_daily(
    #         ts_code="399107.SZ",
    #         start_date="20260211",
    #         end_date="20260213"
    #     )
    #     logger.info(f"多指数日期范围入库影响行数：{affected_multi}")
    #
    #     logger.info("===== 指数日线数据清洗入库测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"指数日线数据清洗入库测试失败：{str(e)} ❌")
    #
    # # ====================== 新增：get_kline_min_by_stock_date 测试用例 ======================
    # 3. 测试场景2：正常参数调用（查库→接口→入库流程）
    # print("\n===== 测试场景2：正常参数调用 =====")
    # test_ts_code = "301550.SZ"
    # test_trade_date = "2026-01-16"
    # result_df =  data_cleaner.get_kline_min_by_stock_date(
    #     ts_code=test_ts_code,
    #     trade_date=test_trade_date,
    #     table_name="kline_min"
    # )
    # print(result_df)
    # """测试分钟线数据（单股票+批量）清洗入库"""
    # try:
    #     # 初始化cleaner实例
    #     data_cleaner = DataCleaner()
    #     test_trade_date = "2026-02-19"  # 测试交易日（可根据实际调整）
    #
    #     logger.info("\n===== 测试2：批量股票分钟线拉取+入库 =====")
    #     test_stock_list = ["000001.SZ", "000002.SZ", "002208.SZ"]  # 测试候选池
    #     batch_result = data_cleaner.batch_get_kline_min_by_date(
    #         stock_codes=test_stock_list,
    #         trade_date=test_trade_date,
    #     )
    #     # 输出批量测试结果
    #     valid_count = len([code for code, df in batch_result.items() if not df.empty])
    #     logger.info(f"批量测试 - 传入股票数：{len(test_stock_list)}")
    #     logger.info(f"批量测试 - 有效数据股票数：{valid_count}")
    #     for ts_code, df in batch_result.items():
    #         logger.info(f"  {ts_code}：拉取数据行数 = {len(df)}")
    #
    #     logger.info("\n===== 分钟线数据（单股票+批量）测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"分钟线数据测试失败：{str(e)} ❌")
