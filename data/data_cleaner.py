import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from utils.log_utils import logger
from utils.db_utils import db
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
        raw_df = data_fetcher.get_stock_company()  # 需确保date_fetcher.py中实现该方法
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
                # 新增字段时添加注释（便于数据库维护）
                col_comment = f"{col}（stock_company接口字段）"
                # 注意：需确保db.add_table_column方法支持添加注释（若不支持，可临时修改该方法）
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


# 全局实例
data_cleaner = DataCleaner()

if __name__ == "__main__":

    """基础信息表入库/更新"""
    affected_rows = data_cleaner.clean_and_insert_stockbase(table_name="stock_basic")
    """stock_company表入库"""
    affected_rows = data_cleaner.clean_and_insert_stockcompany(table_name="stock_company")
    logger.info(f"stock_company入库测试完成：入库/更新行数 {affected_rows}")
