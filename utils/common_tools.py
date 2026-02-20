import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
from typing import List, Dict, Optional
from utils.db_utils import db  # 确保db工具已导入
from utils.log_utils import logger

# 增量更新配置（可统一维护）
UPDATE_RECORD_FILE = Path(__file__).parent.parent / "data" / "update_record.json"
DEFAULT_FIRST_UPDATE_DATE = "2024-01-01"


def init_update_record():
    """初始化更新记录文件"""
    init_data = {
        "last_update_date": DEFAULT_FIRST_UPDATE_DATE,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_tables": []
    }
    with open(UPDATE_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(init_data, f, ensure_ascii=False, indent=4)
    return init_data


def read_last_update_record():
    """读取上次更新记录"""
    if not os.path.exists(UPDATE_RECORD_FILE):
        return init_update_record()
    try:
        with open(UPDATE_RECORD_FILE, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["last_update_date"] = record.get("last_update_date", DEFAULT_FIRST_UPDATE_DATE)
        return record
    except Exception as e:
        logger.error(f"读取更新记录失败，初始化新记录：{e}")
        return init_update_record()


def write_update_record(update_date: str, updated_tables: list):
    """写入本次更新记录"""
    record_data = {
        "last_update_date": update_date,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_tables": updated_tables
    }
    try:
        with open(UPDATE_RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=4)
        logger.info(f"更新记录已保存：本次更新至 {update_date}")
    except Exception as e:
        logger.error(f"写入更新记录失败：{e}")


def calc_incremental_date_range(last_update_date: str, end_date: str = None) -> list:
    """计算增量更新日期列表（YYYYMMDD）"""
    last_dt = datetime.strptime(last_update_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    if last_dt >= end_dt:
        logger.info("无增量日期，无需更新")
        return []

    date_list = []
    current_dt = last_dt + timedelta(days=1)
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%Y%m%d"))
        current_dt += timedelta(days=1)

    logger.info(f"增量更新日期：共{len(date_list)}天 → {date_list}")
    return date_list


def calc_15_years_date_range() -> Tuple[str, str]:
    """
    获取倒推15年内的K线数据
    返回：(start_date, end_date) 均为YYYYMMDD格式字符串
    """
    # 结束日期：当前日期（YYYYMMDD）
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=15*365)).strftime("%Y%m%d")
    logger.info(f"日K数据获取时间范围：{start_date} 至 {end_date}")
    return start_date, end_date

def escape_mysql_reserved_words(field_name: str) -> str:
    """
    转义MySQL保留字字段（给保留字加反引号）
    用于避免字段名与MySQL关键字冲突
    """
    # 常见MySQL保留字（金融数据场景高频）
    reserved_words = {"change", "desc", "order", "select", "insert", "update", "volume"}
    if field_name.lower() in reserved_words:
        return f"`{field_name}`"
    return field_name




def auto_add_missing_table_columns(
    table_name: str,
    missing_columns: List[str],
    col_type_mapping: Dict[str, str] = None
) -> bool:
    """
    通用方法：自动为数据库表新增缺失字段（带默认值，避免插入NULL/NaN报错）
    迁移说明：从DataCleaner类迁移为通用函数，核心逻辑完全不变
    """
    # 默认字段类型映射（金融数据标准化规则）
    default_col_type_mapping = {
        # 日期类字段
        "list_date": "DATE NOT NULL DEFAULT '1970-01-01'",
        "delist_date": "DATE DEFAULT NULL",
        # 数值类字段
        "total_share": "BIGINT DEFAULT 0",
        "float_share": "BIGINT DEFAULT 0",
        "free_share": "BIGINT DEFAULT 0",
        "total_mv": "DECIMAL(20,2) DEFAULT 0.00",
        "circ_mv": "DECIMAL(20,2) DEFAULT 0.00",
        # 核心字符串字段
        "exchange": "VARCHAR(8) NOT NULL DEFAULT 'UNKNOWN'",
        "ts_code": "VARCHAR(9) NOT NULL DEFAULT 'UNKNOWN'",
        "symbol": "VARCHAR(6) NOT NULL DEFAULT 'UNKNOWN'",
        "name": "VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN'",
        # 兜底类型
        "default": "VARCHAR(255) NOT NULL DEFAULT ''"
    }

    # 合并默认映射和自定义映射（自定义优先级更高）
    final_col_map = default_col_type_mapping.copy()
    if col_type_mapping:
        final_col_map.update(col_type_mapping)

    success = True
    for col in missing_columns:
        try:
            col_type = final_col_map.get(col, final_col_map["default"])
            db.add_table_column(table_name, col, col_type)
            logger.info(f"表{table_name}新增字段{col}成功（类型：{col_type}）")
        except Exception as e:
            success = False
            logger.error(f"表{table_name}新增字段{col}失败：{e}")

    return success

