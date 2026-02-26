import json
import os
import threading
from datetime import datetime, timedelta
import pandas as pd
import time
from pathlib import Path
import functools
from typing import List, Dict
from typing import Tuple

from utils.db_utils import db  # 确保db工具已导入
from utils.log_utils import logger

# Token 统计文件配置（默认统计到 data 目录，可根据需要调整）
TOKEN_USAGE_FILE = None
TOKEN_LOCK = threading.Lock()


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




def init_token_file(file_path: str, force: bool = False):
    """
    初始化Token统计文件
    :param file_path: Token统计文件的绝对/相对路径
    :param force: 是否强制覆盖现有文件（默认False）
    """
    global TOKEN_USAGE_FILE
    TOKEN_USAGE_FILE = os.path.abspath(file_path)

    if force or not os.path.exists(TOKEN_USAGE_FILE):
        try:
            init_data = {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0
            }
            # 确保文件所在目录存在
            os.makedirs(os.path.dirname(TOKEN_USAGE_FILE), exist_ok=True)

            with open(TOKEN_USAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(init_data, f, ensure_ascii=False, indent=2)
            logger.info(f"初始化Token统计文件成功，路径：{TOKEN_USAGE_FILE}，初始数据：{init_data}")
        except Exception as e:
            logger.error(f"初始化Token统计文件失败：{str(e)}，路径：{TOKEN_USAGE_FILE}")
            raise e


def update_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int, token_stat: bool = True):
    """
    累加Token消耗到文件
    :param prompt_tokens: 本次输入Token数
    :param completion_tokens: 本次输出Token数
    :param total_tokens: 本次总Token数
    :param token_stat: 是否开启Token统计（开关控制）
    """
    if not token_stat:
        logger.warning("Token统计功能已关闭，跳过累加")
        return
    if not TOKEN_USAGE_FILE:
        raise ValueError("Token统计文件未初始化，请先调用init_token_file设置文件路径")
    with TOKEN_LOCK:
        try:
            with open(TOKEN_USAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            old_prompt = data["total_prompt_tokens"]
            old_completion = data["total_completion_tokens"]
            old_total = data["total_tokens"]
            old_count = data["call_count"]

            new_prompt = old_prompt + prompt_tokens
            new_completion = old_completion + completion_tokens
            new_total = old_total + total_tokens
            new_count = old_count + 1

            # 写入新数据
            data.update({
                "total_prompt_tokens": new_prompt,
                "total_completion_tokens": new_completion,
                "total_tokens": new_total,
                "call_count": new_count
            })
            with open(TOKEN_USAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)



        except json.JSONDecodeError as e:
            logger.error(f"Token统计文件格式错误，重新初始化（原文件已备份）：{str(e)}")
            # 备份错误文件
            if os.path.exists(TOKEN_USAGE_FILE):
                bak_path = f"{TOKEN_USAGE_FILE}.bak"
                os.rename(TOKEN_USAGE_FILE, bak_path)
                logger.info(f"错误文件已备份至：{bak_path}")
            # 重新初始化并累加本次数据
            init_token_file(TOKEN_USAGE_FILE, force=True)
            update_token_usage(prompt_tokens, completion_tokens, total_tokens, token_stat)
        except Exception as e:
            logger.error(f"Token累加失败：{str(e)}")
            raise e


def get_token_usage(file_path: str) -> dict:
    """
    获取Token统计文件的累计数据
    :param file_path: Token统计文件路径
    :return: 累计统计数据
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        logger.warning(f"Token统计文件不存在：{file_path}，返回初始数据")
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"读取Token统计文件失败：{str(e)}")
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0
        }

# -------------------------- 重试装饰器实现 --------------------------
def retry_decorator(max_retries:  int = 3, retry_interval: float = 1.0):
    """
    ：支持异常重试 + 空DataFrame重试
    :param max_retries: 最大重试次数
    :param retry_interval: 每次重试的间隔（秒）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 执行原方法
                    result = func(self, *args, **kwargs)

                    # 检查返回值是否为空DataFrame
                    if isinstance(result, pd.DataFrame) and result.empty:
                        retry_count += 1
                        # 达到最大重试次数，记录警告并返回空DataFrame
                        if retry_count >= max_retries:
                            logger.warning(
                                f"【{func.__name__}】空数据重试次数已达上限（{max_retries}次），最终返回空数据 "
                                f"参数：args={args}, kwargs={kwargs}"
                            )
                            return result
                        # 未达最大次数，记录警告并等待后重试
                        logger.warning(
                            f"【{func.__name__}】返回空DataFrame，将进行第{retry_count}次重试（剩余{max_retries - retry_count}次） "
                            f"间隔{retry_interval}秒"
                        )
                        time.sleep(retry_interval)
                        continue  # 触发下一次重试
                    # 数据非空，直接返回
                    return result

                except Exception as e:
                    retry_count += 1
                    # 达到最大重试次数，记录错误并返回空DataFrame
                    if retry_count >= max_retries:
                        logger.error(
                            f"【{func.__name__}】异常重试次数已达上限（{max_retries}次），最终执行失败 "
                            f"参数：args={args}, kwargs={kwargs} 错误：{str(e)}"
                        )
                        return pd.DataFrame()
                    # 未达最大次数，记录警告并等待后重试
                    logger.warning(
                        f"【{func.__name__}】执行异常，将进行第{retry_count}次重试（剩余{max_retries - retry_count}次） "
                        f"间隔{retry_interval}秒 错误：{str(e)}"
                    )
                    time.sleep(retry_interval)
            # 兜底返回空DataFrame
            return pd.DataFrame()
        return wrapper
    return decorator

