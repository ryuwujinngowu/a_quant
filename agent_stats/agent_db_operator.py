"""
数据库操作封装
和核心引擎解耦，所有表读写统一收口，后续改表结构仅需修改这里
"""
import json
from typing import List, Dict
from datetime import datetime
from utils.db_utils import db
from utils.log_utils import logger
from agent_stats.config import STATS_TABLE_NAME


class AgentStatsDBOperator:
    def __init__(self):
        self.table_name = STATS_TABLE_NAME

    def get_last_processed_date(self, agent_id: str) -> str:
        """获取指定智能体最后一次处理的选股日期"""
        sql = f"""
            SELECT MAX(trade_date) as max_date FROM {self.table_name} 
            WHERE agent_id = %s
        """
        result = db.query_one(sql, (agent_id,))
        if result and result["max_date"]:
            return result["max_date"].strftime("%Y-%m-%d")
        return None

    def get_unclosed_records(self, trade_date: str) -> List[Dict]:
        """获取指定选股日期的待结账记录（隔日字段为空的记录）"""
        sql = f"""
            SELECT agent_id, trade_date, signal_stock_detail 
            FROM {self.table_name} 
            WHERE trade_date = %s AND next_day_avg_close_return IS NULL
        """
        result = db.query_all(sql, (trade_date,))
        # 解析JSON字段
        for row in result:
            row["signal_stock_detail"] = json.loads(row["signal_stock_detail"])
        return result

    def insert_signal_record(self, record: Dict) -> bool:
        """
        幂等插入选股日记录
        重复运行时，已存在的记录会直接覆盖，保证幂等性
        """
        sql = f"""
            INSERT INTO {self.table_name} (
                agent_id, agent_name, trade_date, 
                intraday_avg_return, signal_stock_detail,
                create_time, update_time
            ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                agent_name = VALUES(agent_name),
                intraday_avg_return = VALUES(intraday_avg_return),
                signal_stock_detail = VALUES(signal_stock_detail),
                update_time = NOW()
        """
        try:
            affected_rows = db.execute(sql, (
                record["agent_id"],
                record["agent_name"],
                record["trade_date"],
                record["intraday_avg_return"],
                json.dumps(record["signal_stock_detail"], ensure_ascii=False)
            ))
            logger.debug(f"[{record['agent_id']}][{record['trade_date']}] 选股记录插入成功，影响行数：{affected_rows}")
            return True
        except Exception as e:
            logger.error(f"[{record['agent_id']}][{record['trade_date']}] 选股记录插入失败：{e}", exc_info=True)
            return False

    def update_next_day_stats(self, agent_id: str, trade_date: str, stats: Dict) -> bool:
        """更新选股记录的隔日表现字段（结账操作）"""
        sql = f"""
            UPDATE {self.table_name} SET
                next_day_avg_open_premium = %s,
                next_day_avg_close_return = %s,
                next_day_avg_red_minute = %s,
                next_day_avg_profit_minute = %s,
                next_day_avg_intraday_profit = %s,
                next_day_avg_max_premium = %s,
                next_day_avg_max_drawdown = %s,
                next_day_stock_detail = %s,
                update_time = NOW()
            WHERE agent_id = %s AND trade_date = %s
        """
        try:
            affected_rows = db.execute(sql, (
                stats["next_day_avg_open_premium"],
                stats["next_day_avg_close_return"],
                stats["next_day_avg_red_minute"],
                stats["next_day_avg_profit_minute"],
                stats["next_day_avg_intraday_profit"],
                stats["next_day_avg_max_premium"],
                stats["next_day_avg_max_drawdown"],
                json.dumps(stats["next_day_stock_detail"], ensure_ascii=False),
                agent_id,
                trade_date
            ))
            logger.debug(f"[{agent_id}][{trade_date}] 隔日表现更新成功，影响行数：{affected_rows}")
            return True
        except Exception as e:
            logger.error(f"[{agent_id}][{trade_date}] 隔日表现更新失败：{e}", exc_info=True)
            return False

    def check_date_data_exists(self, trade_date: str) -> bool:
        """检查指定交易日的日线数据是否已入库（前置校验用）
        kline_day.trade_date 存储格式为 YYYYMMDD，需做格式转换"""
        date_fmt = trade_date.replace("-", "")
        sql = "SELECT COUNT(1) as cnt FROM kline_day WHERE trade_date = %s LIMIT 1"
        result = db.query_one(sql, (date_fmt,))
        return result and result["cnt"] > 0