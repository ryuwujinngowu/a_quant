import os
import pymysql
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple, Union
from dbutils.pooled_db import PooledDB
# 新增：导入日志器
from utils.log_utils import logger

# 加载配置文件（拼接项目根目录下的config/.env路径）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
load_dotenv(CONFIG_PATH)


class DBConnector:
    """
    阿里云RDS MySQL通用连接工具类（单例模式）
    核心特性：
    1. 连接池优化：避免频繁创建/销毁连接，提升高并发场景性能
    2. 兼容量化场景：支持DataFrame批量插入（适配K线数据同步）
    3. 安全防护：参数化SQL避免注入，异常捕获+事务回滚保证数据一致性
    4. 版本兼容：适配DBUtils 1.x/2.x、MySQL 5.7+/8.0
    """
    # 单例实例，保证全局只有一个连接池，避免资源浪费
    _instance = None

    def __new__(cls):
        """单例模式初始化：确保全局仅创建一个DBConnector实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_pool()  # 初始化连接池（仅第一次创建实例时执行）
            cls._instance.logger = logger
        return cls._instance

    def _init_pool(self):
        """
        初始化数据库连接池（核心方法）
        可扩展点：
        1. 支持多数据源配置（如读写分离，可新增read_host/read_port等配置）
        2. 增加连接池状态监控（如空闲连接数、活跃连接数统计）
        3. 加入连接重试机制（如初始化失败时重试3次）
        """

        try:
            self.pool = PooledDB(
                creator=pymysql,          # 指定数据库驱动（固定为pymysql）
                host=os.getenv("DB_HOST"),# 从配置文件读取数据库地址
                port=int(os.getenv("DB_PORT")), # 数据库端口（转整型）
                user=os.getenv("DB_USER"), # 数据库账号
                password=os.getenv("DB_PASSWORD"), # 数据库密码
                database=os.getenv("DB_NAME"), # 目标数据库名
                charset="utf8mb4",        # 字符集（适配A股特殊符号/中文）
                mincached=int(os.getenv("DB_MIN_CONNS")), # 最小空闲连接数（初始化时创建）
                maxcached=int(os.getenv("DB_MAX_CONNS")), # 最大空闲连接数（超过则关闭）
                maxconnections=50,        # 最大并发连接数（所有线程可获取的连接上限）
                blocking=True,            # 无空闲连接时是否阻塞（True=等待，False=直接报错）
                cursorclass=pymysql.cursors.DictCursor, # 游标类型（返回字典格式，更易用）
                connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT")) # 连接超时时间（秒）
            )
            logger.debug("数据库连接池初始化成功")
        except Exception as e:
            # 初始化失败抛出明确异常，便于排查（如配置错误、网络不通、白名单问题）
            logger.critical(f"数据库连接池初始化失败：{str(e)}")
            raise RuntimeError(f"数据库连接池初始化失败：{str(e)}")

    def get_conn(self) -> Tuple[pymysql.connections.Connection, pymysql.cursors.Cursor]:
        """
        从连接池获取数据库连接和游标
        返回值：
            Tuple[Connection, Cursor]：数据库连接对象 + 游标对象
        可扩展点：
            1. 增加连接有效性校验（如ping()检测连接是否存活，失效则重新获取）
            2. 记录连接获取/归还日志，便于排查连接泄漏问题
        """
        conn = self.pool.connection()  # 从连接池获取连接（非新建）
        cursor = conn.cursor()         # 创建游标（支持字典格式返回）
        logger.debug("成功从连接池获取数据库连接和游标")
        return conn, cursor

    def close(self, conn: pymysql.connections.Connection, cursor: pymysql.cursors.Cursor):
        """
        关闭游标和连接（连接自动归还连接池，非真正关闭）
        参数：
            conn: 数据库连接对象（get_conn返回的conn）
            cursor: 游标对象（get_conn返回的cursor）
        可扩展点：
            1. 增加异常分级处理（如游标关闭失败不影响连接归还）
            2. 记录关闭失败日志，便于定位资源泄漏
        """
        try:
            cursor.close()  # 先关闭游标
            conn.close()    # 归还连接到池（不是关闭物理连接）
            logger.debug("成功关闭游标并归还连接到连接池")
        except Exception as e:
            logger.error(f"关闭连接/游标失败：{str(e)}")

    def query(
        self,
        sql: str,
        params: Optional[Tuple] = None,
        return_df: bool = False
    ) -> Optional[Union[List[Dict], pd.DataFrame]]:
        """
        通用查询方法（支持单条/批量查询，返回字典列表或DataFrame）
        参数：
            sql: 查询SQL语句（必须参数化，避免%s直接拼接）
            params: SQL参数元组（如("600000.SH", "2024-01-01")），默认None
            return_df: 是否返回DataFrame（量化场景常用），默认False（返回字典列表）
        返回值：
            Optional[Union[List[Dict], pd.DataFrame]]：查询结果（失败返回None）
        可扩展点：
            1. 增加分页功能（如limit/offset参数，支持大数据量分页查询）
            2. 支持查询结果缓存（如高频查询的股票基础信息缓存）
            3. 增加查询超时控制（避免慢查询阻塞）
        """
        conn, cursor = None, None
        try:
            conn, cursor = self.get_conn()          # 获取连接和游标
            cursor.execute(sql, params or ())       # 执行查询（参数化避免SQL注入）
            result = cursor.fetchall()              # 获取所有查询结果
            logger.debug(f"SQL查询执行成功，SQL：{sql}，参数：{params}，结果行数：{len(result) if result else 0}")
            if return_df:
                return pd.DataFrame(result)         # 量化场景返回DataFrame更易用
            return result                           # 通用场景返回字典列表
        except Exception as e:
            # 打印详细错误信息（SQL+参数+异常），便于排查
            logger.error(f"查询执行失败：SQL={sql}, params={params}, error={str(e)}")
            return None
        finally:
            self.close(conn, cursor)  # 无论成败，最终关闭连接/游标

    def execute(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> Optional[int]:
        """
        通用执行方法（支持单条INSERT/UPDATE/DELETE，带事务）
        参数：
            sql: 执行SQL语句（如UPDATE/DELETE/单条INSERT）
            params: SQL参数元组，默认None
        返回值：
            Optional[int]：影响的行数（失败返回None）
        可扩展点：
            1. 支持手动事务控制（如新增begin/commit/rollback方法）
            2. 增加操作日志（记录执行的SQL、参数、影响行数、执行人）
            3. 支持批量执行阈值（如行数>1000自动调用batch_execute）
        """
        conn, cursor = None, None
        try:
            conn, cursor = self.get_conn()          # 获取连接和游标
            rows = cursor.execute(sql, params or ())# 执行SQL
            conn.commit()                           # 提交事务（保证数据一致性）
            logger.info(f"SQL执行成功（INSERT/UPDATE/DELETE），SQL：{sql}，参数：{params}，影响行数：{rows}")
            return rows                             # 返回影响行数
        except Exception as e:
            if conn:
                conn.rollback()                     # 执行失败回滚事务
            logger.error(f"执行失败：SQL={sql}, params={params}, error={str(e)}")
            return None
        finally:
            self.close(conn, cursor)  # 无论成败，关闭连接/游标

    # 在DBConnector类中新增以下方法（其他代码保持不变）
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        获取数据库表的所有字段名
        :param table_name: 表名
        :return: 字段名列表
        """
        try:
            sql = f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """
            result = self.query(sql, (os.getenv("DB_NAME", "a_quant"), table_name))
            if result:
                columns = [row["COLUMN_NAME"] for row in result]
                logger.debug(f"获取表{table_name}字段成功：{columns}")
                return columns
            else:
                logger.warning(f"表{table_name}不存在或无字段")
                return []
        except Exception as e:
            logger.error(f"获取表{table_name}字段失败：{str(e)}")
            return []

    def add_table_column(self, table_name: str, col_name: str, col_type: str = "VARCHAR(255)") -> bool:
        """
        自动给数据库表新增字段（仅当字段不存在时）
        :param table_name: 表名
        :param col_name: 新增字段名
        :param col_type: 字段类型（默认VARCHAR(255)，可根据需求调整）
        :return: 是否新增成功
        """
        # 先检查字段是否已存在
        table_columns = self.get_table_columns(table_name)
        if col_name in table_columns:
            logger.info(f"字段{col_name}已存在于表{table_name}，无需新增")
            return True

        try:
            # 构建新增字段SQL（兼容MySQL）
            sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} DEFAULT '' COMMENT '{col_name}（Tushare自动新增）'"
            self.execute(sql)
            logger.info(f"表{table_name}新增字段{col_name}成功，类型：{col_type}")
            return True
        except Exception as e:
            logger.error(f"表{table_name}新增字段{col_name}失败：{str(e)}")
            return False

    def batch_add_table_columns(self, table_name: str, col_names: List[str], col_type: str = "VARCHAR(255)") -> bool:
        """
        批量新增表字段
        :param table_name: 表名
        :param col_names: 字段名列表
        :param col_type: 字段类型
        :return: 是否全部成功
        """
        success = True
        for col in col_names:
            if not self.add_table_column(table_name, col, col_type):
                success = False
        return success
    def batch_execute(
        self,
        sql: str,
        params_list: List[Tuple]
    ) -> Optional[int]:
        """
        批量执行方法（适配海量数据插入/更新，性能优于单条execute）
        参数：
            sql: 批量执行的SQL模板（如INSERT INTO table (col1) VALUES (%s)）
            params_list: 参数列表（如[(v1,), (v2,), ...]）
        返回值：
            Optional[int]：影响的总行数（失败返回None）
        可扩展点：
            1. 增加批量大小分片（如每1000条分一次，避免单次批量过大）
            2. 支持批量执行进度回调（如每完成10%打印进度）
            3. 增加失败重试机制（如部分失败时重试该分片）
        """
        # 空参数列表直接返回，避免无效执行
        if not params_list:
            logger.warning("批量执行参数列表为空，无需执行")
            return 0

        conn, cursor = None, None
        try:
            conn, cursor = self.get_conn()              # 获取连接和游标
            rows = cursor.executemany(sql, params_list) # 批量执行（底层优化，性能更高）
            conn.commit()                               # 提交事务
            logger.info(f"SQL批量执行成功，SQL模板：{sql}，参数列表长度：{len(params_list)}，影响行数：{rows}")
            return rows                                 # 返回影响总行数
        except Exception as e:
            if conn:
                conn.rollback()                         # 失败回滚
            logger.error(f"批量执行失败：SQL={sql}, error={str(e)}")
            return None
        finally:
            self.close(conn, cursor)  # 关闭连接/游标

    # utils/db_utils.py 核心函数修正
    def batch_insert_df(self, df: pd.DataFrame, table_name: str, ignore_duplicate: bool = True) -> Optional[int]:
        """
        DataFrame批量插入数据库（核心优化：完善异常捕捉+支持重复更新）
        :param df: 待插入的DataFrame（字段需与表结构一致）
        :param table_name: 目标表名
        :param ignore_duplicate: True=重复则更新，False=重复则报错
        :return: 影响行数（失败返回None）
        """
        if df.empty:
            self.logger.warning(f"DataFrame为空，表名：{table_name}，跳过插入")
            return 0

        try:
            # 1. 转换DataFrame为可执行的SQL参数
            columns = df.columns.tolist()
            placeholders = ", ".join(["%s"] * len(columns))
            sql_columns = ", ".join(columns)

            # 2. 处理重复数据：ON DUPLICATE KEY UPDATE（更新所有字段）
            if ignore_duplicate:
                update_clause = ", ".join([f"{col}=VALUES({col})" for col in columns])
                sql = f"""
                    INSERT INTO {table_name} ({sql_columns}) 
                    VALUES ({placeholders}) 
                    ON DUPLICATE KEY UPDATE {update_clause}
                """
            else:
                sql = f"INSERT INTO {table_name} ({sql_columns}) VALUES ({placeholders})"

            # 3. 转换DataFrame数据为元组列表
            data = [tuple(row) for row in df.values]

            # 4. 批量执行
            with self.pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(sql, data)
                    affected_rows = cursor.rowcount
                    conn.commit()
                    self.logger.info(f"DataFrame批量插入成功：表名 {table_name}，影响行数 {affected_rows}")
                    return affected_rows

        except Exception as e:
            # 完善的异常捕捉：区分不同错误类型
            error_msg = str(e)
            if "Duplicate entry" in error_msg:
                self.logger.error(f"批量插入失败：表名 {table_name}，重复数据错误 {error_msg}")
            elif "Data too long" in error_msg:
                self.logger.error(f"批量插入失败：表名 {table_name}，字段长度超限 {error_msg}")
            elif "Unknown column" in error_msg:
                self.logger.error(f"批量插入失败：表名 {table_name}，字段不存在 {error_msg}")
            else:
                self.logger.error(f"批量插入失败：表名 {table_name}，未知错误 {error_msg}")
            # 回滚事务
            if 'conn' in locals():
                conn.rollback()
            return None

    def get_all_a_stock_codes(self) -> List[str]:
        """
        从stock_basic表获取全市场A股代码列表（已上市，list_status='L'）
        返回：有效A股代码列表（格式如600000.SH）
        """
        logger.info("===== 从stock_basic表读取全市场A股代码 =====")
        try:
            # 查询SQL（精准匹配A股代码格式）
            sql = "SELECT DISTINCT ts_code FROM stock_basic WHERE list_status = 'L' "
            # 使用query方法返回DataFrame（return_df=True）
            df_codes = self.query(sql, return_df=True)

            if df_codes.empty:
                logger.warning("stock_basic表中无有效A股代码")
                return []

            codes = df_codes["ts_code"].dropna().tolist()
            logger.info(f"成功获取 {len(codes)} 只有效A股代码")
            return codes
        except Exception as e:
            logger.error(f"读取股票代码失败：{str(e)}", exc_info=True)
            return []

    def add_table_column(self, table_name: str, col_name: str, col_type: str = "VARCHAR(255)",
                         comment: str = "") -> bool:
        """
        自动给数据库表新增字段（仅当字段不存在时）
        扩展：支持添加字段注释
        :param table_name: 表名
        :param col_name: 新增字段名
        :param col_type: 字段类型（默认VARCHAR(255)）
        :param comment: 字段注释（可选）
        :return: 是否新增成功
        """
        # 先检查字段是否已存在
        table_columns = self.get_table_columns(table_name)
        if col_name in table_columns:
            logger.info(f"字段{col_name}已存在于表{table_name}，无需新增")
            return True

        try:
            # 构建新增字段SQL（兼容MySQL，支持注释）
            sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
            if comment:
                sql += f" COMMENT '{comment}'"
            else:
                sql += f" DEFAULT '' COMMENT '{col_name}（Tushare自动新增）'"

            self.execute(sql)
            logger.info(f"表{table_name}新增字段{col_name}成功，类型：{col_type}，注释：{comment}")
            return True
        except Exception as e:
            logger.error(f"表{table_name}新增字段{col_name}失败：{str(e)}")
            return False


# 全局单例实例（项目中直接导入该实例即可使用，无需重复创建）
db = DBConnector()

if __name__ == "__main__":
        """全量测试：修正主键重复后的版本"""
        try:
            logger.info("===== 开始执行数据库工具全量测试 =====")
            logger.info("===== 1. 测试连接池初始化 =====")
            logger.info("连接池初始化成功 ✅")

            # ====================== 测试query（查询函数）======================
            logger.info("\n===== 2. 测试query查询函数 =====")
            # 测试1：查询股票基础信息前5条（返回字典列表）
            sql_query_1 = "SELECT ts_code, name, industry FROM stock_basic LIMIT 5"
            result_dict = db.query(sql_query_1)
            logger.info(f"查询结果（字典列表）：\n{result_dict[:2]}...")  # 只打印前2条避免过长

            # 测试2：查询交易日历（返回DataFrame，量化场景常用）
            sql_query_2 = "SELECT trade_date, is_open FROM trade_calendar WHERE trade_date >= %s LIMIT 3"
            params_query_2 = ("2024-01-01",)
            result_df = db.query(sql_query_2, params_query_2, return_df=True)
            logger.info(f"查询结果（DataFrame）：\n{result_df}")

            # ====================== 测试execute（单条执行）======================
            logger.info("\n===== 3. 测试execute单条执行函数（插入+更新+删除） =====")
            # 插入前清理kline_day测试数据
            sql_clean_kline_day_pre = "DELETE FROM kline_day WHERE ts_code = %s AND trade_date = %s"
            db.execute(sql_clean_kline_day_pre, ("600000.SH", "2024-01-02"))

            # 测试1：插入测试数据到kline_day（ts_code长度修正为9位）
            sql_execute_insert = """
                INSERT INTO kline_day (ts_code, trade_date, open, high, low, close, volume, amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
            params_execute_insert = ("600000.SH", "2024-01-02", 8.0, 8.1, 7.9, 8.05, 10000, 80500.0)
            insert_rows = db.execute(sql_execute_insert, params_execute_insert)
            logger.info(f"单条插入影响行数：{insert_rows}")

            # 测试2：更新测试数据（修改涨跌幅）
            sql_execute_update = "UPDATE kline_day SET pct_chg = %s WHERE ts_code = %s AND trade_date = %s"
            params_execute_update = (0.625, "600000.SH", "2024-01-02")
            update_rows = db.execute(sql_execute_update, params_execute_update)
            logger.info(f"单条更新影响行数：{update_rows}")

            # ====================== 测试batch_execute（批量执行）======================
            logger.info("\n===== 4. 测试batch_execute批量执行函数 =====")
            # 插入前清理trade_calendar测试数据（核心修正：避免主键重复）
            sql_clean_batch = "DELETE FROM trade_calendar WHERE exchange = %s AND trade_date >= %s"
            db.execute(sql_clean_batch, ("SH", "2024-01-01"))

            # 测试：批量插入3条交易日历测试数据
            sql_batch = "INSERT INTO trade_calendar (exchange, trade_date, is_open, is_holiday, weekday) VALUES (%s, %s, %s, %s, %s)"
            params_batch_list = [
                ("SH", "2024-01-01", 0, 1, 1),
                ("SH", "2024-01-02", 1, 0, 2),
                ("SH", "2024-01-03", 1, 0, 3)
            ]
            batch_rows = db.batch_execute(sql_batch, params_batch_list)
            logger.info(f"批量插入影响行数：{batch_rows}")

            # ====================== 测试batch_insert_df（DataFrame批量插入）======================
            logger.info("\n===== 5. 测试batch_insert_df DataFrame批量插入 =====")
            # 插入前清理kline_1min测试数据
            sql_clean_kline_1min_pre = "DELETE FROM kline_1min WHERE ts_code = %s AND trade_date = %s"
            db.execute(sql_clean_kline_1min_pre, ("600000.SH", "2024-01-02"))

            # 构造1分钟K线测试DataFrame
            test_df = pd.DataFrame({
                "ts_code": ["600000.SH", "600000.SH"],
                "trade_time": ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
                "trade_date": ["2024-01-02", "2024-01-02"],
                "open": [8.0, 8.05],
                "high": [8.02, 8.08],
                "low": [7.98, 8.03],
                "close": [8.01, 8.06],
                "volume": [5000, 6000],
                "amount": [40050.0, 48360.0]
            })
            df_insert_rows = db.batch_insert_df(test_df, "kline_1min")
            logger.info(f"DataFrame批量插入影响行数：{df_insert_rows}")

            # ====================== 清理测试数据（避免污染真实数据）======================
            logger.info("\n===== 6. 清理测试数据 =====")
            # 删除kline_day测试数据
            sql_clean_kline_day = "DELETE FROM kline_day WHERE ts_code = %s AND trade_date = %s"
            db.execute(sql_clean_kline_day, ("600000.SH", "2024-01-02"))

            # 删除trade_calendar测试数据
            sql_clean_calendar = "DELETE FROM trade_calendar WHERE trade_date >= %s AND exchange = %s"
            db.execute(sql_clean_calendar, ("2024-01-01", "SH"))

            # 删除kline_1min测试数据
            sql_clean_kline_1min = "DELETE FROM kline_1min WHERE ts_code = %s AND trade_date = %s"
            db.execute(sql_clean_kline_1min, ("600000.SH", "2024-01-02"))
            logger.info("测试数据清理完成 ✅")

            logger.info("\n===== 所有测试完成 ✅ =====")

        except Exception as e:
            logger.error(f"测试过程中出现异常：{str(e)} ❌")