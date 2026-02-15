import os
import pandas as pd
import tushare as ts
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# 导入项目日志模块（同步你使用的单例版本）
from utils.log_utils import logger

# ====================== 全局配置 & 初始化 ======================
# 加载.env配置（可选：也可将token放入.env，避免硬编码）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
load_dotenv(CONFIG_PATH)

# Tushare初始化（优先从.env读取token，无则使用硬编码）
TS_TOKEN = os.getenv("TS_TOKEN", "6a3e1b964b1847a66a6e4c5421006605ab279b9b2d4ca33a8aa3e8b3") #我自己的token
try:
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    pro._DataApi__http_url = 'http://106.54.191.157:5000'   # 淘宝买的token必须要有这个代码
    logger.info("Tushare Pro API 初始化成功")
except Exception as e:
    logger.critical(f"Tushare Pro API 初始化失败：{str(e)}")
    raise RuntimeError(f"Tushare初始化失败：{str(e)}")


# ====================== 数据获取核心函数（按业务分类）======================
class DataFetcher:
    """
    数据请求层核心类
    职责：
    1. 统一封装不同数据源的接口调用（当前Tushare，后续可扩展其他数据源）
    2. 按数据类型拆分函数，便于维护和扩展
    3. 基础异常处理+日志记录，便于排查接口调用问题
    4. 仅负责数据获取，不做清洗/入库（清洗入库在data/date_cleaner.py）
    """

    def get_stockbase(
            self,
            exchange: Optional[str] = None,
            list_status: str = "L",
            fields: Optional[List[str]] = None,
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        获取股票基础信息（对应tushare pro的stock_basic接口）
        接口文档：https://tushare.pro/document/2?doc_id=25
        参数说明（对齐Tushare接口，仅封装不修改）：
            exchange: 交易所代码（如SSE=上交所, SZSE=深交所, BSE=北交所），None则返回全市场
            list_status: 上市状态（L=上市，D=退市，P=暂停上市），默认L
            fields: 需要返回的字段列表（None则返回所有字段），如["ts_code", "symbol", "name", "industry"]
            **kwargs: 其他扩展参数（如start_date/end_date，适配接口升级）

        返回值：
            pd.DataFrame: 股票基础数据（失败返回None）
        """
        # 核心修正：默认指定完整字段列表，确保获取delist_date/fullname等字段
        # DEFAULT_FIELDS = [
        #     "ts_code", "symbol", "name", "exchange", "market",
        #     "industry", "list_date", "delist_date", "fullname",
        #     "enname", "spell"  # cnspell对应spell
        # ]
        # if fields is None:
        #     fields = DEFAULT_FIELDS
        # 1. 参数预处理（兼容Tushare接口要求）
        params: Dict[str, Any] = {
            "list_status": list_status,
            "exchange": exchange,
            **kwargs
        }
        # 过滤空参数（避免Tushare接口报错）
        params = {k: v for k, v in params.items() if v is not None}

        # 2. 接口调用+异常处理
        try:
            logger.info(f"开始获取股票基础数据，参数：{params}，字段：{fields}")
            # 调用Tushare stock_basic接口
            df = pro.stock_basic(**params, fields=fields)

            # 3. 基础校验（空数据日志提醒）
            if df.empty:
                logger.warning(f"股票基础数据获取成功，但返回空数据，参数：{params}")
            else:
                logger.info(f"股票基础数据获取成功，数据行数：{len(df)}，字段：{df.columns.tolist()}")

            return df
        except Exception as e:
            logger.error(f"股票基础数据获取失败，参数：{params}，错误信息：{str(e)}")
            return None

    # ====================== 预留扩展函数框架（后续新增数据类型）======================
    def get_trade_calendar(
            self,
            exchange: str = "SSE",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取交易日历数据（预留扩展）
        接口文档：https://tushare.pro/document/2?doc_id=26
        """
        try:
            logger.info(f"开始获取交易日历数据，交易所：{exchange}，时间范围：{start_date}~{end_date}")
            params = {
                "exchange": exchange,
                "start_date": start_date,
                "end_date": end_date
            }
            params = {k: v for k, v in params.items() if v is not None}

            df = pro.trade_cal(**params)
            if df.empty:
                logger.warning(f"交易日历数据返回空，参数：{params}")
            else:
                logger.info(f"交易日历数据获取成功，行数：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"交易日历数据获取失败，参数：{params}，错误：{str(e)}")
            return None

    def get_kline_day(
            self,
            ts_code: str,
            start_date: str,
            end_date: str,
            adj: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        获取日线K线数据（预留扩展）
        接口文档：https://tushare.pro/document/2?doc_id=109
        """
        try:
            logger.info(f"开始获取日线数据，股票代码：{ts_code}，时间范围：{start_date}~{end_date}，复权类型：{adj}")
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adj
            )
            if df.empty:
                logger.warning(f"日线数据返回空，股票代码：{ts_code}，时间范围：{start_date}~{end_date}")
            else:
                logger.info(f"日线数据获取成功，行数：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"日线数据获取失败，股票代码：{ts_code}，错误：{str(e)}")
            return None

    def get_kline_1min(
            self,
            ts_code: str,
            start_date: str,
            end_date: str,
            freq: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """
        获取1分钟K线数据（预留扩展）
        接口文档：https://tushare.pro/document/2?doc_id=108
        """
        try:
            logger.info(f"开始获取{freq}K线数据，股票代码：{ts_code}，时间范围：{start_date}~{end_date}")
            # 注：分钟线需要用bar接口，与日线不同
            df = pro.min_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                freq=freq
            )
            if df.empty:
                logger.warning(f"{freq}K线数据返回空，股票代码：{ts_code}")
            else:
                logger.info(f"{freq}K线数据获取成功，行数：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"{freq}K线数据获取失败，股票代码：{ts_code}，错误：{str(e)}")
            return None

    # 可继续扩展：财务数据、资金流数据、板块数据等函数...


# ====================== 全局实例（项目中直接导入使用）======================
data_fetcher = DataFetcher()

# ====================== 测试代码（验证核心功能）======================
if __name__ == "__main__":
    """测试数据获取层核心函数"""
    try:
        # 1. 测试股票基础数据获取
        logger.info("===== 测试：获取A股上市股票基础数据 =====")
        stock_base_df = data_fetcher.get_stockbase(
            exchange="SSE",  # 仅上交所
            list_status="L",  # 仅上市
            # fields=["ts_code", "symbol", "name", "industry", "list_date"]  # 指定返回字段
        )
        if stock_base_df is not None:
            logger.info(f"股票基础数据预览：\n{stock_base_df.head(5)}")

        # # 2. 测试交易日历获取（预留函数）
        # logger.info("\n===== 测试：获取2024年交易日历 =====")
        # cal_df = data_fetcher.get_trade_calendar(
        #     exchange="SSE",
        #     start_date="20240101",
        #     end_date="20241231"
        # )
        # if cal_df is not None:
        #     logger.info(f"交易日历数据预览：\n{cal_df.head(3)}")
        #
        # # 3. 测试日线数据获取（预留函数）
        # logger.info("\n===== 测试：获取600000.SH 2024年日线数据 =====")
        # kline_day_df = data_fetcher.get_kline_day(
        #     ts_code="600000.SH",
        #     start_date="20240101",
        #     end_date="20240131"
        # )
        # if kline_day_df is not None:
        #     logger.info(f"日线数据预览：\n{kline_day_df.head(3)}")

        logger.info("\n===== 所有数据获取测试完成 ✅ =====")

    except Exception as e:
        logger.error(f"数据获取测试失败，错误信息：{str(e)} ❌")