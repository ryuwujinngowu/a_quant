import os
import pandas as pd
import tushare as ts
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from utils.log_utils import logger

def initialize_tushare():
    # 初始化配置
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
    load_dotenv(CONFIG_PATH)
    TS_TOKEN = os.getenv("TS_TOKEN", "6a3e1b964b1847a66a6e4c5421006605ab279b9b2d4ca33a8aa3e8b3")
    try:
        ts.set_token(TS_TOKEN)
        pro = ts.pro_api()
        pro._DataApi__http_url = 'http://106.54.191.157:5000'   # 淘宝token专属配置，保留
        logger.info("Tushare Pro API 初始化成功")
    except Exception as e:
        logger.critical(f"Tushare Pro API 初始化失败：{str(e)}")
        raise RuntimeError(f"Tushare初始化失败：{str(e)}")
    return pro


class DataFetcher:
    """数据请求层核心类（强制返回接口所有字段+保留中文）"""

    def __init__(self):
        self.pro = initialize_tushare()

    def get_stockbase(
            self,
            exchange: Optional[str] = None,
            list_status: str = "L",
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        获取股票基础信息（强制返回所有字段）
        接口文档：https://tushare.pro/document/2?doc_id=25
        字段说明：官方文档列出的所有字段均显式指定，确保无遗漏
        """
        # 1. 构造参数（过滤空参数，避免接口异常）
        params: Dict[str, Any] = {
            "list_status": list_status,
            "exchange": exchange,
            **kwargs
        }
        # 核心：过滤空值/空字符串参数，避免接口报错
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        # 2. 显式指定Tushare stock_basic所有字段（从官方文档复制，确保全量）
        # 字段列表来自Tushare官方文档v2版本，覆盖所有可返回字段
        ALL_FIELDS = [
            "ts_code",
            "symbol",
            "name",
            "area",
            "industry",
            "cnspell",
            "market",
            "list_date",
            "act_name",
            "act_ent_type",
            "exchange",
            "curr_type",
            "list_status",
            "delist_date",
            "is_hs",
            "enname",
            "fullname"
        ]

        try:
            logger.debug(f"开始获取股票基础数据（全字段），参数：{params}")
            # 核心：显式传入fields参数，指定所有字段（逗号分隔）
            df = self.pro.stock_basic(**params, fields=",".join(ALL_FIELDS))

            if df.empty:
                logger.warning(f"股票基础数据接口返回空数据，参数：{params}")
            else:
                logger.debug(f"股票基础数据获取成功：行数={len(df)}，返回字段数={len(df.columns)}")
                logger.debug(f"返回字段列表：{df.columns.tolist()}")
                # # 验证关键字段是否存在（如exchange、act_name等）
                # key_fields = ["exchange", "act_name", "fullname", "enname"]
                # missing_key_fields = [f for f in key_fields if f not in df.columns]
                # if missing_key_fields:
                #     logger.warning(f"部分关键字段未返回：{missing_key_fields}（权限/接口版本问题）")
            return df
        except Exception as e:
            logger.error(f"股票基础数据获取失败，参数：{params}，错误信息：{str(e)}")
            return None

    def get_stock_company(
            self,
            ts_code: Optional[str] = None,
            exchange: Optional[str] = None,
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        获取上市公司基本信息（stock_company，doc_id=112）
        接口文档：https://tushare.pro/document/2?doc_id=112
        参数说明：
            ts_code: 股票代码（如600000.SH，可选，不传返回所有）
            exchange: 交易所代码（SHSE/SZSE/BSE，可选）
            **kwargs: 其他扩展参数（如limit/offset，接口支持则生效）
        返回字段：包含公司高管、联系方式、办公地址等全量信息
        """
        # 1. 构造参数（过滤空值，避免接口异常）
        params: Dict[str, Any] = {
            "ts_code": ts_code,
            "exchange": exchange,
            **kwargs
        }
        # params = {k: v for k, v in params.items() if v is not None and v != ""}

        # 2. 显式指定stock_company所有字段（从官方文档复制，全量返回）
        # 字段列表来自Tushare官方文档v2版本，覆盖所有可返回字段
        # ALL_COMPANY_FIELDS = [
        #     "ts_code", "exchange", "chairman", "manager", "secretary",
        #     "reg_capital", "setup_date", "province", "city",  "fax", "email", "website", "business_scope",
        #     "main_business", "employees", "main_products", "office",
        #     "enname", "cnspell", "industry", "csrc_industry", "eastmoney_industry"
        # ]

        try:
            logger.info(f"开始获取上市公司基本信息（全字段），参数：{params}")
            # 调用stock_company接口，传入全字段
            df = self.pro.stock_company(**params)
            if df.empty:
                logger.warning(f"上市公司基本信息获取成功，但返回空数据，参数：{params}")
            else:
                logger.info(f"上市公司基本信息获取成功：行数={len(df)}，返回字段数={len(df.columns)}")
                logger.info(f"返回字段列表：{df.columns.tolist()}")
                # 验证核心字段
                core_fields = ["ts_code", "chairman", "manager"]
                missing_core_fields = [f for f in core_fields if f not in df.columns]
                if missing_core_fields:
                    logger.warning(f"部分核心字段未返回：{missing_core_fields}（权限/接口版本问题）")
            return df
        except Exception as e:
            logger.error(f"上市公司基本信息获取失败，参数：{params}，错误信息：{str(e)}")
            return None

    def fetch_kline_day(self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取A股日K线历史数据，适配kline_day表的字段要求


        输出参数：
            返回DataFrame，包含以下字段：
            ts_code: str - 股票代码
            trade_date: str - 交易日期
            open: float - 开盘价
            high: float - 最高价
            low: float - 最低价
            close: float - 收盘价
            pre_close: float - 昨收价【除权价】
            change: float - 涨跌额
            pct_chg: float - 涨跌幅（基于除权昨收计算）
            vol: float - 成交量（手）
            amount: float - 成交额（千元）
        """
        # 构造请求参数（适配Tushare接口参数）
        params = {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date
        }
        # 调用Tushare日线接口
        df = self.pro.daily(**params)
        logger.debug(f"日线数据获取，参数：\n{params}")
        logger.debug(f"日线数据获取行数：{len(df)}")
        return df

    def fetch_kline_day_qfq(self,
                        ts_code: Optional[str] = None,
                        trade_date: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        adj: Optional[str] = None
                        ) -> pd.DataFrame:

        # 构造请求参数（适配Tushare接口参数）
        params = {
            "ts_code": ts_code,

            "start_date": start_date,
            "end_date": end_date,
            "adj" : "qfq"
        }
        # 调用Tushare日线接口
        df = ts.pro_bar(**params)
        logger.debug(f"日线(前复权)数据获取，参数：\n{params}")
        logger.debug(f"日线(前复权)数据获取行数：{len(df)}")
        return df

    # ===================== 新增：指数日线数据获取函数 =====================
    def fetch_index_daily(
            self,
            ts_code: Optional[str] = None,
            trade_date: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取指数历史日线数据，适配index_daily表的字段要求
        接口文档：https://tushare.pro/document/2?doc_id=403

        输出参数：
            返回DataFrame，包含以下字段：
            ts_code: str - 指数代码（如000001.SH）
            trade_date: str - 交易日期
            open: float - 开盘点位
            high: float - 最高点位
            low: float - 最低点位
            close: float - 收盘点位
            pre_close: float - 昨日收盘点
            change: float - 涨跌点
            pct_chg: float - 涨跌幅(%)
            vol: float - 成交量(手)
            amount: float - 成交额(万元)
        """
        # 构造请求参数（适配Tushare接口参数）
        params = {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
        }
        # 过滤空值/空字符串参数，避免接口报错（和get_stockbase保持一致）
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        try:
            # 调用Tushare指数日线接口
            df = self.pro.index_daily(**params)
            logger.debug(f"指数日线数据获取，参数：\n{params}")
            logger.debug(f"指数日线数据获取行数：{len(df)}")

            if df.empty:
                logger.debug(f"指数日线数据接口返回空数据，参数：{params}")

            return df
        except Exception as e:
            logger.error(f"指数日线数据获取失败，参数：{params}，错误信息：{str(e)}")
            return pd.DataFrame()
    # ===================== 新增结束 =====================
    # 其他预留函数（get_trade_calendar/get_kline_day等）保持不变...


# 全局实例
data_fetcher = DataFetcher()

# 测试代码
if __name__ == "__main__":
    """测试stock_basic"""
    # try:
    #     # 1. 测试股票基础数据获取（修正exchange参数：SHSE=上交所，空则返回全部）
    #     logger.info("===== 测试：获取A股上市股票基础数据（全字段） =====")
    #     stock_base_df = data_fetcher.get_stockbase(
    #         exchange="SSE",  # 可选：仅上交所，注释则返回所有交易所
    #         list_status="L",  # 仅上市股票
    #     )
    #     if stock_base_df is not None:
    #         logger.info(f"股票基础数据预览（前3行）：\n{stock_base_df.head(3)}")
    #
    #     logger.info("\n===== 所有数据获取测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"数据获取测试失败，错误信息：{str(e)} ❌")
    """测试新增的stock_company接口"""
    # try:
    #     # 1. 测试获取单只股票的公司信息
    #     logger.info("===== 测试：获取单只股票（600000.SH）的公司基本信息 =====")
    #     single_company_df = data_fetcher.get_stock_company(ts_code="600000.SH")
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.width', None)
    #     print(single_company_df)
    #     # 2. 测试获取指定交易所的公司信息
    #     # logger.info("\n===== 测试：获取上交所所有上市公司基本信息 =====")
    #     # exchange_company_df = data_fetcher.get_stock_company(exchange="SSE")
    #     # if exchange_company_df is not None:
    #     #     logger.info(f"上交所公司信息行数：{len(exchange_company_df)}")
    #     #     logger.info(f"董事长字段示例：{exchange_company_df['chairman'].dropna().head(5).tolist()}")
    #
    #     logger.info("\n===== 所有接口测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"接口测试失败，错误信息：{str(e)} ❌")
    """测试新增的Kline_day接口"""
    logger.info("===== 测试：获取日线数据 =====")
    stock_Kline_day = data_fetcher.fetch_kline_day(
        ts_code="301613.SZ ",
        trade_date="",
        start_date='20250516',
        end_date= '20250521'

    )
    if stock_Kline_day is not None:
        logger.info(f"日线数据预览：\n{stock_Kline_day.head(3)}")
    logger.info("\n=====获取日线数据测试完成 ✅ =====")
    # ===================== 新增：指数日线接口测试代码 =====================
    """测试新增的index_daily接口"""
    # try:
    #     logger.info("===== 测试：获取指数日线数据 =====")
    #     # 测试1：单指数单日数据
    #     index_daily_single = data_fetcher.fetch_index_daily(
    #         ts_code="000001.SH",
    #         trade_date="20241231"
    #     )
    #     if not index_daily_single.empty:
    #         logger.info(f"单指数单日数据预览：\n{index_daily_single.head(3)}")

        # 测试2：多指数单日数据（批量获取）
    #     index_daily_multi = data_fetcher.fetch_index_daily(
    #         ts_code="000001.SH",
    #         trade_date="",
    #         start_date= "20090101",
    #         end_date= "20260213"
    #     )
    #     if not index_daily_multi.empty:
    #         logger.info(f"指数数据行数：{len(index_daily_multi)}")
    #         logger.info(f"指数数据预览：\n{index_daily_multi.head(3)}")
    #
    #     logger.info("\n===== 获取指数日线数据测试完成 ✅ =====")
    # except Exception as e:
    #     logger.error(f"指数日线数据测试失败，错误信息：{str(e)} ❌")

