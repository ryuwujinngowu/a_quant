import os
import pandas as pd
import tushare as ts
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from utils.log_utils import logger

ts.set_token('6a3e1b964b1847a66a6e4c5421006605ab279b9b2d4ca33a8aa3e8b3')
pd.set_option('display.max_rows', None)
#东财数据
# deal_realtime = ts.realtime_tick(ts_code='600000.SH', src='dc')    #实时成交
#东财数据
# tick_realtime = ts.realtime_quote(ts_code='600000.SH', src='dc')    #实时盘口

df = ts.pro_bar(ts_code='301613.SZ', adj='qfq', start_date='20250516', end_date='20250521')

# rank_realtime= ts.realtime_list(src='dc')

print(df)