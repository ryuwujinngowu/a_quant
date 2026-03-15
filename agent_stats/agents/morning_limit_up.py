"""
早盘打板选手（MorningLimitUpAgent）
=====================================
策略逻辑
--------
模拟早盘阶段对所有涨停板的买入行为（无仓位限制，平铺全部命中标的）。

命中条件：
  1. 当日 high >= 涨停价 * 0.999（惯用日线候选池，避免过滤误差）
  2. 非 ST / *ST 股票
  3. 排除北交所
  4. 一字板（open >= 涨停价 * 0.999）：必须曾经开板（min_low < 涨停价 * 0.999），
     且开板后出现回封（参考 multi_limit_up_strategy.check_reopen）；
     纯一字板全天未开板 → 无法实盘买入 → 跳过
  5. 非一字板：首次触板时间（分钟线 high >= 涨停价 * 0.999）需在 11:30 之前

买入价：涨停价（由前收盘价根据板块规则计算）

注意事项
--------
- 使用分钟线数据判断触板时间，不依赖 limit_list_ths.first_time
- ThreadPoolExecutor 并发拉取分钟线，减少等待时间
- 每只股票只命中一次（dict 去重）
"""
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from agent_stats.agent_base import BaseAgent
from data.data_cleaner import data_cleaner
from utils.common_tools import calc_limit_up_price
from utils.log_utils import logger

MORNING_CUTOFF_H = 11
MORNING_CUTOFF_M = 30
TOL = 0.999


def _get_min_df(ts_code: str, trade_date: str) -> pd.DataFrame:
    """获取分钟线（自动入库），返回空 DF 表示无数据"""
    result = data_cleaner.get_kline_min_by_stock_date(ts_code, trade_date)
    if result is None or result.empty:
        return pd.DataFrame()
    return result


def _get_first_limit_time(min_df: pd.DataFrame, limit_price: float) -> Optional[datetime]:
    """返回分钟线中首次 high >= limit_price * TOL 的时间，无则 None"""
    if not pd.api.types.is_datetime64_any_dtype(min_df["trade_time"]):
        min_df = min_df.copy()
        min_df["trade_time"] = pd.to_datetime(min_df["trade_time"])
    hit = min_df[min_df["high"] >= limit_price * TOL]
    if hit.empty:
        return None
    return hit.sort_values("trade_time")["trade_time"].iloc[0]


def _check_reopen(min_df: pd.DataFrame, limit_price: float) -> Optional[datetime]:
    """
    一字板开板+回封校验：
      - 若全天 min_low >= threshold → 未曾开板 → None
      - 找到第一个 low < threshold 的分钟，检查本分钟或下一分钟 close >= threshold → 回封时间
      - 未回封 → None
    """
    threshold = limit_price * TOL
    if not pd.api.types.is_datetime64_any_dtype(min_df["trade_time"]):
        min_df = min_df.copy()
        min_df["trade_time"] = pd.to_datetime(min_df["trade_time"])
    df = min_df.sort_values("trade_time").reset_index(drop=True)
    if df["low"].min() >= threshold:
        return None   # 全天未开板
    for i, row in df.iterrows():
        if row["low"] < threshold:
            if row["close"] >= threshold:
                return row["trade_time"]
            if i + 1 < len(df):
                nxt = df.iloc[i + 1]
                if nxt["close"] >= threshold:
                    return nxt["trade_time"]
    return None   # 开板后未回封


class MorningLimitUpAgent(BaseAgent):
    agent_id   = "morning_limit_up"
    agent_name = "早盘打板选手"
    agent_desc = (
        "早盘涨停打板策略：当日 high 触板（≥涨停价×0.999），非ST/非北交所；"
        "一字板须曾开板且回封；首次触板/回封时间 < 11:30 为早盘命中；买入价为涨停价。"
    )

    def get_signal_stock_pool(
        self,
        trade_date: str,
        daily_data: pd.DataFrame,
        context: Dict,
    ) -> List[Dict]:
        st_set = set(context.get("st_stock_list", []))

        # ── Step 1: 构建前收价映射 ────────────────────────────────────────
        pre_close_map: Dict[str, float] = {}
        pre_data = context.get("pre_close_data", pd.DataFrame())
        if not pre_data.empty and "ts_code" in pre_data.columns and "close" in pre_data.columns:
            pre_close_map = dict(zip(pre_data["ts_code"], pre_data["close"]))
        if "pre_close" in daily_data.columns:
            for _, row in daily_data.iterrows():
                if row["ts_code"] not in pre_close_map:
                    pre_close_map[row["ts_code"]] = row["pre_close"]

        # ── Step 2: 候选池 — 当日 high 触板 + 非ST + 非北交所 ───────────
        candidates = []
        for _, row in daily_data.iterrows():
            ts_code = row["ts_code"]
            if ts_code in st_set:
                continue
            code_prefix = ts_code.split(".")[0]
            if code_prefix.startswith(("83", "87", "88")) or ts_code.endswith(".BJ"):
                continue
            pre_close = pre_close_map.get(ts_code, 0.0)
            if pre_close <= 0:
                continue
            limit_price = calc_limit_up_price(ts_code, pre_close)
            if limit_price <= 0:
                continue
            high = row.get("high", 0.0) if hasattr(row, "get") else getattr(row, "high", 0.0)
            if float(high) >= limit_price * TOL:
                candidates.append({
                    "ts_code":     ts_code,
                    "stock_name":  str(row.get("name", "") if hasattr(row, "get") else ""),
                    "buy_price":   limit_price,
                    "pre_close":   pre_close,
                    "open":        float(row.get("open", 0.0) if hasattr(row, "get") else 0.0),
                    "limit_price": limit_price,
                })

        if not candidates:
            logger.info(f"[{self.agent_id}][{trade_date}] 无涨停候选，无信号")
            return []

        # ── Step 3: 并发拉取分钟线 ───────────────────────────────────────
        # max_workers=10 时各线程各自 sleep 1s 后同时唤醒，会瞬间并发 10 个 API 请求。
        # data_cleaner 内置全局信号量（_TUSHARE_MIN_API_SEM）已限制并发数，
        # 此处保持 10 以充分利用 DB 缓存命中的并发（缓存命中不占 API 配额）。
        ts_codes = [c["ts_code"] for c in candidates]
        min_data: Dict[str, pd.DataFrame] = {}
        fetch_failed: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_get_min_df, ts, trade_date): ts for ts in ts_codes}
            for future in concurrent.futures.as_completed(futures):
                ts = futures[future]
                try:
                    min_data[ts] = future.result()
                except Exception as e:
                    logger.warning(f"[{self.agent_id}][{trade_date}][{ts}] 分钟线拉取失败：{e}")
                    min_data[ts] = pd.DataFrame()
                    fetch_failed.append(ts)
        if fetch_failed:
            logger.warning(
                f"[{self.agent_id}][{trade_date}] ⚠ 分钟线拉取失败 {len(fetch_failed)}/{len(ts_codes)} 只"
                f"，这些候选将被跳过（影响信号池完整性）：{fetch_failed[:5]}{'...' if len(fetch_failed) > 5 else ''}"
            )

        # ── Step 4: 逐个判断触板时间，筛选早盘命中 ──────────────────────
        result = {}
        for c in candidates:
            ts_code    = c["ts_code"]
            limit_price = c["limit_price"]
            min_df     = min_data.get(ts_code, pd.DataFrame())

            if min_df is None or min_df.empty:
                logger.debug(f"[{self.agent_id}][{trade_date}][{ts_code}] 无分钟线，跳过")
                continue

            is_one_price = c["open"] >= limit_price * TOL

            if is_one_price:
                touch_time = _check_reopen(min_df, limit_price)
                if touch_time is None:
                    logger.debug(f"[{self.agent_id}][{trade_date}][{ts_code}] 一字板未开/未回封，跳过")
                    continue
            else:
                touch_time = _get_first_limit_time(min_df, limit_price)
                if touch_time is None:
                    logger.debug(f"[{self.agent_id}][{trade_date}][{ts_code}] 分钟线中无触板，跳过")
                    continue

            # 早盘判定：触板时间 < 11:30
            t = pd.Timestamp(touch_time)
            if t.hour > MORNING_CUTOFF_H or (t.hour == MORNING_CUTOFF_H and t.minute >= MORNING_CUTOFF_M):
                continue

            if ts_code not in result:
                result[ts_code] = {
                    "ts_code":    ts_code,
                    "stock_name": c["stock_name"],
                    "buy_price":  c["buy_price"],
                }

        final = list(result.values())
        logger.info(f"[{self.agent_id}][{trade_date}] 早盘涨停命中 {len(final)} 只（分钟线过滤）")
        return final
