"""
板块热度策略 — 日频信号推送 (独立运行脚本)
=============================================
功能：
  每日尾盘（建议 14:45 触发）调用 SectorHeatStrategy 的买入逻辑，
  生成今日买入信号后通过 PushPlus 推送到微信，无需启动回测引擎。

与 main.py / backtest/engine 的关系：
  完全解耦。本脚本直接调用策略层的核心方法，不依赖 MultiStockBacktestEngine，
  不维护持仓状态，不执行卖出逻辑（信号仅供人工决策参考）。

运行方式：
  # 直接运行（使用今日日期）
  python runner/sector_heat_runner.py

  # 指定日期（补跑历史信号）
  python runner/sector_heat_runner.py --date 2026-03-13

  # 静默模式（不推送微信，只打印日志，用于调试）
  python runner/sector_heat_runner.py --dry-run

云端 Linux 定时触发（crontab）：
  # 每个工作日 14:45 触发（北京时间）
  00 05 * * 1-5 cd /home/user/a_quant && python runner/sector_heat_runner.py >> logs/runner.log 2>&1

依赖前置：
  1. 已运行 python learnEngine/dataset.py 生成训练集
  2. 已运行 python train.py 训练并保存模型到 sector_heat_xgb_model.pkl
"""

import argparse
import os
import sys
from datetime import datetime

from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_tools import get_trade_dates, get_daily_kline_data
from utils.log_utils import logger
# 修正：保持导入名称正确，后续调用统一使用这个名称
from utils.wechat_push import send_wechat_message_to_multiple_users


# ============================================================
# 辅助工具
# ============================================================


def _is_trade_date(date_str: str) -> bool:
    """判断给定日期是否为 A 股交易日"""
    try:
        dates = get_trade_dates(date_str, date_str)
        return bool(dates)
    except Exception:
        return False


def _format_signal_message(trade_date: str, buy_signals: dict) -> Tuple[str, str]:
    """
    将买入信号 dict 格式化为 PushPlus 推送标题 + 正文

    :param trade_date:   信号日期
    :param buy_signals:  {ts_code: buy_type} 或 {}
    :return: (title, content) 两个字符串
    """
    if not buy_signals:
        title   = f"[量化] {trade_date} | 今日无买入信号"
        content = (
            f"板块热度策略 | {trade_date}\n"
            f"今日无满足条件的买入标的。\n"
            f"可能原因：这他妈的还买尼玛呢。"
        )
        return title, content

    title = f"[量化] {trade_date} | 发现 {len(buy_signals)} 个买入信号"
    lines = [
        f"📈 板块热度 XGBoost 策略",
        f"信号日期：{trade_date}",
        f"买入时机：尾盘（14:50-15:00）",
        f"卖出时机：D+2日开盘",
        "=" * 32,
    ]
    for i, (ts_code, buy_type) in enumerate(buy_signals.items(), 1):
        lines.append(f"  {i:>2}. {ts_code}   [{buy_type}]")
    lines += [
        "=" * 32,
        "⚠️  信号仅供参考，注意风险控制，亏了又叫比渣渣的",
        "⚠️  仓位建议：单股不超过总仓位 20%，要死不拦着",
    ]
    content = "\n".join(lines)
    return title, content


# ============================================================
# 主推送逻辑
# ============================================================

def run_daily_signal(trade_date: str, dry_run: bool = False) -> bool:
    """
    单日信号生成 + 微信推送

    :param trade_date: 交易日，格式 YYYY-MM-DD
    :param dry_run:    True=只打印，不推送微信（调试模式）
    :return: 推送成功返回 True
    """
    logger.info(f"[Runner] ===== 开始执行 | 日期: {trade_date} | dry_run={dry_run} =====")

    # ── Step 1: 检查是否为交易日 ──────────────────────────────────────────
    if not _is_trade_date(trade_date):
        msg = f"[Runner] {trade_date} 非交易日，跳过执行"
        logger.info(msg)
        if not dry_run:
            # 修改1：使用正确的函数名 send_wechat_message_to_multiple_users
            # 补充说明：如果你的函数需要指定用户列表，可添加 users 参数，例如：
            send_wechat_message_to_multiple_users(title=f"[量化] {trade_date} 今日非交易日", content=msg)
            send_wechat_message_to_multiple_users(
                title=f"[量化] {trade_date} 今日非交易日",
                content=msg,
            )
        return True

    # ── Step 2: 获取当日全市场日线（策略构建候选池需要）────────────────────
    daily_df = get_daily_kline_data(trade_date)
    if daily_df.empty:
        msg = f"[Runner] {trade_date} 无法获取日线数据，跳过"
        logger.warning(msg)
        if not dry_run:
            # 修改2：同上，修正函数调用名
            send_wechat_message_to_multiple_users(
                title=f"[量化] {trade_date} 数据异常",
                content=msg
            )
        return False

    logger.info(f"[Runner] 日线数据加载完成 | {len(daily_df)} 只股票")

    # ── Step 3: 调用策略买入信号生成（不依赖引擎，直接调用核心方法）────────
    # 延迟导入：避免在 import 阶段就触发模型加载
    from strategies.sector_heat_strategy import SectorHeatStrategy

    try:
        strategy    = SectorHeatStrategy()
        buy_signals = strategy._generate_buy_signal(trade_date, daily_df)
    except Exception as e:
        msg = f"[Runner] 策略执行异常: {e}"
        logger.error(msg, exc_info=True)
        if not dry_run:
            # 修改3：同上，修正函数调用名
            send_wechat_message_to_multiple_users(
                title=f"[量化] {trade_date} 策略执行异常",
                content=msg
            )
        return False

    # ── Step 4: 格式化 + 推送 ────────────────────────────────────────────
    title, content = _format_signal_message(trade_date, buy_signals)

    if dry_run:
        logger.info(f"[Runner][DRY-RUN] 标题: {title}")
        logger.info(f"[Runner][DRY-RUN] 正文:\n{content}")
        return True

    # 修改4：核心推送逻辑，修正函数调用名
    success = send_wechat_message_to_multiple_users(title, content)
    if success:
        logger.info(f"[Runner] 推送成功 | {title}")
    else:
        logger.warning(f"[Runner] 推送失败，信号已记录到日志")

    return success


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="板块热度策略日频信号推送（独立脚本，无需启动回测引擎）"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="交易日期，格式 YYYY-MM-DD（默认今日）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="调试模式：只打印信号，不推送微信",
    )
    args = parser.parse_args()

    ok = run_daily_signal(trade_date=args.date, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)