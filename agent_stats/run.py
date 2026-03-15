"""
服务器运行入口
==============
crontab 示例（每工作日凌晨 3 点）：
  0 3 * * 1-5 /usr/bin/python3 /home/xxx/a_quant/agent_stats/run.py >> /home/xxx/a_quant/logs/agent_stats.log 2>&1

常用场景
--------
# 日常运行（cron，不传参）：
  nohup python3.8 -u run.py  > /dev/null 2>&1 &

# 首次运行 / 指定历史起始日期（新部署时）：
  nohup python3.8 -u run.py --start-date 2024-11-01 > /dev/null 2>&1 &

# 手动重跑指定 agent（策略逻辑更新后重新计算历史）：
  python3.8 un.py --reset-agent morning_limit_up,afternoon_limit_up --reset-from 2024-10-01

# 不指定 --reset-from 则从 config.START_DATE 起重跑：
  python agent_stats/run.py --reset-agent limit_down_buy

⚠ --reset-agent 会删除 DB 对应记录并重跑，不可逆，系统不会自动触发此参数。
"""
import sys
from pathlib import Path

# 获取当前脚本的绝对路径 (agent_stats/run.py)
current_file = Path(__file__).resolve()
# 项目根目录是 agent_stats 的父目录 (a_quant/)
project_root = current_file.parent.parent

# 将项目根目录加入 sys.path（如果尚未存在）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# -------------------------------------------
# 先执行模块初始化（路径处理），必须放最前
import agent_stats  # noqa: F401
import argparse
import sys
import time
from datetime import datetime, timedelta

from agent_stats.config import MAX_RETRY_TIMES, RETRY_INTERVAL, START_DATE
from agent_stats.stats_engine import AgentStatsEngine
from agent_stats.wechat_reporter import AgentWechatReporter
from data.data_cleaner import is_rate_limit_aborted
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users


def _sleep_until_midnight() -> None:
    """
    睡眠至次日零点（多5分钟缓冲），期间每小时打印一次等待日志。
    用于 Tushare 每日配额耗尽后等待次日配额刷新，进程保持存活不需要人工重启。
    """
    now  = datetime.now()
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    total_secs = (next_midnight - now).total_seconds()
    logger.info(
        f"[限流等待] 当日 Tushare 配额耗尽，进程休眠至次日零点后（{next_midnight.strftime('%m-%d %H:%M')}），"
        f"共需等待 {int(total_secs // 3600)}h{int((total_secs % 3600) // 60)}m"
    )
    slept = 0
    check_interval = 3600  # 每小时 log 一次
    while slept < total_secs:
        sleep_this = min(check_interval, total_secs - slept)
        time.sleep(sleep_this)
        slept += sleep_this
        remaining = total_secs - slept
        if remaining > 60:
            logger.info(f"[限流等待] 还需等待约 {int(remaining // 3600)}h{int((remaining % 3600) // 60)}m")


def _parse_args():
    parser = argparse.ArgumentParser(description="Agent 收益统计引擎")
    parser.add_argument(
        "--start-date", type=str, default=None,
        help=f"统计起始日期（YYYY-MM-DD），新 agent 从此日期回溯。不传则用 config.START_DATE={START_DATE}",
    )
    parser.add_argument(
        "--reset-agent", type=str, default=None,
        help="逗号分隔的 agent_id，强制从 --reset-from 日期删除并重跑。"
             "例：--reset-agent morning_limit_up,limit_down_buy",
    )
    parser.add_argument(
        "--reset-from", type=str, default=None,
        help="配合 --reset-agent，指定重跑起始日期（YYYY-MM-DD）。不传则用 --start-date 或 config.START_DATE。",
    )
    parser.add_argument(
        "--repair-incomplete", action="store_true", default=False,
        help="修复历史遗留的数据不完整记录并重新计算。\n"
             "  [MIN_FAIL] 记录（分钟线聚合告警）：删除后由引擎用断点续跑重新生成信号（缓存后成功率高）。\n"
             "  D+1 NULL 记录（D+1 未结账）：由 run_full_flow 的 dates_unclosed 自动重算。\n"
             "用法：python agent_stats/run.py --repair-incomplete",
    )
    return parser.parse_args()


def _build_reset_agents(args) -> dict:
    """将 CLI 参数转换为引擎所需的 {agent_id: from_date} 字典"""
    if not args.reset_agent:
        return {}
    from_date = args.reset_from or args.start_date or START_DATE
    agent_ids = [aid.strip() for aid in args.reset_agent.split(",") if aid.strip()]
    reset = {aid: from_date for aid in agent_ids}
    logger.info(f"手动重置 agent 列表：{reset}")
    return reset


def main():
    args = _parse_args()

    logger.info("=" * 60)
    logger.info(f"[agent_stats] 启动  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.start_date:
        logger.info(f"  --start-date  : {args.start_date}")
    if args.reset_agent:
        logger.info(f"  --reset-agent : {args.reset_agent}")
        logger.info(f"  --reset-from  : {args.reset_from or '(使用 start_date)'}")
    if args.repair_incomplete:
        logger.info(f"  --repair-incomplete: 开启，将删除 [MIN_FAIL] 记录后由引擎断点续跑重算")
    logger.info("=" * 60)

    reset_agents = _build_reset_agents(args)
    engine   = AgentStatsEngine(start_date=args.start_date)

    # 修复数据不完整记录（在正常运行前执行，run_full_flow 会处理被删除的日期）
    if args.repair_incomplete:
        deleted = engine.repair_incomplete_records()
        logger.info(f"[repair-incomplete] 完成，删除 {deleted} 条 [MIN_FAIL] 记录，"
                    f"继续正常运行引擎（将重算这些日期）...")
    reporter = AgentWechatReporter()

    run_success = False
    retry_count = 0

    while retry_count < MAX_RETRY_TIMES and not run_success:
        try:
            run_success = engine.run_full_flow(reset_agents=reset_agents)
            if run_success:
                logger.info("引擎运行完成")
                # 仅推送最新交易日的统计（历史补全不推，避免刷屏）
                if engine.all_trade_dates:
                    reporter.report_latest(engine.all_trade_dates[-1])
            else:
                # 判断是否因 Tushare 当日配额耗尽触发 abort
                if is_rate_limit_aborted():
                    # 配额耗尽：sleep 至次日零点（不消耗 retry_count），
                    # 次日配额刷新后自动恢复，无需人工干预。
                    _sleep_until_midnight()
                    logger.info("[限流等待] 次日零点已到，限流状态自动重置，恢复正常运行...")
                    # 注意：_THROTTLE_STATE 在进程内存中，_throttle_get_mode 会在
                    # 下次调用时检测到日期变化并自动重置为 normal，无需手动重置。
                else:
                    # 其他原因导致的 False（数据问题等），按常规重试逻辑处理
                    retry_count += 1
                    logger.warning(
                        f"运行返回 False，{RETRY_INTERVAL // 60} 分钟后重试"
                        f"（{retry_count}/{MAX_RETRY_TIMES}）"
                    )
                    time.sleep(RETRY_INTERVAL)
        except Exception as e:
            retry_count += 1
            logger.error(f"运行异常：{e}", exc_info=True)
            try:
                send_wechat_message_to_multiple_users("【agent_stats 异常】", str(e)[:500])
            except Exception:
                pass
            if retry_count < MAX_RETRY_TIMES:
                time.sleep(RETRY_INTERVAL)

    if not run_success:
        logger.error(f"已达最大重试次数 {MAX_RETRY_TIMES}，任务终止")
        try:
            send_wechat_message_to_multiple_users("【agent_stats 最终失败】", "请手动检查日志")
        except Exception:
            pass
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
