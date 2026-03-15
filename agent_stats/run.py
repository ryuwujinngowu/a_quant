"""
服务器运行入口
==============
crontab 示例（每工作日凌晨 3 点）：
  0 3 * * 1-5 /usr/bin/python3 /home/xxx/a_quant/agent_stats/run.py >> /home/xxx/a_quant/logs/agent_stats.log 2>&1

常用场景
--------
# 日常运行（cron，不传参）：
  python agent_stats/run.py

# 首次运行 / 指定历史起始日期（新部署时）：
  python agent_stats/run.py --start-date 2024-10-01

# 手动重跑指定 agent（策略逻辑更新后重新计算历史）：
  python agent_stats/run.py --reset-agent morning_limit_up,afternoon_limit_up --reset-from 2024-10-01

# 不指定 --reset-from 则从 config.START_DATE 起重跑：
  python agent_stats/run.py --reset-agent limit_down_buy

⚠ --reset-agent 会删除 DB 对应记录并重跑，不可逆，系统不会自动触发此参数。
"""
# 先执行模块初始化（路径处理），必须放最前
import agent_stats  # noqa: F401

import argparse
import sys
import time
from datetime import datetime

from agent_stats.config import MAX_RETRY_TIMES, RETRY_INTERVAL, START_DATE
from agent_stats.stats_engine import AgentStatsEngine
from agent_stats.wechat_reporter import AgentWechatReporter
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message_to_multiple_users


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
    logger.info("=" * 60)

    reset_agents = _build_reset_agents(args)
    engine   = AgentStatsEngine(start_date=args.start_date)
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
