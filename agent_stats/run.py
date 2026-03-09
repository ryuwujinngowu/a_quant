"""
服务器运行入口文件
阿里云crontab定时触发的唯一文件
示例crontab配置（每个交易日凌晨3点执行）：
0 3 * * 1-5 /usr/bin/python3 /home/xxx/AQuant/agent_stats/run.py >> /home/xxx/AQuant/logs/agent_stats_run.log 2>&1
"""
# 先执行模块初始化，处理路径问题，必须放在最前面
import agent_stats

import sys
import time
from datetime import datetime
from utils.log_utils import logger
from utils.wechat_push import send_wechat_message
from agent_stats.stats_engine import AgentStatsEngine
from agent_stats.config import MAX_RETRY_TIMES, RETRY_INTERVAL


def main():
    logger.info("=" * 50)
    logger.info(f"智能体收益统计脚本启动，运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # 初始化引擎
    engine = AgentStatsEngine()
    retry_count = 0
    run_success = False

    # 重试机制
    while retry_count < MAX_RETRY_TIMES and not run_success:
        try:
            run_success = engine.run_full_flow()
            if run_success:
                logger.info("脚本执行成功！")
                # 推送成功通知
                try:
                    send_wechat_message(
                        "【智能体统计成功】",
                        f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n执行结果：成功完成所有统计任务"
                    )
                except Exception as e:
                    logger.warning(f"微信推送失败：{e}")
            else:
                retry_count += 1
                logger.warning(f"脚本执行失败，{RETRY_INTERVAL/60}分钟后重试，剩余重试次数：{MAX_RETRY_TIMES - retry_count}")
                time.sleep(RETRY_INTERVAL)
        except Exception as e:
            retry_count += 1
            logger.error(f"脚本执行异常：{e}", exc_info=True)
            # 推送异常告警
            try:
                send_wechat_message(
                    "【智能体统计异常告警】",
                    f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n异常信息：{str(e)[:500]}"
                )
            except Exception as push_e:
                logger.warning(f"告警推送失败：{push_e}")
            # 重试等待
            if retry_count < MAX_RETRY_TIMES:
                logger.warning(f"{RETRY_INTERVAL/60}分钟后重试，剩余重试次数：{MAX_RETRY_TIMES - retry_count}")
                time.sleep(RETRY_INTERVAL)

    # 最终失败处理
    if not run_success:
        logger.error(f"脚本执行失败，已达到最大重试次数{MAX_RETRY_TIMES}")
        try:
            send_wechat_message(
                "【智能体统计最终失败】",
                f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n已达到最大重试次数{MAX_RETRY_TIMES}，请手动检查！"
            )
        except Exception as push_e:
            logger.warning(f"失败告警推送失败：{push_e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()