import requests
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def send_wechat_message_to_multiple_users(title, content, tokens):
    """
    通过PushPlus向多个微信用户发送消息（官方合规方式）
    :param title: 消息标题（必填）
    :param content: 消息内容
    :param tokens: 接收人的PushPlus Token列表，如["token1", "token2"]
    :return: 整体推送结果（True=全部成功/False=至少一个失败）
    """
    # PushPlus的核心API地址
    url = "http://www.pushplus.plus/send"
    # 记录每个用户的推送结果
    push_results = []

    if not tokens:
        print("❌ 未配置任何用户Token！")
        return False

    for idx, token in enumerate(tokens, 1):
        if not token:
            print(f"❌ 第{idx}个用户的Token为空，跳过推送")
            push_results.append(False)
            continue

        # 构造单用户推送参数（符合官方文档要求）
        data = {
            "token": token,  # 单个用户的Token（官方仅支持单个）
            "title": title,
            "content": content,
            "template": "txt"  # 消息格式：txt(纯文本)/markdown(富文本)
        }

        try:
            # 发送POST请求（单用户）
            response = requests.post(
                url=url,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
                timeout=10  # 增加超时控制，避免卡壳
            )
            # 解析响应结果
            result = response.json()
            if result["code"] == 200:
                print(f"✅ 第{idx}个用户消息推送成功！")
                push_results.append(True)
            else:
                print(f"❌ 第{idx}个用户推送失败：{result['msg']}（Token：{token[:8]}...）")
                push_results.append(False)
        except Exception as e:
            print(f"❌ 第{idx}个用户推送过程出错：{str(e)}（Token：{token[:8]}...）")
            push_results.append(False)

    # 整体结果：全部成功才返回True，否则False
    return all(push_results)


# 主程序：模拟Python程序输出并推送
if __name__ == "__main__":
    # 模拟Python程序的输出内容
    program_output = """
✅回测完成，核心指标汇总：
✅ 策略名称：涨停回马枪策略
✅ 回测开始日期：2026-01-05
✅ 回测结束日期：2026-02-01
✅ 初始本金(元)：1000000
✅最终资产(元)：224329.38
✅总收益率(%)：-77.57
✅ 年化收益率(%)：-100.0
✅年化波动率(%)：95.19
✅最大回撤(%)：-77.57
✅ 夏普比率：-17.72
✅信息比率：无基准
✅ 交易胜率(%)：0.0
✅ 总交易次数：25
✅回测交易日数：21
✅ ============================================================
✅ 【盈利最多的Top10股票】（实际均亏损）
✅  1. 000905.SZ：累计净盈亏 -15269.21 元
✅   2. 003037.SZ：累计净盈亏 -15882.87 元
✅ ============================================================
✅ 【亏损最多的Top10股票】
✅  1. 000409.SZ：累计净盈亏 -58391.42 元
✅  2. 001231.SZ：累计净盈亏 -55399.46 元
✅- ============================================================
    """

    # 配置两个接收人的PushPlus Token（替换成你实际的有效Token！）
    user_tokens = [
        "88ae50c3af6c41ab8a94a25b7aabe4f9",  # 第一个人的有效Token
        "028a3ef4df0a41aabb85320660a65bfe"  # 第二个人的有效Token
    ]

    # 调用多用户推送函数
    total_result = send_wechat_message_to_multiple_users(
        title="Python程序运行报告",
        content=program_output,
        tokens=user_tokens
    )

    # 最终结果提示
    if total_result:
        print("🎉 所有用户都推送成功！")
    else:
        print("⚠️ 部分用户推送失败，请检查Token是否正确！")