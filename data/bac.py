import requests
import json


def send_wechat_message(token, title, content):
    """
    通过PushPlus向微信发送消息
    :param token: PushPlus的token（必填，替换成你的）
    :param title: 消息标题（必填）
    :param content: 消息内容（必填，可填Python程序输出）
    :return: 推送结果（True/False）
    """
    # PushPlus的核心API地址
    url = "http://www.pushplus.plus/send"
    # 构造推送参数
    data = {
        "token": token,
        "title": title,
        "content": content,
        "template": "txt"  # 消息格式：txt(纯文本)/markdown(富文本)
    }

    try:
        # 发送POST请求
        response = requests.post(
            url=url,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"}
        )
        # 解析响应结果
        result = response.json()
        if result["code"] == 200:
            print("✅ 消息推送成功！")
            return True
        else:
            print(f"❌ 消息推送失败：{result['msg']}")
            return False
    except Exception as e:
        print(f"❌ 推送过程出错：{str(e)}")
        return False


# 主程序：模拟Python程序输出并推送
if __name__ == "__main__":
    YOUR_PUSHPLUS_TOKEN = "028a3ef4df0a41aabb85320660a65bfe"

    # 模拟Python程序的输出内容（替换成你实际的程序输出即可）
    program_output = """
测试下推送
    """

    # 调用推送函数
    send_wechat_message(
        token=YOUR_PUSHPLUS_TOKEN,
        title="Python程序运行报告",  # 消息标题
        content=program_output  # 程序输出内容
    )