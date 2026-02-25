import requests
import json
import logging
from typing import Dict, List


logger = logging.getLogger(__name__)

def call_doubao_search_api(prompt: str, end_date: str) -> Dict:
    """
    适配新联网应用的API调用（核心修改版）
    :param prompt: 输入提示词
    :param end_date: 搜索截止日期（如2026-02-13）
    :return: 结构化响应 {"raw_text": "", "token_usage": {}, "error": None}
    """
    # 1. 核心配置替换（用你新应用的参数）
    global response
    API_KEY = ''  # API Key 不变，复用原有配置
    # API_URL = "https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"  # 新接口地址
    MODEL_ID = "bot-20260224232147-dfcct"  # 你的新应用model ID

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # 2. 适配新接口的请求参数（关键修改）
    data = {
        "model": MODEL_ID,  # 替换为应用专属ID，不再用原模型名
        "stream": False,  # 关闭流式返回（批量处理更方便，若要流式可改为True）
        "stream_options": {"include_usage": True},  # 新接口必填
        "messages": [
            {
                "role": "system",
                "content": f"""
你是A股全市场股票概念专家，仅检索截止到{end_date}的互联网消息、政策、财经资讯，不使用之后的信息；
仅使用标准化概念标签列表中的标签，不得自创表述。
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        # 联网能力已在应用端配置，无需再传plugins参数！
        "temperature": 0.1,
        "max_tokens": 2048
    }

    try:
        # 3. 发送请求（关闭流式后，返回完整文本）
        response = requests.post(
            url=API_URL,
            headers=headers,
            json=data,
            timeout=30  # 联网检索超时时间加长
        )
        response.raise_for_status()  # 抛出HTTP错误

        # 4. 解析响应（分流式/非流式两种情况）
        result = {
            "raw_text": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": None
        }

        if data["stream"]:
            # 若开启流式，逐行解析（批量处理建议关闭）
            raw_text = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8').lstrip('data: ')
                    if line == '[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        # 提取流式文本
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                            raw_text += chunk["choices"][0]["delta"]["content"]
                        # 提取Token消耗
                        if chunk.get("usage"):
                            result["token_usage"] = {
                                "prompt_tokens": chunk["usage"].get("prompt_tokens", 0),
                                "completion_tokens": chunk["usage"].get("completion_tokens", 0),
                                "total_tokens": chunk["usage"].get("total_tokens", 0)
                            }
                    except:
                        continue
            result["raw_text"] = raw_text.strip()
        else:
            # 非流式（推荐）：直接解析完整响应
            resp_json = response.json()
            # 提取核心文本
            if resp_json.get("choices") and len(resp_json["choices"]) > 0:
                result["raw_text"] = resp_json["choices"][0]["message"]["content"].strip()
            # 提取Token消耗
            if resp_json.get("usage"):
                result["token_usage"] = {
                    "prompt_tokens": resp_json["usage"].get("prompt_tokens", 0),
                    "completion_tokens": resp_json["usage"].get("completion_tokens", 0),
                    "total_tokens": resp_json["usage"].get("total_tokens", 0)
                }

        logger.info(f"新联网API返回文本：\n{result['raw_text']}")
        logger.info(f"Token消耗：输入{result['token_usage']['prompt_tokens']} | 输出{result['token_usage']['completion_tokens']}")
        return result

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP错误 {response.status_code}：{response.text}"
        logger.error(error_msg)
        return {"raw_text": "", "token_usage": {}, "error": error_msg}
    except requests.exceptions.Timeout:
        logger.error("新联网API调用超时")
        return {"raw_text": "", "token_usage": {}, "error": "接口调用超时"}
    except Exception as e:
        logger.error(f"新联网API调用异常：{str(e)}")
        return {"raw_text": "", "token_usage": {}, "error": f"调用异常：{str(e)}"}
if __name__ == "__main__":
    # 测试：调用新联网API补全单只股票概念
    test_prompt = """
仅使用标准化标签，输出格式：1 600879.SH 商业航天,卫星互联网,低空经济,军工电子
股票列表：["600879.SH|航天电子"]
"""
    result = call_doubao_search_api(test_prompt, end_date="2026-02-24")
    if not result["error"]:
        print(f"返回结果：{result['raw_text']}")
    else:
        print(f"调用失败：{result['error']}")