import requests
import json
from utils.log_utils import logger
from typing import Dict, List
from utils.db_utils import db


def call_doubao_search_api(prompt: str) -> Dict:
    """
    适配新联网应用的API调用（核心修改版）
    :param prompt: 输入提示词
    :return: 结构化响应 {"raw_text": "", "token_usage": {}, "error": None}
    """
    # 1. 核心配置替换（用你新应用的参数）
    global response
    API_KEY = '02f9dd23-b023-43bb-ae90-0c91ff7b9324'  # API Key 不变，复用原有配置
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


def get_stock_concepts_batch(stock_batch: List[Dict], batch_index: int = 0) -> Dict:
    """
    批量获取股票全量概念标签（联网搜索+强制标准化）
    """
    stock_info = [f"{item['ts_code']}|{item['name']}" for item in stock_batch]
    stock_count = len(stock_batch)

    # 精简Prompt，明确要求使用标准标签
    prompt = f"""
严格按以下格式输出，无任何多余内容：
每行1只股票，格式：序号 股票代码 概念标签1,概念标签2,概念标签3,...
股票列表（共{stock_count}只）：
{stock_info}
输出示例：
1 002456.SZ AI算力,半导体,先进封装,液冷
2 600879.SH 商业航天,卫星互联网,低空经济,军工电子
"""

    logger.info(f"===== 批次[{batch_index}] 开始批量获取概念，共{stock_count}只股票 =====")
    api_resp = call_doubao_search_api(prompt)

    if api_resp.get("error") or not api_resp.get("raw_text"):
        logger.error(f"批次[{batch_index}] API调用失败：{api_resp.get('error')}")
        return {"error": api_resp.get("error"), "data": []}

    try:
        raw_text = api_resp["raw_text"].strip()
        result_data = []

        for line in raw_text.split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                logger.debug(f"跳过无效行：{line}")
                continue

            # 拆分：序号 代码 概念
            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                logger.warning(f"行格式错误，跳过：{line}")
                continue

            seq_str, ts_code, raw_concepts = parts
            try:
                seq = int(seq_str) - 1  # 转成列表索引（从0开始）
            except ValueError:
                logger.warning(f"序号非数字，跳过行：{line}")
                continue
            if 0 <= seq < stock_count:
                result_data.append({
                    "ts_code": ts_code,
                    "concept_tags": raw_concepts
                })
            else:
                logger.warning(f"序号{seq + 1}超出范围（共{stock_count}只），跳过：{line}")

        logger.info(f"批次[{batch_index}] 解析完成，成功获取{len(result_data)}/{stock_count}只股票概念")
        return {"error": None, "data": result_data}

    except Exception as e:
        logger.error(f"批次[{batch_index}] 解析失败：{str(e)}，原始文本：{raw_text}")
        return {"error": f"解析失败：{str(e)}", "data": []}


def update_concept_tags_for_20_stocks():
    """
    批量更新20只股票的concept_tags字段，测试效果
    """
    logger.info("===== 开始更新20只股票的concept_tags =====")

    # 1. 从数据库读取20只股票（ts_code + name）
    read_sql = "SELECT ts_code, name FROM stock_basic LIMIT 20"
    stocks = db.query(read_sql)
    if not stocks:
        logger.error("未读取到任何股票数据，终止更新")
        return
    logger.info(f"成功读取{len(stocks)}只股票：{[s['ts_code'] for s in stocks]}")

    # 2. 调用API批量获取概念
    concept_result = get_stock_concepts_batch(stocks, batch_index=1)
    if concept_result.get("error") or not concept_result.get("data"):
        logger.error(f"批量获取概念失败，终止更新：{concept_result.get('error')}")
        return

    # 3. 批量更新到数据库
    update_sql = "UPDATE stock_basic SET concept_tags = %s WHERE ts_code = %s"
    params_list = [
        (item["concept_tags"], item["ts_code"])
        for item in concept_result["data"]
    ]

    logger.info(f"准备批量更新{len(params_list)}条记录")
    affected_rows = db.batch_execute(update_sql, params_list)

    if affected_rows is not None:
        logger.info(f"✅ 批量更新成功，影响行数：{affected_rows}")
        # 打印更新详情
        for item in concept_result["data"]:
            logger.info(f"[{item['ts_code']}] concept_tags: {item['concept_tags']}")
    else:
        logger.error("❌ 批量更新失败")



if __name__ == "__main__":
#     # 测试：调用新联网API补全单只股票概念
#     test_prompt = """
# 仅使用标准化标签，输出格式：1 600879.SH 商业航天,卫星互联网,低空经济,军工电子
# 股票列表：["600879.SH|航天电子"]
# """
#     result = call_doubao_search_api(test_prompt, end_date="2026-02-24")
#     if not result["error"]:
#         print(f"返回结果：{result['raw_text']}")
#     else:
#         print(f"调用失败：{result['error']}")
    update_concept_tags_for_20_stocks()

