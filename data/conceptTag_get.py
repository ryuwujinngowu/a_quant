import time
import requests
import json
from utils.log_utils import logger
from typing import Dict, List
from utils.db_utils import db
# 导入Token统计工具


def call_doubao_search_api(prompt: str) -> Dict:
    """
    适配新联网应用的API调用（核心修改版）
    :param prompt: 输入提示词
    :return: 结构化响应 {"raw_text": "", "token_usage": {}, "error": None}
    """
    API_KEY = ''
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
    MODEL_ID = "bot-20260224232147-dfcct"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    data = {
        "model": MODEL_ID,
        "stream": False,
        "stream_options": {"include_usage": True},
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }

    try:
        # 移除危险的 global response，本地变量即可
        response = requests.post(
            url=API_URL,
            headers=headers,
            json=data,
            timeout=40
        )
        response.raise_for_status()

        result = {
            "raw_text": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": None
        }

        resp_json = response.json()
        if resp_json.get("choices") and len(resp_json["choices"]) > 0:
            result["raw_text"] = resp_json["choices"][0]["message"]["content"].strip()

        if resp_json.get("usage"):
            result["token_usage"] = {
                "prompt_tokens": resp_json["usage"].get("prompt_tokens", 0),
                "completion_tokens": resp_json["usage"].get("completion_tokens", 0),
                "total_tokens": resp_json["usage"].get("total_tokens", 0)
            }

        logger.info(f"API返回文本：\n{result['raw_text']}")
        logger.info(f"Token消耗：输入{result['token_usage']['prompt_tokens']} | 输出{result['token_usage']['completion_tokens']}")
        return result

    except requests.exceptions.HTTPError as e:
        # 安全获取状态码，不使用未定义变量
        status = response.status_code if 'response' in locals() else 'unknown'
        text = response.text if 'response' in locals() else ''
        error_msg = f"HTTP错误 {status}：{text}"
        logger.error(error_msg)
        return {"raw_text": "", "token_usage": {}, "error": error_msg}
    except requests.exceptions.Timeout:
        logger.error("API调用超时")
        return {"raw_text": "", "token_usage": {}, "error": "接口调用超时"}
    except Exception as e:
        logger.error(f"API调用异常：{str(e)}")
        return {"raw_text": "", "token_usage": {}, "error": f"调用异常：{str(e)}"}


def get_stock_concepts_batch(stock_batch: List[Dict], batch_index: int = 0) -> Dict:
    """
    批量获取股票全量概念标签（原有逻辑100%保留，不修改）
    """
    stock_info = [f"{item['ts_code']}|{item['name']}" for item in stock_batch]
    stock_count = len(stock_batch)

    prompt = f"""
严格按以下格式输出，无任何多余内容：
每行1只股票，格式：序号 股票代码 概念标签1,概念标签2,概念标签3,...
股票列表（共{stock_count}只）：
{stock_info}
输出示例：
1 002456.SZ AI算力,半导体,先进封装,液冷
2 600879.SH 商业航天,卫星互联网,低空经济,军工电子
注意！二次review你输出的内容,特别是股票代码,是.SH还是.SZ，和给你的列表里的代码一一对应检查,不要写错
我再提醒你一遍，注意你的输出格式，严格按照示例来。仅输出序号 股票代码 标签……不要给我股票代码|股票名称
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

            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                logger.warning(f"行格式错误，跳过：{line}")
                continue

            seq_str, ts_code, raw_concepts = parts
            try:
                seq = int(seq_str) - 1
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
        return {
            "error": None,
            "data": result_data,
            "token_usage": api_resp["token_usage"]  # 只加这一行
        }

    except Exception as e:
        logger.error(f"批次[{batch_index}] 解析失败：{str(e)}，原始文本：{raw_text}")
        return {"error": f"解析失败：{str(e)}", "data": []}

#
# def update_concept_tags_for_20_stocks():
#     """
#     【原有函数100%保留】批量更新20只股票的concept_tags字段，测试效果
#     """
#     logger.info("===== 开始更新20只股票的concept_tags =====")
#     read_sql = "SELECT ts_code, name FROM stock_basic LIMIT 20"
#     stocks = db.query(read_sql)
#     if not stocks:
#         logger.error("未读取到任何股票数据，终止更新")
#         return
#     logger.info(f"成功读取{len(stocks)}只股票：{[s['ts_code'] for s in stocks]}")
#
#     concept_result = get_stock_concepts_batch(stocks, batch_index=1)
#     if concept_result.get("error") or not concept_result.get("data"):
#         logger.error(f"批量获取概念失败，终止更新：{concept_result.get('error')}")
#         return
#
#     update_sql = "UPDATE stock_basic SET concept_tags = %s WHERE ts_code = %s"
#     params_list = [(item["concept_tags"], item["ts_code"]) for item in concept_result["data"]]
#
#     logger.info(f"准备批量更新{len(params_list)}条记录")
#     affected_rows = db.batch_execute(update_sql, params_list)
#
#     if affected_rows is not None:
#         logger.info(f"✅ 批量更新成功，影响行数：{affected_rows}")
#         for item in concept_result["data"]:
#             logger.info(f"[{item['ts_code']}] concept_tags: {item['concept_tags']}")
#     else:
#         logger.error("❌ 批量更新失败")


def batch_update_with_single_fail_log(params_list: List[tuple]) -> int:
    """
    【核心新增】批量更新 + 单条失败时记录ERROR级别股票代码
    满足你要求：个别失败，日志打印具体ts_code
    原有批量逻辑不变，只增加失败日志
    """
    update_sql = "UPDATE stock_basic SET concept_tags = %s WHERE ts_code = %s"
    success_count = 0

    for param in params_list:
        concept_tags, ts_code = param
        try:
            rows = db.execute(update_sql, (concept_tags, ts_code))
            if rows and rows > 0:
                success_count += 1
            else:
                logger.error(f"更新失败[{ts_code}]：未找到股票或无变更")
        except Exception as e:
            # ✅ 你要的：个别失败，ERROR日志记录股票代码
            logger.error(f"更新异常[{ts_code}]：{str(e)}")

    return success_count


def update_all_stock_concept_tags(
        batch_size: int = 20,
        ts_code: list = None,  # 移除Optional，直接用list（新手更易理解，且无导入报错）
        update: bool = True
):
    """
    【核心新增】逐批更新全表，直到全部完成
    1. 有序分页（按ts_code，不漏不重）
    2. 自动分批
    3. 单条失败打error日志
    4. 统计总Token
    新增参数：
        ts_code: 股票代码列表，传值则仅查询该列表范围内的股票；不传则查全表
        update: 是否仅更新concept_tags为空的股票（True=增量更新，False=全量重新查），默认True
    """

    logger.info("===== 开始全量更新 stock_basic 所有股票 concept_tags =====")
    total_stock = 0
    total_success = 0
    batch_index = 0
    last_ts_code = ""

    while True:
        batch_index += 1
        # ==================== 重构SQL拼接逻辑（核心修改） ====================
        # 基础SELECT语句
        base_sql = "SELECT ts_code, name FROM stock_basic"
        # 构建WHERE条件列表
        where_conditions = []

        # 1. 股票代码列表条件（传值则限定范围）
        if ts_code and len(ts_code) > 0:
            # 拼接IN条件（处理列表转SQL格式）
            ts_code_str = ','.join([f"'{code}'" for code in ts_code])
            where_conditions.append(f"ts_code IN ({ts_code_str})")

        # 2. 分页条件（ts_code > last_ts_code）
        if last_ts_code:
            where_conditions.append(f"ts_code > '{last_ts_code}'")

        # 3. update参数条件（True=仅查concept_tags为空，False=不限制）
        if update:
            where_conditions.append("concept_tags IS NULL")

        # 拼接完整SQL
        if where_conditions:
            read_sql = f"{base_sql} WHERE {' AND '.join(where_conditions)} ORDER BY ts_code LIMIT {batch_size}"
        else:
            read_sql = f"{base_sql} ORDER BY ts_code LIMIT {batch_size}"

        # ==================== 原有逻辑保持不变 ====================
        stock_batch = db.query(read_sql)
        if not stock_batch or len(stock_batch) == 0:
            logger.info("===== 全量更新完成！无更多股票 =====")
            break

        current_batch_size = len(stock_batch)
        total_stock += current_batch_size
        last_ts_code = stock_batch[-1]["ts_code"]

        logger.info(f"\n===== 批次[{batch_index}] | 累计处理：{total_stock} 只 =====")

        # 1. 调用API获取概念（原有逻辑）
        concept_result = get_stock_concepts_batch(stock_batch, batch_index)
        if concept_result.get("error") or not concept_result.get("data"):
            logger.error(f"批次[{batch_index}] 跳过，继续下一批")
            time.sleep(1)
            continue

        # 3. 批量更新 + 单条失败日志（满足你要求）
        params_list = [(item["concept_tags"], item["ts_code"]) for item in concept_result["data"]]
        success = batch_update_with_single_fail_log(params_list)
        total_success += success

        logger.info(f"批次[{batch_index}] 成功：{success}/{current_batch_size} | 总成功：{total_success}/{total_stock}")

        # 4. 防API限流
        time.sleep(1)

    logger.info(f"\n🎉 全量更新结束！总股票：{total_stock} | 成功：{total_success}")


if __name__ == "__main__":
    #默认update= True，只做增量更新没有标签的股票。
    update_all_stock_concept_tags(batch_size=20,update= True,ts_code=[])