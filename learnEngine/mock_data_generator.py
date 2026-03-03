# learnEngine/mock_data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.common_tools import get_trade_dates  # 复用你项目已有的交易日工具

# ===================== 配置项：和你的策略完全对齐 =====================
MOCK_START_DATE = "20240101"  # 模拟数据起始日期
MOCK_END_DATE = "20241231"  # 模拟数据结束日期
TOP3_SECTOR_COUNT = 3  # 前3热点板块，和你的策略一致
DAY_OFFSET_RANGE = range(4, -1, -1)  # d4到d0，共5天，和你的预期一致
CANDIDATE_STOCKS_PER_DAY = 10  # 每个交易日模拟10只候选个股（你的策略筛选后的股票池）


def generate_mock_trade_dates() -> list:
    """生成模拟的交易日列表，复用你项目的工具"""
    try:
        return get_trade_dates(MOCK_START_DATE, MOCK_END_DATE)
    except:
        # 如果你项目的工具还没对接上，用备用逻辑生成交易日（周一到周五）
        start_dt = datetime.strptime(MOCK_START_DATE, "%Y%m%d")
        end_dt = datetime.strptime(MOCK_END_DATE, "%Y%m%d")
        all_dates = pd.date_range(start_dt, end_dt, freq="B")  # B=工作日
        return [dt.strftime("%Y%m%d") for dt in all_dates]


def generate_mock_sector_features() -> pd.DataFrame:
    """
    生成模拟的板块特征数据，严格匹配你的格式
    返回：每行=一个交易日的板块特征
    """
    trade_dates = generate_mock_trade_dates()
    sector_feature_list = []

    for trade_date in trade_dates:
        # 1. 模拟板块轮动分adapt_score
        adapt_score = np.random.uniform(10, 60)  # 10-60之间，和你的策略阈值40对齐

        # 2. 模拟3个板块×5天的赚亏钱效应，严格匹配你的字段名格式
        feature_row = {
            "trade_date": trade_date,
            "adapt_score": adapt_score
        }
        # 生成sector1_d4_profit、sector1_d4_loss...sector3_d0_loss
        for sector_idx in range(1, TOP3_SECTOR_COUNT + 1):
            for day_offset in DAY_OFFSET_RANGE:
                # 赚钱效应：0-100的随机数，模拟上涨个股占比
                feature_row[f"sector{sector_idx}_d{day_offset}_profit"] = np.random.uniform(20, 90)
                # 亏钱效应：0-100的随机数，和赚钱效应加起来接近100
                feature_row[f"sector{sector_idx}_d{day_offset}_loss"] = np.random.uniform(10, 80)

        sector_feature_list.append(feature_row)

    sector_feature_df = pd.DataFrame(sector_feature_list)
    print("=" * 50)
    print("✅ 模拟板块特征生成完成，示例数据：")
    print(sector_feature_df.head(1).T)  # 打印第一行的所有字段，让你看到格式
    print("=" * 50)
    return sector_feature_df


def generate_mock_stock_pool(trade_date: str, sector_feature_row: pd.Series) -> pd.DataFrame:
    """
    生成单个交易日的模拟候选股票池（你的策略筛选后的结果）
    返回：每行=一只候选个股的特征
    """
    stock_pool = []
    for stock_idx in range(CANDIDATE_STOCKS_PER_DAY):
        # 模拟个股代码
        ts_code = f"{np.random.randint(600000, 601000)}.SH" if np.random.rand() > 0.5 else f"{np.random.randint(0, 3000)}.SZ"
        # 模拟个股所属板块（1/2/3，对应3个热点板块）
        stock_sector = np.random.randint(1, TOP3_SECTOR_COUNT + 1)
        # 模拟个股因子
        stock_feature = {
            "trade_date": trade_date,
            "ts_code": ts_code,
            "stock_sector_encoded": stock_sector - 1,  # 模型需要0/1/2的编码
            "30d_gain": np.random.uniform(-20, 50),  # 30日涨跌幅，-20%到50%
            "has_limit_up_10d": np.random.randint(0, 2),  # 0=近10日无涨停，1=有涨停
            "amount": np.random.uniform(1e7, 1e9),  # 成交额1000万到10亿
            "pct_chg": np.random.uniform(-5, 10),  # 当日涨跌幅-5%到10%
            # 标签：模拟T+1日是否赚钱，1=涨，0=跌
            "label": np.random.randint(0, 2)
        }
        stock_pool.append(stock_feature)

    return pd.DataFrame(stock_pool)


def generate_full_mock_dataset() -> pd.DataFrame:
    """
    生成完整的训练数据集（板块特征+个股特征+标签合并）
    这就是最终喂给模型的训练数据
    """
    print("开始生成模拟训练数据集...")
    # 1. 生成所有交易日的板块特征
    sector_feature_df = generate_mock_sector_features()
    full_dataset = []

    # 2. 每个交易日生成候选股票池，合并板块特征
    for idx, row in sector_feature_df.iterrows():
        trade_date = row["trade_date"]
        # 生成当日候选股票池
        daily_stock_pool = generate_mock_stock_pool(trade_date, row)
        # 合并当日的板块特征到每只个股上
        daily_stock_pool = daily_stock_pool.merge(
            sector_feature_df[sector_feature_df["trade_date"] == trade_date],
            on="trade_date",
            how="left"
        )
        full_dataset.append(daily_stock_pool)

    # 合并所有交易日的数据
    final_train_df = pd.concat(full_dataset, ignore_index=True)
    # 填充缺失值
    final_train_df.fillna(0, inplace=True)

    print("✅ 完整模拟训练数据集生成完成！")
    print(f"数据集总行数：{len(final_train_df)}，总字段数：{len(final_train_df.columns)}")
    print("=" * 50)
    print("最终训练数据示例（前2行）：")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(final_train_df.head(2))
    print("=" * 50)

    # 保存到本地，方便后续使用
    final_train_df.to_csv("mock_train_data.csv", index=False)
    print("模拟数据已保存到：mock_train_data.csv")
    return final_train_df


# 直接运行这个文件就能生成模拟数据
if __name__ == "__main__":
    generate_full_mock_dataset()