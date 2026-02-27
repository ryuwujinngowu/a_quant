#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块热度XGB模型训练独立入口
职责：仅处理模型训练，不涉及回测逻辑
流程：拉取历史数据 → 生成标签 → 构建数据集 → 训练模型 → 保存模型
"""
import pandas as pd
from utils.log_utils import logger

# 数据层
from data.data_fetcher import DataFetcher

# 机器学习层
from learnEngine.label import LabelGenerator
from learnEngine.dataset import DatasetBuilder
from learnEngine.model import SectorHeatXGBModel


def train_sector_heat_model():
    """核心：板块热度XGB模型训练主逻辑"""
    logger.info("=" * 60)
    logger.info("开始训练【板块热度选股XGB模型】")
    logger.info("=" * 60)

    # ===================== 训练参数配置（可根据需求调整） =====================
    TRAIN_START_DATE = "2023-01-01"  # 训练数据开始日期
    TRAIN_END_DATE = "2024-12-31"    # 训练数据结束日期
    MODEL_SAVE_PATH = "models/sector_heat_xgb.pkl"  # 模型保存路径

    # ===================== 1. 拉取历史日线数据 =====================
    try:
        data_fetcher = DataFetcher()
        # 替换为你实际的日线数据拉取逻辑（后续填充真实代码）
        # daily_df = data_fetcher.fetch_daily_data(
        #     start_date=TRAIN_START_DATE,
        #     end_date=TRAIN_END_DATE
        # )
        daily_df = pd.DataFrame()  # 骨架占位，后续替换为真实数据
        if daily_df.empty:
            logger.error("训练数据为空，终止模型训练！")
            return
        logger.info(f"✅ 成功拉取训练数据：{len(daily_df)} 条")
    except Exception as e:
        logger.error(f"❌ 拉取训练数据失败：{str(e)}")
        return

    # ===================== 2. 生成D+1收益标签 =====================
    try:
        label_gen = LabelGenerator()
        label_df = label_gen.generate_label(
            daily_df=daily_df,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE
        )
        if label_df.empty:
            logger.error("标签生成失败，终止模型训练！")
            return
        logger.info(f"✅ 成功生成标签：{len(label_df)} 条")
    except Exception as e:
        logger.error(f"❌ 生成标签失败：{str(e)}")
        return

    # ===================== 3. 构建训练数据集 =====================
    try:
        dataset_builder = DatasetBuilder()
        X_train, y_train, train_df = dataset_builder.build_train_dataset(
            daily_df=daily_df,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE
        )
        if X_train.size == 0 or len(y_train) == 0:
            logger.error("训练数据集为空，终止模型训练！")
            return
        logger.info(f"✅ 成功构建训练数据集：样本数={len(y_train)}，特征数={X_train.shape[1]}")
    except Exception as e:
        logger.error(f"❌ 构建数据集失败：{str(e)}")
        return

    # ===================== 4. 训练XGB模型 =====================
    try:
        model = SectorHeatXGBModel()
        eval_result = model.train(X_train, y_train)
        logger.info(f"✅ 模型训练完成，评估指标：{eval_result}")
    except Exception as e:
        logger.error(f"❌ 训练模型失败：{str(e)}")
        return

    # ===================== 5. 保存模型 =====================
    try:
        model.save_model(save_path=MODEL_SAVE_PATH)
        logger.info(f"✅ 模型已保存至：{MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"❌ 保存模型失败：{str(e)}")
        return

    logger.info("=" * 60)
    logger.info("【板块热度选股XGB模型】训练完成 ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 直接运行该脚本即可启动模型训练
    train_sector_heat_model()