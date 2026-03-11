"""
模型训练入口 (train.py)
========================
运行方式：python train.py

前置条件：
    已运行 python learnEngine/dataset.py 生成 train_dataset.csv

流程：
    1. 加载训练集 CSV
    2. 数据预处理（清洗、特征/标签分离）
    3. 时间序列切分 train / val（避免未来数据泄漏）
    4. 训练 XGBoost 模型
    5. 输出评估指标 + 特征重要性
    6. 保存模型
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from learnEngine.model import SectorHeatXGBModel
from utils.log_utils import logger


# ============================================================
# 可配置参数
# ============================================================
TRAIN_CSV_PATH   = os.path.join(os.getcwd(), "learnEngine/train_dataset_2026-02-02.csv")
MODEL_SAVE_PATH  = os.path.join(os.getcwd(), "sector_heat_xgb_model.pkl")
TARGET_LABEL     = "label1"       # 训练目标：label1 (日内 5% 收益) 或 label2 (隔夜高开)
VAL_RATIO        = 0.2            # 验证集占比（按时间序列尾部切分）

# 需要排除的非特征列（主键 + 标签 + 辅助信息）
EXCLUDE_COLS = [
    "stock_code", "trade_date",
    "label1", "label2",
    "sector_name", "top3_sectors",
]


# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_prepare(csv_path: str, target_label: str):
    """
    加载训练集 CSV 并预处理

    :return: (X, y, feature_cols, df) — 特征矩阵、标签、特征列名、原始 DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"训练集文件不存在: {csv_path}\n请先运行 python learnEngine/dataset.py 生成训练集")

    df = pd.read_csv(csv_path)
    logger.info(f"加载训练集: {csv_path} | 行数: {len(df)} | 列数: {len(df.columns)}")

    # 检查标签列
    if target_label not in df.columns:
        raise ValueError(f"目标标签列 '{target_label}' 不存在于训练集中，可用列: {df.columns.tolist()}")

    # 删除标签缺失的行
    before = len(df)
    df = df.dropna(subset=[target_label])
    if len(df) < before:
        logger.info(f"删除 {target_label} 缺失行: {before - len(df)}")

    # 去重
    dup = df.duplicated(subset=["stock_code", "trade_date"]).sum()
    if dup > 0:
        df = df.drop_duplicates(subset=["stock_code", "trade_date"])
        logger.info(f"移除重复行: {dup}")

    # 分离特征与标签
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # 仅保留数值型列
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df[target_label].astype(int).values

    # 缺失值填 0（与 DataSetAssembler 一致）
    X = X.fillna(0)

    # inf 替换为 0
    X = X.replace([np.inf, -np.inf], 0)

    logger.info(f"特征列数: {len(feature_cols)} | 正样本率: {y.mean():.2%}")
    return X, y, feature_cols, df


# ============================================================
# 时间序列切分（严格按 trade_date 排序，避免未来数据泄漏）
# ============================================================
def time_series_split(X, y, df, val_ratio: float):
    """
    按 trade_date 排序后尾部切分验证集（不随机打乱）

    :return: (X_train, X_val, y_train, y_val)
    """
    # 按 trade_date 排序
    dates = df["trade_date"].values
    sort_idx = np.argsort(dates)
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]

    split_point = int(len(X) * (1 - val_ratio))
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y[:split_point], y[split_point:]

    # 获取切分日期信息
    sorted_dates = dates[sort_idx]
    train_end   = sorted_dates[split_point - 1] if split_point > 0 else "N/A"
    val_start   = sorted_dates[split_point]      if split_point < len(dates) else "N/A"

    logger.info(
        f"时间序列切分 | 训练集: {len(X_train)} 行 (截至 {train_end}) | "
        f"验证集: {len(X_val)} 行 (从 {val_start} 起)"
    )
    logger.info(f"训练集正样本率: {y_train.mean():.2%} | 验证集正样本率: {y_val.mean():.2%}")

    return X_train, X_val, y_train, y_val


# ============================================================
# 详细评估
# ============================================================
def evaluate_model(model, X_val, y_val, feature_cols):
    """输出详细评估指标"""
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc       = accuracy_score(y_val, y_pred)
    auc       = roc_auc_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall    = recall_score(y_val, y_pred, zero_division=0)

    logger.info("=" * 60)
    logger.info("模型评估结果")
    logger.info("=" * 60)
    logger.info(f"  准确率 (Accuracy):  {acc:.4f}")
    logger.info(f"  AUC:                {auc:.4f}")
    logger.info(f"  精确率 (Precision): {precision:.4f}")
    logger.info(f"  召回率 (Recall):    {recall:.4f}")
    logger.info("")

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    logger.info("混淆矩阵:")
    logger.info(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    logger.info(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    logger.info("")

    # 分类报告
    logger.info("分类报告:")
    logger.info("\n" + classification_report(y_val, y_pred, target_names=["不买", "买入"]))

    # 特征重要性 Top 20
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info("Top 20 重要特征:")
    for i, row in fi.head(20).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return {"accuracy": acc, "auc": auc, "precision": precision, "recall": recall}


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    logger.info("=" * 60)
    logger.info("开始模型训练")
    logger.info("=" * 60)

    # 1. 加载数据
    X, y, feature_cols, df = load_and_prepare(TRAIN_CSV_PATH, TARGET_LABEL)

    if len(X) < 50:
        logger.error(f"训练集样本不足（{len(X)} 行），至少需要 50 行，请扩大日期范围重新生成训练集")
        sys.exit(1)

    # 2. 时间序列切分
    X_train, X_val, y_train, y_val = time_series_split(X, y, df, VAL_RATIO)

    # 3. 训练模型
    xgb_model = SectorHeatXGBModel(model_save_path=MODEL_SAVE_PATH)
    xgb_model.train(X_train, X_val, y_train, y_val, feature_cols)

    # 4. 详细评估
    metrics = evaluate_model(xgb_model.model, X_val, y_val, feature_cols)

    # 5. 完成
    logger.info("=" * 60)
    logger.info(f"训练完成！模型已保存至: {MODEL_SAVE_PATH}")
    logger.info(f"  训练集: {len(X_train)} 行 | 验证集: {len(X_val)} 行")
    logger.info(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    logger.info("=" * 60)
