# train_sector_heat_model.py
from learnEngine import (
    generate_full_mock_dataset,
    load_and_process_dataset,
    split_time_series_dataset,
    SectorHeatXGBModel
)
from utils.log_utils import logger

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("开始板块热度策略XGBoost模型训练全流程")
    logger.info("="*60)

    # ===================== 步骤1：生成模拟训练数据（你后续替换成真实数据） =====================
    generate_full_mock_dataset()

    # ===================== 步骤2：加载并预处理数据集 =====================
    X, y, feature_cols = load_and_process_dataset("mock_train_data.csv")

    # ===================== 步骤3：时间序列划分训练集/验证集 =====================
    X_train, X_val, y_train, y_val = split_time_series_dataset(X, y)

    # ===================== 步骤4：训练模型 =====================
    model = SectorHeatXGBModel()
    model.train(X_train, X_val, y_train, y_val, feature_cols)

    logger.info("="*60)
    logger.info("🎉 模型训练全流程完成！你已经成功跑通了机器学习的整个流程")
    logger.info("="*60)