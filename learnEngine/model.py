# learnEngine/model.py
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.log_utils import logger


class SectorHeatXGBModel:
    def __init__(self, model_save_path: str = "sector_heat_xgb_model.pkl"):
        self.model_save_path = model_save_path
        self.model = None
        # 【极简版】XGBoost参数，去掉所有可能报错的高级参数
        self.params = {
            "objective": "binary:logistic",  # 二分类：预测是否赚钱
            "max_depth": 4,  # 树深度，防止过拟合
            "learning_rate": 0.1,  # 学习率
            "n_estimators": 50,  # 树的数量（减少一点，避免过拟合）
            "random_state": 42,
            "verbosity": 0  # 0=不输出训练过程的冗余日志
        }

    def train(self, X_train, X_val, y_train, y_val, feature_cols: list):
        """【极简兼容版】训练模型，去掉所有可能报错的参数"""
        logger.info("开始训练XGBoost模型（极简兼容版）...")

        # 初始化模型
        self.model = xgb.XGBClassifier(**self.params)

        # 【核心修复】用最简单的fit方法，只传最基础的参数，保证兼容所有版本
        self.model.fit(X_train, y_train)

        # 验证模型效果
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_proba)

        logger.info("=" * 50)
        logger.info(f"✅ 模型训练完成！")
        logger.info(f"验证集准确率：{accuracy:.2%}（模型预测正确的比例）")
        logger.info(f"验证集AUC：{auc:.4f}（模型区分涨跌的能力，0.5以上就有效）")
        logger.info("=" * 50)

        # 输出特征重要性，告诉你哪些因子最有用
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        logger.info("【最有用的前5个因子】：")
        logger.info(feature_importance.head(5))

        # 保存模型
        self.save_model()
        return self.model

    def save_model(self):
        """保存训练好的模型到本地"""
        if self.model is None:
            logger.error("模型还未训练，无法保存")
            return
        with open(self.model_save_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存到：{self.model_save_path}")

    def load_model(self):
        """加载本地保存的模型，用于策略里的预测"""
        try:
            with open(self.model_save_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("模型加载成功！")
            return self.model
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise

    def predict_profit_prob(self, feature_df: pd.DataFrame) -> list:
        """
        预测个股赚钱的概率，用于策略里的筛选
        :param feature_df: 候选个股的特征DataFrame，列必须和训练时一致
        :return: 每只个股赚钱的概率（0-1之间，越高越容易赚钱）
        """
        if self.model is None:
            self.load_model()
        # 预测赚钱的概率（label=1的概率）
        profit_proba = self.model.predict_proba(feature_df)[:, 1]
        return profit_proba.tolist()