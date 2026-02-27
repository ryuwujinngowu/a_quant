# learnEngine/model.py
"""
板块热度策略XGBoost模型核心类
功能：模型训练、评估、保存、加载、胜率推理（无sklearn依赖）
"""
import os
import pickle
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from utils.log_utils import logger


class SectorHeatXGBModel:
    """XGBoost二分类模型（预测D+1日上涨胜率）"""

    def __init__(self, model_config: dict = None):
        # 模型默认配置（仅XGBoost原生参数，无sklearn依赖）
        self.default_config = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss"  # XGBoost原生评估指标
        }
        self.config = model_config if model_config else self.default_config
        self.model = XGBClassifier(**self.config)
        # 模型保存路径（从配置读取）
        self.model_save_path =  "models/sector_heat_xgb.pkl"

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None
    ) -> Dict[str, float]:
        """
        模型训练（无sklearn依赖，手动计算评估指标）
        :param X_train: 训练特征矩阵
        :param y_train: 训练标签
        :param X_val: 验证集特征（可选）
        :param y_val: 验证集标签（可选）
        :return: 训练评估指标
        """
        # ==================== XGBoost原生训练逻辑 ====================
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # 模型拟合（XGBoost原生接口，无需sklearn）
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=20 if eval_set else None
        )

        # ==================== 手动计算评估指标（无sklearn） ====================
        # 1. 训练集指标
        y_train_pred = self.model.predict(X_train)
        train_accuracy = self._calc_accuracy(y_train, y_train_pred)
        train_auc = self._calc_auc(y_train, self.model.predict_proba(X_train)[:, 1])

        # 2. 验证集指标（可选）
        val_accuracy = 0.0
        val_auc = 0.0
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = self._calc_accuracy(y_val, y_val_pred)
            val_auc = self._calc_auc(y_val, self.model.predict_proba(X_val)[:, 1])

        # 评估结果
        eval_result = {
            "train_accuracy": round(train_accuracy, 4),
            "train_auc": round(train_auc, 4),
            "val_accuracy": round(val_accuracy, 4),
            "val_auc": round(val_auc, 4)
        }
        logger.info(f"模型训练完成，评估指标：{eval_result}")
        return eval_result

    def predict_win_rate(self, X_infer: np.ndarray) -> np.ndarray:
        """
        预测个股上涨胜率（核心：策略运行时调用）
        :param X_infer: 推理特征矩阵
        :return: 每只个股的上涨胜率（0-1之间）
        """
        if self.model is None:
            logger.error("模型未加载/训练，无法预测")
            return np.array([])
        # XGBoost原生predict_proba，无sklearn依赖
        win_rate = self.model.predict_proba(X_infer)[:, 1]
        return win_rate

    def save_model(self, save_path: str = None) -> None:
        """保存模型到本地（pickle原生，无sklearn）"""
        save_path = save_path if save_path else self.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存到：{save_path}")

    def load_model(self, model_path: str = None) -> bool:
        """加载本地模型（策略初始化时调用）"""
        model_path = model_path if model_path else self.model_save_path
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在：{model_path}")
            return False
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"模型加载成功：{model_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败：{str(e)}")
            return False

    # ========== 手动实现评估指标（无sklearn依赖） ==========
    @staticmethod
    def _calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """手动计算准确率（Accuracy）"""
        if len(y_true) == 0:
            return 0.0
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def _calc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        手动极简计算AUC（无sklearn依赖，适配二分类）
        注：仅核心逻辑，后续可根据需求优化精度
        """
        if len(y_true) == 0:
            return 0.0
        # 排序后计算AUC核心值
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        pos_count = np.sum(y_true == 1)
        neg_count = len(y_true) - pos_count

        if pos_count == 0 or neg_count == 0:
            return 0.5  # 无正/负样本时返回0.5

        # 计算累计正样本数
        cum_pos = np.cumsum(y_true_sorted)
        auc = (np.sum(cum_pos[y_true_sorted == 0]) + 0.5 * np.sum(cum_pos[y_true_sorted == 1])) / (
                    pos_count * neg_count)
        return auc