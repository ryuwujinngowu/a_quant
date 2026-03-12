# learnEngine/model.py
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.log_utils import logger


class SectorHeatXGBModel:
    """
    XGBoost 二分类模型（金融量化调参版）
    ========================================
    核心参数选择依据：
      - scale_pos_weight : A 股标签高度不平衡（正样本 5-10%），动态按 neg/pos 比例设置
      - n_estimators=500 + early_stopping_rounds=50 : 足够多的树 + 提前停止防过拟合
      - learning_rate=0.05 : 低学习率配合大树数，泛化更好
      - max_depth=4 : 浅树防止对噪声过拟合（金融特征信噪比低）
      - subsample/colsample_bytree=0.8 : 行/列随机采样，Dropout 效果
      - min_child_weight=5 : 叶节点最小样本数，避免对小样本过拟合
      - gamma=0.1 : 分裂所需最小增益，让树只在有显著区分度时才分裂
      - reg_alpha/reg_lambda : L1+L2 正则化
      - eval_metric="auc" : 不平衡数据下 AUC 远比 accuracy 可靠
    """

    def __init__(self, model_save_path: str = "learnEngine/models/sector_heat_xgb_model.pkl"):
        self.model_save_path = model_save_path
        self.model = None
        # scale_pos_weight 在 train() 中根据实际数据动态计算
        self.base_params = {
            "objective":           "binary:logistic",
            "eval_metric":         "auc",
            "max_depth":           4,
            "learning_rate":       0.05,
            "n_estimators":        500,        # 配合 early_stopping 使用
            "early_stopping_rounds": 50,        # 【核心修复】移到这里，兼容新版本XGBoost
            "subsample":           0.8,        # 行采样，防过拟合
            "colsample_bytree":    0.8,        # 列采样，防过拟合
            "min_child_weight":    5,          # 叶节点最小样本数
            "gamma":               0.1,        # 分裂最小增益
            "reg_alpha":           0.1,        # L1 正则
            "reg_lambda":          1.0,        # L2 正则
            "n_jobs":              -1,         # 全核心并行
            "random_state":        42,
            "verbosity":           0,
        }

    def train(self, X_train, X_val, y_train, y_val, feature_cols: list):
        """训练模型，动态计算 scale_pos_weight，启用 early stopping"""
        # ── 动态计算正负样本比（解决 A 股标签高度不平衡问题）──────────────
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = round(neg / pos, 2) if pos > 0 else 1.0
        scale_pos_weight = min(scale_pos_weight,4.0)
        logger.info(
            f"训练集样本分布 | 正样本(买入):{pos} 负样本:{neg} "
            f"→ scale_pos_weight={scale_pos_weight}"
        )

        params = {**self.base_params, "scale_pos_weight": scale_pos_weight}
        self.model = xgb.XGBClassifier(**params)

        # ── 训练（eval_set 用于 early stopping 监控 AUC）──────────────────
        logger.info("开始训练 XGBoost 模型...")
        # 【核心修复】fit()里不再传early_stopping_rounds，只保留eval_set和verbose
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        actual_trees = self.model.best_iteration + 1 if hasattr(self.model, "best_iteration") else params["n_estimators"]
        logger.info(f"实际训练轮数: {actual_trees} / {params['n_estimators']}")

        # ── 评估 ──────────────────────────────────────────────────────────
        y_val_pred  = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        accuracy    = accuracy_score(y_val, y_val_pred)
        auc         = roc_auc_score(y_val, y_val_proba)

        logger.info("=" * 50)
        logger.info(f"模型训练完成")
        logger.info(f"验证集准确率: {accuracy:.2%}")
        logger.info(f"验证集 AUC:   {auc:.4f}")
        logger.info("=" * 50)

        feature_importance = pd.DataFrame({
            "feature":    feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info("Top 5 重要因子:")
        logger.info(feature_importance.head(5).to_string(index=False))

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