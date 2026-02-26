# learnEngine/model.py
import xgboost as xgb

class LimitUpModel:
    """
    机器学习模型：预测涨停板次日赚钱概率
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]