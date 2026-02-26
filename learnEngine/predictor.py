# learnEngine/predictor.py
import pandas as pd

class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, df_features):
        feature_cols = ["limit_up_count", "limit_down_count"]
        X = df_features[feature_cols]
        return self.model.predict_proba(X)