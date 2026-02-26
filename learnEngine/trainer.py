# learnEngine/trainer.py
import pandas as pd


class Trainer:
    def __init__(self, model, dataset, label_generator):
        self.model = model
        self.dataset = dataset
        self.label_generator = label_generator

    def run(self, df_with_features: pd.DataFrame):
        # 1. 打标签
        df = self.label_generator.generate(df_with_features)

        # 2. 划分训练/测试
        train_df, test_df = self.dataset.split_train_test(df)

        # 3. 确定特征列（排除非数字列）
        feature_cols = [
            "limit_up_count",  # 你刚写的特征
            "limit_down_count"  # 你刚写的特征
        ]

        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]

        # 4. 训练
        self.model.fit(X_train, y_train)

        print("✅ 机器学习模型训练完成")
        return self.model