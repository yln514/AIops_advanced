import joblib
import os
from sklearn.ensemble import IsolationForest
from core.data_loader import DataLoader


class AIOpsEngine:
    def __init__(self, model_path='models/iso_forest.pkl'):
        self.model_path = model_path
        self.model = None

    def train_model(self, data_path):
        """
        全流程训练：加载数据 -> 清洗 -> 训练 -> 保存
        """
        print(f"正在加载数据: {data_path}...")

        # 1. 实例化 Loader (传入文件路径)
        loader = DataLoader(file_path=data_path)

        # 2. 获取处理好的特征矩阵 X (会自动保存 scaler/ohe 到 models/artifacts/)
        # 注意：fit_transform 返回 X, y, df，我们只需要 X
        X_train, _, _ = loader.fit_transform(loader.load_raw_data())

        print(f"数据预处理完成，特征维度: {X_train.shape}")

        # 3. 训练模型
        self.model = IsolationForest(n_estimators=100, contamination=0.1, n_jobs=-1, random_state=42)
        self.model.fit(X_train)

        # 4. 保存模型
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"模型已保存至: {self.model_path}")

        return "训练成功"

    def predict_anomalies(self, df_input):
        """
        预测逻辑：接收 DataFrame -> 转换 -> 预测
        """
        # 1. 加载模型
        if not self.model:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise Exception("模型文件不存在，请先训练！")

        # 2. 数据转换 (使用 DataLoader)
        # 注意：这里不需要传入 file_path，只需加载 artifact
        loader = DataLoader()
        X_pred = loader.transform(df_input)

        # 3. 预测 (-1异常, 1正常 -> 转为 1异常, 0正常)
        preds = self.model.predict(X_pred)
        return (preds == -1).astype(int)