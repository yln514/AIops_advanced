import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class DataLoader:
    def __init__(self, file_path=None, artifact_dir='models/artifacts'):
        self.file_path = file_path
        self.artifact_dir = artifact_dir

        # 定义关键列名配置
        self.numeric_features = ['call_duration', 'cost', 'signal_strength', 'drop_rate']
        self.cat_features = ['base_station']

        # [修改点 1] 删除或注释掉这行旧代码，因为它包含 'base_station_encoded'
        # self.train_features = self.numeric_features + ['base_station_encoded', 'create_hour']

        # 预处理器
        self.scaler = StandardScaler()
        self.ohe_base_station = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)

        # 用于存储最终特征列名
        self.final_feature_names = []

        self.fill_values = {
            'cost': 0.0,
            'base_station': 'Unknown'
        }

        os.makedirs(self.artifact_dir, exist_ok=True)

    # ... load_raw_data 和 _clean_data 保持不变 ...
    def load_raw_data(self):
        """加载原始 CSV 数据"""
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"数据文件未找到: {self.file_path}")

        df = pd.read_csv(self.file_path, encoding='utf-8')
        return df

    def _clean_data(self, df):
        """通用清洗逻辑"""
        # 1. 去重
        subset_cols = ['user_id', 'create_hour', 'call_duration']
        if set(subset_cols).issubset(df.columns):
            df = df.drop_duplicates(subset=subset_cols)

        # 2. 过滤显式错误
        if 'call_duration' in df.columns:
            df = df[df['call_duration'] >= 0]

        return df

    def fit_transform(self, df):
        """【训练阶段使用】"""
        df = self._clean_data(df.copy())

        # --- 填充缺失值 ---
        self.fill_values['cost'] = df['cost'].mean()
        df['cost'] = df['cost'].fillna(self.fill_values['cost'])

        if not df['base_station'].mode().empty:
            self.fill_values['base_station'] = df['base_station'].mode()[0]
        df['base_station'] = df['base_station'].fillna(self.fill_values['base_station'])

        df.dropna(inplace=True)

        # --- One-Hot 编码 ---
        ohe_array = self.ohe_base_station.fit_transform(df[['base_station']])
        ohe_columns = self.ohe_base_station.get_feature_names_out(['base_station'])

        df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns, index=df.index)
        df = pd.concat([df, df_ohe], axis=1)

        # --- 数值标准化 ---
        df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])

        # --- 更新最终特征列表 ---
        # 这一步是动态生成的，包含了 base_station_A, base_station_B 等新列名
        self.final_feature_names = self.numeric_features + list(ohe_columns) + ['create_hour']

        # --- 保存状态 ---
        self._save_artifacts()

        # --- [修改点 2] 关键修复！使用 final_feature_names 而不是 train_features ---
        # 之前报错就是因为这里用了旧的 train_features (含 base_station_encoded)
        X = df[self.final_feature_names]

        y = (df['bill_status'] != '正常').astype(int) if 'bill_status' in df.columns else None

        return X, y, df

    def transform(self, df):
        """【预测阶段使用】"""
        if not hasattr(self.scaler, 'mean_'):
            self._load_artifacts()

        df = self._clean_data(df.copy())

        # 1. 填充
        if 'cost' in df.columns:
            df['cost'] = df['cost'].fillna(self.fill_values['cost'])
        if 'base_station' in df.columns:
            df['base_station'] = df['base_station'].fillna(self.fill_values['base_station'])

        # 2. OHE 转换
        ohe_array = self.ohe_base_station.transform(df[['base_station']])
        ohe_columns = self.ohe_base_station.get_feature_names_out(['base_station'])

        df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns, index=df.index)
        df = pd.concat([df, df_ohe], axis=1)

        # 3. 标准化
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])

        # 4. 返回
        # 这里你之前写的是对的，使用了 final_feature_names
        return df[self.final_feature_names]

    # ... _save_artifacts 和 _load_artifacts 保持不变 ...
    def _save_artifacts(self):
        joblib.dump(self.scaler, os.path.join(self.artifact_dir, 'scaler.pkl'))
        joblib.dump(self.ohe_base_station, os.path.join(self.artifact_dir, 'ohe_base.pkl'))
        joblib.dump(self.fill_values, os.path.join(self.artifact_dir, 'fill_values.pkl'))
        joblib.dump(self.final_feature_names, os.path.join(self.artifact_dir, 'feature_names.pkl'))
        print("预处理工具已保存 (OneHotEncoder版)。")

    def _load_artifacts(self):
        try:
            self.scaler = joblib.load(os.path.join(self.artifact_dir, 'scaler.pkl'))
            self.ohe_base_station = joblib.load(os.path.join(self.artifact_dir, 'ohe_base.pkl'))
            self.fill_values = joblib.load(os.path.join(self.artifact_dir, 'fill_values.pkl'))
            self.final_feature_names = joblib.load(os.path.join(self.artifact_dir, 'feature_names.pkl'))
        except FileNotFoundError:
            raise Exception("未找到预处理模型文件，请先执行训练 (fit_transform)！")