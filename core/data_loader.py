import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class DataLoader:
    def __init__(self, file_path=None, artifact_dir='models/artifacts'):
        """
        初始化数据加载器
        :param file_path: CSV 数据文件路径 (训练时必填)
        :param artifact_dir: 存放预处理工具（Scaler/Encoder）的目录
        """
        self.file_path = file_path
        self.artifact_dir = artifact_dir

        # 定义关键列名配置（方便后期维护）
        self.numeric_features = ['call_duration', 'cost', 'signal_strength', 'drop_rate']
        self.cat_features = ['base_station']
        self.train_features = self.numeric_features + ['base_station_encoded', 'create_hour']

        # 预处理器容器
        self.scaler = StandardScaler()
        # 修改点 1: 改用 OneHotEncoder
        # sparse_output=False: 让它返回数组而不是稀疏矩阵，方便转 DataFrame
        # handle_unknown='ignore': 生产环境神器！遇到没见过的基站，自动设为全0，程序不报错
        self.ohe_base_station = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)

        # 用于存储最终进入模型的所有特征列名（因为 OHE 会产生动态数量的列）
        self.final_feature_names = []

        self.fill_values = {
            'cost': 0.0,
            'base_station': 'Unknown'
        }

        # 确保保存目录存在
        os.makedirs(self.artifact_dir, exist_ok=True)

    def load_raw_data(self):
        """加载原始 CSV 数据"""
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"数据文件未找到: {self.file_path}")

        df = pd.read_csv(self.file_path, encoding='utf-8')
        return df

    def _clean_data(self, df):
        """
        通用清洗逻辑：处理去重、异常值过滤
        注意：这里只做行删除操作，不做列值的填充（填充逻辑在 transform 里）
        """
        original_len = len(df)

        # 1. 去重 (业务主键)
        # 仅在有这些字段时才去重
        subset_cols = ['user_id', 'create_hour', 'call_duration']
        if set(subset_cols).issubset(df.columns):
            df = df.drop_duplicates(subset=subset_cols)

        # 2. 过滤显式错误 (如负通话时长)
        if 'call_duration' in df.columns:
            df = df[df['call_duration'] >= 0]

        # 3. 缺失值处理策略
        # 逻辑：如果是训练阶段，我们稍后会计算均值；如果是预测阶段，我们稍后会用保存的均值填充
        # 这里只删除那些无法填充的关键字段缺失的行（如果有的话）

        return df

    def fit_transform(self, df):
        """
        【训练阶段使用】
        1. 清洗数据
        2. 学习统计量（均值、方差、编码映射）
        3. 保存这些“标尺”
        4. 返回处理后的 X 矩阵和 y (如果有)
        """
        df = self._clean_data(df.copy())

        # --- 学习并填充缺失值 ---
        # 1. Cost 均值
        self.fill_values['cost'] = df['cost'].mean()
        df['cost'] = df['cost'].fillna(self.fill_values['cost'])

        # 2. Base Station 众数
        if not df['base_station'].mode().empty:
            self.fill_values['base_station'] = df['base_station'].mode()[0]
        df['base_station'] = df['base_station'].fillna(self.fill_values['base_station'])

        # 删除仍然有空值的行
        df.dropna(inplace=True)

        # --- 修改点 2: One-Hot 编码处理 ---
        # fit_transform 需要二维数组，所以用 [[ ]]
        ohe_array = self.ohe_base_station.fit_transform(df[['base_station']])

        # 获取生成的列名 (例如: base_station_A, base_station_B ...)
        ohe_columns = self.ohe_base_station.get_feature_names_out(['base_station'])

        # 将 OHE 结果转为 DataFrame 并合并回原数据
        df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns, index=df.index)
        df = pd.concat([df, df_ohe], axis=1)

        # --- 数值标准化 ---
        df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])

        # --- 确定最终特征列表 ---
        # 最终特征 = 数值特征 + OHE生成的基站特征 + create_hour
        self.final_feature_names = self.numeric_features + list(ohe_columns) + ['create_hour']

        # --- 保存状态 ---
        self._save_artifacts()

        # --- 准备返回 ---
        X = df[self.train_features]
        # 如果有标签则返回标签，用于监督学习或评估
        y = (df['bill_status'] != '正常').astype(int) if 'bill_status' in df.columns else None

        return X, y, df  # 返回 df 是为了方便后续做 EDA 分析图表

    def transform(self, df):
        """【预测阶段】使用已保存的标尺转换"""
        if not hasattr(self.scaler, 'mean_'):
            self._load_artifacts()

        df = self._clean_data(df.copy())

        # 1. 填充
        if 'cost' in df.columns:
            df['cost'] = df['cost'].fillna(self.fill_values['cost'])
        if 'base_station' in df.columns:
            df['base_station'] = df['base_station'].fillna(self.fill_values['base_station'])

        # --- 修改点 3: One-Hot 编码转换 ---
        # 注意：这里直接调用 transform，遇到新基站会根据 handle_unknown='ignore' 处理
        ohe_array = self.ohe_base_station.transform(df[['base_station']])
        ohe_columns = self.ohe_base_station.get_feature_names_out(['base_station'])

        df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns, index=df.index)
        df = pd.concat([df, df_ohe], axis=1)

        # 2. 标准化
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])

        # 3. 按严格顺序返回列 (防止列顺序错乱导致模型预测错误)
        # 如果新数据缺少某些列（理论上不会，因为OHE补全了），这里会报错提醒
        return df[self.final_feature_names]

    def _save_artifacts(self):
        joblib.dump(self.scaler, os.path.join(self.artifact_dir, 'scaler.pkl'))
        # 保存 OHE 对象
        joblib.dump(self.ohe_base_station, os.path.join(self.artifact_dir, 'ohe_base.pkl'))
        joblib.dump(self.fill_values, os.path.join(self.artifact_dir, 'fill_values.pkl'))
        # 必须保存最终的特征列名列表，因为预测时需要对齐
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