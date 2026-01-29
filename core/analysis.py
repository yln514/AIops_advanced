import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import platform

# 防止中文乱码配置
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Analyzer:
    @staticmethod
    def _fig_to_base64(fig):
        """将 Matplotlib 图片转换为 Web 可显示的 Base64 编码"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)  # 关键：释放内存
        return img_base64

    @staticmethod
    def plot_root_cause_distribution(df, anomalies_pred):
        """画饼图：异常根因分布"""
        # 假设预测为 1 是异常
        df_anomalies = df[anomalies_pred == 1]

        # 如果数据中没有 root_cause 字段，就画基站分布代替
        target_col = 'root_cause' if 'root_cause' in df.columns else 'base_station'

        if len(df_anomalies) == 0:
            return None  # 没有异常，不画图

        counts = df_anomalies[target_col].value_counts().head(10)  # 只取前10防止太挤

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title(f'异常分布 ({target_col})')

        return Analyzer._fig_to_base64(fig)

    @staticmethod
    def plot_hourly_trend(df, anomalies_pred):
        """画折线图：每小时异常数量趋势"""
        df_temp = df.copy()
        df_temp['is_anomaly'] = anomalies_pred

        # 统计每小时异常数
        hourly_counts = df_temp[df_temp['is_anomaly'] == 1].groupby('create_hour').size()

        # 补全 0-23 小时
        all_hours = pd.Series(0, index=range(24))
        hourly_counts = all_hours.add(hourly_counts, fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hourly_counts.index, hourly_counts.values, marker='o', color='#d62728', linestyle='-')
        ax.set_title('每小时异常趋势')
        ax.set_xlabel('小时 (0-23)')
        ax.set_ylabel('异常话单数量')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(range(0, 24))

        return Analyzer._fig_to_base64(fig)