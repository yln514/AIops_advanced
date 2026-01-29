# 文件路径: app/api.py
from flask import Flask, jsonify, render_template_string
import pandas as pd
import sys
import os

# 确保能找到 core 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import AIOpsEngine
from core.data_loader import DataLoader
from core.analysis import Analyzer

app = Flask(__name__)
engine = AIOpsEngine()

# 简单的 HTML 模板，用于展示图表
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AIops 异常检测仪表盘</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f2f5; }
        .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        img { max-width: 100%; height: auto; }
        .stats { display: flex; gap: 20px; }
        .stat-box { background: #007bff; color: white; padding: 15px; border-radius: 5px; flex: 1; text-align: center; }
        .btn { display: inline-block; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>电信异常话单检测系统</h1>
    <div class="stats">
        <div class="stat-box"><h3>总样本数</h3><p>{{ stats.total }}</p></div>
        <div class="stat-box"><h3>检测出异常</h3><p>{{ stats.anomalies }}</p></div>
        <div class="stat-box"><h3>异常率</h3><p>{{ stats.rate }}%</p></div>
    </div>

    <div class="card">
        <h2>时段趋势分析</h2>
        {% if charts.trend %}
            <img src="data:image/png;base64, {{ charts.trend }}" />
        {% else %}
            <p>暂无数据</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>异常分布分析</h2>
        {% if charts.pie %}
            <img src="data:image/png;base64, {{ charts.pie }}" />
        {% else %}
            <p>暂无数据</p>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/')
def index():
    """解决 404 问题，直接跳转到仪表盘"""
    return '<a href="/dashboard" class="btn">进入 AIops 仪表盘</a>'


@app.route('/dashboard')
def dashboard():
    try:
        # 1. 加载数据
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'telecom_bill_data.csv')
        loader = DataLoader(data_path)
        df_raw = loader.load_raw_data()

        # 2. 预测 (如果没有模型，会自动触发训练)
        # 为了演示，我们先转换数据
        X_train, _, _ = loader.fit_transform(df_raw)

        # 确保模型存在
        if not engine.model:
            print("模型未加载，开始训练...")
            engine.train_model(data_path)  # 假设 engine 有 train_model 方法

        # 预测
        preds = engine.predict_anomalies(df_raw)  # 假设 engine 有 predict_anomalies 方法

        # 3. 生成图表
        img_trend = Analyzer.plot_hourly_trend(df_raw, preds)
        img_pie = Analyzer.plot_root_cause_distribution(df_raw, preds)

        # 4. 统计信息
        stats = {
            "total": len(df_raw),
            "anomalies": int(sum(preds)),
            "rate": round(sum(preds) / len(df_raw) * 100, 2)
        }

        return render_template_string(HTML_TEMPLATE, stats=stats, charts={"trend": img_trend, "pie": img_pie})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"系统错误: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)