from flask import Flask, request, jsonify
import pandas as pd

# === 关键引用 ===
# 虽然 Flask 不直接操作 DataLoader，但它需要 Engine，
# 而 Engine 内部已经封装了 DataLoader 的调用，所以这里只需要引 Engine
from core.engine import AIOpsEngine

app = Flask(__name__)
engine = AIOpsEngine()

# 假设数据文件就在本地 (生产环境可能是上传的)
DATA_PATH = "data/telecom_bill_data.csv"


@app.route('/api/train', methods=['POST'])
def train():
    try:
        # 直接调用 Engine 的训练方法
        msg = engine.train_model(DATA_PATH)
        return jsonify({"status": "success", "message": msg})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    接收前端传来的单条或多条数据 (JSON格式)
    """
    try:
        # 1. 获取 JSON 数据并转为 DataFrame
        input_json = request.json
        if not input_json:
            return jsonify({"error": "No data provided"}), 400

        # 假设前端传的是列表 [{'call_duration': 100, ...}, {...}]
        df_input = pd.DataFrame(input_json)

        # 2. 调用 Engine 进行预测
        # Engine 内部会自动调用 DataLoader.transform
        result = engine.predict_anomalies(df_input)

        return jsonify({
            "status": "success",
            "anomalies": result.tolist()  # numpy array 转 list
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)