import sys
import os

# 1. 将项目根目录添加到 python 搜索路径
# 这样 Python 才能找到 'core' 和 'app' 包
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 2. 从 app 包中导入 Flask 实例
from app.api import app

if __name__ == '__main__':
    print(f"AIops 服务正在启动...")
    print(f"项目根路径: {project_root}")
    print(f"请在浏览器访问: http://127.0.0.1:5000/dashboard")

    # 启动 Flask 开发服务器
    app.run(host='0.0.0.0', port=5000, debug=True)