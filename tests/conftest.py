# tests/conftest.py — pytest 共享 fixtures
import os
import sys

# 确保项目根目录在 sys.path 中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置测试环境变量，避免模块导入时报错
os.environ.setdefault("DEEPSEEK_API_KEY", "test-mock-key")
