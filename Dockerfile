# Miemie-Agent-RAG Dockerfile
# ============================================================
# 构建:  docker build -t miemie-rag-app .
# 运行:  docker run -p 8000:8000 --env-file .env miemie-rag-app
# ============================================================

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# --- 系统依赖 ---
# libgomp1: torch 运行所需
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python 依赖层 (利用 Docker 缓存) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 项目代码 ---
COPY . .

# --- 预下载 Embedding 模型 (可选，加速首次启动) ---
# 如需使用 HuggingFace 镜像加速，取消下一行注释:
# ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')"

# --- 运行时配置 ---
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
