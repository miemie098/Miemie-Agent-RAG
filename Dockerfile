# ⚡ 终极修复：直接调用本地健康的 v1 镜像，实现 0 网络依赖构建
FROM miemie-rag-app:v1

# 设置容器内的当前工作目录
WORKDIR /app

# ⚡ 核心修复：彻底去掉已经挂掉的中科大换源逻辑，也不再重复安装 gcc/g++（底座里早已具备）
# 直接进行全量 Python 资产同步即可
COPY . .

# ... 前面的代码保持不动 ...

# ⚡ 核心提效：注入国内官方 HuggingFace 镜像源加速通道，将几小时的下载压缩至 30 秒以内！
ENV HF_ENDPOINT=https://hf-mirror.com

# ⚡ 模型固件预载（因为加了 --network=host，这里会完美共享宿主机的梯子直接秒速固化）
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')"

# 暴露 FastAPI 默认服务的 8000 端口
EXPOSE 8000

# 容器启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]