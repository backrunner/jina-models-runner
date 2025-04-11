FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置HuggingFace令牌环境变量（如果需要的话）
# ENV HUGGINGFACE_TOKEN=""

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 关闭MLX (仅适用于苹果芯片)
ENV USE_MLX=False

# 启用ONNX Runtime
ENV USE_ONNX=True
# 默认使用无量化模型，可在运行时通过ONNX_QUANTIZATION环境变量更改
ENV ONNX_QUANTIZATION=none

# 设置模型缓存目录
ENV MODEL_CACHE_DIR="/app/models"
RUN mkdir -p /app/models /app/logs

# 预下载模型以避免首次启动时下载
RUN pip install huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('jinaai/jina-embeddings-v3', cache_dir='/app/models', local_dir='/app/models/jina-embeddings-v3'); \
    snapshot_download('jinaai/jina-reranker-v2-base-multilingual', cache_dir='/app/models', local_dir='/app/models/jina-reranker-v2-base-multilingual')"

# 设置模型ID环境变量
ENV EMBEDDINGS_MODEL_ID="jinaai/jina-embeddings-v3"
ENV RERANKER_MODEL_ID="jinaai/jina-reranker-v2-base-multilingual"

# 复制应用代码
COPY app/ /app/app/
COPY run.py /app/

# 复制入口脚本并设置权限
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# 暴露端口
EXPOSE 8000

# 设置入口点
ENTRYPOINT ["/app/docker-entrypoint.sh"] 