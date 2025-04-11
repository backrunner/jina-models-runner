#!/bin/bash
set -e

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 设置工作目录
WORKDIR="/app"
cd "$WORKDIR"

# 设置模型目录和日志目录
MODEL_DIR="${MODEL_CACHE_DIR:-/app/models}"
LOG_DIR="/app/logs"
mkdir -p "$MODEL_DIR" "$LOG_DIR"

# 确保huggingface_hub已安装
python -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}huggingface_hub未安装，正在安装...${NC}"
    pip install huggingface_hub
fi

# 设置模型ID
EMBEDDING_MODEL_ID="${EMBEDDINGS_MODEL_ID:-jinaai/jina-embeddings-v3}"
RERANKER_MODEL_ID="${RERANKER_MODEL_ID:-jinaai/jina-reranker-v2-base-multilingual}"

# 设置模型路径
EMBEDDING_MODEL_PATH="$MODEL_DIR/jina-embeddings-v3"
RERANKER_MODEL_PATH="$MODEL_DIR/jina-reranker-v2-base-multilingual"

# 下载模型函数
download_model() {
    local model_id=$1
    local model_path=$2
    local model_name=$3
    
    echo -e "${BLUE}开始下载${model_name}模型: ${model_id}${NC}"
    if [ -d "$model_path" ] && [ "$FORCE_DOWNLOAD" != "true" ]; then
        echo -e "${YELLOW}${model_name}模型已存在于 $model_path${NC}"
    else
        echo -e "${YELLOW}下载中...这可能需要一些时间${NC}"
        
        # 下载选项
        local download_opts=""
        if [ "$FORCE_DOWNLOAD" = "true" ]; then
            download_opts="--force-download"
        fi
        
        # 使用python执行下载，支持HUGGINGFACE_TOKEN环境变量
        python -c "from huggingface_hub import snapshot_download; snapshot_download('$model_id', cache_dir='$MODEL_DIR', local_dir='$model_path', $download_opts)"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}${model_name}模型下载成功!${NC}"
        else
            echo -e "${RED}${model_name}模型下载失败!${NC}"
            exit 1
        fi
    fi
}

# 检查模型并下载
check_and_download_models() {
    # 检查嵌入模型
    if [ ! -d "$EMBEDDING_MODEL_PATH" ] || [ "$FORCE_DOWNLOAD" = "true" ]; then
        download_model "$EMBEDDING_MODEL_ID" "$EMBEDDING_MODEL_PATH" "嵌入"
    else
        echo -e "${GREEN}嵌入模型已存在于 $EMBEDDING_MODEL_PATH${NC}"
    fi
    
    # 检查重排序模型
    if [ ! -d "$RERANKER_MODEL_PATH" ] || [ "$FORCE_DOWNLOAD" = "true" ]; then
        download_model "$RERANKER_MODEL_ID" "$RERANKER_MODEL_PATH" "重排序"
    else
        echo -e "${GREEN}重排序模型已存在于 $RERANKER_MODEL_PATH${NC}"
    fi
}

# 环境设置函数
setup_environment() {
    echo -e "${BLUE}设置环境变量...${NC}"
    
    # 设置模型缓存目录
    export MODEL_CACHE_DIR="$MODEL_DIR"
    
    # 设置模型ID
    export EMBEDDINGS_MODEL_ID="$EMBEDDING_MODEL_ID"
    export RERANKER_MODEL_ID="$RERANKER_MODEL_ID"
    
    # 设置其他环境变量
    export HOST="${HOST:-0.0.0.0}"
    export PORT="${PORT:-8000}"
    
    # Docker环境下默认禁用MLX（MLX主要支持Apple Silicon芯片）
    if [ -z "$USE_MLX" ]; then
        export USE_MLX="False"
    fi
    
    # 检查CUDA可用性
    python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}检测到CUDA可用${NC}"
        export USE_CUDA="True"
    else
        echo -e "${YELLOW}CUDA不可用，使用CPU模式${NC}"
        export USE_CUDA="False"
    fi
    
    echo -e "${GREEN}环境变量设置完成${NC}"
    echo -e "HOST: $HOST"
    echo -e "PORT: $PORT"
    echo -e "MODEL_CACHE_DIR: $MODEL_CACHE_DIR"
    echo -e "USE_MLX: $USE_MLX"
    echo -e "USE_CUDA: $USE_CUDA"
}

# 启动服务函数
start_service() {
    echo -e "${BLUE}启动Jina模型API服务...${NC}"
    
    # 如果提供了额外的命令行参数，则执行这些参数
    if [ $# -gt 0 ]; then
        echo -e "${YELLOW}执行命令: $@${NC}"
        exec "$@"
    else
        # 否则启动默认的服务
        echo -e "${YELLOW}使用默认配置启动服务${NC}"
        exec python run.py
    fi
}

# 主函数
main() {
    echo -e "${BLUE}======== Jina 模型 API 服务 ========${NC}"
    
    # 下载模型
    check_and_download_models
    
    # 设置环境
    setup_environment
    
    # 启动服务
    start_service "$@"
}

# 执行主函数，传递所有命令行参数
main "$@" 