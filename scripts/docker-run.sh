#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录的绝对路径
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help           显示此帮助信息"
    echo "  -b, --build          运行前构建镜像"
    echo "  -f, --force          强制重新下载模型"
    echo "  -d, --detach         后台运行容器"
    echo "  -t, --token TOKEN    设置HuggingFace令牌"
    echo "  -p, --port PORT      设置端口 (默认: 8000)"
    echo "  -c, --compose        使用docker-compose (默认)"
    echo "  -r, --run            使用docker run 而不是 docker-compose"
    echo "  -q, --quantize TYPE  指定ONNX量化类型 (none, fp16, int8, uint8, quantized, q4, bnb4)"
    echo ""
    exit 0
}

# 解析命令行参数
BUILD=false
FORCE_DOWNLOAD=false
DETACH=""
HF_TOKEN=""
PORT=8000
USE_COMPOSE=true
ONNX_QUANTIZATION="none"  # 默认使用无量化的ONNX模型

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -f|--force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        -d|--detach)
            DETACH="-d"
            shift
            ;;
        -t|--token)
            HF_TOKEN="$2"
            shift
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -c|--compose)
            USE_COMPOSE=true
            shift
            ;;
        -r|--run)
            USE_COMPOSE=false
            shift
            ;;
        -q|--quantize)
            ONNX_QUANTIZATION="$2"
            shift
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            ;;
    esac
done

# 验证量化类型
VALID_QUANTIZATION=false
for type in "none" "fp16" "int8" "uint8" "quantized" "q4" "bnb4"; do
    if [ "$ONNX_QUANTIZATION" = "$type" ]; then
        VALID_QUANTIZATION=true
        break
    fi
done

if [ "$VALID_QUANTIZATION" = false ]; then
    echo -e "${RED}无效的量化类型: $ONNX_QUANTIZATION${NC}"
    echo -e "${YELLOW}有效值: none, fp16, int8, uint8, quantized, q4, bnb4${NC}"
    exit 1
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 确保模型目录存在
mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs"

# 根据是否使用docker-compose执行不同的命令
if [ "$USE_COMPOSE" = true ]; then
    # 使用docker-compose
    echo -e "${BLUE}使用docker-compose启动服务...${NC}"
    
    # 构建镜像（如果需要）
    if [ "$BUILD" = true ]; then
        echo -e "${YELLOW}构建镜像...${NC}"
        docker-compose build
    fi
    
    # 设置环境变量
    export PORT="$PORT"
    export USE_ONNX="true"
    export ONNX_QUANTIZATION="$ONNX_QUANTIZATION"
    
    if [ -n "$HF_TOKEN" ]; then
        export HUGGINGFACE_TOKEN="$HF_TOKEN"
    fi
    if [ "$FORCE_DOWNLOAD" = true ]; then
        export FORCE_DOWNLOAD="true"
    fi
    
    # 启动服务
    if [ -n "$DETACH" ]; then
        echo -e "${YELLOW}后台启动容器...${NC}"
        docker-compose up -d
    else
        echo -e "${YELLOW}前台启动容器...${NC}"
        docker-compose up
    fi
else
    # 使用docker run
    echo -e "${BLUE}使用docker run启动服务...${NC}"
    
    # 构建镜像（如果需要）
    if [ "$BUILD" = true ]; then
        echo -e "${YELLOW}构建镜像...${NC}"
        docker build -t jina-models-api .
    fi
    
    # 设置环境变量参数
    ENV_PARAMS="-e HOST=0.0.0.0 -e PORT=$PORT -e USE_MLX=False -e USE_ONNX=true -e ONNX_QUANTIZATION=$ONNX_QUANTIZATION"
    if [ -n "$HF_TOKEN" ]; then
        ENV_PARAMS="$ENV_PARAMS -e HUGGINGFACE_TOKEN=$HF_TOKEN"
    fi
    if [ "$FORCE_DOWNLOAD" = true ]; then
        ENV_PARAMS="$ENV_PARAMS -e FORCE_DOWNLOAD=true"
    fi
    
    # 启动容器
    echo -e "${YELLOW}启动容器...${NC}"
    docker run --name jina-models-api \
        -p $PORT:$PORT \
        $ENV_PARAMS \
        -v "$PROJECT_ROOT/models:/app/models" \
        -v "$PROJECT_ROOT/logs:/app/logs" \
        $DETACH \
        --restart unless-stopped \
        jina-models-api
fi

# 如果是后台启动，显示访问信息
if [ -n "$DETACH" ]; then
    echo -e "${GREEN}✅ 服务已在后台启动${NC}"
    echo -e "${BLUE}服务地址: http://localhost:$PORT${NC}"
    echo -e "${BLUE}健康检查: http://localhost:$PORT/health${NC}"
    echo -e "${BLUE}要查看日志: docker logs jina-models-api${NC}"
    echo -e "${BLUE}要停止服务: docker stop jina-models-api${NC}"
    echo -e "${BLUE}使用的ONNX量化类型: ${ONNX_QUANTIZATION}${NC}"
fi 