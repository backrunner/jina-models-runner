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

# 检测操作系统类型
SYSTEM=$(uname -s)

# 设置日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 设置模型目录
MODEL_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODEL_DIR"

# 检查模型是否已下载
EMBEDDING_MODEL_PATH="$MODEL_DIR/jina-embeddings-v3"
RERANKER_MODEL_PATH="$MODEL_DIR/jina-reranker-v2-base-multilingual"

if [ ! -d "$EMBEDDING_MODEL_PATH" ] || [ ! -d "$RERANKER_MODEL_PATH" ]; then
    echo -e "${YELLOW}模型文件不完整，尝试下载模型...${NC}"
    $SCRIPT_DIR/download-models.sh
    
    # 检查下载是否成功
    if [ ! -d "$EMBEDDING_MODEL_PATH" ] || [ ! -d "$RERANKER_MODEL_PATH" ]; then
        echo -e "${YELLOW}警告: 模型可能未完全下载，服务可能无法正常运行${NC}"
        echo -e "${YELLOW}如果遇到问题，请手动运行: './scripts/download-models.sh'${NC}"
    fi
fi

# 设置环境变量
if [[ "$SYSTEM" == "Darwin" ]]; then
    # 检查是否为Apple Silicon
    if [[ $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
        echo -e "${YELLOW}检测到Apple Silicon芯片，启用MLX加速...${NC}"
        export USE_MLX=True
    else
        echo -e "${YELLOW}检测到Intel芯片，禁用MLX加速...${NC}"
        export USE_MLX=False
    fi
elif [[ "$SYSTEM" == "Linux" ]]; then
    # 检查是否有NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}检测到NVIDIA GPU，使用CUDA...${NC}"
        export USE_MLX=False
        # 可以添加其他CUDA相关环境变量
    else
        echo -e "${YELLOW}未检测到GPU，将使用CPU模式...${NC}"
        export USE_MLX=False
    fi
else
    echo -e "${YELLOW}不支持的操作系统，将使用CPU模式...${NC}"
    export USE_MLX=False
fi

# 设置模型缓存目录
export MODEL_CACHE_DIR="$PROJECT_ROOT/models"

# 检查虚拟环境是否激活
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}虚拟环境未激活，正在激活...${NC}"
    source "$PROJECT_ROOT/bin/activate"
    if [ $? -ne 0 ]; then
        echo -e "${RED}激活虚拟环境失败!${NC}"
        echo -e "${RED}请先运行 'source scripts/activate.sh'${NC}"
        exit 1
    fi
    echo -e "${GREEN}虚拟环境已激活: $(which python)${NC}"
fi

# 获取进程ID
PID_FILE="$PROJECT_ROOT/.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}服务已经在运行 (PID: $OLD_PID)${NC}"
        echo -e "${YELLOW}如需重启，请先运行 'scripts/stop.sh'${NC}"
        exit 0
    else
        echo -e "${YELLOW}发现过期的PID文件，将被删除...${NC}"
        rm -f "$PID_FILE"
    fi
fi

# 启动服务并记录日志
echo -e "${BLUE}启动Jina模型API服务...${NC}"
echo -e "${YELLOW}环境: USE_MLX=$USE_MLX${NC}"
echo -e "${YELLOW}日志位置: $LOG_DIR/app.log${NC}"

cd "$PROJECT_ROOT"
nohup python run.py > "$LOG_DIR/nohup.out" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

# 检查服务是否成功启动
sleep 3
if ps -p $PID > /dev/null; then
    echo -e "${GREEN}✅ 服务已成功启动 (PID: $PID)${NC}"
    echo -e "${BLUE}服务地址: http://localhost:8000${NC}"
    echo -e "${BLUE}健康检查: http://localhost:8000/health${NC}"
    echo -e "${BLUE}要停止服务，请运行: scripts/stop.sh${NC}"
else
    echo -e "${RED}服务启动失败!${NC}"
    echo -e "${YELLOW}请查看日志: $LOG_DIR/nohup.out${NC}"
    rm -f "$PID_FILE"
    exit 1
fi 