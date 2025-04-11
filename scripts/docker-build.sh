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

# 设置镜像名称和标签
IMAGE_NAME="jina-models-api"
IMAGE_TAG="latest"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help           显示此帮助信息"
    echo "  -n, --name NAME      设置镜像名称 (默认: $IMAGE_NAME)"
    echo "  -t, --tag TAG        设置镜像标签 (默认: $IMAGE_TAG)"
    echo "  -p, --push           构建后推送到Docker仓库"
    echo "  -c, --compose        使用docker-compose构建"
    echo "  -f, --force          强制重新构建，不使用缓存"
    echo ""
    exit 0
}

# 解析命令行参数
PUSH=false
USE_COMPOSE=false
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift
            shift
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -c|--compose)
            USE_COMPOSE=true
            shift
            ;;
        -f|--force)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            ;;
    esac
done

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 构建Docker镜像
echo -e "${BLUE}开始构建Docker镜像 ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"

if [ "$USE_COMPOSE" = true ]; then
    # 使用docker-compose构建
    echo -e "${YELLOW}使用docker-compose构建...${NC}"
    
    if [ -n "$NO_CACHE" ]; then
        docker-compose build --no-cache
    else
        docker-compose build
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker Compose构建失败!${NC}"
        exit 1
    fi
else
    # 使用docker build命令构建
    echo -e "${YELLOW}使用docker build构建...${NC}"
    
    if [ -n "$NO_CACHE" ]; then
        docker build $NO_CACHE -t ${IMAGE_NAME}:${IMAGE_TAG} .
    else
        docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker构建失败!${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}镜像构建成功: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"

# 如果需要，推送镜像到仓库
if [ "$PUSH" = true ]; then
    echo -e "${YELLOW}正在推送镜像到仓库...${NC}"
    docker push ${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}镜像推送失败!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}镜像已成功推送到仓库${NC}"
fi

echo -e "${GREEN}✅ 完成!${NC}" 