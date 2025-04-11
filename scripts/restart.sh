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

echo -e "${BLUE}重启Jina模型API服务...${NC}"

# 先停止服务
echo -e "${YELLOW}停止当前服务...${NC}"
$SCRIPT_DIR/stop.sh

# 检查停止是否成功
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告: 服务停止可能不完全${NC}"
fi

# 等待几秒钟确保服务完全停止
sleep 3

# 启动服务
echo -e "${YELLOW}启动服务...${NC}"
$SCRIPT_DIR/start.sh

# 检查启动是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}服务重启失败!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 服务已成功重启${NC}"
fi 