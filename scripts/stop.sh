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

echo -e "${BLUE}停止Jina模型API服务...${NC}"

# 检查PID文件
PID_FILE="$PROJECT_ROOT/.pid"
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}未找到PID文件，服务可能未运行。${NC}"
    
    # 检查是否有运行中的Python进程
    PIDS=$(ps aux | grep "python run.py" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo -e "${YELLOW}找到疑似服务进程: $PIDS${NC}"
        echo -e "${YELLOW}正在强制停止...${NC}"
        for PID in $PIDS; do
            kill -9 $PID 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}已终止进程 $PID${NC}"
            else
                echo -e "${RED}无法终止进程 $PID${NC}"
            fi
        done
    else
        echo -e "${RED}未找到运行中的服务进程${NC}"
    fi
    exit 0
fi

# 读取PID
PID=$(cat "$PID_FILE")
if [ -z "$PID" ]; then
    echo -e "${RED}PID文件为空${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

# 检查进程是否存在
if ! ps -p $PID > /dev/null; then
    echo -e "${YELLOW}进程 $PID 已不存在，删除PID文件...${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

# 停止服务
echo -e "${YELLOW}停止进程 $PID...${NC}"
kill $PID

# 等待进程终止
WAIT_COUNT=0
while ps -p $PID > /dev/null && [ $WAIT_COUNT -lt 10 ]; do
    echo -e "${YELLOW}等待服务停止...${NC}"
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

# 如果进程仍然存在，强制终止
if ps -p $PID > /dev/null; then
    echo -e "${YELLOW}服务未能正常停止，强制终止...${NC}"
    kill -9 $PID
    sleep 1
fi

# 最终检查
if ps -p $PID > /dev/null; then
    echo -e "${RED}无法停止服务 (PID: $PID)${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 服务已成功停止${NC}"
    rm -f "$PID_FILE"
fi 