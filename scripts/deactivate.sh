#!/bin/bash

# 检测脚本是否被source执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "错误: 此脚本必须通过source命令执行，例如:"
    echo "    source $(basename ${BASH_SOURCE[0]})"
    exit 1
fi

# 检查虚拟环境是否激活
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "当前没有激活的虚拟环境"
    return 0
fi

# 保存当前虚拟环境路径用于显示
CURRENT_VENV=$VIRTUAL_ENV

# 退出虚拟环境
deactivate

# 显示结果
echo "已退出虚拟环境: $CURRENT_VENV" 