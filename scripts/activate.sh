#!/bin/bash

# 检测脚本是否被source执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "错误: 此脚本必须通过source命令执行，例如:"
    echo "    source $(basename ${BASH_SOURCE[0]})"
    exit 1
fi

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录的绝对路径
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 检测操作系统类型
SYSTEM=$(uname -s)

# 确定虚拟环境激活脚本的位置
if [[ "$SYSTEM" == "Darwin" ]]; then
    # macOS
    ACTIVATE_SCRIPT="$PROJECT_ROOT/bin/activate"
elif [[ "$SYSTEM" == "Linux" ]]; then
    # Linux
    ACTIVATE_SCRIPT="$PROJECT_ROOT/bin/activate"
else
    echo "不支持的操作系统: $SYSTEM"
    return 1
fi

# 检查虚拟环境激活脚本是否存在
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "找不到虚拟环境激活脚本: $ACTIVATE_SCRIPT"
    echo "请先运行 'scripts/init.sh' 创建虚拟环境"
    return 1
fi

# 激活虚拟环境
echo "正在激活虚拟环境..."
source "$ACTIVATE_SCRIPT"

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "✅ 虚拟环境已激活，当前Python解释器: $(which python)"
echo "要退出虚拟环境，请运行 'deactivate' 或 'source scripts/deactivate.sh'" 