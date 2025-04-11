#!/bin/bash

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录的绝对路径
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 检测操作系统类型
SYSTEM=$(uname -s)
# 检测系统架构
ARCH=$(uname -m)

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}正在初始化Jina模型运行环境...${NC}"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "${YELLOW}检测到Python版本: $PYTHON_VERSION${NC}"

# 创建虚拟环境
cd "$PROJECT_ROOT"
echo -e "${YELLOW}创建虚拟环境...${NC}"

if [ -d "bin" ] && [ -d "lib" ] && [ -f "pyvenv.cfg" ]; then
    echo -e "${YELLOW}检测到已存在的虚拟环境，跳过创建步骤...${NC}"
else
    python3 -m venv .
    if [ $? -ne 0 ]; then
        echo -e "${RED}创建虚拟环境失败!${NC}"
        exit 1
    fi
    echo -e "${GREEN}虚拟环境创建成功!${NC}"
fi

# 激活虚拟环境
source "$PROJECT_ROOT/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}激活虚拟环境失败!${NC}"
    exit 1
fi
echo -e "${GREEN}虚拟环境已激活: $(which python)${NC}"

# 升级pip
echo -e "${YELLOW}正在升级pip...${NC}"
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${RED}升级pip失败!${NC}"
    exit 1
fi
echo -e "${GREEN}pip升级成功: $(pip --version)${NC}"

# 根据系统安装额外依赖
if [[ "$SYSTEM" == "Darwin" ]]; then
    # macOS 特定安装
    echo -e "${YELLOW}检测到macOS系统...${NC}"
    
    # 检查是否是Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${YELLOW}检测到Apple Silicon芯片，安装特定依赖...${NC}"
        
        # 安装MLX
        echo -e "${YELLOW}正在安装MLX...${NC}"
        pip install mlx
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}MLX安装失败，但这不影响整体功能。${NC}"
        else
            echo -e "${GREEN}MLX安装成功!${NC}"
        fi
        
        # 安装ONNX Runtime
        echo -e "${YELLOW}正在安装ONNX Runtime (支持CoreML)...${NC}"
        pip install "onnxruntime>=1.17.0"
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}ONNX Runtime安装失败，但这不影响整体功能。${NC}"
        else
            echo -e "${GREEN}ONNX Runtime安装成功!${NC}"
            # 验证CoreML支持
            COREML_SUPPORT=$(python -c "import onnxruntime as ort; print('CoreMLExecutionProvider' in ort.get_available_providers())")
            if [[ "$COREML_SUPPORT" == "True" ]]; then
                echo -e "${GREEN}ONNX Runtime已支持CoreML加速!${NC}"
            else
                echo -e "${YELLOW}ONNX Runtime不支持CoreML加速，将使用CPU执行。${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}检测到Intel芯片，安装标准依赖...${NC}"
        pip install "onnxruntime>=1.17.0"
    fi
    
elif [[ "$SYSTEM" == "Linux" ]]; then
    # Linux 特定安装
    echo -e "${YELLOW}检测到Linux系统，检查GPU可用性...${NC}"
    
    # 检查是否是ARM架构
    if [[ "$ARCH" == "aarch64" ]]; then
        echo -e "${YELLOW}检测到ARM架构，安装ARM专用依赖...${NC}"
        pip install "onnxruntime>=1.17.0"
    else
        # 检查NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${GREEN}检测到NVIDIA GPU，安装CUDA支持...${NC}"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}CUDA支持安装可能不完整，将尝试使用CPU模式运行。${NC}"
            fi
            
            # 安装ONNX Runtime GPU版本
            echo -e "${YELLOW}正在安装ONNX Runtime GPU版本...${NC}"
            pip install onnxruntime-gpu>=1.17.0
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}ONNX Runtime GPU安装失败，回退到CPU版本...${NC}"
                pip install "onnxruntime>=1.17.0"
            else
                echo -e "${GREEN}ONNX Runtime GPU安装成功!${NC}"
            fi
        else
            echo -e "${YELLOW}未检测到NVIDIA GPU，将使用CPU模式运行。${NC}"
            # 安装CPU版本的ONNX Runtime
            echo -e "${YELLOW}正在安装ONNX Runtime CPU版本...${NC}"
            pip install "onnxruntime>=1.17.0"
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}ONNX Runtime安装失败，但这不影响整体功能。${NC}"
            else
                echo -e "${GREEN}ONNX Runtime安装成功!${NC}"
            fi
        fi
    fi
fi

# 安装依赖
echo -e "${YELLOW}正在安装项目依赖...${NC}"
pip install -r "$PROJECT_ROOT/requirements.txt"
if [ $? -ne 0 ]; then
    echo -e "${RED}安装依赖失败!${NC}"
    exit 1
fi
echo -e "${GREEN}依赖安装成功!${NC}"

# 创建必要的目录
echo -e "${YELLOW}创建必要的目录...${NC}"
mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs"

# 下载模型
echo -e "${YELLOW}正在下载模型...${NC}"
$SCRIPT_DIR/download-models.sh
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}模型下载失败，但这不影响环境初始化。${NC}"
    echo -e "${YELLOW}稍后可以使用 './scripts/download-models.sh' 手动下载模型。${NC}"
fi

# 验证ONNX Runtime
echo -e "${YELLOW}验证ONNX Runtime安装...${NC}"
python -c "import onnxruntime as ort; print(f'ONNX Runtime版本: {ort.__version__}'); print(f'可用执行提供者: {ort.get_available_providers()}')" || echo -e "${YELLOW}ONNX Runtime验证失败，可能未正确安装。${NC}"

# 输出ONNX量化模型使用说明
echo -e "\n${BLUE}ONNX模型量化选项:${NC}"
echo -e "---------------------------------------"
echo -e "系统支持多种ONNX量化模型，可以通过环境变量选择："
echo -e "  ${GREEN}export ONNX_QUANTIZATION=none${NC}      # 使用无量化模型 (model.onnx) - 默认选项，最高精度但文件最大"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=fp16${NC}      # 使用FP16量化 (model_fp16.onnx) - 平衡精度与大小"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=int8${NC}      # 使用INT8量化 (model_int8.onnx) - 体积小，略微降低精度"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=uint8${NC}     # 使用UINT8量化 (model_uint8.onnx) - 体积小，略微降低精度"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=quantized${NC} # 使用通用量化 (model_quantized.onnx) - 通用量化方式"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=q4${NC}        # 使用4位量化 (model_q4.onnx) - 最小体积，但可能明显降低精度"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=bnb4${NC}      # 使用BNB 4位量化 (model_bnb4.onnx) - 极小体积，精度可能受影响"
echo -e "---------------------------------------"
echo -e "推荐使用${YELLOW}fp16${NC}或${YELLOW}quantized${NC}模型获得最佳性能和体积平衡\n"

echo -e "${GREEN}✅ 初始化完成!${NC}"
echo -e "${BLUE}要激活虚拟环境，请运行: source scripts/activate.sh${NC}"
echo -e "${BLUE}要启动服务，请运行: scripts/start.sh${NC}"

# 退出虚拟环境
deactivate 