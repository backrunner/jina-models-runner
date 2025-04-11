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

# 设置模型缓存目录
MODEL_CACHE_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODEL_CACHE_DIR"

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

# 确保huggingface_hub已安装
python -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}huggingface_hub未安装，正在安装...${NC}"
    pip install huggingface_hub
fi

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help           显示此帮助信息"
    echo "  -e, --embedding      仅下载嵌入模型 (jinaai/jina-embeddings-v3)"
    echo "  -r, --reranker       仅下载重排序模型 (jinaai/jina-reranker-v2-base-multilingual)"
    echo "  -t, --token TOKEN    指定HuggingFace令牌"
    echo "  -f, --force          强制重新下载，不使用缓存"
    echo ""
    echo "如果不指定-e或-r选项，则会下载两个模型"
    echo ""
    exit 0
}

# 解析命令行参数
EMBEDDING_ONLY=false
RERANKER_ONLY=false
FORCE=""
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -e|--embedding)
            EMBEDDING_ONLY=true
            shift
            ;;
        -r|--reranker)
            RERANKER_ONLY=true
            shift
            ;;
        -t|--token)
            HF_TOKEN="$2"
            shift
            shift
            ;;
        -f|--force)
            FORCE="--force-download"
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            ;;
    esac
done

# 设置模型路径
EMBEDDING_MODEL_ID="jinaai/jina-embeddings-v3"
EMBEDDING_MODEL_PATH="$MODEL_CACHE_DIR/jina-embeddings-v3"

RERANKER_MODEL_ID="jinaai/jina-reranker-v2-base-multilingual"
RERANKER_MODEL_PATH="$MODEL_CACHE_DIR/jina-reranker-v2-base-multilingual"

# 环境变量设置
MODEL_ENV=""
if [ -n "$HF_TOKEN" ]; then
    MODEL_ENV="HUGGINGFACE_TOKEN=$HF_TOKEN"
fi

# 如果两个标志都未设置，下载两个模型
if [ "$EMBEDDING_ONLY" = false ] && [ "$RERANKER_ONLY" = false ]; then
    EMBEDDING_ONLY=true
    RERANKER_ONLY=true
fi

# 下载嵌入模型
if [ "$EMBEDDING_ONLY" = true ]; then
    echo -e "${BLUE}开始下载嵌入模型: ${EMBEDDING_MODEL_ID}${NC}"
    if [ -d "$EMBEDDING_MODEL_PATH" ] && [ -z "$FORCE" ]; then
        echo -e "${YELLOW}嵌入模型已存在于 $EMBEDDING_MODEL_PATH${NC}"
        echo -e "${YELLOW}使用 --force 参数强制重新下载${NC}"
    else
        echo -e "${YELLOW}下载中...这可能需要一些时间${NC}"
        $MODEL_ENV python -c "from huggingface_hub import snapshot_download; snapshot_download('$EMBEDDING_MODEL_ID', cache_dir='$MODEL_CACHE_DIR', local_dir='$EMBEDDING_MODEL_PATH', $FORCE)"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}嵌入模型下载成功!${NC}"
        else
            echo -e "${RED}嵌入模型下载失败!${NC}"
            exit 1
        fi
    fi
fi

# 下载重排序模型
if [ "$RERANKER_ONLY" = true ]; then
    echo -e "${BLUE}开始下载重排序模型: ${RERANKER_MODEL_ID}${NC}"
    if [ -d "$RERANKER_MODEL_PATH" ] && [ -z "$FORCE" ]; then
        echo -e "${YELLOW}重排序模型已存在于 $RERANKER_MODEL_PATH${NC}"
        echo -e "${YELLOW}使用 --force 参数强制重新下载${NC}"
    else
        echo -e "${YELLOW}下载中...这可能需要一些时间${NC}"
        $MODEL_ENV python -c "from huggingface_hub import snapshot_download; snapshot_download('$RERANKER_MODEL_ID', cache_dir='$MODEL_CACHE_DIR', local_dir='$RERANKER_MODEL_PATH', $FORCE)"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}重排序模型下载成功!${NC}"
            
            # 检查重排序模型是否包含ONNX模型
            ONNX_DIR="$RERANKER_MODEL_PATH/onnx"
            if [ -d "$ONNX_DIR" ]; then
                echo -e "${GREEN}检测到ONNX模型目录: $ONNX_DIR${NC}"
                echo -e "${YELLOW}查看可用的ONNX模型:${NC}"
                ls -lh "$ONNX_DIR" | grep -E "\.onnx$" | awk '{print "  "$9" ("$5")"}'
                
                # 提供量化模型的说明
                echo -e "\n${BLUE}ONNX量化模型选项:${NC}"
                echo -e "  * model.onnx - 无量化模型，最高精度但文件最大"
                echo -e "  * model_fp16.onnx - FP16量化，平衡精度与大小"
                echo -e "  * model_int8.onnx - INT8量化，体积小，略微降低精度"
                echo -e "  * model_uint8.onnx - UINT8量化，体积小，略微降低精度"
                echo -e "  * model_quantized.onnx - 通用量化"
                echo -e "  * model_q4.onnx - 4位量化，最小体积但可能降低精度"
                echo -e "  * model_bnb4.onnx - BNB 4位量化，极小体积"
                
                echo -e "\n通过设置环境变量选择不同的量化模型:"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=none${NC} (默认)"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=fp16${NC} (推荐，在Apple Silicon上有良好表现)"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=int8${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=uint8${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=quantized${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=q4${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=bnb4${NC}"
            else
                echo -e "${YELLOW}未检测到ONNX模型目录，某些功能可能无法使用。${NC}"
            fi
        else
            echo -e "${RED}重排序模型下载失败!${NC}"
            exit 1
        fi
    fi
fi

echo -e "${GREEN}✅ 模型下载完成!${NC}"
echo -e "${BLUE}模型保存在: ${MODEL_CACHE_DIR} 目录下${NC}" 