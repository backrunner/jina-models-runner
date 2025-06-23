#!/bin/bash

# Get absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get absolute path of the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set up colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set model cache directory
MODEL_CACHE_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODEL_CACHE_DIR"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Virtual environment not activated, activating now...${NC}"
    
    # Check if the virtual environment exists
    if [ ! -d "$PROJECT_ROOT/bin" ] || [ ! -f "$PROJECT_ROOT/bin/activate" ]; then
        echo -e "${RED}Virtual environment not found!${NC}"
        echo -e "${YELLOW}Please initialize the environment first: ./scripts/init.sh${NC}"
        exit 1
    fi
    
    # Source the virtual environment activation script
    source "$PROJECT_ROOT/bin/activate"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to activate virtual environment!${NC}"
        echo -e "${RED}Please run 'source scripts/activate.sh' first${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment activated: $(which python)${NC}"
fi

# Ensure huggingface_hub is installed
# Use the virtual environment Python directly
PYTHON_PATH="$PROJECT_ROOT/bin/python"
$PYTHON_PATH -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}huggingface_hub not installed, installing now...${NC}"
    pip install huggingface_hub
fi

# Show help information
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Display this help information"
    echo "  -e, --embedding      Download only embedding model (jinaai/jina-embeddings-v3)"
    echo "  -r, --reranker       Download only reranker model (jinaai/jina-reranker-v2-base-multilingual)"
    echo "  -t, --token TOKEN    Specify HuggingFace token"
    echo "  -f, --force          Force re-download, don't use cache"
    echo ""
    echo "If neither -e nor -r options are specified, both models will be downloaded"
    echo ""
    exit 0
}

# Parse command line arguments
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
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Set model paths
EMBEDDING_MODEL_ID="jinaai/jina-embeddings-v3"
EMBEDDING_MODEL_PATH="$MODEL_CACHE_DIR/jina-embeddings-v3"

RERANKER_MODEL_ID="jinaai/jina-reranker-v2-base-multilingual"
RERANKER_MODEL_PATH="$MODEL_CACHE_DIR/jina-reranker-v2-base-multilingual"

# Environment variable setup
MODEL_ENV=""
if [ -n "$HF_TOKEN" ]; then
    MODEL_ENV="HUGGINGFACE_TOKEN=$HF_TOKEN"
fi

# If neither flag is set, download both models
if [ "$EMBEDDING_ONLY" = false ] && [ "$RERANKER_ONLY" = false ]; then
    EMBEDDING_ONLY=true
    RERANKER_ONLY=true
fi

# Download embedding model
if [ "$EMBEDDING_ONLY" = true ]; then
    echo -e "${BLUE}Starting download of embedding model: ${EMBEDDING_MODEL_ID}${NC}"
    if [ -d "$EMBEDDING_MODEL_PATH" ] && [ -z "$FORCE" ]; then
        echo -e "${YELLOW}Embedding model already exists at $EMBEDDING_MODEL_PATH${NC}"
        echo -e "${YELLOW}Use --force parameter to force re-download${NC}"
    else
        echo -e "${YELLOW}Downloading... this may take some time${NC}"
        $MODEL_ENV $PYTHON_PATH -c "from huggingface_hub import snapshot_download; snapshot_download('$EMBEDDING_MODEL_ID', cache_dir='$MODEL_CACHE_DIR', local_dir='$EMBEDDING_MODEL_PATH', $FORCE)"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Embedding model download successful!${NC}"
        else
            echo -e "${RED}Embedding model download failed!${NC}"
            exit 1
        fi
    fi
fi

# Download reranker model
if [ "$RERANKER_ONLY" = true ]; then
    echo -e "${BLUE}Starting download of reranker model: ${RERANKER_MODEL_ID}${NC}"
    if [ -d "$RERANKER_MODEL_PATH" ] && [ -z "$FORCE" ]; then
        echo -e "${YELLOW}Reranker model already exists at $RERANKER_MODEL_PATH${NC}"
        echo -e "${YELLOW}Use --force parameter to force re-download${NC}"
    else
        echo -e "${YELLOW}Downloading... this may take some time${NC}"
        $MODEL_ENV $PYTHON_PATH -c "from huggingface_hub import snapshot_download; snapshot_download('$RERANKER_MODEL_ID', cache_dir='$MODEL_CACHE_DIR', local_dir='$RERANKER_MODEL_PATH', $FORCE)"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Reranker model download successful!${NC}"
            
            # Check if the reranker model includes ONNX models
            ONNX_DIR="$RERANKER_MODEL_PATH/onnx"
            if [ -d "$ONNX_DIR" ]; then
                echo -e "${GREEN}ONNX model directory detected: $ONNX_DIR${NC}"
                echo -e "${YELLOW}Available ONNX models:${NC}"
                ls -lh "$ONNX_DIR" | grep -E "\.onnx$" | awk '{print "  "$9" ("$5")"}'
                
                # Provide quantized model information
                echo -e "\n${BLUE}ONNX Quantization Options:${NC}"
                echo -e "  * model.onnx - No quantization, highest accuracy but largest file"
                echo -e "  * model_fp16.onnx - FP16 quantization, balance of accuracy and size"
                echo -e "  * model_int8.onnx - INT8 quantization, small size, slightly reduced accuracy"
                echo -e "  * model_uint8.onnx - UINT8 quantization, small size, slightly reduced accuracy"
                echo -e "  * model_quantized.onnx - General quantization"
                echo -e "  * model_q4.onnx - 4-bit quantization, smallest size but may reduce accuracy"
                echo -e "  * model_bnb4.onnx - BNB 4-bit quantization, very small size"
                
                echo -e "\nSelect different quantization models by setting environment variables:"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=none${NC} (default)"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=fp16${NC} (recommended, performs well on Apple Silicon)"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=int8${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=uint8${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=quantized${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=q4${NC}"
                echo -e "  ${GREEN}export ONNX_QUANTIZATION=bnb4${NC}"
            else
                echo -e "${YELLOW}ONNX model directory not detected, some features may not be available.${NC}"
            fi
        else
            echo -e "${RED}Reranker model download failed!${NC}"
            exit 1
        fi
    fi
fi

echo -e "${GREEN}✅ Model download complete!${NC}"
echo -e "${BLUE}Models saved in: ${MODEL_CACHE_DIR} directory${NC}" 