#!/bin/bash
set -e

# Set up colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set working directory
WORKDIR="/app"
cd "$WORKDIR"

# Set model directory and log directory
MODEL_DIR="${MODEL_CACHE_DIR:-/app/models}"
LOG_DIR="/app/logs"
mkdir -p "$MODEL_DIR" "$LOG_DIR"

# Ensure huggingface_hub is installed
python -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}huggingface_hub not installed, installing...${NC}"
    pip install huggingface_hub
fi

# Set model IDs
EMBEDDING_MODEL_ID="${EMBEDDINGS_MODEL_ID:-jinaai/jina-embeddings-v3}"
RERANKER_MODEL_ID="${RERANKER_MODEL_ID:-jinaai/jina-reranker-v2-base-multilingual}"

# Set model paths
EMBEDDING_MODEL_PATH="$MODEL_DIR/jina-embeddings-v3"
RERANKER_MODEL_PATH="$MODEL_DIR/jina-reranker-v2-base-multilingual"

# Download model function
download_model() {
    local model_id=$1
    local model_path=$2
    local model_name=$3
    
    echo -e "${BLUE}Starting download of ${model_name} model: ${model_id}${NC}"
    if [ -d "$model_path" ] && [ "$FORCE_DOWNLOAD" != "true" ]; then
        echo -e "${YELLOW}${model_name} model already exists at $model_path${NC}"
    else
        echo -e "${YELLOW}Downloading... this may take some time${NC}"
        
        # Download options
        local download_opts=""
        if [ "$FORCE_DOWNLOAD" = "true" ]; then
            download_opts="--force-download"
        fi
        
        # Use Python to execute download, supports HUGGINGFACE_TOKEN environment variable
        python -c "from huggingface_hub import snapshot_download; snapshot_download('$model_id', cache_dir='$MODEL_DIR', local_dir='$model_path', $download_opts)"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}${model_name} model download successful!${NC}"
        else
            echo -e "${RED}${model_name} model download failed!${NC}"
            exit 1
        fi
    fi
}

# Check models and download
check_and_download_models() {
    # Check embedding model
    if [ ! -d "$EMBEDDING_MODEL_PATH" ] || [ "$FORCE_DOWNLOAD" = "true" ]; then
        download_model "$EMBEDDING_MODEL_ID" "$EMBEDDING_MODEL_PATH" "Embedding"
    else
        echo -e "${GREEN}Embedding model already exists at $EMBEDDING_MODEL_PATH${NC}"
    fi
    
    # Check reranker model
    if [ ! -d "$RERANKER_MODEL_PATH" ] || [ "$FORCE_DOWNLOAD" = "true" ]; then
        download_model "$RERANKER_MODEL_ID" "$RERANKER_MODEL_PATH" "Reranker"
    else
        echo -e "${GREEN}Reranker model already exists at $RERANKER_MODEL_PATH${NC}"
    fi
}

# Environment setup function
setup_environment() {
    echo -e "${BLUE}Setting up environment variables...${NC}"
    
    # Set model cache directory
    export MODEL_CACHE_DIR="$MODEL_DIR"
    
    # Set model IDs
    export EMBEDDINGS_MODEL_ID="$EMBEDDING_MODEL_ID"
    export RERANKER_MODEL_ID="$RERANKER_MODEL_ID"
    
    # Set other environment variables
    export HOST="${HOST:-0.0.0.0}"
    export PORT="${PORT:-8000}"
    
    # Disable MLX by default in Docker environment (MLX primarily supports Apple Silicon chips)
    if [ -z "$USE_MLX" ]; then
        export USE_MLX="False"
    fi
    
    # Check CUDA availability
    python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}CUDA detected as available${NC}"
        export USE_CUDA="True"
    else
        echo -e "${YELLOW}CUDA not available, using CPU mode${NC}"
        export USE_CUDA="False"
    fi
    
    echo -e "${GREEN}Environment variables setup complete${NC}"
    echo -e "HOST: $HOST"
    echo -e "PORT: $PORT"
    echo -e "MODEL_CACHE_DIR: $MODEL_CACHE_DIR"
    echo -e "USE_MLX: $USE_MLX"
    echo -e "USE_CUDA: $USE_CUDA"
}

# Start service function
start_service() {
    echo -e "${BLUE}Starting Jina Models API Service...${NC}"
    
    # If additional command line arguments are provided, execute them
    if [ $# -gt 0 ]; then
        echo -e "${YELLOW}Executing command: $@${NC}"
        exec "$@"
    else
        # Otherwise start the default service
        echo -e "${YELLOW}Starting service with default configuration${NC}"
        exec python run.py
    fi
}

# Main function
main() {
    echo -e "${BLUE}======== Jina Models API Service ========${NC}"
    
    # Download models
    check_and_download_models
    
    # Setup environment
    setup_environment
    
    # Start service
    start_service "$@"
}

# Execute main function, passing all command line arguments
main "$@" 