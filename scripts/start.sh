#!/bin/bash

set -e

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

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Virtual environment not activated, activating now...${NC}"
    source "$PROJECT_ROOT/bin/activate"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to activate virtual environment!${NC}"
        echo -e "${RED}Please run 'source scripts/activate.sh' first${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment activated: $(which python)${NC}"
fi

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Process command line arguments
PORT_ARG=""
HOST_ARG=""
LOG_ARG=""
RELOAD_ARG=""
WORKERS_ARG=""

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --port=*)
            PORT_ARG="${arg#*=}"
            shift
            ;;
        --host=*)
            HOST_ARG="${arg#*=}"
            shift
            ;;
        --log=*)
            LOG_ARG="${arg#*=}"
            shift
            ;;
        --reload)
            RELOAD_ARG="--reload"
            shift
            ;;
        --workers=*)
            WORKERS_ARG="${arg#*=}"
            shift
            ;;
        *)
            # Unknown option
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# Set default values if not provided
PORT=${PORT_ARG:-8000}
HOST=${HOST_ARG:-0.0.0.0}
LOG_LEVEL=${LOG_ARG:-info}
WORKERS=${WORKERS_ARG:-1}

# Set environment variables
export PORT=$PORT
export HOST=$HOST

echo -e "${BLUE}Starting Jina Models API Service...${NC}"
echo -e "${GREEN}Host: $HOST${NC}"
echo -e "${GREEN}Port: $PORT${NC}"
echo -e "${GREEN}Log level: $LOG_LEVEL${NC}"
echo -e "${GREEN}Workers: $WORKERS${NC}"
if [ -n "$RELOAD_ARG" ]; then
    echo -e "${GREEN}Auto-reload: enabled${NC}"
fi

# Check if models exist
MODEL_DIR="$PROJECT_ROOT/models"
if [ ! -d "$MODEL_DIR/jina-embeddings-v3" ] || [ ! -d "$MODEL_DIR/jina-reranker-v2-base-multilingual" ]; then
    echo -e "${YELLOW}Some models are missing. Downloading models...${NC}"
    $SCRIPT_DIR/download-models.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Model download failed!${NC}"
        echo -e "${YELLOW}You can try to download models manually using ./scripts/download-models.sh${NC}"
        echo -e "${YELLOW}Continuing anyway, models will be downloaded on demand.${NC}"
    fi
fi

# Start the service
cd "$PROJECT_ROOT"
echo -e "${GREEN}Starting service...${NC}"

# Run with Uvicorn
if [ -n "$RELOAD_ARG" ]; then
    python -m uvicorn app.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL $RELOAD_ARG --workers 1
else
    python -m uvicorn app.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL --workers $WORKERS
fi 