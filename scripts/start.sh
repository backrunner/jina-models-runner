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
while [[ $# -gt 0 ]]; do
    case $1 in
        --port=*)
            PORT_ARG="${1#*=}"
            shift
            ;;
        --port)
            PORT_ARG="$2"
            shift 2
            ;;
        --host=*)
            HOST_ARG="${1#*=}"
            shift
            ;;
        --host)
            HOST_ARG="$2"
            shift 2
            ;;
        --log=*)
            LOG_ARG="${1#*=}"
            shift
            ;;
        --log)
            LOG_ARG="$2"
            shift 2
            ;;
        --reload)
            RELOAD_ARG="--reload"
            shift
            ;;
        --workers=*)
            WORKERS_ARG="${1#*=}"
            shift
            ;;
        --workers)
            WORKERS_ARG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -h, --help              Display this help message"
            echo "  --port PORT             Set the port (default: 8000)"
            echo "  --host HOST             Set the host (default: 0.0.0.0)"
            echo "  --log LEVEL             Set log level (default: info)"
            echo "  --workers NUMBER        Set number of workers (default: 1)"
            echo "  --reload                Enable auto-reload for development"
            echo ""
            echo "Examples:"
            echo "  $0 --port 8001"
            echo "  $0 --host 127.0.0.1 --port 8080"
            echo "  $0 --workers 4 --log debug"
            echo ""
            exit 0
            ;;
        *)
            # Unknown option
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "${YELLOW}Use --help for usage information${NC}"
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
# Use the virtual environment Python directly to avoid PATH issues
PYTHON_PATH="$PROJECT_ROOT/bin/python"

if [ -n "$RELOAD_ARG" ]; then
    $PYTHON_PATH -m uvicorn app.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL $RELOAD_ARG --workers 1
else
    $PYTHON_PATH -m uvicorn app.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL --workers $WORKERS
fi 