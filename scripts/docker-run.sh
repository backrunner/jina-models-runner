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

# Default settings
IMAGE_NAME="jinaai/jina-models-runner"
IMAGE_TAG="latest"
HOST="0.0.0.0"
PORT="8000"
CONTAINER_NAME="jina-models-runner"
MODEL_CACHE_DIR="$PROJECT_ROOT/models"
LOGS_DIR="$PROJECT_ROOT/logs"
REMOVE=false
FORCE=false
GPU=false
DETACHED=false

# Show help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  -i, --image IMAGE_NAME  Set Docker image name (default: $IMAGE_NAME)"
    echo "  -t, --tag TAG           Set Docker image tag (default: $IMAGE_TAG)"
    echo "  -n, --name NAME         Set container name (default: $CONTAINER_NAME)"
    echo "  -p, --port PORT         Set host port (default: $PORT)"
    echo "  -m, --models DIR        Set models directory (default: $MODEL_CACHE_DIR)"
    echo "  -l, --logs DIR          Set logs directory (default: $LOGS_DIR)"
    echo "  -g, --gpu               Enable GPU support"
    echo "  -r, --rm                Remove container when stopped"
    echo "  -f, --force             Force recreate container if exists"
    echo "  -d, --detach            Run container in detached mode"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift
            shift
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -m|--models)
            MODEL_CACHE_DIR="$2"
            shift
            shift
            ;;
        -l|--logs)
            LOGS_DIR="$2"
            shift
            shift
            ;;
        -g|--gpu)
            GPU=true
            shift
            ;;
        -r|--rm)
            REMOVE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--detach)
            DETACHED=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $key${NC}"
            show_help
            ;;
    esac
done

# Create necessary directories
mkdir -p "$MODEL_CACHE_DIR" "$LOGS_DIR"

# Check if container exists
CONTAINER_EXISTS=$(docker ps -a -q -f name="^/$CONTAINER_NAME$")
if [ -n "$CONTAINER_EXISTS" ]; then
    if [ "$FORCE" = true ]; then
        echo -e "${YELLOW}Container '$CONTAINER_NAME' already exists. Removing...${NC}"
        docker rm -f "$CONTAINER_NAME" > /dev/null
    else
        # Check if container is running
        CONTAINER_RUNNING=$(docker ps -q -f name="^/$CONTAINER_NAME$")
        if [ -n "$CONTAINER_RUNNING" ]; then
            echo -e "${YELLOW}Container '$CONTAINER_NAME' is already running.${NC}"
            echo -e "${YELLOW}To force recreate, use --force option.${NC}"
            echo -e "${YELLOW}To view logs: docker logs -f $CONTAINER_NAME${NC}"
            exit 0
        else
            echo -e "${YELLOW}Starting existing container '$CONTAINER_NAME'...${NC}"
            docker start "$CONTAINER_NAME"
            
            if [ "$?" -ne 0 ]; then
                echo -e "${RED}Failed to start container!${NC}"
                echo -e "${YELLOW}To force recreate, use --force option.${NC}"
                exit 1
            fi
            
            echo -e "${GREEN}Container started successfully!${NC}"
            echo -e "${YELLOW}To view logs: docker logs -f $CONTAINER_NAME${NC}"
            echo -e "${BLUE}Service available at: http://$HOST:$PORT${NC}"
            exit 0
        fi
    fi
fi

# Build docker run command
DOCKER_CMD="docker run"

# Add options
if [ "$REMOVE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD --rm"
fi

if [ "$DETACHED" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
else
    DOCKER_CMD="$DOCKER_CMD -it"
fi

# Add GPU support if requested
if [ "$GPU" = true ]; then
    # Check if nvidia-docker or docker with gpu support is available
    if command -v nvidia-docker &> /dev/null; then
        echo -e "${YELLOW}Using nvidia-docker for GPU support${NC}"
        DOCKER_CMD="nvidia-docker run"
    elif docker info | grep -q "Runtimes:.*nvidia"; then
        echo -e "${YELLOW}Using docker with nvidia runtime for GPU support${NC}"
        DOCKER_CMD="$DOCKER_CMD --runtime=nvidia"
    elif docker info | grep -q "GPU"; then
        echo -e "${YELLOW}Using docker with GPU support${NC}"
        DOCKER_CMD="$DOCKER_CMD --gpus all"
    else
        echo -e "${RED}GPU support requested but not available!${NC}"
        echo -e "${YELLOW}Continuing without GPU support...${NC}"
    fi
fi

# Add name, port mapping, and volume mounts
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME -p $PORT:8000"
DOCKER_CMD="$DOCKER_CMD -v $MODEL_CACHE_DIR:/app/models"
DOCKER_CMD="$DOCKER_CMD -v $LOGS_DIR:/app/logs"

# Add environment variables
DOCKER_CMD="$DOCKER_CMD -e HOST=0.0.0.0 -e PORT=8000"
DOCKER_CMD="$DOCKER_CMD -e MODEL_CACHE_DIR=/app/models"

# Add image name
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME:$IMAGE_TAG"

echo -e "${BLUE}Starting Jina Models Runner container...${NC}"
echo -e "${GREEN}Container name: $CONTAINER_NAME${NC}"
echo -e "${GREEN}Image: $IMAGE_NAME:$IMAGE_TAG${NC}"
echo -e "${GREEN}Port mapping: $PORT:8000${NC}"
echo -e "${GREEN}Models directory: $MODEL_CACHE_DIR${NC}"
echo -e "${GREEN}Logs directory: $LOGS_DIR${NC}"

# Execute docker run command
echo -e "${YELLOW}Running command: $DOCKER_CMD${NC}"
exec $DOCKER_CMD

# This code won't execute due to the exec above, but kept for reference
if [ "$?" -ne 0 ]; then
    echo -e "${RED}Failed to start container!${NC}"
    exit 1
fi

echo -e "${GREEN}Container started successfully!${NC}"
echo -e "${BLUE}Service available at: http://$HOST:$PORT${NC}" 