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

# Parse command line arguments
VERSION="latest"
PUSH=false
ARCH=""

# Display help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help             Display this help message"
    echo "  -v, --version VERSION  Set Docker image version (default: latest)"
    echo "  -p, --push             Push the image to Docker Hub after building"
    echo "  -a, --arch ARCH        Build for specific architecture (e.g., amd64, arm64)"
    echo ""
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -v|--version)
            VERSION="$2"
            shift
            shift
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -a|--arch)
            ARCH="$2"
            shift
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $key${NC}"
            show_help
            ;;
    esac
done

# Docker image name
IMAGE_NAME="jinaai/jina-models-runner"
TAG="${IMAGE_NAME}:${VERSION}"

echo -e "${BLUE}Building Docker image for Jina Models API Service...${NC}"
echo -e "${GREEN}Image tag: ${TAG}${NC}"

# Determine build command based on architecture
BUILD_CMD="docker build"
if [ -n "$ARCH" ]; then
    echo -e "${YELLOW}Building for architecture: $ARCH${NC}"
    BUILD_CMD="$BUILD_CMD --platform linux/$ARCH"
fi

# Start build
cd "$PROJECT_ROOT"
echo -e "${YELLOW}Starting build process...${NC}"
$BUILD_CMD -t $TAG .

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Docker image built successfully: ${TAG}${NC}"

# Push to Docker Hub if requested
if [ "$PUSH" = true ]; then
    echo -e "${YELLOW}Pushing image to Docker Hub...${NC}"
    docker push $TAG
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker push failed!${NC}"
        echo -e "${YELLOW}You may need to run 'docker login' first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Image pushed to Docker Hub: ${TAG}${NC}"
fi

echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo -e "${BLUE}To run the container: docker run -p 8000:8000 ${TAG}${NC}" 