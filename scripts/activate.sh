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

# Check if the virtual environment exists
if [ ! -d "$PROJECT_ROOT/bin" ] || [ ! -f "$PROJECT_ROOT/bin/activate" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please initialize the environment first: ./scripts/init.sh${NC}"
    return 1
fi

# Activate the virtual environment
echo -e "${BLUE}Activating Jina Models environment...${NC}"
source "$PROJECT_ROOT/bin/activate"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate the virtual environment!${NC}"
    return 1
fi

# Display information
echo -e "${GREEN}✅ Virtual environment activated!${NC}"
echo -e "${YELLOW}Python: $(which python) ($(python --version))${NC}"
echo -e "${YELLOW}Pip: $(pip --version)${NC}"

# Export project path
export JINA_MODELS_ROOT="$PROJECT_ROOT"
echo -e "${GREEN}Environment variable set: JINA_MODELS_ROOT=${JINA_MODELS_ROOT}${NC}"

# Display help information
echo -e "${BLUE}Available commands:${NC}"
echo -e "  ${GREEN}scripts/start.sh${NC} - Start the service"
echo -e "  ${GREEN}scripts/stop.sh${NC} - Stop the service"
echo -e "  ${GREEN}scripts/deactivate.sh${NC} - Deactivate the virtual environment"
echo -e "  ${GREEN}scripts/download-models.sh${NC} - Download models" 