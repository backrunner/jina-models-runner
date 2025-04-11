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

# Check if the virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Virtual environment is not active.${NC}"
    return 0
fi

# Deactivate the virtual environment
echo -e "${BLUE}Deactivating Jina Models environment...${NC}"
deactivate

# Check if deactivation was successful
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${GREEN}✅ Virtual environment deactivated successfully!${NC}"
else
    echo -e "${RED}Failed to deactivate the virtual environment.${NC}"
    return 1
fi 