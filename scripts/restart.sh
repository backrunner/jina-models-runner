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

echo -e "${BLUE}Restarting Jina Models API Service...${NC}"

# Stop the service if it's running
echo -e "${YELLOW}Stopping any running service...${NC}"
$SCRIPT_DIR/stop.sh

# Wait for a moment to ensure the service is fully stopped
sleep 2

# Start the service again
echo -e "${YELLOW}Starting the service...${NC}"
$SCRIPT_DIR/start.sh "$@"

echo -e "${GREEN}✅ Restart completed!${NC}" 