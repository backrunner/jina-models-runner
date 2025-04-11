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

echo -e "${BLUE}Stopping Jina model API service...${NC}"

# PID file location
PID_FILE="$PROJECT_ROOT/.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}No PID file found. The service may not be running.${NC}"
    
    # Try to find running processes
    PIDS=$(ps -ef | grep "python" | grep -v grep | grep "app.main" | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo -e "${RED}No running service found.${NC}"
        exit 0
    else
        echo -e "${YELLOW}Found potential service processes:${NC}"
        for pid in $PIDS; do
            CMD=$(ps -p $pid -o command=)
            echo -e "${YELLOW}PID: $pid - $CMD${NC}"
        done
        
        # Ask for confirmation
        read -p "Do you want to stop these processes? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for pid in $PIDS; do
                echo -e "${YELLOW}Stopping process with PID: $pid${NC}"
                kill -15 $pid
                
                # Wait for process to terminate
                for i in {1..5}; do
                    if ps -p $pid > /dev/null; then
                        echo -e "${YELLOW}Waiting for process to terminate... ($i/5)${NC}"
                        sleep 1
                    else
                        echo -e "${GREEN}Process terminated.${NC}"
                        break
                    fi
                done
                
                # If process still exists, force kill
                if ps -p $pid > /dev/null; then
                    echo -e "${YELLOW}Process is not responding. Force killing...${NC}"
                    kill -9 $pid
                    if ps -p $pid > /dev/null; then
                        echo -e "${RED}Failed to kill process with PID: $pid${NC}"
                    else
                        echo -e "${GREEN}Process forcefully terminated.${NC}"
                    fi
                fi
            done
        else
            echo -e "${YELLOW}Operation cancelled.${NC}"
            exit 0
        fi
    fi
else
    # Read PID from file
    PID=$(cat "$PID_FILE")
    
    # Check if process exists
    if ps -p $PID > /dev/null; then
        echo -e "${YELLOW}Stopping service with PID: $PID${NC}"
        kill -15 $PID
        
        # Wait for process to terminate
        for i in {1..5}; do
            if ps -p $PID > /dev/null; then
                echo -e "${YELLOW}Waiting for process to terminate... ($i/5)${NC}"
                sleep 1
            else
                echo -e "${GREEN}Service stopped successfully.${NC}"
                rm -f "$PID_FILE"
                exit 0
            fi
        done
        
        # If process still exists, force kill
        if ps -p $PID > /dev/null; then
            echo -e "${YELLOW}Service is not responding. Force killing...${NC}"
            kill -9 $PID
            if ps -p $PID > /dev/null; then
                echo -e "${RED}Failed to stop service with PID: $PID${NC}"
                exit 1
            else
                echo -e "${GREEN}Service forcefully stopped.${NC}"
                rm -f "$PID_FILE"
                exit 0
            fi
        fi
    else
        echo -e "${YELLOW}Process with PID $PID not found. Removing stale PID file.${NC}"
        rm -f "$PID_FILE"
    fi
fi

echo -e "${GREEN}Service stopped.${NC}" 