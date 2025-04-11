#!/bin/bash

set -e

# Get absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get absolute path of the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Detect operating system type
SYSTEM=$(uname -s)
# Detect system architecture
ARCH=$(uname -m)

# Set up colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Initializing Jina Models runtime environment...${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION${NC}"

# Create virtual environment
cd "$PROJECT_ROOT"
echo -e "${YELLOW}Creating virtual environment...${NC}"

if [ -d "bin" ] && [ -d "lib" ] && [ -f "pyvenv.cfg" ]; then
    echo -e "${YELLOW}Existing virtual environment detected, skipping creation step...${NC}"
else
    python3 -m venv .
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created successfully!${NC}"
fi

# Activate virtual environment
source "$PROJECT_ROOT/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment!${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated: $(which python)${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to upgrade pip!${NC}"
    exit 1
fi
echo -e "${GREEN}Pip upgrade successful: $(pip --version)${NC}"

# Install extra dependencies based on the system
if [[ "$SYSTEM" == "Darwin" ]]; then
    # macOS specific installation
    echo -e "${YELLOW}Detected macOS system...${NC}"
    
    # Check if Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${YELLOW}Detected Apple Silicon chip, installing specific dependencies...${NC}"
        
        # Install MLX
        echo -e "${YELLOW}Installing MLX...${NC}"
        pip install mlx
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}MLX installation failed, but this doesn't affect overall functionality.${NC}"
        else
            echo -e "${GREEN}MLX installation successful!${NC}"
        fi
        
        # Install ONNX Runtime
        echo -e "${YELLOW}Installing ONNX Runtime (with CoreML support)...${NC}"
        pip install "onnxruntime>=1.17.0"
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}ONNX Runtime installation failed, but this doesn't affect overall functionality.${NC}"
        else
            echo -e "${GREEN}ONNX Runtime installation successful!${NC}"
            # Verify CoreML support
            COREML_SUPPORT=$(python -c "import onnxruntime as ort; print('CoreMLExecutionProvider' in ort.get_available_providers())")
            if [[ "$COREML_SUPPORT" == "True" ]]; then
                echo -e "${GREEN}ONNX Runtime supports CoreML acceleration!${NC}"
            else
                echo -e "${YELLOW}ONNX Runtime does not support CoreML acceleration, will use CPU execution.${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Detected Intel chip, installing standard dependencies...${NC}"
        pip install "onnxruntime>=1.17.0"
    fi
    
elif [[ "$SYSTEM" == "Linux" ]]; then
    # Linux specific installation
    echo -e "${YELLOW}Detected Linux system, checking GPU availability...${NC}"
    
    # Check if ARM architecture
    if [[ "$ARCH" == "aarch64" ]]; then
        echo -e "${YELLOW}Detected ARM architecture, installing ARM specific dependencies...${NC}"
        pip install "onnxruntime>=1.17.0"
    else
        # Check NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${GREEN}Detected NVIDIA GPU, installing CUDA support...${NC}"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}CUDA support installation may be incomplete, will try to run in CPU mode.${NC}"
            fi
            
            # Install ONNX Runtime GPU version
            echo -e "${YELLOW}Installing ONNX Runtime GPU version...${NC}"
            pip install onnxruntime-gpu>=1.17.0
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}ONNX Runtime GPU installation failed, falling back to CPU version...${NC}"
                pip install "onnxruntime>=1.17.0"
            else
                echo -e "${GREEN}ONNX Runtime GPU installation successful!${NC}"
            fi
        else
            echo -e "${YELLOW}No NVIDIA GPU detected, will run in CPU mode.${NC}"
            # Install CPU version of ONNX Runtime
            echo -e "${YELLOW}Installing ONNX Runtime CPU version...${NC}"
            pip install "onnxruntime>=1.17.0"
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}ONNX Runtime installation failed, but this doesn't affect overall functionality.${NC}"
            else
                echo -e "${GREEN}ONNX Runtime installation successful!${NC}"
            fi
        fi
    fi
fi

# Install dependencies
echo -e "${YELLOW}Installing project dependencies...${NC}"
pip install -r "$PROJECT_ROOT/requirements.txt"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies!${NC}"
    exit 1
fi
echo -e "${GREEN}Dependencies installed successfully!${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs"

# Download models
echo -e "${YELLOW}Downloading models...${NC}"
$SCRIPT_DIR/download-models.sh
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Model download failed, but this doesn't affect environment initialization.${NC}"
    echo -e "${YELLOW}You can download models manually later using './scripts/download-models.sh'.${NC}"
fi

# Verify ONNX Runtime
echo -e "${YELLOW}Verifying ONNX Runtime installation...${NC}"
python -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}'); print(f'Available execution providers: {ort.get_available_providers()}')" || echo -e "${YELLOW}ONNX Runtime verification failed, may not be installed correctly.${NC}"

# Output ONNX quantized model usage instructions
echo -e "\n${BLUE}ONNX Model Quantization Options:${NC}"
echo -e "---------------------------------------"
echo -e "The system supports various ONNX quantized models, selectable via environment variables:"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=none${NC}      # Use non-quantized model (model.onnx) - Default option, highest accuracy but largest file"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=fp16${NC}      # Use FP16 quantization (model_fp16.onnx) - Balance of accuracy and size"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=int8${NC}      # Use INT8 quantization (model_int8.onnx) - Small size, slightly reduced accuracy"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=uint8${NC}     # Use UINT8 quantization (model_uint8.onnx) - Small size, slightly reduced accuracy"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=quantized${NC} # Use general quantization (model_quantized.onnx) - Generic quantization method"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=q4${NC}        # Use 4-bit quantization (model_q4.onnx) - Smallest size, but may significantly reduce accuracy"
echo -e "  ${GREEN}export ONNX_QUANTIZATION=bnb4${NC}      # Use BNB 4-bit quantization (model_bnb4.onnx) - Very small size, accuracy may be affected"
echo -e "---------------------------------------"
echo -e "Recommended to use ${YELLOW}fp16${NC} or ${YELLOW}quantized${NC} models for the best balance of performance and size\n"

echo -e "${GREEN}✅ Initialization complete!${NC}"
echo -e "${BLUE}To activate the virtual environment, run: source scripts/activate.sh${NC}"
echo -e "${BLUE}To start the service, run: scripts/start.sh${NC}"

# Exit virtual environment
deactivate 