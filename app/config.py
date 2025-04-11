from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import platform
import subprocess
import logging

logger = logging.getLogger(__name__)

# 系统检测函数
def detect_system_and_hardware():
    """检测当前系统类型和可用硬件"""
    system = platform.system()
    use_mlx = False
    use_cuda = False
    use_mps = False
    use_onnx = False
    
    if system == "Darwin":  # macOS
        # 检查是否为Apple Silicon
        try:
            cpu_brand = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
            if "Apple" in cpu_brand:
                logger.info("检测到Apple Silicon芯片，可启用MLX")
                use_mlx = True
                
                # 优先使用ONNX+CoreML加速，Apple芯片上效果较好
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if 'CoreMLExecutionProvider' in providers:
                        logger.info("检测到ONNX Runtime CoreML支持，将优先使用")
                        use_onnx = True
                        use_mlx = False  # 优先使用ONNX+CoreML而不是MLX
                except Exception as e:
                    logger.warning(f"无法检测ONNX Runtime: {e}")
            else:
                logger.info("检测到Intel芯片，将使用CPU")
        except Exception as e:
            logger.warning(f"无法检测CPU类型: {e}")
        
        # 检查MPS可用性
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("MPS可用于加速")
                use_mps = True
        except Exception as e:
            logger.warning(f"无法检测MPS可用性: {e}")
    
    elif system == "Linux":
        # 检查NVIDIA GPU是否可用
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logger.info("检测到NVIDIA GPU，可使用CUDA")
                use_cuda = True
        except Exception:
            logger.info("未检测到NVIDIA GPU或nvidia-smi不可用")
    
    # 使用环境变量覆盖检测结果
    if os.getenv("USE_MLX", "").lower() in ("true", "1", "t"):
        use_mlx = True
        logger.info("通过环境变量启用MLX")
    elif os.getenv("USE_MLX", "").lower() in ("false", "0", "f"):
        use_mlx = False
        logger.info("通过环境变量禁用MLX")
    
    if os.getenv("USE_CUDA", "").lower() in ("true", "1", "t"):
        use_cuda = True
        logger.info("通过环境变量启用CUDA")
    elif os.getenv("USE_CUDA", "").lower() in ("false", "0", "f"):
        use_cuda = False
        logger.info("通过环境变量禁用CUDA")
    
    if os.getenv("USE_MPS", "").lower() in ("true", "1", "t"):
        use_mps = True
        logger.info("通过环境变量启用MPS")
    elif os.getenv("USE_MPS", "").lower() in ("false", "0", "f"):
        use_mps = False
        logger.info("通过环境变量禁用MPS")
    
    if os.getenv("USE_ONNX", "").lower() in ("true", "1", "t"):
        use_onnx = True
        logger.info("通过环境变量启用ONNX Runtime")
    elif os.getenv("USE_ONNX", "").lower() in ("false", "0", "f"):
        use_onnx = False
        logger.info("通过环境变量禁用ONNX Runtime")
    
    return {
        "system": system,
        "use_mlx": use_mlx,
        "use_cuda": use_cuda,
        "use_mps": use_mps,
        "use_onnx": use_onnx
    }

# 检测系统和硬件
system_info = detect_system_and_hardware()

# ONNX模型量化类型
class ONNXQuantizationType:
    NONE = "none"           # 无量化 (model.onnx)
    FP16 = "fp16"           # FP16量化 (model_fp16.onnx)
    INT8 = "int8"           # INT8量化 (model_int8.onnx)
    UINT8 = "uint8"         # UINT8量化 (model_uint8.onnx)
    QUANTIZED = "quantized" # 通用量化 (model_quantized.onnx)
    Q4 = "q4"               # 4位量化 (model_q4.onnx)
    BNB4 = "bnb4"           # BNB 4位量化 (model_bnb4.onnx)

# 基本配置
class Config:
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # 模型配置
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "models")
    
    # 系统和硬件配置
    SYSTEM: str = system_info["system"]
    USE_MLX: bool = system_info["use_mlx"]
    USE_CUDA: bool = system_info["use_cuda"]
    USE_MPS: bool = system_info["use_mps"]
    USE_ONNX: bool = system_info["use_onnx"]
    
    # ONNX模型配置
    # 可选值: none, fp16, int8, uint8, quantized, q4, bnb4
    ONNX_QUANTIZATION: str = os.getenv("ONNX_QUANTIZATION", ONNXQuantizationType.NONE)
    
    # 嵌入模型配置 - 使用HuggingFace Hub上的模型
    EMBEDDINGS_MODEL_ID: str = os.getenv("EMBEDDINGS_MODEL_ID", "jinaai/jina-embeddings-v3")
    EMBEDDINGS_DIMENSION: int = 1024
    EMBEDDINGS_MAX_LENGTH: int = 8192
    
    # 重排序模型配置 - 使用HuggingFace Hub上的模型
    RERANKER_MODEL_ID: str = os.getenv("RERANKER_MODEL_ID", "jinaai/jina-reranker-v2-base-multilingual")
    RERANKER_MAX_LENGTH: int = 512

# Ollama API 兼容请求模型
class EmbeddingRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None

class RerankerRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    options: Optional[Dict[str, Any]] = None

# Ollama API 兼容响应模型
class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

class RerankerResponse(BaseModel):
    scores: List[float]
    model: str 