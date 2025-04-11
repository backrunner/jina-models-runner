from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import platform
import subprocess
import logging

logger = logging.getLogger(__name__)

# System detection function
def detect_system_and_hardware():
    """Detect current system type and available hardware"""
    system = platform.system()
    use_mlx = False
    use_cuda = False
    use_mps = False
    use_onnx = False
    
    if system == "Darwin":  # macOS
        # Check if Apple Silicon
        try:
            cpu_brand = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
            if "Apple" in cpu_brand:
                logger.info("Detected Apple Silicon chip, MLX can be enabled")
                use_mlx = True
                
                # Prefer ONNX+CoreML acceleration, works better on Apple chips
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if 'CoreMLExecutionProvider' in providers:
                        logger.info("Detected ONNX Runtime CoreML support, will use as priority")
                        use_onnx = True
                        use_mlx = False  # Prefer ONNX+CoreML over MLX
                except Exception as e:
                    logger.warning(f"Unable to detect ONNX Runtime: {e}")
            else:
                logger.info("Detected Intel chip, will use CPU")
        except Exception as e:
            logger.warning(f"Unable to detect CPU type: {e}")
        
        # Check MPS availability
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("MPS available for acceleration")
                use_mps = True
        except Exception as e:
            logger.warning(f"Unable to detect MPS availability: {e}")
    
    elif system == "Linux":
        # Check if NVIDIA GPU is available
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logger.info("Detected NVIDIA GPU, CUDA can be used")
                use_cuda = True
        except Exception:
            logger.info("No NVIDIA GPU detected or nvidia-smi not available")
    
    # Override detection results using environment variables
    if os.getenv("USE_MLX", "").lower() in ("true", "1", "t"):
        use_mlx = True
        logger.info("Enabled MLX via environment variable")
    elif os.getenv("USE_MLX", "").lower() in ("false", "0", "f"):
        use_mlx = False
        logger.info("Disabled MLX via environment variable")
    
    if os.getenv("USE_CUDA", "").lower() in ("true", "1", "t"):
        use_cuda = True
        logger.info("Enabled CUDA via environment variable")
    elif os.getenv("USE_CUDA", "").lower() in ("false", "0", "f"):
        use_cuda = False
        logger.info("Disabled CUDA via environment variable")
    
    if os.getenv("USE_MPS", "").lower() in ("true", "1", "t"):
        use_mps = True
        logger.info("Enabled MPS via environment variable")
    elif os.getenv("USE_MPS", "").lower() in ("false", "0", "f"):
        use_mps = False
        logger.info("Disabled MPS via environment variable")
    
    if os.getenv("USE_ONNX", "").lower() in ("true", "1", "t"):
        use_onnx = True
        logger.info("Enabled ONNX Runtime via environment variable")
    elif os.getenv("USE_ONNX", "").lower() in ("false", "0", "f"):
        use_onnx = False
        logger.info("Disabled ONNX Runtime via environment variable")
    
    return {
        "system": system,
        "use_mlx": use_mlx,
        "use_cuda": use_cuda,
        "use_mps": use_mps,
        "use_onnx": use_onnx
    }

# Detect system and hardware
system_info = detect_system_and_hardware()

# ONNX model quantization types
class ONNXQuantizationType:
    NONE = "none"           # No quantization (model.onnx)
    FP16 = "fp16"           # FP16 quantization (model_fp16.onnx)
    INT8 = "int8"           # INT8 quantization (model_int8.onnx)
    UINT8 = "uint8"         # UINT8 quantization (model_uint8.onnx)
    QUANTIZED = "quantized" # General quantization (model_quantized.onnx)
    Q4 = "q4"               # 4-bit quantization (model_q4.onnx)
    BNB4 = "bnb4"           # BNB 4-bit quantization (model_bnb4.onnx)

# Basic configuration
class Config:
    # Service configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Model configuration
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "models")
    
    # System and hardware configuration
    SYSTEM: str = system_info["system"]
    USE_MLX: bool = system_info["use_mlx"]
    USE_CUDA: bool = system_info["use_cuda"]
    USE_MPS: bool = system_info["use_mps"]
    USE_ONNX: bool = system_info["use_onnx"]
    
    # ONNX model configuration
    # Optional values: none, fp16, int8, uint8, quantized, q4, bnb4
    ONNX_QUANTIZATION: str = os.getenv("ONNX_QUANTIZATION", ONNXQuantizationType.NONE)
    
    # Embedding model configuration - Using models from HuggingFace Hub
    EMBEDDINGS_MODEL_ID: str = os.getenv("EMBEDDINGS_MODEL_ID", "jinaai/jina-embeddings-v3")
    EMBEDDINGS_DIMENSION: int = 1024
    EMBEDDINGS_MAX_LENGTH: int = 8192
    
    # Reranker model configuration - Using models from HuggingFace Hub
    RERANKER_MODEL_ID: str = os.getenv("RERANKER_MODEL_ID", "jinaai/jina-reranker-v2-base-multilingual")
    RERANKER_MAX_LENGTH: int = 512
    
    # API concurrency and cache configuration
    # Concurrency request limits
    MAX_EMBEDDING_CONCURRENCY: int = int(os.getenv("MAX_EMBEDDING_CONCURRENCY", "8"))
    MAX_RERANKER_CONCURRENCY: int = int(os.getenv("MAX_RERANKER_CONCURRENCY", "4"))
    
    # Cache configuration
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "t")
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
    RERANKER_CACHE_SIZE: int = int(os.getenv("RERANKER_CACHE_SIZE", "1000"))
    EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "300"))  # 5 minutes
    RERANKER_CACHE_TTL: int = int(os.getenv("RERANKER_CACHE_TTL", "300"))    # 5 minutes

# Ollama API compatible request models
class EmbeddingRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None

class RerankerRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    options: Optional[Dict[str, Any]] = None

# Ollama API compatible response models
class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

class RerankerResponse(BaseModel):
    scores: List[float]
    model: str 