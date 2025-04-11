import torch
import os
import numpy as np
import platform
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union
from ..config import Config, ONNXQuantizationType
import logging
from huggingface_hub import snapshot_download
import time

logger = logging.getLogger(__name__)

class JinaRerankerModel:
    def __init__(self):
        self.model_id = Config.RERANKER_MODEL_ID
        self.max_length = Config.RERANKER_MAX_LENGTH
        
        # Initialize device settings
        self.use_cuda = Config.USE_CUDA
        self.use_mps = Config.USE_MPS
        self.use_onnx = Config.USE_ONNX if hasattr(Config, 'USE_ONNX') else False
        self.onnx_quantization = Config.ONNX_QUANTIZATION if hasattr(Config, 'ONNX_QUANTIZATION') else ONNXQuantizationType.NONE
        
        # Detect system type
        self.is_macos = platform.system() == "Darwin"
        
        # Determine device priority
        if self.is_macos:
            # On macOS, prioritize ONNX (with CoreML support)
            if self.use_onnx:
                self.device = "onnx"
            elif self.use_mps:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            # On non-macOS, prioritize CUDA
            if self.use_cuda:
                self.device = "cuda"
            elif self.use_onnx:
                # Consider ONNX only if CUDA is not available
                self.device = "onnx"
            else:
                self.device = "cpu"
        
        logger.info(f"Initializing Jina reranker model (device={self.device}, model={self.model_id})")
        
        # Download and load the model
        self._load_model()
    
    def _load_model(self):
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        # Download model to cache directory
        logger.info(f"Downloading model from HuggingFace Hub: {self.model_id}")
        try:
            model_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=Config.MODEL_CACHE_DIR,
                local_dir=os.path.join(Config.MODEL_CACHE_DIR, os.path.basename(self.model_id))
            )
            logger.info(f"Model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            model_path = self.model_id  # On failure, use original model_id, let transformers try to download directly
        
        if self.device == "onnx":
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _get_onnx_model_path(self, onnx_dir):
        """Get appropriate ONNX model file path based on quantization"""
        # Map quantization type to filename
        quantization_to_filename = {
            ONNXQuantizationType.NONE: "model.onnx",
            ONNXQuantizationType.FP16: "model_fp16.onnx",
            ONNXQuantizationType.INT8: "model_int8.onnx",
            ONNXQuantizationType.UINT8: "model_uint8.onnx",
            ONNXQuantizationType.QUANTIZED: "model_quantized.onnx",
            ONNXQuantizationType.Q4: "model_q4.onnx",
            ONNXQuantizationType.BNB4: "model_bnb4.onnx",
        }
        
        # Get filename based on configuration
        target_filename = quantization_to_filename.get(self.onnx_quantization, "model.onnx")
        target_path = os.path.join(onnx_dir, target_filename)
        
        # Check if target file exists
        if os.path.exists(target_path):
            logger.info(f"Using quantized model: {target_filename}")
            return target_path
        
        # If specified quantization model doesn't exist, try others
        logger.warning(f"Specified quantization model {target_filename} doesn't exist, trying other available models...")
        
        # Try other available models in performance/size priority order
        fallback_order = [
            "model_quantized.onnx",  # General quantized model, small size
            "model_quantized.onnx",  # General quantized model, small size
            "model_int8.onnx",        # INT8 quantization
            "model_uint8.onnx",       # UINT8 quantization
            "model_fp16.onnx",        # FP16 quantization
            "model_q4.onnx",          # 4-bit quantization
            "model_bnb4.onnx",        # BNB 4-bit quantization
            "model.onnx",             # Non-quantized original model
        ]
        
        for filename in fallback_order:
            path = os.path.join(onnx_dir, filename)
            if os.path.exists(path) and path != target_path:
                logger.info(f"Falling back to available model: {filename}")
                return path
        
        # If no available model is found, return None
        return None
    
    def _load_onnx_model(self, model_path):
        """Load model using ONNX Runtime"""
        try:
            # Import necessary libraries
            import onnxruntime as ort
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Check if ONNX model exists in the model directory
            onnx_dir = os.path.join(model_path, "onnx")
            if not os.path.exists(onnx_dir):
                raise FileNotFoundError(f"ONNX model directory not found: {onnx_dir}")
            
            # Select appropriate ONNX model based on quantization settings
            onnx_model_path = self._get_onnx_model_path(onnx_dir)
            
            if onnx_model_path is None:
                raise FileNotFoundError(f"No available ONNX model found in {onnx_dir}")
            
            # Configure ONNX Runtime session options
            logger.info("Initializing ONNX Runtime session...")
            session_options = ort.SessionOptions()
            
            # Log available execution providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX execution providers: {available_providers}")
            
            # Use CoreML as execution provider to leverage Apple Silicon performance
            providers = []
            
            # Try to use CoreML on macOS
            if self.is_macos and "CoreMLExecutionProvider" in available_providers:
                logger.info("Using CoreML as ONNX Runtime execution provider")
                
                # Determine best model format based on macOS version
                # macOS 12+/iOS 15+ supports MLProgram format
                import platform
                macos_version = tuple(map(int, platform.mac_ver()[0].split('.')))
                model_format = "MLProgram" if macos_version >= (12, 0) else "NeuralNetwork"
                
                # Configure CoreML execution provider options (using correct API format)
                coreml_options = {
                    "ModelFormat": model_format,
                    "MLComputeUnits": "ALL"  # Use all available compute units (CPU/GPU/ANE)
                }
                
                # Add CoreML as primary execution provider
                providers.append(("CoreMLExecutionProvider", coreml_options))
                
                logger.info(f"CoreML configuration: {coreml_options}")
            else:
                if self.is_macos:
                    logger.warning("CoreML execution provider not available, cannot optimize for Apple Silicon")
            
            # Add CPU execution provider as fallback
            providers.append("CPUExecutionProvider")
            
            # Create ONNX Runtime session
            logger.info(f"Creating ONNX session, provider order: {providers}")
            self.model = ort.InferenceSession(
                onnx_model_path, 
                sess_options=session_options,
                providers=providers
            )
            
            # Get model input and output names
            self.input_names = [input_info.name for input_info in self.model.get_inputs()]
            self.output_names = [output_info.name for output_info in self.model.get_outputs()]
            
            # Log used providers and model path
            active_providers = self.model.get_providers()
            model_file = os.path.basename(onnx_model_path)
            logger.info(f"ONNX Runtime model initialization successful: {model_file}")
            logger.info(f"Actual execution providers used: {active_providers}")
            
            # Add advanced debug information (if possible)
            if hasattr(self.model, "get_session_options"):
                session_info = self.model.get_session_options()
                logger.info(f"Session configuration: {session_info}")
            
        except Exception as e:
            logger.error(f"ONNX model loading failed: {e}")
            # Log detailed error information
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            
            # Choose fallback strategy based on system
            if self.is_macos:
                self.device = "mps" if self.use_mps else "cpu"
            else:
                self.device = "cuda" if self.use_cuda else "cpu"
            logger.info(f"Falling back to PyTorch model, device: {self.device}")
            self._load_pytorch_model(model_path)
    
    def _load_pytorch_model(self, model_path):
        # Load tokenizer and model
        logger.info(f"Loading reranker model using PyTorch to {self.device} device...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Use trust_remote_code parameter to load Jina model, which may have custom code
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Move model to appropriate device
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif self.device == "mps":
                if not torch.backends.mps.is_available():
                    logger.warning("MPS not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    logger.info("Using Apple MPS")
            
            self.model.to(self.device)
            # Set to evaluation mode
            self.model.eval()
            logger.info(f"Reranker model loaded to {self.device} device")
        except Exception as e:
            logger.error(f"Reranker model loading failed: {e}")
            raise
    
    def _batch_tokenize(self, query: str, passages: List[str]) -> List[Dict[str, Union[torch.Tensor, np.ndarray]]]:
        """Batch convert queries and passages to model input format"""
        features = []
        for passage in passages:
            # Use numpy tensors for ONNX, torch tensors for PyTorch
            return_tensors = "np" if self.device == "onnx" else "pt"
            
            try:
                tokenized = self.tokenizer(
                    query,
                    passage,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors=return_tensors
                )
                
                # Ensure input data types are correct (especially for ONNX+CoreML)
                if self.device == "onnx":
                    # For some models, CoreML requires specific data types
                    # Ensure input tensors are float32, which works better with CoreML
                    for key in tokenized:
                        if key in ["input_ids", "attention_mask", "token_type_ids"]:
                            # Keep integer type values unchanged
                            continue
                        elif isinstance(tokenized[key], np.ndarray):
                            # Convert float data to float32, improving CoreML compatibility
                            tokenized[key] = tokenized[key].astype(np.float32)
                
                features.append(tokenized)
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                # Create empty tokenization result to avoid entire process failure
                empty_result = {}
                if return_tensors == "np":
                    empty_result = {
                        "input_ids": np.zeros((1, self.max_length), dtype=np.int64),
                        "attention_mask": np.zeros((1, self.max_length), dtype=np.int64),
                        "token_type_ids": np.zeros((1, self.max_length), dtype=np.int64)
                    }
                else:
                    empty_result = {
                        "input_ids": torch.zeros((1, self.max_length), dtype=torch.long),
                        "attention_mask": torch.zeros((1, self.max_length), dtype=torch.long),
                        "token_type_ids": torch.zeros((1, self.max_length), dtype=torch.long)
                    }
                features.append(empty_result)
                
        return features
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents and return relevance scores"""
        # Try to use the model's built-in method if available
        try:
            if hasattr(self.model, 'compute_similarity') and self.device != "onnx":
                logger.info("Using model's built-in compute_similarity method")
                scores = []
                for doc in documents:
                    score = self.model.compute_similarity(query, doc)
                    scores.append(score)
                return scores
        except Exception as e:
            logger.warning(f"Model's built-in method failed: {e}")
            logger.info("Falling back to custom reranking implementation")
        
        # Use different reranking methods based on the device
        if self.device == "onnx":
            return self._rerank_with_onnx(query, documents)
        else:
            return self._rerank_with_pytorch(query, documents)
    
    def _rerank_with_pytorch(self, query: str, documents: List[str]) -> List[float]:
        """Perform reranking using PyTorch"""
        # Batch convert queries and documents to model input
        features = self._batch_tokenize(query, documents)
        
        scores = []
        for feature in features:
            # Move to appropriate device
            feature = {k: v.to(self.device) for k, v in feature.items()}
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(**feature)
            
            # Get relevance scores
            logits = outputs.logits
            score = torch.nn.functional.softmax(logits, dim=1)[:, 1].item()
            scores.append(score)
            
        return scores
    
    def _rerank_with_onnx(self, query: str, documents: List[str]) -> List[float]:
        """Perform reranking using ONNX Runtime"""
        # Batch convert queries and documents to model input
        features = self._batch_tokenize(query, documents)
        
        scores = []
        for idx, feature in enumerate(features):
            try:
                # Prepare input data
                onnx_inputs = {}
                
                # Check if input data matches what the model needs
                missing_inputs = [name for name in self.input_names if name not in feature]
                extra_inputs = [name for name in feature if name not in self.input_names]
                
                if missing_inputs:
                    logger.warning(f"Required inputs missing in features: {missing_inputs}")
                
                if extra_inputs and idx == 0:  # Only log once
                    logger.info(f"Extra input fields in features (will not be used): {extra_inputs}")
                
                # Only use inputs that the model needs
                for key, value in feature.items():
                    if key in self.input_names:
                        onnx_inputs[key] = value
                
                # Check if necessary inputs are missing
                if len(onnx_inputs) < len(self.input_names):
                    logger.error(f"Missing necessary inputs: need {self.input_names}, provided {list(onnx_inputs.keys())}")
                    # Provide a default low score instead of failing
                    scores.append(0.01)
                    continue
                
                # Log first document's input information (for debugging)
                if idx == 0:
                    input_shapes = {k: v.shape for k, v in onnx_inputs.items()}
                    input_types = {k: str(v.dtype) for k, v in onnx_inputs.items()}
                    logger.debug(f"ONNX input shapes: {input_shapes}")
                    logger.debug(f"ONNX input types: {input_types}")
                
                # Model inference
                start_time = time.time()
                outputs = self.model.run(self.output_names, onnx_inputs)
                inference_time = time.time() - start_time
                
                # Log first document's inference time (for performance analysis)
                if idx == 0:
                    logger.debug(f"ONNX inference time: {inference_time:.3f} seconds")
                
                # Get relevance score - handle different possible output formats
                logits = outputs[0]  # First output should be logits
                
                # Log first document's output shape (for debugging)
                if idx == 0:
                    logger.debug(f"ONNX output shape: {logits.shape}, type: {logits.dtype}")
                
                # Check logits dimensions and process accordingly
                if len(logits.shape) == 2 and logits.shape[1] >= 2:
                    # Standard format with multiple classes (typically [batch_size, 2])
                    # Calculate relevance score using NumPy softmax
                    logits_max = np.max(logits, axis=1, keepdims=True)
                    exp_logits = np.exp(logits - logits_max)
                    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    
                    # Get positive class score
                    score = softmax[0, 1].item()
                elif len(logits.shape) == 2 and logits.shape[1] == 1:
                    # Single output format with just one score (e.g., [batch_size, 1])
                    # Normalize raw score to [0,1] range
                    score = 1.0 / (1.0 + np.exp(-logits[0, 0]))
                elif len(logits.shape) == 1:
                    # One-dimensional output, directly a score (e.g., [batch_size])
                    score = 1.0 / (1.0 + np.exp(-logits[0]))
                else:
                    # Unknown format, log warning and return default score
                    logger.warning(f"Unknown logits format: shape={logits.shape}, using default score")
                    score = 0.5
                
                scores.append(score)
                
            except Exception as e:
                logger.error(f"ONNX inference error (document {idx}): {e}")
                import traceback
                logger.debug(f"Detailed error: {traceback.format_exc()}")
                
                # Provide a default low score instead of failing the entire process
                scores.append(0.1)
        
        # Check if all inferences failed
        if all(score <= 0.1 for score in scores) and len(scores) > 0:
            logger.error("All ONNX inference requests failed, results may be unreliable")
        
        return scores 