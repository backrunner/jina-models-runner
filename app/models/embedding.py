import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
from ..config import Config
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class JinaEmbeddingsModel:
    def __init__(self):
        self.model_id = Config.EMBEDDINGS_MODEL_ID
        self.max_length = Config.EMBEDDINGS_MAX_LENGTH
        self.model_dimension = Config.EMBEDDINGS_DIMENSION
        
        # Initialize device settings
        self.use_mlx = Config.USE_MLX
        self.use_cuda = Config.USE_CUDA
        self.use_mps = Config.USE_MPS
        
        # Determine device priority: MLX > CUDA > MPS > CPU
        if self.use_mlx:
            self.device = "mlx"
        elif self.use_cuda:
            self.device = "cuda"
        elif self.use_mps:
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Initializing Jina embedding model (device={self.device}, model={self.model_id})")
        
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
        
        if self.device == "mlx":
            self._load_mlx_model(model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _load_mlx_model(self, model_path):
        try:
            # Try to import MLX
            import mlx.core as mx
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model with MLX
            logger.info("Loading model with MLX...")
            
            # Use trust_remote_code parameter to load Jina model, which may have custom code
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            logger.info("MLX model loading successful")
        except Exception as e:
            logger.warning(f"MLX model loading failed: {e}")
            logger.info("Falling back to PyTorch...")
            self.device = "cuda" if self.use_cuda else ("mps" if self.use_mps else "cpu")
            self._load_pytorch_model(model_path)
    
    def _load_pytorch_model(self, model_path):
        # Load tokenizer and model
        logger.info(f"Loading model with PyTorch to {self.device} device...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Use trust_remote_code parameter to load Jina model, which may have custom code
            self.model = AutoModel.from_pretrained(
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
            logger.info(f"Model loaded to {self.device} device")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _normalize(self, embeddings):
        """Normalize embedding vectors"""
        if self.device == "mlx":
            # MLX normalization
            import mlx.core as mx
            return embeddings / mx.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            # PyTorch normalization
            return embeddings / embeddings.norm(dim=1, keepdim=True)
    
    def embed(self, text: Union[str, List[str]], task: str = None) -> List[List[float]]:
        """Generate embedding vectors
        
        Args:
            text: Text or list of texts to embed
            task: Embedding task type, options include: 'retrieval.query', 'retrieval.passage', 
                 'separation', 'classification', 'text-matching' or None
        """
        # Ensure input is in list format
        if isinstance(text, str):
            text = [text]
        
        # Use Jina model's built-in encode method if available
        try:
            if hasattr(self.model, 'encode'):
                logger.info(f"Using model's built-in encode method to generate embeddings, task type: {task}")
                with torch.no_grad():
                    # Use model's encode method, which will handle device and normalization issues
                    embeddings = self.model.encode(text, task=task)
                    # Ensure result is Python list, not numpy array
                    if hasattr(embeddings, 'tolist'):
                        embeddings = embeddings.tolist()
                    elif isinstance(embeddings, np.ndarray):
                        embeddings = embeddings.tolist()
                return embeddings
        except Exception as e:
            logger.warning(f"Model's built-in encode method failed: {e}")
            logger.info("Falling back to manual embedding implementation")
        
        # Use different embedding methods based on device
        if self.device == "mlx":
            return self._embed_with_mlx(text)
        else:
            return self._embed_with_pytorch(text)
    
    def _embed_with_pytorch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using PyTorch"""
        # Tokenize input
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling as sentence embedding
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        
        # Calculate masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings
        embeddings = self._normalize(embeddings)
        
        # Convert to list
        result = embeddings.cpu().numpy().tolist()
        
        # Ensure result is standard Python list
        if not isinstance(result, list):
            result = result.tolist() if hasattr(result, 'tolist') else list(result)
        
        return result
    
    def _embed_with_mlx(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using MLX"""
        try:
            import mlx.core as mx
            
            # Tokenize input
            inputs = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Convert PyTorch tensors to MLX arrays
            mlx_inputs = {k: mx.array(v.numpy()) for k, v in inputs.items()}
            
            # Generate embeddings
            outputs = self.model(**mlx_inputs)
            
            # Use mean pooling
            attention_mask = mlx_inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            
            # Calculate masked mean pooling
            mask_expanded = mx.expand_dims(attention_mask, -1)
            mask_expanded = mx.broadcast_to(mask_expanded, embeddings.shape)
            sum_embeddings = mx.sum(embeddings * mask_expanded, axis=1)
            sum_mask = mx.sum(mask_expanded, axis=1)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings
            embeddings = self._normalize(embeddings)
            
            # Convert to list
            result = embeddings.tolist()
            
            # Ensure result is standard Python list
            if not isinstance(result, list):
                result = result.tolist() if hasattr(result, 'tolist') else list(result)
            
            return result
        except Exception as e:
            logger.error(f"MLX embedding failed: {e}")
            logger.info("Falling back to PyTorch for this request")
            return self._embed_with_pytorch(texts) 