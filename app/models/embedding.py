import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Optional, Union
from ..config import Config
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class JinaEmbeddingsModel:
    def __init__(self):
        self.model_id = Config.EMBEDDINGS_MODEL_ID
        self.max_length = Config.EMBEDDINGS_MAX_LENGTH
        self.model_dimension = Config.EMBEDDINGS_DIMENSION
        
        # 初始化设备设置
        self.use_mlx = Config.USE_MLX
        self.use_cuda = Config.USE_CUDA
        self.use_mps = Config.USE_MPS
        
        # 确定设备优先级：MLX > CUDA > MPS > CPU
        if self.use_mlx:
            self.device = "mlx"
        elif self.use_cuda:
            self.device = "cuda"
        elif self.use_mps:
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"初始化Jina嵌入模型 (device={self.device}, model={self.model_id})")
        
        # 下载和加载模型
        self._load_model()
    
    def _load_model(self):
        # 确保模型缓存目录存在
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        # 下载模型到缓存目录
        logger.info(f"从HuggingFace Hub下载模型: {self.model_id}")
        try:
            model_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=Config.MODEL_CACHE_DIR,
                local_dir=os.path.join(Config.MODEL_CACHE_DIR, os.path.basename(self.model_id))
            )
            logger.info(f"模型已下载到: {model_path}")
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            model_path = self.model_id  # 失败时使用原始model_id，让transformers尝试直接下载
        
        if self.device == "mlx":
            self._load_mlx_model(model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _load_mlx_model(self, model_path):
        try:
            # 尝试导入MLX
            import mlx.core as mx
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 使用MLX加载模型
            logger.info("使用MLX加载模型...")
            
            # 使用trust_remote_code参数加载Jina模型，该模型可能有自定义代码
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            logger.info("MLX模型加载成功")
        except Exception as e:
            logger.warning(f"MLX模型加载失败: {e}")
            logger.info("回退到PyTorch...")
            self.device = "cuda" if self.use_cuda else ("mps" if self.use_mps else "cpu")
            self._load_pytorch_model(model_path)
    
    def _load_pytorch_model(self, model_path):
        # 加载tokenizer和模型
        logger.info(f"使用PyTorch加载模型到{self.device}设备...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 使用trust_remote_code参数加载Jina模型，该模型可能有自定义代码
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 移动模型到合适的设备
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("CUDA不可用，回退到CPU")
                    self.device = "cpu"
                else:
                    logger.info(f"使用CUDA: {torch.cuda.get_device_name(0)}")
            elif self.device == "mps":
                if not torch.backends.mps.is_available():
                    logger.warning("MPS不可用，回退到CPU")
                    self.device = "cpu"
                else:
                    logger.info("使用Apple MPS")
            
            self.model.to(self.device)
            # 设置为评估模式
            self.model.eval()
            logger.info(f"模型已加载到{self.device}设备")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _normalize(self, embeddings):
        """规范化嵌入向量"""
        if self.device == "mlx":
            # MLX规范化
            import mlx.core as mx
            return embeddings / mx.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            # PyTorch规范化
            return embeddings / embeddings.norm(dim=1, keepdim=True)
    
    def embed(self, text: Union[str, List[str]], task: str = None) -> List[List[float]]:
        """生成嵌入向量
        
        Args:
            text: 要嵌入的文本或文本列表
            task: 嵌入任务类型，可选值包括：'retrieval.query', 'retrieval.passage', 
                 'separation', 'classification', 'text-matching' 或 None
        """
        # 确保输入是列表格式
        if isinstance(text, str):
            text = [text]
        
        # 使用Jina模型自带的encode方法，如果可用
        try:
            if hasattr(self.model, 'encode'):
                logger.info(f"使用模型自带的encode方法生成嵌入向量，任务类型: {task}")
                with torch.no_grad():
                    # 使用模型的encode方法，它会处理设备和规范化问题
                    embeddings = self.model.encode(text, task=task)
                    # 确保结果是Python列表而不是numpy数组
                    if hasattr(embeddings, 'tolist'):
                        embeddings = embeddings.tolist()
                    elif isinstance(embeddings, np.ndarray):
                        embeddings = embeddings.tolist()
                return embeddings
        except Exception as e:
            logger.warning(f"模型自带encode方法失败: {e}")
            logger.info("回退到手动实现的嵌入方法")
        
        # 根据不同设备使用不同的嵌入方法
        if self.device == "mlx":
            return self._embed_with_mlx(text)
        else:
            return self._embed_with_pytorch(text)
    
    def _embed_with_pytorch(self, texts: List[str]) -> List[List[float]]:
        """使用PyTorch生成嵌入向量"""
        # Tokenize输入
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到合适的设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成嵌入向量
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 使用平均池化作为句子嵌入
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        
        # 计算掩码平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        embeddings = sum_embeddings / sum_mask
        
        # 规范化嵌入向量
        embeddings = self._normalize(embeddings)
        
        # 转为列表
        result = embeddings.cpu().numpy().tolist()
        
        # 确保结果是标准Python列表
        if not isinstance(result, list):
            result = result.tolist() if hasattr(result, 'tolist') else list(result)
        
        return result
    
    def _embed_with_mlx(self, texts: List[str]) -> List[List[float]]:
        """使用MLX生成嵌入向量"""
        try:
            import mlx.core as mx
            
            # Tokenize输入
            inputs = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # 将PyTorch张量转为MLX数组
            mlx_inputs = {k: mx.array(v.numpy()) for k, v in inputs.items()}
            
            # 生成嵌入向量
            outputs = self.model(**mlx_inputs)
            
            # 使用平均池化
            attention_mask = mlx_inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            
            # 计算掩码平均池化
            mask_expanded = mx.expand_dims(attention_mask, -1)
            mask_expanded = mx.broadcast_to(mask_expanded, embeddings.shape)
            sum_embeddings = mx.sum(embeddings * mask_expanded, axis=1)
            sum_mask = mx.sum(mask_expanded, axis=1)
            embeddings = sum_embeddings / sum_mask
            
            # 规范化嵌入向量
            embeddings = self._normalize(embeddings)
            
            # 转为列表
            result = embeddings.tolist()
            
            # 确保结果是标准Python列表
            if not isinstance(result, list):
                result = result.tolist() if hasattr(result, 'tolist') else list(result)
            
            return result
        except Exception as e:
            logger.error(f"MLX嵌入失败: {e}")
            logger.info("回退到PyTorch进行此次请求")
            return self._embed_with_pytorch(texts) 