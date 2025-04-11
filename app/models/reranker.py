import torch
import os
import numpy as np
import platform
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any, Optional, Union, Tuple
from ..config import Config, ONNXQuantizationType
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class JinaRerankerModel:
    def __init__(self):
        self.model_id = Config.RERANKER_MODEL_ID
        self.max_length = Config.RERANKER_MAX_LENGTH
        
        # 初始化设备设置
        self.use_cuda = Config.USE_CUDA
        self.use_mps = Config.USE_MPS
        self.use_onnx = Config.USE_ONNX if hasattr(Config, 'USE_ONNX') else False
        self.onnx_quantization = Config.ONNX_QUANTIZATION if hasattr(Config, 'ONNX_QUANTIZATION') else ONNXQuantizationType.NONE
        
        # 检测系统类型
        self.is_macos = platform.system() == "Darwin"
        
        # 确定设备优先级
        if self.is_macos:
            # 在macOS上优先使用ONNX (支持CoreML)
            if self.use_onnx:
                self.device = "onnx"
            elif self.use_mps:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            # 在非macOS上优先使用CUDA
            if self.use_cuda:
                self.device = "cuda"
            elif self.use_onnx:
                # 在没有CUDA的情况下才考虑ONNX
                self.device = "onnx"
            else:
                self.device = "cpu"
        
        logger.info(f"初始化Jina重排序模型 (device={self.device}, model={self.model_id})")
        
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
        
        if self.device == "onnx":
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _get_onnx_model_path(self, onnx_dir):
        """根据量化选择获取合适的ONNX模型文件路径"""
        # 根据配置的量化类型映射文件名
        quantization_to_filename = {
            ONNXQuantizationType.NONE: "model.onnx",
            ONNXQuantizationType.FP16: "model_fp16.onnx",
            ONNXQuantizationType.INT8: "model_int8.onnx",
            ONNXQuantizationType.UINT8: "model_uint8.onnx",
            ONNXQuantizationType.QUANTIZED: "model_quantized.onnx",
            ONNXQuantizationType.Q4: "model_q4.onnx",
            ONNXQuantizationType.BNB4: "model_bnb4.onnx",
        }
        
        # 根据配置获取文件名
        target_filename = quantization_to_filename.get(self.onnx_quantization, "model.onnx")
        target_path = os.path.join(onnx_dir, target_filename)
        
        # 检查目标文件是否存在
        if os.path.exists(target_path):
            logger.info(f"使用量化模型: {target_filename}")
            return target_path
        
        # 如果指定的量化模型不存在，则尝试其他模型
        logger.warning(f"指定的量化模型 {target_filename} 不存在，尝试其他可用模型...")
        
        # 按照性能/大小优先级尝试其他可用模型
        fallback_order = [
            "model_quantized.onnx",  # 通用量化模型，体积小
            "model_int8.onnx",        # INT8量化
            "model_uint8.onnx",       # UINT8量化
            "model_fp16.onnx",        # FP16量化
            "model_q4.onnx",          # 4位量化
            "model_bnb4.onnx",        # BNB 4位量化
            "model.onnx",             # 无量化原始模型
        ]
        
        for filename in fallback_order:
            path = os.path.join(onnx_dir, filename)
            if os.path.exists(path) and path != target_path:
                logger.info(f"回退到可用模型: {filename}")
                return path
        
        # 如果没有找到任何可用模型，返回None
        return None
    
    def _load_onnx_model(self, model_path):
        """使用ONNX Runtime加载模型"""
        try:
            # 导入必要的库
            import onnxruntime as ort
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 检查模型目录中是否有ONNX模型
            onnx_dir = os.path.join(model_path, "onnx")
            if not os.path.exists(onnx_dir):
                raise FileNotFoundError(f"没有找到ONNX模型目录: {onnx_dir}")
            
            # 根据量化设置选择合适的ONNX模型
            onnx_model_path = self._get_onnx_model_path(onnx_dir)
            
            if onnx_model_path is None:
                raise FileNotFoundError(f"在 {onnx_dir} 中没有找到可用的ONNX模型")
            
            # 配置ONNX Runtime会话选项
            logger.info("初始化ONNX Runtime会话...")
            session_options = ort.SessionOptions()
            
            # 使用CoreML作为执行提供者，以利用Apple Silicon的性能
            providers = []
            
            # 在macOS上尝试使用CoreML
            if self.is_macos and hasattr(ort, "CoreMLExecutionProvider"):
                logger.info("使用CoreML作为ONNX Runtime执行提供者")
                provider_options = {
                    "CoreML": {"ModelFormat": "MLProgram"}
                }
                providers.append(("CoreMLExecutionProvider", provider_options["CoreML"]))
            
            # 添加CPU执行提供者作为备选
            providers.append("CPUExecutionProvider")
            
            # 创建ONNX Runtime会话
            self.model = ort.InferenceSession(
                onnx_model_path, 
                sess_options=session_options,
                providers=providers
            )
            
            # 获取模型输入和输出名称
            self.input_names = [input_info.name for input_info in self.model.get_inputs()]
            self.output_names = [output_info.name for output_info in self.model.get_outputs()]
            
            # 记录使用的提供者和模型路径
            active_providers = self.model.get_providers()
            model_file = os.path.basename(onnx_model_path)
            logger.info(f"ONNX Runtime模型初始化成功: {model_file}, 使用执行提供者: {active_providers}")
            
        except Exception as e:
            logger.error(f"ONNX模型加载失败: {e}")
            # 根据系统选择回退策略
            if self.is_macos:
                self.device = "mps" if self.use_mps else "cpu"
            else:
                self.device = "cuda" if self.use_cuda else "cpu"
            logger.info(f"回退到PyTorch模型，设备: {self.device}")
            self._load_pytorch_model(model_path)
    
    def _load_pytorch_model(self, model_path):
        # 加载tokenizer和模型
        logger.info(f"使用PyTorch加载重排序模型到{self.device}设备...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 使用trust_remote_code参数加载Jina模型，该模型可能有自定义代码
            self.model = AutoModelForSequenceClassification.from_pretrained(
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
            logger.info(f"重排序模型已加载到{self.device}设备")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            raise
    
    def _batch_tokenize(self, query: str, passages: List[str]) -> List[Dict[str, Union[torch.Tensor, np.ndarray]]]:
        """批量将查询和段落转换为模型输入格式"""
        features = []
        for passage in passages:
            tokenized = self.tokenizer(
                query,
                passage,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt" if self.device != "onnx" else "np"
            )
            features.append(tokenized)
        return features
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """对文档重新排序并返回相关性分数"""
        # 使用Jina模型自带的方法，如果有的话
        try:
            if hasattr(self.model, 'compute_similarity') and self.device != "onnx":
                logger.info("使用模型自带的compute_similarity方法")
                scores = []
                for doc in documents:
                    score = self.model.compute_similarity(query, doc)
                    scores.append(score)
                return scores
        except Exception as e:
            logger.warning(f"模型自带方法失败: {e}")
            logger.info("回退到自定义实现的重排序方法")
        
        # 根据不同设备使用不同的重排序方法
        if self.device == "onnx":
            return self._rerank_with_onnx(query, documents)
        else:
            return self._rerank_with_pytorch(query, documents)
    
    def _rerank_with_pytorch(self, query: str, documents: List[str]) -> List[float]:
        """使用PyTorch进行重排序"""
        # 批量将查询和文档转换为模型输入
        features = self._batch_tokenize(query, documents)
        
        scores = []
        for feature in features:
            # 移动到合适的设备
            feature = {k: v.to(self.device) for k, v in feature.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**feature)
            
            # 获取相关性分数
            logits = outputs.logits
            score = torch.nn.functional.softmax(logits, dim=1)[:, 1].item()
            scores.append(score)
            
        return scores
    
    def _rerank_with_onnx(self, query: str, documents: List[str]) -> List[float]:
        """使用ONNX Runtime进行重排序"""
        # 批量将查询和文档转换为模型输入
        features = self._batch_tokenize(query, documents)
        
        scores = []
        for feature in features:
            # 准备输入数据
            onnx_inputs = {}
            for key, value in feature.items():
                if key in self.input_names:
                    onnx_inputs[key] = value
            
            # 模型推理
            outputs = self.model.run(self.output_names, onnx_inputs)
            
            # 获取相关性分数 - 处理不同可能的输出格式
            logits = outputs[0]  # 第一个输出应该是logits
            
            # 检查logits的维度并相应处理
            if len(logits.shape) == 2 and logits.shape[1] >= 2:
                # 标准格式，有多个类别 (通常是 [batch_size, 2])
                # 使用NumPy softmax计算相关性分数
                logits_max = np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                # 获取正类的分数
                score = softmax[0, 1].item()
            elif len(logits.shape) == 2 and logits.shape[1] == 1:
                # 单一输出格式，只有一个分数 (如 [batch_size, 1])
                # 将原始分数标准化到 [0,1] 范围
                score = 1.0 / (1.0 + np.exp(-logits[0, 0]))
            elif len(logits.shape) == 1:
                # 一维输出，直接是分数 (如 [batch_size])
                score = 1.0 / (1.0 + np.exp(-logits[0]))
            else:
                # 未知格式，记录警告并返回默认分数
                logger.warning(f"未知的logits格式: shape={logits.shape}, 使用默认分数")
                score = 0.5
            
            scores.append(score)
            
            # 记录调试信息
            if len(scores) == 1:
                logger.debug(f"ONNX模型输出格式: shape={logits.shape}")
        
        return scores 