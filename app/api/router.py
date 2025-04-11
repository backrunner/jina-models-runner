from fastapi import APIRouter, HTTPException, Depends
from ..config import EmbeddingRequest, EmbeddingResponse, RerankerRequest, RerankerResponse
from ..models.embedding import JinaEmbeddingsModel
from ..models.reranker import JinaRerankerModel
import logging
import numpy as np

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# 模型实例缓存
_embedding_model = None
_reranker_model = None

# 获取嵌入模型实例
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = JinaEmbeddingsModel()
    return _embedding_model

# 获取重排序模型实例
def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = JinaRerankerModel()
    return _reranker_model

# 映射模型名称到任务类型
def map_model_to_task(model_name: str) -> str:
    """根据模型名称映射到合适的任务类型"""
    # 针对jina-embeddings-v3模型的任务映射
    if 'query' in model_name.lower():
        return 'retrieval.query'
    elif 'passage' in model_name.lower():
        return 'retrieval.passage'
    elif 'classification' in model_name.lower():
        return 'classification'
    elif 'separation' in model_name.lower():
        return 'separation'
    else:
        # 默认使用text-matching任务
        return 'text-matching'

@router.post("/api/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    embedding_model: JinaEmbeddingsModel = Depends(get_embedding_model)
):
    """生成文本嵌入向量的API端点，兼容Ollama API"""
    try:
        logger.info(f"处理嵌入请求，模型: {request.model}")
        
        # 只支持jina-embeddings模型
        if "jina" not in request.model.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的模型: {request.model}。只支持Jina嵌入模型。"
            )
        
        # 从模型名称或选项中获取任务类型
        task = None
        if request.options and "task" in request.options:
            task = request.options["task"]
        elif 'query' in request.model.lower() or 'passage' in request.model.lower():
            task = map_model_to_task(request.model)
        
        # 生成嵌入向量
        embeddings = embedding_model.embed(request.prompt, task=task)
        
        # 确保嵌入向量是标准Python列表
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        # 返回结果
        if isinstance(embeddings, list) and isinstance(embeddings[0], list):
            # 如果嵌入了多个文本，只返回第一个的嵌入向量（符合API期望）
            return EmbeddingResponse(
                embedding=embeddings[0],
                model=request.model
            )
        else:
            # 单个嵌入向量
            return EmbeddingResponse(
                embedding=embeddings,
                model=request.model
            )
    
    except Exception as e:
        logger.error(f"生成嵌入向量时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成嵌入向量失败: {str(e)}")

@router.post("/api/rerank", response_model=RerankerResponse)
async def rerank_documents(
    request: RerankerRequest,
    reranker_model: JinaRerankerModel = Depends(get_reranker_model)
):
    """文档重排序API端点，兼容Ollama API扩展"""
    try:
        logger.info(f"处理重排序请求，模型: {request.model}")
        
        # 只支持jina-reranker模型
        if "jina" not in request.model.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的模型: {request.model}。只支持Jina重排序模型。"
            )
        
        # 检查文档列表
        if not request.documents or len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="没有提供重排序文档")
        
        # 执行重排序
        scores = reranker_model.rerank(request.query, request.documents)
        
        # 返回结果
        return RerankerResponse(
            scores=scores,
            model=request.model
        )
    
    except Exception as e:
        logger.error(f"重排序文档时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重排序文档失败: {str(e)}")

# 为了兼容Ollama API，添加模型信息端点
@router.get("/api/tags")
async def get_models():
    """获取可用模型列表的API端点，兼容Ollama API"""
    return {
        "models": [
            {
                "name": "jinaai/jina-embeddings-v3",
                "modified_at": "2023-10-01T00:00:00Z",
                "size": 350000000,
                "digest": "jina-embeddings-v3",
                "details": {
                    "family": "jina",
                    "parameter_size": "350M",
                    "quantization_level": "none"
                }
            },
            {
                "name": "jinaai/jina-embeddings-v3-query",
                "modified_at": "2023-10-01T00:00:00Z",
                "size": 350000000,
                "digest": "jina-embeddings-v3-query",
                "details": {
                    "family": "jina",
                    "parameter_size": "350M",
                    "quantization_level": "none",
                    "task": "retrieval.query"
                }
            },
            {
                "name": "jinaai/jina-embeddings-v3-passage",
                "modified_at": "2023-10-01T00:00:00Z",
                "size": 350000000,
                "digest": "jina-embeddings-v3-passage",
                "details": {
                    "family": "jina",
                    "parameter_size": "350M",
                    "quantization_level": "none",
                    "task": "retrieval.passage"
                }
            },
            {
                "name": "jinaai/jina-reranker-v2-base-multilingual",
                "modified_at": "2023-10-01T00:00:00Z",
                "size": 290000000,
                "digest": "jina-reranker-v2-base-multilingual",
                "details": {
                    "family": "jina",
                    "parameter_size": "290M",
                    "quantization_level": "none"
                }
            }
        ]
    } 