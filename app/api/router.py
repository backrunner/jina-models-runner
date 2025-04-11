from fastapi import APIRouter, HTTPException, Depends
from ..config import EmbeddingRequest, EmbeddingResponse, RerankerRequest, RerankerResponse, Config
from ..models.embedding import JinaEmbeddingsModel
from ..models.reranker import JinaRerankerModel
import logging
import numpy as np
import asyncio
import hashlib
import time
from typing import Dict, List, Any
from cachetools import TTLCache, LRUCache

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Model instance cache
_embedding_model = None
_reranker_model = None

# Concurrency control locks
_embedding_lock = asyncio.Lock()
_reranker_lock = asyncio.Lock()

# Request merging dictionaries, for tracking identical requests being processed
_embedding_requests_in_flight = {}
_reranker_requests_in_flight = {}

# Custom cache class combining LRU and TTL features
class LRUTTLCache:
    """Cache with both LRU and TTL features"""
    
    def __init__(self, maxsize, ttl):
        self.ttl_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.lru_cache = LRUCache(maxsize=maxsize)
    
    def __contains__(self, key):
        return key in self.ttl_cache and key in self.lru_cache
    
    def __getitem__(self, key):
        # First check if it exists in TTL cache to ensure it hasn't expired
        if key in self.ttl_cache:
            # Then get from LRU cache, which will update access order
            return self.lru_cache[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        # Update both caches simultaneously
        self.ttl_cache[key] = value
        self.lru_cache[key] = value
    
    def __delitem__(self, key):
        # Delete from both caches
        if key in self.ttl_cache:
            del self.ttl_cache[key]
        if key in self.lru_cache:
            del self.lru_cache[key]
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

# Result cache - Using config settings with LRU+TTL dual strategy
_embedding_cache = LRUTTLCache(
    maxsize=Config.EMBEDDING_CACHE_SIZE, 
    ttl=Config.EMBEDDING_CACHE_TTL
) if Config.CACHE_ENABLED else None

_reranker_cache = LRUTTLCache(
    maxsize=Config.RERANKER_CACHE_SIZE, 
    ttl=Config.RERANKER_CACHE_TTL
) if Config.CACHE_ENABLED else None

# Concurrency limiting semaphores
_embedding_semaphore = asyncio.Semaphore(Config.MAX_EMBEDDING_CONCURRENCY)
_reranker_semaphore = asyncio.Semaphore(Config.MAX_RERANKER_CONCURRENCY)

# Log configuration information at startup
logger.info(f"API configuration: Cache {'enabled' if Config.CACHE_ENABLED else 'disabled'}")
logger.info(f"Embedding API concurrency limit: {Config.MAX_EMBEDDING_CONCURRENCY}")
logger.info(f"Reranker API concurrency limit: {Config.MAX_RERANKER_CONCURRENCY}")
if Config.CACHE_ENABLED:
    logger.info(f"Embedding cache: capacity={Config.EMBEDDING_CACHE_SIZE}, TTL={Config.EMBEDDING_CACHE_TTL}s")
    logger.info(f"Reranker cache: capacity={Config.RERANKER_CACHE_SIZE}, TTL={Config.RERANKER_CACHE_TTL}s")

# Get embedding model instance
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = JinaEmbeddingsModel()
    return _embedding_model

# Get reranker model instance
def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = JinaRerankerModel()
    return _reranker_model

# Generate request hash
def _generate_hash(obj: Any) -> str:
    """Generate a hash for an object, used as cache key"""
    if isinstance(obj, dict):
        # Sort dictionary to ensure consistency
        return hashlib.md5(str(sorted(obj.items())).encode()).hexdigest()
    else:
        return hashlib.md5(str(obj).encode()).hexdigest()

# Map model name to task type
def map_model_to_task(model_name: str) -> str:
    """Map model name to appropriate task type"""
    # Task mapping for jina-embeddings-v3 models
    if 'query' in model_name.lower():
        return 'retrieval.query'
    elif 'passage' in model_name.lower():
        return 'retrieval.passage'
    elif 'classification' in model_name.lower():
        return 'classification'
    elif 'separation' in model_name.lower():
        return 'separation'
    else:
        # Default to text-matching task
        return 'text-matching'

@router.post("/api/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    embedding_model: JinaEmbeddingsModel = Depends(get_embedding_model)
):
    """API endpoint for generating text embeddings, compatible with Ollama API"""
    try:
        # Record request start time (for performance monitoring)
        start_time = time.time()
        logger.info(f"Processing embedding request, model: {request.model}")
        
        # Only support jina-embeddings models
        if "jina" not in request.model.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model}. Only Jina embedding models are supported."
            )
        
        # Get task type from model name or options
        task = None
        if request.options and "task" in request.options:
            task = request.options["task"]
        elif 'query' in request.model.lower() or 'passage' in request.model.lower():
            task = map_model_to_task(request.model)
        
        # Create cache key
        cache_key = _generate_hash({
            "prompt": request.prompt,
            "model": request.model,
            "task": task
        })
        
        # Check if result already exists in cache (if caching is enabled)
        if Config.CACHE_ENABLED and cache_key in _embedding_cache:
            logger.info(f"Returning embedding result from cache (cache_key={cache_key[:8]})")
            embeddings = _embedding_cache[cache_key]
            elapsed = time.time() - start_time
            logger.info(f"Embedding request completed (cached=True, time={elapsed:.2f}s)")
            return EmbeddingResponse(
                embedding=embeddings,
                model=request.model
            )
            
        # Check if identical request is already being processed
        if cache_key in _embedding_requests_in_flight:
            logger.info(f"Waiting for identical embedding request in progress (cache_key={cache_key[:8]})")
            embeddings = await _embedding_requests_in_flight[cache_key]
            elapsed = time.time() - start_time
            logger.info(f"Embedding request completed (deduplicated=True, time={elapsed:.2f}s)")
            return EmbeddingResponse(
                embedding=embeddings,
                model=request.model
            )
        
        # Create async Future to track the request being processed
        future = asyncio.Future()
        _embedding_requests_in_flight[cache_key] = future
        
        try:
            # Use semaphore to limit concurrency
            async with _embedding_semaphore:
                # Use lock to ensure embedding generation exclusivity
                async with _embedding_lock:
                    # Check cache again, as it might have been populated while waiting for lock
                    if Config.CACHE_ENABLED and cache_key in _embedding_cache:
                        embeddings = _embedding_cache[cache_key]
                    else:
                        # Generate embedding vector
                        embeddings = embedding_model.embed(request.prompt, task=task)
                        
                        # Ensure embedding vector is standard Python list
                        if isinstance(embeddings, np.ndarray):
                            embeddings = embeddings.tolist()
                        
                        # Store result in cache (if caching is enabled)
                        if Config.CACHE_ENABLED:
                            if isinstance(embeddings, list) and isinstance(embeddings[0], list):
                                # If multiple texts were embedded, use first one's vector (matches API expectation)
                                _embedding_cache[cache_key] = embeddings[0]
                            else:
                                # Single embedding vector
                                _embedding_cache[cache_key] = embeddings
            
            # Prepare result for return
            result_embeddings = None
            if Config.CACHE_ENABLED:
                result_embeddings = _embedding_cache[cache_key]
            else:
                if isinstance(embeddings, list) and isinstance(embeddings[0], list):
                    result_embeddings = embeddings[0]
                else:
                    result_embeddings = embeddings
                    
            # Set Future result, unblocking all awaiting identical requests
            future.set_result(result_embeddings)
            
            # Record processing time
            elapsed = time.time() - start_time
            logger.info(f"Embedding request completed (new=True, time={elapsed:.2f}s)")
            
            # Return result
            return EmbeddingResponse(
                embedding=result_embeddings,
                model=request.model
            )
            
        finally:
            # Remove from in-flight requests
            del _embedding_requests_in_flight[cache_key]
    
    except Exception as e:
        logger.error(f"Error generating embedding vector: {str(e)}")
        # If Future was created but result not set, set exception
        if 'cache_key' in locals() and cache_key in _embedding_requests_in_flight:
            if not _embedding_requests_in_flight[cache_key].done():
                _embedding_requests_in_flight[cache_key].set_exception(e)
            del _embedding_requests_in_flight[cache_key]
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding vector: {str(e)}")

@router.post("/api/rerank", response_model=RerankerResponse)
async def rerank_documents(
    request: RerankerRequest,
    reranker_model: JinaRerankerModel = Depends(get_reranker_model)
):
    """Document reranking API endpoint, compatible with Ollama API extension"""
    try:
        # Record request start time
        start_time = time.time()
        logger.info(f"Processing reranking request, model: {request.model}")
        
        # Only support jina-reranker models
        if "jina" not in request.model.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model}. Only Jina reranker models are supported."
            )
        
        # Check document list
        if not request.documents or len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="No documents provided for reranking")
        
        # Create cache key
        cache_key = _generate_hash({
            "query": request.query,
            "documents": request.documents,
            "model": request.model
        })
        
        # Check if result already exists in cache (if caching is enabled)
        if Config.CACHE_ENABLED and cache_key in _reranker_cache:
            logger.info(f"Returning reranking result from cache (cache_key={cache_key[:8]})")
            scores = _reranker_cache[cache_key]
            elapsed = time.time() - start_time
            logger.info(f"Reranking request completed (cached=True, time={elapsed:.2f}s)")
            return RerankerResponse(
                scores=scores,
                model=request.model
            )
            
        # Check if identical request is already being processed
        if cache_key in _reranker_requests_in_flight:
            logger.info(f"Waiting for identical reranking request in progress (cache_key={cache_key[:8]})")
            scores = await _reranker_requests_in_flight[cache_key]
            elapsed = time.time() - start_time
            logger.info(f"Reranking request completed (deduplicated=True, time={elapsed:.2f}s)")
            return RerankerResponse(
                scores=scores,
                model=request.model
            )
        
        # Create async Future
        future = asyncio.Future()
        _reranker_requests_in_flight[cache_key] = future
        
        try:
            # Use semaphore to limit concurrency
            async with _reranker_semaphore:
                # Use lock to ensure reranking exclusivity
                async with _reranker_lock:
                    # Check cache again, as it might have been populated while waiting for lock
                    if Config.CACHE_ENABLED and cache_key in _reranker_cache:
                        scores = _reranker_cache[cache_key]
                    else:
                        # Perform reranking
                        scores = reranker_model.rerank(request.query, request.documents)
                        # Store result in cache (if caching is enabled)
                        if Config.CACHE_ENABLED:
                            _reranker_cache[cache_key] = scores
            
            # Set Future result
            future.set_result(scores if not Config.CACHE_ENABLED else _reranker_cache[cache_key])
            
            # Record processing time
            elapsed = time.time() - start_time
            logger.info(f"Reranking request completed (new=True, time={elapsed:.2f}s)")
            
            # Return result
            return RerankerResponse(
                scores=scores if not Config.CACHE_ENABLED else _reranker_cache[cache_key],
                model=request.model
            )
            
        finally:
            # Remove from in-flight requests
            del _reranker_requests_in_flight[cache_key]
    
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        # If Future was created but result not set, set exception
        if 'cache_key' in locals() and cache_key in _reranker_requests_in_flight:
            if not _reranker_requests_in_flight[cache_key].done():
                _reranker_requests_in_flight[cache_key].set_exception(e)
            del _reranker_requests_in_flight[cache_key]
        raise HTTPException(status_code=500, detail=f"Failed to rerank documents: {str(e)}")

# For Ollama API compatibility, add model info endpoint
@router.get("/api/tags")
async def get_models():
    """API endpoint for getting available models list, compatible with Ollama API"""
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