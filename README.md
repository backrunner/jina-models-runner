# Jina Models API Service

This is a service that provides Ollama-compatible APIs, using Jina embedding and reranking models from HuggingFace Hub. The service is built with Python 3.13 and PyTorch, with MLX or ONNX Runtime (CoreML) acceleration on macOS.

## Features

- Support for [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) model to generate multi-task text embeddings
- Support for [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) model for document reranking
- Provides Ollama-compatible API interfaces
- Supports multiple embedding task types (retrieval query, retrieval passage, classification, etc.)
- Supports ONNX Runtime execution (with optional CoreML acceleration)
- Supports various ONNX quantized models (FP16, INT8, UINT8, etc.)
- Supports CUDA acceleration (NVIDIA GPU)
- Supports MPS acceleration (Apple Metal Performance Shaders)

## Installation

1. Ensure you have a Python 3.13 environment

2. Initialize the environment:

```bash
./scripts/init.sh
```

3. Activate the virtual environment:

```bash
source scripts/activate.sh
```

## Usage

### Starting the Service

```bash
./scripts/start.sh
```

The service starts by default at `http://0.0.0.0:8000`

### Environment Variables

The service configuration can be customized using the following environment variables:

- `HOST`: Service host address (default: "0.0.0.0")
- `PORT`: Service port (default: 8000)
- `MODEL_CACHE_DIR`: Model cache directory (default: "models")
- `USE_ONNX`: Whether to use ONNX Runtime (auto-detected)
- `ONNX_QUANTIZATION`: ONNX model quantization type (default: "none")
- `USE_CUDA`: Whether to use CUDA acceleration (auto-detected)
- `USE_MPS`: Whether to use MPS acceleration (auto-detected)
- `EMBEDDINGS_MODEL_ID`: Embedding model ID (default: "jinaai/jina-embeddings-v3")
- `RERANKER_MODEL_ID`: Reranker model ID (default: "jinaai/jina-reranker-v2-base-multilingual")

For example:

```bash
HOST=127.0.0.1 PORT=8080 USE_ONNX=True ONNX_QUANTIZATION=fp16 python run.py
```

### ONNX Quantized Model Options

The system supports various ONNX quantized models, which can be selected using environment variables:

- `ONNX_QUANTIZATION=none`: Use non-quantized model (model.onnx) - Default option, highest accuracy but largest file
- `ONNX_QUANTIZATION=fp16`: Use FP16 quantization (model_fp16.onnx) - Balance of accuracy and size
- `ONNX_QUANTIZATION=int8`: Use INT8 quantization (model_int8.onnx) - Small size, slightly lower accuracy
- `ONNX_QUANTIZATION=uint8`: Use UINT8 quantization (model_uint8.onnx) - Small size, slightly lower accuracy
- `ONNX_QUANTIZATION=quantized`: Use general quantization (model_quantized.onnx) - Generic quantization method
- `ONNX_QUANTIZATION=q4`: Use 4-bit quantization (model_q4.onnx) - Smallest size, but may significantly reduce accuracy
- `ONNX_QUANTIZATION=bnb4`: Use BNB 4-bit quantization (model_bnb4.onnx) - Very small size, accuracy may be affected

On Apple Silicon devices, `fp16` or `quantized` models are recommended for the best balance of performance and size.

## Model Information

### Jina Embeddings V3

This service uses the [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) model from HuggingFace Hub, which is a multilingual multi-task embedding model that supports various task types:

- `retrieval.query`: Query embeddings for asymmetric retrieval tasks
- `retrieval.passage`: Passage embeddings for asymmetric retrieval tasks
- `separation`: Embeddings for clustering and reranking applications
- `classification`: Embeddings for classification tasks
- `text-matching`: Task embeddings for quantifying similarity between two texts

### Jina Reranker V2

This service uses the [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) model from HuggingFace Hub, which is a multilingual reranker model optimized for document reranking tasks.

The model is available in various ONNX formats, including different quantized versions, allowing for optimized performance and memory usage across different hardware conditions.

## API Interfaces

### Health Check

```
GET /health
```

### Get Supported Models

```
GET /api/tags
```

### Generate Embedding Vectors

```
POST /api/embeddings
```

Request body:

```json
{
  "model": "jinaai/jina-embeddings-v3",
  "prompt": "This is text that needs to be embedded",
  "options": {
    "task": "text-matching"
  }
}
```

Available task types include: `retrieval.query`, `retrieval.passage`, `separation`, `classification`, `text-matching`

Alternatively, you can specify the task through the model name:

```json
{
  "model": "jinaai/jina-embeddings-v3-query",
  "prompt": "This is a retrieval query"
}
```

### Document Reranking

```
POST /api/rerank
```

Request body:

```json
{
  "model": "jinaai/jina-reranker-v2-base-multilingual",
  "query": "Query text",
  "documents": ["Document 1", "Document 2", "Document 3"]
}
```

## Client Examples

### Using Python to Request Embedding Vectors

```python
import requests

url = "http://localhost:8000/api/embeddings"
payload = {
    "model": "jinaai/jina-embeddings-v3",
    "prompt": "This is a test text",
    "options": {
        "task": "text-matching"
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

### Using Python to Request Document Reranking

```python
import requests

url = "http://localhost:8000/api/rerank"
payload = {
    "model": "jinaai/jina-reranker-v2-base-multilingual",
    "query": "What is the best Python framework?",
    "documents": [
        "Django is a high-level Python Web framework that encourages rapid development and clean, pragmatic design.",
        "Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.",
        "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python based on standard Python type hints."
    ]
}

response = requests.post(url, json=payload)
print(response.json())
```

## Docker Support

Build Docker image:

```bash
./scripts/docker-build.sh
```

Run using docker-compose:

```bash
docker-compose up -d
```

## Script Descriptions

This project provides several convenience scripts:

- `scripts/init.sh`: Initialize environment and dependencies
- `scripts/activate.sh`: Activate virtual environment
- `scripts/deactivate.sh`: Exit virtual environment
- `scripts/start.sh`: Start the service
- `scripts/stop.sh`: Stop the service
- `scripts/restart.sh`: Restart the service
- `scripts/docker-build.sh`: Build Docker image

## License

This project is licensed under the MIT License. Please note that Jina models themselves are subject to different licenses; please refer to each model's HuggingFace page for details. 