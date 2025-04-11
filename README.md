# Jina 模型 API 服务

这是一个提供Ollama兼容API的服务，使用来自HuggingFace Hub的Jina嵌入和重排序模型。该服务使用Python 3.13和PyTorch，在macOS上可以通过MLX或ONNX Runtime(CoreML)加速。

## 特性

- 支持[jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)模型生成多任务文本嵌入
- 支持[jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)模型进行文档重排序
- 提供Ollama兼容的API接口
- 支持多种嵌入任务类型（检索查询、检索段落、分类等）
- 支持ONNX Runtime执行(可选CoreML加速)
- 支持多种ONNX量化模型（FP16, INT8, UINT8等）
- 支持CUDA加速（NVIDIA GPU）
- 支持MPS加速（Apple Metal Performance Shaders）

## 安装

1. 确保你有Python 3.13环境

2. 初始化环境:

```bash
./scripts/init.sh
```

3. 激活虚拟环境:

```bash
source scripts/activate.sh
```

## 使用方法

### 启动服务

```bash
./scripts/start.sh
```

服务默认在 `http://0.0.0.0:8000` 启动

### 环境变量

可以通过以下环境变量自定义服务配置:

- `HOST`: 服务主机地址 (默认: "0.0.0.0")
- `PORT`: 服务端口 (默认: 8000)
- `MODEL_CACHE_DIR`: 模型缓存目录 (默认: "models")
- `USE_ONNX`: 是否使用ONNX Runtime (自动检测)
- `ONNX_QUANTIZATION`: ONNX模型量化类型 (默认: "none")
- `USE_CUDA`: 是否使用CUDA加速 (自动检测)
- `USE_MPS`: 是否使用MPS加速 (自动检测)
- `EMBEDDINGS_MODEL_ID`: 嵌入模型ID (默认: "jinaai/jina-embeddings-v3")
- `RERANKER_MODEL_ID`: 重排序模型ID (默认: "jinaai/jina-reranker-v2-base-multilingual")

例如:

```bash
HOST=127.0.0.1 PORT=8080 USE_ONNX=True ONNX_QUANTIZATION=fp16 python run.py
```

### ONNX量化模型选项

系统支持多种ONNX量化模型，可以通过环境变量选择：

- `ONNX_QUANTIZATION=none`: 使用无量化模型 (model.onnx) - 默认选项，最高精度但文件最大
- `ONNX_QUANTIZATION=fp16`: 使用FP16量化 (model_fp16.onnx) - 平衡精度与大小
- `ONNX_QUANTIZATION=int8`: 使用INT8量化 (model_int8.onnx) - 体积小，略微降低精度
- `ONNX_QUANTIZATION=uint8`: 使用UINT8量化 (model_uint8.onnx) - 体积小，略微降低精度
- `ONNX_QUANTIZATION=quantized`: 使用通用量化 (model_quantized.onnx) - 通用量化方式
- `ONNX_QUANTIZATION=q4`: 使用4位量化 (model_q4.onnx) - 最小体积，但可能明显降低精度
- `ONNX_QUANTIZATION=bnb4`: 使用BNB 4位量化 (model_bnb4.onnx) - 极小体积，精度可能受影响

推荐在Apple Silicon设备上使用`fp16`或`quantized`模型获得最佳性能和体积平衡。

## 模型说明

### Jina Embeddings V3

本服务使用的是HuggingFace Hub上的[jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)模型，这是一个多语言多任务嵌入模型，支持多种任务类型：

- `retrieval.query`: 用于非对称检索任务的查询嵌入
- `retrieval.passage`: 用于非对称检索任务的段落嵌入
- `separation`: 用于聚类和重排序应用的嵌入
- `classification`: 用于分类任务的嵌入
- `text-matching`: 用于量化两文本相似度的任务嵌入

### Jina Reranker V2

本服务使用的是HuggingFace Hub上的[jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)模型，这是一个多语言重排序模型，专为文档重排序任务优化。

模型提供了多种ONNX格式，包括不同的量化版本，允许在不同硬件条件下优化性能和内存使用。

## API接口

### 健康检查

```
GET /health
```

### 获取支持的模型

```
GET /api/tags
```

### 生成嵌入向量

```
POST /api/embeddings
```

请求体:

```json
{
  "model": "jinaai/jina-embeddings-v3",
  "prompt": "这是一段需要嵌入的文本",
  "options": {
    "task": "text-matching"
  }
}
```

可用的任务类型包括：`retrieval.query`, `retrieval.passage`, `separation`, `classification`, `text-matching`

或者，可以通过模型名称指定任务：

```json
{
  "model": "jinaai/jina-embeddings-v3-query",
  "prompt": "这是一个检索查询"
}
```

### 文档重排序

```
POST /api/rerank
```

请求体:

```json
{
  "model": "jinaai/jina-reranker-v2-base-multilingual",
  "query": "查询文本",
  "documents": ["文档1", "文档2", "文档3"]
}
```

## 客户端示例

### 使用Python请求嵌入向量

```python
import requests

url = "http://localhost:8000/api/embeddings"
payload = {
    "model": "jinaai/jina-embeddings-v3",
    "prompt": "这是一段测试文本",
    "options": {
        "task": "text-matching"
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

### 使用Python请求文档重排序

```python
import requests

url = "http://localhost:8000/api/rerank"
payload = {
    "model": "jinaai/jina-reranker-v2-base-multilingual",
    "query": "最好的Python框架是什么?",
    "documents": [
        "Django是一个高级Python Web框架，它鼓励快速开发和干净，实用的设计。",
        "Flask是一个轻量级的WSGI Web应用框架。它旨在使入门变得快速简单，能够扩展到复杂应用。",
        "FastAPI是一个现代、快速（高性能）的web框架，用于基于标准Python类型提示构建API。"
    ]
}

response = requests.post(url, json=payload)
print(response.json())
```

## Docker支持

构建Docker镜像:

```bash
./scripts/docker-build.sh
```

使用docker-compose运行:

```bash
docker-compose up -d
```

## 脚本说明

本项目提供了多个便捷脚本：

- `scripts/init.sh`: 初始化环境和依赖
- `scripts/activate.sh`: 激活虚拟环境
- `scripts/deactivate.sh`: 退出虚拟环境
- `scripts/start.sh`: 启动服务
- `scripts/stop.sh`: 停止服务
- `scripts/restart.sh`: 重启服务
- `scripts/docker-build.sh`: 构建Docker镜像

## 许可证

此项目采用MIT许可证。请注意，Jina模型本身受不同的许可证约束，详情请参考各模型的HuggingFace页面。 