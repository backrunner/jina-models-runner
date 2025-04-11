import uvicorn
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from .api.router import router
from .config import Config
from .utils.logger import setup_logger

# 设置日志
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger(
    log_file=os.path.join(log_dir, "app.log"),
    log_level=logging.INFO
)

# 创建FastAPI应用
app = FastAPI(
    title="Jina Models API Service",
    description="An Ollama-compatible API server for Jina embeddings and reranker models",
    version="0.1.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 包含路由
app.include_router(router)

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# 主函数
def main():
    """启动服务器的主入口点"""
    logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
    logger.info(f"MLX acceleration: {Config.USE_MLX}")
    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        workers=1,
        log_level="info"
    )

if __name__ == "__main__":
    main() 