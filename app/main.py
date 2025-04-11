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

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger(
    log_file=os.path.join(log_dir, "app.log"),
    log_level=logging.INFO
)

# Create FastAPI application
app = FastAPI(
    title="Jina Models API Service",
    description="An Ollama-compatible API server for Jina embeddings and reranker models",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routes
app.include_router(router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Main function
def main():
    """Main entry point for starting the server"""
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