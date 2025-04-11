import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_file=None, log_level=logging.INFO):
    """设置应用的日志配置"""
    # 创建日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有的处理程序
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加控制台处理程序
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理程序
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 添加文件处理程序，最大10MB，保留3个备份
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=3
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger 