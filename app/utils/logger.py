import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_file=None, log_level=logging.INFO):
    """Set up application logging configuration"""
    # Create log format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # If log file is specified, add file handler
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Add file handler, max 10MB, keep 3 backups
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=3
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger 