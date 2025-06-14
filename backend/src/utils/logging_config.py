import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(name: str = None, log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Optional specific log file name. If not provided, uses default naming.
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = logs_dir / f"{log_file}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 