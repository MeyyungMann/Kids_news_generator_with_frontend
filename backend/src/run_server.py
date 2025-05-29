# cd /c/Users/meyyu/Desktop/Kids_news_generator_with_frontend/backend/src
# python run_server.py


# backend/src/run_server.py
import uvicorn
from api import app
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/server.log')
    ]
)
logger = logging.getLogger(__name__)

def verify_model_files():
    """Verify that all required model files exist."""
    model_path = Path(__file__).parent.parent / "models"
    required_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "pytorch_model.bin.index.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required model files: {', '.join(missing_files)}")
        return False
    
    return True

def verify_tokenizer():
    """Verify that the tokenizer can be loaded."""
    try:
        model_path = Path(__file__).parent.parent / "models"
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            padding_side="right",
            pad_token="<pad>",
            model_max_length=2048
        )
        logger.info("Tokenizer verification successful")
        return True
    except Exception as e:
        logger.error(f"Tokenizer verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        
        # Verify CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. Using CPU.")
        
        # Verify model files
        if not verify_model_files():
            logger.error("Model files verification failed. Please check the model directory.")
            sys.exit(1)
        
        # Verify tokenizer
        if not verify_tokenizer():
            logger.error("Tokenizer verification failed. Please check the tokenizer files.")
            sys.exit(1)
        
        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False  # Disable reload to prevent multiple initializations
        )
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)