import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in src directory
env_path = Path(__file__).parent.parent / '.env'  # Look in backend directory
load_dotenv(env_path)

class Config:
    # API Keys
    HUGGING_FACE_HUB_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    # Model Settings
    MODEL_NAME = "gpt2"
    MODEL_PATH = os.getenv('MODEL_PATH', '../models')
    
    # Server Settings
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Directory Settings
    BASE_DIR = Path(__file__).parent.parent  # This will point to backend directory
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    MODELS_DIR = BASE_DIR / 'models'
    
    # New configuration options
    OFFLINE_MODE = False
    SKIP_ARTICLE_LOADING = False
    
    @classmethod
    def validate(cls):
        """Validate environment variables and configuration."""
        missing_vars = []
        
        if not cls.HUGGING_FACE_HUB_TOKEN:
            missing_vars.append('HUGGING_FACE_HUB_TOKEN')
        if not cls.NEWS_API_KEY:
            missing_vars.append('NEWS_API_KEY')
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Add validation for new configuration options
        if not hasattr(cls, 'OFFLINE_MODE'):
            cls.OFFLINE_MODE = False
        if not hasattr(cls, 'SKIP_ARTICLE_LOADING'):
            cls.SKIP_ARTICLE_LOADING = False
        
        # Create required directories
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Environment variables validated successfully") 