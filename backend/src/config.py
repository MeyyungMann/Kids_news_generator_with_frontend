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
    MODEL_NAME = os.getenv('MODEL_NAME', '../models')
    MODEL_PATH = os.getenv('MODEL_PATH', '../models')
    
    # Server Settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    
    # Directory Settings
    BASE_DIR = Path(__file__).parent.parent  # This will point to backend directory
    RESULTS_DIR = BASE_DIR / 'results'
    MODELS_DIR = BASE_DIR / 'models'
    
    @classmethod
    def validate(cls):
        """Validate required environment variables."""
        missing_vars = []
        
        if not cls.HUGGING_FACE_HUB_TOKEN:
            missing_vars.append('HUGGING_FACE_HUB_TOKEN')
        if not cls.NEWS_API_KEY:
            missing_vars.append('NEWS_API_KEY')
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create required directories
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Environment variables validated successfully") 