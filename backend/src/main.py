#!/usr/bin/env python3
import logging
import argparse
import sys
from pathlib import Path
from huggingface_hub import login, HfFolder, snapshot_download
from ml_pipeline import KidsNewsGenerator
from news_api_handler import NewsAPIHandler
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_internet_connection():
    """Check if we can connect to HuggingFace."""
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def verify_hf_auth():
    """Verify Hugging Face authentication."""
    try:
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            logger.error("""
Hugging Face token not found. Please follow these steps:

1. Get your HuggingFace token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read access

2. Add your token to .env file:
   HUGGING_FACE_HUB_TOKEN=your_token_here

3. Or set the environment variable:
   export HUGGING_FACE_HUB_TOKEN=your_token_here

After adding your token, run the script again.
""")
            return False
        
        # Try to login with the token
        login(token)
        logger.info("Successfully authenticated with HuggingFace")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace: {str(e)}")
        return False

def download_model_offline(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", cache_dir: str = "models"):
    """Download model files for offline use."""
    try:
        # First verify authentication
        if not verify_hf_auth():
            return False

        logger.info(f"Downloading model {model_name} for offline use...")
        
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        # Verify the download
        model_path = Path(cache_dir) / "mistral_local"
        if not model_path.exists():
            logger.error("Model files not found after download")
            return False
            
        logger.info(f"Model downloaded successfully to {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def setup_environment():
    """Set up the environment and check requirements."""
    try:
        # Check if running in a virtual environment
        if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix'):
            logger.warning("Not running in a virtual environment. It's recommended to use one.")
        
        # Create necessary directories
        Path("results").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        # Check internet connection
        if not check_internet_connection():
            logger.warning("No internet connection. Will try to use cached models.")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        return False

def verify_model_exists():
    """Verify if the model exists in the correct location."""
    # Check the Hugging Face cache location
    model_paths = [
        Path("models/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/latest"),
        Path("models/mistral_local"),
        Path("models/models--mistralai--Mistral-7B-Instruct-v0.2")
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            logger.info(f"Found model in: {model_path}")
            # Set the model path as an environment variable for the KidsNewsGenerator
            os.environ["MODEL_PATH"] = str(model_path)
            return True
    
    logger.error("""
Model not found in any of the expected locations. Please ensure the model files are in one of these locations:
- models/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/latest
- models/mistral_local
- models/models--mistralai--Mistral-7B-Instruct-v0.2
""")
    return False

async def run_pipeline(args):
    """Run the news generation pipeline."""
    try:
        # Verify model exists first
        if not verify_model_exists():
            logger.error("Cannot proceed without model. Please download the model first.")
            return []

        has_internet = check_internet_connection()
        if not has_internet:
            logger.warning("No internet connection. Will try to use cached models.")
        
        logger.info("Initializing news generator...")
        generator = KidsNewsGenerator(offline_mode=not has_internet)
        generated_results = []
        
        if has_internet:
            logger.info("Initializing news fetcher...")
            news_fetcher = NewsAPIHandler()
            logger.info(f"Fetching recent {args.category} articles...")
            articles = await news_fetcher.fetch_articles(
                category=args.category,
                days=3,
                max_articles=args.max_articles
            )
            if not articles:
                logger.error(f"No articles found for category: {args.category}")
                return []
            logger.info("Adding articles to RAG system...")
            for article in articles:
                generator.rag.add_documents([article["content"]])
            logger.info("\nGenerating kid-friendly news...")
            for article in articles[:args.max_articles]:
                result = generator.generate_news(
                    topic=article["title"],
                    age_group=args.age_group
                )
                generator.save_summary(result, article)
                generated_results.append({
                    'title': article['title'],
                    'source': article['source'],
                    'kid_friendly': result["text"],
                    'safety_score': result['safety_score']
                })
                logger.info(f"\nOriginal Title: {article['title']}")
                logger.info(f"Source: {article['source']}")
                logger.info("\nKid-Friendly Version:")
                logger.info(result["text"])
                logger.info(f"\nSafety Score: {result['safety_score']:.2f}")
                logger.info("-" * 80)
        else:
            logger.info("Running in offline mode with cached models...")
            # Add offline mode handling here if needed
            # For now, just return empty
            return []
        logger.info(f"\nPipeline completed successfully. Results saved in: {generator.summaries_dir}")
        return generated_results
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

def display_menu():
    """Display the main menu."""
    print("\n=== Kids-Friendly News Generator ===")
    print("1. Generate Science News")
    print("2. Generate Technology News")
    print("3. Generate Health News")
    print("4. Generate Environment News")
    print("5. Generate Economy News")
    print("6. Download Model for Offline Use")
    print("7. System Diagnostics")
    print("8. Exit")
    return input("\nSelect an option (1-8): ")

def select_age_group():
    """Select target age group."""
    print("\n=== Select Age Group ===")
    print("1. Ages 3-6 (Preschool)")
    print("2. Ages 7-9 (Early Elementary)")
    print("3. Ages 10-12 (Upper Elementary)")
    while True:
        try:
            choice = input("\nSelect age group (1-3): ")
            if choice == '1':
                return 3  # Start of Preschool range
            elif choice == '2':
                return 7  # Start of Early Elementary range
            elif choice == '3':
                return 10  # Start of Upper Elementary range
            else:
                print("Invalid option. Please select 1-3.")
        except ValueError:
            print("Please enter a valid number.")

def run_diagnostics():
    """Run and display system diagnostics."""
    try:
        # Initialize generator to get diagnostics
        generator = KidsNewsGenerator()
        
        # Display detailed diagnostics
        print("\n=== System Diagnostics ===")
        print(f"Device: {generator.device}")
        print(f"Model: {generator.model_name}")
        
        if generator.diagnostics:
            print("\nSystem Information:")
            print(f"OS: {generator.diagnostics['system_info']['os']} {generator.diagnostics['system_info']['os_version']}")
            print(f"Python: {generator.diagnostics['system_info']['python_version']}")
            print(f"CPU Cores: {generator.diagnostics['system_info']['cpu_count']}")
            print(f"Available Memory: {generator.diagnostics['system_info']['available_memory'] / (1024**3):.1f} GB")
            
            if generator.diagnostics['gpu_info']['available']:
                print("\nGPU Information:")
                for device in generator.diagnostics['gpu_info']['devices']:
                    print(f"\nGPU {device['id']}: {device['name']}")
                    print(f"  Memory: {device['memory_free']/1024:.1f} GB free of {device['memory_total']/1024:.1f} GB")
                    print(f"  Temperature: {device['temperature']}°C")
                    print(f"  Load: {device['load']:.1f}%")
            else:
                print("\nNo GPU available")
        
        input("\nPress Enter to continue...")
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {str(e)}")
        input("\nPress Enter to continue...")

async def run_pipeline_with_category(category: str, offline_mode: bool = False):
    """Run pipeline for a specific category."""
    try:
        # Get age group selection
        age_group = select_age_group()
        
        # Adjust content based on age group using consistent ranges
        if 3 <= age_group <= 6:  # Preschool (3-6)
            max_articles = 2  # Fewer articles for younger children
            max_length = 200  # Shorter content
            temperature = 0.8  # More creative
        elif 7 <= age_group <= 9:  # Early Elementary (7-9)
            max_articles = 3
            max_length = 512
            temperature = 0.7
        elif 10 <= age_group <= 12:  # Upper Elementary (10-12)
            max_articles = 3
            max_length = 512
            temperature = 0.7
        else:
            # Default to middle range if age group is invalid
            max_articles = 3
            max_length = 512
            temperature = 0.7
        
        # Map age group to reading level using consistent ranges
        if 3 <= age_group <= 6:
            reading_level = "Preschool (3-6)"
        elif 7 <= age_group <= 9:
            reading_level = "Early Elementary (7-9)"
        elif 10 <= age_group <= 12:
            reading_level = "Upper Elementary (10-12)"
        else:
            reading_level = "Early Elementary (7-9)"  # Default to middle range
        
        args = argparse.Namespace(
            category=category,
            age_group=age_group,
            max_articles=max_articles,
            days=3,
            offline=offline_mode,
            max_length=max_length,
            temperature=temperature
        )
        
        # Check if model exists
        if not verify_model_exists():
            logger.error("Model not found. Please download the model first (Option 6).")
            return
        
        # Check internet connection
        has_internet = check_internet_connection()
        if not has_internet and not offline_mode:
            logger.warning("No internet connection. Switching to offline mode...")
            offline_mode = True
        
        # Run pipeline
        results = await run_pipeline(args)
        
        if results:
            print("\n=== Kid-Friendly News Summaries ===")
            print(f"Target Age Group: {reading_level}")
            for i, item in enumerate(results, 1):
                print(f"\n{i}. {item['title']}")
                print(f"Source: {item['source']}")
                print(f"Kid-Friendly Version:\n{item['kid_friendly']}")
                print(f"Safety Score: {item['safety_score']:.2f}")
                print("-" * 60)
        else:
            print("\nNo kid-friendly news was generated.")
            
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")

def main():
    """Main entry point."""
    try:
        # Set up environment
        setup_environment()
        
        while True:
            try:
                choice = display_menu()
                
                if choice == '1':
                    asyncio.run(run_pipeline_with_category('science'))
                elif choice == '2':
                    asyncio.run(run_pipeline_with_category('technology'))
                elif choice == '3':
                    asyncio.run(run_pipeline_with_category('health'))
                elif choice == '4':
                    asyncio.run(run_pipeline_with_category('environment'))
                elif choice == '5':
                    asyncio.run(run_pipeline_with_category('economy'))
                elif choice == '6':
                    if download_model_offline("mistralai/Mistral-7B-Instruct-v0.2"):
                        logger.info("Model downloaded successfully. You can now run in offline mode.")
                    else:
                        logger.error("Failed to download model.")
                elif choice == '7':
                    run_diagnostics()
                elif choice == '8':
                    logger.info("Exiting program...")
                    break
                else:
                    print("Invalid option. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                continue
            except Exception as e:
                logger.error(f"Error in menu option: {str(e)}")
                input("\nPress Enter to continue...")
            
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 