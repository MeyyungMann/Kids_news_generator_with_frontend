import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from image_generator import KidFriendlyImageGenerator
from config import Config
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KidsNewsGenerator:
    def __init__(self, offline_mode: bool = False):
        """Initialize the news generator."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = Config.MODEL_NAME
        self.offline_mode = offline_mode
        
        # Set up directories using config
        self.base_dir = Config.RESULTS_DIR
        self.summaries_dir = self.base_dir / "summaries"
        self.images_dir = self.base_dir / "images"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_model()
        self.image_generator = KidFriendlyImageGenerator()
        
        # System diagnostics
        self.diagnostics = self._get_system_diagnostics()
        
        # Initialize RAG system without loading articles
        self.rag = RAGSystem()
        self.articles_loaded = False
    
    def _init_model(self):
        """Initialize the language model."""
        try:
            model_path = Path(__file__).parent.parent / 'models'
            logger.info(f"Loading local model from: {model_path}")
            
            # Check if model files exist
            required_files = [
                'config.json',
                'pytorch_model.bin.index.json',
                'tokenizer.json',
                'tokenizer_config.json',
                'special_tokens_map.json',
                'generation_config.json'
            ]
            
            # Check for split model files
            model_files = list(model_path.glob('pytorch_model-*-of-*.bin'))
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            logger.info(f"Found {len(model_files)} model files")
            
            # Check CUDA availability and device info
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info("Tokenizer loaded successfully")
            
            # Load model with explicit device mapping
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                offload_folder=str(model_path / "offload")  # Add offload folder
            )
            
            # Log model device information
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            logger.error(f"Model path: {model_path}")
            logger.error(f"Directory contents: {list(model_path.glob('*'))}")
            raise
    
    def _get_system_diagnostics(self) -> Dict[str, Any]:
        """Get system diagnostics."""
        try:
            import psutil
            import platform
            
            return {
                "system_info": {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "python_version": platform.python_version(),
                    "cpu_count": psutil.cpu_count(),
                    "available_memory": psutil.virtual_memory().available
                },
                "gpu_info": {
                    "available": torch.cuda.is_available(),
                    "devices": [
                        {
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_total": torch.cuda.get_device_properties(i).total_memory,
                            "memory_free": torch.cuda.memory_allocated(i),
                            "temperature": 0,  # Add GPU temperature monitoring if available
                            "load": 0  # Add GPU load monitoring if available
                        }
                        for i in range(torch.cuda.device_count())
                    ] if torch.cuda.is_available() else []
                }
            }
        except Exception as e:
            logger.error(f"Error getting system diagnostics: {str(e)}")
            return {}
    
    def ensure_articles_loaded(self):
        """Ensure articles are loaded into the RAG system."""
        if not self.articles_loaded:
            logger.info("Loading articles into RAG system...")
            self.rag.load_news_articles()
            self.articles_loaded = True
    
    def generate_news(self, topic: str, age_group: int) -> Dict[str, Any]:
        """Generate kid-friendly news content."""
        try:
            logger.info(f"Starting news generation for topic: {topic}, age group: {age_group}")
            
            # Ensure articles are loaded before searching
            self.ensure_articles_loaded()
            
            # First, search for relevant news articles
            logger.info("Searching for relevant documents...")
            try:
                relevant_docs = self.rag.search(topic, k=3)
                logger.info(f"Found {len(relevant_docs)} relevant documents")
            except Exception as e:
                logger.error(f"Error searching for relevant documents: {str(e)}")
                raise
            
            # Create prompt with relevant facts
            try:
                context = "\n".join([doc['document'] for doc in relevant_docs])
                logger.info("Created context from relevant documents")
            except Exception as e:
                logger.error(f"Error creating context: {str(e)}")
                raise
            
            # Create prompt based on age group
            try:
                if 3 <= age_group <= 6:  # Preschool
                    prompt = f"""You are a children's storyteller. Write a story about "{topic}" using these facts:
                    {context}
                    
                    The story must:
                    1. Start with "Once upon a time" and introduce a friendly character
                    2. Explain the topic in simple terms
                    3. End with a happy conclusion
                    
                    Use very simple words and write exactly 3 paragraphs. End with "Wasn't that fun?" or "The end!"
                 
                    """

                elif 7 <= age_group <= 9:  # Early Elementary
                    prompt = f"""You are a children's storyteller. Write a story about "{topic}" using these facts:
                    {context}
                    
                    The story must:
                    1. Start with an engaging opening and introduce the setting
                    2. Present the topic through a simple adventure
                    3. Show how the character learns about the topic
                    4. End with a satisfying conclusion
                    
                    Use clear, simple language and write exactly 4 paragraphs. Make sure to tie everything together at the end.
                   
                    """

                else:  # Upper Elementary
                    prompt = f"""You are a children's storyteller. Write a story about "{topic}" using these facts:
                    {context}
                    
                    The story must:
                    1. Start with an interesting opening
                    2. Set up the main character and their challenge
                    3. Present the topic through an engaging narrative
                    4. Include key facts and explanations
                    5. End with a satisfying conclusion
                    
                    Use age-appropriate vocabulary and write exactly 5 paragraphs.
                    
                    After the story, add a "Glossary" section with exactly 3 key terms from the story, formatted like this:
                    Glossary:
                    - Term 1: Simple definition
                    - Term 2: Simple definition
                    - Term 3: Simple definition
                    
                    """

                logger.info("Created prompt for model")
            except Exception as e:
                logger.error(f"Error creating prompt: {str(e)}")
                raise
            
            # Generate content
            logger.info("Tokenizing input...")
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                logger.info(f"Input shape: {inputs['input_ids'].shape}")
                logger.info(f"Input device: {inputs['input_ids'].device}")
            except Exception as e:
                logger.error(f"Error during tokenization: {str(e)}")
                raise
            
            logger.info("Generating content...")
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_length=8192,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                logger.info(f"Output shape: {outputs.shape}")
                logger.info("Content generation completed")
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                logger.error(f"Model device: {next(self.model.parameters()).device}")
                raise
            
            logger.info("Decoding generated text...")
            try:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Generated text length: {len(generated_text)}")
                logger.info(f"First 100 chars of generated text: {generated_text[:100]}")
            except Exception as e:
                logger.error(f"Error during decoding: {str(e)}")
                raise
            
            # Clean the generated text
            try:
                cleaned_text = self._clean_generated_text(generated_text)
                logger.info(f"Cleaned text length: {len(cleaned_text)}")
                logger.info(f"First 100 chars of cleaned text: {cleaned_text[:100]}")
            except Exception as e:
                logger.error(f"Error cleaning text: {str(e)}")
                raise
            
            # Validate facts against news articles
            logger.info("Validating facts...")
            try:
                validation_results = self.rag.validate_facts(cleaned_text)
                logger.info(f"Fact validation completed with score: {validation_results['overall_score']}")
            except Exception as e:
                logger.error(f"Error during fact validation: {str(e)}")
                raise
            
            # Prepare result
            try:
                result = {
                    "text": cleaned_text,
                    "safety_score": validation_results['overall_score'],
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "age_group": age_group,
                    "fact_validation": validation_results,
                    "source_articles": [
                        {
                            'title': doc['metadata']['title'],
                            'source': doc['metadata']['source'],
                            'url': doc['metadata']['url']
                        }
                        for doc in relevant_docs
                    ]
                }
                
                logger.info("Successfully generated news content")
                return result
            except Exception as e:
                logger.error(f"Error preparing result: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error generating news: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            raise
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean the generated text by removing prompt-related content and formatting paragraphs."""
        if not text:
            return ""
        
        # Split into lines and filter out prompt-related content
        lines = text.split('\n')
        cleaned_lines = []
        
        # List of story starters to identify where the actual story begins
        story_starters = [
            "once upon a time",
            "let me tell you a story about",
            "here's a fun story about"
        ]
        
        # List of phrases that indicate prompt content to remove
        prompt_phrases = [
            # Role and context
            'you are',
            'children\'s storyteller',
            'storyteller',
            
            # Story instructions
            'write a story',
            'write a complete story',
            'write exactly',
            'the story must',
            'the story must:',
            'story must:',
            
            # Content markers
            'using these facts',
            'using these facts:',
            'use these facts',
            'use these facts:',
            'facts to include',
            
            # Language requirements
            'use very simple words',
            'use clear, simple language',
            'use age-appropriate vocabulary',
            'use simple vocabulary',
            'use detailed but clear explanations',
            
            # Structure markers
            'start with',
            'end with',
            'make sure to',
            'include',
            'present',
            'show how',
            'set up',
            'explain',
            'introduce',
            
            # Content elements
            'friendly character',
            'friendly characters',
            'simple terms',
            'simple adventure',
            'engaging narrative',
            'interesting opening',
            'happy conclusion',
            'happy ending',
            'satisfying conclusion',
            'proper conclusion',
            'brief glossary',
            'key terms',
            'glossary of',
            
            # Formatting
            'paragraphs',
            'paragraph',
            'exactly 3 paragraphs',
            'exactly 4 paragraphs',
            'exactly 5 paragraphs',
            
            # Common instruction words
            'must',
            'should',
            'need to',
            'have to',
            'ensure',
            'make sure',
            'remember to',
            'don\'t forget to'
        ]
        
        # Process each line
        current_paragraph = []
        story_started = False
        
        for line in lines:
            line = line.strip()
            if not line:
                # If we have content in current paragraph, add it to cleaned lines
                if current_paragraph:
                    cleaned_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            lower_line = line.lower()
            
            # Check if this line starts the story
            if not story_started and any(starter in lower_line for starter in story_starters):
                story_started = True
            
            # Skip lines that are clearly part of the prompt or before the story starts
            if not story_started:
                continue
                
            # Skip lines that match any prompt phrase
            if any(phrase in lower_line for phrase in prompt_phrases):
                continue
                
            # Skip lines that match numbered list pattern
            if re.match(r'^\d+[\.\)]', line):
                continue
            
            # Add line to current paragraph
            current_paragraph.append(line)
        
        # Add any remaining paragraph
        if current_paragraph:
            cleaned_lines.append(' '.join(current_paragraph))
        
        # If we have no content after cleaning, return the original text
        if not cleaned_lines:
            logger.warning("No content after cleaning, returning original text")
            return text
        
        # Join paragraphs with double newlines for proper spacing
        cleaned_text = '\n\n'.join(cleaned_lines)
        
        # Clean up any extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Ensure proper paragraph spacing
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        return cleaned_text
    
    def save_summary(self, result: Dict[str, Any], article_data: Dict[str, Any]) -> None:
        """Save the generated summary and metadata."""
        try:
            # Create safe filename
            safe_title = "".join(c if c.isalnum() else "_" for c in article_data.get("title", "article"))
            timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            # Get category
            category = article_data.get("category", "general").capitalize()
            
            # Create category directory if it doesn't exist
            category_dir = self.summaries_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            summary_path = category_dir / f"{safe_title}_{timestamp}.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "topic": article_data.get("title", ""),
                    "text": result["text"],  # The actual generated article text
                    "safety_score": result.get("safety_score", 0.0),
                    "timestamp": timestamp,
                    "age_group": article_data.get("age_group", 7),
                    "category": category,
                    "original_article": article_data
                }, f, indent=2)
            
            logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            raise 