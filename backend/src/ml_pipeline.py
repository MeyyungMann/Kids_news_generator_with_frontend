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
from rl_system import RLSystem  # Import the RL system
import asyncio

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
        self.feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_model()
        self.image_generator = KidFriendlyImageGenerator()
        
        # System diagnostics
        self.diagnostics = self._get_system_diagnostics()
        
        # Initialize RAG system with proper error handling
        try:
            self.rag = RAGSystem()
            # Load articles immediately during initialization
            self.rag.load_news_articles()
            self.articles_loaded = True
            logger.info("RAG system initialized and articles loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            self.rag = None
            self.articles_loaded = False
        
        # Initialize RL system
        self.rl_system = RLSystem()
        
        # Load feedback data
        self.feedback_data = self._load_feedback_data()
    
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
            try:
                logger.info("Loading articles into RAG system...")
                if self.rag is None:
                    self.rag = RAGSystem()
                self.rag.load_news_articles()
                self.articles_loaded = True
                logger.info("Articles loaded successfully")
            except Exception as e:
                logger.error(f"Error loading articles: {str(e)}")
                raise RuntimeError(f"Failed to load articles: {str(e)}")

    def update_rag_with_new_article(self, article: Dict[str, Any]):
        """Update RAG system with a new article."""
        try:
            # Prepare article for RAG
            rag_doc = {
                'title': article['title'],
                'source': article['source'],
                'url': article['url'],
                'published_at': article['published_at'],
                'category': article['category'],
                'chunks': [article['content']],  # Use full content as one chunk
                'metadata': {
                    'source': article['source'],
                    'url': article['url'],
                    'published_at': article['published_at'],
                    'category': article['category']
                }
            }
            
            # Add to RAG system
            self.rag._maintain_sliding_window(
                new_documents=[article['content']],
                new_metadata=[{
                    'title': article['title'],
                    'source': article['source'],
                    'url': article['url'],
                    'category': article['category'],
                    'published_at': article['published_at']
                }]
            )
            
            # Update index
            self.rag._create_or_update_index()
            logger.info(f"Updated RAG system with new article: {article['title']}")
            
        except Exception as e:
            logger.error(f"Error updating RAG with new article: {str(e)}")

    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load and aggregate feedback data."""
        feedback_data = {
            "age_appropriate": {},
            "engagement": {},
            "clarity": {}
        }
        
        try:
            # Load feedback from all category directories
            for category_dir in self.feedback_dir.glob("*"):
                if not category_dir.is_dir():
                    continue
                    
                for feedback_file in category_dir.glob("*.json"):
                    try:
                        with open(feedback_file, "r") as f:
                            feedback = json.load(f)
                            
                            # Aggregate feedback by type and age group
                            feedback_type = feedback["feedback_type"]
                            age_group = str(feedback["age_group"])
                            
                            if age_group not in feedback_data[feedback_type]:
                                feedback_data[feedback_type][age_group] = {
                                    "total_rating": 0,
                                    "count": 0,
                                    "comments": []
                                }
                            
                            feedback_data[feedback_type][age_group]["total_rating"] += feedback["rating"]
                            feedback_data[feedback_type][age_group]["count"] += 1
                            
                            if feedback.get("comments"):
                                feedback_data[feedback_type][age_group]["comments"].append(feedback["comments"])
                    except Exception as e:
                        logger.error(f"Error loading feedback file {feedback_file}: {str(e)}")
                        continue
                        
            return feedback_data
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
            return feedback_data

    def _get_feedback_insights(self, age_group: int) -> Dict[str, Any]:
        """Get insights from feedback for a specific age group."""
        age_group_str = str(age_group)
        insights = {
            "age_appropriate": 0,
            "engagement": 0,
            "clarity": 0,
            "suggestions": []
        }
        
        # Calculate average ratings
        for feedback_type in ["age_appropriate", "engagement", "clarity"]:
            if age_group_str in self.feedback_data[feedback_type]:
                data = self.feedback_data[feedback_type][age_group_str]
                if data["count"] > 0:
                    insights[feedback_type] = data["total_rating"] / data["count"]
                    insights["suggestions"].extend(data["comments"])
        
        return insights

    async def generate_news(self, topic: str, age_group: int) -> Dict[str, Any]:
        """Generate kid-friendly news content."""
        try:
            logger.info(f"Starting news generation for topic: {topic}, age_group: {age_group}")
            
            # Validate URL if topic is a URL
            if topic.startswith(('http://', 'https://')):
                try:
                    # Basic URL validation
                    from urllib.parse import urlparse
                    parsed_url = urlparse(topic)
                    if not all([parsed_url.scheme, parsed_url.netloc]):
                        raise ValueError("Invalid URL format")
                    
                    # Try to extract content from URL
                    try:
                        import requests
                        from bs4 import BeautifulSoup
                        
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        response = requests.get(topic, headers=headers, timeout=10)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                            element.decompose()
                        
                        # Extract main content
                        article_text = soup.get_text(separator=' ', strip=True)
                        
                        if not article_text or len(article_text) < 100:
                            raise ValueError("Could not extract meaningful content from URL")
                        
                        # Use the extracted content as the topic
                        topic = article_text[:500]  # Use first 500 characters as topic
                        logger.info("Successfully extracted content from URL")
                        
                    except requests.RequestException as e:
                        logger.error(f"Error fetching URL: {str(e)}")
                        raise ValueError(f"Could not access URL: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error extracting content: {str(e)}")
                        raise ValueError(f"Error extracting content: {str(e)}")
                        
                except ValueError as e:
                    logger.error(f"URL validation error: {str(e)}")
                    raise ValueError(f"Invalid URL or content extraction failed: {str(e)}")
            
            # Simplify title for younger age groups
            if 3 <= age_group <= 6:  # Preschool
                # Remove source and complex parts
                topic = re.sub(r'\s*-\s*[^-]+$', '', topic)  # Remove source
                topic = re.sub(r'\d+\.?\d*\s*billion', '', topic)  # Remove numbers
                topic = re.sub(r'\b(urgent|warning|breaking|critical)\b', '', topic, flags=re.IGNORECASE)
                topic = re.sub(r'\b(users|devices|technology)\b', '', topic, flags=re.IGNORECASE)
                topic = f"Fun Story About {topic}"
                
            elif 7 <= age_group <= 9:  # Early Elementary
                # Keep basic structure but simplify
                topic = re.sub(r'\s*-\s*[^-]+$', '', topic)  # Remove source
                topic = re.sub(r'\d+\.?\d*\s*billion', 'many', topic)  # Replace numbers
                topic = re.sub(r'\b(urgent|warning|breaking)\b', 'important', topic, flags=re.IGNORECASE)
                topic = f"Kids News: {topic}"
            
            logger.info(f"Using simplified topic: {topic}")
            
            # Ensure RAG system is initialized and articles are loaded
            if self.rag is None:
                raise RuntimeError("RAG system not initialized")
            await self.ensure_articles_loaded()
            
            # Get feedback insights
            feedback_insights = self._get_feedback_insights(age_group)
            logger.info(f"Using feedback insights: {feedback_insights}")
            
            # Get RL-based generation guidelines
            rl_guidelines = self.rl_system.get_generation_guidelines(age_group, "general")
            logger.info(f"Using RL guidelines: {rl_guidelines}")
            
            # Search for relevant news articles with retry logic
            max_retries = 3
            retry_delay = 1  # seconds
            relevant_docs = []
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Searching for relevant documents (attempt {attempt + 1}/{max_retries})...")
                    relevant_docs = await self.rag.search(topic, k=3)
                    if relevant_docs:
                        logger.info(f"Found {len(relevant_docs)} relevant documents")
                        break
                    else:
                        logger.warning(f"No relevant documents found on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                except Exception as e:
                    logger.error(f"Error searching for documents (attempt {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        raise
            
            if not relevant_docs:
                raise RuntimeError("No relevant documents found after all retries")
            
            # Create prompt with relevant facts and feedback insights
            try:
                context = "\n".join([doc['document'] for doc in relevant_docs])
                
                # Add feedback-based instructions
                feedback_instructions = []
                if feedback_insights["age_appropriate"] < 4:
                    feedback_instructions.append("Use simpler vocabulary and shorter sentences")
                if feedback_insights["engagement"] < 4:
                    feedback_instructions.append("Add more interactive elements and questions")
                if feedback_insights["clarity"] < 4:
                    feedback_instructions.append("Provide more examples and explanations")
                
                # Add RL-based instructions
                if rl_guidelines['vocabulary_complexity'] < 0.4:
                    feedback_instructions.append("Use very simple vocabulary")
                if rl_guidelines['interactive_elements']:
                    feedback_instructions.append("Include interactive elements and questions")
                if rl_guidelines['example_count'] > 2:
                    feedback_instructions.append("Provide multiple examples")
                
                if feedback_instructions:
                    context += "\n\nBased on user feedback and learning, please ensure to:\n" + "\n".join(f"- {instruction}" for instruction in feedback_instructions)
                
                logger.info("Created context from relevant documents and feedback")
            except Exception as e:
                logger.error(f"Error creating context: {str(e)}")
                raise
            
            # Create prompt based on age group
            try:
                if 3 <= age_group <= 6:  # Preschool
                    prompt = f"""
                    You are a friendly storyteller for young children.

                    Write a complete story for a child aged {age_group} about "{topic}" using these facts:
                    {context}

                    STORY STARTS BELOW:
                    The story must:
                    1. Start with "Once upon a time" and introduce a friendly character
                    2. Use very simple words and short sentences (3-5 words)
                    3. Include basic emotions (happy, sad, excited)
                    4. **Separate each paragraph with a blank line.**
                    5. End with a happy conclusion and a simple lesson
                    6. Make sure to complete the entire story before ending
                    
                    Write exactly 3 short paragraphs. Each paragraph should be 2-3 sentences.
                    Use words that a {age_group}-year-old can understand.
                    End with "Wasn't that fun?" or "The end!"


                    """

                elif 7 <= age_group <= 9:  # Early Elementary
                    prompt = f"""
                    You are a friendly storyteller for young children.

                    Write a complete story for a child aged {age_group} about "{topic}" using these facts:
                    {context}

                    STORY STARTS BELOW:
                    The story must:
                    1. Start with an engaging opening and introduce the setting
                    2. Use simple sentences (5-8 words) and basic vocabulary
                    3. Include a character who learns something new
                    4. Add one or two simple questions for the reader
                    5. **Separate each paragraph with a blank line.**
                    6. End with a satisfying conclusion and a clear lesson
                    7. Make sure to complete the entire story before ending
                    
                    Write exactly 4 paragraphs. Each paragraph should be 3-4 sentences.
                    Use words that a {age_group}-year-old can understand.
                    Include one "Did you know?" fact.
                    """

                else:  # Upper Elementary (10-12)
                    prompt = f"""
                    You are a friendly storyteller for young children.

                    Write a complete story for a child aged {age_group} about "{topic}" using these facts:
                    {context}

                    STORY STARTS BELOW:
                    The story must:
                    1. Start with an interesting opening that grabs attention
                    2. Set up the main character and their challenge
                    3. Present the topic through an engaging narrative
                    4. Include clear explanations of complex concepts
                    5. **Separate each paragraph with a blank line.**
                    6. End with a satisfying conclusion and key takeaways
                    7. Make sure to complete the entire story before ending

                    Write exactly 5 paragraphs. Each paragraph should be 4-5 sentences.
                    Separate each paragraph with a new line.
                    Use age-appropriate vocabulary for {age_group}-year-olds.
                    """

                logger.info("Created age-appropriate prompt for model")
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
                validation_results = await self.rag.validate_facts(cleaned_text)
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
        
        # Find the story content after the delimiter
        if "STORY STARTS BELOW:" in text:
            text = text.split("STORY STARTS BELOW:")[1].strip()
        
        # Remove any repeated dashes
        text = re.sub(r'-{3,}', '', text)
        
        # Split into paragraphs and clean each one
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # Skip paragraphs that are just instructions
            if any(phrase in paragraph.lower() for phrase in [
                'write', 'story', 'must', 'should', 'need to', 'have to',
                'ensure', 'make sure', 'remember to', 'don\'t forget to',
                'use', 'include', 'present', 'show', 'set up', 'explain',
                'introduce', 'add', 'provide', 'follow', 'format', 'structure',
                'guidelines', 'instructions', 'requirements', 'elements',
                'content', 'markers', 'patterns', 'phrases', 'words',
                'sentences', 'paragraphs', 'sections', 'parts', 'pieces',
                'bits', 'chunks', 'blocks', 'segments', 'portions',
                'fragments', 'snippets', 'excerpts', 'passages', 'texts',
                'contents', 'materials', 'resources', 'sources', 'references',
                'citations', 'quotes', 'facts', 'information', 'data',
                'details', 'points', 'items', 'elements', 'components'
            ]):
                continue
            
            # Clean up the paragraph
            paragraph = re.sub(r'\s+', ' ', paragraph).strip()
            if paragraph:
                cleaned_paragraphs.append(paragraph)
        
        # Join paragraphs with double newlines
        cleaned_text = '\n\n'.join(cleaned_paragraphs)
        
        # Final cleanup
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
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
                    "original_article": article_data,
                    "rl_guidelines": self.rl_system.get_generation_guidelines(
                        article_data.get("age_group", 7),
                        category
                    )
                }, f, indent=2)
            
            logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            raise

    def update_with_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Update the system with new feedback."""
        try:
            # Update RL system
            self.rl_system.update_with_feedback(feedback_data)
            
            # Save feedback
            category = feedback_data.get("category", "general").capitalize()
            feedback_dir = self.feedback_dir / category
            feedback_dir.mkdir(parents=True, exist_ok=True)
            
            feedback_file = feedback_dir / f"{feedback_data['article_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Feedback saved to {feedback_file}")
            
        except Exception as e:
            logger.error(f"Error updating with feedback: {str(e)}")
            raise 