import torch
from diffusers import StableDiffusionPipeline
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from PIL import Image
import os
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KidFriendlyImageGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None
    ):
        """
        Initialize the image generator.
        
        Args:
            model_id: Stable Diffusion model ID
            device: Device to run the model on
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Update paths to use absolute paths
        self.base_dir = Path(__file__).parent.parent / "results"  # This will point to backend/results
        self.images_dir = self.base_dir / "images"
        self.summaries_dir = self.base_dir / "summaries"
        
        # Create category directories
        categories = ["Science", "Technology", "Environment", "Health", "Economy"]
        for category in categories:
            (self.images_dir / category).mkdir(parents=True, exist_ok=True)
            (self.summaries_dir / category).mkdir(parents=True, exist_ok=True)
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.has_gpu = torch.cuda.is_available()
        
        # Initialize environment
        if not self._initialize_environment():
            raise RuntimeError("Failed to initialize environment")
        
        # Initialize the pipeline
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize the Stable Diffusion pipeline."""
        try:
            logger.info(f"Loading Stable Diffusion model: {self.model_id}")
            
            # Calculate available GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Reserve 10% for buffers
                max_memory = {0: int(gpu_memory * 0.9)}
            else:
                max_memory = None
            
            # Load model with memory optimization
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for kid-friendly content
                use_safetensors=True,  # Use safetensors for better compatibility
                variant="fp16" if self.device == "cuda" else None,  # Use fp16 variant for GPU
                low_cpu_mem_usage=True
            )
            
            # Move pipeline to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimization
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
                if max_memory:
                    self.pipeline.enable_model_cpu_offload()
            
            logger.info("Stable Diffusion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Stable Diffusion model: {str(e)}")
            raise
    
    def _create_prompt(self, content: Dict[str, Any], age_group: int) -> str:
        """
        Create a kid-friendly prompt for image generation.
        
        Args:
            content: Generated news content
            age_group: Target age group
            
        Returns:
            Formatted prompt for image generation
        """
        # Extract key elements from content
        topic = content.get("topic", "")
        text = content.get("text", "")
        category = content.get("category", "").lower()
        
        # Create age-appropriate style using consistent ranges
        if 3 <= age_group <= 6:  # Preschool (3-6)
            style = "cute cartoon style, bright colors, simple shapes, friendly characters, very simple and clear, perfect for preschoolers, no text, no letters, no words"
            complexity = "very simple, basic concepts, friendly and approachable, visual only"
        elif 7 <= age_group <= 9:  # Early Elementary (7-9)
            style = "colorful cartoon style, fun characters, educational illustration, engaging and playful, perfect for early readers, no text, no letters, no words"
            complexity = "clear and engaging, educational but fun, easy to understand, visual only"
        elif 10 <= age_group <= 12:  # Upper Elementary (10-12)
            style = "engaging illustration style, educational, clear and colorful, slightly more detailed, perfect for older elementary students, no text, no letters, no words"
            complexity = "more detailed but still kid-friendly, educational and informative, visual only"
        else:
            # Default to middle range if age group is invalid
            style = "colorful cartoon style, fun characters, educational illustration, engaging and playful, perfect for early readers, no text, no letters, no words"
            complexity = "clear and engaging, educational but fun, easy to understand, visual only"
        
        # Add category-specific elements
        category_elements = {
            "science": "scientific concepts, experiments, nature, space, animals, visual representation only",
            "technology": "computers, robots, gadgets, innovation, digital world, visual representation only",
            "health": "healthy habits, body, nutrition, exercise, wellness, visual representation only",
            "environment": "nature, animals, plants, conservation, weather, visual representation only",
            "economy": "money, saving, shopping, business, community, visual representation only"
        }
        
        category_element = category_elements.get(category, "")
        
        # Create the prompt
        prompt = f"Create a kid-friendly illustration about {topic}. {text[:100]}... Style: {style}, {complexity}, {category_element}, kid-friendly, educational, safe for children, visual only, no text, no letters, no words, no writing"
        
        # Add negative prompt to ensure kid-friendly content and no text
        negative_prompt = "text, letters, words, writing, numbers, symbols, signs, labels, captions, scary, violent, inappropriate, complex, realistic, photorealistic, dark, scary, frightening, text, words, letters, writing, numbers, symbols, signs, labels, captions"
        
        return prompt, negative_prompt
    
    def generate_image(
        self,
        content: Dict[str, Any],
        age_group: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """
        Generate a kid-friendly image based on the content.
        
        Args:
            content: Generated news content
            age_group: Target age group
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            
        Returns:
            Dictionary containing image path and metadata
        """
        try:
            # Create prompt
            prompt, negative_prompt = self._create_prompt(content, age_group)
            
            # Generate image
            logger.info("Generating image...")
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            # Save image
            timestamp = content.get("timestamp", "")
            safe_title = "".join(c if c.isalnum() else "_" for c in content.get("topic", "image"))
            
            # Get category from content
            category = content.get("category", "general").capitalize()
            
            # Save image in category-specific directory
            image_path = self.images_dir / category / f"{safe_title}_{timestamp}.png"
            image.save(image_path)
            
            # Save metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "age_group": age_group,
                "topic": content.get("topic", ""),
                "timestamp": timestamp,
                "image_path": str(image_path)
            }
            
            # Save metadata in category-specific directory
            metadata_path = self.summaries_dir / category / f"{safe_title}_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Image saved to {image_path}")
            
            return {
                "image_path": str(image_path),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

    def _initialize_environment(self):
        """Initialize environment for model loading."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create offload directory if it doesn't exist
            offload_dir = Path("offload")
            offload_dir.mkdir(exist_ok=True)
            
            # Set environment variables for better memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
            return True
        except Exception as e:
            logger.error(f"Error initializing environment: {str(e)}")
            return False

async def main():
    """Test the image generator."""
    try:
        # Initialize generator
        generator = KidFriendlyImageGenerator()
        
        # Test content
        test_content = {
            "topic": "Why is the sky blue?",
            "text": "The sky appears blue because of how sunlight interacts with our atmosphere. When sunlight travels through the air, it gets scattered in all directions. Blue light is scattered more than other colors because it travels in shorter, smaller waves.",
            "timestamp": "2024-03-20_123456"
        }
        
        # Generate image
        result = generator.generate_image(test_content, age_group=8)
        print(f"Generated image: {result['image_path']}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 