import torch
from diffusers import StableDiffusionPipeline
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from PIL import Image
import os
import gc
from clip_handler import CLIPHandler

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
        
        # Initialize CLIP handler
        self.clip_handler = CLIPHandler()
    
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
    
    def _create_clip_prompt(self, content: Dict[str, Any], age_group: int) -> str:
        """Create a concise prompt for CLIP validation."""
        topic = content.get("topic", "")
        category = content.get("category", "").lower()
        
        # Create a simplified prompt for CLIP
        prompt = f"kid-friendly illustration about {topic}"
        
        # Add category-specific elements
        category_elements = {
            "science": "scientific illustration",
            "technology": "technology illustration",
            "health": "health education illustration",
            "environment": "nature illustration",
            "economy": "economic education illustration"
        }
        
        category_element = category_elements.get(category, "")
        if category_element:
            prompt += f", {category_element}"
        
        return prompt
    
    def _validate_image(self, image: Image.Image, content: Dict[str, Any], age_group: int, min_similarity: float = 0.5) -> bool:
        """Validate generated image using CLIP."""
        try:
            # Save image temporarily
            temp_path = self.base_dir / "temp_validation.png"
            image.save(temp_path)
            
            # Create a concise prompt for CLIP
            clip_prompt = self._create_clip_prompt(content, age_group)
            
            # Compute similarity
            similarity = self.clip_handler.compute_similarity(str(temp_path), clip_prompt)
            
            # Clean up
            temp_path.unlink()
            
            return similarity >= min_similarity
        except Exception as e:
            logger.error(f"Error validating image with CLIP: {str(e)}")
            return True  # Return True on error to not block generation
    
    def _create_style_guidance(self, age_group: int) -> str:
        """Create age-appropriate style guidance for image generation."""
        styles = {
            # Ages 3-6 (Preschool)
            "3-6": {
                "style": "very simple cartoon style",
                "colors": "bright, primary colors",
                "shapes": "simple, basic shapes",
                "characters": "friendly, cute characters",
                "details": "minimal details, clear outlines",
                "mood": "happy, cheerful, playful"
            },
            
            # Ages 7-9 (Early Elementary)
            "7-9": {
                "style": "colorful cartoon style",
                "colors": "vibrant, engaging colors",
                "shapes": "clear, recognizable shapes",
                "characters": "expressive, relatable characters",
                "details": "moderate details, educational elements",
                "mood": "engaging, educational, fun"
            },
            
            # Ages 10-12 (Upper Elementary)
            "10-12": {
                "style": "detailed illustration style",
                "colors": "rich, varied colors",
                "shapes": "complex, realistic shapes",
                "characters": "realistic, detailed characters",
                "details": "more detailed, informative elements",
                "mood": "educational, informative, engaging"
            }
        }
        
        # Get style for age group or default to 7-9
        age_key = str(age_group)
        style = styles.get(age_key, styles["7-9"])
        
        # Combine style elements into a prompt
        style_prompt = f"{style['style']}, {style['colors']}, {style['shapes']}, {style['characters']}, {style['details']}, {style['mood']}"
        
        return style_prompt
    
    def _create_prompt(self, content: Dict[str, Any], age_group: int) -> str:
        """
        Create a kid-friendly prompt for image generation.
        
        Args:
            content: Generated news content
            age_group: Target age group
            
        Returns:
            Formatted prompt for image generation
        """
        # Get style guidance
        style_guidance = self._create_style_guidance(age_group)
        
        # Create the prompt with style guidance
        prompt = f"Create a kid-friendly illustration about {content.get('topic', '')}. {content.get('text', '')[:100]}... {style_guidance}"
        
        # Add negative prompt
        negative_prompt = "text, letters, words, writing, numbers, symbols, signs, labels, captions, scary, violent, inappropriate"
        
        return prompt, negative_prompt
    
    def generate_image(
        self,
        content: Dict[str, Any],
        age_group: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a kid-friendly image based on the content.
        
        Args:
            content: Generated news content
            age_group: Target age group
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            max_attempts: Number of attempts to generate a valid image
            
        Returns:
            Dictionary containing image path and metadata
        """
        try:
            # Create prompt
            prompt, negative_prompt = self._create_prompt(content, age_group)
            
            # Generate image with validation
            for attempt in range(max_attempts):
                logger.info(f"Generating image (attempt {attempt + 1}/{max_attempts})...")
                
                # Generate image
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
                
                # Validate with CLIP
                if self._validate_image(image, content, age_group):
                    logger.info("Image validated successfully with CLIP")
                    break
                else:
                    logger.warning(f"Image validation failed (attempt {attempt + 1})")
                    if attempt == max_attempts - 1:
                        logger.warning("Using last generated image despite validation failure")
            
            # Save image
            timestamp = content.get("timestamp", "")
            safe_title = "".join(c if c.isalnum() else "_" for c in content.get("topic", "image"))
            
            # Get category from content
            category = content.get("category", "general").capitalize()
            
            # Save image in category-specific directory
            image_path = self.images_dir / category / f"{safe_title}_{timestamp}.png"
            image.save(image_path)
            
            # Compute CLIP similarity score
            similarity_score = self.clip_handler.compute_similarity(str(image_path), self._create_clip_prompt(content, age_group))
            
            # Save metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "age_group": age_group,
                "topic": content.get("topic", ""),
                "timestamp": timestamp,
                "image_path": str(image_path),
                "clip_similarity_score": similarity_score
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