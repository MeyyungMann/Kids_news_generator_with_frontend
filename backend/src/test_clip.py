from clip_handler import CLIPHandler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_clip():
    try:
        # Initialize CLIP
        logger.info("Initializing CLIP...")
        clip_handler = CLIPHandler()
        
        # Test text encoding
        logger.info("Testing text encoding...")
        text = "A happy child reading a book"
        text_features = clip_handler.encode_text(text)
        logger.info(f"Text features shape: {text_features.shape}")
        
        # Test image encoding (if you have a test image)
        # logger.info("Testing image encoding...")
        # image_path = "path/to/your/test/image.jpg"
        # image_features = clip_handler.encode_image(image_path)
        # logger.info(f"Image features shape: {image_features.shape}")
        
        # Test similarity computation (if you have a test image)
        # logger.info("Testing similarity computation...")
        # similarity = clip_handler.compute_similarity(image_path, text)
        # logger.info(f"Similarity score: {similarity:.2f}")
        
        logger.info("CLIP test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during CLIP test: {str(e)}")
        raise

if __name__ == "__main__":
    test_clip() 