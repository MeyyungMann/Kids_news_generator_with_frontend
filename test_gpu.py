import torch
import sys
from loguru import logger

def test_gpu():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU memory allocation
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            logger.info("GPU memory allocation and computation test passed!")
            logger.info(f"Result shape: {z.shape}")
        except Exception as e:
            logger.error(f"GPU test failed: {str(e)}")
    else:
        logger.warning("CUDA is not available. Please check your GPU installation.")

if __name__ == "__main__":
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    test_gpu() 