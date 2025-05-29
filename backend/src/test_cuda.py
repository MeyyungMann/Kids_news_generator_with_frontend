import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cuda():
    """Test CUDA availability and configuration."""
    logger.info("Testing CUDA configuration...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        # Test CUDA tensor operations
        x = torch.rand(5, 3).cuda()
        print(f"\nTest tensor on CUDA: {x.device}")
        print(f"Tensor dtype: {x.dtype}")
    else:
        logger.warning("CUDA is not available. Please check your PyTorch installation.")
        logger.info("PyTorch version:", torch.__version__)
        logger.info("CUDA version:", torch.version.cuda)

if __name__ == "__main__":
    test_cuda() 