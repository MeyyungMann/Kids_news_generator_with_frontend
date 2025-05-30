import torch
import sys

def check_pytorch_cuda():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("Number of CUDA Devices:", torch.cuda.device_count())
        print("Current CUDA Device:", torch.cuda.current_device())
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        
        # Check CUDA device properties
        device = torch.cuda.current_device()
        print("\nCUDA Device Properties:")
        print("Total Memory:", torch.cuda.get_device_properties(device).total_memory / 1024**3, "GB")
        print("Multi Processor Count:", torch.cuda.get_device_properties(device).multi_processor_count)
        print("Compute Capability:", torch.cuda.get_device_properties(device).major, ".", torch.cuda.get_device_properties(device).minor)
    else:
        print("\nCUDA is not available. Please check your PyTorch installation and GPU drivers.")
        print("You might need to:")
        print("1. Install CUDA toolkit")
        print("2. Install the correct version of PyTorch with CUDA support")
        print("3. Update your GPU drivers")

if __name__ == "__main__":
    check_pytorch_cuda() 