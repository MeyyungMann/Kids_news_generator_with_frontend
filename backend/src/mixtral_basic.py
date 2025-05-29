import torch
import logging
import platform
import psutil
from typing import Dict, Any, Tuple, List
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUCompatibilityChecker:
    """Class to check GPU compatibility and system resources."""
    
    @staticmethod
    def get_optimal_device() -> Tuple[str, Dict[str, Any]]:
        """
        Determine the optimal device (CPU/GPU) for model execution.
        
        Returns:
            Tuple containing:
            - str: Device to use ('cuda' or 'cpu')
            - Dict: System diagnostics information
        """
        try:
            # Get system information
            system_info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'available_memory': psutil.virtual_memory().available
            }
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            cuda_info = {
                'available': cuda_available,
                'version': torch.version.cuda if cuda_available else None
            }
            
            # Get GPU information if available
            gpu_info = {
                'available': cuda_available,
                'devices': []
            }
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    try:
                        # Get total memory
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        # Get allocated memory
                        allocated_memory = torch.cuda.memory_allocated(i)
                        # Calculate free memory
                        free_memory = total_memory - allocated_memory
                        
                        device = {
                            'id': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_total': total_memory,
                            'memory_free': free_memory,
                            'memory_allocated': allocated_memory,
                            'temperature': GPUCompatibilityChecker._get_gpu_temperature(i),
                            'load': GPUCompatibilityChecker._get_gpu_load(i)
                        }
                        gpu_info['devices'].append(device)
                        
                        # Log memory status
                        logger.info(f"GPU {i} Memory Status:")
                        logger.info(f"  Total: {total_memory / (1024**3):.1f} GB")
                        logger.info(f"  Allocated: {allocated_memory / (1024**3):.1f} GB")
                        logger.info(f"  Free: {free_memory / (1024**3):.1f} GB")
                        
                    except Exception as e:
                        logger.warning(f"Error getting information for GPU {i}: {str(e)}")
            
            # Determine optimal device
            if cuda_available and gpu_info['devices']:
                # Check if any GPU has enough free memory (at least 8GB)
                for device in gpu_info['devices']:
                    if device['memory_free'] >= 8 * 1024 * 1024 * 1024:  # 8GB in bytes
                        logger.info(f"Using GPU: {device['name']}")
                        return 'cuda', {
                            'system_info': system_info,
                            'gpu_info': gpu_info,
                            'cuda_info': cuda_info
                        }
            
            logger.info("Using CPU (no suitable GPU found)")
            return 'cpu', {
                'system_info': system_info,
                'gpu_info': gpu_info,
                'cuda_info': cuda_info
            }
            
        except Exception as e:
            logger.error(f"Error checking GPU compatibility: {str(e)}")
            return 'cpu', {
                'system_info': system_info,
                'gpu_info': {'available': False, 'devices': []},
                'cuda_info': {'available': False, 'version': None}
            }
    
    @staticmethod
    def _get_gpu_temperature(device_id: int) -> float:
        """Get GPU temperature in Celsius."""
        try:
            if platform.system() == 'Windows':
                # Windows-specific temperature check
                import wmi
                w = wmi.WMI(namespace="root\OpenHardwareMonitor")
                temperature_infos = w.Sensor()
                for sensor in temperature_infos:
                    if sensor.SensorType == 'Temperature' and 'GPU' in sensor.Name:
                        return float(sensor.Value)
            else:
                # Linux-specific temperature check
                temp_file = f"/sys/class/drm/card{device_id}/device/hwmon/hwmon*/temp1_input"
                import glob
                temp_files = glob.glob(temp_file)
                if temp_files:
                    with open(temp_files[0], 'r') as f:
                        return float(f.read().strip()) / 1000.0
        except Exception as e:
            logger.warning(f"Error getting GPU temperature: {str(e)}")
        return 0.0
    
    @staticmethod
    def _get_gpu_load(device_id: int) -> float:
        """Get GPU utilization percentage."""
        try:
            if platform.system() == 'Windows':
                # Windows-specific GPU load check
                import wmi
                w = wmi.WMI(namespace="root\OpenHardwareMonitor")
                load_infos = w.Sensor()
                for sensor in load_infos:
                    if sensor.SensorType == 'Load' and 'GPU' in sensor.Name:
                        return float(sensor.Value)
            else:
                # Linux-specific GPU load check
                import subprocess
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
                ).decode().strip()
                return float(result)
        except Exception as e:
            logger.warning(f"Error getting GPU load: {str(e)}")
        return 0.0

class MixtralHandler:
    """Handler class for Mixtral model operations."""
    
    def __init__(self, device: str = None):
        """
        Initialize Mixtral handler.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). If None, will be determined automatically.
        """
        self.device, self.diagnostics = (
            (device, None) if device else GPUCompatibilityChecker.get_optimal_device()
        )
        
        # Log system information
        if self.diagnostics:
            logger.info("System Diagnostics:")
            logger.info(f"OS: {self.diagnostics['system_info']['os']} {self.diagnostics['system_info']['os_version']}")
            logger.info(f"Python: {self.diagnostics['system_info']['python_version']}")
            logger.info(f"CPU Cores: {self.diagnostics['system_info']['cpu_count']}")
            logger.info(f"Available Memory: {self.diagnostics['system_info']['available_memory'] / (1024**3):.1f} GB")
            
            if self.diagnostics['gpu_info']['available']:
                for device in self.diagnostics['gpu_info']['devices']:
                    logger.info(f"GPU {device['id']}: {device['name']}")
                    logger.info(f"  Memory: {device['memory_free']/1024:.1f} GB free of {device['memory_total']/1024:.1f} GB")
                    logger.info(f"  Temperature: {device['temperature']}°C")
                    logger.info(f"  Load: {device['load']:.1f}%")
    
    def generate_text(self, prompt: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Generate text using the Mixtral model.
        
        Args:
            prompt: Input prompt for text generation
            max_length: Maximum length of generated text
            
        Returns:
            Dictionary containing generated text and metadata
        """
        try:
            # This is a placeholder for the actual model implementation
            # You would typically load and run the Mixtral model here
            return {
                'text': f"Generated text for prompt: {prompt}",
                'metadata': {
                    'device': self.device,
                    'max_length': max_length
                }
            }
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise 