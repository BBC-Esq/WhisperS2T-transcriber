"""System utility functions."""
import torch
import ctranslate2
import psutil

def get_compute_and_platform_info() -> list[str]:
    """Get available compute devices."""
    devices = ["cpu"]
    
    if ctranslate2.get_cuda_device_count() > 0:
        devices.append('cuda')
        
    return devices

def get_logical_core_count() -> int:
    """Get number of logical CPU cores."""
    return psutil.cpu_count(logical=True) or 1

def has_bfloat16_support() -> bool:
    """Check if current GPU supports bfloat16."""
    if not torch.cuda.is_available():
        return False
        
    capability = torch.cuda.get_device_capability()
    # bfloat16 requires compute capability 8.6 or higher
    return capability >= (8, 6)

def get_optimal_cpu_threads() -> int:
    """Get optimal number of CPU threads for processing."""
    logical_cores = get_logical_core_count()
    # Reserve some cores for system and UI
    return max(4, logical_cores - 8)