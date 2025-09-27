"""System metrics data structures and monitoring utilities."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import psutil
import torch

@dataclass
class SystemMetrics:
    """Container for system metrics data."""
    timestamp: datetime
    cpu_usage: float
    ram_usage_percent: float
    gpu_utilization: Optional[float] = None
    vram_usage_percent: Optional[float] = None
    power_usage_percent: Optional[float] = None
    power_limit_percent: Optional[float] = None


class SystemMonitor:
    """System resource monitoring utilities."""
    
    def __init__(self):
        self.has_nvidia = self._init_nvidia()
        
    def _init_nvidia(self) -> bool:
        """Initialize NVIDIA monitoring if available."""
        try:
            if torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower():
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return True
        except Exception:
            pass
        
        self.gpu_handle = None
        return False
    
    def is_nvidia_gpu_available(self) -> bool:
        """Check if NVIDIA GPU is available."""
        return self.has_nvidia
    
    def collect_cpu_metrics(self) -> float:
        """Collect CPU usage metrics."""
        percentages = psutil.cpu_percent(interval=0, percpu=True)
        return sum(percentages) / len(percentages) if percentages else 0
    
    def collect_ram_metrics(self) -> tuple[float, int]:
        """Collect RAM usage metrics."""
        ram = psutil.virtual_memory()
        return ram.percent, ram.used
    
    def collect_gpu_metrics(self) -> tuple[float, float, float, float]:
        """Collect GPU metrics if available."""
        if not self.gpu_handle:
            return 0, 0, 0, 0
            
        try:
            import pynvml
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
            vram_usage_percent = (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0
            power_usage_percent, power_limit_percent = self._collect_power_metrics()
            return gpu_utilization, vram_usage_percent, power_usage_percent, power_limit_percent
        except Exception:
            return 0, 0, 0, 0
    
    def _collect_power_metrics(self) -> tuple[float, float]:
        """Collect GPU power metrics."""
        if not self.gpu_handle:
            return 0, 0
            
        try:
            import pynvml
            power_usage = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0
            power_percentage = (power_usage / power_limit) * 100 if power_limit > 0 else 0
            return power_percentage, power_limit
        except Exception:
            return 0, 0
    
    def collect_all_metrics(self) -> SystemMetrics:
        """Collect all system metrics."""
        cpu_usage = self.collect_cpu_metrics()
        ram_usage_percent, _ = self.collect_ram_metrics()
        
        if self.has_nvidia:
            gpu_util, vram_percent, power_percent, power_limit = self.collect_gpu_metrics()
        else:
            gpu_util = vram_percent = power_percent = power_limit = None
            
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            ram_usage_percent=ram_usage_percent,
            gpu_utilization=gpu_util,
            vram_usage_percent=vram_percent,
            power_usage_percent=power_percent,
            power_limit_percent=power_limit
        )