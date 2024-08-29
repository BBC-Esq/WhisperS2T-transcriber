import torch
import ctranslate2
import psutil

def get_compute_and_platform_info():
    available_devices = ["cpu"]
    
    if ctranslate2.get_cuda_device_count() > 0:
        available_devices.append('cuda')
    
    return available_devices


# def get_supported_quantizations(device_type):
    # types = ctranslate2.get_supported_compute_types(device_type)
    # filtered_types = [q for q in types if q != 'int16']
    # desired_order = ['float32', 'float16', 'bfloat16', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int8']
    # sorted_types = [q for q in desired_order if q in filtered_types]
    # return sorted_types

def get_logical_core_count():
    return psutil.cpu_count(logical=True)

def has_bfloat16_support():
    if not torch.cuda.is_available():
        return False
    
    capability = torch.cuda.get_device_capability()
    return capability >= (8, 6)