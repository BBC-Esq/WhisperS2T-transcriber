import torch
import ctranslate2

def get_compute_and_platform_info():
    available_devices = ["cpu"]
    
    if ctranslate2.get_cuda_device_count() > 0:
        available_devices.append('cuda')
    
    return available_devices

def get_supported_quantizations(device_type):
    types = ctranslate2.get_supported_compute_types(device_type)
    filtered_types = [q for q in types if q != 'int16']
    desired_order = ['float32', 'float16', 'bfloat16', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int8']
    sorted_types = [q for q in desired_order if q in filtered_types]
    return sorted_types

