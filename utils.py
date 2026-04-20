from __future__ import annotations

import os
import sys

import psutil


def get_resource_path(relative_path: str) -> str:
    """Resolve a path relative to the application root, whether frozen or not."""
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if os.path.basename(script_dir) == "app" and os.path.exists(os.path.join(parent_dir, "compute_type.txt")):
            base_path = script_dir
        else:
            base_path = script_dir

    return os.path.join(base_path, relative_path)


def get_install_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    if os.path.basename(script_dir) == "app" and os.path.exists(os.path.join(parent_dir, "compute_type.txt")):
        return parent_dir
    return script_dir


def is_gpu_install() -> bool:
    install_dir = get_install_dir()
    compute_type_file = os.path.join(install_dir, "compute_type.txt")

    if os.path.exists(compute_type_file):
        try:
            with open(compute_type_file, 'r') as f:
                return f.read().strip().lower() == "gpu"
        except Exception:
            pass

    return True


def get_logical_core_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def get_physical_core_count() -> int:
    return psutil.cpu_count(logical=False) or 1


def get_optimal_cpu_threads() -> int:
    """Reserve a few cores for the UI and system."""
    logical_cores = get_logical_core_count()
    return max(4, logical_cores - 8)


def has_bfloat16_support() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability()
        return capability >= (8, 6)
    except Exception:
        return False
