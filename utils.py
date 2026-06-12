from __future__ import annotations

import os
import sys

import psutil


def get_resource_path(relative_path: str) -> str:
    """Resolve a path relative to the application root, whether frozen or not."""
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def get_logical_core_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def get_optimal_cpu_threads() -> int:
    """Reserve a few cores for the UI and system."""
    logical_cores = get_logical_core_count()
    return max(4, logical_cores - 8)
