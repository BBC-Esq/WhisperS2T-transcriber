from __future__ import annotations

import os
from pathlib import Path

import psutil

APP_FOLDER_NAME = "WhisperS2TTranscriber"


def get_settings_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / APP_FOLDER_NAME


def get_local_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_FOLDER_NAME


def get_logical_core_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def get_optimal_cpu_threads() -> int:
    """Reserve a few cores for the UI and system."""
    logical_cores = get_logical_core_count()
    return max(4, logical_cores - 8)
