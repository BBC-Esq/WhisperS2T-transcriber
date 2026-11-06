from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QSettings

from core.logging_config import get_logger
from utils import get_settings_dir

logger = get_logger(__name__)

LEGACY_ORGANIZATION = "WhisperS2TTranscriber"
LEGACY_APPLICATION = "Transcriber"
LEGACY_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def migrate_legacy_settings() -> None:
    settings_dir = get_settings_dir()
    _migrate_registry(settings_dir / "Transcriber.ini")
    _migrate_config(settings_dir / "config.yaml")


def _migrate_registry(ini_path: Path) -> None:
    if ini_path.exists():
        return
    legacy = QSettings(LEGACY_ORGANIZATION, LEGACY_APPLICATION)
    keys = legacy.allKeys()
    if not keys:
        return
    ini_path.parent.mkdir(parents=True, exist_ok=True)
    settings = QSettings(str(ini_path), QSettings.IniFormat)
    for key in keys:
        settings.setValue(key, legacy.value(key))
    settings.sync()
    logger.info(f"Migrated {len(keys)} settings from the registry to {ini_path}")


def _migrate_config(config_path: Path) -> None:
    if config_path.exists() or not LEGACY_CONFIG_PATH.is_file():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_CONFIG_PATH, config_path)
    logger.info(f"Copied {LEGACY_CONFIG_PATH} to {config_path}")
