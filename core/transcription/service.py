"""Transcription service for processing audio files."""
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from config.settings import TranscriptionSettings
from config.constants import WHISPER_MODELS
from .batch_processor import BatchProcessor

class TranscriptionService(QObject):

    progress_updated = Signal(int, int, str)
    completed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._processor: Optional[BatchProcessor] = None

    def process_files(self, files: List[Path], settings: TranscriptionSettings,
                      model_manager) -> None:
        if not files:
            self.error_occurred.emit("No matching files found")
            return

        model_info = WHISPER_MODELS[settings.model_key]

        self._processor = BatchProcessor(
            files=files,
            settings=settings,
            model_info=model_info,
            model_manager=model_manager
        )

        self._processor.progress.connect(self.progress_updated)
        self._processor.finished.connect(self.completed)
        self._processor.error.connect(self.error_occurred)

        self._processor.start()

    def stop(self) -> None:
        if self._processor and self._processor.isRunning():
            self._processor.request_stop()

    def cleanup(self) -> None:
        if self._processor:
            if self._processor.isRunning():
                self._processor.request_stop()
                self._processor.wait()
            self._processor = None
