"""Main application controller."""
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional

from config.settings import TranscriptionSettings
from core.models.manager import ModelManager
from core.transcription.service import TranscriptionService
from core.monitoring.metrics_store import MetricsStore

class BatchTranscriberController(QObject):
    """Coordinates between UI, models, and transcription service."""
    
    # Signals
    status_updated = Signal(str)
    processing_started = Signal()
    processing_stopped = Signal()
    processing_completed = Signal(str)
    progress_updated = Signal(int, int, str)  # current, total, message
    
    def __init__(self):
        super().__init__()
        
        self.model_manager = ModelManager()
        self.transcription_service = TranscriptionService()
        self.metrics_store = MetricsStore()
        
        self._connect_signals()
        
    def _connect_signals(self):
        """Connect internal component signals."""
        self.transcription_service.progress_updated.connect(self.progress_updated)
        self.transcription_service.completed.connect(self.processing_completed)
        self.transcription_service.error_occurred.connect(self.status_updated)
        
    def start_processing(self, directory: str, settings: TranscriptionSettings) -> None:
        """Start batch transcription processing."""
        self.processing_started.emit()
        self.transcription_service.process_directory(
            directory=directory,
            settings=settings,
            model_manager=self.model_manager
        )
        
    def stop_processing(self) -> None:
        """Stop ongoing processing."""
        self.transcription_service.stop()
        self.processing_stopped.emit()
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.transcription_service.cleanup()
        self.model_manager.cleanup()