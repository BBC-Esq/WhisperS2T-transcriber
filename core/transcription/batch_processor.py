"""Batch processing worker thread."""
from pathlib import Path
from typing import List, Dict, Any
from queue import Queue
from threading import Event

from PySide6.QtCore import QThread, Signal, QElapsedTimer
import whisper_s2t

from config.settings import TranscriptionSettings

class BatchProcessor(QThread):
    """Worker thread for batch transcription processing."""
    
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, files: List[Path], settings: TranscriptionSettings,
                 model_info: Dict[str, Any], model_manager):
        super().__init__()
        self.files = files
        self.settings = settings
        self.model_info = model_info
        self.model_manager = model_manager
        self.stop_requested = Event()
        
    def request_stop(self):
        """Request processing to stop."""
        self.stop_requested.set()
        
    def run(self):
        """Process all files."""
        timer = QElapsedTimer()
        timer.start()
        
        try:
            # Load model
            model = self.model_manager.get_or_load_model(
                self.settings.model_key,
                self.settings.device,
                self.settings.beam_size,
                self.model_info['precision']
            )
            
            if not model:
                self.error.emit("Failed to load model")
                return
                
            # Process files
            total_files = len(self.files)
            
            for idx, audio_file in enumerate(self.files, 1):
                if self.stop_requested.is_set():
                    break
                    
                self.progress.emit(idx, total_files, f"Processing {audio_file.name}")
                
                try:
                    # Transcribe
                    out = model.transcribe_with_vad(
                        [str(audio_file)],
                        lang_codes=['en'],
                        tasks=[self.settings.task_mode],
                        initial_prompts=[None],
                        batch_size=self.settings.batch_size
                    )
                    
                    # Write output
                    output_file = audio_file.with_suffix(f'.{self.settings.output_format}')
                    whisper_s2t.write_outputs(
                        out,
                        format=self.settings.output_format,
                        op_files=[str(output_file)]
                    )
                    
                    self.progress.emit(idx, total_files, f"Completed {audio_file.name}")
                    
                except Exception as e:
                    self.error.emit(f"Error processing {audio_file.name}: {e}")
                    
        except Exception as e:
            self.error.emit(f"Processing failed: {e}")
            
        finally:
            elapsed = timer.elapsed() / 1000.0
            self.finished.emit(f"Processing time: {elapsed:.2f} seconds")