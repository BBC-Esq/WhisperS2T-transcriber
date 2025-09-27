"""Model management and caching."""
import gc
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal, QMutex

import torch
import whisper_s2t

from config.constants import WHISPER_MODELS
from utils.system_utils import get_logical_core_count

class ModelManager(QObject):
    """Manages model loading, caching, and resource cleanup."""
    
    model_loaded = Signal(str, str)  # model_name, device
    model_error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._current_model = None
        self._current_config = None
        self._model_mutex = QMutex()
        self._cpu_threads = max(4, get_logical_core_count() - 8)
        
    def get_or_load_model(self, model_key: str, device: str, 
                         beam_size: int, precision: str) -> Optional[Any]:
        """Get cached model or load new one if config changed."""
        config = {
            'model_key': model_key,
            'device': device,
            'beam_size': beam_size,
            'precision': precision
        }
        
        self._model_mutex.lock()
        try:
            if self._current_model is None or self._current_config != config:
                self._release_current_model()
                self._current_model = self._load_model(config)
                self._current_config = config
                self.model_loaded.emit(model_key, device)
            
            return self._current_model
        except Exception as e:
            self.model_error.emit(str(e))
            return None
        finally:
            self._model_mutex.unlock()
            
    def _load_model(self, config: Dict[str, Any]) -> Any:
        """Load a new model with given configuration."""
        model_info = WHISPER_MODELS[config['model_key']]
        
        model_kwargs = {}
        if 'large-v3' in model_info['repo_id']:
            model_kwargs['n_mels'] = 128

        asr_options = {
            'beam_size': config['beam_size'],
            'word_timestamps': True,  # NEW
        }

        return whisper_s2t.load_model(
            model_identifier=model_info['repo_id'],
            backend='CTranslate2',
            device=config['device'],
            compute_type=config['precision'],
            asr_options={'beam_size': config['beam_size']},
            cpu_threads=self._cpu_threads if config['device'] == "cpu" else 4,
            **model_kwargs
        )
        
    def _release_current_model(self) -> None:
        """Release current model and free resources."""
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    def cleanup(self) -> None:
        """Clean up all resources."""
        self._model_mutex.lock()
        self._release_current_model()
        self._model_mutex.unlock()