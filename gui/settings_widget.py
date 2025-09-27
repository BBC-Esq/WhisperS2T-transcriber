"""Settings widget for transcription configuration."""
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSlider
)
from PySide6.QtCore import Qt, Signal

from config.constants import (
    WHISPER_MODELS, OUTPUT_FORMATS, TASK_MODES,
    DEFAULT_BEAM_SIZE, DEFAULT_BATCH_SIZE, 
    DEFAULT_OUTPUT_FORMAT, DEFAULT_TASK_MODE
)
from utils.system_utils import get_compute_and_platform_info, has_bfloat16_support

class SettingsWidget(QGroupBox):
    """Widget for configuring transcription settings."""
    
    device_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self._init_ui()
        self._populate_devices()
        
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # First row - Model, Device, Output, Task
        row1 = QHBoxLayout()
        
        row1.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        row1.addWidget(self.model_combo)
        
        row1.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        row1.addWidget(self.device_combo)
        
        row1.addWidget(QLabel("Output:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(OUTPUT_FORMATS)
        self.format_combo.setCurrentText(DEFAULT_OUTPUT_FORMAT)
        row1.addWidget(self.format_combo)
        
        row1.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(TASK_MODES)
        self.task_combo.setCurrentText(DEFAULT_TASK_MODE)
        row1.addWidget(self.task_combo)
        
        layout.addLayout(row1)
        
        # Beam size slider
        beam_layout = QHBoxLayout()
        beam_layout.addWidget(QLabel("Beam Size:"))
        
        self.beam_slider = QSlider(Qt.Horizontal)
        self.beam_slider.setMinimum(1)
        self.beam_slider.setMaximum(5)
        self.beam_slider.setValue(DEFAULT_BEAM_SIZE)
        self.beam_slider.setTickPosition(QSlider.TicksBelow)
        self.beam_slider.setTickInterval(1)
        beam_layout.addWidget(self.beam_slider)
        
        self.beam_label = QLabel(str(DEFAULT_BEAM_SIZE))
        beam_layout.addWidget(self.beam_label)
        self.beam_slider.valueChanged.connect(
            lambda v: self.beam_label.setText(str(v))
        )
        
        layout.addLayout(beam_layout)
        
        # Batch size slider
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setMinimum(1)
        self.batch_slider.setMaximum(200)
        self.batch_slider.setValue(DEFAULT_BATCH_SIZE)
        self.batch_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_slider.setTickInterval(10)
        batch_layout.addWidget(self.batch_slider)
        
        self.batch_label = QLabel(str(DEFAULT_BATCH_SIZE))
        batch_layout.addWidget(self.batch_label)
        self.batch_slider.valueChanged.connect(
            lambda v: self.batch_label.setText(str(v))
        )
        
        layout.addLayout(batch_layout)
        
    def _populate_devices(self):
        """Populate compute device options."""
        devices = get_compute_and_platform_info()
        self.device_combo.addItems(devices)
        
        # Default to CUDA if available
        if "cuda" in devices:
            self.device_combo.setCurrentText("cuda")
        else:
            self.device_combo.setCurrentText("cpu")
            
        self._update_model_options()
        
    def _on_device_changed(self, device: str):
        """Handle device change."""
        self.device_changed.emit(device)
        self._update_model_options()
        
    def _update_model_options(self):
        """Update available models based on device."""
        current_device = self.device_combo.currentText()
        self.model_combo.clear()
        
        for model_name, model_info in WHISPER_MODELS.items():
            # CPU only supports float32
            if current_device == "cpu" and model_info['precision'] == 'float32':
                self.model_combo.addItem(model_name)
            # CUDA supports multiple precisions
            elif current_device == "cuda":
                if model_info['precision'] in ['float32', 'float16']:
                    self.model_combo.addItem(model_name)
                elif model_info['precision'] == 'bfloat16' and has_bfloat16_support():
                    self.model_combo.addItem(model_name)
    
    # Getter methods for settings
    def get_model(self) -> str:
        """Get selected model."""
        return self.model_combo.currentText()
    
    def get_device(self) -> str:
        """Get selected device."""
        return self.device_combo.currentText()
    
    def get_output_format(self) -> str:
        """Get selected output format."""
        return self.format_combo.currentText()
    
    def get_task_mode(self) -> str:
        """Get selected task mode."""
        return self.task_combo.currentText()
    
    def get_beam_size(self) -> int:
        """Get beam size value."""
        return self.beam_slider.value()
    
    def get_batch_size(self) -> int:
        """Get batch size value."""
        return self.batch_slider.value()