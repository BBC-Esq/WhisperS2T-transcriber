"""Main application window."""
from pathlib import Path
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton,
    QHBoxLayout, QCheckBox, QFileDialog, QMessageBox
)

from config.constants import SUPPORTED_AUDIO_EXTENSIONS
from config.settings import TranscriptionSettings
from core.controller import BatchTranscriberController
from core.transcription.file_scanner import FileScanner
from gui.settings_widget import SettingsWidget
from gui.widgets.metrics_bar import MetricsBar

class MainWindow(QWidget):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.controller = BatchTranscriberController()
        self.file_scanner = FileScanner()
        self.selected_directory = None
        
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("Batch Whisper Transcriber")
        self.setGeometry(100, 100, 680, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout(self)
        
        # Transcriber group
        transcriber_group = QGroupBox("Batch Transcriber (ctranslate2 edition)")
        transcriber_layout = QVBoxLayout()
        
        self.dir_label = QLabel("No directory selected")
        transcriber_layout.addWidget(self.dir_label)
        
        self.progress_label = QLabel("Status: Idle")
        transcriber_layout.addWidget(self.progress_label)
        
        transcriber_group.setLayout(transcriber_layout)
        layout.addWidget(transcriber_group)
        
        # File extensions group
        extensions_group = QGroupBox("File Extensions")
        extensions_layout = QHBoxLayout()
        
        self.extension_checkboxes = {}
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            checkbox = QCheckBox(ext)
            checkbox.setChecked(True)
            self.extension_checkboxes[ext] = checkbox
            extensions_layout.addWidget(checkbox)
            
        extensions_group.setLayout(extensions_layout)
        layout.addWidget(extensions_group)
        
        # Settings widget
        self.settings_widget = SettingsWidget()
        layout.addWidget(self.settings_widget)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.clicked.connect(self._select_directory)
        controls_layout.addWidget(self.select_dir_button)
        
        self.recursive_checkbox = QCheckBox("Process Sub-Folders?")
        controls_layout.addWidget(self.recursive_checkbox)
        
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self._process_files)
        self.process_button.setEnabled(False)
        controls_layout.addWidget(self.process_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_processing)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        
        layout.addLayout(controls_layout)
        
        # Metrics bar
        self.metrics_bar = MetricsBar()
        layout.addWidget(self.metrics_bar)
        
    def _connect_signals(self):
        """Connect controller signals."""
        self.controller.status_updated.connect(self._update_status)
        self.controller.progress_updated.connect(self._update_progress)
        self.controller.processing_started.connect(self._on_processing_started)
        self.controller.processing_stopped.connect(self._on_processing_stopped)
        self.controller.processing_completed.connect(self._on_processing_completed)
        
    @Slot()
    def _select_directory(self):
        """Select directory for processing."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.selected_directory = dir_path
            self.dir_label.setText(f"Directory: {dir_path}")
            self.process_button.setEnabled(True)
            
    @Slot()
    def _process_files(self):
        """Start processing files."""
        if not self.selected_directory:
            return
            
        # Get settings
        settings = self._build_settings()
        
        # Validate settings
        warnings = settings.validate()
        if warnings:
            reply = QMessageBox.warning(
                self, "Warning",
                "\n".join(warnings) + "\n\nContinue anyway?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
                
        # Count files
        file_count = self.file_scanner.count_files(
            Path(self.selected_directory),
            settings.selected_extensions,
            settings.recursive
        )
        
        # Confirm processing
        reply = QMessageBox.question(
            self, "Confirm Process",
            f"{file_count} files found. Proceed?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok
        )
        
        if reply == QMessageBox.Ok:
            self.controller.start_processing(self.selected_directory, settings)
            
    @Slot()
    def _stop_processing(self):
        """Stop ongoing processing."""
        self.controller.stop_processing()
        
    def _build_settings(self) -> TranscriptionSettings:
        """Build settings from UI state."""
        return TranscriptionSettings(
            model_key=self.settings_widget.get_model(),
            device=self.settings_widget.get_device(),
            beam_size=self.settings_widget.get_beam_size(),
            batch_size=self.settings_widget.get_batch_size(),
            output_format=self.settings_widget.get_output_format(),
            task_mode=self.settings_widget.get_task_mode(),
            recursive=self.recursive_checkbox.isChecked(),
            selected_extensions=[
                ext for ext, cb in self.extension_checkboxes.items() 
                if cb.isChecked()
            ]
        )
        
    @Slot(str)
    def _update_status(self, message: str):
        """Update status label."""
        self.progress_label.setText(f"Status: {message}")
        
    @Slot(int, int, str)
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress display."""
        self.progress_label.setText(f"Status: {message} ({current}/{total})")
        
    @Slot()
    def _on_processing_started(self):
        """Handle processing started."""
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.settings_widget.setEnabled(False)
        
    @Slot()
    def _on_processing_stopped(self):
        """Handle processing stopped."""
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.settings_widget.setEnabled(True)
        
    @Slot(str)
    def _on_processing_completed(self, message: str):
        """Handle processing completed."""
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.settings_widget.setEnabled(True)
        self.progress_label.setText(f"Status: Completed | {message}")
        
    def closeEvent(self, event):
        """Handle window close."""
        self.controller.cleanup()
        self.metrics_bar.cleanup()
        super().closeEvent(event)