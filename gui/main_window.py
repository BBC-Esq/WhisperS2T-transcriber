from pathlib import Path
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton,
    QHBoxLayout, QCheckBox, QFileDialog, QMessageBox
)

from config.constants import SUPPORTED_AUDIO_EXTENSIONS
from config.settings import TranscriptionSettings
from core.models.manager import ModelManager
from core.transcription.file_scanner import FileScanner
from core.transcription.service import TranscriptionService
from gui.settings_widget import SettingsWidget
from gui.widgets.metrics_bar import MetricsBar

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.transcription_service = TranscriptionService()
        self.file_scanner = FileScanner()
        self.selected_directory = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.setWindowTitle("Batch Whisper Transcriber")
        self.setGeometry(100, 100, 680, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        transcriber_group = QGroupBox("Batch Transcriber (ctranslate2 edition)")
        transcriber_layout = QVBoxLayout()

        self.dir_label = QLabel("No directory selected")
        transcriber_layout.addWidget(self.dir_label)

        self.progress_label = QLabel("Status: Idle")
        transcriber_layout.addWidget(self.progress_label)

        transcriber_group.setLayout(transcriber_layout)
        layout.addWidget(transcriber_group)

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

        self.settings_widget = SettingsWidget()
        layout.addWidget(self.settings_widget)

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

        self.metrics_bar = MetricsBar()
        layout.addWidget(self.metrics_bar)

    def _connect_signals(self):
        self.transcription_service.error_occurred.connect(self._update_status)
        self.transcription_service.progress_updated.connect(self._update_progress)
        self.transcription_service.completed.connect(self._on_processing_completed)

    @Slot()
    def _select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.selected_directory = dir_path
            self.dir_label.setText(f"Directory: {dir_path}")
            self.process_button.setEnabled(True)

    @Slot()
    def _process_files(self):
        if not self.selected_directory:
            return

        settings = self._build_settings()

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

        files = self.file_scanner.scan_directory(
            Path(self.selected_directory),
            settings.selected_extensions,
            settings.recursive
        )

        reply = QMessageBox.question(
            self, "Confirm Process",
            f"{len(files)} files found. Proceed?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok
        )

        if reply == QMessageBox.Ok:
            self._on_processing_started()
            self.transcription_service.process_files(
                files=files,
                settings=settings,
                model_manager=self.model_manager
            )

    @Slot()
    def _stop_processing(self):
        self.transcription_service.stop()
        self._on_processing_stopped()

    def _build_settings(self) -> TranscriptionSettings:
        return TranscriptionSettings(
            model_key=self.settings_widget.get_model(),
            device=self.settings_widget.get_device(),
            beam_size=self.settings_widget.get_beam_size(),
            batch_size=self.settings_widget.get_batch_size(),
            output_format=self.settings_widget.get_output_format(),
            task_mode=self.settings_widget.get_task_mode(),
            language=self.settings_widget.get_language(),
            recursive=self.recursive_checkbox.isChecked(),
            selected_extensions=[
                ext for ext, cb in self.extension_checkboxes.items()
                if cb.isChecked()
            ]
        )

    @Slot(str)
    def _update_status(self, message: str):
        self.progress_label.setText(f"Status: {message}")

    @Slot(int, int, str)
    def _update_progress(self, current: int, total: int, message: str):
        self.progress_label.setText(f"Status: {message} ({current}/{total})")

    def _on_processing_started(self):
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.settings_widget.setEnabled(False)

    def _on_processing_stopped(self):
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.settings_widget.setEnabled(True)

    @Slot(str)
    def _on_processing_completed(self, message: str):
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.settings_widget.setEnabled(True)
        self.progress_label.setText(f"Status: Completed | {message}")

    def closeEvent(self, event):
        self.transcription_service.cleanup()
        self.model_manager.cleanup()
        self.metrics_bar.cleanup()
        super().closeEvent(event)
