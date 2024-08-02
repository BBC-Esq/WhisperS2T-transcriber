import os
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox, QLabel, QGroupBox, QMessageBox
from PySide6.QtCore import Qt
import torch    
from utilities import get_compute_and_platform_info, get_supported_quantizations
from whispers2t_batch_transcriber import Worker
from metrics_bar import MetricsBar
from settings import SettingsGroupBox
import logging
import traceback
from constants import WHISPER_MODELS

def set_cuda_paths():
    try:
        venv_base = Path(sys.executable).parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
        for env_var in ['CUDA_PATH', 'CUDA_PATH_V12_1', 'PATH']:
            current_path = os.environ.get(env_var, '')
            os.environ[env_var] = os.pathsep.join(filter(None, [str(nvidia_base_path), current_path]))
        logging.info("CUDA paths set successfully")
    except Exception as e:
        logging.error(f"Error setting CUDA paths: {str(e)}")
        logging.debug(traceback.format_exc())

set_cuda_paths()

def is_nvidia_gpu_available():
    return torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("chintellalaw.com - for non-commercial use")
        initial_height = 400 if is_nvidia_gpu_available() else 370
        self.setGeometry(100, 100, 680, initial_height)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        main_layout = QVBoxLayout()

        transcriberGroupBox = QGroupBox("Batch Transcriber (ctranslate2 edition)")
        transcriberLayout = QVBoxLayout()

        self.dirLabel = QLabel("No directory selected")
        transcriberLayout.addWidget(self.dirLabel)

        self.progressLabel = QLabel("Status: Idle")
        transcriberLayout.addWidget(self.progressLabel)

        transcriberGroupBox.setLayout(transcriberLayout)
        main_layout.addWidget(transcriberGroupBox)

        fileExtensionsGroupBox = QGroupBox("File Extensions")
        fileExtensionsLayout = QHBoxLayout()

        self.file_extension_checkboxes = []

        file_extensions = [".aac", ".amr", ".asf", ".avi", ".flac", ".m4a", ".mkv", ".mp3", ".mp4", ".wav", ".wma"]
        for extension in file_extensions:
            checkbox = QCheckBox(extension)
            checkbox.setChecked(True)
            self.file_extension_checkboxes.append(checkbox)
            fileExtensionsLayout.addWidget(checkbox)

        fileExtensionsGroupBox.setLayout(fileExtensionsLayout)
        main_layout.addWidget(fileExtensionsGroupBox)

        self.settingsGroupBox = SettingsGroupBox(get_compute_and_platform_info, self)
        main_layout.addWidget(self.settingsGroupBox)

        selectDirLayout = QHBoxLayout()
        self.selectDirButton = QPushButton("Select Directory")
        self.selectDirButton.clicked.connect(self.selectDirectory)
        selectDirLayout.addWidget(self.selectDirButton)

        self.recursiveCheckbox = QCheckBox("Process Sub-Folders?")
        selectDirLayout.addWidget(self.recursiveCheckbox)

        self.processButton = QPushButton("Process")
        self.processButton.clicked.connect(self.processFiles)
        selectDirLayout.addWidget(self.processButton)

        self.stopButton = QPushButton("Stop")
        self.stopButton.setEnabled(False)
        self.stopButton.setVisible(True)
        self.stopButton.clicked.connect(self.stopProcessing)
        selectDirLayout.addWidget(self.stopButton)

        main_layout.addLayout(selectDirLayout)

        self.metricsBar = MetricsBar()
        main_layout.addWidget(self.metricsBar)

        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.metricsBar.stop_metrics_collector()
        super().closeEvent(event)

    def selectDirectory(self):
        dirPath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirPath:
            self.directory = dirPath
            self.dirLabel.setText(f"Directory: {dirPath}")
            self.processButton.setEnabled(True)

    def calculate_files_to_process(self):
        selected_extensions = [checkbox.text() for checkbox in self.file_extension_checkboxes if checkbox.isChecked()]
        patterns = [f'*{ext}' for ext in selected_extensions]

        directory_path = Path(self.directory)
        total_files = 0
        for pattern in patterns:
            if self.recursiveCheckbox.isChecked():
                total_files += len(list(directory_path.rglob(pattern)))
            else:
                total_files += len(list(directory_path.glob(pattern)))
        return total_files

    def perform_checks(self):
        model = self.settingsGroupBox.modelComboBox.currentText()
        device = self.settingsGroupBox.computeDeviceComboBox.currentText()
        batch_size = self.settingsGroupBox.batchSizeSlider.value()
        beam_size = self.settingsGroupBox.beamSizeSlider.value()

        # Check 1: CPU and non-float32 model
        if "float32" not in model.lower() and device.lower() == "cpu":
            QMessageBox.warning(self, "Invalid Configuration", 
                                "CPU only supports Float 32 computation. Please select a different Whisper model.")
            return False

        # Check 2: CPU with high batch size
        if device.lower() == "cpu" and batch_size > 16:
            reply = QMessageBox.warning(self, "Performance Warning", 
                "When using CPU it is generally recommended to use a batch size of no more than 16 "
                "otherwise compute time will actually be worse.\n\n"
                "Moreover, if you select a Beam Size greater than one, you should reduce the Batch Size accordingly.\n\n"
                "For example:\n"
                "- If you select a Beam Size of 2 (double the default value of 1) you would reduce the Batch Size (default value 16) by half.\n"
                "- If Beam Size is set to 3 you should reduce the Batch Size to 1/3 of the default level, and so on.\n\nClick OK to proceed.",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False

        # Check 3: GPU compatibility
        # Only perform this check if the device is not CPU
        if device.lower() != "cpu":
            supported_quantizations = get_supported_quantizations(device)
            if "float16" in model.lower() and "float16" not in supported_quantizations:
                QMessageBox.warning(self, "Incompatible Configuration", 
                                    "Your GPU does not support the selected floating point value (float16). "
                                    "Please make another selection.")
                return False
            if "bfloat16" in model.lower() and "bfloat16" not in supported_quantizations:
                QMessageBox.warning(self, "Incompatible Configuration", 
                                    "Your GPU does not support the selected floating point value (bfloat16). "
                                    "Please make another selection.")
                return False

        return True  # All checks passed

    def processFiles(self):
        if hasattr(self, 'directory'):
            total_files = self.calculate_files_to_process()

            # Perform checks
            if not self.perform_checks():
                return

            reply = QMessageBox.question(
                self, 
                'Confirm Process', 
                f"{total_files} files found that match the selected criteria. Click OK to proceed or Cancel to abort.",
                QMessageBox.Ok | QMessageBox.Cancel, 
                QMessageBox.Ok)

            if reply == QMessageBox.Ok:
                model_key = self.settingsGroupBox.modelComboBox.currentText()
                device = self.settingsGroupBox.computeDeviceComboBox.currentText()
                beam_size = self.settingsGroupBox.beamSizeSlider.value()
                batch_size = self.settingsGroupBox.batchSizeSlider.value()
                output_format = self.settingsGroupBox.formatComboBox.currentText()
                task = self.settingsGroupBox.transcribeTranslateComboBox.currentText()

                selected_extensions = [checkbox.text() for checkbox in self.file_extension_checkboxes if checkbox.isChecked()]

                self.worker = Worker(directory=self.directory, 
                                     recursive=self.recursiveCheckbox.isChecked(), 
                                     output_format=output_format, 
                                     device=device, 
                                     model_key=model_key,
                                     beam_size=beam_size, 
                                     batch_size=batch_size,
                                     task=task,
                                     selected_extensions=selected_extensions)
                
                self.worker.progress.connect(self.updateProgress)
                self.worker.finished.connect(self.workerFinished)
                self.worker.start()
                self.processButton.setEnabled(False)
                self.stopButton.setEnabled(True)
            else:
                self.updateProgress("Processing cancelled by user.")
        else:
            self.updateProgress("Please select a directory first.")

    def stopProcessing(self):
        if hasattr(self, 'worker'):
            self.worker.request_stop()
            self.stopButton.setEnabled(False)

    def updateProgress(self, message):
        self.progressLabel.setText(f"Status: {message}")

    def workerFinished(self, message):
        self.processButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.progressLabel.setText(f"Status: Completed | {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
