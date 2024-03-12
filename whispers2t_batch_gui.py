import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox, QLabel, QGroupBox
from PySide6.QtCore import Qt
from utilities import get_compute_and_platform_info
from whispers2t_batch_transcriber import Worker
from metrics_bar import MetricsBar
from settings import SettingsGroupBox

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("chintellalaw.com - For Non-Commercial Use")
        self.setGeometry(100, 100, 680, 400)
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

        self.recursiveCheckbox = QCheckBox("Process All Sub-Folders?")
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

    def processFiles(self):
        if hasattr(self, 'directory'):
            device = self.settingsGroupBox.computeDeviceComboBox.currentText()
            size = self.settingsGroupBox.sizeComboBox.currentText()
            quantization = self.settingsGroupBox.quantizationComboBox.currentText()
            beam_size = self.settingsGroupBox.beamSizeSlider.value()
            batch_size = self.settingsGroupBox.batchSizeSlider.value()
            output_format = self.settingsGroupBox.formatComboBox.currentText()
            task = self.settingsGroupBox.transcribeTranslateComboBox.currentText()

            selected_extensions = [checkbox.text() for checkbox in self.file_extension_checkboxes if checkbox.isChecked()]

            self.worker = Worker(directory=self.directory, 
                                 recursive=self.recursiveCheckbox.isChecked(), 
                                 output_format=output_format, 
                                 device=device, 
                                 size=size, 
                                 quantization=quantization, 
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
