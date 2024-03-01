import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox, QLabel, QComboBox, QGroupBox, QSlider)
from PySide6.QtCore import Qt
from utilities import get_compute_and_platform_info, get_supported_quantizations
from whispers2t_batch_transcriber import Worker
from metrics_bar import MetricsBar

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("www.chintellalaw.com - Free Non-Commercial Use")
        self.setGeometry(100, 100, 520, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        main_layout = QVBoxLayout()

        transcriberGroupBox = QGroupBox("Batch Transcriber (ctranslate2)")
        transcriberLayout = QVBoxLayout()

        self.dirLabel = QLabel("No directory selected")
        transcriberLayout.addWidget(self.dirLabel)

        self.progressLabel = QLabel("Status: Idle")
        transcriberLayout.addWidget(self.progressLabel)

        transcriberGroupBox.setLayout(transcriberLayout)
        main_layout.addWidget(transcriberGroupBox)

        settingsGroupBox = QGroupBox("Settings")
        settingsLayout = QVBoxLayout()

        hbox1_layout = QHBoxLayout()
        computeDeviceLabel = QLabel("Device:")
        hbox1_layout.addWidget(computeDeviceLabel)

        self.computeDeviceComboBox = QComboBox()
        hbox1_layout.addWidget(self.computeDeviceComboBox)
        
        sizeLabel = QLabel("Size:")
        hbox1_layout.addWidget(sizeLabel)

        self.sizeComboBox = QComboBox()
        model_sizes = ["large-v2", "medium.en", "medium", "small.en", "small", "base.en", "base", "tiny.en", "tiny"]
        self.sizeComboBox.addItems(model_sizes)
        hbox1_layout.addWidget(self.sizeComboBox)
        
        quantizationLabel = QLabel("Quantization:")
        hbox1_layout.addWidget(quantizationLabel)

        self.quantizationComboBox = QComboBox()
        hbox1_layout.addWidget(self.quantizationComboBox)

        settingsLayout.addLayout(hbox1_layout)

        hbox2_layout = QHBoxLayout()
        
        self.recursiveCheckbox = QCheckBox("Process All Sub-Folders?")
        hbox2_layout.addWidget(self.recursiveCheckbox)

        formatLabel = QLabel("Output Format:")
        hbox2_layout.addWidget(formatLabel)

        self.formatComboBox = QComboBox()
        self.formatComboBox.addItems(["txt", "vtt", "srt", "tsv", "json"])
        hbox2_layout.addWidget(self.formatComboBox)

        settingsLayout.addLayout(hbox2_layout)

        beam_and_batch_size_layout = QHBoxLayout()

        beamSizeLabel = QLabel("Beam Size:")
        beam_and_batch_size_layout.addWidget(beamSizeLabel)

        self.beamSizeSlider = QSlider(Qt.Horizontal)
        self.beamSizeSlider.setMinimum(1)
        self.beamSizeSlider.setMaximum(10)
        self.beamSizeSlider.setValue(5)
        self.beamSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.beamSizeSlider.setTickInterval(1)
        beam_and_batch_size_layout.addWidget(self.beamSizeSlider)

        self.beamSizeValueLabel = QLabel("5")
        beam_and_batch_size_layout.addWidget(self.beamSizeValueLabel)

        self.beamSizeSlider.valueChanged.connect(lambda: self.update_slider_label(self.beamSizeSlider, self.beamSizeValueLabel))

        batchSizeLabel = QLabel("Batch Size:")
        beam_and_batch_size_layout.addWidget(batchSizeLabel)

        self.batchSizeSlider = QSlider(Qt.Horizontal)
        self.batchSizeSlider.setMinimum(1)
        self.batchSizeSlider.setMaximum(100)
        self.batchSizeSlider.setValue(16)
        self.batchSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.batchSizeSlider.setTickInterval(10)
        beam_and_batch_size_layout.addWidget(self.batchSizeSlider)

        self.batchSizeValueLabel = QLabel("16")
        beam_and_batch_size_layout.addWidget(self.batchSizeValueLabel)

        self.batchSizeSlider.valueChanged.connect(lambda: self.update_slider_label(self.batchSizeSlider, self.batchSizeValueLabel))

        settingsLayout.addLayout(beam_and_batch_size_layout)

        settingsGroupBox.setLayout(settingsLayout)

        main_layout.addWidget(settingsGroupBox)

        self.selectDirButton = QPushButton("Select Directory")
        self.selectDirButton.clicked.connect(self.selectDirectory)
        main_layout.addWidget(self.selectDirButton)

        self.processButton = QPushButton("Process")
        self.processButton.clicked.connect(self.processFiles)
        main_layout.addWidget(self.processButton)

        self.metricsBar = MetricsBar()
        main_layout.addWidget(self.metricsBar)

        self.setLayout(main_layout)

        self.populateComputeDeviceComboBox()

        self.computeDeviceComboBox.currentIndexChanged.connect(self.updateQuantizationComboBox)

    def closeEvent(self, event):
        self.metricsBar.stop_metrics_collector()
        super().closeEvent(event)

    def update_slider_label(self, slider, label):
        label.setText(str(slider.value()))
    
    def populateComputeDeviceComboBox(self):
        available_devices, _ = get_compute_and_platform_info()
        self.computeDeviceComboBox.addItems(available_devices)
        self.computeDeviceComboBox.setCurrentIndex(self.computeDeviceComboBox.findText("cpu"))
        self.updateQuantizationComboBox()

    def updateQuantizationComboBox(self):
        device_type = self.computeDeviceComboBox.currentText()
        quantizations = get_supported_quantizations(device_type)
        self.quantizationComboBox.clear()
        self.quantizationComboBox.addItems(quantizations)

    def selectDirectory(self):
        dirPath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirPath:
            self.directory = dirPath
            self.dirLabel.setText(f"Directory: {dirPath}")

    def processFiles(self):
        if hasattr(self, 'directory'):
            # Retrieve additional parameters
            device = self.computeDeviceComboBox.currentText()
            size = self.sizeComboBox.currentText()
            quantization = self.quantizationComboBox.currentText()
            beam_size = self.beamSizeSlider.value()
            batch_size = self.batchSizeSlider.value()
            output_format = self.formatComboBox.currentText()

            # Pass all parameters to the Worker
            self.worker = Worker(directory=self.directory, 
                                 recursive=self.recursiveCheckbox.isChecked(), 
                                 output_format=output_format, 
                                 device=device, 
                                 size=size, 
                                 quantization=quantization, 
                                 beam_size=beam_size, 
                                 batch_size=batch_size)
            
            self.worker.progress.connect(self.updateProgress)
            self.worker.finished.connect(self.workerFinished)
            self.worker.start()
            self.processButton.setEnabled(False)
        else:
            self.updateProgress("Please select a directory first.")

    def updateProgress(self, message):
        self.progressLabel.setText(f"Status: {message}")

    def workerFinished(self):
        self.processButton.setEnabled(True)
        self.progressLabel.setText("Status: Completed")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
