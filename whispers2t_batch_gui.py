import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox, QLabel, QComboBox, QGroupBox, QSlider
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
import whisper_s2t
import torch
import platform
import ctranslate2

class Worker(QThread):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, directory, recursive, output_format):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.output_format = output_format

    def run(self):
        directory_path = Path(self.directory)
        patterns = ['*.mp3', '*.wav', '*.flac', '*.wma']
        audio_files = []

        if self.recursive:
            for pattern in patterns:
                audio_files.extend(directory_path.rglob(pattern))
        else:
            for pattern in patterns:
                audio_files.extend(directory_path.glob(pattern))

        model = whisper_s2t.load_model(model_identifier="medium.en", backend='CTranslate2', device='cuda', asr_options={'beam_size': 5})

        for original_audio_file in audio_files:
            self.progress.emit(f"Processing {original_audio_file}...")
            lang_codes = ['en']
            tasks = ['transcribe']
            initial_prompts = [None]

            out = model.transcribe_with_vad([str(original_audio_file)], lang_codes=lang_codes, tasks=tasks, initial_prompts=initial_prompts, batch_size=70)

            output_file_path = original_audio_file.with_suffix(f'.{self.output_format}')
            whisper_s2t.write_outputs(out, format=self.output_format, op_files=[str(output_file_path)])
            self.progress.emit(f"Transcribed {original_audio_file} to {output_file_path}")

        self.finished.emit()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("www.chintellalaw.com - Free Non-Commercial Use")
        self.setGeometry(100, 100, 520, 250)
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

        self.setLayout(main_layout)

        self.populateComputeDeviceComboBox()

        self.computeDeviceComboBox.currentIndexChanged.connect(self.updateQuantizationComboBox)

    def update_slider_label(self, slider, label):
        label.setText(str(slider.value()))
    
    def populateComputeDeviceComboBox(self):
        available_devices, _ = self.get_compute_and_platform_info()
        self.computeDeviceComboBox.addItems(available_devices)
        self.computeDeviceComboBox.setCurrentIndex(self.computeDeviceComboBox.findText("cpu"))
        self.updateQuantizationComboBox()

    def get_compute_and_platform_info(self):
        available_devices = ["cpu"]
        os_name = platform.system().lower()
        if torch.cuda.is_available():
            available_devices.append('cuda')
        return available_devices, os_name

    def updateQuantizationComboBox(self):
        device_type = self.computeDeviceComboBox.currentText()
        quantizations = self.get_supported_quantizations(device_type)
        self.quantizationComboBox.clear()
        self.quantizationComboBox.addItems(quantizations)

    def get_supported_quantizations(self, device_type):
        types = ctranslate2.get_supported_compute_types(device_type)
        filtered_types = [q for q in types if q != 'int16']
        desired_order = ['float32', 'float16', 'bfloat16', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int8']
        sorted_types = [q for q in desired_order if q in filtered_types]
        return sorted_types

    def selectDirectory(self):
        dirPath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirPath:
            self.directory = dirPath
            self.dirLabel.setText(f"Directory: {dirPath}")

    def processFiles(self):
        if hasattr(self, 'directory'):
            output_format = self.formatComboBox.currentText()
            self.worker = Worker(self.directory, self.recursiveCheckbox.isChecked(), output_format)
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