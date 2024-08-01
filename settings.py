from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider)
from PySide6.QtCore import Qt
from constants import WHISPER_MODELS

class SettingsGroupBox(QGroupBox):
    def __init__(self, get_compute_and_platform_info_callback, parent=None):
        super().__init__("Settings", parent)
        self.get_compute_and_platform_info = get_compute_and_platform_info_callback
        self.initUI()

    def initUI(self):
        self.setLayout(QVBoxLayout())

        hbox1_layout = QHBoxLayout()
        
        # Replace size and quantization combo boxes with a single model combo box
        modelLabel = QLabel("Model:")
        hbox1_layout.addWidget(modelLabel)

        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(WHISPER_MODELS.keys())
        hbox1_layout.addWidget(self.modelComboBox)

        # Keep the rest of the widgets
        computeDeviceLabel = QLabel("Device:")
        hbox1_layout.addWidget(computeDeviceLabel)

        self.computeDeviceComboBox = QComboBox()
        hbox1_layout.addWidget(self.computeDeviceComboBox)

        formatLabel = QLabel("Output:")
        hbox1_layout.addWidget(formatLabel)

        self.formatComboBox = QComboBox()
        self.formatComboBox.addItems(["txt", "vtt", "srt", "tsv", "json"])
        hbox1_layout.addWidget(self.formatComboBox)

        taskLabel = QLabel("Task:")
        hbox1_layout.addWidget(taskLabel)

        self.transcribeTranslateComboBox = QComboBox()
        self.transcribeTranslateComboBox.addItems(["transcribe", "translate"])
        hbox1_layout.addWidget(self.transcribeTranslateComboBox)

        self.layout().addLayout(hbox1_layout)

        beam_size_layout = QHBoxLayout()
        beamSizeLabel = QLabel("Beam Size:")
        beam_size_layout.addWidget(beamSizeLabel)

        self.beamSizeSlider = QSlider(Qt.Horizontal)
        self.beamSizeSlider.setMinimum(1)
        self.beamSizeSlider.setMaximum(10)
        self.beamSizeSlider.setValue(5)
        self.beamSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.beamSizeSlider.setTickInterval(1)
        beam_size_layout.addWidget(self.beamSizeSlider)

        self.beamSizeValueLabel = QLabel("5")
        beam_size_layout.addWidget(self.beamSizeValueLabel)
        self.beamSizeSlider.valueChanged.connect(lambda: self.update_slider_label(self.beamSizeSlider, self.beamSizeValueLabel))

        self.layout().addLayout(beam_size_layout)

        batch_size_layout = QHBoxLayout()
        batchSizeLabel = QLabel("Batch Size:")
        batch_size_layout.addWidget(batchSizeLabel)

        self.batchSizeSlider = QSlider(Qt.Horizontal)
        self.batchSizeSlider.setMinimum(1)
        self.batchSizeSlider.setMaximum(200)
        self.batchSizeSlider.setValue(16)
        self.batchSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.batchSizeSlider.setTickInterval(10)
        batch_size_layout.addWidget(self.batchSizeSlider)

        self.batchSizeValueLabel = QLabel("16")
        batch_size_layout.addWidget(self.batchSizeValueLabel)
        self.batchSizeSlider.valueChanged.connect(lambda: self.update_slider_label(self.batchSizeSlider, self.batchSizeValueLabel))

        self.layout().addLayout(batch_size_layout)

        self.populateComputeDeviceComboBox()

    def update_slider_label(self, slider, label):
        label.setText(str(slider.value()))

    def populateComputeDeviceComboBox(self):
        available_devices = self.get_compute_and_platform_info()
        self.computeDeviceComboBox.addItems(available_devices)
        self.computeDeviceComboBox.setCurrentIndex(self.computeDeviceComboBox.findText("cpu"))