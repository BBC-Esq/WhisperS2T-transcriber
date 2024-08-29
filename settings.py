from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider)
from PySide6.QtCore import Qt, Signal
from constants import WHISPER_MODELS
import torch

from utilities import has_bfloat16_support

class SettingsGroupBox(QGroupBox):
    device_changed = Signal(str)

    def __init__(self, get_compute_and_platform_info_callback, parent=None):
        super().__init__("Settings", parent)
        self.get_compute_and_platform_info = get_compute_and_platform_info_callback
        self.initUI()

    def initUI(self):
        self.setLayout(QVBoxLayout())

        hbox1_layout = QHBoxLayout()
        
        modelLabel = QLabel("Model:")
        hbox1_layout.addWidget(modelLabel)

        self.modelComboBox = QComboBox()
        hbox1_layout.addWidget(self.modelComboBox)

        computeDeviceLabel = QLabel("Device:")
        hbox1_layout.addWidget(computeDeviceLabel)

        self.computeDeviceComboBox = QComboBox()
        hbox1_layout.addWidget(self.computeDeviceComboBox)
        self.computeDeviceComboBox.currentTextChanged.connect(self.on_device_changed)

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
        self.beamSizeSlider.setMaximum(5)
        self.beamSizeSlider.setValue(1)
        self.beamSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.beamSizeSlider.setTickInterval(1)
        beam_size_layout.addWidget(self.beamSizeSlider)

        self.beamSizeValueLabel = QLabel("1")
        beam_size_layout.addWidget(self.beamSizeValueLabel)
        self.beamSizeSlider.valueChanged.connect(lambda: self.update_slider_label(self.beamSizeSlider, self.beamSizeValueLabel))

        self.layout().addLayout(beam_size_layout)

        batch_size_layout = QHBoxLayout()
        batchSizeLabel = QLabel("Batch Size:")
        batch_size_layout.addWidget(batchSizeLabel)

        self.batchSizeSlider = QSlider(Qt.Horizontal)
        self.batchSizeSlider.setMinimum(1)
        self.batchSizeSlider.setMaximum(200)
        self.batchSizeSlider.setValue(8)
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
        if "cuda" in available_devices:
            self.computeDeviceComboBox.setCurrentIndex(self.computeDeviceComboBox.findText("cuda"))
        else:
            self.computeDeviceComboBox.setCurrentIndex(self.computeDeviceComboBox.findText("cpu"))
        self.update_model_combobox()

    def on_device_changed(self, device):
        self.device_changed.emit(device)
        self.update_model_combobox()

    def update_model_combobox(self):
        current_device = self.computeDeviceComboBox.currentText()
        self.modelComboBox.clear()
        
        for model_name, model_info in WHISPER_MODELS.items():
            if current_device == "cpu" and model_info['precision'] == 'float32':
                self.modelComboBox.addItem(model_name)
            elif current_device == "cuda":
                if model_info['precision'] in ['float32', 'float16']:
                    self.modelComboBox.addItem(model_name)
                elif model_info['precision'] == 'bfloat16' and has_bfloat16_support():
                    self.modelComboBox.addItem(model_name)
