from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider
)
from PySide6.QtCore import Qt, Signal

from config.constants import (
    MODEL_NAMES, MODEL_PRECISIONS, DISTIL_MODELS, WHISPER_LANGUAGES,
    OUTPUT_FORMATS, TASK_MODES,
    DEFAULT_BEAM_SIZE, DEFAULT_BATCH_SIZE,
    DEFAULT_OUTPUT_FORMAT, DEFAULT_TASK_MODE, DEFAULT_LANGUAGE
)
from utils.system_utils import get_compute_and_platform_info, has_bfloat16_support


class SettingsWidget(QGroupBox):

    device_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self._bfloat16_supported = has_bfloat16_support()
        self._init_ui()
        self._populate_devices()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_NAMES)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row1.addWidget(self.model_combo)

        row1.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        row1.addWidget(self.precision_combo)

        row1.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        row1.addWidget(self.device_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Output:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(OUTPUT_FORMATS)
        self.format_combo.setCurrentText(DEFAULT_OUTPUT_FORMAT)
        row2.addWidget(self.format_combo)

        row2.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(TASK_MODES)
        self.task_combo.setCurrentText(DEFAULT_TASK_MODE)
        row2.addWidget(self.task_combo)

        row2.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        for code, name in WHISPER_LANGUAGES.items():
            self.language_combo.addItem(f"{name} ({code})", code)
        self._set_language_by_code(DEFAULT_LANGUAGE)
        row2.addWidget(self.language_combo)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Beam Size:"))
        self.beam_combo = QComboBox()
        self.beam_combo.addItems([str(i) for i in range(1, 6)])
        self.beam_combo.setCurrentText(str(DEFAULT_BEAM_SIZE))
        row3.addWidget(self.beam_combo)

        row3.addWidget(QLabel("Batch Size:"))
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setMinimum(1)
        self.batch_slider.setMaximum(200)
        self.batch_slider.setValue(DEFAULT_BATCH_SIZE)
        self.batch_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_slider.setTickInterval(10)
        row3.addWidget(self.batch_slider)

        self.batch_label = QLabel(str(DEFAULT_BATCH_SIZE))
        self.batch_slider.valueChanged.connect(
            lambda v: self.batch_label.setText(str(v))
        )
        row3.addWidget(self.batch_label)
        layout.addLayout(row3)

    def _populate_devices(self):
        devices = get_compute_and_platform_info()
        self.device_combo.addItems(devices)

        if "cuda" in devices:
            self.device_combo.setCurrentText("cuda")
        else:
            self.device_combo.setCurrentText("cpu")

        self._update_precision_options()
        self._update_model_constraints()

    def _on_device_changed(self, device: str):
        self.device_changed.emit(device)
        self._update_precision_options()

    def _on_model_changed(self, model_name: str):
        self._update_precision_options()
        self._update_model_constraints()

    def _update_precision_options(self):
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        current_device = self.device_combo.currentText()
        model_precs = MODEL_PRECISIONS.get(model_name, [])

        if current_device == "cpu":
            available = [p for p in model_precs if p == "float32"]
        else:
            available = []
            for p in model_precs:
                if p == "bfloat16" and not self._bfloat16_supported:
                    continue
                available.append(p)

        previous = self.precision_combo.currentText()
        self.precision_combo.blockSignals(True)
        self.precision_combo.clear()
        self.precision_combo.addItems(available)

        if previous in available:
            self.precision_combo.setCurrentText(previous)
        self.precision_combo.blockSignals(False)

    def _update_model_constraints(self):
        model_name = self.model_combo.currentText()
        is_english_only = model_name.endswith(".en")
        is_distil = model_name in DISTIL_MODELS
        can_translate = not is_english_only and not is_distil

        self.task_combo.blockSignals(True)
        if can_translate:
            current_task = self.task_combo.currentText()
            self.task_combo.clear()
            self.task_combo.addItems(TASK_MODES)
            if current_task in TASK_MODES:
                self.task_combo.setCurrentText(current_task)
            self.task_combo.setEnabled(True)
        else:
            self.task_combo.clear()
            self.task_combo.addItem("transcribe")
            self.task_combo.setCurrentText("transcribe")
            self.task_combo.setEnabled(False)
        self.task_combo.blockSignals(False)

        if is_english_only:
            self.language_combo.setEnabled(False)
            self._set_language_by_code("en")
        else:
            self.language_combo.setEnabled(True)

    def _set_language_by_code(self, code: str):
        for i in range(self.language_combo.count()):
            if self.language_combo.itemData(i) == code:
                self.language_combo.setCurrentIndex(i)
                return

    def get_model(self) -> str:
        model_name = self.model_combo.currentText()
        precision = self.precision_combo.currentText()
        return f"{model_name} - {precision}"

    def get_device(self) -> str:
        return self.device_combo.currentText()

    def get_output_format(self) -> str:
        return self.format_combo.currentText()

    def get_task_mode(self) -> str:
        return self.task_combo.currentText()

    def get_language(self) -> str:
        return self.language_combo.currentData()

    def get_beam_size(self) -> int:
        return int(self.beam_combo.currentText())

    def get_batch_size(self) -> int:
        return self.batch_slider.value()
