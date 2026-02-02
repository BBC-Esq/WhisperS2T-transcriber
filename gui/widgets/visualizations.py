from collections import deque
from math import sin, cos, pi

from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QProgressBar, QLabel,
    QVBoxLayout
)
from PySide6.QtGui import (
    QPainter, QColor, QPolygon, QPainterPath, QPen,
    QPixmap, QLinearGradient
)

from core.monitoring.metrics_store import MetricsStore
from core.monitoring.system_metrics import SystemMetrics


class BaseVisualization(QWidget):

    def __init__(self, metrics_store: MetricsStore, has_nvidia_gpu: bool):
        super().__init__()
        self.metrics_store = metrics_store
        self.has_nvidia_gpu = has_nvidia_gpu
        self.metrics_store.metrics_ready.connect(self.update_metrics)

    def update_metrics(self, metrics: SystemMetrics):
        raise NotImplementedError

    def cleanup(self):
        self.metrics_store.metrics_ready.disconnect(self.update_metrics)


class BarVisualization(BaseVisualization):

    def __init__(self, metrics_store: MetricsStore, has_nvidia_gpu: bool):
        super().__init__(metrics_store, has_nvidia_gpu)
        self._init_ui()

    def _init_ui(self):
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        self.cpu_bar, self.cpu_label = self._add_metric_row(
            "CPU Usage:", "#FF4136", grid_layout, 0
        )
        self.ram_bar, self.ram_label = self._add_metric_row(
            "RAM Usage:", "#B10DC9", grid_layout, 1
        )

        if self.has_nvidia_gpu:
            self.gpu_bar, self.gpu_label = self._add_metric_row(
                "GPU Usage:", "#0074D9", grid_layout, 2
            )
            self.vram_bar, self.vram_label = self._add_metric_row(
                "VRAM Usage:", "#2ECC40", grid_layout, 3
            )
            self.power_bar, self.power_label = self._add_metric_row(
                "GPU Power:", "#FFD700", grid_layout, 4
            )

    def _add_metric_row(self, label_text: str, color: str,
                        layout: QGridLayout, row: int):
        label = QLabel(label_text)
        layout.addWidget(label, row, 0)

        percent_label = QLabel("0%")
        layout.addWidget(percent_label, row, 1)

        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setMaximumHeight(11)
        bar.setStyleSheet(
            f"QProgressBar {{ background-color: #1e2126; border: none; }}"
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )
        bar.setTextVisible(False)
        layout.addWidget(bar, row, 2)

        return bar, percent_label

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_bar.setValue(int(metrics.cpu_usage))
        self.cpu_label.setText(f"{int(metrics.cpu_usage)}%")

        self.ram_bar.setValue(int(metrics.ram_usage_percent))
        self.ram_label.setText(f"{int(metrics.ram_usage_percent)}%")

        if self.has_nvidia_gpu and metrics.gpu_utilization is not None:
            self.gpu_bar.setValue(int(metrics.gpu_utilization))
            self.gpu_label.setText(f"{int(metrics.gpu_utilization)}%")

            self.vram_bar.setValue(int(metrics.vram_usage_percent))
            self.vram_label.setText(f"{int(metrics.vram_usage_percent)}%")

            self.power_bar.setValue(int(metrics.power_usage_percent))
            self.power_label.setText(f"{int(metrics.power_usage_percent)}%")


class Sparkline(QWidget):

    def __init__(self, max_values: int = 125, color: str = "#0074D9"):
        super().__init__()
        self.max_values = max_values
        self.values = deque(maxlen=max_values)
        self.setFixedSize(125, 65)
        self.color = QColor(color)
        self._gradient_pixmap = self._create_gradient()

    def add_value(self, value: float):
        self.values.append(value)
        self.update()

    def _create_gradient(self):
        pixmap = QPixmap(1, self.height())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, 0, self.height())

        fill_color = QColor(self.color)
        fill_color.setAlpha(60)
        gradient.setColorAt(0, fill_color)
        gradient.setColorAt(1, QColor(0, 0, 0, 0))

        painter.fillRect(pixmap.rect(), gradient)
        painter.end()

        return pixmap

    def paintEvent(self, event):
        if not self.values:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 5

        path = QPainterPath()
        x_step = (width - 2 * margin) / (len(self.values) - 1) if len(self.values) > 1 else 0
        points = []

        for i, value in enumerate(self.values):
            x = margin + i * x_step
            y = height - margin - (value / 100) * (height - 2 * margin)
            points.append(QPointF(x, y))

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill_path = QPainterPath(path)
        if points:
            fill_path.lineTo(points[-1].x(), height - margin)
            fill_path.lineTo(points[0].x(), height - margin)
            fill_path.closeSubpath()

        painter.save()
        painter.setClipPath(fill_path)
        for x in range(0, width, self._gradient_pixmap.width()):
            painter.drawPixmap(x, 0, self._gradient_pixmap)
        painter.restore()

        painter.setPen(QPen(self.color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


class SparklineVisualization(BaseVisualization):

    def __init__(self, metrics_store: MetricsStore, has_nvidia_gpu: bool):
        super().__init__(metrics_store, has_nvidia_gpu)
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)

        cpu_widget, self.cpu_sparkline, self.cpu_label = self._create_sparkline_group(
            "CPU", "#FF4136"
        )
        layout.addWidget(cpu_widget, 0, 0)

        ram_widget, self.ram_sparkline, self.ram_label = self._create_sparkline_group(
            "RAM", "#B10DC9"
        )
        layout.addWidget(ram_widget, 0, 1)

        if self.has_nvidia_gpu:
            gpu_widget, self.gpu_sparkline, self.gpu_label = self._create_sparkline_group(
                "GPU", "#0074D9"
            )
            layout.addWidget(gpu_widget, 0, 2)

            vram_widget, self.vram_sparkline, self.vram_label = self._create_sparkline_group(
                "VRAM", "#2ECC40"
            )
            layout.addWidget(vram_widget, 0, 3)

            power_widget, self.power_sparkline, self.power_label = self._create_sparkline_group(
                "GPU Power", "#FFD700"
            )
            layout.addWidget(power_widget, 0, 4)

        for i in range(layout.columnCount()):
            layout.setColumnStretch(i, 1)

    def _create_sparkline_group(self, name: str, color: str):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)

        sparkline = Sparkline(color=color)
        layout.addWidget(sparkline, alignment=Qt.AlignCenter)

        label = QLabel(f"{name} 0.0%")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, alignment=Qt.AlignCenter)

        return widget, sparkline, label

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_sparkline.add_value(metrics.cpu_usage)
        self.cpu_label.setText(f"CPU {metrics.cpu_usage:.1f}%")

        self.ram_sparkline.add_value(metrics.ram_usage_percent)
        self.ram_label.setText(f"RAM {metrics.ram_usage_percent:.1f}%")

        if self.has_nvidia_gpu and metrics.gpu_utilization is not None:
            self.gpu_sparkline.add_value(metrics.gpu_utilization)
            self.gpu_label.setText(f"GPU {metrics.gpu_utilization:.1f}%")

            self.vram_sparkline.add_value(metrics.vram_usage_percent)
            self.vram_label.setText(f"VRAM {metrics.vram_usage_percent:.1f}%")

            self.power_sparkline.add_value(metrics.power_usage_percent)
            self.power_label.setText(f"GPU Power {metrics.power_usage_percent:.1f}%")


class Speedometer(QWidget):

    def __init__(self, min_value: float = 0, max_value: float = 100,
                 colors: list = None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.colors = colors or ["#00FF00", "#FFFF00", "#FF0000"]
        self.setFixedSize(105, 105)
        self._arc_background = self._create_arc_background()

    def set_value(self, value: float):
        self.current_value = max(self.min_value, min(self.max_value, value))
        self.update()

    def _create_arc_background(self):
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) / 2 * 0.7

        start_angle = 180 * 16
        for i in range(180):
            color = self._get_color_at_angle(i)
            painter.setPen(color)
            painter.drawArc(
                int(center_x - radius),
                int(center_y - radius),
                int(radius * 2),
                int(radius * 2),
                start_angle - i * 16,
                -16
            )

        painter.end()
        return pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._arc_background)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) / 2 * 0.7

        angle = 180 - (self.current_value - self.min_value) / (
            self.max_value - self.min_value
        ) * 180

        needle_length = radius * 0.9
        needle_width = 5
        needle_angle = angle * (pi / 180)

        needle_tip_x = center_x + needle_length * cos(needle_angle)
        needle_tip_y = center_y - needle_length * sin(needle_angle)

        perpendicular_angle = needle_angle + pi / 2
        half_width = needle_width / 2

        point1 = QPointF(
            center_x + half_width * cos(perpendicular_angle),
            center_y - half_width * sin(perpendicular_angle)
        )
        point2 = QPointF(
            center_x - half_width * cos(perpendicular_angle),
            center_y + half_width * sin(perpendicular_angle)
        )
        point3 = QPointF(needle_tip_x, needle_tip_y)

        needle = QPolygon([point1.toPoint(), point2.toPoint(), point3.toPoint()])

        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawPolygon(needle)

    def _get_color_at_angle(self, angle: int) -> QColor:
        t = angle / 180

        if t <= 0:
            return QColor(self.colors[0])
        elif t >= 1:
            return QColor(self.colors[-1])
        else:
            segment = t * (len(self.colors) - 1)
            index = int(segment)
            t = segment - index
            index = min(index, len(self.colors) - 2)

            c1 = QColor(self.colors[index])
            c2 = QColor(self.colors[index + 1])

            r = int(c1.red() * (1 - t) + c2.red() * t)
            g = int(c1.green() * (1 - t) + c2.green() * t)
            b = int(c1.blue() * (1 - t) + c2.blue() * t)

            return QColor(r, g, b)


class SpeedometerVisualization(BaseVisualization):

    def __init__(self, metrics_store: MetricsStore, has_nvidia_gpu: bool):
        super().__init__(metrics_store, has_nvidia_gpu)
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)

        cpu_layout, self.cpu_meter, self.cpu_label = self._create_speedometer_group("CPU")
        layout.addLayout(cpu_layout, 0, 0)

        ram_layout, self.ram_meter, self.ram_label = self._create_speedometer_group("RAM")
        layout.addLayout(ram_layout, 0, 1)

        if self.has_nvidia_gpu:
            gpu_layout, self.gpu_meter, self.gpu_label = self._create_speedometer_group("GPU")
            layout.addLayout(gpu_layout, 0, 2)

            vram_layout, self.vram_meter, self.vram_label = self._create_speedometer_group("VRAM")
            layout.addLayout(vram_layout, 0, 3)

            power_layout, self.power_meter, self.power_label = self._create_speedometer_group("GPU Power")
            layout.addLayout(power_layout, 0, 4)

        for i in range(layout.columnCount()):
            layout.setColumnStretch(i, 1)

    def _create_speedometer_group(self, name: str):
        layout = QVBoxLayout()
        layout.setSpacing(2)

        speedometer = Speedometer(colors=["#00FF00", "#FFFF00", "#FF0000"])
        speedometer.setFixedSize(105, 105)
        layout.addWidget(speedometer, alignment=Qt.AlignCenter)

        label = QLabel(f"{name} 0.0%")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, alignment=Qt.AlignCenter)

        return layout, speedometer, label

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_meter.set_value(metrics.cpu_usage)
        self.cpu_label.setText(f"CPU {metrics.cpu_usage:.1f}%")

        self.ram_meter.set_value(metrics.ram_usage_percent)
        self.ram_label.setText(f"RAM {metrics.ram_usage_percent:.1f}%")

        if self.has_nvidia_gpu and metrics.gpu_utilization is not None:
            self.gpu_meter.set_value(metrics.gpu_utilization)
            self.gpu_label.setText(f"GPU {metrics.gpu_utilization:.1f}%")

            self.vram_meter.set_value(metrics.vram_usage_percent)
            self.vram_label.setText(f"VRAM {metrics.vram_usage_percent:.1f}%")

            self.power_meter.set_value(metrics.power_usage_percent)
            self.power_label.setText(f"GPU Power {metrics.power_usage_percent:.1f}%")


class ArcGraph(QWidget):

    def __init__(self, color: str = "#0074D9"):
        super().__init__()
        self.color = QColor(color)
        self.value = 0
        self.setFixedSize(100, 100)
        self._arc_background = self._create_background()

    def set_value(self, value: float):
        self.value = min(100, max(0, value))
        self.update()

    def _create_background(self):
        background = QPixmap(self.size())
        background.fill(Qt.transparent)

        painter = QPainter(background)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        radius = min(width, height) / 2 - 10
        center = QPointF(width / 2, height / 2)

        painter.setPen(QPen(QColor("#1e2126"), 8))
        painter.drawArc(
            int(center.x() - radius),
            int(center.y() - radius),
            int(radius * 2),
            int(radius * 2),
            180 * 16,
            -180 * 16
        )
        painter.end()

        return background

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._arc_background)

        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        radius = min(width, height) / 2 - 10
        center = QPointF(width / 2, height / 2)

        painter.setPen(QPen(self.color, 8))
        span_angle = -(self.value / 100.0) * 180
        painter.drawArc(
            int(center.x() - radius),
            int(center.y() - radius),
            int(radius * 2),
            int(radius * 2),
            180 * 16,
            int(span_angle * 16)
        )

        painter.setPen(Qt.white)
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self.value)}%")


class ArcGraphVisualization(BaseVisualization):

    def __init__(self, metrics_store: MetricsStore, has_nvidia_gpu: bool):
        super().__init__(metrics_store, has_nvidia_gpu)
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)

        cpu_layout, self.cpu_arc, self.cpu_label = self._create_arc_group("CPU", "#FF4136")
        layout.addLayout(cpu_layout, 0, 0)

        ram_layout, self.ram_arc, self.ram_label = self._create_arc_group("RAM", "#B10DC9")
        layout.addLayout(ram_layout, 0, 1)

        if self.has_nvidia_gpu:
            gpu_layout, self.gpu_arc, self.gpu_label = self._create_arc_group("GPU", "#0074D9")
            layout.addLayout(gpu_layout, 0, 2)

            vram_layout, self.vram_arc, self.vram_label = self._create_arc_group("VRAM", "#2ECC40")
            layout.addLayout(vram_layout, 0, 3)

            power_layout, self.power_arc, self.power_label = self._create_arc_group("GPU Power", "#FFD700")
            layout.addLayout(power_layout, 0, 4)

        for i in range(layout.columnCount()):
            layout.setColumnStretch(i, 1)

    def _create_arc_group(self, name: str, color: str):
        layout = QVBoxLayout()
        layout.setSpacing(2)

        arc = ArcGraph(color=color)
        layout.addWidget(arc, alignment=Qt.AlignCenter)

        label = QLabel(name)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, alignment=Qt.AlignCenter)

        return layout, arc, label

    def update_metrics(self, metrics: SystemMetrics):
        self.cpu_arc.set_value(metrics.cpu_usage)
        self.cpu_label.setText(f"CPU {metrics.cpu_usage:.1f}%")

        self.ram_arc.set_value(metrics.ram_usage_percent)
        self.ram_label.setText(f"RAM {metrics.ram_usage_percent:.1f}%")

        if self.has_nvidia_gpu and metrics.gpu_utilization is not None:
            self.gpu_arc.set_value(metrics.gpu_utilization)
            self.gpu_label.setText(f"GPU {metrics.gpu_utilization:.1f}%")

            self.vram_arc.set_value(metrics.vram_usage_percent)
            self.vram_label.setText(f"VRAM {metrics.vram_usage_percent:.1f}%")

            self.power_arc.set_value(metrics.power_usage_percent)
            self.power_label.setText(f"GPU Power {metrics.power_usage_percent:.1f}%")