from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu

from core.monitoring.collectors import MetricsCollector
from core.monitoring.metrics_store import MetricsStore
from core.monitoring.system_metrics import SystemMonitor
from gui.widgets.visualizations import (
    BarVisualization, SparklineVisualization,
    SpeedometerVisualization, ArcGraphVisualization
)


class MetricsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("Right click for display options")

        monitor = SystemMonitor()
        self._has_nvidia_gpu = monitor.is_nvidia_gpu_available()
        monitor.shutdown()

        self.metrics_store = MetricsStore(buffer_size=100)
        self.current_visualization_type = 1

        self._init_ui()
        self._create_visualization(self.current_visualization_type)
        self._start_metrics_collector()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def _create_visualization(self, viz_type: int):
        if hasattr(self, 'current_visualization'):
            self.current_visualization.cleanup()
            self.layout.removeWidget(self.current_visualization)
            self.current_visualization.deleteLater()

        visualizations = [
            BarVisualization,
            SparklineVisualization,
            SpeedometerVisualization,
            ArcGraphVisualization
        ]

        self.current_visualization = visualizations[viz_type](
            self.metrics_store, self._has_nvidia_gpu
        )
        self.current_visualization.setToolTip("Right click for display options")
        self.layout.addWidget(self.current_visualization)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        visual_menu = menu.addMenu("Visualization")

        viz_actions = [
            visual_menu.addAction("Bar"),
            visual_menu.addAction("Sparkline"),
            visual_menu.addAction("Speedometer"),
            visual_menu.addAction("Arc")
        ]

        viz_actions[self.current_visualization_type].setCheckable(True)
        viz_actions[self.current_visualization_type].setChecked(True)

        menu.addSeparator()

        is_running = hasattr(self, 'metrics_collector') and self.metrics_collector.isRunning()
        control_action = menu.addAction("Stop Monitoring" if is_running else "Start Monitoring")

        action = menu.exec_(event.globalPos())

        if action in viz_actions:
            new_type = viz_actions.index(action)
            if new_type != self.current_visualization_type:
                self.current_visualization_type = new_type
                self._create_visualization(new_type)
        elif action == control_action:
            if is_running:
                self._stop_metrics_collector()
            else:
                self._start_metrics_collector()

    def _start_metrics_collector(self):
        if hasattr(self, 'metrics_collector'):
            self._stop_metrics_collector()

        self.metrics_collector = MetricsCollector()
        self.metrics_collector.metrics_updated.connect(
            lambda metrics: self.metrics_store.add_metrics(metrics)
        )
        self.metrics_collector.start()

    def _stop_metrics_collector(self):
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.stop()
            self.metrics_collector.wait()
            self.metrics_collector.cleanup()
            self.metrics_collector.deleteLater()
            del self.metrics_collector

    def cleanup(self):
        self._stop_metrics_collector()
        if hasattr(self, 'current_visualization'):
            self.current_visualization.cleanup()