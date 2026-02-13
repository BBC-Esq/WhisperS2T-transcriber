from PySide6.QtCore import QObject, Signal

from core.monitoring.system_metrics import SystemMetrics


class MetricsStore(QObject):

    metrics_ready = Signal(object)

    def add_metrics(self, metrics: SystemMetrics) -> None:
        self.metrics_ready.emit(metrics)
