import threading
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from core.monitoring.system_metrics import SystemMetrics


class MetricsStore(QObject):

    metrics_ready = Signal(object)

    def __init__(self, buffer_size: int = 100, parent=None):
        super().__init__(parent)
        self.buffer_size = buffer_size
        self.metrics_history: List[SystemMetrics] = []
        self._lock = threading.Lock()

    def add_metrics(self, metrics: SystemMetrics) -> None:
        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.buffer_size:
                self.metrics_history.pop(0)
        self.metrics_ready.emit(metrics)

    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def clear(self) -> None:
        with self._lock:
            self.metrics_history.clear()