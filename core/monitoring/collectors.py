from threading import Event
from PySide6.QtCore import QThread, Signal

from core.monitoring.system_metrics import SystemMonitor


class MetricsCollector(QThread):

    metrics_updated = Signal(object)

    def __init__(self, interval: int = 400):
        super().__init__()
        self.interval = interval
        self._stop_event = Event()
        self.monitor = SystemMonitor()

    def run(self):
        while not self._stop_event.is_set():
            try:
                metrics = self.monitor.collect_all_metrics()
                self.metrics_updated.emit(metrics)
            except Exception as e:
                print(f"Error collecting metrics: {e}")

            self.msleep(self.interval)

    def stop(self):
        self._stop_event.set()

    def cleanup(self):
        self.monitor.shutdown()