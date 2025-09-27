"""Metrics collection threads."""
from PySide6.QtCore import QThread, Signal

from core.monitoring.system_metrics import SystemMetrics, SystemMonitor

class MetricsCollector(QThread):
    """Background thread for collecting system metrics."""
    
    metrics_updated = Signal(object)  # SystemMetrics
    
    def __init__(self, interval: int = 200):
        super().__init__()
        self.interval = interval
        self.running = True
        self.monitor = SystemMonitor()

    def run(self):
        """Collect metrics at regular intervals."""
        while self.running:
            try:
                metrics = self.monitor.collect_all_metrics()
                self.metrics_updated.emit(metrics)
            except Exception as e:
                print(f"Error collecting metrics: {e}")

            self.msleep(self.interval)

    def stop(self):
        """Stop metrics collection."""
        self.running = False