"""Metrics data storage and subscription management."""
import threading
from typing import List, Callable

from core.monitoring.system_metrics import SystemMetrics

class MetricsStore:
    """Store and distribute system metrics to subscribers."""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.metrics_history: List[SystemMetrics] = []
        self._subscribers: List[Callable[[SystemMetrics], None]] = []
        self._lock = threading.Lock()

    def add_metrics(self, metrics: SystemMetrics) -> None:
        """Add new metrics and notify subscribers."""
        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.buffer_size:
                self.metrics_history.pop(0)
        self._notify_subscribers(metrics)

    def subscribe(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Subscribe to metrics updates."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Unsubscribe from metrics updates."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify_subscribers(self, metrics: SystemMetrics) -> None:
        """Notify all subscribers of new metrics."""
        with self._lock:
            subscribers = self._subscribers.copy()
        
        for subscriber in subscribers:
            try:
                subscriber(metrics)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def get_latest_metrics(self) -> SystemMetrics:
        """Get the most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def clear(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self.metrics_history.clear()