"""Main metrics bar widget."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMenu
from PySide6.QtGui import QPixmapCache

from core.monitoring.collectors import MetricsCollector
from core.monitoring.metrics_store import MetricsStore
from gui.widgets.visualizations import (
    BarVisualization, SparklineVisualization,
    SpeedometerVisualization, ArcGraphVisualization
)

class MetricsBar(QWidget):
    """Main metrics display widget with multiple visualization options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        QPixmapCache.setCacheLimit(10 * 1024)
        self.setToolTip("Right click for display options")
        
        self.metrics_store = MetricsStore(buffer_size=100)
        self.current_visualization_type = 1  # Default to sparkline
        
        self._init_ui()
        self._create_visualization(self.current_visualization_type)
        self._start_metrics_collector()
        
    def _init_ui(self):
        """Initialize UI layout."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
    def _create_visualization(self, viz_type: int):
        """Create visualization widget."""
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
        
        self.current_visualization = visualizations[viz_type](self.metrics_store)
        self.current_visualization.setToolTip("Right click for display options")
        self.layout.addWidget(self.current_visualization)
        
    def contextMenuEvent(self, event):
        """Show context menu for visualization options."""
        menu = QMenu(self)
        
        # Visualization submenu
        visual_menu = menu.addMenu("Visualization")
        
        viz_actions = [
            visual_menu.addAction("Bar"),
            visual_menu.addAction("Sparkline"),
            visual_menu.addAction("Speedometer"),
            visual_menu.addAction("Arc")
        ]
        
        # Mark current visualization
        viz_actions[self.current_visualization_type].setCheckable(True)
        viz_actions[self.current_visualization_type].setChecked(True)
        
        menu.addSeparator()
        
        # Monitoring control
        is_running = hasattr(self, 'metrics_collector') and self.metrics_collector.isRunning()
        control_action = menu.addAction("Stop Monitoring" if is_running else "Start Monitoring")
        
        # Execute menu
        action = menu.exec_(event.globalPos())
        
        # Handle action
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
        """Start metrics collection."""
        self.metrics_collector = MetricsCollector()
        self.metrics_collector.metrics_updated.connect(
            lambda metrics: self.metrics_store.add_metrics(metrics)
        )
        self.metrics_collector.start()
        
    def _stop_metrics_collector(self):
        """Stop metrics collection."""
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.stop()
            self.metrics_collector.wait()
            
    def cleanup(self):
        """Clean up resources."""
        self._stop_metrics_collector()
        if hasattr(self, 'current_visualization'):
            self.current_visualization.cleanup()
        QPixmapCache.clear()
        
    def closeEvent(self, event):
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)