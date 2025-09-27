"""Application entry point."""
import sys
from PySide6.QtWidgets import QApplication

from utils.cuda_setup import setup_cuda_paths
from gui.main_window import MainWindow

def main():
    """Run the application."""
    # Setup CUDA paths if needed
    setup_cuda_paths()
    
    # Create and run application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()