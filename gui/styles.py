"""UI styling constants and utilities."""

# Window styles
WINDOW_STYLE = """
    QWidget {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12px;
    }
"""

# Progress bar styles
PROGRESS_BAR_STYLE = """
    QProgressBar {
        background-color: #1e2126;
        border: none;
        border-radius: 2px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: {color};
        border-radius: 2px;
    }
"""

# Button styles
BUTTON_NORMAL = """
    QPushButton {
        padding: 5px 15px;
        border-radius: 3px;
        background-color: #3498db;
        color: white;
    }
    QPushButton:hover {
        background-color: #2980b9;
    }
    QPushButton:pressed {
        background-color: #21618c;
    }
    QPushButton:disabled {
        background-color: #7f8c8d;
        color: #bdc3c7;
    }
"""

BUTTON_STOP = """
    QPushButton {
        padding: 5px 15px;
        border-radius: 3px;
        background-color: #e74c3c;
        color: white;
    }
    QPushButton:hover {
        background-color: #c0392b;
    }
    QPushButton:pressed {
        background-color: #a93226;
    }
    QPushButton:disabled {
        background-color: #7f8c8d;
        color: #bdc3c7;
    }
"""

# GroupBox styles
GROUP_BOX_STYLE = """
    QGroupBox {
        font-weight: bold;
        border: 1px solid #bdc3c7;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }
"""