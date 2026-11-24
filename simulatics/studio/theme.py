"""Dark workstation theme: one palette, one stylesheet."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#1b1d22"        # window
PANEL = "#23262d"     # dock contents
RAISED = "#2c3039"    # inputs, headers
BORDER = "#15171b"
LINE = "#3a3f4b"
TEXT = "#d6dae2"
DIM = "#8b93a3"
ACCENT = "#4c9aff"
ACCENT_DIM = "#2d5fa8"
GOOD = "#3ecf8e"
WARN = "#f0b429"
BAD = "#ef5350"

PORT_COLORS = {
    "robot": "#4c9aff", "policy": "#b388ff", "motion": "#ff8a65", "number": "#3ecf8e", "int": "#3ecf8e",
    "bool": "#f0b429", "text": "#cfd8dc", "choice": "#cfd8dc", "code": "#cfd8dc", "series": "#ff5c8a",
    "points": "#26c6da", "image": "#ffd54f", "table": "#a1887f", "any": "#9aa3b2",
}

QSS = f"""
* {{ font-size: 12px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; }}
QMainWindow::separator {{ background: {BORDER}; width: 4px; height: 4px; }}
QMainWindow::separator:hover {{ background: {ACCENT_DIM}; }}
QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {RAISED}; padding: 5px 8px; border-bottom: 1px solid {BORDER};
                      text-transform: uppercase; font-weight: 600; font-size: 11px; }}
QDockWidget > QWidget {{ background: {PANEL}; }}
QTabBar::tab {{ background: {BG}; color: {DIM}; padding: 5px 12px; border: none; border-right: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::close-button {{ image: none; width: 8px; height: 8px; border-radius: 4px; background: {LINE}; margin: 2px; }}
QTabBar::close-button:hover {{ background: {BAD}; }}
QTabWidget::pane {{ border: none; background: {PANEL}; }}
QMenuBar {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}
QToolBar {{ background: {BG}; border: none; spacing: 2px; padding: 3px; }}
QToolBar::separator {{ background: {LINE}; width: 1px; height: 1px; margin: 4px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px; }}
QToolButton:hover {{ background: {RAISED}; }}
QToolButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QPushButton {{ background: {RAISED}; border: 1px solid {LINE}; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_DIM}; }}
QPushButton:disabled {{ color: {DIM}; border-color: {RAISED}; }}
QPushButton[primary="true"] {{ background: {ACCENT_DIM}; border-color: {ACCENT}; font-weight: 600; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BG}; border: 1px solid {LINE}; border-radius: 3px; padding: 3px 5px;
    selection-background-color: {ACCENT_DIM}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM}; }}
QTreeWidget, QListWidget, QTableWidget, QTreeView, QTableView {{
    background: {PANEL}; alternate-background-color: #262a32; border: none; outline: none; }}
QTreeWidget::item, QListWidget::item {{ padding: 2px 0; }}
QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected {{ background: {ACCENT_DIM}; color: white; }}
QHeaderView::section {{ background: {RAISED}; border: none; border-right: 1px solid {BORDER}; padding: 4px 6px; color: {DIM}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 10px; }}
QScrollBar:horizontal {{ background: {PANEL}; height: 10px; }}
QScrollBar::handle {{ background: {LINE}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; height: 0; width: 0; }}
QSlider::groove:horizontal {{ height: 4px; background: {BG}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {LINE}; border-radius: 3px; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QStatusBar {{ background: {BG}; border-top: 1px solid {BORDER}; color: {DIM}; }}
QStatusBar::item {{ border: none; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 4px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {DIM}; text-transform: uppercase; font-size: 10px; }}
QProgressBar {{ background: {BG}; border: 1px solid {LINE}; border-radius: 3px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QSplitter::handle {{ background: {BORDER}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px; }}
QLabel[dim="true"] {{ color: {DIM}; }}
QLabel[heading="true"] {{ color: {DIM}; text-transform: uppercase; font-size: 10px; font-weight: 600; padding-top: 6px; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    for role, c in ((QPalette.Window, BG), (QPalette.Base, BG), (QPalette.AlternateBase, PANEL),
                    (QPalette.Button, RAISED), (QPalette.Text, TEXT), (QPalette.WindowText, TEXT),
                    (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT_DIM), (QPalette.HighlightedText, "#ffffff"),
                    (QPalette.ToolTipBase, RAISED), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, DIM)):
        p.setColor(role, QColor(c))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(DIM))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(DIM))
    app.setPalette(p)
    f = QFont("Segoe UI", 9)
    app.setFont(f)
    app.setStyleSheet(QSS)


def mono() -> QFont:
    f = QFont("Cascadia Mono", 9)
    f.setStyleHint(QFont.Monospace)
    return f
