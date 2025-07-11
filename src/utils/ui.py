'''
Utility Functions for UI
'''
# Standard Import
import logging

# PySide 6
from PySide6.QtWidgets import (
    QLabel, QWidget, QHBoxLayout, QSizePolicy, QKeySequenceEdit
)
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QKeySequence, QKeyEvent, QFontDatabase, QFont

# Local import
from src.utils.logger import logger
from src.utils.lang import tr

def validate_numerical_input(input_str: str, error_label: QLabel,
                              val_lowest: float, val_highest: float):
    try:
        val = float(input_str)
        if not (val_lowest <= val <= val_highest):
            raise ValueError
        error_label.setVisible(False)
        return True
    except ValueError:
        error_label.setText(tr("Please enter number between {val_lowest} and {val_highest}.").format(val_lowest=val_lowest, val_highest=val_highest))
        error_label.setVisible(True)
        logger.debug(f"Invalid numerical input: {input_str}")
        return False

def create_error_label():
    error_label = QLabel()
    error_label.setStyleSheet("color: red;")
    error_label.setVisible(False)
    return error_label

def create_field(label_text, field_widget):
    container = QWidget()
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)  # Slight spacing

    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(field_widget)

    # Avoid expanding
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
    field_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    layout.setAlignment(Qt.AlignLeft)
    container.setLayout(layout)
    container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

    return container

def clear_debug_canvas(canvas):
    '''
    Flush the debug window viz canvas
    '''
    canvas.clear()
    canvas.setText(tr("Press start or 'F1' to start AutoBot"))
    canvas.setAlignment(Qt.AlignCenter)

class SingleKeyEdit(QKeySequenceEdit):
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()

        # Allow modifier-only keys
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            self.setKeySequence(QKeySequence(key))
            return

        # Otherwise, record only one key (replace previous)
        if modifiers:
            # This prevents combos like Ctrl+A
            self.setKeySequence(QKeySequence(key))
        else:
            self.setKeySequence(QKeySequence(key))

    def set_key(self, key_str):
        """
        Set the key sequence from string like 'A', 'F1', etc.
        """
        self.setKeySequence(QKeySequence(key_str))

    def get_key(self):
        """
        Return the currently set key as a string, like 'A', 'F1', etc.
        """
        seq = self.keySequence()
        return seq.toString(QKeySequence.NativeText).strip().lower()

class QtLogHandler(logging.Handler, QObject):
    '''
    QtLogHandler
    '''
    log_signal = Signal(str, int)  # message, level

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg, record.levelno)

def apply_chinese_font(app):
    '''
    Set application font to one that supports Chinese characters if available.
    '''
    candidates = [
        "Microsoft JhengHei",  # Windows Traditional Chinese
        "Microsoft YaHei",     # Windows Simplified Chinese
        "PingFang TC",         # macOS Traditional Chinese
        "PingFang SC",         # macOS Simplified Chinese
        "SimHei",              # Generic fallback
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ]
    db = QFontDatabase()
    for name in candidates:
        if name in db.families():
            app.setFont(QFont(name))
            break
