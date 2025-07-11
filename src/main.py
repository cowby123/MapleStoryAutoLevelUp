# Standard Import
import sys
import argparse

# Pyside
from PySide6.QtWidgets import QApplication

from src.utils.ui import apply_chinese_font
from src.utils.lang import set_language

# Load Import
from src.ui.ui import MainWindow
from src.ui.AutoBotController import AutoBotController

def main():
    '''
    Main Function
    Run: python -m ui.main
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='en', choices=['en', 'zh'])
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    apply_chinese_font(app)
    set_language(args.lang)

    autoBotController = AutoBotController()
    ui = MainWindow(autoBotController)

    autoBotController.update_signal(ui)

    ui.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
