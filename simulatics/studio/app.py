"""Entry point of the desktop app: `simulatics studio` or `python -m simulatics.studio`."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Simulatics Studio needs PySide6. Install it with:  pip install PySide6-Essentials")
        return 1
    from . import theme
    from .viewport import configure_gl
    configure_gl()                      # before the QApplication exists
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Simulatics Studio")
    app.setOrganizationName("Simulatics")
    theme.apply(app)
    from .main_window import StudioWindow
    win = StudioWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
