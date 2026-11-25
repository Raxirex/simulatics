"""Entry point of the desktop app: `simulatics studio` or `python -m simulatics.studio`."""
from __future__ import annotations

import os
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
    selftest = os.environ.get("SIMULATICS_SELFTEST")
    win = StudioWindow(restore=not selftest)
    win.show()
    if selftest:
        _selftest(app, win, selftest)
    code = app.exec()
    if getattr(sys, "frozen", False):
        hard_exit(code)
    return code


def hard_exit(code: int = 0) -> None:
    """End the process at once. In the packaged build an ordinary exit (even os._exit) crashes inside the Qt
    binding while Windows unloads it, so on Windows the process is terminated without unloading anything.
    Call it only after everything worth keeping has been saved."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.TerminateProcess(k32.GetCurrentProcess(), int(code))
    os._exit(int(code))


def _selftest(app, win, spec: str) -> None:
    """SIMULATICS_SELFTEST="sample_a,sample_b": run those samples for a few seconds each, write
    selftest.json and selftest_<sample>.png into the working folder, and quit. Used to check a packaged build."""
    import json
    import time
    from PySide6.QtCore import QTimer
    from .. import __version__, samples
    ids = [s for s in spec.split(",") if s]
    report = {"version": __version__, "samples_available": len(samples.SAMPLES), "runs": []}
    state = {"i": -1, "t0": 0.0}

    def step():
        i = state["i"]
        if i >= 0:
            s = win.sess
            win.viewport.grabFramebuffer().save(f"selftest_{ids[i]}.png")
            report["runs"].append({"sample": ids[i], "sim_seconds": round(float(s.data.time), 2),
                                   "fps": round(win.viewport._fps, 1), "controller": getattr(s.controller, "name", ""),
                                   "falls": s.falls, "error": s.error, "log": win.log_panel.toPlainText().splitlines()[-1:]})
        state["i"] = i = i + 1
        if i >= len(ids):
            with open("selftest.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1)
            win.close()
            app.quit()
            return
        win.run_sample(ids[i])
        QTimer.singleShot(5000, step)
    QTimer.singleShot(2500, step)


if __name__ == "__main__":
    raise SystemExit(main())
