"""Entry point of the packaged app (SimulaticsStudio.exe)."""
import multiprocessing
import os
import shutil
import sys
from pathlib import Path


def _workspace() -> Path:
    """Where runs/ (trained policies) and robots/ (downloaded models) live for the packaged app."""
    root = Path(os.environ.get("SIMULATICS_HOME") or Path.home() / "Documents" / "Simulatics")
    root.mkdir(parents=True, exist_ok=True)
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "bundled_runs"
    if bundled.is_dir():
        for run in bundled.iterdir():           # the shipped policies, copied once
            dst = root / "runs" / run.name
            if not dst.exists():
                shutil.copytree(run, dst)
    return root


def main() -> int:
    multiprocessing.freeze_support()            # design search starts worker processes
    root = _workspace()
    os.chdir(root)
    if sys.stderr is None or sys.stdout is None:  # windowed build: no console, keep errors in a log file
        log = open(root / "studio.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    from simulatics.studio.app import main as studio_main
    return studio_main([sys.argv[0]])


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:       # noqa: BLE001 - report and really exit (Qt threads would keep a failed start alive)
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))
